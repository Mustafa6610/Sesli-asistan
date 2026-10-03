"""
Sesli Asistan Sunucusu (Bulut Barindirma + Bilgisayar Komut Kontrolu)
---------------------------------------------------------------------
ESP32'den gelen ham 16-bit PCM sesi alir:
  1) OpenAI Whisper ile metne cevirir (STT)
  2) OpenAI GPT ile cevap uretir (LLM) ve gerekirse bir BILGISAYAR KOMUTU
     tespit eder (ornegin "youtube'u ac")
  3) Komut varsa bir kuyruga (queue) yazar; kullanicinin bilgisayarindaki
     ajan programi bu kuyrugu duzenli araliklarla kontrol edip komutu
     gercekten calistirir
  4) OpenAI TTS ile cevabi tekrar sese cevirir
  5) Ham PCM olarak ESP32'ye geri gonderir

Yerel test icin:
  pip install fastapi uvicorn openai numpy
  set OPENAI_API_KEY=sk-...        (Windows CMD)
  set AGENT_SECRET=herhangi-bir-sifre
  uvicorn server:app --host 0.0.0.0 --port 8000

Render.com'da:
  - API anahtarini kod icine YAZMAYIN, Render dashboard'unda
    "Environment" sekmesinden OPENAI_API_KEY olarak ekleyin.
  - Ayni sekilde AGENT_SECRET adinda, kendi uydurdugunuz bir sifreyi de
    ekleyin (bilgisayarinizdaki ajan programinin kimligini dogrulamak icin).
  - Start command: uvicorn server:app --host 0.0.0.0 --port $PORT
"""

import io
import os
import json
import wave
from fastapi import FastAPI, Request, Response, Header, HTTPException
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY ortam degiskeni bulunamadi. "
        "Yerelde 'set OPENAI_API_KEY=sk-...' ile, Render'da Environment sekmesinden ekleyin."
    )

# Bilgisayardaki ajan programinin /commands uc noktasina erisirken
# kullanacagi basit paylasilan sifre. Boylece baskasi sunucunuzun
# adresini bulsa bile komut kuyrugunuzu goremez/calistiramaz.
AGENT_SECRET = os.environ.get("AGENT_SECRET", "")
if not AGENT_SECRET:
    raise RuntimeError(
        "AGENT_SECRET ortam degiskeni bulunamadi. "
        "Render'da Environment sekmesinden, kendi uydurdugunuz bir sifreyi ekleyin."
    )

openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

conversation_history = []           # basit hafiza, tek kullanici icin
pending_computer_commands = []      # bilgisayardaki ajana gonderilmeyi bekleyen komutlar
pending_relay_commands = []         # ESP32-C3 role kartina gonderilmeyi bekleyen komutlar


# ---------- Rolelerin tanimi ----------
# Buradaki anahtarlar (sol taraf) ESP32-C3 kodundaki RELAY_PINS ile
# BIREBIR AYNI olmali. Sag taraf ise GPT'nin anlamasi ve sesli
# cevapta kullanmasi icin dogal dildeki isim.
RELAY_DEVICES = {
    "salon_isigi": "salon ışığı",
    "fan": "fan",
    "role_3": "3. röle",
    "role_4": "4. röle",
}

SYSTEM_PROMPT = (
    "Sen sesli bir asistansin. Cevaplarin kisa, dogal ve konusma diline uygun olsun. "
    "Uzun listeler veya markdown kullanma, cunku cevabin sesli okunacak. "
    "Kullanici senden bilgisayarinda bir islem yapmani istiyorsa "
    "(ornegin bir web sitesi acmani ya da bir program acmani istiyorsa), "
    "run_computer_command aracini cagir. "
    "Kullanici evdeki bir roleyi (isik, fan vb.) acmani/kapatmani istiyorsa "
    "control_relay aracini cagir. Kontrol edebileceginiz cihazlar: "
    + ", ".join(RELAY_DEVICES.values()) + ". "
    "Sadece gercekten boyle bir istek oldugunda bu araclari cagir; "
    "normal sohbet sorularinda cagirma."
)

# GPT'nin cagirabilecegi araclar (function calling)
COMMAND_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_computer_command",
            "description": (
                "Kullanicinin bilgisayarinda bir islem calistirir: "
                "bir web sitesi acmak ya da bir program/uygulama acmak."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open_url", "open_app"],
                        "description": "open_url: bir web sitesi ac. open_app: bir masaustu programi ac.",
                    },
                    "target": {
                        "type": "string",
                        "description": (
                            "open_url icin alan adi (orn. 'youtube.com'). "
                            "open_app icin program adi (orn. 'spotify', 'chrome', 'word', 'notepad', 'calculator')."
                        ),
                    },
                },
                "required": ["action", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_relay",
            "description": "Evdeki bir roleyi (lamba, fan vb.) acar veya kapatir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "enum": list(RELAY_DEVICES.keys()),
                        "description": "Kontrol edilecek cihazin anahtari.",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["on", "off"],
                    },
                },
                "required": ["device", "state"],
            },
        },
    },
]


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


def transcribe(wav_bytes: bytes) -> str:
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "audio.wav"
    result = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="tr",
    )
    return result.text


def friendly_confirmation(action: str, target: str) -> str:
    if action in ("open_url", "open_app"):
        return f"Tamam, {target} aciliyor."
    return "Tamam, yapiyorum."


def friendly_relay_confirmation(device: str, state: str) -> str:
    label = RELAY_DEVICES.get(device, device)
    if state == "on":
        return f"Tamam, {label} aciliyor."
    return f"Tamam, {label} kapatiliyor."


def ask_gpt(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=messages,
        tools=COMMAND_TOOLS,
        tool_choice="auto",
    )
    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}

        if tool_call.function.name == "run_computer_command":
            action = args.get("action")
            target = args.get("target", "")
            if action in ("open_url", "open_app") and target:
                pending_computer_commands.append({"action": action, "target": target})
                print(f"Bilgisayar komutu kuyruga eklendi: {action} -> {target}")
            reply = friendly_confirmation(action, target)

        elif tool_call.function.name == "control_relay":
            device = args.get("device")
            state = args.get("state")
            if device in RELAY_DEVICES and state in ("on", "off"):
                pending_relay_commands.append({"device": device, "state": state})
                print(f"Role komutu kuyruga eklendi: {device} -> {state}")
            reply = friendly_relay_confirmation(device, state)

        else:
            reply = "Tamam."

        # Gecmise, modelin ne yaptigini de (kisaca) kaydedelim
        conversation_history.append({"role": "assistant", "content": reply})
    else:
        reply = message.content
        conversation_history.append({"role": "assistant", "content": reply})

    if len(conversation_history) > 12:
        del conversation_history[:2]
    return reply


def synthesize_speech(text: str) -> tuple[bytes, int]:
    tts_response = openai_client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="wav",
    )
    wav_bytes = tts_response.read()

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())

    return pcm_data, sample_rate


@app.get("/")
async def health_check():
    # Render'in servisi "ayakta" tutmak icin kullanabilecegi basit bir uc nokta
    return {"status": "ok"}


@app.post("/assistant")
async def assistant(request: Request):
    sample_rate = int(request.headers.get("X-Sample-Rate", "16000"))
    pcm_bytes = await request.body()

    wav_bytes = pcm_to_wav_bytes(pcm_bytes, sample_rate)
    user_text = transcribe(wav_bytes)
    print(f"Kullanici: {user_text}")

    reply_text = ask_gpt(user_text)
    print(f"Asistan: {reply_text}")

    reply_pcm, reply_rate = synthesize_speech(reply_text)

    return Response(
        content=reply_pcm,
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(reply_rate)},
    )


@app.get("/commands/computer")
async def get_computer_commands(x_agent_key: str = Header(default="")):
    """
    Bilgisayardaki ajan programi (windows_agent.py) bu uc noktayi
    duzenli araliklarla kontrol eder.
    """
    if x_agent_key != AGENT_SECRET:
        raise HTTPException(status_code=403, detail="Gecersiz anahtar")

    commands = list(pending_computer_commands)
    pending_computer_commands.clear()
    return {"commands": commands}


@app.get("/commands/relay")
async def get_relay_commands(x_agent_key: str = Header(default="")):
    """
    ESP32-C3 role karti bu uc noktayi duzenli araliklarla kontrol eder.
    """
    if x_agent_key != AGENT_SECRET:
        raise HTTPException(status_code=403, detail="Gecersiz anahtar")

    commands = list(pending_relay_commands)
    pending_relay_commands.clear()
    return {"commands": commands}
