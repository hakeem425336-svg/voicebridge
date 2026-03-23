from __future__ import annotations
import os
import tempfile
import time
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =========================
# ⚙️ CONFIG
# =========================
@dataclass
class AppConfig:
    groq_api_key: str
    groq_model_id: str
    whisper_model: str
    whisper_language: str
    tts_lang: str
    system_prompt: str


CFG = AppConfig(
    groq_api_key=os.getenv("API_KEY", ""),
    groq_model_id=os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile"),
    whisper_model=os.getenv("WHISPER_MODEL", "tiny"),
    whisper_language=os.getenv("WHISPER_LANGUAGE", "en"),
    tts_lang=os.getenv("GTTS_LANG", "en"),
    system_prompt=os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful assistant. Keep replies short and clear."
    ),
)

# =========================
# 🎧 ASR
# =========================
@st.cache_resource
def load_model():
    from faster_whisper import WhisperModel
    return WhisperModel(
        CFG.whisper_model,
        device="cpu",
        compute_type="int8"
    )


def transcribe(audio_bytes):
    model = load_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    segments, _ = model.transcribe(
        path,
        language=CFG.whisper_language,
        vad_filter=True
    )

    text = "".join([seg.text for seg in segments]).strip()
    os.remove(path)
    return text


# =========================
# 🤖 GROQ LLM (FIXED)
# =========================
def generate_reply(user_text, history):
    import requests

    if not CFG.groq_api_key:
        return f"(Demo Mode) {user_text}"

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {CFG.groq_api_key}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "system", "content": CFG.system_prompt}]
    messages += history[-6:]
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": CFG.groq_model_id,
        "messages": messages,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)

        if r.status_code != 200:
            return f"API ERROR: {r.text}"

        data = r.json()

        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")

    except Exception as e:
        return f"Error: {str(e)}"


# =========================
# 🔊 TTS
# =========================
def tts(text):
    from gtts import gTTS

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name

    tts = gTTS(text=text, lang=CFG.tts_lang)
    tts.save(path)

    audio = open(path, "rb").read()
    os.remove(path)
    return audio


# =========================
# 🎨 UI
# =========================
st.set_page_config(page_title="VoiceBridge AI", layout="centered")

st.title("🎙️ VoiceBridge AI")
st.caption("Urdu • Arabic • English Voice Assistant")

# 🌍 Language selection
lang = st.selectbox("🌍 Language", ["English", "Urdu", "Arabic"])

if lang == "Urdu":
    CFG.whisper_language = "ur"
    CFG.tts_lang = "ur"
elif lang == "Arabic":
    CFG.whisper_language = "ar"
    CFG.tts_lang = "ar"
else:
    CFG.whisper_language = "en"
    CFG.tts_lang = "en"

CFG.system_prompt = f"Reply ONLY in {lang}. Keep answers short and clear."

# 🧹 Clear Chat
if st.button("🗑️ Clear Chat"):
    st.session_state.chat = []
    st.rerun()

if "chat" not in st.session_state:
    st.session_state.chat = []

# 💬 Chat display
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 🎤 Input
audio = st.audio_input("🎤 Speak")

# =========================
# ⚡ PIPELINE
# =========================
if audio is not None:

    st.write("DEBUG: Audio received")

    wav = audio.getvalue()

    # 🎧 ASR
    with st.spinner("🎧 Transcribing..."):
        t1 = time.time()
        text = transcribe(wav)
        t_asr = time.time() - t1

    st.write("DEBUG: Transcription:", text)

    if not text:
        st.error("❌ No speech detected")
        st.stop()

    # 👤 USER MESSAGE
    with st.chat_message("user"):
        st.write(text)

    # 🤖 LLM
    with st.spinner("🤖 Thinking..."):
        t2 = time.time()
        reply = generate_reply(text, st.session_state.chat)
        t_llm = time.time() - t2

    # 🤖 AI MESSAGE
    with st.chat_message("assistant"):
        st.write(reply)

    # save history
    st.session_state.chat.append({"role": "user", "content": text})
    st.session_state.chat.append({"role": "assistant", "content": reply})

    # 🔊 TTS
    with st.spinner("🔊 Speaking..."):
        t3 = time.time()
        audio_bytes = tts(reply)
        t_tts = time.time() - t3

    st.audio(audio_bytes)

    # ⏱️ Latency
    st.caption(f"ASR: {t_asr:.2f}s | LLM: {t_llm:.2f}s | TTS: {t_tts:.2f}s")