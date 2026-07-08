"""
stt/groq_stt.py — Speech-to-Text via Groq Whisper

Records from the mic using sounddevice, then sends a WAV buffer
to Groq's whisper-large-v3-turbo endpoint.
"""

import io
import wave
import numpy as np
import sounddevice as sd
from groq import Groq
from dotenv import load_dotenv

load_dotenv(".env.local")

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16_000   # Hz — Whisper native rate
SILENCE_RMS      = 0.01     # amplitude threshold for silence detection
SILENCE_SECS     = 1.5      # seconds of silence before stop
MIN_SPEECH_SECS  = 0.4      # minimum speech length to bother transcribing
CHUNK_SECS       = 0.1      # read granularity (100 ms)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _record() -> np.ndarray:
    """Block until the user speaks, then record until silence. Returns float32 audio."""
    chunk_frames   = int(SAMPLE_RATE * CHUNK_SECS)
    silence_chunks = int(SILENCE_SECS / CHUNK_SECS)

    audio_chunks   : list[np.ndarray] = []
    silent_count   = 0
    started        = False

    print("🎙️  Listening...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(chunk_frames)
            flat = chunk.flatten()
            rms  = float(np.sqrt(np.mean(flat ** 2)))

            if rms > SILENCE_RMS:
                started = True
                silent_count = 0
                audio_chunks.append(flat)
            elif started:
                audio_chunks.append(flat)
                silent_count += 1
                if silent_count >= silence_chunks:
                    break

    return np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype="float32")


def _to_wav_bytes(audio: np.ndarray) -> io.BytesIO:
    """Convert float32 numpy array → 16-bit PCM WAV in a BytesIO buffer."""
    pcm = (audio * 32_767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    buf.name = "audio.wav"
    return buf


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray) -> str:
    """Send audio to Groq Whisper and return the transcript string."""
    if len(audio) < SAMPLE_RATE * MIN_SPEECH_SECS:
        return ""

    client = Groq()
    result = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=_to_wav_bytes(audio),
        language="en",
    )
    return result.text.strip()


def listen() -> str:
    """
    Full pipeline: record from mic → transcribe via Groq.
    Returns the transcribed string (empty string if nothing usable was captured).
    """
    audio = _record()
    if not len(audio):
        return ""

    print("⚙️  Transcribing...")
    text = transcribe(audio)

    if text:
        print(f"🗣️  You: {text}")

    return text

