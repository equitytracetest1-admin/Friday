"""
tts/google_tts.py — Text-to-Speech via Kokoro (local, offline)

Synthesises speech entirely on-device using the Kokoro ONNX model.
No internet connection or API key required.

Prerequisites — place these two files in the friday/ project root:
  • kokoro-v1.0.onnx
  • voices-v1.0.bin
Download: https://github.com/thewh1teagle/kokoro-onnx/releases
"""

import numpy as np
import sounddevice as sd

# ── Config ────────────────────────────────────────────────────────────────────
VOICE       = "af_heart"   # Kokoro voice ID  (af_heart, af_sky, am_adam, …)
SPEED       = 1.0          # 1.0 = normal; increase for faster speech
LANG        = "en-us"
SAMPLE_RATE = 24_000       # Kokoro's native output rate

# Paths to the model files (relative to where you run the script)
ONNX_PATH   = "kokoro-v1.0.onnx"
VOICES_PATH = "voices-v1.0.bin"

# ── Lazy model loader ─────────────────────────────────────────────────────────
_kokoro = None

def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(ONNX_PATH, VOICES_PATH)
    return _kokoro


# ── Public API ────────────────────────────────────────────────────────────────

def speak(text: str) -> None:
    """Synthesise `text` with Kokoro and play it through the default audio output."""
    if not text.strip():
        return

    print(f"🔊 Friday: {text}")

    kokoro = _get_kokoro()
    samples, sample_rate = kokoro.create(
        text,
        voice=VOICE,
        speed=SPEED,
        lang=LANG,
    )

    audio = np.array(samples, dtype=np.float32)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
