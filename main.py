"""
main.py — Friday Voice AI Assistant
Run from inside the Friday/ folder with: python main.py
"""

import sys
import signal

from stt.groq_stt    import listen
from tts.kokoro_tts  import speak
from agent.assistant import Assistant
from memory.vault    import Vault
from memory.writer   import add_user, add_assistant

WAKE_WORD    = "friday"
EXIT_PHRASES = {"goodbye", "bye", "exit", "quit", "stop", "shut down"}


def _handle_exit(sig, frame):
    print("\n👋 Shutting down Friday. Goodbye!")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, _handle_exit)

    print("=" * 52)
    print("  🤖  Friday — Voice AI Assistant")
    print("=" * 52)
    print(f"  STT       : Groq Whisper (whisper-large-v3-turbo)")
    print(f"  LLM       : Gemini 2.5 Flash")
    print(f"  TTS       : Kokoro (local)")
    print(f"  Wake word : '{WAKE_WORD}' — say it once to activate")
    print("  Ctrl-C to quit.")
    print("=" * 52 + "\n")

    vault     = Vault()
    assistant = Assistant()
    assistant.start_session(vault)

    # ── Two-state loop ────────────────────────────────────────────────────────
    # State 1: SLEEPING — waiting for wake word
    # State 2: ACTIVE   — responding freely, no wake word needed
    active = False

    print("😴 Sleeping... say 'Friday' to wake me up.\n")

    while True:
        text = listen()
        if not text:
            continue

        # ── SLEEPING: wait for wake word ──────────────────────────────────
        if not active:
            if WAKE_WORD in text.lower():
                active = True
                # Check if there's a query after the wake word
                query = text.lower().replace(WAKE_WORD, "", 1).strip()
                print("✅ Activated!\n")
                if not query:
                    speak("Hey! What can I do for you?")
                    continue
                # Fall through to process the inline query
            else:
                continue  # Not awake yet, ignore

        else:
            query = text

        # ── ACTIVE: process every utterance ──────────────────────────────
        # Exit check
        if any(phrase in query.lower() for phrase in EXIT_PHRASES):
            speak("Goodbye! Have a great day.")
            active = False
            print("😴 Sleeping... say 'Friday' to wake me up.\n")
            continue  # Go back to sleep instead of quitting entirely

        # Think and respond
        add_user(vault, query)
        print("⚙️  Thinking...")

        reply = assistant.respond(query, vault)
        add_assistant(vault, reply)

        speak(reply)


if __name__ == "__main__":
    main()