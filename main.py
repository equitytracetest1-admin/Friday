"""
main.py — Friday Voice AI Assistant
Run from inside the Friday/ folder with:

    python main.py              → hybrid mode (mic + keyboard both active)
    python main.py --voice      → voice-only mode (mic, TTS, wake word)
    python main.py --text       → text mode (no mic, no TTS — for testing)
"""

import sys
import signal
import re
import threading
import queue

from agent.assistant import Assistant
from memory.vault    import Vault
from memory.writer   import add_user, add_assistant, load_last_session

WAKE_WORD    = "friday"
EXIT_PHRASES = {"goodbye", "bye", "exit", "quit", "stop", "shut down", "go to sleep", "peace out"}


def _handle_exit(sig, frame):
    print("\n👋 i'm going to sleep boss. See ya!")
    sys.exit(0)


def _is_exit_phrase(text: str) -> bool:
    return any(re.search(rf'\b{re.escape(p)}\b', text.lower()) for p in EXIT_PHRASES)


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 2 — TEXT MODE (no mic, no TTS)
# ══════════════════════════════════════════════════════════════════════════════

def text_mode(vault: Vault, assistant: Assistant) -> None:
    """
    Level 2 testing — type directly to Friday in the terminal.
    Full LLM + skill + memory pipeline runs exactly as in voice mode.
    No microphone, no Kokoro TTS. Ctrl-C to quit.
    """
    print("=" * 52)
    print("  🖥️   Friday — Text Mode")
    print("=" * 52)
    print("  LLM       : Groq LLaMA (llama-3.3-70b-versatile)")
    print("  Skills    : All skills active")
    print("  Memory    : Session + long-term memory active")
    print("  TTS       : Disabled (text only)")
    print("  Ctrl-C    : Quit")
    print("=" * 52)
    print()
    print("  Tip: type naturally — same as you would say it to Friday.")
    print("  Exit phrases (bye, goodbye, etc.) still work.")
    print()

    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            if _is_exit_phrase(query):
                print("Friday: Goodbye! Have a great day.")
                print("\n😴 Session ended.\n")
                break

            add_user(vault, query)
            print("⚙️  Thinking...\n")

            reply = assistant.respond(query, vault)
            add_assistant(vault, reply)

            print(f"Friday: {reply}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Bye, Boss.\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 3 — VOICE MODE (mic only, wake word required)
# ══════════════════════════════════════════════════════════════════════════════

def voice_mode(vault: Vault, assistant: Assistant) -> None:
    """
    Level 3 — full voice pipeline.
    Mic → Groq Whisper STT → LLM → Kokoro TTS.
    """
    from stt.groq_stt   import listen
    from tts.kokoro_tts import speak

    signal.signal(signal.SIGINT, _handle_exit)

    print("=" * 52)
    print("  🤖  Friday — Voice Mode")
    print("=" * 52)
    print(f"  STT       : Groq Whisper (whisper-large-v3-turbo)")
    print(f"  LLM       : Groq LLaMA (llama-3.3-70b-versatile)")
    print(f"  TTS       : Kokoro (local)")
    print(f"  Wake word : '{WAKE_WORD}' — say it once to activate")
    print("  Ctrl-C    : Quit")
    print("=" * 52 + "\n")

    active = False
    print("😴 Sleeping... say 'Friday' to wake me up.\n")

    while True:
        text = listen()
        if not text:
            continue

        if not active:
            if WAKE_WORD in text.lower():
                active = True
                query  = text.lower().replace(WAKE_WORD, "", 1).strip()
                print("✅ Activated!\n")
                if not query:
                    speak("Friday, Online boss. What can I do for you?")
                    continue
            else:
                continue
        else:
            query = text

        if _is_exit_phrase(query):
            speak("Goodbye! Have a great day.")
            active = False
            print("😴 Sleeping... say 'Friday' to wake up.\n")
            continue

        add_user(vault, query)
        print("⚙️  Thinking...")

        reply = assistant.respond(query, vault)
        add_assistant(vault, reply)

        speak(reply)


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 4 — HYBRID MODE (mic + keyboard both active in one session)
# ══════════════════════════════════════════════════════════════════════════════
#
# Two background producer threads (mic listener, keyboard reader) push queries
# onto a shared queue. The main thread is the single consumer — it pulls one
# query at a time and runs it through Assistant.respond(), because the
# Assistant/Vault objects aren't safe for concurrent calls from two threads.
#
# Voice queries still require the wake word ("Friday") to activate, same as
# voice_mode. Typed queries are always active — no wake word needed, since
# typing is already an explicit, deliberate action.
#
# Every reply is printed AND spoken, regardless of which input it came from.

_INPUT_Q: "queue.Queue[tuple[str, str]]" = queue.Queue()  # (source, text)


def _keyboard_producer() -> None:
    """Reads typed input, forever, and pushes it to the shared queue."""
    while True:
        try:
            line = input()
        except EOFError:
            break
        line = line.strip()
        if line:
            _INPUT_Q.put(("text", line))


def _mic_producer() -> None:
    """Listens on the mic, forever, applies wake-word gating, pushes to queue."""
    from stt.groq_stt import listen

    active = False
    while True:
        text = listen()
        if not text:
            continue

        if not active:
            if WAKE_WORD in text.lower():
                active = True
                query = text.lower().replace(WAKE_WORD, "", 1).strip()
                _INPUT_Q.put(("voice_wake", query))
            continue
        else:
            _INPUT_Q.put(("voice", text))

        if _is_exit_phrase(text):
            active = False


def hybrid_mode(vault: Vault, assistant: Assistant) -> None:
    """
    Mic and keyboard both active at once. Type any time — no wake word needed
    for typed input. Speak "Friday" to wake the mic for voice input.
    Replies are always printed and spoken. Ctrl-C to quit.
    """
    from tts.kokoro_tts import speak

    signal.signal(signal.SIGINT, _handle_exit)

    print("=" * 52)
    print("  🤖⌨️   Friday — Hybrid Mode (mic + keyboard)")
    print("=" * 52)
    print("  Type any time — no wake word needed for typed input.")
    print(f"  Say '{WAKE_WORD}' to activate the mic for voice input.")
    print("  Replies are printed and spoken either way.")
    print("  Ctrl-C : Quit")
    print("=" * 52 + "\n")

    threading.Thread(target=_keyboard_producer, daemon=True).start()
    threading.Thread(target=_mic_producer, daemon=True).start()

    while True:
        source, query = _INPUT_Q.get()

        if not query:
            if source == "voice_wake":
                speak("Friday, online boss. What can I do for you?")
            continue

        if _is_exit_phrase(query):
            print("Friday: Goodbye! Have a great day.\n")
            speak("Goodbye! Have a great day.")
            continue

        if source == "text":
            print(f"You: {query}")
        else:
            print(f"🗣️  You: {query}")

        add_user(vault, query)
        print("⚙️  Thinking...")

        reply = assistant.respond(query, vault)
        add_assistant(vault, reply)

        print(f"Friday: {reply}\n")
        speak(reply)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    vault     = Vault()
    load_last_session(vault)
    assistant = Assistant()
    assistant.start_session(vault)

    if "--text" in sys.argv:
        text_mode(vault, assistant)
    elif "--voice" in sys.argv:
        voice_mode(vault, assistant)
    else:
        hybrid_mode(vault, assistant)


if __name__ == "__main__":
    main()