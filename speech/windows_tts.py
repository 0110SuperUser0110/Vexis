from __future__ import annotations

from typing import Optional

try:
    import pyttsx3
except ModuleNotFoundError:
    pyttsx3 = None  # type: ignore


class WindowsTTSEngine:
    """
    Simple Windows fallback TTS using pyttsx3.
    """

    def __init__(self, rate: int = 180, volume: float = 1.0, voice_id: Optional[str] = None) -> None:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not installed for the Windows TTS backend.")

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        if voice_id:
            self.engine.setProperty("voice", voice_id)

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return
        self.engine.say(text)
        self.engine.runAndWait()

    def stop(self) -> None:
        self.engine.stop()
