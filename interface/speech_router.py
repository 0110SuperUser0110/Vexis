from __future__ import annotations

from speech.speech_governor import SpeechGovernor
from speech.tts_engine import TTSEngine
from speech.voice_queue import VoiceQueue


class SpeechRouter:
    """
    Routes approved speech to the configured TTS engine.
    """

    def __init__(self, tts_engine: TTSEngine, governor: SpeechGovernor, queue: VoiceQueue) -> None:
        self.tts_engine = tts_engine
        self.governor = governor
        self.queue = queue

    def say(self, text: str, speech_type: str = "status", user_present: bool = True) -> tuple[bool, str]:
        allowed, reason = self.governor.can_speak(
            speech_type=speech_type,
            user_present=user_present,
        )
        if not allowed:
            return False, reason

        self.queue.enqueue(text=text, speech_type=speech_type)
        item = self.queue.dequeue()
        if item is None:
            return False, "queue failure"

        self.tts_engine.speak(item.text)
        self.governor.mark_spoken(speech_type)
        return True, "spoken"