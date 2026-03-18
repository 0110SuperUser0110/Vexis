from __future__ import annotations

import json
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


@dataclass
class STTResult:
    success: bool
    text: str
    language: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class STTEngine:
    """
    Local speech-to-text engine for VEX.

    Default design:
    - local transcription
    - push-to-talk friendly
    - optimized for conversational use
    - can also transcribe saved audio files
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16",
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.channels = channels

        self.model = WhisperModel(
            model_size_or_path=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = "en",
        beam_size: int = 3,
    ) -> STTResult:
        path = Path(audio_path)
        if not path.exists():
            return STTResult(
                success=False,
                text="",
                error=f"audio file not found: {audio_path}",
            )

        try:
            info = sf.info(str(path))
            duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0

            segments, detected = self.model.transcribe(
                str(path),
                language=language,
                beam_size=beam_size,
                vad_filter=True,
            )

            text = " ".join(segment.text.strip() for segment in segments).strip()

            return STTResult(
                success=True,
                text=text,
                language=detected.language if detected else language,
                duration_seconds=duration,
                metadata={
                    "audio_path": str(path),
                    "probability": getattr(detected, "language_probability", None),
                },
            )

        except Exception as exc:
            return STTResult(
                success=False,
                text="",
                error=str(exc),
                metadata={"audio_path": str(path)},
            )

    def record_and_transcribe(
        self,
        seconds: float = 5.0,
        language: Optional[str] = "en",
        beam_size: int = 3,
    ) -> STTResult:
        try:
            recording = sd.rec(
                int(seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
            )
            sd.wait()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            sf.write(str(tmp_path), recording, self.sample_rate)
            try:
                result = self.transcribe_file(
                    audio_path=str(tmp_path),
                    language=language,
                    beam_size=beam_size,
                )
                result.metadata["recorded_seconds"] = seconds
                return result
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        except Exception as exc:
            return STTResult(
                success=False,
                text="",
                error=str(exc),
            )

    def transcribe_pcm_bytes(
        self,
        pcm_bytes: bytes,
        language: Optional[str] = "en",
        beam_size: int = 3,
    ) -> STTResult:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                with wave.open(str(tmp_path), "wb") as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(pcm_bytes)

                return self.transcribe_file(
                    audio_path=str(tmp_path),
                    language=language,
                    beam_size=beam_size,
                )
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        except Exception as exc:
            return STTResult(
                success=False,
                text="",
                error=str(exc),
            )

    def save_config(self, path: str) -> None:
        data = {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")