from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    import winsound
except ImportError:
    winsound = None  # type: ignore


class PiperTTSEngine:
    """
    Piper CLI-based offline neural TTS engine for VEXIS.
    Uses: python -m piper
    This avoids PATH issues inside Visual Studio.
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        speaker: Optional[int] = 0,
        use_cuda: bool = False,
        piper_exe: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.speaker = speaker
        self.use_cuda = use_cuda

        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {self.model_path}")

        if self.config_path is not None and not self.config_path.exists():
            raise FileNotFoundError(f"Piper config not found: {self.config_path}")

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return

        if winsound is None:
            raise RuntimeError("winsound is unavailable on this system.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            wav_path = Path(temp_wav.name)

        try:
            command = [
                sys.executable,
                "-m",
                "piper",
                "--model",
                str(self.model_path),
                "--config",
                str(self.config_path) if self.config_path else "",
                "--output_file",
                str(wav_path),
            ]

            if self.speaker is not None:
                command.extend(["--speaker", str(self.speaker)])

            if self.use_cuda:
                command.append("--cuda")

            command = [arg for arg in command if arg != ""]

            result = subprocess.run(
                command,
                input=text,
                text=True,
                capture_output=True,
                check=False,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip() if result.stderr else "unknown Piper CLI error"
                raise RuntimeError(f"Piper CLI failed: {stderr}")

            if not wav_path.exists() or wav_path.stat().st_size == 0:
                raise RuntimeError("Piper CLI did not produce a valid WAV file.")

            winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)

        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

    def stop(self) -> None:
        if winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)