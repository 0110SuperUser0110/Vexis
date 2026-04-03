from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class TTSEngine:
    """
    Backend loader for VEXIS speech.
    Chooses a speech backend from config/voices.json.
    """

    def __init__(self, config_path: str = "config/voices.json") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()

        backend = str(self.config.get("backend", "windows")).lower()

        if backend == "piper":
            from speech.piper_tts import PiperTTSEngine

            self.engine = PiperTTSEngine(
                piper_exe=self.config.get("piper_exe"),
                model_path=self.config["model_path"],
                config_path=self.config.get("config_path"),
                speaker=self.config.get("speaker"),
                use_cuda=bool(self.config.get("use_cuda", False)),
            )
        else:
            from speech.windows_tts import WindowsTTSEngine

            self.engine = WindowsTTSEngine()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {"backend": "windows"}

        with self.config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def speak(self, text: str) -> None:
        self.engine.speak(text)

    def stop(self) -> None:
        self.engine.stop()
