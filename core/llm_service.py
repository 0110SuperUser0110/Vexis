from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LLMServiceConfig:
    server_path: str = r"E:\Vexis\bin\llama.cpp\llama-server.exe"
    model_path: str = r"E:\Vexis\models\text\qwen3_8b\Qwen3-8B-Q4_K_M.gguf"
    host: str = "127.0.0.1"
    port: int = 8080
    gpu_layers: int = 999
    context_size: int = 4096
    temperature: float = 0.05


class LLMService:
    def __init__(self, config: Optional[LLMServiceConfig] = None) -> None:
        self.config = config or LLMServiceConfig()
        self.process: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def is_running(self) -> bool:
        if self.process is not None and self.process.poll() is not None:
            self.process = None
            return False

        return self._port_open(self.config.host, self.config.port)

    def start(self, timeout_seconds: int = 90) -> None:
        if self.is_running():
            return

        server_path = Path(self.config.server_path)
        model_path = Path(self.config.model_path)

        if not server_path.exists():
            raise FileNotFoundError(f"llama-server not found: {server_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")

        command = [
            str(server_path),
            "-m",
            str(model_path),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "-ngl",
            str(self.config.gpu_layers),
            "-c",
            str(self.config.context_size),
            "--temp",
            str(self.config.temperature),
            "--jinja",
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self.process.poll() is not None:
                code = self.process.returncode
                self.process = None
                raise RuntimeError(f"llama-server exited during startup with code {code}")

            if self._port_open(self.config.host, self.config.port):
                return

            time.sleep(0.5)

        raise TimeoutError("llama-server did not become ready in time")

    def stop(self, timeout_seconds: int = 10) -> None:
        if self.process is None:
            return

        if self.process.poll() is not None:
            self.process = None
            return

        self.process.terminate()
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            if self.process.poll() is not None:
                self.process = None
                return
            time.sleep(0.2)

        self.process.kill()
        self.process = None

    def _port_open(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            return False