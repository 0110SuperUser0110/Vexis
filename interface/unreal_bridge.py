from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from typing import Any

from core.schemas import utc_now_iso
class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class UnrealPresenceBridgeServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._lock = Lock()
        self._snapshot: dict[str, Any] = self._default_snapshot()
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    def _default_snapshot(self) -> dict[str, Any]:
        return {
            "timestamp": utc_now_iso(),
            "bridge_online": True,
            "visual_state": "idle",
            "status_text": "ready",
            "is_thinking": False,
            "is_speaking": False,
            "boot_in_progress": False,
            "current_focus": "",
            "last_classification": "",
            "active_presence_action": "",
            "last_response_text": "",
            "last_spoken_text": "",
            "thought_lines": [],
            "open_question_count": 0,
            "unsupported_claim_count": 0,
            "completed_file_count": 0,
        }

    def start(self) -> None:
        if self._server is not None:
            return

        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    payload = {
                        "ok": True,
                        "service": "vexis_unreal_bridge",
                        "timestamp": utc_now_iso(),
                    }
                    self._write_json(payload)
                    return

                if self.path == "/v1/presence":
                    self._write_json(bridge.get_snapshot())
                    return

                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

            def _write_json(self, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded)

        self._server = _ReusableThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True, name="vexis-unreal-bridge")
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return

        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def publish_snapshot(self, snapshot: dict[str, Any]) -> None:
        packed = json.loads(json.dumps(snapshot, ensure_ascii=True))
        packed["timestamp"] = utc_now_iso()
        packed["bridge_online"] = True
        with self._lock:
            self._snapshot = packed

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

