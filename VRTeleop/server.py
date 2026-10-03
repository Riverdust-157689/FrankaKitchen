from __future__ import annotations

import asyncio
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .protocol import LatestControllerState, parse_controller_packet


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


class TeleopServer:
    """Serve the local WebXR page and receive right-controller packets."""

    def __init__(
        self,
        latest: LatestControllerState,
        host: str = "0.0.0.0",
        http_port: int = 8000,
        websocket_port: int = 8001,
        web_root: Path | None = None,
    ) -> None:
        self.latest = latest
        self.host = host
        self.http_port = http_port
        self.websocket_port = websocket_port
        self.web_root = web_root or Path(__file__).with_name("web-ui")
        self._http: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()

    async def _handle_websocket(self, websocket: Any) -> None:
        async for message in websocket:
            try:
                self.latest.set(parse_controller_packet(json.loads(message), hand="right"))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    def _run_websocket(self) -> None:
        import websockets

        async def serve() -> None:
            async with websockets.serve(
                self._handle_websocket,
                self.host,
                self.websocket_port,
                max_size=1 << 20,
                ping_interval=20,
            ):
                self._ready.set()
                await asyncio.to_thread(self._stop.wait)

        loop = asyncio.new_event_loop()
        self._ws_loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(serve())
        except (asyncio.CancelledError, RuntimeError):
            pass
        finally:
            loop.close()

    def start(self) -> None:
        if not self.web_root.is_dir():
            raise FileNotFoundError(f"WebXR assets not found: {self.web_root}")
        self._ready.clear()
        self._stop.clear()
        handler = partial(_QuietHandler, directory=str(self.web_root))
        self._http = ThreadingHTTPServer((self.host, self.http_port), handler)
        self._http_thread = threading.Thread(target=self._http.serve_forever, daemon=True)
        self._http_thread.start()
        self._ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self._ws_thread.start()
        if not self._ready.wait(timeout=5):
            self.close()
            raise RuntimeError("WebSocket server did not start within 5 seconds")

    def close(self) -> None:
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
            self._http = None
        self._stop.set()
        if self._http_thread is not None:
            self._http_thread.join(timeout=2)
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=2)
