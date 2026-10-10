"""
server.py: a small local web server for the observer page.

It serves these on http://localhost:<port>/ :
    /              the observer page (ui/static/index.html)
    /state         the latest suggestion and status, as JSON
    /minimap.jpg   the latest minimap picture with the suggestion drawn on
    /killfeed.jpg  the latest kill-feed picture, cut from the broadcast

The live runner (scripts/run_live.py) calls ObserverServer.publish() every
frame; the page asks for /state and /minimap.jpg several times a second.
Only Python's standard library is used, so nothing extra has to be
installed. It only listens on this computer (127.0.0.1).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).parent / "static" / "index.html"


class ObserverServer:
    """Holds the latest state and serves it in a background thread."""

    def __init__(self, port: int = 8765):
        self.port = port
        self._lock = threading.Lock()        # the page reads while the runner writes
        self._state: dict = {"status": "starting"}
        self._images: dict[str, bytes] = {}   # "/minimap.jpg" -> JPEG bytes, ...
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"

    def start(self) -> None:
        """Start answering requests (returns immediately)."""
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()

    def publish(self, state: dict, jpeg: bytes | None = None, images: dict[str, bytes] | None = None) -> None:
        """Replace the state and pictures the page shows.

        jpeg:   the minimap picture (served at /minimap.jpg)
        images: any other pictures, by name, e.g. {"killfeed.jpg": ...}
        """
        with self._lock:
            self._state = state
            if jpeg is not None:
                self._images["/minimap.jpg"] = jpeg
            for name, data in (images or {}).items():
                self._images["/" + name] = data

    def _handler(self):
        """Build the request handler class, bound to this server's state."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (name required by http.server)
                path = self.path.split("?")[0]   # the page adds ?t=... to avoid caching
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
                elif path == "/state":
                    with server._lock:
                        body = json.dumps(server._state).encode()
                    self._send(200, "application/json", body)
                elif path.endswith(".jpg"):
                    with server._lock:
                        body = server._images.get(path, b"")
                    self._send(200 if body else 404, "image/jpeg", body)
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code: int, kind: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass   # keep the terminal quiet; the page asks many times a second

        return Handler
