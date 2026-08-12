"""Minimal WOLKE-compatible Socket.IO + HTTP .npy server for BLITZ.

Contract: WOLKE/BLITZ_Receiver_Contract.md
  - Socket.IO: emit send_file_message {file_name}
  - HTTP GET /{token}?filename=... → .npy body
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Callable

import numpy as np
from flask import Flask, abort, request, send_file
from flask_socketio import SocketIO

log = logging.getLogger("evt_sidecar.server")

DEFAULT_TOKEN = "evt"
STACK_NAME = "stack.npy"


class StackPublisher:
    """Holds the latest frame stack and pushes it to connected BLITZ clients."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5055,
        token: str = DEFAULT_TOKEN,
    ) -> None:
        self.host = host
        self.port = port
        self.token = token
        self._stack: np.ndarray | None = None
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._app = Flask("evt_sidecar")
        self._sio = SocketIO(self._app, cors_allowed_origins="*", async_mode="threading")
        self._register()
        self.on_viewer_index: Callable[[int], None] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def has_stack(self) -> bool:
        with self._lock:
            return self._stack is not None

    def set_stack(self, stack: np.ndarray, push: bool = True) -> None:
        """Replace served stack; optionally notify BLITZ to re-download."""
        arr = np.ascontiguousarray(stack)
        with self._lock:
            self._stack = arr
        mb = arr.nbytes / (1024 * 1024)
        log.info("stack ready shape=%s dtype=%s (%.1f MB)", arr.shape, arr.dtype, mb)
        if push:
            self.push()

    def push(self) -> None:
        """Emit send_file_message so BLITZ downloads the current stack."""
        log.info("push %s", STACK_NAME)
        self._sio.emit("send_file_message", {"file_name": STACK_NAME})

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _run() -> None:
            log.info("serving %s (token=%s)", self.base_url, self.token)
            self._sio.run(
                self._app,
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                allow_unsafe_werkzeug=True,
            )

        self._thread = threading.Thread(target=_run, name="evt-sidecar-http", daemon=True)
        self._thread.start()

    def _register(self) -> None:
        app = self._app
        sio = self._sio
        publisher = self

        @app.get("/<tok>")
        def get_file(tok: str):
            if tok != publisher.token:
                return abort(404)
            file_name = request.args.get("filename")
            if not file_name:
                return abort(400)
            with publisher._lock:
                stack = publisher._stack
            if stack is None:
                return abort(404)
            # Serve same payload for stack.npy or any requested name in v1
            buf = io.BytesIO()
            np.save(buf, stack)
            buf.seek(0)
            return send_file(
                buf,
                mimetype="application/octet-stream",
                download_name=file_name if file_name.endswith(".npy") else STACK_NAME,
            )

        @sio.on("connect")
        def on_connect():
            log.info("BLITZ client connected")
            sio.emit("Connected successfully")
            # Auto-push current stack if already binned
            with publisher._lock:
                ready = publisher._stack is not None
            if ready:
                sio.emit("send_file_message", {"file_name": STACK_NAME})

        @sio.on("disconnect")
        def on_disconnect():
            log.info("BLITZ client disconnected")

        @sio.on("viewer_index")
        def on_viewer_index(data):
            idx = data.get("index") if isinstance(data, dict) else None
            if isinstance(idx, int) and publisher.on_viewer_index:
                publisher.on_viewer_index(idx)
