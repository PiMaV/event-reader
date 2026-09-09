"""Minimal WETTER Viewer Contract Socket.IO + HTTP .npy server for BLITZ and DONNER.

Contract: WOLKE/WETTER_Viewer_Contract.md (legacy alias BLITZ_Receiver_Contract.md)
  - Socket.IO: emit send_file_message {file_name[, index]}
  - HTTP GET /{token}?filename=... → .npy body
  - viewer_index from a client is rebroadcast as index seek to other clients
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np
from flask import Flask, abort, request
from flask_socketio import SocketIO

from .npy_http import apply_cors, npy_response

log = logging.getLogger("evt_sidecar.server")

DEFAULT_TOKEN = "evt"
STACK_NAME = "stack.npy"


class StackPublisher:
    """Holds the latest frame stack and pushes it to connected viewers."""

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
        self.on_served: Callable[[int], None] | None = None
        self.on_client_count: Callable[[int], None] | None = None
        self._clients = 0
        self.gzip_enabled = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def has_stack(self) -> bool:
        with self._lock:
            return self._stack is not None

    def set_stack(self, stack: np.ndarray, push: bool = True) -> None:
        """Replace served stack; optionally notify viewers to re-download."""
        arr = np.ascontiguousarray(stack)
        with self._lock:
            self._stack = arr
        mb = arr.nbytes / (1024 * 1024)
        log.info("stack ready shape=%s dtype=%s (%.1f MB)", arr.shape, arr.dtype, mb)
        if push:
            self.push()

    def push(self) -> None:
        """Emit send_file_message so viewers download the current stack."""
        log.info("push %s", STACK_NAME)
        self._sio.emit("send_file_message", {"file_name": STACK_NAME})

    def push_index(self, index: int, *, skip_sid: str | None = None) -> None:
        """Broadcast playhead seek for the current stack (no cube bytes)."""
        payload = {"file_name": STACK_NAME, "index": int(index)}
        log.info("push index %s", index)
        if skip_sid:
            self._sio.emit("send_file_message", payload, skip_sid=skip_sid)
        else:
            self._sio.emit("send_file_message", payload)

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

        @app.after_request
        def add_cors(resp):
            return apply_cors(
                resp,
                origin=request.headers.get("Origin"),
                request_headers=request.headers.get("Access-Control-Request-Headers"),
            )

        @app.route("/<tok>", methods=["OPTIONS"])
        def options_file(tok: str):  # noqa: ARG001
            return ("", 204)

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
            download_name = (
                file_name if file_name.endswith(".npy") else STACK_NAME
            )
            want_gzip = publisher.gzip_enabled or (
                request.args.get("gzip", "").lower() in {"1", "true", "yes"}
            )
            resp, raw_n, wire_n, used_gzip = npy_response(
                stack,
                download_name=download_name,
                accept_encoding=request.headers.get("Accept-Encoding"),
                compress=want_gzip,
            )
            log.info(
                "serve %s raw=%.1f MB wire=%.1f MB encoding=%s",
                download_name,
                raw_n / (1024 * 1024),
                wire_n / (1024 * 1024),
                "gzip" if used_gzip else "identity",
            )
            if publisher.on_served is not None:
                publisher.on_served(wire_n)
            return resp

        @sio.on("connect")
        def on_connect():
            publisher._clients += 1
            log.info("viewer client connected")
            sio.emit("Connected successfully")
            if publisher.on_client_count is not None:
                publisher.on_client_count(publisher._clients)
            with publisher._lock:
                ready = publisher._stack is not None
            if ready:
                sio.emit("send_file_message", {"file_name": STACK_NAME})

        @sio.on("disconnect")
        def on_disconnect():
            publisher._clients = max(0, publisher._clients - 1)
            log.info("viewer client disconnected")
            if publisher.on_client_count is not None:
                publisher.on_client_count(publisher._clients)

        @sio.on("viewer_index")
        def on_viewer_index(data):
            idx = data.get("index") if isinstance(data, dict) else None
            if not isinstance(idx, int):
                return
            if publisher.on_viewer_index:
                publisher.on_viewer_index(idx)
            # Hub-and-spoke: relay playhead to other viewers (skip emitter).
            sid = getattr(request, "sid", None)
            publisher.push_index(idx, skip_sid=sid)
