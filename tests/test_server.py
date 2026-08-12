"""HTTP + Socket.IO contract smoke (WOLKE-compatible)."""

from __future__ import annotations

import io
import time
import urllib.request

import numpy as np

from evt_sidecar.server import STACK_NAME, StackPublisher


def test_publisher_push_and_download() -> None:
    pub = StackPublisher(host="127.0.0.1", port=5067, token="evt")
    pub.start_background()
    time.sleep(0.5)

    stack = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    test_client = pub._sio.test_client(pub._app, flask_test_client=pub._app.test_client())
    try:
        # clear connect handshake noise
        test_client.get_received()
        pub.set_stack(stack, push=True)
        time.sleep(0.1)
        received = test_client.get_received()
        names = [pkt["name"] for pkt in received]
        assert "send_file_message" in names
        msg = next(pkt for pkt in received if pkt["name"] == "send_file_message")
        assert msg["args"][0]["file_name"] == STACK_NAME

        with urllib.request.urlopen(
            f"http://127.0.0.1:5067/evt?filename={STACK_NAME}", timeout=5
        ) as resp:
            body = resp.read()
        arr = np.load(io.BytesIO(body))
        assert arr.shape == stack.shape
        assert np.allclose(arr, stack)
    finally:
        test_client.disconnect()
