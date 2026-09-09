"""HTTP + Socket.IO contract smoke (WOLKE-compatible)."""

from __future__ import annotations

import gzip
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
            assert resp.headers.get("Content-Encoding") != "gzip"
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        arr = np.load(io.BytesIO(body))
        assert arr.shape == stack.shape
        assert np.allclose(arr, stack)
    finally:
        test_client.disconnect()


def test_viewer_index_rebroadcasts_seek() -> None:
    pub = StackPublisher(host="127.0.0.1", port=5069, token="evt")
    pub.start_background()
    time.sleep(0.5)
    stack = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    pub.set_stack(stack, push=False)

    a = pub._sio.test_client(pub._app, flask_test_client=pub._app.test_client())
    b = pub._sio.test_client(pub._app, flask_test_client=pub._app.test_client())
    try:
        a.get_received()
        b.get_received()
        a.emit("viewer_index", {"index": 1})
        time.sleep(0.1)
        received_b = b.get_received()
        names = [pkt["name"] for pkt in received_b]
        assert "send_file_message" in names
        msg = next(pkt for pkt in received_b if pkt["name"] == "send_file_message")
        assert msg["args"][0] == {"file_name": STACK_NAME, "index": 1}
        # Emitter should not see its own seek echoed (skip_sid).
        echoed = [
            pkt
            for pkt in a.get_received()
            if pkt["name"] == "send_file_message"
            and pkt["args"]
            and pkt["args"][0].get("index") == 1
        ]
        assert echoed == []
    finally:
        a.disconnect()
        b.disconnect()


def test_publisher_gzip_is_opt_in() -> None:
    pub = StackPublisher(host="127.0.0.1", port=5068, token="evt")
    stack = np.zeros((16, 32, 32), dtype=np.uint8)
    stack[0, 0, 0] = 1
    stack[3, 10, 10] = 255
    pub.set_stack(stack, push=False)

    client = pub._app.test_client()
    # BLITZ/requests always send Accept-Encoding: gzip — that alone must not zip
    identity = client.get(
        f"/evt?filename={STACK_NAME}",
        headers={"Accept-Encoding": "gzip"},
    )
    assert identity.status_code == 200
    assert identity.headers.get("Content-Encoding") != "gzip"
    assert identity.headers.get("Access-Control-Allow-Origin") == "*"
    assert identity.headers.get("Access-Control-Allow-Private-Network") == "true"
    ident = np.load(io.BytesIO(identity.data))
    assert np.array_equal(ident, stack)

    zipped = client.get(f"/evt?filename={STACK_NAME}&gzip=1")
    assert zipped.status_code == 200
    assert zipped.headers.get("Content-Encoding") == "gzip"
    raw = gzip.decompress(zipped.data)
    arr = np.load(io.BytesIO(raw))
    assert arr.shape == stack.shape
    assert np.array_equal(arr, stack)
    assert len(zipped.data) < len(raw)

    preflight = client.options(
        f"/evt?filename={STACK_NAME}",
        headers={
            "Origin": "https://lab.ole.icu",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert preflight.status_code == 204
    assert preflight.headers.get("Access-Control-Allow-Origin") == "https://lab.ole.icu"
    assert preflight.headers.get("Access-Control-Allow-Private-Network") == "true"

    cross = client.get(
        f"/evt?filename={STACK_NAME}",
        headers={"Origin": "https://lab.ole.icu"},
    )
    assert cross.status_code == 200
    assert cross.headers.get("Access-Control-Allow-Origin") == "https://lab.ole.icu"
    assert cross.headers.get("Access-Control-Allow-Private-Network") == "true"
