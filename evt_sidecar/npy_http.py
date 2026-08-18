"""Serve a NumPy array as .npy, optionally gzip-compressed (WOLKE/BLITZ contract)."""

from __future__ import annotations

import gzip
import io

import numpy as np
from flask import Response


def accepts_gzip(accept_encoding: str | None) -> bool:
    if not accept_encoding:
        return False
    for part in accept_encoding.split(","):
        token = part.strip().split(";", 1)[0].strip().lower()
        if token == "gzip":
            return True
    return False


def npy_response(
    arr: np.ndarray,
    *,
    download_name: str,
    accept_encoding: str | None,
    compress: bool = False,
) -> tuple[Response, int, int, bool]:
    """Return (response, raw_npy_bytes, wire_bytes, used_gzip).

    Gzip is **opt-in** (``compress=True``). Default is raw ``.npy`` — localhost
    zip+unzip is wasted CPU. The Event reader checkbox and ``?gzip=1`` set this.
    """
    buf = io.BytesIO()
    np.save(buf, arr)
    raw = buf.getvalue()
    raw_n = len(raw)
    used_gzip = bool(compress)
    if used_gzip:
        payload = gzip.compress(raw, compresslevel=6)
        headers = {
            "Content-Encoding": "gzip",
            "Vary": "Accept-Encoding",
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "Content-Length": str(len(payload)),
        }
    else:
        payload = raw
        headers = {
            "Vary": "Accept-Encoding",
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "Content-Length": str(len(payload)),
        }
    resp = Response(payload, mimetype="application/octet-stream", headers=headers)
    return resp, raw_n, len(payload), used_gzip
