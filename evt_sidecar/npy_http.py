"""Serve a NumPy array as .npy, optionally gzip-compressed (WOLKE viewer contract)."""

from __future__ import annotations

import gzip
import io

import numpy as np
from flask import Response

# Browser viewers (DONNER) fetch from another origin than the sidecar.
# Echo Origin (not always *) so Chrome Local Network Access from a public
# HTTPS page (lab.ole.icu) to loopback can pass the private-network preflight.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Private-Network": "true",
}


def apply_cors(
    resp: Response,
    *,
    origin: str | None = None,
    request_headers: str | None = None,
) -> Response:
    allow_origin = origin.strip() if origin and origin.strip() else "*"
    resp.headers["Access-Control-Allow-Origin"] = allow_origin
    resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = (
        request_headers.strip() if request_headers and request_headers.strip() else "Content-Type"
    )
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    # * + credentials is invalid CORS; Socket.IO middleware may have set this.
    resp.headers.pop("Access-Control-Allow-Credentials", None)
    vary = {part.strip() for part in (resp.headers.get("Vary") or "").split(",") if part.strip()}
    vary.update({"Origin", "Access-Control-Request-Headers"})
    resp.headers["Vary"] = ", ".join(sorted(vary))
    return resp


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
            **CORS_HEADERS,
        }
    else:
        payload = raw
        headers = {
            "Vary": "Accept-Encoding",
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "Content-Length": str(len(payload)),
            **CORS_HEADERS,
        }
    resp = Response(payload, mimetype="application/octet-stream", headers=headers)
    return resp, raw_n, len(payload), used_gzip
