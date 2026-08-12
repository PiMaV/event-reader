"""EVT 3.0 decoder (little-endian) — no Metavision SDK dependency.

Word layout from Prophesee EVT 3.0 documentation. IMX636 / IDS RAW uses LE.
Hot path is Numba-accelerated (count pass + fill pass).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from .raw_header import RawHeader, parse_raw_header


@dataclass
class EventStore:
    """Decoded CD events plus sensor geometry."""

    t: np.ndarray  # uint64, microseconds
    x: np.ndarray  # uint16
    y: np.ndarray  # uint16
    p: np.ndarray  # uint8, 0=OFF, 1=ON
    width: int
    height: int
    header: RawHeader

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def t_min(self) -> int:
        return int(self.t[0]) if len(self) else 0

    @property
    def t_max(self) -> int:
        return int(self.t[-1]) if len(self) else 0

    @property
    def duration_us(self) -> int:
        if len(self) < 2:
            return 0
        return int(self.t[-1] - self.t[0])


def load_evt3_raw(path: Path | str, max_events: int | None = None) -> EventStore:
    """Parse header and decode EVT3 payload into column arrays."""
    path = Path(path)
    header = parse_raw_header(path)
    fmt_u = header.format_name.upper()
    if "EVT3" not in fmt_u and not header.evt_version.startswith("3"):
        raise ValueError(
            f"Unsupported format {header.format_name!r} / evt {header.evt_version!r}"
        )
    payload = np.memmap(path, dtype="<u2", mode="r", offset=header.header_bytes)
    words = np.asarray(payload, dtype=np.uint16).ravel()
    t, x, y, p = decode_evt3_words(words, max_events=max_events)
    return EventStore(
        t=t, x=x, y=y, p=p, width=header.width, height=header.height, header=header
    )


def decode_evt3_words(
    words: np.ndarray,
    max_events: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode little-endian EVT3 16-bit words → t,x,y,p arrays."""
    words = np.ascontiguousarray(words, dtype=np.uint16)
    limit = -1 if max_events is None else int(max_events)
    n = int(_count_evt3(words, limit))
    t_out = np.empty(n, dtype=np.uint64)
    x_out = np.empty(n, dtype=np.uint16)
    y_out = np.empty(n, dtype=np.uint16)
    p_out = np.empty(n, dtype=np.uint8)
    _fill_evt3(words, t_out, x_out, y_out, p_out, n)
    return t_out, x_out, y_out, p_out


@njit(cache=True)
def _count_evt3(words, limit):
    n = 0
    have_y = False
    n_words = words.shape[0]
    for i in range(n_words):
        if limit >= 0 and n >= limit:
            break
        word = int(words[i])
        ev_type = word >> 12
        data = word & 0x0FFF
        if ev_type == 0x0:
            have_y = True
        elif ev_type == 0x2:
            if have_y:
                n += 1
        elif ev_type == 0x4:
            if have_y:
                valid = data & 0x0FFF
                for k in range(12):
                    if (valid >> k) & 1:
                        n += 1
                        if limit >= 0 and n >= limit:
                            break
        elif ev_type == 0x5:
            if have_y:
                valid = data & 0x00FF
                for k in range(8):
                    if (valid >> k) & 1:
                        n += 1
                        if limit >= 0 and n >= limit:
                            break
    if limit >= 0 and n > limit:
        n = limit
    return n


@njit(cache=True)
def _fill_evt3(words, t_out, x_out, y_out, p_out, limit):
    n = 0
    current_y = 0
    base_x = 0
    pol = 0
    time_low = 0
    time_high = 0
    time_high_loop = 0
    last_time_high = 0
    have_y = False
    n_words = words.shape[0]

    for i in range(n_words):
        if n >= limit:
            break
        word = int(words[i])
        ev_type = word >> 12
        data = word & 0x0FFF

        if ev_type == 0x0:
            current_y = data & 0x07FF
            have_y = True
        elif ev_type == 0x2:
            if have_y:
                pol = (data >> 11) & 0x1
                x = data & 0x07FF
                ts = (time_high_loop << 24) | (time_high << 12) | time_low
                t_out[n] = ts
                x_out[n] = x
                y_out[n] = current_y
                p_out[n] = pol
                n += 1
        elif ev_type == 0x3:
            pol = (data >> 11) & 0x1
            base_x = data & 0x07FF
        elif ev_type == 0x4:
            if have_y:
                valid = data & 0x0FFF
                ts = (time_high_loop << 24) | (time_high << 12) | time_low
                for k in range(12):
                    if n >= limit:
                        break
                    if (valid >> k) & 1:
                        t_out[n] = ts
                        x_out[n] = base_x + k
                        y_out[n] = current_y
                        p_out[n] = pol
                        n += 1
                base_x += 12
        elif ev_type == 0x5:
            if have_y:
                valid = data & 0x00FF
                ts = (time_high_loop << 24) | (time_high << 12) | time_low
                for k in range(8):
                    if n >= limit:
                        break
                    if (valid >> k) & 1:
                        t_out[n] = ts
                        x_out[n] = base_x + k
                        y_out[n] = current_y
                        p_out[n] = pol
                        n += 1
                base_x += 8
        elif ev_type == 0x6:
            time_low = data & 0x0FFF
        elif ev_type == 0x8:
            th = data & 0x0FFF
            if th < last_time_high:
                time_high_loop += 1
            time_high = th
            last_time_high = th

    return n
