"""Unit tests for EVT3 header + synthetic decode + binning."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from evt_sidecar.binning import BinParams, PolarityMode, bin_events
from evt_sidecar.evt3 import EventStore, decode_evt3_words, load_evt3_raw
from evt_sidecar.raw_header import parse_raw_header


def _word(ev_type: int, data: int) -> int:
    return ((ev_type & 0xF) << 12) | (data & 0x0FFF)


def test_parse_header(tmp_path: Path) -> None:
    raw = tmp_path / "t.raw"
    header = (
        b"% camera_integrator_name IDS\n"
        b"% evt 3.0\n"
        b"% format EVT3;height=720;width=1280\n"
        b"% geometry 1280x720\n"
        b"% end\n"
    )
    raw.write_bytes(header + b"\x00\x00")
    h = parse_raw_header(raw)
    assert h.width == 1280
    assert h.height == 720
    assert h.header_bytes == len(header)
    assert "EVT3" in h.format_name.upper()


def test_decode_single_event() -> None:
    # TIME_HIGH=0, TIME_LOW=100, Y=10, X=20 ON
    words = np.array(
        [
            _word(0x8, 0),
            _word(0x6, 100),
            _word(0x0, 10),
            _word(0x2, (1 << 11) | 20),
        ],
        dtype=np.uint16,
    )
    t, x, y, p = decode_evt3_words(words)
    assert len(t) == 1
    assert int(t[0]) == 100
    assert int(x[0]) == 20
    assert int(y[0]) == 10
    assert int(p[0]) == 1


def test_decode_vect12() -> None:
    words = np.array(
        [
            _word(0x8, 0),
            _word(0x6, 50),
            _word(0x0, 5),
            _word(0x3, (0 << 11) | 100),  # OFF base x=100
            _word(0x4, 0b0000_0000_0101),  # bits 0 and 2
        ],
        dtype=np.uint16,
    )
    t, x, y, p = decode_evt3_words(words)
    assert list(x) == [100, 102]
    assert list(y) == [5, 5]
    assert list(p) == [0, 0]
    assert list(t) == [50, 50]


def test_bin_signed() -> None:
    from evt_sidecar.raw_header import RawHeader

    store = EventStore(
        t=np.array([0, 10, 20, 1100], dtype=np.uint64),
        x=np.array([1, 1, 2, 1], dtype=np.uint16),
        y=np.array([1, 1, 1, 1], dtype=np.uint16),
        p=np.array([1, 0, 1, 1], dtype=np.uint8),
        width=4,
        height=4,
        header=RawHeader(4, 4, "EVT3", "3.0", {}, 0),
    )
    stack = bin_events(
        store,
        BinParams(dt_us=1000, polarity=PolarityMode.SIGNED, max_frames=10),
    )
    assert stack.shape[0] == 2
    # frame0 at (1,1): +1 -1 = 0; (2,1): +1
    assert stack[0, 1, 1] == pytest.approx(0.0)
    assert stack[0, 1, 2] == pytest.approx(1.0)
    assert stack[1, 1, 1] == pytest.approx(1.0)


def test_stack_for_network_uint8() -> None:
    from evt_sidecar.binning import stack_for_network

    # Sparse heavy-tailed counts: linear min-max would zero almost everything
    counts = np.zeros((1, 4, 4), dtype=np.float32)
    counts[0, 1, 1] = 1.0
    counts[0, 2, 2] = 3.0
    counts[0, 3, 3] = 8000.0
    u = stack_for_network(counts)
    assert u.dtype == np.uint8
    assert u.shape == counts.shape
    assert int(u[0, 1, 1]) > 0
    assert int(u[0, 2, 2]) > int(u[0, 1, 1])

    signed = np.array([[[-2.0, 0.0, 2.0]]], dtype=np.float32)
    s = stack_for_network(signed)
    assert s.dtype == np.uint8
    assert int(s[0, 0, 1]) == 128


def test_plan_pictures() -> None:
    from evt_sidecar.binning import HARD_FRAME_CAP, plan_pictures

    dt, n, capped, used = plan_pictures(1_500_000, dt_us=1000)
    assert dt == 1000
    assert n == 1500
    assert not capped
    assert used == 1_500_000

    dt, n, capped, used = plan_pictures(1_500_000, n_frames=150)
    assert n == 150
    assert dt == 10_000
    assert not capped

    dt, n, capped, _used = plan_pictures(10_000_000, dt_us=1000)
    assert capped
    assert n == HARD_FRAME_CAP


def test_min_dt_us() -> None:
    from evt_sidecar.binning import min_dt_us

    assert min_dt_us(1_500_000) == 750  # 1500 ms / 2000 pictures
    assert min_dt_us(100) == 1



def test_event_rate_ms() -> None:
    from evt_sidecar.binning import event_rate_ms
    from evt_sidecar.raw_header import RawHeader

    store = EventStore(
        t=np.array([0, 100, 200, 5000], dtype=np.uint64),
        x=np.zeros(4, dtype=np.uint16),
        y=np.zeros(4, dtype=np.uint16),
        p=np.ones(4, dtype=np.uint8),
        width=8,
        height=8,
        header=RawHeader(8, 8, "EVT3", "3.0", {}, 0),
    )
    x_ms, counts = event_rate_ms(store, n_bins=10)
    assert x_ms.shape == counts.shape
    assert counts.sum() == 4
    assert x_ms[-1] > 0


def test_write_synthetic_raw_roundtrip(tmp_path: Path) -> None:
    header = (
        b"% evt 3.0\n"
        b"% format EVT3;height=8;width=8\n"
        b"% geometry 8x8\n"
        b"% end\n"
    )
    words = [
        _word(0x8, 0),
        _word(0x6, 0),
        _word(0x0, 3),
        _word(0x2, (1 << 11) | 4),
        _word(0x6, 500),
        _word(0x2, (0 << 11) | 4),
    ]
    payload = b"".join(struct.pack("<H", w) for w in words)
    path = tmp_path / "syn.raw"
    path.write_bytes(header + payload)
    store = load_evt3_raw(path)
    assert len(store) == 2
    assert store.width == 8
    stack = bin_events(store, BinParams(dt_us=1000, polarity=PolarityMode.BOTH))
    assert stack.shape == (1, 8, 8)
    assert stack[0, 3, 4] == pytest.approx(2.0)
