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
    assert h.source_lines[0].startswith("% ")
    assert h.source_lines[-1] == "% end"
    assert any("format EVT3" in line for line in h.source_lines)


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

    counts = np.zeros((1, 4, 4), dtype=np.float32)
    counts[0, 1, 1] = 1.0
    counts[0, 2, 2] = 3.0
    counts[0, 3, 3] = 8000.0
    raw = stack_for_network(counts)
    assert raw.dtype == np.uint8
    assert int(raw[0, 1, 1]) == 1
    assert int(raw[0, 2, 2]) == 3
    assert int(raw[0, 3, 3]) == 255

    logged = stack_for_network(counts, log_stretch=True)
    assert int(logged[0, 1, 1]) > 0
    assert int(logged[0, 2, 2]) > int(logged[0, 1, 1])
    assert int(logged[0, 3, 3]) == 255

    signed = np.array([[[-2.0, 0.0, 2.0]]], dtype=np.float32)
    s = stack_for_network(signed)
    assert s.dtype == np.uint8
    assert int(s[0, 0, 1]) == 128
    assert int(s[0, 0, 0]) == 126
    assert int(s[0, 0, 2]) == 130


def test_encode_stack_sends_float_counts_by_default() -> None:
    from evt_sidecar.binning import encode_stack_for_send

    counts = np.zeros((1, 4, 4), dtype=np.float32)
    counts[0, 1, 1] = 1.0
    counts[0, 2, 2] = 6.0
    out = encode_stack_for_send(counts)
    assert out.dtype == np.float32
    assert out[0, 1, 1] == pytest.approx(1.0)
    assert out[0, 2, 2] == pytest.approx(6.0)

    packed = encode_stack_for_send(counts, eight_bit=True)
    assert packed.dtype == np.uint8
    assert int(packed[0, 2, 2]) == 6


def test_plan_pictures() -> None:
    from evt_sidecar.binning import plan_pictures

    dt, n, used = plan_pictures(1_500_000, dt_us=1000)
    assert dt == 1000
    assert n == 1500
    assert used == 1_500_000

    dt, n, used = plan_pictures(1_500_000, n_frames=150)
    assert n == 150
    assert dt == 10_000

    dt, n, used = plan_pictures(10_000_000, dt_us=1000)
    assert n == 10_000
    assert used == 10_000_000


def test_plan_overview_floors_dt_at_1ms() -> None:
    from evt_sidecar.binning import OVERVIEW_MIN_DT_US, plan_overview, plan_pictures

    dt, n, _u = plan_pictures(40_000, n_frames=150)
    assert dt < OVERVIEW_MIN_DT_US
    dt_ov, n_ov, _u = plan_overview(40_000)
    assert dt_ov == OVERVIEW_MIN_DT_US
    assert n_ov == 40


def test_activity_preview_levels() -> None:
    from evt_sidecar.binning import activity_preview

    empty = np.zeros((4, 4), dtype=np.float32)
    disp, lo, hi, n_pos = activity_preview(empty)
    assert disp.shape == (1, 4, 4)
    assert n_pos == 0
    assert hi > lo

    img = np.zeros((8, 8), dtype=np.float32)
    img[2, 2] = 1.0
    img[3, 3] = 10_000.0
    disp, lo, hi, n_pos = activity_preview(img)
    assert n_pos == 2
    assert disp[0, 2, 2] > 0
    # Hot pixel must not set the LUT ceiling to log1p(10000) alone
    assert hi <= float(np.log1p(10_000.0))
    assert hi >= disp[0, 2, 2]


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

    x_win, c_win = event_rate_ms(store, n_bins=10, t0_us=0, t1_us=300)
    assert c_win.sum() == 3
    assert x_win[-1] < 1.0


def test_bin_crop_and_even_odd_ratio() -> None:
    from evt_sidecar.binning import even_odd_row_ratio
    from evt_sidecar.raw_header import RawHeader

    store = EventStore(
        t=np.array([0, 10, 20], dtype=np.uint64),
        x=np.array([5, 6, 1], dtype=np.uint16),
        y=np.array([5, 5, 1], dtype=np.uint16),
        p=np.ones(3, dtype=np.uint8),
        width=8,
        height=8,
        header=RawHeader(8, 8, "EVT3", "3.0", {}, 0),
    )
    cropped = bin_events(
        store,
        BinParams(dt_us=1000, x0=4, y0=4, x1=8, y1=8, max_frames=1),
    )
    assert cropped.shape == (1, 4, 4)
    assert cropped[0, 1, 1] == pytest.approx(1.0)  # (5,5) → (1,1)
    assert cropped[0, 1, 2] == pytest.approx(1.0)  # (6,5) → (1,2)
    assert cropped[0].sum() == pytest.approx(2.0)

    balanced = np.ones((2, 4, 4), dtype=np.float32)
    assert even_odd_row_ratio(balanced) == pytest.approx(1.0)
    striped = np.zeros((1, 4, 4), dtype=np.float32)
    striped[:, 0::2, :] = 10.0
    assert even_odd_row_ratio(striped) > 3.0


def test_bin_neighbor_and_isolated() -> None:
    from evt_sidecar.raw_header import RawHeader

    store = EventStore(
        t=np.array([0, 200, 10_000], dtype=np.uint64),
        x=np.array([2, 3, 6], dtype=np.uint16),
        y=np.array([2, 2, 6], dtype=np.uint16),
        p=np.ones(3, dtype=np.uint8),
        width=8,
        height=8,
        header=RawHeader(8, 8, "EVT3", "3.0", {}, 0),
    )
    raw = bin_events(store, BinParams(dt_us=20_000, max_frames=1))
    assert raw[0, 2, 2] == pytest.approx(1.0)
    assert raw[0, 6, 6] == pytest.approx(1.0)

    nn = bin_events(
        store,
        BinParams(dt_us=20_000, max_frames=1, neighbor_dt_us=1000),
    )
    assert nn[0, 2, 2] == pytest.approx(0.0)  # first of pair dropped
    assert nn[0, 2, 3] == pytest.approx(1.0)
    assert nn[0, 6, 6] == pytest.approx(0.0)

    despike = bin_events(
        store,
        BinParams(dt_us=20_000, max_frames=1, drop_isolated=True),
    )
    assert despike[0, 6, 6] == pytest.approx(0.0)
    assert despike[0, 2, 2] == pytest.approx(1.0)
    assert despike[0, 2, 3] == pytest.approx(1.0)


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


def test_raw_paths_from_dropped(tmp_path: Path) -> None:
    from evt_sidecar.gui import raw_paths_from_dropped

    raw_a = tmp_path / "a.raw"
    raw_b = tmp_path / "sub" / "b.raw"
    other = tmp_path / "note.txt"
    nested_dir = tmp_path / "sub"
    nested_dir.mkdir()
    raw_a.write_bytes(b"x")
    raw_b.write_bytes(b"x")
    other.write_text("nope")

    assert raw_paths_from_dropped([raw_a]) == [raw_a]
    assert raw_paths_from_dropped([other]) == []
    assert raw_paths_from_dropped([tmp_path]) == [raw_a]
    assert raw_paths_from_dropped([nested_dir]) == [raw_b]
    assert raw_paths_from_dropped([raw_a, raw_a]) == [raw_a]
