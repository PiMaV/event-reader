"""Unit tests for EVT3 header + synthetic decode + binning."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from evt_sidecar.binning import (
    BinParams,
    PolarityMode,
    Representation,
    bin_events,
    occupancy_view,
    spatial_out_size,
    stack_for_polarity,
    stack_for_send,
)
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
    planes = bin_events(store, BinParams(dt_us=1000, max_frames=10))
    assert planes.dtype == np.uint16
    assert planes.shape == (2, 4, 4, 2)
    stack = stack_for_polarity(planes, PolarityMode.SIGNED)
    assert stack.dtype == np.int16
    assert stack.shape[0] == 2
    # frame0 at (1,1): +1 -1 = 0; (2,1): +1
    assert int(stack[0, 1, 1]) == 0
    assert int(stack[0, 1, 2]) == 1
    assert int(stack[1, 1, 1]) == 1


def test_bin_color_rgb_from_polarity() -> None:
    from evt_sidecar.raw_header import RawHeader

    store = EventStore(
        t=np.array([0, 10], dtype=np.uint64),
        x=np.array([1, 2], dtype=np.uint16),
        y=np.array([1, 1], dtype=np.uint16),
        p=np.array([0, 1], dtype=np.uint8),
        width=4,
        height=4,
        header=RawHeader(4, 4, "EVT3", "3.0", {}, 0),
    )
    planes = bin_events(store, BinParams(dt_us=1000, max_frames=1))
    rgb = stack_for_polarity(planes, PolarityMode.COLOR)
    assert rgb.dtype == np.uint16
    assert rgb.shape == (1, 4, 4, 3)
    assert int(rgb[0, 1, 1, 0]) == 1  # OFF → red
    assert int(rgb[0, 1, 1, 1]) == 0
    assert int(rgb[0, 1, 2, 1]) == 1  # ON → green
    assert int(rgb[0, 1, 2, 0]) == 0


def test_mixed_pixel_counts_and_occupancy_yellow() -> None:
    from evt_sidecar.raw_header import RawHeader

    store = EventStore(
        t=np.array([0, 10, 20], dtype=np.uint64),
        x=np.array([1, 1, 1], dtype=np.uint16),
        y=np.array([1, 1, 1], dtype=np.uint16),
        p=np.array([1, 1, 0], dtype=np.uint8),  # two ON, one OFF
        width=4,
        height=4,
        header=RawHeader(4, 4, "EVT3", "3.0", {}, 0),
    )
    planes = bin_events(store, BinParams(dt_us=1000, max_frames=1))
    assert int(planes[0, 1, 1, 0]) == 1
    assert int(planes[0, 1, 1, 1]) == 2
    gray = stack_for_send(planes, PolarityMode.COLOR, Representation.COUNTS)
    assert gray.ndim == 3
    assert gray.shape == (1, 4, 4)
    assert gray.dtype == np.uint16
    assert int(gray[0, 1, 1]) == 3  # activity: 2 ON + 1 OFF
    occ_send = stack_for_send(planes, PolarityMode.COLOR, Representation.OCCUPANCY)
    assert occ_send.ndim == 3
    assert occ_send.dtype == np.uint8
    assert int(occ_send[0, 1, 1]) == 255
    states = stack_for_send(planes, PolarityMode.COLOR, Representation.STATES)
    assert states.dtype == np.uint8
    assert states.ndim == 3
    assert states.shape == (1, 4, 4)
    assert int(states[0, 1, 1]) == 255  # both → top rung
    assert int(states[0, 0, 0]) == 0
    occ = occupancy_view(planes, PolarityMode.COLOR)
    assert occ.dtype == np.uint8
    assert list(occ[0, 1, 1]) == [255, 255, 0]  # yellow = both polarities
    from evt_sidecar.binning import preview_from_on_off

    occ_preview = preview_from_on_off(
        planes, PolarityMode.COLOR, Representation.OCCUPANCY
    )
    assert occ_preview.shape[-1] == 3
    assert occ_preview[0, 1, 1, 0] == pytest.approx(1.0)
    assert occ_preview[0, 1, 1, 1] == pytest.approx(1.0)
    assert occ_preview[0, 1, 1, 2] == pytest.approx(1.0)
    states_preview = preview_from_on_off(
        planes, PolarityMode.COLOR, Representation.STATES
    )
    assert states_preview.shape[-1] == 3
    assert states_preview[0, 1, 1, 0] == pytest.approx(1.0)
    assert states_preview[0, 1, 1, 1] == pytest.approx(1.0)
    assert states_preview[0, 1, 1, 2] == pytest.approx(0.0)

    from evt_sidecar.binning import INFERNO_LUT

    counts_rgb = preview_from_on_off(
        planes, PolarityMode.COLOR, Representation.COUNTS, hi=3
    )
    # Activity 3 at p99=3 → top Inferno (not red/green/yellow)
    assert counts_rgb.shape[-1] == 3
    assert counts_rgb[0, 1, 1] == pytest.approx(INFERNO_LUT[-1], abs=1e-5)
    assert counts_rgb[0, 0, 0] == pytest.approx(INFERNO_LUT[0], abs=1e-5)
    assert float(INFERNO_LUT[0].sum()) == pytest.approx(0.0)
    assert counts_rgb[0, 1, 1, 2] > 0.5  # top Inferno is yellow-white, not RGB yellow
    signed = stack_for_polarity(planes, PolarityMode.SIGNED)
    assert signed.dtype == np.int16
    assert int(signed[0, 1, 1]) == 1  # 2 ON − 1 OFF


def test_states_view_even_rungs() -> None:
    from evt_sidecar.binning import STATE_BOTH, STATE_OFF, STATE_ON, states_view

    planes = np.zeros((1, 2, 2, 2), dtype=np.uint16)
    planes[0, 0, 0, 0] = 1  # OFF only
    planes[0, 0, 1, 1] = 4  # ON only
    planes[0, 1, 0, 0] = 1
    planes[0, 1, 0, 1] = 1  # both
    codes = states_view(planes)
    assert codes.dtype == np.uint8
    assert codes.shape == (1, 2, 2)
    assert int(codes[0, 0, 0]) == int(STATE_OFF)
    assert int(codes[0, 0, 1]) == int(STATE_ON)
    assert int(codes[0, 1, 0]) == int(STATE_BOTH)
    assert int(codes[0, 1, 1]) == 0


def test_npy_save_roundtrip_matches_send_cube(tmp_path: Path) -> None:
    planes = np.zeros((2, 3, 3, 2), dtype=np.uint16)
    planes[0, 1, 1, 1] = 4
    planes[1, 0, 0, 0] = 2
    cube = stack_for_send(planes, PolarityMode.COLOR, Representation.STATES)
    out = tmp_path / "clip_states.npy"
    np.save(out, cube)
    loaded = np.load(out)
    assert loaded.dtype == cube.dtype
    assert loaded.shape == cube.shape
    np.testing.assert_array_equal(loaded, cube)


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

    u16 = np.zeros((1, 4, 4, 3), dtype=np.uint16)
    u16[0, 1, 1, 1] = 7
    kept = encode_stack_for_send(u16)
    assert kept.dtype == np.uint16
    assert int(kept[0, 1, 1, 1]) == 7


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

    # Short window: ~150 pictures may be finer than 1 ms — that is allowed
    dt, n, _u = plan_pictures(40_000, n_frames=150)
    assert dt < 1000
    assert n >= 150


def test_nice_dt_us_is_1_2_5() -> None:
    from evt_sidecar.binning import nice_dt_us, plan_nice_pictures

    assert nice_dt_us(1) == 1
    assert nice_dt_us(603) == 500
    assert nice_dt_us(1000) == 1000
    assert nice_dt_us(10_000) == 10_000
    assert nice_dt_us(66_666) == 50_000
    assert nice_dt_us(800) == 1000

    dt, n, _u = plan_nice_pictures(90_450, n_frames=150)
    assert dt == 500
    assert n == 181


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
    # Hot pixel saturates the top rung; 1-event pixel stays a distinct step
    assert hi == pytest.approx(1.0)
    assert disp[0, 2, 2] > 0
    assert disp[0, 2, 2] < disp[0, 3, 3]

    rgb = np.zeros((1, 4, 4, 3), dtype=np.float32)
    rgb[0, 1, 1, 0] = 2.0
    rgb[0, 1, 2, 1] = 4.0
    disp_c, lo_c, hi_c, n_c = activity_preview(rgb)
    assert disp_c.shape == (1, 4, 4, 3)
    assert n_c == 2
    assert disp_c[0, 1, 1, 0] > disp_c[0, 1, 1, 1]
    assert 0.0 <= lo_c < hi_c <= 1.0


def test_color_preview_scales() -> None:
    from evt_sidecar.binning import (
        apply_event_scale,
        color_preview,
        event_count_ceiling,
    )

    empty = np.zeros((2, 3, 3, 3), dtype=np.float32)
    assert color_preview(empty).shape == empty.shape
    stacked = np.zeros((1, 2, 2, 3), dtype=np.float32)
    stacked[0, 0, 0, 1] = 1.0
    stacked[0, 0, 1, 1] = 2.0
    stacked[0, 1, 1, 1] = 4.0
    hi = event_count_ceiling(stacked)
    assert hi == 4
    out = color_preview(stacked, hi=hi)
    assert out[0, 0, 0, 1] == pytest.approx(0.25)
    assert out[0, 0, 1, 1] == pytest.approx(0.5)
    assert out[0, 1, 1, 1] == pytest.approx(1.0)
    a = apply_event_scale(stacked, hi)
    b = apply_event_scale(np.zeros_like(stacked), hi)
    assert a.max() > b.max()


def test_removed_fraction() -> None:
    from evt_sidecar.binning import removed_fraction

    before = np.ones((2, 4, 4), dtype=np.float32)
    after = before.copy()
    after[0] = 0.0
    assert removed_fraction(before, after, frame=0) == pytest.approx(1.0)
    assert removed_fraction(before, after, frame=1) == pytest.approx(0.0)
    assert removed_fraction(before, after) == pytest.approx(0.5)


def test_roi_event_series() -> None:
    from evt_sidecar.binning import roi_event_series

    stack = np.zeros((3, 8, 8), dtype=np.float32)
    stack[0, 2:4, 2:4] = 4.0
    stack[1, 2:4, 2:4] = 8.0
    y = roi_event_series(stack, 2, 2, 4, 4)
    assert y.shape == (3,)
    assert y[0] == pytest.approx(16.0)
    assert y[1] == pytest.approx(32.0)
    assert y[2] == pytest.approx(0.0)
    rgb = np.zeros((2, 4, 4, 3), dtype=np.float32)
    rgb[0, 1, 1, 1] = 3.0
    y_rgb = roi_event_series(rgb, 1, 1, 2, 2)
    assert y_rgb[0] == pytest.approx(3.0)


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
    assert cropped.shape == (1, 4, 4, 2)
    assert cropped.dtype == np.uint16
    activity = stack_for_polarity(cropped, PolarityMode.BOTH)
    assert activity.shape == (1, 4, 4)
    assert int(activity[0, 1, 1]) == 1  # (5,5) → (1,1)
    assert int(activity[0, 1, 2]) == 1  # (6,5) → (1,2)
    assert int(activity[0].sum()) == 2

    balanced = np.ones((2, 4, 4), dtype=np.float32)
    assert even_odd_row_ratio(balanced) == pytest.approx(1.0)
    striped = np.zeros((1, 4, 4), dtype=np.float32)
    striped[:, 0::2, :] = 10.0
    assert even_odd_row_ratio(striped) > 3.0


def test_spatial_bin_pools_counts() -> None:
    from evt_sidecar.raw_header import RawHeader

    # Four ON events in the top-left 2×2, plus one at (3,3).
    store = EventStore(
        t=np.array([0, 1, 2, 3, 4], dtype=np.uint64),
        x=np.array([0, 1, 0, 1, 3], dtype=np.uint16),
        y=np.array([0, 0, 1, 1, 3], dtype=np.uint16),
        p=np.ones(5, dtype=np.uint8),
        width=4,
        height=4,
        header=RawHeader(4, 4, "EVT3", "3.0", {}, 0),
    )
    assert spatial_out_size(4, 4, 2) == (2, 2)
    assert spatial_out_size(5, 4, 2) == (2, 2)
    assert BinParams(spatial_bin=0).clamp_spatial_bin() == 1

    stacked = bin_events(store, BinParams(dt_us=1000, spatial_bin=2, max_frames=1))
    assert stacked.shape == (1, 2, 2, 2)
    activity = stack_for_polarity(stacked, PolarityMode.BOTH)
    assert int(activity[0, 0, 0]) == 4
    assert int(activity[0, 1, 1]) == 1
    assert int(activity[0].sum()) == 5


def test_spatial_bin_after_crop() -> None:
    from evt_sidecar.raw_header import RawHeader

    store = EventStore(
        t=np.array([0, 1, 2], dtype=np.uint64),
        x=np.array([4, 5, 0], dtype=np.uint16),
        y=np.array([4, 5, 0], dtype=np.uint16),
        p=np.ones(3, dtype=np.uint8),
        width=8,
        height=8,
        header=RawHeader(8, 8, "EVT3", "3.0", {}, 0),
    )
    stacked = bin_events(
        store,
        BinParams(dt_us=1000, x0=4, y0=4, x1=8, y1=8, spatial_bin=2, max_frames=1),
    )
    assert stacked.shape == (1, 2, 2, 2)
    activity = stack_for_polarity(stacked, PolarityMode.BOTH)
    assert int(activity[0, 0, 0]) == 2  # (4,4) and (5,5) in the same 2×2
    assert int(activity[0].sum()) == 2


def test_aligned_crop_then_spatial_matches_slice() -> None:
    from evt_sidecar.raw_header import RawHeader

    rng = np.random.default_rng(0)
    n = 40
    store = EventStore(
        t=np.arange(n, dtype=np.uint64),
        x=rng.integers(0, 8, size=n, dtype=np.uint16),
        y=rng.integers(0, 8, size=n, dtype=np.uint16),
        p=np.ones(n, dtype=np.uint8),
        width=8,
        height=8,
        header=RawHeader(8, 8, "EVT3", "3.0", {}, 0),
    )
    full = bin_events(store, BinParams(dt_us=1000, spatial_bin=2, max_frames=1))
    cropped = bin_events(
        store,
        BinParams(
            dt_us=1000, x0=2, y0=2, x1=6, y1=6, spatial_bin=2, max_frames=1
        ),
    )
    np.testing.assert_array_equal(cropped, full[:, 1:3, 1:3, :])


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
    raw = stack_for_polarity(
        bin_events(store, BinParams(dt_us=20_000, max_frames=1)),
        PolarityMode.BOTH,
    )
    assert int(raw[0, 2, 2]) == 1
    assert int(raw[0, 6, 6]) == 1

    nn = stack_for_polarity(
        bin_events(
            store,
            BinParams(dt_us=20_000, max_frames=1, neighbor_dt_us=1000),
        ),
        PolarityMode.BOTH,
    )
    assert int(nn[0, 2, 2]) == 0  # first of pair dropped
    assert int(nn[0, 2, 3]) == 1
    assert int(nn[0, 6, 6]) == 0

    despike = stack_for_polarity(
        bin_events(
            store,
            BinParams(dt_us=20_000, max_frames=1, drop_isolated=True),
        ),
        PolarityMode.BOTH,
    )
    assert int(despike[0, 6, 6]) == 0
    assert int(despike[0, 2, 2]) == 1
    assert int(despike[0, 2, 3]) == 1


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
    planes = bin_events(store, BinParams(dt_us=1000))
    assert planes.shape == (1, 8, 8, 2)
    assert int(planes[0, 3, 4].sum()) == 2
    activity = stack_for_polarity(planes, PolarityMode.BOTH)
    assert activity.shape == (1, 8, 8)
    assert int(activity[0, 3, 4]) == 2


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


def test_preview_contain_ranges() -> None:
    from evt_sidecar.gui import PREVIEW_VOID, preview_contain_ranges

    assert PREVIEW_VOID != (0, 0, 0)

    # Wide short pane: full height, extra void left/right — whole 1280×720 visible.
    xr, yr = preview_contain_ranges(1280, 720, 800, 280)
    assert yr == (0.0, 720.0)
    assert xr[0] < 0.0
    assert xr[1] > 1280.0
    assert abs((xr[1] - xr[0]) / (yr[1] - yr[0]) - 800 / 280) < 1e-6

    # Tall square pane: full width, extra void top/bottom — not a width crop.
    xr, yr = preview_contain_ranges(1280, 720, 400, 400)
    assert xr == (0.0, 1280.0)
    assert yr[0] < 0.0
    assert yr[1] > 720.0
    assert abs(yr[0] + yr[1] - 720.0) < 1e-6
    assert abs((xr[1] - xr[0]) / (yr[1] - yr[0]) - 1.0) < 1e-6

    xr, yr = preview_contain_ranges(1280, 720, 800, 280, margin=0.12)
    assert yr[0] < 0.0
    assert yr[1] > 720.0
    assert abs((yr[1] - yr[0]) - 720.0 * 1.24) < 1e-6
    assert abs((xr[1] - xr[0]) / (yr[1] - yr[0]) - 800 / 280) < 1e-6
