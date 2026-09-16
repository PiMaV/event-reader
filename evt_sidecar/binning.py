"""Accumulate CD events into ON/OFF count planes, then view for Stream."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .evt3 import EventStore
from .filters import drop_isolated_pixels, neighbor_keep_mask

# Last axis of the dense bin: 0 = OFF counts, 1 = ON counts
OFF_CH = 0
ON_CH = 1

# States cube (one gray uint8): even rungs 0…255 so any BLITZ colormap works.
# Yellow = both at the top (fullest occupancy). Metavision frame gen is
# last-event-wins (no “both”); this fourth rung is occupancy in the Δt.
STATE_NONE = np.uint8(0)
STATE_OFF = np.uint8(85)  # 255 // 3
STATE_ON = np.uint8(170)  # 2 * (255 // 3)
STATE_BOTH = np.uint8(255)
STATE_RUNGS = (int(STATE_NONE), int(STATE_OFF), int(STATE_ON), int(STATE_BOTH))


class PolarityMode(str, Enum):
    ON = "on"
    OFF = "off"
    BOTH = "both"
    SIGNED = "signed"  # ON=+1, OFF=-1
    COLOR = "color"  # OFF=red, ON=green, both=yellow


class Representation(str, Enum):
    STATES = "states"  # uint8 gray: none / OFF / ON / both (0, 85, 170, 255)
    COUNTS = "counts"  # events/pixel/Δt
    OCCUPANCY = "occupancy"  # binary who-fired (0 or 255; signed −1/0/+1)


class AccumMode(str, Enum):
    COUNT = "count"


@dataclass
class BinParams:
    dt_us: int = 1000
    polarity: PolarityMode = PolarityMode.BOTH  # unused by bin; views apply it
    accum: AccumMode = AccumMode.COUNT
    t0_us: int | None = None  # absolute; None = store.t_min
    t1_us: int | None = None  # absolute; None = store.t_max
    max_frames: int | None = None  # None = no hard picture cap
    # Exclusive-end crop in sensor pixels; None = that edge is full frame
    x0: int | None = None
    y0: int | None = None
    x1: int | None = None
    y1: int | None = None
    drop_isolated: bool = False
    neighbor_dt_us: int | None = None  # None = temporal neighbour filter off
    neighbor_radius: int = 1
    spatial_bin: int = 1  # 1 = full sensor; 2/4/8 pool that many pixels

    def clamp_dt(self) -> int:
        return max(1, int(self.dt_us))

    def clamp_spatial_bin(self) -> int:
        return max(1, int(self.spatial_bin or 1))


def crop_box(store: EventStore, params: BinParams) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) exclusive-end, clipped to the sensor."""
    w, h = int(store.width), int(store.height)
    x0 = 0 if params.x0 is None else int(params.x0)
    y0 = 0 if params.y0 is None else int(params.y0)
    x1 = w if params.x1 is None else int(params.x1)
    y1 = h if params.y1 is None else int(params.y1)
    x0 = max(0, min(w, x0))
    y0 = max(0, min(h, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, w, h
    return x0, y0, x1, y1


def spatial_out_size(height: int, width: int, spatial_bin: int) -> tuple[int, int]:
    """Height/width after pooling ``spatial_bin``×``spatial_bin`` sensor pixels."""
    sb = max(1, int(spatial_bin))
    return max(1, int(height) // sb), max(1, int(width) // sb)


def bin_events(store: EventStore, params: BinParams) -> np.ndarray:
    """
    Bin CD events into uint16 ON/OFF counts ``(T, height, width, 2)``.

    Channel 0 = OFF, channel 1 = ON. Polarity and representation are views
    (``stack_for_polarity`` / ``stack_for_send``), not a second bin.
    ``states`` is one gray uint8 channel (0 / 85 / 170 / 255);
    ``counts`` / ``occupancy`` stay one gray channel. RGB is local preview.
    """
    x0, y0, x1, y1 = crop_box(store, params)
    sb = params.clamp_spatial_bin()
    src_h, src_w = y1 - y0, x1 - x0
    out_h, out_w = spatial_out_size(src_h, src_w, sb)

    def empty(n: int = 1) -> np.ndarray:
        return np.zeros((n, out_h, out_w, 2), dtype=np.uint16)

    if len(store) == 0:
        return empty()

    dt = params.clamp_dt()
    t0 = store.t_min if params.t0_us is None else int(params.t0_us)
    t1 = store.t_max if params.t1_us is None else int(params.t1_us)
    if t1 < t0:
        t0, t1 = t1, t0

    t = store.t
    lo = int(np.searchsorted(t, t0, side="left"))
    hi = int(np.searchsorted(t, t1, side="right"))
    if hi <= lo:
        return empty()

    t_s = t[lo:hi]
    x_s = store.x[lo:hi]
    y_s = store.y[lo:hi]
    p_s = store.p[lo:hi]

    if t_s.shape[0] == 0:
        return empty()

    if params.neighbor_dt_us is not None:
        keep_nn = neighbor_keep_mask(
            t_s,
            x_s,
            y_s,
            height=store.height,
            width=store.width,
            dt_us=int(params.neighbor_dt_us),
            radius=int(params.neighbor_radius),
        )
        t_s, x_s, y_s, p_s = t_s[keep_nn], x_s[keep_nn], y_s[keep_nn], p_s[keep_nn]
        if t_s.shape[0] == 0:
            return empty()

    frame_idx = ((t_s.astype(np.int64) - t0) // dt).astype(np.int64)
    n_frames = int(frame_idx.max()) + 1
    if params.max_frames is not None and n_frames > params.max_frames:
        keep = frame_idx < params.max_frames
        frame_idx = frame_idx[keep]
        x_s = x_s[keep]
        y_s = y_s[keep]
        p_s = p_s[keep]
        n_frames = params.max_frames

    inb = (x_s >= x0) & (x_s < x1) & (y_s >= y0) & (y_s < y1)
    frame_idx = frame_idx[inb]
    x_s = x_s[inb]
    y_s = y_s[inb]
    p_s = p_s[inb]
    if frame_idx.shape[0] == 0:
        return empty(n_frames)

    xs = (x_s.astype(np.int64) - x0) // sb
    ys = (y_s.astype(np.int64) - y0) // sb
    in_grid = (xs >= 0) & (xs < out_w) & (ys >= 0) & (ys < out_h)
    frame_idx = frame_idx[in_grid]
    xs = xs[in_grid]
    ys = ys[in_grid]
    p_s = p_s[in_grid]
    if frame_idx.shape[0] == 0:
        return empty(n_frames)

    stack32 = np.zeros((n_frames, out_h, out_w, 2), dtype=np.uint32)
    hw = out_h * out_w
    ch = np.where(p_s == 1, ON_CH, OFF_CH).astype(np.int64)
    flat = (frame_idx * hw + ys * out_w + xs) * 2 + ch
    np.add.at(stack32.ravel(), flat, 1)
    stack = np.clip(stack32, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if params.drop_isolated:
        stack = drop_isolated_pixels(stack)
    return stack


def _as_on_off(on_off: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(T, H, W, 2)`` and whether a leading T was added."""
    arr = np.asarray(on_off)
    squeeze = False
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[None, ...]
        squeeze = True
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(f"expected (T, H, W, 2) ON/OFF counts, got {arr.shape}")
    return arr, squeeze


def send_polarity(polarity: PolarityMode | str) -> PolarityMode:
    """Gray-cube polarity: Color becomes activity / any-fire (BOTH)."""
    mode = PolarityMode(polarity)
    if mode == PolarityMode.COLOR:
        return PolarityMode.BOTH
    return mode


def stack_for_polarity(
    on_off: np.ndarray, polarity: PolarityMode | str = PolarityMode.COLOR
) -> np.ndarray:
    """Counts view. Color is RGB uint16 for the local preview only."""
    arr, squeeze = _as_on_off(on_off)
    off = arr[..., OFF_CH]
    on = arr[..., ON_CH]
    mode = PolarityMode(polarity)
    if mode == PolarityMode.COLOR:
        out = np.zeros((*arr.shape[:3], 3), dtype=np.uint16)
        out[..., 0] = off
        out[..., 1] = on
    elif mode == PolarityMode.ON:
        out = np.asarray(on, dtype=np.uint16)
    elif mode == PolarityMode.OFF:
        out = np.asarray(off, dtype=np.uint16)
    elif mode == PolarityMode.SIGNED:
        net = on.astype(np.int32) - off.astype(np.int32)
        out = np.clip(net, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(
            np.int16
        )
    else:
        summed = off.astype(np.uint32) + on.astype(np.uint32)
        out = np.clip(summed, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return out[0] if squeeze else out


def occupancy_view(
    on_off: np.ndarray, polarity: PolarityMode | str = PolarityMode.COLOR
) -> np.ndarray:
    """Who fired in the Δt. Color is RGB 0/255 (yellow = both)."""
    arr, squeeze = _as_on_off(on_off)
    off = arr[..., OFF_CH] > 0
    on = arr[..., ON_CH] > 0
    mode = PolarityMode(polarity)
    lit = np.uint8(255)
    dark = np.uint8(0)
    if mode == PolarityMode.COLOR:
        out = np.zeros((*arr.shape[:3], 3), dtype=np.uint8)
        out[..., 0] = np.where(off, lit, dark)
        out[..., 1] = np.where(on, lit, dark)
    elif mode == PolarityMode.ON:
        out = np.where(on, lit, dark)
    elif mode == PolarityMode.OFF:
        out = np.where(off, lit, dark)
    elif mode == PolarityMode.SIGNED:
        net = arr[..., ON_CH].astype(np.int32) - arr[..., OFF_CH].astype(np.int32)
        out = np.sign(net).astype(np.int8)
    else:
        out = np.where(off | on, lit, dark)
    return out[0] if squeeze else out


def states_view(on_off: np.ndarray) -> np.ndarray:
    """Four occupancy rungs as one uint8 channel: none / OFF / ON / both.

    Values are 0, 85, 170, 255 (even steps). Both is the top rung so a
    sequential colormap reads nothing < OFF < ON < both. Local preview
    stays RGB; this is the BLITZ cube.
    """
    arr, squeeze = _as_on_off(on_off)
    off = arr[..., OFF_CH] > 0
    on = arr[..., ON_CH] > 0
    out = np.zeros(arr.shape[:3], dtype=np.uint8)
    out[off & ~on] = STATE_OFF
    out[on & ~off] = STATE_ON
    out[off & on] = STATE_BOTH
    return out[0] if squeeze else out


def stack_for_send(
    on_off: np.ndarray,
    polarity: PolarityMode | str = PolarityMode.COLOR,
    representation: Representation | str = Representation.STATES,
) -> np.ndarray:
    """Cube for BLITZ / DONNER — always one channel, never RGB.

    States (default): uint8 0 / 85 / 170 / 255 (none / OFF / ON / both).
    Counts: uint16 (ON+OFF, ON, OFF) or int16 signed; color polarity → activity.
    Occupancy: uint8 0/255 binary, or int8 signed; color polarity → any-fire.
    """
    rep = Representation(representation)
    if rep == Representation.STATES:
        return states_view(on_off)
    pol = send_polarity(polarity)
    if rep == Representation.OCCUPANCY:
        return occupancy_view(on_off, pol)
    return stack_for_polarity(on_off, pol)


def polarity_count_ceiling(
    on_off: np.ndarray, polarity: PolarityMode | str = PolarityMode.COLOR
) -> int:
    """p99 count rung for the local preview (unfiltered stack)."""
    pol = PolarityMode(polarity)
    arr, _squeeze = _as_on_off(on_off)
    if pol == PolarityMode.COLOR:
        energy = arr[..., OFF_CH].astype(np.uint32) + arr[..., ON_CH].astype(np.uint32)
        return event_count_ceiling(energy)
    viewed = stack_for_polarity(on_off, pol)
    if pol == PolarityMode.SIGNED:
        return event_count_ceiling(np.abs(viewed))
    return event_count_ceiling(viewed)


# Matplotlib Inferno stops (sRGB 0…1). Local counts preview; the wire is gray.
_INFERNO_STOPS = np.array(
    [
        [0.001462, 0.000466, 0.013866],
        [0.107138, 0.046545, 0.267002],
        [0.258234, 0.038571, 0.406485],
        [0.416331, 0.090203, 0.432994],
        [0.578304, 0.148039, 0.404411],
        [0.735683, 0.215906, 0.330245],
        [0.865006, 0.316822, 0.226055],
        [0.954480, 0.477102, 0.098161],
        [0.987622, 0.683002, 0.072076],
        [0.988362, 0.998364, 0.644924],
    ],
    dtype=np.float64,
)


def inferno_lut(n: int = 256) -> np.ndarray:
    """Inferno RGB ``(n, 3)`` float32 in 0…1."""
    x = np.linspace(0.0, 1.0, len(_INFERNO_STOPS), dtype=np.float64)
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
    out = np.empty((int(n), 3), dtype=np.float32)
    for c in range(3):
        out[:, c] = np.interp(t, x, _INFERNO_STOPS[:, c]).astype(np.float32)
    out[0] = 0.0
    return out


INFERNO_LUT = inferno_lut()


def apply_inferno(level: np.ndarray) -> np.ndarray:
    """Map 0…1 gray ``(T, H, W)`` (or ``H, W``) to Inferno RGB float32."""
    arr = np.clip(np.asarray(level, dtype=np.float32), 0.0, 1.0)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[None, ...]
    n = INFERNO_LUT.shape[0]
    idx = np.clip(np.rint(arr * (n - 1)), 0, n - 1).astype(np.intp)
    out = INFERNO_LUT[idx]
    return out[0] if squeeze else out


def gray_as_rgb(level: np.ndarray) -> np.ndarray:
    """Repeat 0…1 gray as RGB (occupancy preview)."""
    arr = np.clip(np.asarray(level, dtype=np.float32), 0.0, 1.0)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[None, ...]
    out = np.repeat(arr[..., None], 3, axis=-1)
    return out[0] if squeeze else out


def preview_from_on_off(
    on_off: np.ndarray,
    polarity: PolarityMode | str,
    representation: Representation | str,
    hi: float | None = None,
) -> np.ndarray:
    """Local display matching Send as: states RGB, occupancy gray, counts Inferno."""
    rep = Representation(representation)
    pol = PolarityMode(polarity)
    if rep == Representation.STATES:
        viewed = occupancy_view(on_off, PolarityMode.COLOR)
        return viewed.astype(np.float32) / 255.0
    if rep == Representation.OCCUPANCY:
        viewed = occupancy_view(on_off, send_polarity(pol))
        if viewed.dtype == np.uint8:
            return gray_as_rgb(viewed.astype(np.float32) / 255.0)
        signed = (viewed.astype(np.float32) + 1.0) * 0.5
        return gray_as_rgb(signed)
    viewed = stack_for_polarity(on_off, send_polarity(pol))
    if send_polarity(pol) == PolarityMode.SIGNED:
        return viewed
    cap = event_count_ceiling(viewed) if hi is None else hi
    return apply_inferno(apply_event_scale(viewed, cap))


SENSOR_DT_US = 1  # EVT3 timestamp tick
OVERVIEW_PICTURES = 150
INTERLACE_DT_US = 1000  # even/odd hint is extra-loud below 1 ms
INTERLACE_RATIO_WARN = 3.0


def plan_pictures(
    window_us: int,
    *,
    dt_us: int | None = None,
    n_frames: int | None = None,
) -> tuple[int, int, int]:
    """
    Map a time window to (dt_us, n_frames, used_window_us).

    Drive with either frame time (dt_us) or picture count (n_frames).
    No hard picture cap — RAM warnings live in ``ram.assess_stack``.
    """
    window_us = max(1, int(window_us))
    if dt_us is not None:
        dt = max(1, int(dt_us))
        n = max(1, int((window_us + dt - 1) // dt))
    else:
        n = max(1, int(n_frames or 1))
        dt = max(1, int(round(window_us / n)))
        n = max(1, int((window_us + dt - 1) // dt))
    used = min(window_us, n * dt)
    return dt, n, used


def nice_dt_us(dt_us: int) -> int:
    """Snap a frame time to 1-2-5 × 10^n microseconds (at least 1 µs)."""
    raw = max(1, int(dt_us))
    exp = int(math.floor(math.log10(raw)))
    scale = 10**exp
    mant = raw / scale
    best = 1
    best_err = abs(mant - 1.0)
    for candidate in (1, 2, 5, 10):
        err = abs(mant - candidate)
        if err < best_err:
            best_err = err
            best = candidate
    return max(1, int(best * scale))


def plan_nice_pictures(
    window_us: int,
    n_frames: int = OVERVIEW_PICTURES,
) -> tuple[int, int, int]:
    """``plan_pictures`` for ~n frames, with Δt snapped to 1-2-5 µs."""
    dt, _n, _used = plan_pictures(window_us, n_frames=n_frames)
    return plan_pictures(window_us, dt_us=nice_dt_us(dt))


def activity_preview(img: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """Discrete count display plus LUT levels (0…1) for the activity image.

    Gray ``(T, H, W)`` returns ``(display, lo, hi, n_positive)`` with T=1.
    RGB ``(T, H, W, 3)`` is OFF=red / ON=green, same integer steps.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    hi = event_count_ceiling(arr)
    disp = apply_event_scale(arr, hi)
    if arr.ndim == 4:
        energy = np.max(np.abs(arr), axis=-1)
    else:
        energy = np.abs(arr)
    n_pos = int(np.count_nonzero(energy > 0))
    return disp, 0.0, 1.0, n_pos


def event_count_ceiling(stack: np.ndarray, *, percentile: float = 99.0) -> int:
    """Integer event-count ceiling for discrete colour (EVT preview).

    CD bins are counts, not analog intensity: 0, 1, 2, … events per pixel
    per Δt. ``p99`` of **positive raw counts** (unfiltered) is the top
    rung. Empty stacks return 1.
    """
    arr = np.asarray(stack, dtype=np.float32)
    pos = arr[arr > 0]
    if pos.size == 0:
        return 1
    return max(1, int(np.ceil(float(np.percentile(pos, percentile)))))


def event_log1p_ceiling(stack: np.ndarray, *, percentile: float = 99.0) -> float:
    """Deprecated name: integer count ceiling as float (not log1p)."""
    return float(event_count_ceiling(stack, percentile=percentile))


def apply_event_scale(stack: np.ndarray, hi: float) -> np.ndarray:
    """Map integer counts onto ``hi`` discrete rungs in 0…1.

    ``k`` events → ``k / hi`` for ``k = 0…hi`` (clip above). No log1p —
    neighbouring counts stay distinct steps, not a wash.
    """
    cap = max(1, int(round(float(hi))))
    arr = np.maximum(np.asarray(stack, dtype=np.float32), 0.0)
    stepped = np.clip(np.rint(arr), 0, cap)
    return (stepped / cap).astype(np.float32)


def color_preview(stack: np.ndarray, hi: float | None = None) -> np.ndarray:
    """Discrete RGB 0…1. Pass ``hi`` to keep before/after on one ladder."""
    arr = np.asarray(stack, dtype=np.float32)
    if hi is None:
        hi = event_count_ceiling(arr)
    return apply_event_scale(arr, hi)


def removed_fraction(
    before: np.ndarray,
    after: np.ndarray,
    *,
    frame: int | None = None,
) -> float:
    """Fraction of activity removed by filters (0…1).

    Uses the sum of |counts| (ON/OFF planes or RGB: sum of channels).
    Optional ``frame`` indexes the current picture; ``None`` is the whole
    stack.

    Always slices first so a GUI playhead update never materializes a
    full-volume abs() of a colour volume.
    """
    b = np.asarray(before)
    a = np.asarray(after)
    if frame is not None:
        idx = int(frame)
        if idx < 0 or idx >= b.shape[0]:
            return 0.0
        b = b[idx]
        a = a[idx]
    sb = float(np.abs(b).sum(dtype=np.float64))
    if sb <= 0.0:
        return 0.0
    sa = float(np.abs(a).sum(dtype=np.float64))
    return max(0.0, min(1.0, (sb - sa) / sb))


def roi_event_series(
    stack: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    """Event count per picture inside a pixel box (exclusive-end x1, y1).

    Sum, not mean: CD bins are discrete counts, and a 10 % box is mostly
    zeros. ON/OFF planes and RGB sum all channels (R+G = OFF+ON; B is 0).
    """
    arr = np.asarray(stack)
    if arr.ndim < 3:
        return np.zeros((1,), dtype=np.float64)
    n_t = int(arr.shape[0])
    height, width = int(arr.shape[1]), int(arr.shape[2])
    xa = max(0, min(width, int(x0)))
    xb = max(xa + 1, min(width, int(x1)))
    ya = max(0, min(height, int(y0)))
    yb = max(ya + 1, min(height, int(y1)))
    if xa >= xb or ya >= yb:
        return np.zeros((n_t,), dtype=np.float64)
    tile = arr[:, ya:yb, xa:xb]
    return tile.reshape(n_t, -1).sum(axis=1).astype(np.float64, copy=False)


def event_rate_ms(
    store: EventStore,
    n_bins: int = 400,
    *,
    t0_us: int | None = None,
    t1_us: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative time (ms from first event in the file) vs event counts.

    Optional ``t0_us`` / ``t1_us`` histogram only that window; x stays
    aligned to the file origin so a zoomed plot still matches the axis.
    """
    if len(store) == 0:
        return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    origin = int(store.t_min)
    t0 = origin if t0_us is None else int(t0_us)
    t1 = int(store.t_max) if t1_us is None else int(t1_us)
    if t1 <= t0:
        t1 = t0 + 1
    counts, edges = np.histogram(store.t, bins=n_bins, range=(t0, t1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    x_ms = (centers - origin) / 1000.0
    return x_ms, counts.astype(np.float64)


def even_odd_row_ratio(stack: np.ndarray) -> float:
    """max(even, odd) / max(min(even, odd), eps) of per-row mean |count|.

    1.0 means balanced even/odd rows. Empty or single-row stacks return 1.0.
    """
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4:
        arr = np.max(np.abs(arr), axis=-1)
    if arr.ndim != 3 or arr.shape[1] < 2:
        return 1.0
    even = float(np.mean(np.abs(arr[:, 0::2, :])))
    odd = float(np.mean(np.abs(arr[:, 1::2, :])))
    hi = max(even, odd)
    lo = min(even, odd)
    if hi <= 0.0:
        return 1.0
    return hi / max(lo, 1e-12)


def _flatten_rgb_activity(arr: np.ndarray) -> np.ndarray:
    """R+G event activity (not photo luma). Occupancy stays 0/255."""
    energy = arr[..., 0].astype(np.float32) + arr[..., 1].astype(np.float32)
    if arr.dtype == np.uint8:
        return np.where(energy > 0, np.uint8(255), np.uint8(0))
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(energy, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return energy.astype(np.float32)


def encode_stack_for_send(
    stack: np.ndarray,
    *,
    eight_bit: bool = False,
    log_stretch: bool = False,
    normalize: bool = False,
    grayscale: bool = False,
) -> np.ndarray:
    """Apply File-tab-like options, then optionally pack to uint8.

    Default: keep the view dtype (uint8 states, uint16 counts, uint8
    occupancy, int16 signed). 8-bit / log stretch / normalize match the
    BLITZ File tab. Grayscale on Color RGB is ON+OFF activity, not photo luma.
    """
    arr = np.asarray(stack)
    if grayscale and arr.ndim == 4 and arr.shape[-1] >= 2:
        arr = _flatten_rgb_activity(arr)
    if not eight_bit and not log_stretch:
        if normalize:
            return _normalize_per_frame(np.asarray(arr, dtype=np.float32), uint8=False)
        return arr
    if log_stretch:
        return stack_for_network(arr, log_stretch=True)
    if normalize:
        return _normalize_per_frame(np.asarray(arr, dtype=np.float32), uint8=True)
    return stack_for_network(arr, log_stretch=False)


def _normalize_per_frame(arr: np.ndarray, *, uint8: bool) -> np.ndarray:
    out_dtype = np.uint8 if uint8 else np.float32
    hi = 255.0 if uint8 else 1.0
    frames = []
    for frame in arr:
        lo = float(frame.min())
        span = float(frame.max()) - lo
        if span <= 0:
            frames.append(np.zeros(frame.shape, dtype=out_dtype))
            continue
        scaled = (frame - lo) / span * hi
        frames.append(np.clip(scaled, 0, hi).astype(out_dtype))
    return np.stack(frames, axis=0)


def stack_for_network(stack: np.ndarray, *, log_stretch: bool = False) -> np.ndarray:
    """
    Compact stack → uint8 for WOLKE/BLITZ HTTP transfer.

    Default ``log_stretch=False``: raw event counts, clipped to 0…255 (signed:
    mid-grey 128). ``log_stretch=True``: log1p into the full 0…255 range so a
    few hot pixels do not crush typical counts.
    """
    arr = np.asarray(stack, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)

    amin = float(arr.min())
    amax = float(arr.max())

    if not log_stretch:
        if amin < 0.0 < amax:
            return np.clip(np.round(arr) + 128.0, 0, 255).astype(np.uint8)
        return np.clip(np.round(np.maximum(arr, 0.0)), 0, 255).astype(np.uint8)
    if amin < 0.0 < amax:
        mag = np.log1p(np.abs(arr))
        m = float(np.percentile(mag, 99.5)) if mag.size else 0.0
        m = max(m, 1e-6)
        scaled = np.sign(arr) * (mag / m) * 127.0 + 128.0
        return np.clip(scaled, 0, 255).astype(np.uint8)

    if amax <= amin:
        return np.zeros(arr.shape, dtype=np.uint8)

    logged = np.log1p(np.maximum(arr, 0.0))
    pos = logged[logged > 0]
    if pos.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    hi = float(np.percentile(pos, 99.5))
    hi = max(hi, 1e-6)
    scaled = (logged / hi) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)
