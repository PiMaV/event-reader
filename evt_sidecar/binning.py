"""Accumulate CD events into a dense (T, H, W) frame stack for BLITZ."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .evt3 import EventStore
from .filters import drop_isolated_pixels, neighbor_keep_mask


class PolarityMode(str, Enum):
    ON = "on"
    OFF = "off"
    BOTH = "both"
    SIGNED = "signed"  # ON=+1, OFF=-1


class AccumMode(str, Enum):
    COUNT = "count"


@dataclass
class BinParams:
    dt_us: int = 1000
    polarity: PolarityMode = PolarityMode.BOTH
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

    def clamp_dt(self) -> int:
        return max(1, int(self.dt_us))


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


def bin_events(store: EventStore, params: BinParams) -> np.ndarray:
    """
    Return float32 stack shaped (T, height, width) — OpenCV/image convention.
    BLITZ DataLoader swapaxes(1,2) → ImageData (T, W, H).
    """
    x0, y0, x1, y1 = crop_box(store, params)
    out_h, out_w = y1 - y0, x1 - x0

    def empty(n: int = 1) -> np.ndarray:
        return np.zeros((n, out_h, out_w), dtype=np.float32)

    if len(store) == 0:
        return empty()

    dt = params.clamp_dt()
    t0 = store.t_min if params.t0_us is None else int(params.t0_us)
    t1 = store.t_max if params.t1_us is None else int(params.t1_us)
    if t1 < t0:
        t0, t1 = t1, t0

    # Slice events in [t0, t1]
    t = store.t
    lo = int(np.searchsorted(t, t0, side="left"))
    hi = int(np.searchsorted(t, t1, side="right"))
    if hi <= lo:
        return empty()

    t_s = t[lo:hi]
    x_s = store.x[lo:hi]
    y_s = store.y[lo:hi]
    p_s = store.p[lo:hi]

    # Polarity filter / weights
    if params.polarity == PolarityMode.ON:
        mask = p_s == 1
        weights = np.ones(mask.sum(), dtype=np.float32)
        t_s, x_s, y_s = t_s[mask], x_s[mask], y_s[mask]
    elif params.polarity == PolarityMode.OFF:
        mask = p_s == 0
        weights = np.ones(mask.sum(), dtype=np.float32)
        t_s, x_s, y_s = t_s[mask], x_s[mask], y_s[mask]
    elif params.polarity == PolarityMode.SIGNED:
        weights = np.where(p_s == 1, 1.0, -1.0).astype(np.float32)
    else:
        weights = np.ones(t_s.shape[0], dtype=np.float32)

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
        t_s, x_s, y_s, weights = t_s[keep_nn], x_s[keep_nn], y_s[keep_nn], weights[keep_nn]
        if t_s.shape[0] == 0:
            return empty()

    # Frame index relative to t0
    frame_idx = ((t_s.astype(np.int64) - t0) // dt).astype(np.int64)
    n_frames = int(frame_idx.max()) + 1
    if params.max_frames is not None and n_frames > params.max_frames:
        # Keep earliest max_frames bins
        keep = frame_idx < params.max_frames
        frame_idx = frame_idx[keep]
        x_s = x_s[keep]
        y_s = y_s[keep]
        weights = weights[keep]
        n_frames = params.max_frames

    inb = (x_s >= x0) & (x_s < x1) & (y_s >= y0) & (y_s < y1)
    frame_idx = frame_idx[inb]
    x_s = x_s[inb]
    y_s = y_s[inb]
    weights = weights[inb]
    if frame_idx.shape[0] == 0:
        return empty(n_frames)

    xs = x_s.astype(np.int64) - x0
    ys = y_s.astype(np.int64) - y0
    stack = np.zeros((n_frames, out_h, out_w), dtype=np.float32)
    flat = frame_idx * (out_h * out_w) + ys * out_w + xs
    np.add.at(stack.ravel(), flat, weights)
    if params.drop_isolated:
        stack = drop_isolated_pixels(stack)
    return stack


SENSOR_DT_US = 1  # EVT3 timestamp tick
OVERVIEW_PICTURES = 150
OVERVIEW_MIN_DT_US = 1000  # 1 ms — finer overview bins look like empty stripes
INTERLACE_DT_US = 1000  # hint only when send Δt is below 1 ms
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


def plan_overview(
    window_us: int,
    *,
    n_frames: int = OVERVIEW_PICTURES,
    min_dt_us: int = OVERVIEW_MIN_DT_US,
) -> tuple[int, int, int]:
    """Overview pictures for a window, but never finer than ``min_dt_us``.

    A short zoom would otherwise make Δt ≪ 1 ms (150 pictures over 40 ms
    → 0.27 ms). Those frames are almost empty and look like readout stripes.
    Send Δt can still go down to 1 µs.
    """
    dt_us, n, used = plan_pictures(window_us, n_frames=n_frames)
    if dt_us < min_dt_us:
        return plan_pictures(window_us, dt_us=min_dt_us)
    return dt_us, n, used


def activity_preview(img: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """log1p activity stack (1, H, W) plus LUT levels so hot pixels do not crush.

    Returns ``(display, lo, hi, n_positive)``. All-zero images get levels 0…1.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    disp = np.log1p(np.maximum(arr, 0.0))
    pos = disp[disp > 0]
    n_pos = int(pos.size)
    if n_pos == 0:
        return disp[None, ...], 0.0, 1.0, 0
    hi = float(np.percentile(pos, 99.0))
    hi = max(hi, float(np.min(pos)), 1e-6)
    return disp[None, ...], 0.0, hi, n_pos


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
    if arr.ndim != 3 or arr.shape[1] < 2:
        return 1.0
    even = float(np.mean(np.abs(arr[:, 0::2, :])))
    odd = float(np.mean(np.abs(arr[:, 1::2, :])))
    hi = max(even, odd)
    lo = min(even, odd)
    if hi <= 0.0:
        return 1.0
    return hi / max(lo, 1e-12)


def encode_stack_for_send(
    stack: np.ndarray,
    *,
    eight_bit: bool = False,
    log_stretch: bool = False,
    normalize: bool = False,
    grayscale: bool = False,
) -> np.ndarray:
    """Apply File-tab-like options, then optionally pack to uint8.

    Default: float32 event counts (no 255 clip). 8-bit / log stretch / normalize
    match the BLITZ File tab knobs so sidecar and Stream stay consistent.
    """
    arr = np.asarray(stack, dtype=np.float32)
    if grayscale and arr.ndim == 4 and arr.shape[-1] == 3:
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        arr = np.sum(arr * weights, axis=-1).astype(np.float32)
    if not eight_bit:
        if normalize:
            return _normalize_per_frame(arr, uint8=False)
        return arr
    if log_stretch:
        return stack_for_network(arr, log_stretch=True)
    if normalize:
        return _normalize_per_frame(arr, uint8=True)
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
    Compact float stack → uint8 for WOLKE/BLITZ HTTP transfer.

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
        # Signed: log-magnitude around mid-grey
        mag = np.log1p(np.abs(arr))
        m = float(np.percentile(mag, 99.5)) if mag.size else 0.0
        m = max(m, 1e-6)
        scaled = np.sign(arr) * (mag / m) * 127.0 + 128.0
        return np.clip(scaled, 0, 255).astype(np.uint8)

    if amax <= amin:
        return np.zeros(arr.shape, dtype=np.uint8)

    # Non-negative counts: log1p, clip to p99.5 of positive pixels
    logged = np.log1p(np.maximum(arr, 0.0))
    pos = logged[logged > 0]
    if pos.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    hi = float(np.percentile(pos, 99.5))
    hi = max(hi, 1e-6)
    scaled = (logged / hi) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)
