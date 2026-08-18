"""Accumulate CD events into a dense (T, H, W) frame stack for BLITZ."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .evt3 import EventStore


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
    max_frames: int | None = 2000  # safety cap for RAM / Network transfer

    def clamp_dt(self) -> int:
        return max(1, int(self.dt_us))


def bin_events(store: EventStore, params: BinParams) -> np.ndarray:
    """
    Return float32 stack shaped (T, height, width) — OpenCV/image convention.
    BLITZ DataLoader swapaxes(1,2) → ImageData (T, W, H).
    """
    if len(store) == 0:
        return np.zeros((1, store.height, store.width), dtype=np.float32)

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
        return np.zeros((1, store.height, store.width), dtype=np.float32)

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
        return np.zeros((1, store.height, store.width), dtype=np.float32)

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

    h, w = store.height, store.width
    # Drop out-of-bounds coordinates
    inb = (x_s < w) & (y_s < h)
    frame_idx = frame_idx[inb]
    x_s = x_s[inb]
    y_s = y_s[inb]
    weights = weights[inb]

    stack = np.zeros((n_frames, h, w), dtype=np.float32)
    # Flat index: frame * H*W + y * W + x
    flat = frame_idx * (h * w) + y_s.astype(np.int64) * w + x_s.astype(np.int64)
    np.add.at(stack.ravel(), flat, weights)
    return stack


HARD_FRAME_CAP = 2000
SENSOR_DT_US = 1  # EVT3 timestamp tick
OVERVIEW_PICTURES = 150


def min_dt_us(window_us: int, cap: int = HARD_FRAME_CAP) -> int:
    """Finest frame time that still fits `cap` pictures (not below 1 µs)."""
    window_us = max(1, int(window_us))
    return max(SENSOR_DT_US, int((window_us + cap - 1) // cap))


def plan_pictures(
    window_us: int,
    *,
    dt_us: int | None = None,
    n_frames: int | None = None,
    cap: int = HARD_FRAME_CAP,
) -> tuple[int, int, bool, int]:
    """
    Map a time window to (dt_us, n_frames, capped, used_window_us).

    Drive with either frame time (dt_us) or picture count (n_frames).
    If the result exceeds `cap`, keep dt and take the first `cap` frames
    (used_window_us may be shorter than window_us).
    """
    window_us = max(1, int(window_us))
    if dt_us is not None:
        dt = max(1, int(dt_us))
        n = max(1, int((window_us + dt - 1) // dt))
    else:
        n = max(1, int(n_frames or 1))
        dt = max(1, int(round(window_us / n)))
        n = max(1, int((window_us + dt - 1) // dt))
    capped = n > cap
    if capped:
        n = cap
    used = min(window_us, n * dt)
    return dt, n, capped, used


def event_rate_ms(store: EventStore, n_bins: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Relative time (ms) vs event counts for a region slider plot."""
    if len(store) == 0:
        return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    t0 = int(store.t_min)
    t1 = int(store.t_max)
    if t1 <= t0:
        t1 = t0 + 1
    counts, edges = np.histogram(store.t, bins=n_bins, range=(t0, t1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    x_ms = (centers - t0) / 1000.0
    return x_ms, counts.astype(np.float64)


def stack_for_network(stack: np.ndarray) -> np.ndarray:
    """
    Compact float stack → uint8 for WOLKE/BLITZ HTTP transfer.

    Sparse event counts are heavy-tailed (max ≫ typical). Linear min–max mapping
    rounds almost all pixels to 0 — use log1p (+ high percentile clip) instead.
    """
    arr = np.asarray(stack, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)

    amin = float(arr.min())
    amax = float(arr.max())
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
