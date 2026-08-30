"""Optional noise filters for the EVT → BLITZ export path.

Overview pictures stay raw. Both filters are off by default.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def drop_isolated_pixels(stack: np.ndarray) -> np.ndarray:
    """Zero pixels that are non-zero while all 8 neighbours are zero.

    Works on (H, W) or (T, H, W). Signed counts use abs() so an isolated
    OFF-only pixel is removed too.
    """
    arr = np.array(stack, dtype=np.float32, copy=True)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    if arr.ndim != 3:
        raise ValueError(f"expected (H, W) or (T, H, W), got shape {arr.shape}")

    abs_a = np.abs(arr)
    padded = np.pad(abs_a, ((0, 0), (1, 1), (1, 1)), mode="constant")
    neigh = np.maximum.reduce(
        [
            padded[:, 0:-2, 0:-2],
            padded[:, 0:-2, 1:-1],
            padded[:, 0:-2, 2:],
            padded[:, 1:-1, 0:-2],
            padded[:, 1:-1, 2:],
            padded[:, 2:, 0:-2],
            padded[:, 2:, 1:-1],
            padded[:, 2:, 2:],
        ]
    )
    isolated = (abs_a > 0) & (neigh == 0)
    arr[isolated] = 0.0
    if squeeze:
        return arr[0]
    return arr


def neighbor_keep_mask(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    height: int,
    width: int,
    dt_us: int,
    radius: int = 1,
) -> np.ndarray:
    """Keep events that have a 3×3 (radius) neighbour within ``dt_us``.

    Causal background-activity / nearest-neighbour filter: an event is kept
    if any pixel in the square neighbourhood (including the same pixel)
    already fired at a time ``t_prev`` with ``0 <= t - t_prev <= dt_us``.
    Events are assumed sorted by time.
    """
    n = int(t.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.bool_)
    dt = max(1, int(dt_us))
    rad = max(0, int(radius))
    h = max(1, int(height))
    w = max(1, int(width))
    t64 = np.ascontiguousarray(t, dtype=np.int64)
    x32 = np.ascontiguousarray(x, dtype=np.int32)
    y32 = np.ascontiguousarray(y, dtype=np.int32)
    return np.asarray(_neighbor_keep(t64, x32, y32, h, w, dt, rad))


@njit(cache=True)
def _neighbor_keep(t, x, y, height, width, dt_us, radius):
    n = t.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    last = np.full(height * width, np.int64(-1))
    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        ti = int(t[i])
        if xi < 0 or yi < 0 or xi >= width or yi >= height:
            continue
        y0 = yi - radius
        if y0 < 0:
            y0 = 0
        y1 = yi + radius + 1
        if y1 > height:
            y1 = height
        x0 = xi - radius
        if x0 < 0:
            x0 = 0
        x1 = xi + radius + 1
        if x1 > width:
            x1 = width
        found = False
        yy = y0
        while yy < y1:
            row = yy * width
            xx = x0
            while xx < x1:
                prev = last[row + xx]
                if prev >= 0:
                    delta = ti - prev
                    if 0 <= delta <= dt_us:
                        found = True
                        break
                xx += 1
            if found:
                break
            yy += 1
        keep[i] = found
        last[yi * width + xi] = ti
    return keep
