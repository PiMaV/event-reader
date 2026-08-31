"""Optional noise filters for the local preview and the BLITZ send.

Both filters are off by default. Preview and send use the same flags.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def drop_isolated_pixels(stack: np.ndarray) -> np.ndarray:
    """Zero pixels that are non-zero while all 8 neighbours are zero.

    Works on (H, W), (T, H, W), ON/OFF (T, H, W, 2), or RGB (T, H, W, 3).
    Signed counts and multi-channel use abs() / channel-max so an isolated
    OFF-only pixel is removed too. Preserves the input dtype.

    Call from a worker thread for full overview stacks — a
    150×720×1280×2 volume must not run on the Qt GUI thread.
    """
    orig = np.asarray(stack)
    orig_dtype = orig.dtype
    arr = np.array(orig, dtype=np.float32, copy=True)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    if arr.ndim == 4:
        if arr.shape[-1] not in (2, 3):
            raise ValueError(
                f"expected last-axis 2 (ON/OFF) or 3 (RGB), got shape {arr.shape}"
            )
        _zero_isolated_n4(arr)
    elif arr.ndim == 3:
        _zero_isolated_n3(arr)
    else:
        raise ValueError(
            f"expected (H, W), (T, H, W), (T, H, W, 2), or (T, H, W, 3), "
            f"got shape {arr.shape}"
        )
    if squeeze:
        arr = arr[0]
    if orig_dtype == np.float32:
        return arr
    return arr.astype(orig_dtype, copy=False)


@njit(cache=True)
def _zero_isolated_n3(arr):
    """In-place: zero isolated pixels in (T, H, W)."""
    t_n, height, width = arr.shape
    iso = np.zeros((t_n, height, width), dtype=np.uint8)
    for t in range(t_n):
        for y in range(height):
            for x in range(width):
                energy = arr[t, y, x]
                if energy < 0.0:
                    energy = -energy
                if energy <= 0.0:
                    continue
                found = False
                y0 = 0 if y == 0 else y - 1
                y1 = height if y >= height - 1 else y + 2
                x0 = 0 if x == 0 else x - 1
                x1 = width if x >= width - 1 else x + 2
                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        if yy == y and xx == x:
                            continue
                        neigh = arr[t, yy, xx]
                        if neigh < 0.0:
                            neigh = -neigh
                        if neigh > 0.0:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    iso[t, y, x] = 1
    for t in range(t_n):
        for y in range(height):
            for x in range(width):
                if iso[t, y, x]:
                    arr[t, y, x] = 0.0


@njit(cache=True)
def _zero_isolated_n4(arr):
    """In-place: zero isolated pixels in (T, H, W, C) using channel-max."""
    t_n, height, width, n_c = arr.shape
    iso = np.zeros((t_n, height, width), dtype=np.uint8)
    for t in range(t_n):
        for y in range(height):
            for x in range(width):
                energy = 0.0
                for c in range(n_c):
                    val = arr[t, y, x, c]
                    if val < 0.0:
                        val = -val
                    if val > energy:
                        energy = val
                if energy <= 0.0:
                    continue
                found = False
                y0 = 0 if y == 0 else y - 1
                y1 = height if y >= height - 1 else y + 2
                x0 = 0 if x == 0 else x - 1
                x1 = width if x >= width - 1 else x + 2
                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        if yy == y and xx == x:
                            continue
                        neigh = 0.0
                        for c in range(n_c):
                            val = arr[t, yy, xx, c]
                            if val < 0.0:
                                val = -val
                            if val > neigh:
                                neigh = val
                        if neigh > 0.0:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    iso[t, y, x] = 1
    for t in range(t_n):
        for y in range(height):
            for x in range(width):
                if iso[t, y, x]:
                    for c in range(n_c):
                        arr[t, y, x, c] = 0.0


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
