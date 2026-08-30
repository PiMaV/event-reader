"""Noise filters: 1-pixel spatial + temporal nearest-neighbour."""

from __future__ import annotations

import numpy as np

from evt_sidecar.filters import drop_isolated_pixels, neighbor_keep_mask


def test_drop_isolated_pixel_keeps_neighbours() -> None:
    frame = np.zeros((5, 5), dtype=np.float32)
    frame[2, 2] = 4.0
    frame[0, 0] = 1.0
    frame[0, 1] = 2.0
    out = drop_isolated_pixels(frame)
    assert out[2, 2] == 0.0
    assert out[0, 0] == 1.0
    assert out[0, 1] == 2.0


def test_drop_isolated_signed_and_stack() -> None:
    stack = np.zeros((2, 3, 3), dtype=np.float32)
    stack[0, 1, 1] = -3.0
    stack[1, 0, 0] = 1.0
    stack[1, 0, 1] = 1.0
    out = drop_isolated_pixels(stack)
    assert out[0, 1, 1] == 0.0
    assert out[1, 0, 0] == 1.0
    assert out[1, 0, 1] == 1.0


def test_neighbor_keeps_second_of_pair() -> None:
    # Isolated first event at (2,2); second at neighbour (3,2) 200 µs later.
    t = np.array([1000, 1200, 50_000], dtype=np.int64)
    x = np.array([2, 3, 7], dtype=np.int32)
    y = np.array([2, 2, 7], dtype=np.int32)
    keep = neighbor_keep_mask(
        t, x, y, height=10, width=10, dt_us=5_000, radius=1
    )
    assert list(keep) == [False, True, False]


def test_neighbor_same_pixel_within_dt() -> None:
    t = np.array([0, 500], dtype=np.int64)
    x = np.array([1, 1], dtype=np.int32)
    y = np.array([1, 1], dtype=np.int32)
    keep = neighbor_keep_mask(t, x, y, height=4, width=4, dt_us=1000, radius=1)
    assert list(keep) == [False, True]
    keep_tight = neighbor_keep_mask(
        t, x, y, height=4, width=4, dt_us=100, radius=1
    )
    assert list(keep_tight) == [False, False]
