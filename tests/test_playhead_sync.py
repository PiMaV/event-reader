"""Unit tests for stack index ↔ timeline mapping."""

from __future__ import annotations

from evt_sidecar.playhead_sync import playhead_s_to_stack_index, stack_index_to_playhead_s


def test_stack_index_to_playhead_roundtrip() -> None:
    store_t_min = 1_000_000
    sent_t0 = 1_500_000
    dt = 10_000
    n = 5
    dur = 2.0
    for i in range(n):
        t_s = stack_index_to_playhead_s(
            i,
            sent_t0_us=sent_t0,
            sent_dt_us=dt,
            store_t_min=store_t_min,
            duration_s=dur,
            n_frames=n,
        )
        back = playhead_s_to_stack_index(
            t_s,
            sent_t0_us=sent_t0,
            sent_dt_us=dt,
            store_t_min=store_t_min,
            n_frames=n,
        )
        assert back == i


def test_playhead_outside_sent_window_is_none() -> None:
    assert (
        playhead_s_to_stack_index(
            0.0,
            sent_t0_us=1_000_000,
            sent_dt_us=10_000,
            store_t_min=0,
            n_frames=3,
        )
        is None
    )
    assert (
        playhead_s_to_stack_index(
            5.0,
            sent_t0_us=1_000_000,
            sent_dt_us=10_000,
            store_t_min=0,
            n_frames=3,
        )
        is None
    )


def test_index_clamped_to_stack() -> None:
    t = stack_index_to_playhead_s(
        99,
        sent_t0_us=0,
        sent_dt_us=1_000_000,
        store_t_min=0,
        duration_s=10.0,
        n_frames=3,
    )
    assert t == 2.0
