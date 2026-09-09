"""Map Viewer Contract stack index ↔ Event reader timeline seconds."""

from __future__ import annotations


def stack_index_to_playhead_s(
    index: int,
    *,
    sent_t0_us: int,
    sent_dt_us: int,
    store_t_min: int,
    duration_s: float,
    n_frames: int,
) -> float:
    """Absolute playhead time (s from recording start) for BLITZ/DONNER frame ``index``."""
    n = max(1, int(n_frames))
    idx = max(0, min(int(index), n - 1))
    dt = max(1, int(sent_dt_us))
    t_abs = int(sent_t0_us) + idx * dt
    t_s = (t_abs - int(store_t_min)) / 1_000_000.0
    dur = max(0.0, float(duration_s))
    return max(0.0, min(dur, t_s))


def playhead_s_to_stack_index(
    t_s: float,
    *,
    sent_t0_us: int,
    sent_dt_us: int,
    store_t_min: int,
    n_frames: int,
) -> int | None:
    """
    Stack frame under the playhead, or ``None`` if outside the last-sent window.

    Used so Event reader scrub can ``push_index`` only while the playhead
    sits on a frame that exists in the cube BLITZ/DONNER already hold.
    """
    n = int(n_frames)
    dt = int(sent_dt_us)
    if n <= 0 or dt <= 0:
        return None
    t_abs = int(store_t_min) + int(round(float(t_s) * 1_000_000.0))
    idx = int(round((t_abs - int(sent_t0_us)) / dt))
    if idx < 0 or idx >= n:
        return None
    return idx
