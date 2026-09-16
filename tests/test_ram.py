"""RAM soft-limit colours for planned stacks (comfort zone)."""

from __future__ import annotations

from evt_sidecar.ram import (
    COMFORT_WIRE_BYTES,
    RED_WARN_FRAMES,
    RED_WIRE_BYTES,
    RamSnapshot,
    assess_stack,
    fmt_bytes,
    fmt_count,
    payload_bytes,
)


def test_fmt_bytes() -> None:
    assert fmt_bytes(512) == "512 B"
    assert "MB" in fmt_bytes(5 * 1024**2)
    assert "GB" in fmt_bytes(3 * 1024**3)


def test_fmt_count() -> None:
    assert fmt_count(921) == "921"
    assert fmt_count(921_000) == "921k"
    assert fmt_count(138_000_000) == "138M"


def test_levels_comfort_not_installed_share() -> None:
    # 1280×720 uint8: 500 frames ≈ 0.43 GiB → under 2 GiB and ≤1000 → ok
    ram = RamSnapshot(total=16 * 1024**3, available=16 * 1024**3)
    ok = assess_stack(500, 720, 1280, ram, wire_itemsize=1)
    assert ok.level == "ok"
    assert not ok.nav_warn
    assert not ok.wire_warn
    assert ok.wire_bytes == payload_bytes(500, 720, 1280, 1)
    assert ok.n_voxels == 500 * 720 * 1280

    # ≥2000 pictures → red (allowed with confirm)
    red_frames = assess_stack(2000, 720, 1280, ram, wire_itemsize=1)
    assert red_frames.level == "red"
    assert red_frames.nav_warn
    assert red_frames.n_frames >= RED_WARN_FRAMES

    red_more = assess_stack(5000, 720, 1280, ram, wire_itemsize=1)
    assert red_more.level == "red"


def test_nav_warn_above_1000_even_when_wire_ok() -> None:
    ram = RamSnapshot(total=16 * 1024**3, available=16 * 1024**3)
    budget = assess_stack(1500, 720, 1280, ram, wire_itemsize=1)
    assert budget.nav_warn
    assert budget.wire_bytes < COMFORT_WIRE_BYTES
    assert budget.ram_level == "ok"
    assert budget.level == "yellow"
    assert budget.banner_theme()[2] == "MANY PICTURES"


def test_64gb_machine_2gb_stack_is_yellow() -> None:
    """Comfort is absolute — 2.2 GB on 64 GB must not stay green."""
    ram = RamSnapshot(total=64 * 1024**3, available=64 * 1024**3)
    # 1200 × 720 × 1280 × uint16 ≈ 2.06 GiB
    budget = assess_stack(1200, 720, 1280, ram, wire_itemsize=2)
    assert budget.wire_bytes > COMFORT_WIRE_BYTES
    assert budget.wire_bytes < RED_WIRE_BYTES
    assert budget.level == "yellow"
    assert budget.wire_warn
    assert "LARGE" in budget.banner_theme()[2]


def test_block_when_wire_exceeds_free() -> None:
    ram = RamSnapshot(total=32 * 1024**3, available=500 * 1024**2)
    # 800 frames × 720 × 1280 ≈ 703 MB uint8 > 90% of 500 MB
    blocked = assess_stack(800, 720, 1280, ram, wire_itemsize=1)
    assert blocked.level == "block"


def test_banner_theme_is_loud() -> None:
    from evt_sidecar.ram import BANNER_THEME

    assert BANNER_THEME["ok"][2] == "RAM OK"
    assert BANNER_THEME["yellow"][2] == "YELLOW"
    assert BANNER_THEME["red"][2] == "RED"
    assert BANNER_THEME["block"][2] == "TOO BIG"
    assert BANNER_THEME["yellow"][0] != BANNER_THEME["ok"][0]
    assert BANNER_THEME["red"][0] != BANNER_THEME["yellow"][0]


def test_escalate_to_red_when_build_is_tight() -> None:
    # wire under 2 GiB comfort, but 2×uint16 build exceeds 90% of a small free pool
    ram = RamSnapshot(total=64 * 1024**3, available=700 * 1024**2)
    budget = assess_stack(200, 720, 1280, ram, wire_itemsize=1)
    assert budget.wire_bytes < COMFORT_WIRE_BYTES
    assert budget.level == "red"
