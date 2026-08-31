"""RAM soft-limit colours for planned stacks."""

from __future__ import annotations

from evt_sidecar.ram import RamSnapshot, assess_stack, fmt_bytes, payload_bytes


def test_fmt_bytes() -> None:
    assert fmt_bytes(512) == "512 B"
    assert "MB" in fmt_bytes(5 * 1024**2)
    assert "GB" in fmt_bytes(3 * 1024**3)


def test_levels_16gb_hd_stack() -> None:
    # 1280×720 uint8: 500 frames ≈ 0.44 GiB → ~2.7% of 16 GB → ok
    ram = RamSnapshot(total=16 * 1024**3, available=16 * 1024**3)
    ok = assess_stack(500, 720, 1280, ram, wire_itemsize=1)
    assert ok.level == "ok"
    assert not ok.nav_warn
    assert ok.wire_bytes == payload_bytes(500, 720, 1280, 1)

    yellow = assess_stack(3000, 720, 1280, ram, wire_itemsize=1)
    assert yellow.level == "yellow"
    assert yellow.nav_warn

    red = assess_stack(5000, 720, 1280, ram, wire_itemsize=1)
    assert red.level == "red"


def test_nav_warn_above_1000_even_when_ram_ok() -> None:
    ram = RamSnapshot(total=16 * 1024**3, available=16 * 1024**3)
    budget = assess_stack(1500, 720, 1280, ram, wire_itemsize=1)
    assert budget.nav_warn
    assert budget.ram_level == "ok"
    assert budget.level == "yellow"
    assert budget.banner_theme()[2] == "MANY PICTURES"


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
    # uint8 well under 1/8 of 64 GB, but 2×uint16 build (~4×) exceeds 90% of a small free pool
    ram = RamSnapshot(total=64 * 1024**3, available=700 * 1024**2)
    budget = assess_stack(200, 720, 1280, ram, wire_itemsize=1)
    assert budget.fraction_of_total < 1 / 8
    assert budget.level == "red"
