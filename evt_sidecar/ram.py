"""Soft limits for picture stacks: comfort zone, not share of installed RAM."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

# Internal bin is 2×uint16 ON/OFF planes. Wire depends on the send view.
WIRE_BYTES_PER_PIXEL = 2
BUILD_BYTES_PER_PIXEL = 4  # 2 channels × uint16

# Comfort / pain thresholds (absolute). Installed-RAM share is display-only.
COMFORT_WIRE_BYTES = 2 * 1024**3  # yellow above this
RED_WIRE_BYTES = 8 * 1024**3  # red at or above this

# Refuse if the wire stack itself would not fit in *free* RAM
WIRE_AVAILABLE_FRAC = 0.90

# Timeline: more frames still work, but scrubbing stops being comfortable
NAV_WARN_FRAMES = 1000
RED_WARN_FRAMES = 2000

# Bar ticks as fractions of RED_WIRE_BYTES (fill = wire / RED_WIRE_BYTES)
YELLOW_TICK = COMFORT_WIRE_BYTES / RED_WIRE_BYTES  # 0.25 → 2 GB
RED_TICK = 1.0  # 8 GB

# Kept for callers that still import old names (bar / tooltips).
YELLOW_FRAC = YELLOW_TICK
RED_FRAC = RED_TICK

RamLevel = Literal["ok", "yellow", "red", "block"]

_LEVEL_RANK: dict[RamLevel, int] = {
    "ok": 0,
    "yellow": 1,
    "red": 2,
    "block": 3,
}

BANNER_THEME: dict[RamLevel, tuple[str, str, str]] = {
    # background, foreground, headline tag
    "ok": ("#1b4332", "#d8f3dc", "RAM OK"),
    "yellow": ("#f1c40f", "#1a1400", "YELLOW"),
    "red": ("#e74c3c", "#ffffff", "RED"),
    "block": ("#6b0000", "#ffffff", "TOO BIG"),
}

FILL_COLOR: dict[RamLevel, str] = {
    "ok": "#2ecc71",
    "yellow": "#f4d03f",
    "red": "#ff6b5a",
    "block": "#ff8a80",
}

_FALLBACK_TOTAL = 16 * 1024**3
_FALLBACK_AVAILABLE = 8 * 1024**3


@dataclass(frozen=True)
class RamSnapshot:
    total: int
    available: int


@dataclass(frozen=True)
class StackBudget:
    n_frames: int
    height: int
    width: int
    n_voxels: int
    wire_bytes: int
    build_bytes: int
    fraction_of_total: float
    level: RamLevel
    ram_level: RamLevel
    nav_warn: bool
    wire_warn: bool

    def banner_theme(self) -> tuple[str, str, str]:
        bg, fg, tag = BANNER_THEME[self.level]
        if self.level == "ok":
            return bg, fg, tag
        if self.level == "yellow":
            parts: list[str] = []
            if self.wire_warn:
                parts.append("LARGE")
            if self.nav_warn:
                parts.append("MANY PICTURES")
            if parts:
                return bg, fg, " · ".join(parts)
            return bg, fg, tag
        extras: list[str] = []
        if self.wire_warn:
            extras.append("LARGE")
        if self.nav_warn:
            extras.append("MANY PICTURES")
        if extras:
            return bg, fg, f"{tag} · " + " · ".join(extras)
        return bg, fg, tag

    def fill_color(self) -> str:
        return FILL_COLOR[self.level]

    def bar_fraction(self) -> float:
        """Fill for the meter: wire size relative to the red comfort ceiling."""
        if RED_WIRE_BYTES <= 0:
            return 0.0
        return self.wire_bytes / RED_WIRE_BYTES


def read_ram() -> RamSnapshot:
    """Installed RAM and currently available RAM (best effort)."""
    if sys.platform.startswith("linux"):
        snap = _read_proc_meminfo()
        if snap is not None:
            return snap
    if sys.platform == "win32":
        snap = _read_windows()
        if snap is not None:
            return snap
    return RamSnapshot(total=_FALLBACK_TOTAL, available=_FALLBACK_AVAILABLE)


def _read_proc_meminfo() -> RamSnapshot | None:
    total = 0
    available = 0
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except OSError:
        return None
    if total <= 0:
        return None
    return RamSnapshot(total=total, available=available or total)


def _read_windows() -> RamSnapshot | None:
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)) == 0:
            return None
        total = int(st.ullTotalPhys)
        avail = int(st.ullAvailPhys)
        if total <= 0:
            return None
        return RamSnapshot(total=total, available=avail or total)
    except (OSError, AttributeError, ValueError):
        return None


def payload_bytes(n_frames: int, height: int, width: int, bytes_per: int) -> int:
    n = max(0, int(n_frames))
    h = max(0, int(height))
    w = max(0, int(width))
    return n * h * w * int(bytes_per)


def _worse(a: RamLevel, b: RamLevel) -> RamLevel:
    return a if _LEVEL_RANK[a] >= _LEVEL_RANK[b] else b


def assess_stack(
    n_frames: int,
    height: int,
    width: int,
    ram: RamSnapshot | None = None,
    *,
    wire_itemsize: int = 2,
    build_itemsize: int | None = None,
) -> StackBudget:
    """
    Colour the planned stack against absolute comfort thresholds.

    OK: wire ≤ 2 GiB and ≤ 1000 pictures.
    Yellow: above that (still allowed).
    Red: wire ≥ 8 GiB or ≥ 2000 pictures (confirm); or build buffer tight on free RAM.
    ``block`` if the wire stack itself would not fit in *available* RAM.
    Share of installed RAM is informational only — it does not pick the colour.
    """
    ram = ram if ram is not None else read_ram()
    n = max(1, int(n_frames))
    h = max(0, int(height))
    w = max(0, int(width))
    item = max(1, int(wire_itemsize))
    build_item = BUILD_BYTES_PER_PIXEL if build_itemsize is None else max(1, int(build_itemsize))
    wire = payload_bytes(n, h, w, item)
    build = payload_bytes(n, h, w, build_item)
    voxels = n * h * w
    frac = wire / ram.total if ram.total > 0 else 0.0

    nav_warn = n > NAV_WARN_FRAMES
    wire_warn = wire > COMFORT_WIRE_BYTES

    if ram.available > 0 and wire > WIRE_AVAILABLE_FRAC * ram.available:
        wire_level: RamLevel = "block"
    elif wire >= RED_WIRE_BYTES:
        wire_level = "red"
    elif wire_warn:
        wire_level = "yellow"
    else:
        wire_level = "ok"

    if n >= RED_WARN_FRAMES:
        frame_level: RamLevel = "red"
    elif nav_warn:
        frame_level = "yellow"
    else:
        frame_level = "ok"

    level = _worse(wire_level, frame_level)
    ram_level = wire_level
    if (
        level in ("ok", "yellow")
        and ram.available > 0
        and build > WIRE_AVAILABLE_FRAC * ram.available
    ):
        level = "red"

    return StackBudget(
        n_frames=n,
        height=h,
        width=w,
        n_voxels=voxels,
        wire_bytes=wire,
        build_bytes=build,
        fraction_of_total=frac,
        level=level,
        ram_level=ram_level,
        nav_warn=nav_warn,
        wire_warn=wire_warn,
    )


def fmt_bytes(n: int) -> str:
    n = max(0, int(n))
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def fmt_count(n: int) -> str:
    """Compact count for voxel / picture headlines (921k, 138M)."""
    n = max(0, int(n))
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B".replace(".0B", "B")
    if n >= 1_000_000:
        val = n / 1_000_000
        return f"{val:.0f}M" if val >= 10 else f"{val:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        val = n / 1_000
        return f"{val:.0f}k" if val >= 10 else f"{val:.1f}k".replace(".0k", "k")
    return str(n)
