"""RAM-based soft limits for picture stacks (no magic frame count)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

# Internal bin is 2×uint16 ON/OFF planes. Wire depends on the send view.
WIRE_BYTES_PER_PIXEL = 2
BUILD_BYTES_PER_PIXEL = 4  # 2 channels × uint16

# Fractions of *installed* RAM (MemTotal / ullTotalPhys)
YELLOW_FRAC = 1 / 8
RED_FRAC = 1 / 4

# Refuse if the wire stack itself would not fit in *free* RAM
WIRE_AVAILABLE_FRAC = 0.90

# BLITZ timeline: more frames still work, but scrubbing stops being comfortable
NAV_WARN_FRAMES = 1000

RamLevel = Literal["ok", "yellow", "red", "block"]

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
    wire_bytes: int
    build_bytes: int
    fraction_of_total: float
    level: RamLevel
    ram_level: RamLevel
    nav_warn: bool

    def banner_theme(self) -> tuple[str, str, str]:
        bg, fg, tag = BANNER_THEME[self.level]
        if self.nav_warn and self.ram_level == "ok":
            return bg, fg, "MANY PICTURES"
        if self.nav_warn:
            return bg, fg, f"{tag} · MANY PICTURES"
        return bg, fg, tag

    def fill_color(self) -> str:
        return FILL_COLOR[self.level]


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
    Colour the planned stack against installed RAM (default uint16 = 2 bytes/px).

    Yellow / red use the user's fractions of *total* RAM (stable).
    ``block`` if the wire stack itself would not fit in *available* RAM.
    If the ON/OFF uint16 build buffer is tight on free RAM, escalate to red.
    """
    ram = ram if ram is not None else read_ram()
    n = max(1, int(n_frames))
    item = max(1, int(wire_itemsize))
    build_item = BUILD_BYTES_PER_PIXEL if build_itemsize is None else max(1, int(build_itemsize))
    wire = payload_bytes(n, height, width, item)
    build = payload_bytes(n, height, width, build_item)
    frac = wire / ram.total if ram.total > 0 else 0.0
    if ram.available > 0 and wire > WIRE_AVAILABLE_FRAC * ram.available:
        level: RamLevel = "block"
    elif frac >= RED_FRAC:
        level = "red"
    elif frac >= YELLOW_FRAC:
        level = "yellow"
    else:
        level = "ok"
    if (
        level in ("ok", "yellow")
        and ram.available > 0
        and build > WIRE_AVAILABLE_FRAC * ram.available
    ):
        level = "red"
    ram_level = level
    nav_warn = n > NAV_WARN_FRAMES
    if ram_level == "ok" and nav_warn:
        level = "yellow"
    return StackBudget(
        n_frames=n,
        height=int(height),
        width=int(width),
        wire_bytes=wire,
        build_bytes=build,
        fraction_of_total=frac,
        level=level,
        ram_level=ram_level,
        nav_warn=nav_warn,
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
