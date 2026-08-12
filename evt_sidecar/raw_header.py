"""Parse Prophesee/IDS EVT3 RAW ASCII header (lines starting with '%')."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawHeader:
    width: int
    height: int
    format_name: str
    evt_version: str
    metadata: dict[str, str] = field(default_factory=dict)
    header_bytes: int = 0

    @property
    def geometry(self) -> tuple[int, int]:
        return self.width, self.height


def parse_raw_header(path: Path | str) -> RawHeader:
    """Read ASCII header until '% end'; return geometry and byte offset of payload."""
    path = Path(path)
    metadata: dict[str, str] = {}
    header_bytes = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"RAW header incomplete (no % end): {path}")
            header_bytes += len(line)
            if not line.startswith(b"%"):
                raise ValueError(f"Expected '%' header line in {path}, got {line[:40]!r}")
            text = line[1:].decode("ascii", errors="replace").strip()
            if text.lower() == "end":
                break
            if " " not in text:
                metadata[text] = ""
                continue
            key, value = text.split(" ", 1)
            metadata[key] = value.strip()

    width = height = None
    fmt = metadata.get("format", "")
    # format EVT3;height=720;width=1280
    for part in fmt.split(";"):
        part = part.strip()
        if part.lower().startswith("height="):
            height = int(part.split("=", 1)[1])
        elif part.lower().startswith("width="):
            width = int(part.split("=", 1)[1])
        elif part and "=" not in part:
            metadata.setdefault("format_name", part)

    if "geometry" in metadata and ("x" in metadata["geometry"].lower()):
        w_s, h_s = metadata["geometry"].lower().split("x", 1)
        width = width or int(w_s)
        height = height or int(h_s)

    if width is None or height is None:
        raise ValueError(f"Could not parse geometry from RAW header: {path}")

    format_name = metadata.get("format_name") or fmt.split(";", 1)[0].strip() or "EVT3"
    return RawHeader(
        width=width,
        height=height,
        format_name=format_name,
        evt_version=metadata.get("evt", ""),
        metadata=metadata,
        header_bytes=header_bytes,
    )
