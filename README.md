# Event reader

Open **event-camera recordings** (Prophesee/IDS **EVT3** `.raw`) and view them
in [BLITZ](https://github.com/PiMaV/BLITZ) as a normal image stack.

No Metavision / MDK install. You choose the time bin (Δt), polarity, and window;
the reader builds frames and sends them to BLITZ.

## Run

**Release binary:** download `EventReader.exe` (Windows) or `EventReader` (Linux)
from GitHub Releases.

**From source:**

```bash
uv sync
uv run evt-sidecar path/to/recording.raw
```

Optional: `--host`, `--port`, `--token` (defaults `127.0.0.1`, `5055`, `evt`).

## In BLITZ

1. Start the event reader and load a `.raw` file (wait until it shows event count).
2. BLITZ → **Network**: address `http://127.0.0.1:5055`, token `evt` → Connect.
3. Change Δt / polarity / window in the reader; BLITZ updates the stack.

## Controls

| Control | Meaning |
|---------|---------|
| Δt | Time per frame |
| Polarity | ON, OFF, both, or signed (ON−OFF) |
| Window | Start/end within the recording |
| Max frames | Caps how many frames are sent (keeps RAM/network bounded) |
| Live apply | Recompute while you move the controls |

Headless export without BLITZ:

```bash
uv run evt-sidecar recording.raw --export-npy out.npy --dt-ms 1 --polarity both
```

License: GPL-3.0-or-later.
