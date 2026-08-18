# Event reader

Open **event-camera recordings** (Prophesee/IDS **EVT3** `.raw`) and view them
in [BLITZ](https://github.com/PiMaV/BLITZ) as a normal image stack.

You choose the **time range** (yellow band) and either the **frame time** or
the **number of pictures**. The reader builds a stack and sends it to BLITZ.

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
3. Drag the yellow time band and set frame time (or picture count). A green
   status means BLITZ downloaded the stack.

## Controls

| Control | Meaning |
|---------|---------|
| Yellow band | Which part of the recording becomes pictures |
| Frame time | How long each picture integrates (e.g. 1 ms) |
| Pictures | Alternative: set how many pictures, frame time follows |
| Polarity | ON, OFF, both, or signed (ON−OFF) |
| Preview | Same stack that will be sent; scrub with the mouse |

A coloured status (decoding / making pictures / sending / **BLITZ received**)
replaces guessing from the log line.

Headless export without BLITZ:

```bash
uv run evt-sidecar recording.raw --export-npy out.npy --dt-ms 1 --polarity both
```

License: GPL-3.0-or-later.
