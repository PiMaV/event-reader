# Event reader

Open **event-camera recordings** (Prophesee/IDS **EVT3** `.raw`) and view them
in [BLITZ](https://github.com/PiMaV/BLITZ) as a normal image stack.

Work in order. Nothing is sent to BLITZ until you click **Build pictures and
send to BLITZ**.

1. **Open** a `.raw` file. The reader builds a coarse **overview** (~150 pictures
   of the whole recording) so you can click through time.
2. Set **start / end** (overview pictures or the yellow band). This is only the
   time range you care about.
3. Set **frame time (Δt)** — how long each picture for BLITZ integrates. A
   **minimum Δt** is shown so you stay within the picture cap.
4. Connect BLITZ → Network (`http://127.0.0.1:5055`, token `evt`), then send.
   Green status = BLITZ downloaded the stack.

## Run

**Release binary:** `EventReader.exe` / `EventReader` from GitHub Releases.

```bash
uv sync
uv run evt-sidecar path/to/recording.raw
```

Headless export:

```bash
uv run evt-sidecar recording.raw --export-npy out.npy --dt-ms 1 --polarity both
```

License: GPL-3.0-or-later.
