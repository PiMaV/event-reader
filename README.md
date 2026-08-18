# Event reader

Open **event-camera recordings** (Prophesee/IDS **EVT3** `.raw`) and view them
in [BLITZ](https://github.com/PiMaV/BLITZ) as a normal image stack.

Work in order. Nothing is sent to BLITZ until you click **Build pictures and
send to BLITZ**.

1. **Open** a `.raw` file. The reader shows the **RAW header and first events**
   (about ten lines) plus a coarse **overview** (~150 pictures, local only).
2. **Scrub** on the timeline under the image (click, drag the white playhead, or
   mouse wheel). Time is **seconds from the first event in this file**, not
   wall-clock — EVT3 ticks are 1 µs, but the file does not store when you pressed
   record.
3. Drag the **yellow band** to set start → end of the range you will send.
4. Set **frame time (Δt)** and File-tab-like **8-bit / Normalize / Grayscale**
   (defaults: float32 counts, no 8-bit clip). RAM yellow ≥ 1/8, red ≥ 1/4;
   more than 1000 pictures is allowed but uncomfortable in BLITZ.
5. Connect BLITZ → Network (`http://127.0.0.1:5055`, token `evt`), then send.
   BLITZ File-tab options apply on Connect too (including Floor). **Gzip** is a
   checkbox, default off. Optional **Log stretch** only with 8-bit.

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

## Later (backlog)

- **Interlace / black horizontal bars.** Some recordings look like classic
  interlaced fields (dark even/odd lines). Likely how the camera wrote the
  stream, not the EVT3 decoder. Needs a look at a known clip (field order,
  binning vs line skip) before any deinterlace filter — discuss, then maybe an
  optional post-bin step. Not in the reader yet.

