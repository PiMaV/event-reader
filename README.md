# Event reader

Open **event-camera recordings** (Prophesee/IDS **EVT3** `.raw`) and view them
in [BLITZ](https://github.com/PiMaV/BLITZ) as a normal image stack.

Work in order. Nothing is sent to BLITZ until you click **Build pictures and
send to BLITZ**.

1. **Open** a `.raw` file. The reader builds a coarse **overview** (~150 pictures
   of the whole recording, local only).
2. **Scrub** on the timeline under the image (click, drag the white playhead, or
   mouse wheel). Time is **seconds from the first event in this file**, not
   wall-clock — EVT3 ticks are 1 µs, but the file does not store when you pressed
   record.
3. Drag the **yellow band** to set start → end of the range you will send.
4. Set **frame time (Δt)** — how long each picture for BLITZ integrates. Sensor
   resolution is 1 µs (0.001 ms). There is **no hard picture count**. The planned
   uint8 stack is compared to this PC's RAM: **yellow ≥ 1/8** (send anyway),
   **red ≥ 1/4** (confirm). Independently, **more than 1000 pictures** is allowed
   but not comfortable in BLITZ (timeline too fine) — the panel turns yellow
   (`MANY PICTURES`) and send asks once. The RAM panel fills **yellow or red**
   so the size is obvious; the send button uses the same colour. Send is refused
   only if the pictures would not fit in free RAM. Building uses ~4× RAM
   (float32) before the uint8 send.
5. Connect BLITZ → Network (`http://127.0.0.1:5055`, token `evt`), then send.
   Green status = BLITZ downloaded the stack. **Gzip** is a checkbox, default
   off (only useful on a weak network). Pictures are **raw event counts**
   (clipped at 255); optional **Log stretch** squeezes outliers into 0…255.

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

