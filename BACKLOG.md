# Event reader backlog

Not scheduled. Already in the app: yellow-band restag at the **same Δt as
send**, suggested Δt you can override, even/odd warn, optional noise
filters with **before/after** in the GUI (neighbour window = frame Δt),
color polarity from the RAW ON/OFF bit, crop from window max.

## Parked (suite Viewer Contract)

Playhead ↔ BLITZ `viewer_index` is in. Still later (see
[`../WETTER/TODO.MD`](../WETTER/TODO.MD)):

- DONNER Streamer UI (local) for 3D dual-scrub on this hub
- Open-in / ROI handoffs via the contract
- WOLKE multi-row selection parity with DONNER

## Open

### Interlace / even–odd rows

Confirmed on real recordings, including when Δt is ≥ 1 ms. Looks like sensor
readout or vendor packing, not EVT3 decode and not a BLITZ LUT. The reader
**reports** even/odd imbalance on preview rebuild and on send. Optional
merge/interpolate only after a **known sample**.
