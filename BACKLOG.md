# Event reader backlog

Not scheduled. Already in the app: yellow-band restag at the **same Δt as
send** (no 1 ms preview floor), suggested Δt for ~150 pictures that you can
override, even/odd warn on the local preview and on send, optional noise
filters on **preview and send**, crop from window max.

## Open

### Interlace / even–odd rows

Confirmed on real recordings, including when Δt is ≥ 1 ms. Looks like sensor
readout or vendor packing, not EVT3 decode and not a BLITZ LUT. The reader
**reports** even/odd imbalance on preview rebuild and on send. Optional
merge/interpolate only after a **known sample**.
