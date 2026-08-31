# Changelog

All notable changes to the Event reader are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.0] - 2026-08-31

First stable EVT3 `.raw` → BLITZ line. **Send as** is the contract: what you see is what the cube holds.

### Added

- **Send as states** (default): one `uint8` gray channel for BLITZ — 0 / 85 / 170 / 255 (nothing / OFF / ON / both). Even rungs so any colormap works; both is the top rung. Local preview is red / green / yellow.
- **Send as** also has counts (`uint16` events/pixel/Δt) and occupancy (`uint8` binary 0/255). Internal bin is always two `uint16` planes; Send as is a view (no re-bin).
- **Full recording** button (Esc) restores the cached full-file overview after a yellow-band rebuild. Busy overlay stays up while the restore runs (cached or re-bin).
- **Spatial bin** (`1×1` / `2×2` / `4×4` / `8×8`) pools sensor pixels before preview and send. Counts in a block are summed (clipped to uint16). Headless: `--spatial-bin`.
- **Save as NumPy…** writes the same cube a BLITZ send would hold (Send as, crop, Δt, filters). Headless `--export-npy` is unchanged.

### Changed

- **Send as** is the only GUI choice for what BLITZ gets. Preview matches: states = red / green / yellow, counts = Inferno, occupancy = black / white. The Polarity dropdown is gone (CLI `--polarity` remains for ON / OFF / signed).
- Color polarity RGB is **local preview only**. **States** send the four rungs as gray uint8. Counts and occupancy stay one gray channel (GUI: activity / any-fire). Occupancy preview is black / white.
- Preview default view is **contain** (the whole 1280×720 is visible, letterboxed with slate blue). Fit-to-height was cropping the sides in the tall side-by-side panes; zoom-out was blocked at that crop.
- Letterbox around the picture is slate blue so black pixels read as measured zero, not empty UI.
- Suggested Δt snaps to **1-2-5** µs (0.5 / 1 / 2 / 5 ms, …). The yellow band writes that value into the spinbox (RAM/plan only). Rebuild (O) uses it unless you typed your own. Send as shows the busy overlay before the pictures update.
- **Frame time (Δt)** no longer re-bins the preview as you type or when the yellow band fills the 1-2-5 value. Pictures catch up on **Rebuild (O)** and send. Dropped **Use suggested** — the band already writes that Δt.
- Overview pictures take the **full window width** (two panes side by side), with the timeline underneath. Frame time, filters, and send sit in a compact row below. Guides are **How this works** / **RAW header** (off by default) and tooltips, not walls of text.
- Status banner and **RAM** bar sit in **Send or save** (panel 3), not under the file path / under the filters. Panel 2 stays the compact Δt and filter row.
- **Crop (M)** sits under the timeline (after Rebuild / Full recording), not on the picture. The green rectangle stays movable; **Apply crop** slices both previews and the send. Crop-edit zoom is slightly out so handles sit in the blue margin. Dropdowns stay content-sized.
- Temporal neighbour filter no longer re-bins when you drag the yellow band or Δt. It runs when you enable it, on Rebuild (O), and on send. 1-pixel spatial still updates live.
- Opening a new `.raw` resets Δt, filters, Send as, crop, and File-tab options to first-open defaults.
- Crop, Reset crop, Rebuild, and Full recording reset the pictures to **full frame**. Window resize keeps that full frame (no leftover zoom clipping the image).
- Toggling the 1-pixel filter no longer grows the window or resets picture pan/zoom.
- Default send is **states** (`uint8` 0/85/170/255). Headless `--representation states` (default); `counts` writes activity gray unless `--polarity` picks ON / OFF / signed.
- Default send dtype for counts is `uint16`, not float32 rungs. Preview still uses discrete 0…1 rungs locally.
- RAM estimate uses 2×uint16 for the build buffer; wire is 1 byte/pixel (states / occupancy) or 2 (counts).

### Removed

- **Send last pictures again** — BLITZ already re-downloads when Stream connects if a cube is ready.

[Unreleased]: https://github.com/PiMaV/event-reader/compare/build-v1.0.0...HEAD
[1.0.0]: https://github.com/PiMaV/event-reader/compare/build-0.2.0-1...build-v1.0.0
