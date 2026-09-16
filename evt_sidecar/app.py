"""CLI / GUI entry for EVT Sidecar."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EVT Sidecar: EVT3 → frame stacks for BLITZ / DONNER (WOLKE contract)",
    )
    parser.add_argument(
        "raw",
        nargs="?",
        type=Path,
        help="Optional path to an EVT3 .raw recording",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Listen address")
    parser.add_argument("--port", type=int, default=5055, help="Listen port")
    parser.add_argument("--token", default="evt", help="Stream token (BLITZ / DONNER)")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    parser.add_argument(
        "--export-npy",
        type=Path,
        metavar="OUT.npy",
        help="Headless: bin once and write .npy (no GUI / no server)",
    )
    parser.add_argument(
        "--dt-ms",
        type=float,
        default=1.0,
        help="Bin width in milliseconds (with --export-npy)",
    )
    parser.add_argument(
        "--polarity",
        choices=["on", "off", "both", "signed", "color"],
        default="color",
        help="Headless only: polarity for counts/occupancy export "
        "(on / off / both / signed / color). Ignored for --representation "
        "states. The GUI always sends activity / any-fire.",
    )
    parser.add_argument(
        "--representation",
        choices=["states", "counts", "occupancy"],
        default="states",
        help="BLITZ cube: uint8 states 0/85/170/255 (default), uint16 counts, or uint8 binary occupancy.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional hard picture cap (0 = none; RAM warnings are GUI-only)",
    )
    parser.add_argument(
        "--spatial-bin",
        type=int,
        default=1,
        choices=(1, 2, 4, 8),
        help="Pool that many sensor pixels into one (with --export-npy)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.export_npy is not None:
        return _export_headless(args)

    from PyQt6.QtWidgets import QApplication

    from .gui import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow(
        host=args.host,
        port=args.port,
        token=args.token,
        initial_raw=args.raw,
    )
    win.show()
    return app.exec()


def _export_headless(args: argparse.Namespace) -> int:
    import numpy as np

    from .binning import (
        BinParams,
        PolarityMode,
        Representation,
        bin_events,
        stack_for_send,
    )
    from .evt3 import load_evt3_raw

    if args.raw is None:
        print("error: RAW path required with --export-npy", file=sys.stderr)
        return 2
    store = load_evt3_raw(args.raw)
    cap = args.max_frames if args.max_frames > 0 else None
    params = BinParams(
        dt_us=max(1, int(round(args.dt_ms * 1000.0))),
        max_frames=cap,
        spatial_bin=int(args.spatial_bin),
    )
    planes = bin_events(store, params)
    stack = stack_for_send(
        planes,
        PolarityMode(args.polarity),
        Representation(args.representation),
    )
    out: Path = args.export_npy
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, stack)
    print(f"wrote {out} shape={stack.shape} dtype={stack.dtype} events={len(store)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
