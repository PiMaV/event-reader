"""CLI / GUI entry for EVT Sidecar."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EVT Sidecar: EVT3 → frame stacks for BLITZ (WOLKE contract)",
    )
    parser.add_argument(
        "raw",
        nargs="?",
        type=Path,
        help="Optional path to an EVT3 .raw recording",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Listen address")
    parser.add_argument("--port", type=int, default=5055, help="Listen port")
    parser.add_argument("--token", default="evt", help="BLITZ Network token")
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
        choices=["on", "off", "both", "signed"],
        default="both",
        help="Polarity mode (with --export-npy)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=2000,
        help="Frame cap (with --export-npy)",
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

    from .binning import BinParams, PolarityMode, bin_events
    from .evt3 import load_evt3_raw

    if args.raw is None:
        print("error: RAW path required with --export-npy", file=sys.stderr)
        return 2
    store = load_evt3_raw(args.raw)
    params = BinParams(
        dt_us=max(1, int(round(args.dt_ms * 1000.0))),
        polarity=PolarityMode(args.polarity),
        max_frames=args.max_frames,
    )
    stack = bin_events(store, params)
    out: Path = args.export_npy
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, stack)
    print(f"wrote {out} shape={stack.shape} dtype={stack.dtype} events={len(store)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
