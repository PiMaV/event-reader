"""PyQt6 event-reader UI: overview → range → Δt → send to BLITZ."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .binning import (
    INTERLACE_DT_US,
    INTERLACE_RATIO_WARN,
    OVERVIEW_PICTURES,
    SENSOR_DT_US,
    AccumMode,
    BinParams,
    PolarityMode,
    Representation,
    bin_events,
    encode_stack_for_send,
    even_odd_row_ratio,
    event_rate_ms,
    plan_nice_pictures,
    plan_pictures,
    polarity_count_ceiling,
    preview_from_on_off,
    removed_fraction,
    roi_event_series,
    spatial_out_size,
    stack_for_send,
)
from .evt3 import EventStore, load_evt3_raw
from .filters import drop_isolated_pixels
from .ram import (
    FILL_COLOR,
    NAV_WARN_FRAMES,
    RED_FRAC,
    YELLOW_FRAC,
    StackBudget,
    assess_stack,
    fmt_bytes,
    read_ram,
)
from .server import DEFAULT_TOKEN, StackPublisher

log = logging.getLogger("evt_sidecar.gui")


class _LoadWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            store = load_evt3_raw(self.path)
            log.info("decoded %s: %s events in %.2fs", self.path.name, f"{len(store):,}", elapsed := time.perf_counter() - t0)
            self.finished.emit(store)
        except Exception as exc:  # noqa: BLE001
            log.exception("load failed")
            self.failed.emit(str(exc))


class _BinWorker(QObject):
    finished = pyqtSignal(object, object, float, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        store: EventStore | None,
        params: BinParams,
        kind: str,
        stack: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.store = store
        self.params = params
        self.kind = kind
        self.stack = stack
        self.result_after: np.ndarray | None = None
        self.mass_before: float | None = None
        self.mass_mid: float | None = None
        self.mass_after: float | None = None

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            if self.kind == "spatial":
                if self.stack is None:
                    raise RuntimeError("spatial filter needs a cached stack")
                src = np.asarray(self.stack)
                out = drop_isolated_pixels(src)
                self.mass_mid = float(src.sum(dtype=np.float64))
                self.mass_after = float(out.sum(dtype=np.float64))
                self.result_after = out
                elapsed = time.perf_counter() - t0
                log.info("spatial filtered shape=%s in %.2fs", out.shape, elapsed)
                self.finished.emit(src, out, elapsed, self.kind)
                return
            if self.kind in ("export", "export_npy"):
                if self.store is None:
                    raise RuntimeError("export bin needs a loaded store")
                after = bin_events(self.store, self.params)
                elapsed = time.perf_counter() - t0
                log.info(
                    "%s binned shape=%s in %.2fs",
                    self.kind,
                    after.shape,
                    elapsed,
                )
                self.result_after = after
                self.finished.emit(None, after, elapsed, self.kind)
                return
            if self.store is None:
                raise RuntimeError("overview bin needs a loaded store")
            before_p = replace(
                self.params, drop_isolated=False, neighbor_dt_us=None
            )
            before = bin_events(self.store, before_p)
            if self.params.neighbor_dt_us is not None:
                mid_p = replace(self.params, drop_isolated=False)
                mid = bin_events(self.store, mid_p)
            else:
                mid = before
            if self.params.drop_isolated:
                after = drop_isolated_pixels(mid)
            else:
                after = mid
            self.result_after = after
            self.mass_before = float(before.sum(dtype=np.float64))
            self.mass_mid = float(mid.sum(dtype=np.float64))
            self.mass_after = float(after.sum(dtype=np.float64))
            elapsed = time.perf_counter() - t0
            log.info(
                "%s binned before=%s mid=%s in %.2fs",
                self.kind,
                before.shape,
                mid.shape,
                elapsed,
            )
            self.finished.emit(before, mid, elapsed, self.kind)
        except Exception as exc:  # noqa: BLE001
            log.exception("bin failed")
            self.failed.emit(str(exc))


class _NetBridge(QObject):
    served = pyqtSignal(int)
    clients = pyqtSignal(int)


_STATUS_THEME = {
    # kind: (banner bg, text, dot, short title)
    "idle": ("#2c2c2c", "#dddddd", "#888888", "Idle"),
    "work": ("#5a3e08", "#ffe9a8", "#e6a817", "Working"),
    "wait": ("#0d3a6e", "#cfe4ff", "#3d8bfd", "Waiting"),
    "ok": ("#0d4a2a", "#c8f5d8", "#2ecc71", "Ready"),
    "err": ("#5a1212", "#ffd0d0", "#e74c3c", "Error"),
}


class _Led(QFrame):
    """Status banner (load / bin / BLITZ) in the Send panel."""

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(48)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        self.kind = "idle"
        self._cursor_overridden = False
        self._pulse_on = True

        col = QVBoxLayout(self)
        col.setContentsMargins(8, 6, 8, 6)
        col.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(12)
        self.dot = QFrame()
        self.dot.setFixedSize(22, 22)
        self.dot.setFrameShape(QFrame.Shape.NoFrame)
        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(0)
        self.headline = QLabel("Idle")
        head_font = QFont(self.headline.font())
        head_font.setBold(True)
        head_font.setPointSize(head_font.pointSize() + 2)
        self.headline.setFont(head_font)
        self.label = QLabel(
            "Drop a .raw file or Open RAW…. This bar shows decoding, send, "
            "and BLITZ Stream status."
        )
        self.label.setWordWrap(True)
        self.label.setMaximumHeight(48)
        titles.addWidget(self.headline)
        titles.addWidget(self.label)
        row.addWidget(self.dot, alignment=Qt.AlignmentFlag.AlignTop)
        row.addLayout(titles, stretch=1)
        col.addLayout(row)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setTextVisible(False)
        self.busy.setFixedHeight(8)
        self.busy.setStyleSheet(
            "QProgressBar { background:#111111; border:none; border-radius:4px; }"
            "QProgressBar::chunk { background:#ffe9a8; }"
        )
        self.busy.hide()
        col.addWidget(self.busy)

        self._pulse = QTimer(self)
        self._pulse.setInterval(450)
        self._pulse.timeout.connect(self._toggle_pulse)
        self.set_state("idle", self.label.text())

    def _toggle_pulse(self) -> None:
        self._pulse_on = not self._pulse_on
        self._paint_dot()

    def _paint_dot(self) -> None:
        _bg, _fg, dot, _title = _STATUS_THEME.get(self.kind, _STATUS_THEME["idle"])
        color = dot if self._pulse_on or self.kind not in ("work", "wait") else "#111111"
        self.dot.setStyleSheet(f"background:{color}; border-radius:11px;")

    def _set_busy_cursor(self, on: bool) -> None:
        if on and not self._cursor_overridden:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._cursor_overridden = True
        elif not on and self._cursor_overridden:
            QApplication.restoreOverrideCursor()
            self._cursor_overridden = False

    def set_state(self, kind: str, text: str) -> None:
        self.kind = kind if kind in _STATUS_THEME else "idle"
        bg, fg, _dot, title = _STATUS_THEME[self.kind]
        self.headline.setText(title)
        self.label.setText(text)
        self.setStyleSheet(
            f"QFrame {{ background:{bg}; border-radius:6px; }}"
            f"QLabel {{ color:{fg}; background:transparent; }}"
        )
        self._pulse_on = True
        self._paint_dot()
        busy = self.kind in ("work", "wait")
        self.busy.setVisible(busy)
        if busy:
            self._pulse.start()
        else:
            self._pulse.stop()
        self._set_busy_cursor(self.kind == "work")


class _RamBar(QWidget):
    """Horizontal meter: fill = stack / installed RAM, ticks at 1/8 and 1/4."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(28)
        self.setMaximumHeight(32)
        self._frac = 0.0
        self._level = "ok"

    def set_fraction(self, frac: float, level: str) -> None:
        self._frac = max(0.0, float(frac))
        self._level = level
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w = max(1, self.width())
        h = max(1, self.height())
        span = max(0.50, min(1.0, self._frac * 1.25 + 0.08))

        def px(frac: float) -> int:
            return int(round(w * min(1.0, max(0.0, frac) / span)))

        painter.fillRect(0, 0, w, h, QColor("#111111"))
        painter.fillRect(0, 0, px(YELLOW_FRAC), h, QColor("#16351f"))
        painter.fillRect(
            px(YELLOW_FRAC),
            0,
            max(0, px(RED_FRAC) - px(YELLOW_FRAC)),
            h,
            QColor("#4a3b08"),
        )
        painter.fillRect(px(RED_FRAC), 0, max(0, w - px(RED_FRAC)), h, QColor("#4a1212"))
        fill = QColor(FILL_COLOR.get(self._level, "#2ecc71"))
        painter.fillRect(0, 0, px(self._frac), h, fill)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(px(YELLOW_FRAC), 0, px(YELLOW_FRAC), h)
        painter.drawLine(px(RED_FRAC), 0, px(RED_FRAC), h)
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSize(max(8, font.pointSize() - 1))
        painter.setFont(font)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(px(YELLOW_FRAC) + 5, h - 8, "1/8")
        painter.drawText(px(RED_FRAC) + 5, h - 8, "1/4")


class _RamBanner(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(56)
        col = QVBoxLayout(self)
        col.setContentsMargins(8, 6, 8, 6)
        col.setSpacing(4)
        self.headline = QLabel("RAM")
        head_font = QFont(self.headline.font())
        head_font.setBold(True)
        self.headline.setFont(head_font)
        self.bar = _RamBar()
        self.detail = QLabel("Choose Δt to see how much memory this send needs.")
        self.detail.setWordWrap(True)
        col.addWidget(self.headline)
        col.addWidget(self.bar)
        col.addWidget(self.detail)
        self.set_idle()

    def set_idle(self) -> None:
        self._apply_colors("#2c2c2c", "#dddddd")
        self.headline.setText("RAM")
        self.bar.set_fraction(0.0, "ok")
        self.detail.setText("Choose Δt to see how much memory this send needs.")

    def set_budget(self, budget: StackBudget, ram) -> None:  # noqa: ANN001
        bg, fg, tag = budget.banner_theme()
        self._apply_colors(bg, fg)
        self.headline.setText(
            f"{tag}   ·   {budget.fraction_of_total:.0%} of {fmt_bytes(ram.total)} RAM"
        )
        self.bar.set_fraction(budget.fraction_of_total, budget.level)
        notes = {
            "ok": "Fits comfortably.",
            "yellow": "Large — BLITZ will feel it, but this should still run.",
            "red": "Huge — confirm before send. BLITZ holds another copy.",
            "block": "Will not send: pictures would not fit in free RAM.",
        }
        nav = ""
        if budget.nav_warn:
            nav = (
                f" {budget.n_frames} pictures — above {NAV_WARN_FRAMES} the BLITZ "
                "timeline is very fine-grained. Allowed, but only if you know "
                "you need it."
            )
        self.detail.setText(
            f"{fmt_bytes(budget.wire_bytes)} on the wire  ·  "
            f"{budget.n_frames} pictures  ·  "
            f"~{fmt_bytes(budget.build_bytes)} to bin (2×uint16 ON/OFF)  ·  "
            f"{fmt_bytes(ram.available)} free.  {notes[budget.level]}{nav}"
        )

    def _apply_colors(self, bg: str, fg: str) -> None:
        self.setStyleSheet(
            f"QFrame {{ background:{bg}; border-radius:6px; }}"
            f"QLabel {{ color:{fg}; background:transparent; }}"
        )


def _file_preview_text(store: EventStore, n_header: int = 10, n_events: int = 8) -> str:
    """First header lines of the RAW plus a few decoded events (scientist orientation)."""
    lines = list(store.header.source_lines[:n_header])
    if len(store.header.source_lines) > n_header:
        lines.append("…")
    n_show = min(n_events, len(store))
    if n_show:
        lines.append("")
        lines.append("# first events    t[µs]    x    y  polarity")
        for i in range(n_show):
            pol = "ON" if int(store.p[i]) else "OFF"
            lines.append(
                f"{int(store.t[i]):16d}  {int(store.x[i]):4d}  {int(store.y[i]):4d}  {pol}"
            )
    return "\n".join(lines)


# Slate blue around the picture: black pixels are a measured zero, not empty UI.
PREVIEW_VOID = (28, 52, 84)
# Extra image-heights of void while placing the crop ROI, so edge handles
# sit in the blue instead of on the viewport frame.
CROP_EDIT_MARGIN = 0.12
# Default green rectangle inset (fraction of stack size) — not full-frame.
CROP_ROI_INSET = 0.08

OVERVIEW_HELP = (
    "Wheel or the white playhead scrubs pictures (like BLITZ). "
    "Ctrl+wheel or right-drag zooms time. Yellow band = the range you send "
    "(it also writes a 1-2-5 Δt). Typing Δt updates the RAM plan only. "
    "O rebuilds that band at the Δt in panel 2 (same Δt BLITZ gets). "
    "Esc or double-click the plot restores the coarse full-file overview. "
    "Left = unfiltered, right = noise filters. Pan/zoom stay locked. "
    "Yellow box on the pictures is the time-series probe (plot = event count "
    "in that box, sum not mean). Double-click a picture to fit the full frame. "
    "Send as is the one choice: states (red / green / yellow), "
    "counts (Inferno), occupancy (black / white). "
    "The cube is always one gray channel; those colours are the legend here. "
    "Crop (M) comes after the yellow band: activity image, then move the "
    "green rectangle (zoom is slightly out so the handles sit in the blue). "
    "Apply crop when it fits — mouse-up does not lock the crop. "
    "Spatial bin (2×2 / 4×4 / 8×8) pools sensor pixels when you do not "
    "need the full resolution. Panel 3 sends that cube to BLITZ or "
    "saves it as NumPy (same bytes)."
)


def preview_contain_ranges(
    width: float,
    height: float,
    view_w: float,
    view_h: float,
    margin: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """View ranges that show the **entire** picture (letterbox with void).

    ``margin`` grows the box uniformly so crop-ROI handles sit in the blue.
    Zoom-out stops here; the image is never cropped by the default view.
    """
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    vw = max(1.0, float(view_w))
    vh = max(1.0, float(view_h))
    m = max(0.0, float(margin))
    box_w = w * (1.0 + 2.0 * m)
    box_h = h * (1.0 + 2.0 * m)
    view_aspect = vw / vh
    box_aspect = box_w / box_h
    if view_aspect >= box_aspect:
        y0 = (h - box_h) / 2.0
        x_span = box_h * view_aspect
        x0 = (w - x_span) / 2.0
        return (x0, x0 + x_span), (y0, y0 + box_h)
    x0 = (w - box_w) / 2.0
    y_span = box_w / view_aspect
    y0 = (h - y_span) / 2.0
    return (x0, x0 + box_w), (y0, y0 + y_span)


def preview_fit_height_ranges(
    width: float,
    height: float,
    view_w: float,
    view_h: float,
    margin: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Alias: default view is contain (whole frame), not height-crop."""
    return preview_contain_ranges(width, height, view_w, view_h, margin)


def raw_paths_from_dropped(paths: list[str | Path]) -> list[Path]:
    """Turn dropped files/folders into EVT3 ``.raw`` paths (one directory level)."""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            candidates = sorted(p.glob("*.raw"))
        elif p.is_file() and p.suffix.lower() == ".raw":
            candidates = [p]
        else:
            continue
        for cand in candidates:
            key = cand.resolve()
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return out


class MainWindow(QMainWindow):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5055,
        token: str = DEFAULT_TOKEN,
        initial_raw: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Event reader → BLITZ")
        self.resize(720, 940)

        self._store: EventStore | None = None
        self._overview: np.ndarray | None = None
        self._overview_before: np.ndarray | None = None
        self._overview_full: np.ndarray | None = None
        self._overview_full_before: np.ndarray | None = None
        self._counts_before: np.ndarray | None = None
        self._counts_mid: np.ndarray | None = None
        self._counts_after: np.ndarray | None = None
        self._counts_before_full: np.ndarray | None = None
        self._counts_mid_full: np.ndarray | None = None
        self._counts_after_full: np.ndarray | None = None
        self._activity_before: np.ndarray | None = None
        self._activity_mid: np.ndarray | None = None
        self._activity_after: np.ndarray | None = None
        self._mass_before: float | None = None
        self._mass_mid: float | None = None
        self._mass_after: float | None = None
        self._spatial_src: np.ndarray | None = None
        self._overview_dt_us = 1
        self._overview_full_dt_us = 1
        self._overview_t0_us = 0
        self._overview_full_t0_us = 0
        self._overview_is_detail = False
        self._overview_applied_sig: tuple | None = None
        self._overview_full_sig: tuple | None = None
        self._showing_activity = False
        self._crop_roi: pg.RectROI | None = None
        self._crop_box: tuple[int, int, int, int] | None = None
        self._crop_editing = False
        self._syncing_crop = False
        self._load_thread: QThread | None = None
        self._load_worker: _LoadWorker | None = None
        self._bin_thread: QThread | None = None
        self._bin_worker: _BinWorker | None = None
        self._bin_busy_kind: str | None = None
        self._bin_inflight_params: BinParams | None = None
        self._pending_bin: tuple[BinParams, str] | None = None
        self._blitz_clients = 0
        self._syncing_range = False
        self._syncing_view = False
        self._syncing_probe = False
        self._saved_view_range: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._event_scale_hi = 1.0
        self._probe_roi_before: pg.ROI | None = None
        self._probe_roi_after: pg.ROI | None = None
        self._probe_spatial: tuple[int, int] | None = None
        self._preview_wh: tuple[int, int] | None = None
        self._preview_user_zoom = False
        self._force_fit_preview = False
        self._dt_user_set = False
        self._npy_out: Path | None = None
        self._playhead_s = 0.0
        self._stack_budget: StackBudget | None = None
        self._rate_zoom_timer = QTimer(self)
        self._rate_zoom_timer.setSingleShot(True)
        self._rate_zoom_timer.setInterval(120)
        self._rate_zoom_timer.timeout.connect(self._refresh_rate_curve_from_view)
        self._view_timer = QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(0)
        self._view_timer.timeout.connect(self._run_preview_view_update)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(0)
        self._fit_timer.timeout.connect(self._fit_preview_full_frame)
        self._roi_series_timer = QTimer(self)
        self._roi_series_timer.setSingleShot(True)
        self._roi_series_timer.setInterval(50)
        self._roi_series_timer.timeout.connect(self._refresh_roi_series)

        self._bridge = _NetBridge(self)
        self._bridge.served.connect(self._on_blitz_downloaded)
        self._bridge.clients.connect(self._on_client_count)

        self.publisher = StackPublisher(host=host, port=port, token=token)
        self.publisher.on_served = lambda n: self._bridge.served.emit(n)
        self.publisher.on_client_count = lambda n: self._bridge.clients.emit(n)
        self.publisher.start_background()

        self._build_ui()
        self._install_shortcuts()
        self.setAcceptDrops(True)
        self._update_connect_hint()
        if initial_raw is not None:
            QTimer.singleShot(0, lambda: self._load_path(Path(initial_raw)))

    def _set_status(self, kind: str, text: str) -> None:
        self.led.set_state(kind, text)
        self.statusBar().showMessage(text)
        if kind == "work" and getattr(self, "_busy_before", None) is not None:
            self._set_preview_busy(True, text)

    def _make_preview_pane(
        self,
        title: str,
        extras: list[QWidget] | None = None,
    ) -> tuple[pg.ImageView, QWidget, QLabel]:
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        lab = QLabel(title)
        lab.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        title_row.addWidget(lab, stretch=1)
        for extra in extras or []:
            title_row.addWidget(extra)
        wrap = QWidget()
        grid = QGridLayout(wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        view = pg.ImageView()
        view.setAcceptDrops(False)
        view.ui.roiBtn.hide()
        view.ui.menuBtn.hide()
        view.ui.histogram.hide()
        view.ui.roiPlot.hide()
        view.ui.histogram.setMaximumWidth(0)
        view.ui.roiPlot.setMaximumHeight(0)
        view.ui.splitter.setChildrenCollapsible(True)
        view.ui.splitter.setStretchFactor(0, 1)
        if view.ui.splitter.count() > 1:
            view.ui.splitter.setStretchFactor(1, 0)
            view.ui.splitter.setCollapsible(1, True)
        view.ui.splitter.setHandleWidth(0)
        view.ui.splitter.setSizes([10_000, 0])
        view.setMinimumHeight(240)
        view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        vb = view.getView()
        vb.setAspectLocked(True)
        vb.setDefaultPadding(0.0)
        vb.enableAutoRange(enable=False)
        void = pg.mkColor(*PREVIEW_VOID)
        view.ui.graphicsView.setBackground(void)
        vb.setBackgroundColor(void)
        busy = QLabel("")
        busy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        busy.setWordWrap(True)
        busy.setStyleSheet(
            "QLabel { background: rgba(18, 14, 6, 210); color: #ffe9a8; "
            "font-weight: bold; font-size: 14px; padding: 16px; }"
        )
        busy.hide()
        busy.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        grid.addWidget(view, 0, 0)
        grid.addWidget(busy, 0, 0)
        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)
        busy.raise_()
        col.addLayout(title_row)
        col.addWidget(wrap, stretch=1)
        return view, box, busy

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        file_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Drop an EVT3 .raw here, or Open RAW…")
        self.path_edit.setReadOnly(True)
        self.path_edit.setAcceptDrops(False)
        browse = QPushButton("Open RAW…")
        browse.clicked.connect(self._browse)
        file_row.addWidget(self.path_edit, stretch=1)
        file_row.addWidget(browse)
        layout.addLayout(file_row)

        meta_row = QHBoxLayout()
        self.meta_label = QLabel("No file loaded.")
        self.meta_label.setWordWrap(True)
        meta_row.addWidget(self.meta_label, stretch=1)
        self.header_btn = QToolButton()
        self.header_btn.setText("RAW header")
        self.header_btn.setCheckable(True)
        self.header_btn.setToolTip(
            "First RAW header lines and a few decoded events. Hidden by "
            "default so the pictures keep the space."
        )
        self.help_btn = QToolButton()
        self.help_btn.setText("How this works")
        self.help_btn.setCheckable(True)
        self.help_btn.setToolTip(
            "Short guide. Hover any control for the same detail in a tooltip."
        )
        meta_row.addWidget(self.header_btn)
        meta_row.addWidget(self.help_btn)
        layout.addLayout(meta_row)

        self.file_preview = QPlainTextEdit()
        self.file_preview.setReadOnly(True)
        self.file_preview.setMaximumHeight(110)
        self.file_preview.setPlaceholderText("RAW header and first events appear here.")
        self.file_preview.setFont(QFont("monospace", 9))
        self.file_preview.setAcceptDrops(False)
        self.file_preview.hide()
        self.header_btn.toggled.connect(self.file_preview.setVisible)
        layout.addWidget(self.file_preview)

        self.help_text = QLabel(OVERVIEW_HELP)
        self.help_text.setWordWrap(True)
        self.help_text.hide()
        self.help_btn.toggled.connect(self.help_text.setVisible)
        layout.addWidget(self.help_text)

        step1 = QGroupBox("1 — Overview")
        s1 = QVBoxLayout(step1)
        s1.setContentsMargins(8, 8, 8, 8)
        s1.setSpacing(4)

        self.activity_btn = QPushButton("Crop (M)")
        self.activity_btn.setEnabled(False)
        self.activity_btn.setToolTip(
            "After the yellow band is right: one activity picture of that "
            "window, then move the green rectangle. The view zooms out a "
            "little so the handles sit in the blue. Mouse-up does not lock "
            "the crop — press Apply crop when it fits."
        )
        self.activity_btn.clicked.connect(self._show_activity_crop)
        self.apply_crop_btn = QPushButton("Apply crop")
        self.apply_crop_btn.setEnabled(False)
        self.apply_crop_btn.setToolTip(
            "Slice both previews and the send to the green rectangle. "
            "Drag and drop the rectangle as often as you need first."
        )
        self.apply_crop_btn.clicked.connect(self._apply_crop)
        self.reset_crop_btn = QPushButton("Reset crop")
        self.reset_crop_btn.setEnabled(False)
        self.reset_crop_btn.setToolTip("Use the full sensor again.")
        self.reset_crop_btn.clicked.connect(self._reset_crop)

        pair = QWidget()
        pair.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        pair_row = QHBoxLayout(pair)
        pair_row.setContentsMargins(0, 0, 0, 0)
        pair_row.setSpacing(8)
        self.preview_before, before_box, self._busy_before = self._make_preview_pane(
            "Before — unfiltered"
        )
        self.preview, after_box, self._busy_after = self._make_preview_pane(
            "After — noise filters",
        )
        pair_row.addWidget(before_box, stretch=1)
        pair_row.addWidget(after_box, stretch=1)
        s1.addWidget(pair, stretch=1)

        self.color_legend = QLabel(
            "States: red = OFF · green = ON · yellow = both. Hover for more."
        )
        self.color_legend.setWordWrap(False)
        self.filter_stats = QLabel("Filters off.")
        self.filter_stats.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(0, 0, 0, 0)
        legend_row.addWidget(self.color_legend, stretch=1)
        legend_row.addWidget(self.filter_stats)
        s1.addLayout(legend_row)

        self.playhead_label = QLabel("t = —")
        self.timeline_legend = QLabel("")
        self.timeline_legend.setTextFormat(Qt.TextFormat.RichText)
        self.timeline_legend.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.timeline_legend.setToolTip(
            "Cyan = events in the yellow box (unfiltered). "
            "Gold = the same box after noise filters."
        )
        head_row = QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.addWidget(self.playhead_label, stretch=1)
        head_row.addWidget(self.timeline_legend)
        s1.addLayout(head_row)

        self.time_plot = pg.PlotWidget()
        self.time_plot.setAcceptDrops(False)
        self.time_plot.setMinimumHeight(100)
        self.time_plot.setMaximumHeight(140)
        self.time_plot.setLabel("bottom", "Time", units="s")
        self.time_plot.setLabel("left", "Events")
        self.time_plot.showGrid(x=True, y=True, alpha=0.2)
        self.time_plot.setMouseEnabled(x=True, y=False)
        vb = self.time_plot.getViewBox()
        vb.setMouseEnabled(x=True, y=False)
        vb.enableAutoRange(axis=vb.YAxis)
        vb.disableAutoRange(axis=vb.XAxis)
        vb.sigXRangeChanged.connect(self._on_xrange_changed)
        self.rate_curve = self.time_plot.plot(pen=pg.mkPen("#3d5a66", width=1))
        self.roi_before_curve = self.time_plot.plot(pen=pg.mkPen("#7ec8e3", width=2))
        self.roi_after_curve = self.time_plot.plot(pen=pg.mkPen("#ffcc33", width=2))
        self.region = pg.LinearRegionItem(
            [0.0, 1.0],
            brush=(230, 180, 40, 55),
            pen=pg.mkPen("#ffcc33", width=3),
            hoverBrush=(230, 180, 40, 95),
            hoverPen=pg.mkPen("#ffffff", width=4),
        )
        self.region.setZValue(5)
        self.time_plot.addItem(self.region)
        self.playhead = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True, pen=pg.mkPen("#ffffff", width=2)
        )
        self.playhead.setZValue(10)
        self.time_plot.addItem(self.playhead)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.region.sigRegionChangeFinished.connect(self._on_region_change_finished)
        self.playhead.sigPositionChanged.connect(self._on_playhead_moved)
        self.time_plot.scene().sigMouseClicked.connect(self._on_timeline_clicked)
        self.time_plot.installEventFilter(self)
        self.time_plot.viewport().installEventFilter(self)
        self.preview.sigTimeChanged.connect(self._on_preview_index)
        self.preview_before.sigTimeChanged.connect(self._on_preview_index)
        self._wire_preview_views()
        s1.addWidget(self.time_plot)
        ov_row = QHBoxLayout()
        self.restag_btn = QPushButton("Rebuild overview for selection (O)")
        self.restag_btn.setEnabled(False)
        self.restag_btn.setToolTip(
            "Zoom to the yellow band and re-bin at the Δt in panel 2 — the "
            "same pictures you will send to BLITZ. Moving the yellow band "
            "writes a 1-2-5 Δt (~150 pictures) into the spinbox; typing Δt "
            "updates the RAM plan only until you Rebuild or send. Full "
            "recording restores the coarse full-file overview."
        )
        self.restag_btn.clicked.connect(self._restag_overview_for_view)
        self.full_rec_btn = QPushButton("Full recording (Esc)")
        self.full_rec_btn.setEnabled(False)
        self.full_rec_btn.setToolTip(
            "Un-zoom the timeline and restore the cached full-file overview "
            "(same as double-clicking the plot). Does not reload the .raw."
        )
        self.full_rec_btn.clicked.connect(self._reset_time_zoom)
        ov_row.addWidget(self.restag_btn)
        ov_row.addWidget(self.full_rec_btn)
        ov_row.addStretch(1)
        ov_row.addWidget(self.activity_btn)
        ov_row.addWidget(self.apply_crop_btn)
        ov_row.addWidget(self.reset_crop_btn)
        s1.addLayout(ov_row)

        step2 = QGroupBox("2 — Frame time & filters")
        s2 = QVBoxLayout(step2)
        s2.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )
        self.dt_ms = QDoubleSpinBox()
        self.dt_ms.setRange(SENSOR_DT_US / 1000.0, 1_000_000.0)
        self.dt_ms.setDecimals(3)
        self.dt_ms.setSingleStep(0.01)
        self.dt_ms.setValue(1.0)
        self.dt_ms.setSuffix(" ms")
        self.dt_ms.setToolTip(
            "How long each picture integrates. Typing here updates the RAM "
            "plan only — Rebuild (O) and send re-bin at this Δt. Moving the "
            "yellow band writes a 1-2-5 value (~150 pictures). Sensor "
            "timestamps step by 1 µs (0.001 ms). Picture count is limited "
            "only by RAM (yellow/red bar in Send to BLITZ)."
        )
        self.dt_ms.valueChanged.connect(self._on_dt_changed)
        form.addRow("Frame time (Δt)", self.dt_ms)
        self.spatial_bin = QComboBox()
        for k, label in (
            (1, "1×1 (full)"),
            (2, "2×2"),
            (4, "4×4"),
            (8, "8×8"),
        ):
            self.spatial_bin.addItem(label, k)
        self.spatial_bin.setCurrentIndex(0)
        self.spatial_bin.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.spatial_bin.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        self.spatial_bin.setToolTip(
            "Pool that many sensor pixels into one before preview and send. "
            "Use this when you do not need the full 1280×720. Counts in a "
            "block are summed. 1×1 is the native resolution."
        )
        self.spatial_bin.currentIndexChanged.connect(self._on_spatial_bin_changed)
        form.addRow("Spatial bin", self.spatial_bin)
        self.min_dt_label = QLabel(
            "Timestamps at 1 µs from the first event in this file."
        )
        self.min_dt_label.setWordWrap(True)
        form.addRow(self.min_dt_label)
        self.representation = QComboBox()
        for mode, label in (
            (Representation.STATES, "states (ON / OFF / both)"),
            (Representation.COUNTS, "counts (events / pixel / Δt)"),
            (Representation.OCCUPANCY, "occupancy (fired / not)"),
        ):
            self.representation.addItem(label, mode)
        self.representation.setCurrentIndex(0)
        self.representation.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.representation.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        self.representation.setToolTip(
            "What BLITZ holds — the pictures here use a matching legend. "
            "States (default): polarity as red / green / yellow; cube is "
            "uint8 0 / 85 / 170 / 255 (nothing / OFF / ON / both). "
            "Counts: Inferno for how many events; cube is uint16 activity "
            "(ON+OFF). Occupancy: black / white fired-or-not; cube is "
            "uint8 0 or 255."
        )
        self.representation.currentIndexChanged.connect(self._on_send_as_changed)
        form.addRow("Send as", self.representation)
        self.eight_bit = QCheckBox("8-bit")
        self.eight_bit.setChecked(False)
        self.eight_bit.setToolTip(
            "Off (default): keep the send dtype (uint8 states, "
            "uint16 counts, uint8 occupancy). "
            "On: clip counts to uint8 here (same idea as the BLITZ File tab)."
        )
        self.eight_bit.toggled.connect(self._on_encode_options_changed)
        self.normalize_box = QCheckBox("Normalize")
        self.normalize_box.setChecked(False)
        self.normalize_box.setToolTip(
            "Per-picture min–max stretch. Same idea as the BLITZ File tab. "
            "BLITZ can still apply File-tab options again on Connect."
        )
        self.normalize_box.toggled.connect(self._on_encode_options_changed)
        self.grayscale_box = QCheckBox("Grayscale")
        self.grayscale_box.setChecked(True)
        self.grayscale_box.setEnabled(False)
        self.grayscale_box.setToolTip(
            "Always on: the BLITZ cube is one channel. "
            "Colour here is the Send-as legend (not the cube)."
        )
        encode_wrap = QWidget()
        encode_row = QHBoxLayout(encode_wrap)
        encode_row.setContentsMargins(0, 0, 0, 0)
        encode_row.addWidget(self.eight_bit)
        encode_row.addWidget(self.normalize_box)
        encode_row.addWidget(self.grayscale_box)
        encode_row.addStretch(1)
        form.addRow("Send like File tab", encode_wrap)
        self.log_stretch = QCheckBox("Log stretch (log1p → 0…255)")
        self.log_stretch.setChecked(False)
        self.log_stretch.setEnabled(False)
        self.log_stretch.setToolTip(
            "Only with 8-bit. Squeeze outliers into 0…255 so a few hot pixels "
            "do not crush typical counts."
        )
        form.addRow(self.log_stretch)
        self.log_stretch.toggled.connect(self._on_encode_options_changed)
        self._sync_send_widgets()
        noise_wrap = QWidget()
        noise_col = QHBoxLayout(noise_wrap)
        noise_col.setContentsMargins(0, 0, 0, 0)
        noise_col.setSpacing(8)
        self.drop_isolated_box = QCheckBox("1-pixel spatial")
        self.drop_isolated_box.setChecked(False)
        self.drop_isolated_box.setToolTip(
            "After binning: zero pixels that have a count but all 8 neighbours "
            "are empty. Optional — a real 1-pixel event is removed too. "
            "Runs in the background (busy overlay on both pictures); "
            "send uses the same setting."
        )
        self.drop_isolated_box.toggled.connect(self._on_filter_checkbox)
        self.neighbor_box = QCheckBox("Temporal neighbour")
        self.neighbor_box.setChecked(False)
        self.neighbor_box.setToolTip(
            "Before binning: keep an event only if a pixel in its 3×3 "
            "neighbourhood already fired within the frame Δt above. "
            "Isolated salt-and-pepper events drop. This is expensive — "
            "it runs when you tick the box, on Rebuild (O), and on send. "
            "Moving the yellow band or Δt does not re-bin the preview."
        )
        self.neighbor_box.toggled.connect(self._on_neighbor_toggled)
        noise_col.addWidget(self.drop_isolated_box)
        noise_col.addWidget(self.neighbor_box)
        noise_col.addStretch(1)
        form.addRow("Noise filter", noise_wrap)
        s2.addLayout(form)

        step3 = QGroupBox("3 — Send or save")
        s3 = QVBoxLayout(step3)
        s3.setContentsMargins(8, 8, 8, 8)
        self.apply_btn = QPushButton("Build pictures and send to BLITZ")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setToolTip(
            "BLITZ does not update while you scrub. Connect Stream first, "
            "then send. Green status = BLITZ downloaded the stack."
        )
        self.apply_btn.clicked.connect(self._export_to_blitz)
        self.save_npy_btn = QPushButton("Save as NumPy…")
        self.save_npy_btn.setEnabled(False)
        self.save_npy_btn.setToolTip(
            "Write the same cube a BLITZ send would hold (Send as, crop, "
            "Δt, filters, 8-bit / Normalize). Does not need Stream."
        )
        self.save_npy_btn.clicked.connect(self._save_as_npy)
        s3.addWidget(self.apply_btn)
        s3.addWidget(self.save_npy_btn)
        self.gzip_box = QCheckBox("Gzip on the wire")
        self.gzip_box.setChecked(False)
        self.gzip_box.setToolTip(
            "Only useful over a weak network. Leave off on this computer — "
            "zip then unzip is wasted CPU."
        )
        self.gzip_box.toggled.connect(self._on_gzip_toggled)
        s3.addWidget(self.gzip_box)
        self.connect_hint = QLabel()
        self.connect_hint.setWordWrap(True)
        s3.addWidget(self.connect_hint)
        net_row = QHBoxLayout()
        self.host_edit = QLineEdit(self.publisher.host)
        self.port_edit = QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(self.publisher.port)
        self.token_edit = QLineEdit(self.publisher.token)
        net_row.addWidget(QLabel("Host"))
        net_row.addWidget(self.host_edit)
        net_row.addWidget(QLabel("Port"))
        net_row.addWidget(self.port_edit)
        net_row.addWidget(QLabel("Token"))
        net_row.addWidget(self.token_edit)
        s3.addLayout(net_row)
        s3.addStretch(1)
        self.led = _Led()
        s3.addWidget(self.led)
        self.ram_banner = _RamBanner()
        self.ram_banner.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        s3.addWidget(self.ram_banner)

        bottom = QWidget()
        bottom.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        bottom_row = QHBoxLayout(bottom)
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(8)
        bottom_row.addWidget(step2, stretch=1)
        bottom_row.addWidget(step3, stretch=1)

        layout.addWidget(step1, stretch=1)
        layout.addWidget(bottom)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"Serving {self.publisher.base_url}")

    def _update_connect_hint(self) -> None:
        self.connect_hint.setText(
            f"BLITZ Stream: <b>{self.publisher.base_url}</b>  ·  token "
            f"<b>{self.publisher.token}</b>"
        )
        self.connect_hint.setToolTip(
            "In BLITZ → Stream: Connect with that address and token, then send."
        )

    def _on_encode_options_changed(self) -> None:
        eight = self.eight_bit.isChecked()
        self.log_stretch.setEnabled(eight)
        if not eight:
            self.log_stretch.setChecked(False)
        self._refresh_plan_label()
        if self._counts_mid is not None and not self._showing_activity:
            self._refresh_filter_views()

    def _on_gzip_toggled(self, checked: bool) -> None:
        self.publisher.gzip_enabled = bool(checked)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open EVT3 RAW", "", "RAW recordings (*.raw);;All files (*)"
        )
        if path:
            self._load_path(Path(path))

    def _dropped_local_paths(
        self, event: QDragEnterEvent | QDragMoveEvent | QDropEvent
    ) -> list[Path]:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return []
        locals_: list[str] = []
        for url in mime.urls():
            if url.isLocalFile():
                locals_.append(url.toLocalFile())
        return raw_paths_from_dropped(locals_)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._dropped_local_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        if self._dropped_local_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = self._dropped_local_paths(event)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self._load_path(paths[0])

    def _load_path(self, path: Path) -> None:
        if self._load_thread is not None and self._load_thread.isRunning():
            return
        self._reset_session_controls()
        self.path_edit.setText(str(path))
        self.meta_label.setText("Decoding…")
        self._set_status("work", "1/3 Decoding recording…")
        self.apply_btn.setEnabled(False)
        self.save_npy_btn.setEnabled(False)
        thread = QThread(self)
        worker = _LoadWorker(path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_loaded)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._clear_load_worker)
        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    def _reset_session_controls(self) -> None:
        """Restore Δt, filters, and send options to first-open defaults."""
        self._view_timer.stop()
        widgets = (
            self.neighbor_box,
            self.drop_isolated_box,
            self.spatial_bin,
            self.representation,
            self.eight_bit,
            self.normalize_box,
            self.log_stretch,
            self.dt_ms,
            self.gzip_box,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            self.neighbor_box.setChecked(False)
            self.drop_isolated_box.setChecked(False)
            self.spatial_bin.setCurrentIndex(0)
            self.representation.setCurrentIndex(0)
            self.eight_bit.setChecked(False)
            self.normalize_box.setChecked(False)
            self.log_stretch.setChecked(False)
            self.log_stretch.setEnabled(False)
            self.dt_ms.setValue(1.0)
            self.gzip_box.setChecked(False)
            self.publisher.gzip_enabled = False
            self._dt_user_set = False
            self._crop_box = None
            self._crop_editing = False
            self._remove_crop_roi()
            self._sync_send_widgets()
        finally:
            for w in widgets:
                w.blockSignals(False)
        self.filter_stats.setText("Filters off.")

    def _clear_load_worker(self) -> None:
        if self._load_worker is not None:
            self._load_worker.deleteLater()
            self._load_worker = None
        if self._load_thread is not None:
            self._load_thread.deleteLater()
            self._load_thread = None

    def _on_loaded(self, store: EventStore) -> None:
        self._store = store
        self._overview = None
        self._overview_before = None
        self._overview_full = None
        self._overview_full_before = None
        self._counts_before = None
        self._counts_mid = None
        self._counts_after = None
        self._counts_before_full = None
        self._counts_mid_full = None
        self._counts_after_full = None
        self._activity_before = None
        self._activity_mid = None
        self._activity_after = None
        self._mass_before = None
        self._mass_mid = None
        self._mass_after = None
        self._spatial_src = None
        self._probe_spatial = None
        self._request_full_frame()
        self._dt_user_set = False
        self._saved_view_range = None
        self.filter_stats.setText("Filters off.")
        self._overview_is_detail = False
        self._overview_applied_sig = None
        self._overview_full_sig = None
        self._showing_activity = False
        self._crop_box = None
        self._crop_editing = False
        self._remove_crop_roi()
        dur_s = max(store.duration_us / 1_000_000.0, 0.001)
        self.meta_label.setText(
            f"{store.width}×{store.height}  |  {len(store):,} events  |  "
            f"{dur_s:.3f} s (from first event in this file)"
        )
        x_ms, counts = event_rate_ms(store)
        self.rate_curve.setData(x_ms / 1000.0, counts)
        self.rate_curve.setVisible(True)
        self.roi_before_curve.setData([], [])
        self.roi_after_curve.setData([], [])
        self.time_plot.setLabel("left", "Events")
        self._set_timeline_legend("file")
        self.region.blockSignals(True)
        self.playhead.blockSignals(True)
        self.region.setBounds((0.0, dur_s))
        self.region.setRegion((0.0, dur_s))
        self.playhead.setBounds((0.0, dur_s))
        self.playhead.setValue(0.0)
        self.playhead.blockSignals(False)
        self.region.blockSignals(False)
        self._apply_time_limits(dur_s)
        self._set_time_view(0.0, dur_s)
        self.file_preview.setPlainText(_file_preview_text(store))
        self.restag_btn.setEnabled(False)
        self.full_rec_btn.setEnabled(False)
        self._sync_crop_buttons()
        self._set_status("work", f"1/3 Building ~{OVERVIEW_PICTURES}-picture overview…")
        self._start_bin(self._overview_params(), "overview")

    def _overview_params(self, *, yellow: bool = False) -> BinParams:
        assert self._store is not None
        if yellow:
            t0, t1 = self._selected_window_us()
        else:
            t0, t1 = int(self._store.t_min), int(self._store.t_max)
        window_us = max(1, t1 - t0)
        if yellow:
            dt_us = self._user_dt_us()
            dt_us, n, _u = plan_pictures(window_us, dt_us=dt_us)
        else:
            dt_us, n, _u = plan_nice_pictures(window_us, n_frames=OVERVIEW_PICTURES)
        return BinParams(
            dt_us=dt_us,
            polarity=self._gui_polarity(),
            accum=AccumMode.COUNT,
            t0_us=t0,
            t1_us=t1,
            max_frames=n,
            drop_isolated=self.drop_isolated_box.isChecked(),
            neighbor_dt_us=self._neighbor_dt_us(),
            spatial_bin=self._spatial_bin(),
        )

    def _on_load_failed(self, message: str) -> None:
        self._set_status("err", "Load failed")
        QMessageBox.critical(self, "Load failed", message)

    def _selected_window_us(self) -> tuple[int, int]:
        assert self._store is not None
        t0_s, t1_s = self.region.getRegion()
        if t1_s < t0_s:
            t0_s, t1_s = t1_s, t0_s
        t_base = int(self._store.t_min)
        t0 = t_base + int(round(t0_s * 1_000_000.0))
        t1 = t_base + int(round(t1_s * 1_000_000.0))
        return t0, max(t0 + 1, t1)

    def _visible_window_us(self) -> tuple[int, int]:
        assert self._store is not None
        (x0, x1), _yr = self.time_plot.viewRange()
        if x1 < x0:
            x0, x1 = x1, x0
        dur = self._duration_s()
        x0 = max(0.0, min(dur, x0))
        x1 = max(0.0, min(dur, x1))
        t_base = int(self._store.t_min)
        t0 = t_base + int(round(x0 * 1_000_000.0))
        t1 = t_base + int(round(x1 * 1_000_000.0))
        return t0, max(t0 + 1, t1)

    def _duration_s(self) -> float:
        if self._store is None:
            return 0.001
        return max(self._store.duration_us / 1_000_000.0, 0.001)

    def _overview_origin_s(self) -> float:
        if self._store is None:
            return 0.0
        return (int(self._overview_t0_us) - int(self._store.t_min)) / 1_000_000.0

    def _frame_at_s(self, t_s: float) -> int:
        if self._overview is None:
            return 0
        n = self._overview.shape[0]
        rel_s = t_s - self._overview_origin_s()
        idx = int(round(rel_s * 1_000_000.0 / max(1, self._overview_dt_us)))
        return max(0, min(n - 1, idx))

    def _set_playhead_s(self, t_s: float) -> None:
        dur = self._duration_s()
        t_s = max(0.0, min(dur, t_s))
        if self._showing_activity and self._overview is not None:
            self._show_overview_stack()
            busy = (
                self._bin_thread is not None and self._bin_thread.isRunning()
            )
            if (
                not busy
                and self._overview_applied_sig != self._filter_sig()
                and not self.neighbor_box.isChecked()
            ):
                self._rebuild_preview_for_filters()
        self._playhead_s = t_s
        self._syncing_range = True
        self.playhead.setValue(t_s)
        idx = 0
        if self._overview is not None and not self._showing_activity:
            idx = self._frame_at_s(t_s)
            self.preview.setCurrentIndex(idx)
            self.preview_before.setCurrentIndex(idx)
        self._syncing_range = False
        if self._overview is not None:
            n = self._overview.shape[0]
            kind = "detail" if self._overview_is_detail else "overview"
            self.playhead_label.setText(
                f"t = {t_s:.4f} s   ·   {kind} picture {idx + 1} / {n}   "
                f"({self._overview_dt_us / 1000.0:.2f} ms per picture)"
            )
            self._update_filter_stats(frame=idx)
        else:
            self.playhead_label.setText(f"t = {t_s:.4f} s")

    def _on_playhead_moved(self) -> None:
        if self._syncing_range:
            return
        self._set_playhead_s(float(self.playhead.value()))

    def _on_preview_index(self, ind, _time) -> None:  # noqa: ANN001
        if self._syncing_range or self._overview is None or self._showing_activity:
            return
        t_s = self._overview_origin_s() + int(ind) * self._overview_dt_us / 1_000_000.0
        self._set_playhead_s(t_s)

    def _on_timeline_clicked(self, event) -> None:  # noqa: ANN001
        if event.button() != Qt.MouseButton.LeftButton:
            return
        vb = self.time_plot.plotItem.vb
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        if event.double():
            self._reset_time_zoom()
            return
        x = vb.mapSceneToView(event.scenePos()).x()
        self._set_playhead_s(x)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        preview_ports = {
            self.preview.ui.graphicsView.viewport(),
            self.preview_before.ui.graphicsView.viewport(),
        }
        if (
            obj in preview_ports
            and event.type() == QEvent.Type.MouseButtonDblClick
            and isinstance(event, QMouseEvent)
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._fit_picture()
            return True
        watched = {self.time_plot, self.time_plot.viewport()}
        if obj in watched and event.type() == QEvent.Type.Wheel:
            wheel = event
            mods = QApplication.keyboardModifiers()
            if mods & Qt.KeyboardModifier.ControlModifier:
                return False
            if isinstance(wheel, QWheelEvent) and self._overview is not None:
                step = 1 if wheel.angleDelta().y() < 0 else -1
                idx = self._frame_at_s(self._playhead_s) + step
                t_s = (
                    self._overview_origin_s()
                    + idx * self._overview_dt_us / 1_000_000.0
                )
                self._set_playhead_s(t_s)
                return True
        return super().eventFilter(obj, event)

    def _refresh_plan_label(self) -> None:
        if self._store is None:
            return
        t0, t1 = self._selected_window_us()
        window_us = t1 - t0
        dt_us = self._user_dt_us()
        dt_us, n, _used_us = plan_pictures(window_us, dt_us=dt_us)
        span_s = window_us / 1_000_000.0
        t0_s = (t0 - self._store.t_min) / 1_000_000.0
        t1_s = (t1 - self._store.t_min) / 1_000_000.0
        ram = read_ram()
        itemsize = self._wire_itemsize()
        h, w = self._send_hw()
        budget = assess_stack(n, h, w, ram, wire_itemsize=itemsize)
        self._stack_budget = budget
        y_b = YELLOW_FRAC * ram.total
        r_b = RED_FRAC * ram.total
        crop = self._crop_xyxy()
        if crop is None:
            crop_txt = f"full sensor {self._store.width}×{self._store.height}"
        else:
            x0, y0, x1, y1 = crop
            crop_txt = f"crop {x1 - x0}×{y1 - y0} at ({x0},{y0})"
        sb = self._spatial_bin()
        if sb > 1:
            crop_txt += f" · {sb}×{sb} → {w}×{h}"
        filt_txt = self._filter_brief()
        sug_us, sug_n, _u = plan_nice_pictures(window_us, n_frames=OVERVIEW_PICTURES)
        short = (
            f"{t0_s:.3f}–{t1_s:.3f} s  ·  Δt {dt_us / 1000.0:.3f} ms → {n} pics"
            f"  ·  suggested {sug_us / 1000.0:.3f} ms ({sug_n})"
            f"  ·  {crop_txt}"
        )
        self.min_dt_label.setText(short)
        self.min_dt_label.setToolTip(
            f"Time in this file starts at 0 (first event), not the clock. "
            f"Timestamps step by {SENSOR_DT_US} µs. "
            f"Selected {t0_s:.4f}–{t1_s:.4f} s ({span_s:.3f} s). "
            f"Δt = {dt_us / 1000.0:.3f} ms → {n} pictures (Rebuild (O) "
            f"and BLITZ send use this). "
            f"Suggested (1-2-5, ~{OVERVIEW_PICTURES} pictures): "
            f"{sug_us / 1000.0:.3f} ms ({sug_n} pics). "
            f"{crop_txt}. {filt_txt}. "
            f"Yellow ≥ {fmt_bytes(int(y_b))} (1/8 RAM), "
            f"red ≥ {fmt_bytes(int(r_b))} (1/4 RAM). "
            f"Above {NAV_WARN_FRAMES} pictures BLITZ becomes uncomfortable to scrub."
        )
        self.ram_banner.set_budget(budget, ram)
        self._style_send_button(budget.level)

    def _style_send_button(self, level: str) -> None:
        ready = self._overview is not None
        if level == "block":
            self.apply_btn.setEnabled(False)
            self.save_npy_btn.setEnabled(False)
            self.apply_btn.setText("Too big for free RAM — raise Δt or shrink range")
            self.apply_btn.setStyleSheet(
                "QPushButton { background:#6b0000; color:#ffffff; font-weight:bold; "
                "padding:8px; }"
            )
            return
        self.apply_btn.setEnabled(ready)
        self.save_npy_btn.setEnabled(ready)
        self.apply_btn.setText("Build pictures and send to BLITZ")
        if level == "yellow":
            self.apply_btn.setStyleSheet(
                "QPushButton { background:#f1c40f; color:#1a1400; font-weight:bold; "
                "padding:8px; }"
            )
        elif level == "red":
            self.apply_btn.setStyleSheet(
                "QPushButton { background:#e74c3c; color:#ffffff; font-weight:bold; "
                "padding:8px; }"
            )
        else:
            self.apply_btn.setStyleSheet("")

    def _on_region_changed(self) -> None:
        if self._syncing_range:
            return
        self._refresh_plan_label()

    def _on_region_change_finished(self) -> None:
        if self._syncing_range:
            return
        self._apply_suggested_dt(from_user=False)

    def _suggest_dt_from_overview(self) -> None:
        self._set_dt_ms(self._overview_dt_us / 1000.0, from_user=False)
        self._refresh_plan_label()

    def _set_dt_ms(self, dt_ms: float, *, from_user: bool) -> None:
        self._dt_user_set = from_user
        self.dt_ms.blockSignals(True)
        self.dt_ms.setValue(dt_ms)
        self.dt_ms.blockSignals(False)

    def _user_dt_us(self) -> int:
        return max(SENSOR_DT_US, int(round(self.dt_ms.value() * 1000.0)))

    def _neighbor_dt_us(self) -> int | None:
        if not self.neighbor_box.isChecked():
            return None
        return self._user_dt_us()

    def _spatial_bin(self) -> int:
        data = self.spatial_bin.currentData()
        return max(1, int(data if data is not None else 1))

    def _wire_itemsize(self) -> int:
        if self._representation() in (
            Representation.STATES,
            Representation.OCCUPANCY,
        ):
            return 1
        if self.eight_bit.isChecked():
            return 1
        return 2

    def _gui_polarity(self) -> PolarityMode:
        """GUI send is activity / any-fire; ON/OFF/signed stay CLI-only."""
        return PolarityMode.COLOR

    def _representation(self) -> Representation:
        data = self.representation.currentData()
        return Representation.STATES if data is None else Representation(data)

    def _sync_send_widgets(self) -> None:
        self.grayscale_box.setChecked(True)

    def _filter_sig(self) -> tuple:
        return (self._neighbor_dt_us(), self._spatial_bin())

    def _filter_brief(self) -> str:
        parts: list[str] = []
        if self.drop_isolated_box.isChecked():
            parts.append("1-pixel")
        if self.neighbor_box.isChecked():
            parts.append(f"neighbour (= Δt {self.dt_ms.value():.3f} ms)")
        return ("filters: " + ", ".join(parts)) if parts else "filters off"

    def _filter_status_suffix(self) -> str:
        brief = self._filter_brief()
        if brief == "filters off":
            return ""
        return f" {brief}."

    def _on_dt_changed(self, *_args) -> None:  # noqa: ANN001
        self._dt_user_set = True
        self._refresh_plan_label()

    def _on_spatial_bin_changed(self, *_args) -> None:  # noqa: ANN001
        self._snap_crop_to_spatial()
        self._refresh_plan_label()
        if self._store is None or self._overview is None:
            return
        self._request_full_frame()
        k = self._spatial_bin()
        self._set_status(
            "work",
            f"Spatial bin {k}×{k} — updating preview…",
        )
        self._rebuild_preview_for_filters()

    def _snap_crop_to_spatial(self) -> None:
        """Keep a committed crop on the spatial-bin grid (preview = send)."""
        if self._crop_box is None or self._store is None:
            return
        sb = self._spatial_bin()
        if sb <= 1:
            return
        x0, y0, x1, y1 = self._crop_box
        x0 = ((x0 + sb - 1) // sb) * sb
        y0 = ((y0 + sb - 1) // sb) * sb
        x1 = (x1 // sb) * sb
        y1 = (y1 // sb) * sb
        x1 = min(int(self._store.width), x1)
        y1 = min(int(self._store.height), y1)
        if x1 - x0 < sb or y1 - y0 < sb:
            self._crop_box = None
            return
        self._crop_box = (x0, y0, x1, y1)

    def _on_filter_checkbox(self, *_args) -> None:  # noqa: ANN001
        self._refresh_plan_label()
        if self._counts_mid is None:
            return
        if not self.drop_isolated_box.isChecked():
            self._counts_after = self._counts_mid
            self._activity_after = self._activity_mid
            self._mass_after = self._mass_mid
            self._set_status("work", "Updating preview…")
            self._view_timer.start()
            return
        self._start_spatial()

    def _start_spatial(self) -> None:
        if self._counts_mid is None:
            return
        self._spatial_src = self._counts_mid
        self._set_status("work", "Applying 1-pixel filter…")
        if self._store is None:
            return
        yellow = self._overview_is_detail
        self._start_bin(self._overview_params(yellow=yellow), "spatial")

    def _rebuild_preview_for_filters(self) -> None:
        if self._store is None or self._overview is None:
            return
        brief = self._filter_brief()
        if self._showing_activity:
            self._set_status("work", f"Updating window max ({brief})…")
            self._start_bin(self._activity_params(), "activity")
            return
        yellow = self._overview_is_detail
        kind = "overview_view" if yellow else "overview"
        self._set_status(
            "work",
            f"Applying filters ({brief}) — pictures stay until this finishes…",
        )
        self._start_bin(self._overview_params(yellow=yellow), kind)

    def _display_counts(self, arr: np.ndarray) -> np.ndarray:
        return preview_from_on_off(
            arr,
            self._gui_polarity(),
            self._representation(),
            hi=self._event_scale_hi,
        )

    def _image_axes(self, arr: np.ndarray) -> dict[str, int]:
        if arr.ndim == 4:
            return {"t": 0, "y": 1, "x": 2, "c": 3}
        return {"t": 0, "y": 1, "x": 2}

    def _wire_preview_views(self) -> None:
        vb0 = self.preview_before.getView()
        vb1 = self.preview.getView()
        vb0.sigRangeChanged.connect(self._on_preview_view_range)
        vb1.sigRangeChanged.connect(self._on_preview_view_range)
        vb0.sigRangeChangedManually.connect(self._on_preview_user_zoom)
        vb1.sigRangeChangedManually.connect(self._on_preview_user_zoom)
        vb0.sigResized.connect(self._on_preview_resized)
        vb1.sigResized.connect(self._on_preview_resized)
        self.preview_before.ui.graphicsView.viewport().installEventFilter(self)
        self.preview.ui.graphicsView.viewport().installEventFilter(self)

    def _on_preview_resized(self, *_args) -> None:  # noqa: ANN001
        if self._syncing_view or self._preview_wh is None:
            return
        self._hide_preview_extras()
        self._apply_preview_view_limits(*self._preview_wh)
        if not self._preview_user_zoom:
            self._fit_preview_full_frame()

    def _apply_preview_view_limits(self, width: int, height: int) -> None:
        """Zoom in freely; zoom out stops when the whole picture is visible."""
        w = max(1.0, float(width))
        h = max(1.0, float(height))
        self._preview_wh = (int(width), int(height))
        margin = CROP_EDIT_MARGIN if self._crop_editing else 0.0
        self._syncing_view = True
        try:
            for view in (self.preview_before, self.preview):
                vb = view.getView()
                vb.setAspectLocked(True)
                vb.setDefaultPadding(0.0)
                vw, vh = self._preview_pixel_size(view)
                xr, yr = preview_contain_ranges(w, h, vw, vh, margin)
                max_x = max(w, xr[1] - xr[0])
                max_y = max(h, yr[1] - yr[0])
                vb.setLimits(
                    xMin=xr[0],
                    xMax=xr[1],
                    yMin=yr[0],
                    yMax=yr[1],
                    minXRange=1.0,
                    minYRange=1.0,
                    maxXRange=max_x,
                    maxYRange=max_y,
                )
        finally:
            self._syncing_view = False

    def _fit_preview_full_frame(self) -> None:
        """Default view: the entire picture, letterboxed with slate blue."""
        if self._preview_wh is None:
            return
        width, height = self._preview_wh
        margin = CROP_EDIT_MARGIN if self._crop_editing else 0.0
        self._syncing_view = True
        try:
            for view in (self.preview_before, self.preview):
                vb = view.getView()
                vw, vh = self._preview_pixel_size(view)
                if vw < 8.0 or vh < 8.0:
                    continue
                xr, yr = preview_contain_ranges(
                    width,
                    height,
                    vw,
                    vh,
                    margin=margin,
                )
                vb.setAspectLocked(True)
                vb.setDefaultPadding(0.0)
                vb.enableAutoRange(enable=False)
                vb.setRange(xRange=xr, yRange=yr, padding=0.0)
        finally:
            self._syncing_view = False

    def _fit_preview_to_height(self) -> None:
        self._fit_preview_full_frame()

    def _schedule_fit_preview(self) -> None:
        """Fit again after layout (ViewBox size is often stale in the same event)."""
        self._fit_timer.start()

    def _fit_picture(self) -> None:
        """Reset pan/zoom so the entire picture is visible."""
        self._preview_user_zoom = False
        self._fit_preview_full_frame()

    def _on_preview_view_range(self, vb, *_args) -> None:  # noqa: ANN001
        if self._syncing_view:
            return
        other = (
            self.preview.getView()
            if vb is self.preview_before.getView()
            else self.preview_before.getView()
        )
        xr, yr = vb.viewRange()
        self._syncing_view = True
        try:
            other.setRange(xRange=xr, yRange=yr, padding=0)
        finally:
            self._syncing_view = False

    def _on_preview_user_zoom(self, *_args) -> None:  # noqa: ANN001
        if self._syncing_view:
            return
        self._preview_user_zoom = True

    def _capture_view_range(self) -> None:
        vb = self.preview.getView()
        xr, yr = vb.viewRange()
        self._saved_view_range = (tuple(xr), tuple(yr))

    def _restore_view_range(self) -> None:
        if self._saved_view_range is None:
            return
        xr, yr = self._saved_view_range
        self._syncing_view = True
        try:
            for view in (self.preview_before, self.preview):
                view.getView().setRange(xRange=xr, yRange=yr, padding=0)
        finally:
            self._syncing_view = False

    def _make_probe_roi(self, pos: list[float], size: list[float]) -> pg.ROI:
        roi = pg.ROI(
            pos,
            size,
            pen=pg.mkPen("#ffcc33", width=2),
            hoverPen=pg.mkPen("#ffffff", width=2),
            handlePen=pg.mkPen("#ffcc33", width=1),
            rotatable=False,
        )
        roi.addScaleHandle([1, 1], [0, 0])
        roi.setZValue(15)
        return roi

    def _ensure_probe_rois(self, width: int, height: int) -> None:
        spatial = (int(width), int(height))
        pos = [width * 9 / 20, height * 9 / 20]
        size = [max(8.0, 0.1 * width), max(8.0, 0.1 * height)]
        if self._probe_roi_before is None:
            self._probe_roi_before = self._make_probe_roi(pos, size)
            self._probe_roi_after = self._make_probe_roi(pos, size)
            self.preview_before.view.addItem(self._probe_roi_before)
            self.preview.view.addItem(self._probe_roi_after)
            self._probe_roi_before.sigRegionChanged.connect(
                lambda: self._on_probe_moved(self._probe_roi_before)
            )
            self._probe_roi_after.sigRegionChanged.connect(
                lambda: self._on_probe_moved(self._probe_roi_after)
            )
            self._probe_spatial = spatial
            self._roi_series_timer.start()
            return
        if self._probe_spatial != spatial:
            self._syncing_probe = True
            try:
                for roi in (self._probe_roi_before, self._probe_roi_after):
                    roi.setPos(pos, update=False)
                    roi.setSize(size, update=True)
            finally:
                self._syncing_probe = False
            self._probe_spatial = spatial
            self._roi_series_timer.start()

    def _on_probe_moved(self, src: pg.ROI) -> None:
        if self._syncing_probe:
            return
        other = (
            self._probe_roi_after
            if src is self._probe_roi_before
            else self._probe_roi_before
        )
        if other is None:
            return
        self._syncing_probe = True
        try:
            other.setPos(src.pos(), update=False)
            other.setSize(src.size(), update=True)
        finally:
            self._syncing_probe = False
        self._roi_series_timer.start()

    def _set_preview_busy(self, on: bool, text: str = "") -> None:
        for lab in (self._busy_before, self._busy_after):
            lab.setText(text)
            lab.setVisible(on)

    def _after_stack(self) -> np.ndarray | None:
        if self.drop_isolated_box.isChecked() and self._counts_after is not None:
            return self._counts_after
        return self._counts_mid

    def _activity_after_stack(self) -> np.ndarray | None:
        if self.drop_isolated_box.isChecked() and self._activity_after is not None:
            return self._activity_after
        return self._activity_mid

    def _probe_xyxy(self, width: int, height: int) -> tuple[int, int, int, int]:
        roi = self._probe_roi_before
        if roi is None:
            x0 = int(width * 9 / 20)
            y0 = int(height * 9 / 20)
            return (
                x0,
                y0,
                x0 + max(8, int(0.1 * width)),
                y0 + max(8, int(0.1 * height)),
            )
        pos = roi.pos()
        size = roi.size()
        x0 = int(np.floor(float(pos.x())))
        y0 = int(np.floor(float(pos.y())))
        x1 = int(np.ceil(float(pos.x()) + float(size.x())))
        y1 = int(np.ceil(float(pos.y()) + float(size.y())))
        return x0, y0, x1, y1

    def _refresh_roi_series(self) -> None:
        before = self._counts_before
        if before is None or before.shape[0] < 1:
            self.roi_before_curve.setData([], [])
            self.roi_after_curve.setData([], [])
            self.roi_after_curve.setVisible(False)
            self.rate_curve.setVisible(True)
            self.time_plot.setLabel("left", "Events")
            self._set_timeline_legend("file")
            return
        ox, oy = 0, 0
        h, w = int(before.shape[1]), int(before.shape[2])
        if not self._crop_editing:
            box = self._display_crop_xyxy(before)
            if box is not None:
                ox, oy = box[0], box[1]
                w, h = box[2] - box[0], box[3] - box[1]
        x0, y0, x1, y1 = self._probe_xyxy(w, h)
        x0, x1 = x0 + ox, x1 + ox
        y0, y1 = y0 + oy, y1 + oy
        origin = self._overview_origin_s()
        dt_s = self._overview_dt_us / 1_000_000.0
        t = origin + np.arange(before.shape[0], dtype=np.float64) * dt_s
        y_b = roi_event_series(before, x0, y0, x1, y1)
        self.roi_before_curve.setData(t, y_b)
        self.roi_before_curve.setVisible(True)
        after = self._after_stack()
        filters_on = (
            self.drop_isolated_box.isChecked() or self.neighbor_box.isChecked()
        )
        if (
            filters_on
            and after is not None
            and after.shape[0] == before.shape[0]
        ):
            y_a = roi_event_series(after, x0, y0, x1, y1)
            self.roi_after_curve.setData(t, y_a)
            self.roi_after_curve.setVisible(True)
        else:
            self.roi_after_curve.setData([], [])
            self.roi_after_curve.setVisible(False)
        self.rate_curve.setVisible(False)
        self.time_plot.setLabel("left", "ROI events")
        self._set_timeline_legend("roi_filtered" if filters_on else "roi")

    def _set_timeline_legend(self, kind: str) -> None:
        if kind == "file":
            html = '<span style="color:#8aa4ad">━ file events</span>'
        elif kind == "roi_filtered":
            html = (
                '<span style="color:#7ec8e3">━ ROI</span>'
                "&nbsp;&nbsp;"
                '<span style="color:#ffcc33">━ ROI filtered</span>'
            )
        else:
            html = '<span style="color:#7ec8e3">━ ROI</span>'
        self.timeline_legend.setText(html)

    def _update_filter_stats(self, *, frame: int | None = None) -> None:
        filters_on = (
            self.drop_isolated_box.isChecked() or self.neighbor_box.isChecked()
        )
        if self._showing_activity:
            before = self._activity_before
            after = self._activity_after_stack()
        else:
            before = self._counts_before
            after = self._after_stack()
        if not filters_on or before is None or after is None:
            self.filter_stats.setText("Filters off.")
            return
        if self._showing_activity:
            glob = removed_fraction(before, after, frame=0)
            self.filter_stats.setText(
                f"Removed {glob:.0%} in this window (activity image)."
            )
        else:
            if frame is None:
                frame = self._frame_at_s(self._playhead_s)
            pic = removed_fraction(before, after, frame=frame)
            if (
                self._mass_before is not None
                and self._mass_after is not None
                and self._mass_before > 0.0
            ):
                glob = max(
                    0.0,
                    min(
                        1.0,
                        (self._mass_before - self._mass_after) / self._mass_before,
                    ),
                )
            else:
                glob = pic
            self.filter_stats.setText(
                f"Removed {pic:.0%} in this picture · {glob:.0%} in all "
                f"{before.shape[0]} pictures (this window)."
            )

    def _set_view_image(
        self,
        view: pg.ImageView,
        arr: np.ndarray,
        *,
        levels: tuple[float, float] | None = None,  # noqa: ARG002
        auto_range: bool = True,  # noqa: ARG002  (range is contain / full frame)
    ) -> None:
        axes = self._image_axes(arr)
        view.setImage(arr, autoLevels=False, autoRange=False, axes=axes)
        item = view.getImageItem()
        item.setLevels((0.0, 1.0))
        item.setOpts(smooth=False)

    def _count_ladder_note(self) -> str:
        rep = self._representation()
        if rep == Representation.STATES:
            short = "States: black none · red OFF · green ON · yellow both"
            tip = (
                "Polarity in this Δt: black = nothing, red = OFF, "
                "green = ON, yellow = both. "
                "BLITZ gets one uint8 channel: 0 / 85 / 170 / 255."
            )
        elif rep == Representation.OCCUPANCY:
            short = "Occupancy: black = nothing · white = fired"
            tip = (
                "Binary who-fired. Preview is black / white. "
                "BLITZ gets uint8 0 or 255 (any polarity)."
            )
        else:
            n = max(1, int(round(float(self._event_scale_hi))))
            short = f"Counts: Inferno  ·  rungs 0…{n} events/pixel  ·  BLITZ gets uint16"
            tip = (
                f"How many events (ON+OFF) in this Δt. Inferno: dark = none, "
                f"bright = many. Rungs 0…{n} (p99 of unfiltered). "
                "BLITZ receives uint16 activity. Its LUT is separate."
            )
        self.color_legend.setToolTip(tip)
        return short

    def _refresh_filter_views(self) -> None:
        if self._counts_before is None or self._counts_mid is None:
            return
        after = self._after_stack()
        if after is None:
            after = self._counts_mid
        pol = self._gui_polarity()
        self._event_scale_hi = float(polarity_count_ceiling(self._counts_before, pol))
        self._overview_before = self._display_counts(self._counts_before)
        self._overview = self._display_counts(after)
        self._overview_applied_sig = self._filter_sig()
        self.color_legend.setText(self._count_ladder_note())
        if not self._showing_activity:
            self._show_overview_stack()
        self._update_filter_stats()
        self._refresh_roi_series()

    def _apply_suggested_dt(self, *, from_user: bool = False) -> None:
        if self._store is None:
            return
        t0, t1 = self._selected_window_us()
        dt_us, _n, _u = plan_nice_pictures(t1 - t0, n_frames=OVERVIEW_PICTURES)
        self._set_dt_ms(dt_us / 1000.0, from_user=from_user)
        self._refresh_plan_label()

    def _interlace_hint(self, arr: np.ndarray) -> str:
        ratio = even_odd_row_ratio(arr)
        if ratio <= INTERLACE_RATIO_WARN:
            return ""
        dt = self._user_dt_us()
        scale = (
            f" Δt is {dt / 1000.0:.3f} ms (below 1 ms)."
            if dt < INTERLACE_DT_US
            else f" Δt is {dt / 1000.0:.3f} ms."
        )
        return (
            f" Even/odd rows differ ×{ratio:.1f} — likely sensor readout, "
            f"not the decoder.{scale}"
        )

    def _preview_budget(self) -> StackBudget | None:
        if self._store is None:
            return None
        t0, t1 = self._selected_window_us()
        dt_us, n, _u = plan_pictures(t1 - t0, dt_us=self._user_dt_us())
        ram = read_ram()
        itemsize = self._wire_itemsize()
        return assess_stack(
            n,
            self._store.height,
            self._store.width,
            ram,
            wire_itemsize=itemsize,
        )

    def _confirm_stack(self, budget: StackBudget | None, *, preview: bool) -> bool:
        if budget is None:
            return True
        place = "this preview" if preview else "the wire"
        if budget.level == "block":
            QMessageBox.critical(
                self,
                "Not enough RAM",
                f"{budget.n_frames} pictures = {fmt_bytes(budget.wire_bytes)} "
                f"for {place}, but only {fmt_bytes(read_ram().available)} is free. "
                "Raise Δt, shrink the yellow band, or crop.",
            )
            return False
        if budget.level != "red" and not budget.nav_warn:
            return True
        parts: list[str] = []
        if budget.nav_warn:
            parts.append(
                f"{budget.n_frames} pictures — more than {NAV_WARN_FRAMES} is "
                "not comfortable to scrub. Continue only if you know you need this."
            )
        if budget.level == "red":
            parts.append(
                f"{fmt_bytes(budget.wire_bytes)} is "
                f"{budget.fraction_of_total:.0%} of this PC's RAM. "
                "BLITZ will hold another copy if you send."
            )
        yes = QMessageBox.warning(
            self,
            "Check before preview" if preview else "Check before send",
            "\n\n".join(parts),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return yes == QMessageBox.StandardButton.Yes

    def _queue_or_keep_pending(self, params: BinParams, kind: str) -> None:
        pending = self._pending_bin
        if pending is not None and pending[1] == "export" and kind != "export":
            return
        self._pending_bin = (params, kind)

    def _start_bin(self, params: BinParams, kind: str) -> None:
        if self._store is None:
            return
        if self._bin_thread is not None and self._bin_thread.isRunning():
            self._queue_or_keep_pending(params, kind)
            return
        self._pending_bin = None
        self._bin_busy_kind = kind
        self._bin_inflight_params = params
        self._set_preview_busy(True, self.led.label.text() or "Working…")
        thread = QThread(self)
        worker = _BinWorker(
            self._store,
            params,
            kind,
            stack=self._spatial_src if kind == "spatial" else None,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_binned)
        worker.failed.connect(self._on_bin_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._clear_bin_worker)
        self._bin_thread = thread
        self._bin_worker = worker
        thread.start()

    def _clear_bin_worker(self) -> None:
        if self._bin_worker is not None:
            self._bin_worker.deleteLater()
            self._bin_worker = None
        if self._bin_thread is not None:
            self._bin_thread.deleteLater()
            self._bin_thread = None
        self._bin_busy_kind = None
        pending = self._pending_bin
        self._pending_bin = None
        if pending is not None:
            self._start_bin(*pending)
        else:
            self._set_preview_busy(False)

    def _hide_preview_extras(self) -> None:
        for view in (self.preview, self.preview_before):
            view.ui.roiPlot.hide()
            view.ui.histogram.hide()
            view.ui.roiPlot.setMaximumHeight(0)
            view.ui.histogram.setMaximumWidth(0)
            view.ui.splitter.setSizes([10_000, 0])
            view.ui.splitter.setHandleWidth(0)

    def _request_full_frame(self) -> None:
        """Next picture update fills the pane (full frame), ignoring leftover zoom."""
        self._force_fit_preview = True
        self._preview_user_zoom = False
        self._preview_wh = None

    def _keep_preview_view(self, width: int, height: int) -> bool:
        if self._force_fit_preview:
            self._force_fit_preview = False
            self._preview_user_zoom = False
            return False
        return (
            self._preview_user_zoom
            and self._preview_wh == (width, height)
        )

    def _preview_pixel_size(self, view: pg.ImageView) -> tuple[float, float]:
        port = view.ui.graphicsView.viewport()
        return max(1.0, float(port.width())), max(1.0, float(port.height()))

    def _show_overview_stack(self) -> None:
        if self._overview is None:
            return
        before = (
            self._overview_before
            if self._overview_before is not None
            else self._overview
        )
        self._showing_activity = False
        self._syncing_range = True
        before = self._slice_preview(before)
        ov = self._slice_preview(self._overview)
        h, w = int(before.shape[1]), int(before.shape[2])
        keep_view = self._keep_preview_view(w, h)
        if keep_view:
            self._capture_view_range()
        self._set_view_image(self.preview_before, before, auto_range=not keep_view)
        self._set_view_image(self.preview, ov, auto_range=not keep_view)
        self._hide_preview_extras()
        self._apply_preview_view_limits(w, h)
        self._ensure_probe_rois(w, h)
        if keep_view:
            self._restore_view_range()
        else:
            self._fit_preview_full_frame()
            self._schedule_fit_preview()
        idx = self._frame_at_s(self._playhead_s)
        self.preview.setCurrentIndex(idx)
        self.preview_before.setCurrentIndex(idx)
        self._syncing_range = False
        if self._crop_editing:
            self._ensure_crop_roi(show=True, width=w, height=h)
        else:
            self._ensure_crop_roi(show=False, width=w, height=h)

    def _show_activity_pair(self) -> None:
        if self._activity_before is None or self._activity_mid is None:
            return
        after = self._activity_after_stack()
        if after is None:
            after = self._activity_mid
        pol = self._gui_polarity()
        rep = self._representation()
        hi = float(polarity_count_ceiling(self._activity_before, pol))
        self._event_scale_hi = hi
        disp_b = preview_from_on_off(self._activity_before, pol, rep, hi=hi)
        disp_a = preview_from_on_off(after, pol, rep, hi=hi)
        disp_b = self._slice_preview(disp_b)
        disp_a = self._slice_preview(disp_a)
        self.color_legend.setText(self._count_ladder_note())
        n_pos_b = int(np.count_nonzero(np.abs(disp_b) > 0))
        n_pos = int(np.count_nonzero(np.abs(disp_a) > 0))
        h, w = int(disp_b.shape[1]), int(disp_b.shape[2])
        keep_view = self._keep_preview_view(w, h)
        if keep_view:
            self._capture_view_range()
        self._showing_activity = True
        self._syncing_range = True
        self._set_view_image(
            self.preview_before, disp_b, auto_range=not keep_view
        )
        self._set_view_image(
            self.preview, disp_a, auto_range=not keep_view
        )
        self._hide_preview_extras()
        self._apply_preview_view_limits(w, h)
        self._ensure_probe_rois(w, h)
        if keep_view:
            self._restore_view_range()
        else:
            self._fit_preview_full_frame()
            self._schedule_fit_preview()
        self._syncing_range = False
        if self._crop_editing:
            self._ensure_crop_roi(show=True, width=w, height=h)
        else:
            self._ensure_crop_roi(show=False, width=w, height=h)
        self._update_filter_stats(frame=0)
        if n_pos == 0 and n_pos_b == 0:
            self._set_status(
                "wait",
                "Window max is empty — no events in the yellow band. "
                "Widen the band, then press M again.",
            )
        else:
            if self._crop_editing:
                extra = (
                    "Move the green rectangle — mouse-up does not lock it. "
                    "Handles sit in the blue margin. Apply crop when it fits. "
                )
            elif self._crop_box is not None:
                extra = "Crop (M) to change the crop. "
            else:
                extra = "After the yellow band is right, Crop (M) to draw a crop. "
            self._set_status(
                "ok",
                f"Window max ({n_pos:,} lit pixels after filters, "
                f"{n_pos_b:,} before)."
                f"{self._filter_status_suffix()} "
                "Left unfiltered, right filtered. "
                + extra
                + "Scrub the timeline to return to the overview.",
            )
        self._refresh_plan_label()

    def _adopt_worker_masses(self, kind: str) -> None:
        worker = self._bin_worker
        if worker is None:
            return
        if kind == "spatial":
            if worker.mass_after is not None:
                self._mass_after = worker.mass_after
            if worker.mass_mid is not None:
                self._mass_mid = worker.mass_mid
            return
        if worker.mass_before is not None:
            self._mass_before = worker.mass_before
        if worker.mass_mid is not None:
            self._mass_mid = worker.mass_mid
        if worker.mass_after is not None:
            self._mass_after = worker.mass_after

    def _on_binned(
        self, before: object, after: object, elapsed: float, kind: str
    ) -> None:
        arr = np.asarray(after)
        params = self._bin_inflight_params
        result_after = (
            self._bin_worker.result_after
            if self._bin_worker is not None
            and self._bin_worker.result_after is not None
            else arr
        )
        self._adopt_worker_masses(kind)
        if kind == "spatial":
            self._counts_after = arr
            if not self._overview_is_detail:
                self._counts_after_full = arr
            if self._activity_mid is not None:
                self._activity_after = drop_isolated_pixels(self._activity_mid)
            self._refresh_filter_views()
            if self._showing_activity:
                self._show_activity_pair()
            self._set_preview_busy(False)
            self._set_status(
                "ok",
                "1-pixel filter applied."
                f"{self._filter_status_suffix()} "
                "Plot: cyan = ROI unfiltered, gold = ROI filtered.",
            )
            return
        if kind == "activity":
            if before is None:
                self._activity_before = arr
                self._activity_mid = arr
            else:
                self._activity_before = np.asarray(before)
                self._activity_mid = arr
            self._activity_after = result_after
            self._show_activity_pair()
            self._set_preview_busy(False)
            return

        if kind in ("overview", "overview_view"):
            first_full = self._overview_full is None
            t0 = int(self._store.t_min) if self._store is not None else 0
            dt = params.dt_us if params is not None else self._overview_dt_us
            if params is not None and params.t0_us is not None:
                t0 = int(params.t0_us)
            raw_before = np.asarray(before) if before is not None else arr
            self._counts_before = raw_before
            self._counts_mid = arr
            self._counts_after = result_after
            self._overview_dt_us = dt
            self._overview_t0_us = t0
            if kind == "overview":
                self._counts_before_full = raw_before
                self._counts_mid_full = arr
                self._counts_after_full = result_after
                self._overview_full_dt_us = dt
                self._overview_full_t0_us = t0
                self._overview_is_detail = False
            else:
                self._overview_is_detail = True
            self._refresh_filter_views()
            if kind == "overview":
                self._overview_full = self._overview
                self._overview_full_before = self._overview_before
                self._overview_full_sig = self._filter_sig()
            n = int(self._overview.shape[0]) if self._overview is not None else 0
            self.restag_btn.setEnabled(True)
            self.full_rec_btn.setEnabled(True)
            self._sync_crop_buttons()
            if first_full and kind == "overview":
                self._suggest_dt_from_overview()
            if first_full and kind == "overview":
                self._set_playhead_s(0.0)
            self.apply_btn.setEnabled(True)
            self.save_npy_btn.setEnabled(True)
            if kind == "overview_view":
                dt_ms = self._overview_dt_us / 1000.0
                hint = self._interlace_hint(arr)
                self._set_status(
                    "ok",
                    f"Preview ready ({n} pictures, {dt_ms:.3f} ms each) — "
                    f"same Δt as a BLITZ send.{self._filter_status_suffix()}"
                    f"{hint} "
                    "Left unfiltered, right filtered. "
                    "The plot is the yellow-box ROI (cyan unfiltered, "
                    "gold filtered). "
                    "Scrub here to inspect; send when it looks right.",
                )
            else:
                self._set_status(
                    "ok",
                    f"Overview ready ({n} pictures)."
                    f"{self._filter_status_suffix()} "
                    "Left unfiltered, right filtered. "
                    "The plot is the yellow-box ROI (cyan unfiltered, "
                    "gold filtered). "
                    "Scrub the timeline, set the yellow start/end, choose Δt, "
                    "then send to BLITZ.",
                )
            self._refresh_plan_label()
            self._set_preview_busy(False)
            if (
                self._neighbor_dt_us() is not None
                and (params is None or params.neighbor_dt_us is None)
            ):
                self._rebuild_preview_for_filters()
            return

        self._set_preview_busy(False)
        net = self._wire_cube(arr)
        hint = self._interlace_hint(arr)
        if kind == "export_npy":
            self._write_npy(net, elapsed, hint)
            return

        self.publisher.set_stack(net, push=True)
        if self._blitz_clients <= 0:
            self._set_status(
                "wait",
                "3/3 Pictures ready — connect BLITZ Stream, or wait for download…"
                + hint,
            )
        else:
            self._set_status("wait", "3/3 Sending to BLITZ…" + hint)
        self.statusBar().showMessage(
            f"Sent {tuple(net.shape)} in {elapsed * 1000:.0f} ms{hint}"
        )

    def _crop_xyxy(self) -> tuple[int, int, int, int] | None:
        """Crop in sensor pixels: live ROI while editing, else committed box."""
        if self._crop_editing:
            src = (
                self._activity_before
                if self._showing_activity
                else self._counts_before
            )
            if src is not None:
                return self._sensor_box_from_roi(
                    int(src.shape[2]), int(src.shape[1])
                )
        return self._crop_box

    def _display_crop_xyxy(
        self, arr: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """Crop in the current stack's pixel coords, or None while editing."""
        if self._crop_editing or self._crop_box is None:
            return None
        sb = self._spatial_bin()
        x0, y0, x1, y1 = self._crop_box
        x0, y0, x1, y1 = x0 // sb, y0 // sb, x1 // sb, y1 // sb
        h, w = int(arr.shape[1]), int(arr.shape[2])
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        return x0, y0, x1, y1

    def _slice_preview(self, arr: np.ndarray) -> np.ndarray:
        box = self._display_crop_xyxy(arr)
        if box is None:
            return arr
        x0, y0, x1, y1 = box
        return arr[:, y0:y1, x0:x1, ...]

    def _send_hw(self) -> tuple[int, int]:
        assert self._store is not None
        crop = self._crop_xyxy()
        if crop is None:
            h, w = int(self._store.height), int(self._store.width)
        else:
            x0, y0, x1, y1 = crop
            h, w = y1 - y0, x1 - x0
        return spatial_out_size(h, w, self._spatial_bin())

    def _export_params(self) -> BinParams:
        assert self._store is not None
        t0, t1 = self._selected_window_us()
        dt_us = self._user_dt_us()
        dt_us, n, _u = plan_pictures(t1 - t0, dt_us=dt_us)
        crop = self._crop_xyxy()
        x0 = y0 = x1 = y1 = None
        if crop is not None:
            x0, y0, x1, y1 = crop
        return BinParams(
            dt_us=dt_us,
            polarity=self._gui_polarity(),
            accum=AccumMode.COUNT,
            t0_us=t0,
            t1_us=t0 + n * dt_us,
            max_frames=n,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            drop_isolated=self.drop_isolated_box.isChecked(),
            neighbor_dt_us=self._neighbor_dt_us(),
            spatial_bin=self._spatial_bin(),
        )

    def _export_to_blitz(self) -> None:
        if self._store is None:
            return
        self._refresh_plan_label()
        if not self._confirm_stack(self._stack_budget, preview=False):
            return
        self._set_status("work", "2/3 Building pictures for BLITZ…")
        self._start_bin(self._export_params(), "export")

    def _default_npy_name(self) -> str:
        raw = Path(self.path_edit.text().strip() or "recording")
        return f"{raw.stem}_{self._representation().value}.npy"

    def _save_as_npy(self) -> None:
        if self._store is None:
            return
        self._refresh_plan_label()
        if not self._confirm_stack(self._stack_budget, preview=False):
            return
        raw = Path(self.path_edit.text().strip() or ".")
        start = raw.parent if raw.suffix else Path(".")
        suggested = start / self._default_npy_name()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save NumPy cube",
            str(suggested),
            "NumPy (*.npy);;All files (*)",
        )
        if not path:
            return
        out = Path(path)
        if out.suffix.lower() != ".npy":
            out = out.with_suffix(".npy")
        self._npy_out = out
        self._set_status("work", f"Building pictures for {out.name}…")
        self._start_bin(self._export_params(), "export_npy")

    def _wire_cube(self, arr: np.ndarray) -> np.ndarray:
        net = stack_for_send(arr, self._gui_polarity(), self._representation())
        return encode_stack_for_send(
            net,
            eight_bit=self.eight_bit.isChecked(),
            log_stretch=self.log_stretch.isChecked(),
            normalize=self.normalize_box.isChecked(),
            grayscale=False,
        )

    def _write_npy(self, net: np.ndarray, elapsed: float, hint: str) -> None:
        path = self._npy_out
        self._npy_out = None
        if path is None:
            self._set_status("err", "Save as NumPy: no path")
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, net)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            self._set_status("err", f"Could not write {path.name}")
            return
        self._set_status(
            "ok",
            f"Wrote {path.name}  {tuple(net.shape)} {net.dtype}  "
            f"({elapsed * 1000:.0f} ms).{hint}",
        )

    def _on_blitz_downloaded(self, nbytes: int) -> None:
        mb = nbytes / (1024 * 1024)
        self._set_status("ok", f"BLITZ received the stack ({mb:.1f} MB)")

    def _on_client_count(self, n: int) -> None:
        prev = self._blitz_clients
        self._blitz_clients = n
        if n == prev:
            return
        detail = self.led.label.text()
        if n > 0:
            suffix = f"BLITZ Stream connected ({n})"
        else:
            suffix = "waiting for BLITZ Stream"
        self.statusBar().showMessage(
            f"Serving {self.publisher.base_url}  ·  {suffix}"
        )
        if self.led.kind == "work" or "received the stack" in detail:
            return
        if n > 0:
            if "Pictures ready" in detail or "Sending" in detail or "download" in detail.lower():
                self._set_status("wait", "BLITZ Stream connected — downloading…")
            else:
                self._set_status("ok", "BLITZ Stream connected — send when ready.")
        elif prev > 0:
            self._set_status(
                "wait",
                "BLITZ Stream disconnected — Connect in BLITZ → Stream.",
            )

    def _on_bin_failed(self, message: str) -> None:
        self._npy_out = None
        self._set_preview_busy(False)
        self._set_status("err", "Failed")
        QMessageBox.critical(self, "Failed", message)

    def _shortcut_ok(self) -> bool:
        w = QApplication.focusWidget()
        return not isinstance(
            w, (QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox)
        )

    def _install_shortcuts(self) -> None:
        sc_o = QShortcut(QKeySequence("O"), self)
        sc_o.activated.connect(self._shortcut_restag)
        sc_esc = QShortcut(QKeySequence("Escape"), self)
        sc_esc.activated.connect(self._shortcut_full_recording)
        sc_m = QShortcut(QKeySequence("M"), self)
        sc_m.activated.connect(self._shortcut_activity)

    def _shortcut_restag(self) -> None:
        if self._shortcut_ok():
            self._restag_overview_for_view()

    def _shortcut_full_recording(self) -> None:
        if self._shortcut_ok():
            self._reset_time_zoom()

    def _shortcut_activity(self) -> None:
        if self._shortcut_ok():
            self._show_activity_crop()

    def _on_xrange_changed(self, *_args) -> None:  # noqa: ANN001
        if self._syncing_range or self._store is None:
            return
        self._rate_zoom_timer.start()

    def _refresh_rate_curve_from_view(self) -> None:
        if self._store is None:
            return
        if self._counts_before is not None:
            return
        t0, t1 = self._visible_window_us()
        x_ms, counts = event_rate_ms(self._store, t0_us=t0, t1_us=t1)
        self.rate_curve.setData(x_ms / 1000.0, counts)

    def _time_edge_pad(self, dur_s: float) -> float:
        """Empty time on each side of the plot so yellow handles are not on the frame."""
        return max(dur_s * 0.04, 1e-4)

    def _apply_time_limits(self, dur_s: float) -> None:
        pad = self._time_edge_pad(dur_s)
        vb = self.time_plot.getViewBox()
        vb.setLimits(
            xMin=-pad,
            xMax=dur_s + pad,
            minXRange=min(dur_s, max(1e-4, SENSOR_DT_US / 1_000_000.0)),
        )

    def _set_time_view(self, x0: float, x1: float, *, pad_frac: float = 0.05) -> None:
        if x1 < x0:
            x0, x1 = x1, x0
        dur = self._duration_s()
        span = max(x1 - x0, 1e-4)
        pad = span * pad_frac
        edge = self._time_edge_pad(dur)
        self._apply_time_limits(dur)
        self._syncing_range = True
        self.time_plot.setXRange(
            max(-edge, x0 - pad),
            min(dur + edge, x1 + pad),
            padding=0,
        )
        self._syncing_range = False
        self._refresh_rate_curve_from_view()

    def _region_range_s(self) -> tuple[float, float]:
        t0_s, t1_s = self.region.getRegion()
        if t1_s < t0_s:
            t0_s, t1_s = t1_s, t0_s
        return float(t0_s), float(t1_s)

    def _reset_time_zoom(self) -> None:
        if self._store is None:
            return
        dur = self._duration_s()
        self._set_time_view(0.0, dur)
        if not self._overview_is_detail:
            self._set_playhead_s(self._playhead_s)
            return
        self._set_status("work", "Restoring full recording…")
        QTimer.singleShot(0, self._restore_full_recording)

    def _restore_full_recording(self) -> None:
        if self._store is None:
            self._set_preview_busy(False)
            return
        cache_ok = (
            self._counts_before_full is not None
            and self._counts_mid_full is not None
            and self._overview_full_sig == self._filter_sig()
        )
        if cache_ok:
            self._counts_before = self._counts_before_full
            self._counts_mid = self._counts_mid_full
            if (
                self.drop_isolated_box.isChecked()
                and self._counts_after_full is not None
            ):
                self._counts_after = self._counts_after_full
            else:
                self._counts_after = self._counts_mid_full
            self._overview_dt_us = self._overview_full_dt_us
            self._overview_t0_us = self._overview_full_t0_us
            self._overview_is_detail = False
            self._overview_applied_sig = self._overview_full_sig
            self._showing_activity = False
            self._request_full_frame()
            self._refresh_filter_views()
            self._set_preview_busy(False)
            self._set_status("ok", "Full recording.")
        else:
            self._overview_is_detail = False
            self._showing_activity = False
            self._request_full_frame()
            self._set_status(
                "work",
                f"Rebuilding full-file overview ({self._filter_brief()})…",
            )
            self._start_bin(self._overview_params(yellow=False), "overview")
        self._set_playhead_s(self._playhead_s)

    def _restag_overview_for_view(self) -> None:
        if self._store is None or self._overview is None:
            return
        if not self._dt_user_set:
            self._apply_suggested_dt(from_user=False)
        if not self._confirm_stack(self._preview_budget(), preview=True):
            return
        self._showing_activity = False
        self._crop_editing = False
        if self._crop_roi is not None:
            self._crop_roi.setVisible(False)
        self._sync_crop_buttons()
        self._request_full_frame()
        x0, x1 = self._region_range_s()
        self._set_time_view(x0, x1)
        if self._playhead_s < x0 or self._playhead_s > x1:
            self._set_playhead_s(x0)
        dt_ms = self._user_dt_us() / 1000.0
        self._set_status(
            "work",
            f"Rebuilding preview at Δt = {dt_ms:.3f} ms for "
            f"{x0:.4f}–{x1:.4f} s…",
        )
        self._start_bin(self._overview_params(yellow=True), "overview_view")

    def _on_send_as_changed(self, *_args) -> None:  # noqa: ANN001
        self._sync_send_widgets()
        self._refresh_plan_label()
        if self._store is None or self._counts_before is None:
            return
        self._set_status("work", "Updating preview (send as)…")
        self._view_timer.start()

    def _run_preview_view_update(self) -> None:
        try:
            self._refresh_filter_views()
            if self._showing_activity:
                self._show_activity_pair()
            self._set_status("ok", "Preview updated.")
        finally:
            busy = (
                self._bin_thread is not None and self._bin_thread.isRunning()
            )
            if not busy:
                self._set_preview_busy(False)

    def _on_neighbor_toggled(self, checked: bool) -> None:
        self._refresh_plan_label()
        if self._store is None or self._overview is None:
            return
        if not checked and self._counts_before is not None:
            self._counts_mid = self._counts_before
            self._mass_mid = self._mass_before
            if self._activity_before is not None:
                self._activity_mid = self._activity_before
            if self.drop_isolated_box.isChecked():
                self._start_spatial()
                return
            self._counts_after = self._counts_mid
            self._activity_after = self._activity_mid
            self._mass_after = self._mass_mid
            self._set_status("work", "Updating preview…")
            self._view_timer.start()
            return
        self._rebuild_preview_for_filters()

    def _activity_params(self) -> BinParams:
        assert self._store is not None
        t0, t1 = self._selected_window_us()
        window = max(1, t1 - t0)
        return BinParams(
            dt_us=window,
            polarity=self._gui_polarity(),
            accum=AccumMode.COUNT,
            t0_us=t0,
            t1_us=t1,
            max_frames=1,
            drop_isolated=self.drop_isolated_box.isChecked(),
            neighbor_dt_us=self._neighbor_dt_us(),
            spatial_bin=self._spatial_bin(),
        )

    def _sync_crop_buttons(self) -> None:
        ready = self._overview is not None
        self.activity_btn.setEnabled(ready)
        self.reset_crop_btn.setEnabled(ready)
        self.apply_crop_btn.setEnabled(ready and self._crop_editing)

    def _show_activity_crop(self) -> None:
        if self._store is None or self._overview is None:
            return
        self._crop_editing = True
        self._request_full_frame()
        self._sync_crop_buttons()
        exp_h, exp_w = spatial_out_size(
            int(self._store.height), int(self._store.width), self._spatial_bin()
        )
        if (
            self._activity_before is not None
            and self._activity_mid is not None
            and self._activity_before.shape[1:3] == (exp_h, exp_w)
        ):
            self._show_activity_pair()
            return
        self._set_status("work", "Building window activity image…")
        self._start_bin(self._activity_params(), "activity")

    def _inset_roi_size(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) for a default inset rectangle in stack pixels."""
        w, h = max(1, int(width)), max(1, int(height))
        mx = min(max(8, int(round(CROP_ROI_INSET * w))), max(0, w // 4))
        my = min(max(8, int(round(CROP_ROI_INSET * h))), max(0, h // 4))
        if w - 2 * mx < 1:
            mx = 0
        if h - 2 * my < 1:
            my = 0
        return mx, my, max(1, w - 2 * mx), max(1, h - 2 * my)

    def _ensure_crop_roi(self, *, show: bool, width: int, height: int) -> None:
        if self._store is None:
            return
        w, h = max(1, int(width)), max(1, int(height))
        fresh = False
        if self._crop_roi is None:
            ix, iy, iw, ih = self._inset_roi_size(w, h)
            self._crop_roi = pg.RectROI(
                [ix, iy],
                [iw, ih],
                pen=pg.mkPen("#32cd32", width=2),
                hoverPen=pg.mkPen("#7cfc00", width=2),
                rotatable=False,
                sideScalers=True,
            )
            self.preview.view.addItem(self._crop_roi)
            self._crop_roi.sigRegionChangeFinished.connect(self._on_crop_roi_finished)
            fresh = True
        if not show:
            self._crop_roi.setVisible(False)
            return
        if not fresh and self._crop_roi.isVisible():
            self._crop_roi.setZValue(20)
            return
        sb = self._spatial_bin()
        self._syncing_crop = True
        self._crop_roi.blockSignals(True)
        try:
            if self._crop_box is None:
                ix, iy, iw, ih = self._inset_roi_size(w, h)
                self._crop_roi.setPos((ix, iy), update=False)
                self._crop_roi.setSize((iw, ih), update=True)
            else:
                x0, y0, x1, y1 = self._crop_box
                dx0, dy0 = x0 // sb, y0 // sb
                dx1, dy1 = min(w, x1 // sb), min(h, y1 // sb)
                dw, dh = max(1, dx1 - dx0), max(1, dy1 - dy0)
                self._crop_roi.setPos((dx0, dy0), update=False)
                self._crop_roi.setSize((dw, dh), update=True)
            self._crop_roi.setVisible(True)
            self._crop_roi.setZValue(20)
        finally:
            self._crop_roi.blockSignals(False)
            self._syncing_crop = False

    def _sensor_box_from_roi(
        self, stack_w: int, stack_h: int
    ) -> tuple[int, int, int, int] | None:
        if self._crop_roi is None or self._store is None:
            return None
        pos = self._crop_roi.pos()
        size = self._crop_roi.size()
        x0 = int(np.floor(float(pos.x())))
        y0 = int(np.floor(float(pos.y())))
        x1 = int(np.ceil(float(pos.x()) + float(size.x())))
        y1 = int(np.ceil(float(pos.y()) + float(size.y())))
        x0 = max(0, min(stack_w, x0))
        x1 = max(0, min(stack_w, x1))
        y0 = max(0, min(stack_h, y0))
        y1 = max(0, min(stack_h, y1))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        if x0 <= 0 and y0 <= 0 and x1 >= stack_w and y1 >= stack_h:
            return None
        sb = self._spatial_bin()
        sx0, sy0 = x0 * sb, y0 * sb
        sx1 = min(int(self._store.width), x1 * sb)
        sy1 = min(int(self._store.height), y1 * sb)
        if sx1 - sx0 < 1 or sy1 - sy0 < 1:
            return None
        return sx0, sy0, sx1, sy1

    def _on_crop_roi_finished(self) -> None:
        if self._syncing_crop or not self._crop_editing:
            return
        self._refresh_plan_label()

    def _apply_crop(self) -> None:
        if not self._crop_editing or self._store is None:
            return
        src = (
            self._activity_before if self._showing_activity else self._counts_before
        )
        if src is None:
            return
        sh, sw = int(src.shape[1]), int(src.shape[2])
        self._crop_box = self._sensor_box_from_roi(sw, sh)
        self._crop_editing = False
        self._request_full_frame()
        self._sync_crop_buttons()
        if self._showing_activity:
            self._show_activity_pair()
        else:
            self._show_overview_stack()
        self._refresh_plan_label()
        if self._crop_box is None:
            self._set_status("ok", "Crop is the full sensor.")
        else:
            x0, y0, x1, y1 = self._crop_box
            oh, ow = self._send_hw()
            self._set_status(
                "ok",
                f"Crop applied: {x1 - x0}×{y1 - y0} sensor → {ow}×{oh} send.",
            )

    def _reset_crop(self) -> None:
        self._crop_box = None
        self._crop_editing = False
        self._request_full_frame()
        if self._crop_roi is not None:
            self._crop_roi.setVisible(False)
        self._sync_crop_buttons()
        if self._showing_activity:
            self._show_activity_pair()
        elif self._overview is not None:
            self._show_overview_stack()
        self._refresh_plan_label()
        self._set_status("ok", "Crop reset — full sensor.")

    def _remove_crop_roi(self) -> None:
        if self._crop_roi is None:
            return
        self.preview.view.removeItem(self._crop_roi)
        self._crop_roi = None

    def closeEvent(self, event) -> None:  # noqa: N802
        self.led._set_busy_cursor(False)
        for thread in (self._bin_thread, self._load_thread):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(10_000)
        super().closeEvent(event)

