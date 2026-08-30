"""PyQt6 event-reader UI: overview → range → Δt → send to BLITZ."""

from __future__ import annotations

import logging
import time
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStatusBar,
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
    activity_preview,
    bin_events,
    encode_stack_for_send,
    even_odd_row_ratio,
    event_rate_ms,
    plan_pictures,
)
from .evt3 import EventStore, load_evt3_raw
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
    finished = pyqtSignal(object, float, str)
    failed = pyqtSignal(str)

    def __init__(self, store: EventStore, params: BinParams, kind: str) -> None:
        super().__init__()
        self.store = store
        self.params = params
        self.kind = kind

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            stack = bin_events(self.store, self.params)
            elapsed = time.perf_counter() - t0
            log.info("%s binned shape=%s in %.2fs", self.kind, stack.shape, elapsed)
            self.finished.emit(stack, elapsed, self.kind)
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
    """Full-width status banner (load / bin / BLITZ). Replaces the tiny traffic light."""

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(64)
        self.kind = "idle"
        self._cursor_overridden = False
        self._pulse_on = True

        col = QVBoxLayout(self)
        col.setContentsMargins(12, 8, 12, 8)
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
        self.setMinimumHeight(92)
        col = QVBoxLayout(self)
        col.setContentsMargins(12, 10, 12, 10)
        col.setSpacing(6)
        self.headline = QLabel("RAM")
        head_font = QFont(self.headline.font())
        head_font.setBold(True)
        head_font.setPointSize(head_font.pointSize() + 4)
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
            f"~{fmt_bytes(budget.build_bytes)} to build (float32)  ·  "
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
        self._overview_full: np.ndarray | None = None
        self._overview_dt_us = 1
        self._overview_full_dt_us = 1
        self._overview_t0_us = 0
        self._overview_full_t0_us = 0
        self._overview_is_detail = False
        self._overview_applied_sig: tuple | None = None
        self._overview_full_sig: tuple | None = None
        self._showing_activity = False
        self._crop_roi: pg.RectROI | None = None
        self._crop_roi_wanted = False
        self._load_thread: QThread | None = None
        self._load_worker: _LoadWorker | None = None
        self._bin_thread: QThread | None = None
        self._bin_worker: _BinWorker | None = None
        self._bin_busy_kind: str | None = None
        self._bin_inflight_params: BinParams | None = None
        self._pending_bin: tuple[BinParams, str] | None = None
        self._blitz_clients = 0
        self._syncing_range = False
        self._playhead_s = 0.0
        self._stack_budget: StackBudget | None = None
        self._rate_zoom_timer = QTimer(self)
        self._rate_zoom_timer.setSingleShot(True)
        self._rate_zoom_timer.setInterval(120)
        self._rate_zoom_timer.timeout.connect(self._refresh_rate_curve_from_view)
        self._filter_preview_timer = QTimer(self)
        self._filter_preview_timer.setSingleShot(True)
        self._filter_preview_timer.setInterval(280)
        self._filter_preview_timer.timeout.connect(self._rebuild_preview_for_filters)

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

        self.led = _Led()
        layout.addWidget(self.led)
        self.meta_label = QLabel("No file loaded.")
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)
        self.file_preview = QPlainTextEdit()
        self.file_preview.setReadOnly(True)
        self.file_preview.setMaximumHeight(150)
        self.file_preview.setPlaceholderText("RAW header and first events appear here.")
        self.file_preview.setFont(QFont("monospace", 9))
        self.file_preview.setAcceptDrops(False)
        layout.addWidget(self.file_preview)

        step1 = QGroupBox("1 — Overview (local only, not sent to BLITZ)")
        s1 = QVBoxLayout(step1)
        s1.addWidget(QLabel(
            "Wheel or the white playhead scrubs pictures (like BLITZ). "
            "Ctrl+wheel or right-drag zooms time; double-click resets to the "
            "full file. Yellow band = the range you will send. "
            "O rebuilds the overview for that yellow range at the Δt below "
            "(same pictures BLITZ will get) and zooms the timeline to it. "
            "M shows a window activity image so you can set "
            "a crop rectangle."
        ))
        self.preview = pg.ImageView()
        self.preview.setAcceptDrops(False)
        self.preview.ui.roiBtn.hide()
        self.preview.ui.menuBtn.hide()
        self.preview.ui.histogram.hide()
        self.preview.ui.roiPlot.hide()
        self.preview.setMinimumHeight(280)
        s1.addWidget(self.preview, stretch=1)

        self.playhead_label = QLabel("t = —")
        s1.addWidget(self.playhead_label)

        self.time_plot = pg.PlotWidget()
        self.time_plot.setAcceptDrops(False)
        self.time_plot.setMinimumHeight(110)
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
        self.rate_curve = self.time_plot.plot(pen=pg.mkPen("#7ec8e3", width=1))
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
        self.playhead.sigPositionChanged.connect(self._on_playhead_moved)
        self.time_plot.scene().sigMouseClicked.connect(self._on_timeline_clicked)
        self.time_plot.installEventFilter(self)
        self.time_plot.viewport().installEventFilter(self)
        self.preview.sigTimeChanged.connect(self._on_preview_index)
        s1.addWidget(self.time_plot)
        ov_row = QHBoxLayout()
        self.restag_btn = QPushButton("Rebuild overview for selection (O)")
        self.restag_btn.setEnabled(False)
        self.restag_btn.setToolTip(
            "Zoom to the yellow band and re-bin the overview at the current "
            "Δt — the same pictures you will send to BLITZ. "
            "The coarse full-file overview stays cached (double-click the plot)."
        )
        self.restag_btn.clicked.connect(self._restag_overview_for_view)
        self.activity_btn = QPushButton("Window max / set crop (M)")
        self.activity_btn.setEnabled(False)
        self.activity_btn.setToolTip(
            "One picture of all events in the yellow band. Drag the green "
            "rectangle to crop the send (same idea as a BLITZ load ROI)."
        )
        self.activity_btn.clicked.connect(self._show_activity_crop)
        self.reset_crop_btn = QPushButton("Reset crop")
        self.reset_crop_btn.setEnabled(False)
        self.reset_crop_btn.clicked.connect(self._reset_crop)
        ov_row.addWidget(self.restag_btn)
        ov_row.addWidget(self.activity_btn)
        ov_row.addWidget(self.reset_crop_btn)
        s1.addLayout(ov_row)
        layout.addWidget(step1, stretch=1)

        step2 = QGroupBox("2 — Frame time for the selected range")
        s2 = QVBoxLayout(step2)
        form = QFormLayout()
        self.dt_ms = QDoubleSpinBox()
        self.dt_ms.setRange(SENSOR_DT_US / 1000.0, 1_000_000.0)
        self.dt_ms.setDecimals(3)
        self.dt_ms.setSingleStep(0.01)
        self.dt_ms.setValue(1.0)
        self.dt_ms.setSuffix(" ms")
        self.dt_ms.setToolTip(
            "How long each picture integrates — preview rebuild and BLITZ send "
            "use this same Δt. Sensor timestamps step by 1 µs (0.001 ms). "
            "Use suggested Δt for ~150 pictures in the yellow band, or type "
            "your own. Picture count is limited only by RAM (yellow/red below)."
        )
        self.dt_ms.valueChanged.connect(self._refresh_plan_label)
        self.suggest_dt_btn = QPushButton("Use suggested")
        self.suggest_dt_btn.setEnabled(False)
        self.suggest_dt_btn.setToolTip(
            "Set Δt so the yellow band becomes about 150 pictures. "
            "You can still type a finer or coarser value."
        )
        self.suggest_dt_btn.clicked.connect(self._apply_suggested_dt)
        dt_wrap = QWidget()
        dt_row = QHBoxLayout(dt_wrap)
        dt_row.setContentsMargins(0, 0, 0, 0)
        dt_row.addWidget(self.dt_ms)
        dt_row.addWidget(self.suggest_dt_btn)
        form.addRow("Frame time (Δt)", dt_wrap)
        self.min_dt_label = QLabel(
            "Sensor timestamps at 1 µs (relative to the start of this file, "
            "not wall-clock). Stack size is a soft RAM limit — see the bar."
        )
        self.min_dt_label.setWordWrap(True)
        form.addRow(self.min_dt_label)
        self.polarity = QComboBox()
        for mode in PolarityMode:
            self.polarity.addItem(mode.value, mode)
        self.polarity.setCurrentIndex(2)
        self.polarity.currentIndexChanged.connect(self._on_polarity_changed)
        form.addRow("Polarity", self.polarity)
        self.eight_bit = QCheckBox("8-bit")
        self.eight_bit.setChecked(False)
        self.eight_bit.setToolTip(
            "Off (default): send float32 event counts to BLITZ. "
            "On: pack to uint8 here (same idea as the BLITZ File tab)."
        )
        self.eight_bit.toggled.connect(self._on_encode_options_changed)
        self.normalize_box = QCheckBox("Normalize")
        self.normalize_box.setChecked(False)
        self.normalize_box.setToolTip(
            "Per-picture min–max stretch. Same idea as the BLITZ File tab. "
            "BLITZ can still apply File-tab options again on Connect."
        )
        self.normalize_box.toggled.connect(self._refresh_plan_label)
        self.grayscale_box = QCheckBox("Grayscale")
        self.grayscale_box.setChecked(True)
        self.grayscale_box.setToolTip(
            "Event pictures are already one channel; this matches the File tab "
            "and only changes RGB stacks."
        )
        encode_wrap = QWidget()
        encode_row = QHBoxLayout(encode_wrap)
        encode_row.setContentsMargins(0, 0, 0, 0)
        encode_row.addWidget(self.eight_bit)
        encode_row.addWidget(self.normalize_box)
        encode_row.addWidget(self.grayscale_box)
        form.addRow("Send like File tab", encode_wrap)
        self.log_stretch = QCheckBox("Log stretch (log1p → 0…255)")
        self.log_stretch.setChecked(False)
        self.log_stretch.setEnabled(False)
        self.log_stretch.setToolTip(
            "Only with 8-bit. Squeeze outliers into 0…255 so a few hot pixels "
            "do not crush typical counts."
        )
        form.addRow(self.log_stretch)
        noise_wrap = QWidget()
        noise_col = QVBoxLayout(noise_wrap)
        noise_col.setContentsMargins(0, 0, 0, 0)
        noise_col.setSpacing(4)
        self.drop_isolated_box = QCheckBox("1-pixel spatial")
        self.drop_isolated_box.setChecked(False)
        self.drop_isolated_box.setToolTip(
            "After binning: zero pixels that have a count but all 8 neighbours "
            "are empty. Optional — a real 1-pixel event is removed too. "
            "Updates the local preview immediately; send uses the same setting."
        )
        self.drop_isolated_box.toggled.connect(self._on_filter_checkbox)
        self.neighbor_box = QCheckBox("Temporal neighbour")
        self.neighbor_box.setChecked(False)
        self.neighbor_box.setToolTip(
            "Before binning: keep an event only if a pixel in its 3×3 "
            "neighbourhood already fired within Δt. Isolated salt-and-pepper "
            "events drop. Updates the local preview; send uses the same setting."
        )
        self.neighbor_box.toggled.connect(self._on_neighbor_toggled)
        self.neighbor_dt_ms = QDoubleSpinBox()
        self.neighbor_dt_ms.setRange(SENSOR_DT_US / 1000.0, 1_000_000.0)
        self.neighbor_dt_ms.setDecimals(3)
        self.neighbor_dt_ms.setSingleStep(0.5)
        self.neighbor_dt_ms.setValue(3.0)
        self.neighbor_dt_ms.setSuffix(" ms")
        self.neighbor_dt_ms.setEnabled(False)
        self.neighbor_dt_ms.setToolTip(
            "Neighbourhood time window for the temporal filter. "
            "The preview rebuilds after a short pause while you edit."
        )
        self.neighbor_dt_ms.valueChanged.connect(self._on_neighbor_dt_changed)
        nn_row = QHBoxLayout()
        nn_row.setContentsMargins(0, 0, 0, 0)
        nn_row.addWidget(self.neighbor_box)
        nn_row.addWidget(self.neighbor_dt_ms)
        nn_row.addStretch(1)
        noise_col.addWidget(self.drop_isolated_box)
        noise_col.addLayout(nn_row)
        form.addRow("Noise filter (optional)", noise_wrap)
        s2.addLayout(form)
        self.ram_banner = _RamBanner()
        s2.addWidget(self.ram_banner)
        layout.addWidget(step2)

        step3 = QGroupBox("3 — Send to BLITZ (only when you click)")
        s3 = QVBoxLayout(step3)
        s3.addWidget(QLabel(
            "BLITZ does not update by itself while you scrub. Connect BLITZ "
            "Stream first, then send. Green status = BLITZ downloaded the stack."
        ))
        self.apply_btn = QPushButton("Build pictures and send to BLITZ")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._export_to_blitz)
        self.push_btn = QPushButton("Send last pictures again (no rebuild)")
        self.push_btn.setEnabled(False)
        self.push_btn.clicked.connect(self._repush)
        s3.addWidget(self.apply_btn)
        s3.addWidget(self.push_btn)
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
        layout.addWidget(step3)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"Serving {self.publisher.base_url}")

    def _update_connect_hint(self) -> None:
        self.connect_hint.setText(
            f"In BLITZ → <b>Stream</b> tab: address <b>{self.publisher.base_url}</b>, "
            f"token <b>{self.publisher.token}</b> → Connect. "
            "Then use the button above."
        )

    def _on_encode_options_changed(self) -> None:
        eight = self.eight_bit.isChecked()
        self.log_stretch.setEnabled(eight)
        if not eight:
            self.log_stretch.setChecked(False)
        self._refresh_plan_label()

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
        self.path_edit.setText(str(path))
        self.meta_label.setText("Decoding…")
        self._set_status("work", "1/3 Decoding recording…")
        self.apply_btn.setEnabled(False)
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
        self._overview_full = None
        self._overview_is_detail = False
        self._overview_applied_sig = None
        self._overview_full_sig = None
        self._showing_activity = False
        self._crop_roi_wanted = False
        self._remove_crop_roi()
        dur_s = max(store.duration_us / 1_000_000.0, 0.001)
        self.meta_label.setText(
            f"{store.width}×{store.height}  |  {len(store):,} events  |  "
            f"{dur_s:.3f} s (from first event in this file)"
        )
        x_ms, counts = event_rate_ms(store)
        self.rate_curve.setData(x_ms / 1000.0, counts)
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
        self.activity_btn.setEnabled(False)
        self.reset_crop_btn.setEnabled(False)
        self.suggest_dt_btn.setEnabled(False)
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
            dt_us, n, _u = plan_pictures(window_us, n_frames=OVERVIEW_PICTURES)
        return BinParams(
            dt_us=dt_us,
            polarity=self.polarity.currentData(),
            accum=AccumMode.COUNT,
            t0_us=t0,
            t1_us=t1,
            max_frames=n,
            drop_isolated=self.drop_isolated_box.isChecked(),
            neighbor_dt_us=self._neighbor_dt_us(),
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
            ):
                self._rebuild_preview_for_filters()
        self._playhead_s = t_s
        self._syncing_range = True
        self.playhead.setValue(t_s)
        idx = 0
        if self._overview is not None and not self._showing_activity:
            idx = self._frame_at_s(t_s)
            self.preview.setCurrentIndex(idx)
        self._syncing_range = False
        if self._overview is not None:
            n = self._overview.shape[0]
            kind = "detail" if self._overview_is_detail else "overview"
            self.playhead_label.setText(
                f"t = {t_s:.4f} s   ·   {kind} picture {idx + 1} / {n}   "
                f"({self._overview_dt_us / 1000.0:.2f} ms per picture)"
            )
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
        itemsize = 1 if self.eight_bit.isChecked() else 4
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
        filt_txt = self._filter_brief()
        sug_us, sug_n, _u = plan_pictures(window_us, n_frames=OVERVIEW_PICTURES)
        self.min_dt_label.setText(
            f"Time in this file starts at 0 (first event), not the clock. "
            f"Timestamps step by {SENSOR_DT_US} µs. "
            f"Selected {t0_s:.4f}–{t1_s:.4f} s ({span_s:.3f} s). "
            f"Δt = {dt_us / 1000.0:.3f} ms → {n} pictures (preview rebuild "
            f"and BLITZ send use this). "
            f"Suggested for ~{OVERVIEW_PICTURES} pictures: "
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
            self.apply_btn.setText("Too big for free RAM — raise Δt or shrink range")
            self.apply_btn.setStyleSheet(
                "QPushButton { background:#6b0000; color:#ffffff; font-weight:bold; "
                "padding:8px; }"
            )
            return
        self.apply_btn.setEnabled(ready)
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

    def _suggest_dt_from_overview(self) -> None:
        self.dt_ms.blockSignals(True)
        self.dt_ms.setValue(self._overview_dt_us / 1000.0)
        self.dt_ms.blockSignals(False)
        self._refresh_plan_label()

    def _user_dt_us(self) -> int:
        return max(SENSOR_DT_US, int(round(self.dt_ms.value() * 1000.0)))

    def _neighbor_dt_us(self) -> int | None:
        if not self.neighbor_box.isChecked():
            return None
        return max(SENSOR_DT_US, int(round(self.neighbor_dt_ms.value() * 1000.0)))

    def _filter_sig(self) -> tuple:
        return (
            bool(self.drop_isolated_box.isChecked()),
            self._neighbor_dt_us(),
            self.polarity.currentData(),
        )

    def _filter_brief(self) -> str:
        parts: list[str] = []
        if self.drop_isolated_box.isChecked():
            parts.append("1-pixel")
        if self.neighbor_box.isChecked():
            parts.append(f"neighbour {self.neighbor_dt_ms.value():.3f} ms")
        return ("filters: " + ", ".join(parts)) if parts else "filters off"

    def _filter_status_suffix(self) -> str:
        brief = self._filter_brief()
        if brief == "filters off":
            return ""
        return f" {brief}."

    def _on_filter_checkbox(self, *_args) -> None:  # noqa: ANN001
        self._filter_preview_timer.stop()
        self._refresh_plan_label()
        self._rebuild_preview_for_filters()

    def _on_neighbor_dt_changed(self, *_args) -> None:  # noqa: ANN001
        self._refresh_plan_label()
        if not self.neighbor_box.isChecked():
            return
        if self._store is None or self._overview is None:
            return
        self._filter_preview_timer.start()

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
        self._set_status("work", f"Updating preview ({brief})…")
        self._start_bin(self._overview_params(yellow=yellow), kind)

    def _apply_suggested_dt(self) -> None:
        if self._store is None:
            return
        t0, t1 = self._selected_window_us()
        dt_us, _n, _u = plan_pictures(t1 - t0, n_frames=OVERVIEW_PICTURES)
        self.dt_ms.setValue(dt_us / 1000.0)

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
        itemsize = 1 if self.eight_bit.isChecked() else 4
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
        thread = QThread(self)
        worker = _BinWorker(self._store, params, kind)
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

    def _hide_preview_extras(self) -> None:
        self.preview.ui.roiPlot.hide()
        self.preview.ui.histogram.hide()

    def _show_overview_stack(self) -> None:
        if self._overview is None:
            return
        self._showing_activity = False
        self._syncing_range = True
        self.preview.setImage(
            self._overview, autoLevels=True, axes={"t": 0, "y": 1, "x": 2}
        )
        self._hide_preview_extras()
        idx = self._frame_at_s(self._playhead_s)
        self.preview.setCurrentIndex(idx)
        self._syncing_range = False
        if self._crop_roi_wanted:
            self._ensure_crop_roi(show=True)

    def _on_binned(self, stack: object, elapsed: float, kind: str) -> None:
        arr = np.asarray(stack)
        params = self._bin_inflight_params
        if kind == "activity":
            disp, lo, hi, n_pos = activity_preview(arr)
            self._showing_activity = True
            self._syncing_range = True
            self.preview.setImage(
                disp, autoLevels=False, axes={"t": 0, "y": 1, "x": 2}
            )
            self.preview.getImageItem().setLevels((lo, hi))
            self._hide_preview_extras()
            self._syncing_range = False
            self._ensure_crop_roi(show=True)
            if n_pos == 0:
                self._set_status(
                    "wait",
                    "Window max is empty — no events in the yellow band. "
                    "Widen the band, then press M again.",
                )
            else:
                self._set_status(
                    "ok",
                    f"Window max ({n_pos:,} lit pixels, log counts)."
                    f"{self._filter_status_suffix()} "
                    "Drag the green rectangle to crop before send. "
                    "Scrub the timeline to return to the overview.",
                )
            self._refresh_plan_label()
            return

        net = encode_stack_for_send(
            arr,
            eight_bit=self.eight_bit.isChecked(),
            log_stretch=self.log_stretch.isChecked(),
            normalize=self.normalize_box.isChecked(),
            grayscale=self.grayscale_box.isChecked(),
        )
        if kind in ("overview", "overview_view"):
            first_full = self._overview_full is None
            t0 = int(self._store.t_min) if self._store is not None else 0
            dt = params.dt_us if params is not None else self._overview_dt_us
            if params is not None and params.t0_us is not None:
                t0 = int(params.t0_us)
            self._overview = net
            self._overview_dt_us = dt
            self._overview_t0_us = t0
            if kind == "overview":
                self._overview_full = net
                self._overview_full_dt_us = dt
                self._overview_full_t0_us = t0
                self._overview_is_detail = False
            else:
                self._overview_is_detail = True
            self._overview_applied_sig = self._filter_sig()
            if kind == "overview":
                self._overview_full_sig = self._overview_applied_sig
            n = net.shape[0]
            self.restag_btn.setEnabled(True)
            self.activity_btn.setEnabled(True)
            self.reset_crop_btn.setEnabled(True)
            self.suggest_dt_btn.setEnabled(True)
            if first_full and kind == "overview":
                self._suggest_dt_from_overview()
            self._show_overview_stack()
            if first_full and kind == "overview":
                self._set_playhead_s(0.0)
            self.apply_btn.setEnabled(True)
            if kind == "overview_view":
                dt_ms = self._overview_dt_us / 1000.0
                hint = self._interlace_hint(arr)
                self._set_status(
                    "ok",
                    f"Preview ready ({n} pictures, {dt_ms:.3f} ms each) — "
                    f"same Δt as a BLITZ send.{self._filter_status_suffix()}"
                    f"{hint} "
                    "Scrub here to inspect; send when it looks right.",
                )
            else:
                self._set_status(
                    "ok",
                    f"Overview ready ({n} pictures)."
                    f"{self._filter_status_suffix()} "
                    "Scrub the timeline, set the yellow start/end, choose Δt, "
                    "then send to BLITZ.",
                )
            self._refresh_plan_label()
            return

        self.publisher.set_stack(net, push=True)
        self.push_btn.setEnabled(True)
        hint = self._interlace_hint(arr)
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
        if (
            self._store is None
            or not self._crop_roi_wanted
            or self._crop_roi is None
            or not self._crop_roi.isVisible()
        ):
            return None
        w, h = int(self._store.width), int(self._store.height)
        pos = self._crop_roi.pos()
        size = self._crop_roi.size()
        x0 = int(np.floor(float(pos.x())))
        y0 = int(np.floor(float(pos.y())))
        x1 = int(np.ceil(float(pos.x()) + float(size.x())))
        y1 = int(np.ceil(float(pos.y()) + float(size.y())))
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        if x0 <= 1 and y0 <= 1 and x1 >= w - 1 and y1 >= h - 1:
            return None
        return x0, y0, x1, y1

    def _send_hw(self) -> tuple[int, int]:
        assert self._store is not None
        crop = self._crop_xyxy()
        if crop is None:
            return int(self._store.height), int(self._store.width)
        x0, y0, x1, y1 = crop
        return y1 - y0, x1 - x0

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
            polarity=self.polarity.currentData(),
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
        )

    def _export_to_blitz(self) -> None:
        if self._store is None:
            return
        self._refresh_plan_label()
        if not self._confirm_stack(self._stack_budget, preview=False):
            return
        self._set_status("work", "2/3 Building pictures for BLITZ…")
        self._start_bin(self._export_params(), "export")

    def _repush(self) -> None:
        self.publisher.push()
        self._set_status(
            "wait",
            "Sending last pictures again…"
            if self._blitz_clients
            else "Waiting for BLITZ Stream — Connect in BLITZ → Stream."
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
        sc_m = QShortcut(QKeySequence("M"), self)
        sc_m.activated.connect(self._shortcut_activity)

    def _shortcut_restag(self) -> None:
        if self._shortcut_ok():
            self._restag_overview_for_view()

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
        if (
            self._overview_full is not None
            and self._overview_full_sig == self._filter_sig()
        ):
            self._overview = self._overview_full
            self._overview_dt_us = self._overview_full_dt_us
            self._overview_t0_us = self._overview_full_t0_us
            self._overview_is_detail = False
            self._overview_applied_sig = self._overview_full_sig
            self._show_overview_stack()
        elif self._store is not None:
            self._overview_is_detail = False
            self._set_status(
                "work",
                f"Rebuilding full-file overview ({self._filter_brief()})…",
            )
            self._start_bin(self._overview_params(yellow=False), "overview")
        self._set_playhead_s(self._playhead_s)

    def _restag_overview_for_view(self) -> None:
        if self._store is None or self._overview is None:
            return
        if not self._confirm_stack(self._preview_budget(), preview=True):
            return
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

    def _on_polarity_changed(self, *_args) -> None:  # noqa: ANN001
        if self._store is None or self._overview is None:
            return
        if self._overview_is_detail:
            self._restag_overview_for_view()
        else:
            self._set_status("work", "Rebuilding overview for this polarity…")
            self._start_bin(self._overview_params(yellow=False), "overview")

    def _on_neighbor_toggled(self, checked: bool) -> None:
        self.neighbor_dt_ms.setEnabled(bool(checked))
        self._on_filter_checkbox()

    def _activity_params(self) -> BinParams:
        assert self._store is not None
        t0, t1 = self._selected_window_us()
        window = max(1, t1 - t0)
        return BinParams(
            dt_us=window,
            polarity=self.polarity.currentData(),
            accum=AccumMode.COUNT,
            t0_us=t0,
            t1_us=t1,
            max_frames=1,
            drop_isolated=self.drop_isolated_box.isChecked(),
            neighbor_dt_us=self._neighbor_dt_us(),
        )

    def _show_activity_crop(self) -> None:
        if self._store is None or self._overview is None:
            return
        self._set_status("work", "Building window activity image…")
        self._start_bin(self._activity_params(), "activity")

    def _ensure_crop_roi(self, *, show: bool) -> None:
        if self._store is None:
            return
        w, h = int(self._store.width), int(self._store.height)
        if self._crop_roi is None:
            self._crop_roi = pg.RectROI(
                [0, 0],
                [w, h],
                pen=pg.mkPen("#32cd32", width=2),
                hoverPen=pg.mkPen("#7cfc00", width=2),
                rotatable=False,
                sideScalers=True,
            )
            self.preview.view.addItem(self._crop_roi)
            self._crop_roi.sigRegionChangeFinished.connect(self._on_crop_roi_finished)
        self._crop_roi_wanted = show
        self._crop_roi.setVisible(show)
        if show:
            self._crop_roi.setZValue(20)

    def _on_crop_roi_finished(self) -> None:
        self._refresh_plan_label()

    def _reset_crop(self) -> None:
        self._crop_roi_wanted = False
        if self._crop_roi is not None and self._store is not None:
            self._crop_roi.setPos((0, 0), update=False)
            self._crop_roi.setSize(
                (int(self._store.width), int(self._store.height)), update=True
            )
            self._crop_roi.setVisible(False)
        self._refresh_plan_label()

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

