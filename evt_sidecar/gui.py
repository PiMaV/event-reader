"""PyQt6 event-reader UI: overview → range → Δt → send to BLITZ."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QWheelEvent
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
    OVERVIEW_PICTURES,
    SENSOR_DT_US,
    AccumMode,
    BinParams,
    PolarityMode,
    bin_events,
    encode_stack_for_send,
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
            "Open a RAW file. This bar shows decoding, send, and BLITZ Stream status."
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
        self.resize(720, 860)

        self._store: EventStore | None = None
        self._overview: np.ndarray | None = None
        self._overview_dt_us = 1
        self._load_thread: QThread | None = None
        self._load_worker: _LoadWorker | None = None
        self._bin_thread: QThread | None = None
        self._bin_worker: _BinWorker | None = None
        self._bin_busy_kind: str | None = None
        self._pending_export = False
        self._blitz_clients = 0
        self._syncing_range = False
        self._playhead_s = 0.0
        self._stack_budget: StackBudget | None = None

        self._bridge = _NetBridge(self)
        self._bridge.served.connect(self._on_blitz_downloaded)
        self._bridge.clients.connect(self._on_client_count)

        self.publisher = StackPublisher(host=host, port=port, token=token)
        self.publisher.on_served = lambda n: self._bridge.served.emit(n)
        self.publisher.on_client_count = lambda n: self._bridge.clients.emit(n)
        self.publisher.start_background()

        self._build_ui()
        self._update_connect_hint()
        if initial_raw is not None:
            from PyQt6.QtCore import QTimer

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
        self.path_edit.setPlaceholderText("Event-camera EVT3 .raw …")
        self.path_edit.setReadOnly(True)
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
        layout.addWidget(self.file_preview)

        step1 = QGroupBox("1 — Overview (local only, not sent to BLITZ)")
        s1 = QVBoxLayout(step1)
        s1.addWidget(QLabel(
            "Scrub the pictures with the mouse on the timeline under the image "
            "(like BLITZ). Yellow band = start → end of the range you will send. "
            "White line = the picture you are looking at."
        ))
        self.preview = pg.ImageView()
        self.preview.ui.roiBtn.hide()
        self.preview.ui.menuBtn.hide()
        self.preview.ui.histogram.hide()
        self.preview.ui.roiPlot.hide()
        self.preview.setMinimumHeight(280)
        s1.addWidget(self.preview, stretch=1)

        self.playhead_label = QLabel("t = —")
        s1.addWidget(self.playhead_label)

        self.time_plot = pg.PlotWidget()
        self.time_plot.setMinimumHeight(110)
        self.time_plot.setMaximumHeight(140)
        self.time_plot.setLabel("bottom", "Time", units="s")
        self.time_plot.setLabel("left", "Events")
        self.time_plot.showGrid(x=True, y=True, alpha=0.2)
        self.time_plot.setMouseEnabled(x=False, y=False)
        self.rate_curve = self.time_plot.plot(pen=pg.mkPen("#7ec8e3", width=1))
        self.region = pg.LinearRegionItem([0.0, 1.0], brush=(230, 180, 40, 55))
        self.region.setZValue(0)
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
        layout.addWidget(step1, stretch=1)

        step2 = QGroupBox("2 — Frame time for the selected range")
        s2 = QVBoxLayout(step2)
        form = QFormLayout()
        self.dt_ms = QDoubleSpinBox()
        self.dt_ms.setRange(SENSOR_DT_US / 1000.0, 1_000_000.0)
        self.dt_ms.setDecimals(3)
        self.dt_ms.setSingleStep(0.1)
        self.dt_ms.setValue(1.0)
        self.dt_ms.setSuffix(" ms")
        self.dt_ms.setToolTip(
            "How long each picture integrates. Sensor timestamps step by 1 µs "
            "(0.001 ms). Picture count is limited only by RAM (yellow/red below)."
        )
        self.dt_ms.valueChanged.connect(self._refresh_plan_label)
        form.addRow("Frame time (Δt)", self.dt_ms)
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
        self.time_plot.setXRange(0.0, dur_s, padding=0.02)
        self.file_preview.setPlainText(_file_preview_text(store))
        self._set_status("work", f"1/3 Building ~{OVERVIEW_PICTURES}-picture overview…")
        self._start_bin(self._overview_params(), "overview")

    def _overview_params(self) -> BinParams:
        assert self._store is not None
        window_us = max(1, self._store.duration_us)
        dt_us, n, _u = plan_pictures(window_us, n_frames=OVERVIEW_PICTURES)
        self._overview_dt_us = dt_us
        return BinParams(
            dt_us=dt_us,
            polarity=self.polarity.currentData(),
            accum=AccumMode.COUNT,
            t0_us=self._store.t_min,
            t1_us=self._store.t_max,
            max_frames=n,
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

    def _duration_s(self) -> float:
        if self._store is None:
            return 0.001
        return max(self._store.duration_us / 1_000_000.0, 0.001)

    def _frame_at_s(self, t_s: float) -> int:
        if self._overview is None:
            return 0
        n = self._overview.shape[0]
        idx = int(round(t_s * 1_000_000.0 / max(1, self._overview_dt_us)))
        return max(0, min(n - 1, idx))

    def _set_playhead_s(self, t_s: float) -> None:
        dur = self._duration_s()
        t_s = max(0.0, min(dur, t_s))
        self._playhead_s = t_s
        self._syncing_range = True
        self.playhead.setValue(t_s)
        if self._overview is not None:
            idx = self._frame_at_s(t_s)
            self.preview.setCurrentIndex(idx)
        self._syncing_range = False
        if self._overview is not None:
            n = self._overview.shape[0]
            self.playhead_label.setText(
                f"t = {t_s:.4f} s   ·   overview picture {idx + 1} / {n}   "
                f"({self._overview_dt_us / 1000.0:.2f} ms per overview picture)"
            )
        else:
            self.playhead_label.setText(f"t = {t_s:.4f} s")

    def _on_playhead_moved(self) -> None:
        if self._syncing_range:
            return
        self._set_playhead_s(float(self.playhead.value()))

    def _on_preview_index(self, ind, _time) -> None:  # noqa: ANN001
        if self._syncing_range or self._overview is None:
            return
        t_s = int(ind) * self._overview_dt_us / 1_000_000.0
        self._set_playhead_s(t_s)

    def _on_timeline_clicked(self, event) -> None:  # noqa: ANN001
        if event.button() != Qt.MouseButton.LeftButton:
            return
        vb = self.time_plot.plotItem.vb
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        x = vb.mapSceneToView(event.scenePos()).x()
        self._set_playhead_s(x)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        watched = {self.time_plot, self.time_plot.viewport()}
        if obj in watched and event.type() == QEvent.Type.Wheel:
            wheel = event
            if isinstance(wheel, QWheelEvent) and self._overview is not None:
                step = 1 if wheel.angleDelta().y() < 0 else -1
                idx = self._frame_at_s(self._playhead_s) + step
                t_s = idx * self._overview_dt_us / 1_000_000.0
                self._set_playhead_s(t_s)
                return True
        return super().eventFilter(obj, event)

    def _refresh_plan_label(self) -> None:
        if self._store is None:
            return
        t0, t1 = self._selected_window_us()
        window_us = t1 - t0
        dt_us = max(SENSOR_DT_US, int(round(self.dt_ms.value() * 1000.0)))
        dt_us, n, _used_us = plan_pictures(window_us, dt_us=dt_us)
        span_s = window_us / 1_000_000.0
        t0_s = (t0 - self._store.t_min) / 1_000_000.0
        t1_s = (t1 - self._store.t_min) / 1_000_000.0
        ram = read_ram()
        itemsize = 1 if self.eight_bit.isChecked() else 4
        budget = assess_stack(
            n, self._store.height, self._store.width, ram, wire_itemsize=itemsize
        )
        self._stack_budget = budget
        y_b = YELLOW_FRAC * ram.total
        r_b = RED_FRAC * ram.total
        self.min_dt_label.setText(
            f"Time in this file starts at 0 (first event), not the clock. "
            f"Timestamps step by {SENSOR_DT_US} µs. "
            f"Selected {t0_s:.4f}–{t1_s:.4f} s ({span_s:.3f} s). "
            f"Δt = {dt_us / 1000.0:.3f} ms. "
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

    def _start_bin(self, params: BinParams, kind: str) -> None:
        if self._store is None:
            return
        if self._bin_thread is not None and self._bin_thread.isRunning():
            if kind == "export":
                self._pending_export = True
            return
        self._pending_export = False
        self._bin_busy_kind = kind
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
        if self._pending_export:
            self._pending_export = False
            self._export_to_blitz()

    def _on_binned(self, stack: object, elapsed: float, kind: str) -> None:
        arr = np.asarray(stack)
        net = encode_stack_for_send(
            arr,
            eight_bit=self.eight_bit.isChecked(),
            log_stretch=self.log_stretch.isChecked(),
            normalize=self.normalize_box.isChecked(),
            grayscale=self.grayscale_box.isChecked(),
        )
        if kind == "overview":
            self._overview = net
            n = net.shape[0]
            self.preview.setImage(net, autoLevels=True, axes={"t": 0, "y": 1, "x": 2})
            self.preview.ui.roiPlot.hide()
            self.preview.ui.histogram.hide()
            self._suggest_dt_from_overview()
            self.apply_btn.setEnabled(True)
            self._set_playhead_s(0.0)
            self._set_status(
                "ok",
                f"Overview ready ({n} pictures). Scrub the timeline, set the yellow "
                "start/end, choose Δt, then send to BLITZ.",
            )
            self._refresh_plan_label()
            return
        self.publisher.set_stack(net, push=True)
        self.push_btn.setEnabled(True)
        if self._blitz_clients <= 0:
            self._set_status(
                "wait",
                "3/3 Pictures ready — connect BLITZ Stream, or wait for download…",
            )
        else:
            self._set_status("wait", "3/3 Sending to BLITZ…")
        self.statusBar().showMessage(f"Sent {tuple(net.shape)} in {elapsed * 1000:.0f} ms")

    def _export_params(self) -> BinParams:
        assert self._store is not None
        t0, t1 = self._selected_window_us()
        dt_us = max(SENSOR_DT_US, int(round(self.dt_ms.value() * 1000.0)))
        dt_us, n, _u = plan_pictures(t1 - t0, dt_us=dt_us)
        return BinParams(
            dt_us=dt_us,
            polarity=self.polarity.currentData(),
            accum=AccumMode.COUNT,
            t0_us=t0,
            t1_us=t0 + n * dt_us,
            max_frames=n,
        )

    def _export_to_blitz(self) -> None:
        if self._store is None:
            return
        self._refresh_plan_label()
        budget = self._stack_budget
        if budget is not None and budget.level == "block":
            QMessageBox.critical(
                self,
                "Not enough RAM",
                f"{budget.n_frames} pictures = {fmt_bytes(budget.wire_bytes)} "
                f"on the wire, but only {fmt_bytes(read_ram().available)} is free. "
                "Raise Δt or shrink the yellow band.",
            )
            return
        if budget is not None and (
            budget.level == "red" or budget.nav_warn
        ):
            parts: list[str] = []
            if budget.nav_warn:
                parts.append(
                    f"{budget.n_frames} pictures — more than {NAV_WARN_FRAMES} is "
                    "not comfortable in BLITZ (timeline too fine-grained). "
                    "Continue only if you know you need this."
                )
            if budget.level == "red":
                parts.append(
                    f"{fmt_bytes(budget.wire_bytes)} is "
                    f"{budget.fraction_of_total:.0%} of this PC's RAM on the wire. "
                    "BLITZ will hold another copy."
                )
            yes = QMessageBox.warning(
                self,
                "Check before send",
                "\n\n".join(parts),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if yes != QMessageBox.StandardButton.Yes:
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

    def closeEvent(self, event) -> None:  # noqa: N802
        self.led._set_busy_cursor(False)
        for thread in (self._bin_thread, self._load_thread):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(10_000)
        super().closeEvent(event)
