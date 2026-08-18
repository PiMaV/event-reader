"""PyQt6 event-reader UI: overview → range → Δt → send to BLITZ."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QComboBox,
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
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .binning import (
    HARD_FRAME_CAP,
    OVERVIEW_PICTURES,
    SENSOR_DT_US,
    AccumMode,
    BinParams,
    PolarityMode,
    bin_events,
    event_rate_ms,
    min_dt_us,
    plan_pictures,
    stack_for_network,
)
from .evt3 import EventStore, load_evt3_raw
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


class _Led(QWidget):
    def __init__(self) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.dot = QFrame()
        self.dot.setFixedSize(14, 14)
        self.dot.setFrameShape(QFrame.Shape.NoFrame)
        self.label = QLabel("Idle")
        row.addWidget(self.dot)
        row.addWidget(self.label, stretch=1)
        self.set_state("idle", "Idle")

    def set_state(self, kind: str, text: str) -> None:
        colors = {
            "idle": "#888888",
            "work": "#e6a817",
            "wait": "#3d8bfd",
            "ok": "#2ecc71",
            "err": "#e74c3c",
        }
        self.dot.setStyleSheet(f"background:{colors.get(kind, '#888')}; border-radius:7px;")
        self.label.setText(text)


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

        step1 = QGroupBox("1 — Overview of the whole recording (not sent to BLITZ)")
        s1 = QVBoxLayout(step1)
        s1.addWidget(QLabel(
            "After opening a file, a coarse picture stack is built so you can "
            "click through time. Choose start and end here; nothing is streamed yet."
        ))
        self.rate_plot = pg.PlotWidget()
        self.rate_plot.setMinimumHeight(100)
        self.rate_plot.setLabel("bottom", "Time", units="ms")
        self.rate_plot.setLabel("left", "Events")
        self.rate_plot.showGrid(x=True, y=True, alpha=0.2)
        self.rate_curve = self.rate_plot.plot(pen=pg.mkPen("#7ec8e3", width=1))
        self.region = pg.LinearRegionItem([0.0, 1.0], brush=(230, 180, 40, 60))
        self.rate_plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        s1.addWidget(self.rate_plot)

        range_row = QHBoxLayout()
        self.start_frame = QSpinBox()
        self.end_frame = QSpinBox()
        self.start_frame.valueChanged.connect(self._on_frame_range_changed)
        self.end_frame.valueChanged.connect(self._on_frame_range_changed)
        range_row.addWidget(QLabel("Start picture"))
        range_row.addWidget(self.start_frame)
        range_row.addWidget(QLabel("End picture"))
        range_row.addWidget(self.end_frame)
        s1.addLayout(range_row)
        self.range_time_label = QLabel("Load a recording first.")
        s1.addWidget(self.range_time_label)

        self.preview = pg.ImageView()
        self.preview.ui.roiBtn.hide()
        self.preview.ui.menuBtn.hide()
        self.preview.setMinimumHeight(220)
        s1.addWidget(self.preview)
        layout.addWidget(step1)

        step2 = QGroupBox("2 — Frame time for the selected range")
        form = QFormLayout(step2)
        self.dt_ms = QDoubleSpinBox()
        self.dt_ms.setRange(0.001, 1_000_000.0)
        self.dt_ms.setDecimals(3)
        self.dt_ms.setSingleStep(0.1)
        self.dt_ms.setValue(1.0)
        self.dt_ms.setSuffix(" ms")
        self.dt_ms.valueChanged.connect(self._refresh_plan_label)
        form.addRow("Frame time (Δt)", self.dt_ms)
        self.min_dt_label = QLabel("Min Δt: —")
        form.addRow(self.min_dt_label)
        self.polarity = QComboBox()
        for mode in PolarityMode:
            self.polarity.addItem(mode.value, mode)
        self.polarity.setCurrentIndex(2)
        form.addRow("Polarity", self.polarity)
        self.plan_label = QLabel("Select a range, then set Δt.")
        self.plan_label.setWordWrap(True)
        form.addRow(self.plan_label)
        layout.addWidget(step2)

        step3 = QGroupBox("3 — Send to BLITZ (only when you click)")
        s3 = QVBoxLayout(step3)
        s3.addWidget(QLabel(
            "BLITZ does not update by itself while you scrub. Connect BLITZ "
            "Network first, then send. Green status = BLITZ downloaded the stack."
        ))
        self.apply_btn = QPushButton("Build pictures and send to BLITZ")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._export_to_blitz)
        self.push_btn = QPushButton("Send last pictures again (no rebuild)")
        self.push_btn.setEnabled(False)
        self.push_btn.clicked.connect(self._repush)
        s3.addWidget(self.apply_btn)
        s3.addWidget(self.push_btn)
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
            f"In BLITZ → Network: address <b>{self.publisher.base_url}</b>, "
            f"token <b>{self.publisher.token}</b> → Connect. "
            "Then use the button above."
        )

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
        self.led.set_state("work", "1/3 Decoding recording…")
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
        dur_ms = store.duration_us / 1000.0
        self.meta_label.setText(
            f"{store.width}×{store.height}  |  {len(store):,} events  |  {dur_ms:.1f} ms"
        )
        x_ms, counts = event_rate_ms(store)
        self.rate_curve.setData(x_ms, counts)
        xmax = max(dur_ms, 0.001)
        self.region.blockSignals(True)
        self.region.setBounds((0.0, xmax))
        self.region.setRegion((0.0, xmax))
        self.region.blockSignals(False)
        self.rate_plot.setXRange(0.0, xmax, padding=0.02)
        self.led.set_state("work", f"1/3 Building ~{OVERVIEW_PICTURES}-picture overview…")
        self._start_bin(self._overview_params(), "overview")

    def _overview_params(self) -> BinParams:
        assert self._store is not None
        window_us = max(1, self._store.duration_us)
        dt_us, n, _c, _u = plan_pictures(window_us, n_frames=OVERVIEW_PICTURES)
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
        self.led.set_state("err", "Load failed")
        QMessageBox.critical(self, "Load failed", message)

    def _selected_window_us(self) -> tuple[int, int]:
        assert self._store is not None
        t0_rel_ms, t1_rel_ms = self.region.getRegion()
        if t1_rel_ms < t0_rel_ms:
            t0_rel_ms, t1_rel_ms = t1_rel_ms, t0_rel_ms
        t_base = int(self._store.t_min)
        t0 = t_base + int(round(t0_rel_ms * 1000.0))
        t1 = t_base + int(round(t1_rel_ms * 1000.0))
        return t0, max(t0 + 1, t1)

    def _refresh_plan_label(self) -> None:
        if self._store is None:
            return
        t0, t1 = self._selected_window_us()
        window_us = t1 - t0
        dt_us = max(1, int(round(self.dt_ms.value() * 1000.0)))
        dt_us, n, capped, used_us = plan_pictures(window_us, dt_us=dt_us)
        lo = min_dt_us(window_us)
        self.dt_ms.setMinimum(max(0.001, lo / 1000.0))
        self.min_dt_label.setText(
            f"Min Δt for this range: {lo / 1000.0:.3f} ms  "
            f"({HARD_FRAME_CAP} pictures max; sensor tick {SENSOR_DT_US} µs)"
        )
        span_ms = window_us / 1000.0
        mb = n * self._store.height * self._store.width / (1024 * 1024)
        text = (
            f"{span_ms:.1f} ms selected  →  {n} pictures × {dt_us / 1000.0:.3f} ms "
            f"(~{mb:.0f} MB to BLITZ)"
        )
        if capped:
            text += (
                f"  —  Δt is below the minimum; only the first {used_us / 1000.0:.1f} ms "
                f"would be sent. Raise Δt to at least {lo / 1000.0:.3f} ms."
            )
            pal = QPalette(self.plan_label.palette())
            pal.setColor(QPalette.ColorRole.WindowText, QColor("#e6a817"))
            self.plan_label.setPalette(pal)
        else:
            self.plan_label.setPalette(QLabel().palette())
        self.plan_label.setText(text)
        t0_rel = (t0 - self._store.t_min) / 1000.0
        t1_rel = (t1 - self._store.t_min) / 1000.0
        self.range_time_label.setText(
            f"Selected {t0_rel:.2f}–{t1_rel:.2f} ms  "
            f"(overview pictures {self.start_frame.value()}–{self.end_frame.value()})"
        )

    def _on_region_changed(self) -> None:
        if self._syncing_range or self._store is None or self._overview is None:
            self._refresh_plan_label()
            return
        t0, t1 = self._selected_window_us()
        dt = self._overview_dt_us
        i0 = int((t0 - self._store.t_min) // dt)
        i1 = int((t1 - self._store.t_min) // dt)
        n = self._overview.shape[0]
        i0 = max(0, min(n - 1, i0))
        i1 = max(i0, min(n - 1, i1))
        self._syncing_range = True
        self.start_frame.setValue(i0)
        self.end_frame.setValue(i1)
        self._syncing_range = False
        self._refresh_plan_label()

    def _on_frame_range_changed(self) -> None:
        if self._syncing_range or self._store is None:
            return
        if self.start_frame.value() > self.end_frame.value():
            self._syncing_range = True
            if self.sender() is self.start_frame:
                self.end_frame.setValue(self.start_frame.value())
            else:
                self.start_frame.setValue(self.end_frame.value())
            self._syncing_range = False
        i0, i1 = self.start_frame.value(), self.end_frame.value()
        t0_ms = i0 * self._overview_dt_us / 1000.0
        t1_ms = (i1 + 1) * self._overview_dt_us / 1000.0
        self._syncing_range = True
        self.region.setRegion((t0_ms, t1_ms))
        self._syncing_range = False
        if self._overview is not None:
            self.preview.setCurrentIndex(i0)
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
        net = stack_for_network(arr)
        if kind == "overview":
            self._overview = net
            n = net.shape[0]
            self.start_frame.blockSignals(True)
            self.end_frame.blockSignals(True)
            self.start_frame.setRange(0, max(0, n - 1))
            self.end_frame.setRange(0, max(0, n - 1))
            self.start_frame.setValue(0)
            self.end_frame.setValue(max(0, n - 1))
            self.start_frame.blockSignals(False)
            self.end_frame.blockSignals(False)
            self.preview.setImage(net, autoLevels=True, axes={"t": 0, "y": 1, "x": 2})
            self._suggest_dt_from_overview()
            self.apply_btn.setEnabled(True)
            self.led.set_state(
                "ok",
                f"1/3 Overview ready ({n} pictures, {self._overview_dt_us / 1000.0:.2f} ms each). "
                "Pick start/end, set Δt, then send.",
            )
            self._refresh_plan_label()
            return
        self.publisher.set_stack(net, push=True)
        self.push_btn.setEnabled(True)
        if self._blitz_clients <= 0:
            self.led.set_state("wait", "3/3 Pictures ready — connect BLITZ Network, or wait for download…")
        else:
            self.led.set_state("wait", "3/3 Sending to BLITZ…")
        self.statusBar().showMessage(f"Sent {tuple(net.shape)} in {elapsed * 1000:.0f} ms")

    def _export_params(self) -> BinParams:
        assert self._store is not None
        t0, t1 = self._selected_window_us()
        dt_us = max(1, int(round(self.dt_ms.value() * 1000.0)))
        dt_us, n, _c, _u = plan_pictures(t1 - t0, dt_us=dt_us)
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
        self.led.set_state("work", "2/3 Building pictures for BLITZ…")
        self._start_bin(self._export_params(), "export")

    def _repush(self) -> None:
        self.publisher.push()
        self.led.set_state(
            "wait",
            "Sending last pictures again…" if self._blitz_clients else "Waiting for BLITZ to connect…",
        )

    def _on_blitz_downloaded(self, nbytes: int) -> None:
        mb = nbytes / (1024 * 1024)
        self.led.set_state("ok", f"BLITZ received the stack ({mb:.1f} MB)")

    def _on_client_count(self, n: int) -> None:
        self._blitz_clients = n

    def _on_bin_failed(self, message: str) -> None:
        self.led.set_state("err", "Failed")
        QMessageBox.critical(self, "Failed", message)

    def closeEvent(self, event) -> None:  # noqa: N802
        for thread in (self._bin_thread, self._load_thread):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(10_000)
        super().closeEvent(event)
