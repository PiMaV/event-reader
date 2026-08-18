"""PyQt6 event-reader UI: time region + frame time → pictures for BLITZ."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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
    QRadioButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .binning import (
    HARD_FRAME_CAP,
    AccumMode,
    BinParams,
    PolarityMode,
    bin_events,
    event_rate_ms,
    plan_pictures,
    stack_for_network,
)
from .evt3 import EventStore, load_evt3_raw
from .server import DEFAULT_TOKEN, StackPublisher

log = logging.getLogger("evt_sidecar.gui")


class _LoadWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            self.status.emit(f"Decoding {self.path.name}…")
            t0 = time.perf_counter()
            store = load_evt3_raw(self.path)
            elapsed = time.perf_counter() - t0
            log.info("decoded %s: %s events in %.2fs", self.path.name, f"{len(store):,}", elapsed)
            self.finished.emit(store)
        except Exception as exc:  # noqa: BLE001
            log.exception("load failed")
            self.failed.emit(str(exc))


class _BinWorker(QObject):
    finished = pyqtSignal(object, float)
    failed = pyqtSignal(str)

    def __init__(self, store: EventStore, params: BinParams, generation: int) -> None:
        super().__init__()
        self.store = store
        self.params = params
        self.generation = generation

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            stack = bin_events(self.store, self.params)
            elapsed = time.perf_counter() - t0
            log.info("binned shape=%s in %.2fs", stack.shape, elapsed)
            self.finished.emit((stack, self.generation), elapsed)
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
        c = colors.get(kind, "#888888")
        self.dot.setStyleSheet(f"background:{c}; border-radius:7px;")
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
        self.resize(720, 820)

        self._store: EventStore | None = None
        self._load_thread: QThread | None = None
        self._load_worker: _LoadWorker | None = None
        self._bin_thread: QThread | None = None
        self._bin_worker: _BinWorker | None = None
        self._bin_generation = 0
        self._bin_restart_pending = False
        self._drive_dt = True
        self._blitz_clients = 0
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self._apply_bin)

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

        explain = QLabel(
            "Event cameras do not store pictures. The yellow band is the time "
            "you turn into pictures. <b>Frame time</b> is how long each picture "
            "integrates (1.5 s at 1 ms → 1500 pictures). Or set the number of "
            "pictures instead — the frame time follows."
        )
        explain.setWordWrap(True)
        layout.addWidget(explain)

        self.rate_plot = pg.PlotWidget()
        self.rate_plot.setMinimumHeight(120)
        self.rate_plot.setLabel("bottom", "Time", units="ms")
        self.rate_plot.setLabel("left", "Events")
        self.rate_plot.showGrid(x=True, y=True, alpha=0.2)
        self.rate_curve = self.rate_plot.plot(pen=pg.mkPen("#7ec8e3", width=1))
        self.region = pg.LinearRegionItem([0.0, 1.0], brush=(230, 180, 40, 60))
        self.region.setZValue(10)
        self.rate_plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.region.sigRegionChangeFinished.connect(self._on_param_changed)
        layout.addWidget(self.rate_plot)

        params = QGroupBox("Pictures")
        form = QFormLayout(params)

        drive_row = QHBoxLayout()
        self.radio_dt = QRadioButton("Set frame time")
        self.radio_n = QRadioButton("Set number of pictures")
        self.radio_dt.setChecked(True)
        drive = QButtonGroup(self)
        drive.addButton(self.radio_dt)
        drive.addButton(self.radio_n)
        self.radio_dt.toggled.connect(self._on_drive_toggled)
        drive_row.addWidget(self.radio_dt)
        drive_row.addWidget(self.radio_n)
        form.addRow(drive_row)

        self.dt_ms = QDoubleSpinBox()
        self.dt_ms.setRange(0.001, 1_000_000.0)
        self.dt_ms.setDecimals(3)
        self.dt_ms.setSingleStep(0.1)
        self.dt_ms.setValue(1.0)
        self.dt_ms.setSuffix(" ms")
        self.dt_ms.valueChanged.connect(self._on_dt_edited)
        form.addRow("Frame time", self.dt_ms)

        self.n_frames = QSpinBox()
        self.n_frames.setRange(1, HARD_FRAME_CAP)
        self.n_frames.setValue(40)
        self.n_frames.valueChanged.connect(self._on_n_edited)
        form.addRow("Pictures", self.n_frames)

        self.polarity = QComboBox()
        for mode in PolarityMode:
            self.polarity.addItem(mode.value, mode)
        self.polarity.setCurrentIndex(2)
        self.polarity.currentIndexChanged.connect(self._on_param_changed)
        form.addRow("Polarity", self.polarity)

        self.plan_label = QLabel("Load a recording to plan pictures.")
        self.plan_label.setWordWrap(True)
        form.addRow(self.plan_label)

        self.live_apply = QCheckBox("Update while dragging (debounce)")
        self.live_apply.setChecked(True)
        form.addRow(self.live_apply)

        layout.addWidget(params)
        self._apply_drive_enabled()

        preview_box = QGroupBox("Preview (what BLITZ will get)")
        pv = QVBoxLayout(preview_box)
        self.preview = pg.ImageView()
        self.preview.ui.roiBtn.hide()
        self.preview.ui.menuBtn.hide()
        self.preview.setMinimumHeight(200)
        pv.addWidget(self.preview)
        layout.addWidget(preview_box)

        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Make pictures → BLITZ")
        self.apply_btn.clicked.connect(self._apply_bin)
        self.apply_btn.setEnabled(False)
        self.push_btn = QPushButton("Send again")
        self.push_btn.clicked.connect(self._repush)
        self.push_btn.setEnabled(False)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.push_btn)
        layout.addLayout(btn_row)

        net = QGroupBox("BLITZ Network")
        net_form = QFormLayout(net)
        self.host_edit = QLineEdit(self.publisher.host)
        self.port_edit = QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(self.publisher.port)
        self.token_edit = QLineEdit(self.publisher.token)
        net_form.addRow("Listen host", self.host_edit)
        net_form.addRow("Port", self.port_edit)
        net_form.addRow("Token", self.token_edit)
        self.connect_hint = QLabel()
        self.connect_hint.setWordWrap(True)
        net_form.addRow(self.connect_hint)
        layout.addWidget(net)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"Serving {self.publisher.base_url}")

    def _update_connect_hint(self) -> None:
        self.connect_hint.setText(
            f"In BLITZ → Network: <b>{self.publisher.base_url}</b>, "
            f"token <b>{self.publisher.token}</b>"
        )

    def _apply_drive_enabled(self) -> None:
        self.dt_ms.setEnabled(self._drive_dt)
        self.n_frames.setEnabled(not self._drive_dt)

    def _on_drive_toggled(self, _checked: bool) -> None:
        self._drive_dt = self.radio_dt.isChecked()
        self._apply_drive_enabled()
        self._refresh_plan_label()
        self._on_param_changed()

    def _window_ms(self) -> tuple[float, float]:
        lo, hi = self.region.getRegion()
        if hi < lo:
            lo, hi = hi, lo
        return float(lo), float(hi)

    def _current_plan(self) -> tuple[int, int, bool, int]:
        t0, t1 = self._window_ms()
        window_us = max(1, int(round((t1 - t0) * 1000.0)))
        if self._drive_dt:
            dt_us = max(1, int(round(self.dt_ms.value() * 1000.0)))
            return plan_pictures(window_us, dt_us=dt_us)
        return plan_pictures(window_us, n_frames=int(self.n_frames.value()))

    def _refresh_plan_label(self) -> None:
        if self._store is None:
            return
        dt_us, n, capped, used_us = self._current_plan()
        t0, t1 = self._window_ms()
        span = max(0.001, t1 - t0)
        mb = n * self._store.height * self._store.width / (1024 * 1024)
        text = (
            f"{span:.1f} ms window  →  {n} pictures × {dt_us / 1000.0:.3f} ms "
            f"(~{mb:.0f} MB)"
        )
        if capped:
            text += (
                f"  —  would exceed {HARD_FRAME_CAP}; using the first "
                f"{used_us / 1000.0:.1f} ms of the yellow band. "
                "Raise frame time or shrink the region."
            )
            pal = QPalette(self.plan_label.palette())
            pal.setColor(QPalette.ColorRole.WindowText, QColor("#e6a817"))
            self.plan_label.setPalette(pal)
        else:
            self.plan_label.setPalette(QLabel().palette())
        self.plan_label.setText(text)
        self._sync_peer_spins(dt_us, n)

    def _sync_peer_spins(self, dt_us: int, n: int) -> None:
        self.dt_ms.blockSignals(True)
        self.n_frames.blockSignals(True)
        if not self._drive_dt:
            self.dt_ms.setValue(dt_us / 1000.0)
        else:
            self.n_frames.setValue(min(HARD_FRAME_CAP, n))
        self.dt_ms.blockSignals(False)
        self.n_frames.blockSignals(False)

    def _on_dt_edited(self) -> None:
        if not self._drive_dt:
            return
        self._refresh_plan_label()
        self._on_param_changed()

    def _on_n_edited(self) -> None:
        if self._drive_dt:
            return
        self._refresh_plan_label()
        self._on_param_changed()

    def _on_region_changed(self) -> None:
        self._refresh_plan_label()

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
        self.led.set_state("work", "Decoding recording…")
        self.apply_btn.setEnabled(False)

        thread = QThread(self)
        worker = _LoadWorker(path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status.connect(self.statusBar().showMessage)
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
            f"{store.width}×{store.height}  |  {len(store):,} events  |  "
            f"{dur_ms:.1f} ms"
        )
        x_ms, counts = event_rate_ms(store)
        self.rate_curve.setData(x_ms, counts)
        self.region.blockSignals(True)
        self.region.setBounds((float(x_ms[0]), float(x_ms[-1]) if len(x_ms) else dur_ms))
        self.region.setRegion((0.0, max(dur_ms, 0.001)))
        self.region.blockSignals(False)
        self.rate_plot.setXRange(0.0, max(dur_ms, 0.001), padding=0.02)
        self.apply_btn.setEnabled(True)
        self._refresh_plan_label()
        self.led.set_state("work", "Making pictures…")
        self._apply_bin()

    def _on_load_failed(self, message: str) -> None:
        self.meta_label.setText("Load failed.")
        self.led.set_state("err", "Load failed")
        QMessageBox.critical(self, "Load failed", message)

    def _on_param_changed(self) -> None:
        if self._store is None:
            return
        self._refresh_plan_label()
        if self.live_apply.isChecked():
            self._debounce.start()

    def _current_params(self) -> BinParams:
        assert self._store is not None
        dt_us, n, _capped, _used = self._current_plan()
        t0_rel, _t1_rel = self._window_ms()
        t_base = self._store.t_min
        t0 = t_base + int(round(t0_rel * 1000.0))
        t1 = t0 + n * dt_us
        return BinParams(
            dt_us=dt_us,
            polarity=self.polarity.currentData(),
            accum=AccumMode.COUNT,
            t0_us=t0,
            t1_us=t1,
            max_frames=n,
        )

    def _apply_bin(self) -> None:
        if self._store is None:
            return
        self._bin_generation += 1
        if self._bin_thread is not None and self._bin_thread.isRunning():
            self._bin_restart_pending = True
            self.led.set_state("work", "Busy — will use latest settings…")
            return
        self._bin_restart_pending = False
        self._start_bin_worker(self._bin_generation)

    def _start_bin_worker(self, generation: int) -> None:
        assert self._store is not None
        params = self._current_params()
        self.led.set_state("work", f"Making {params.max_frames} pictures…")
        thread = QThread(self)
        worker = _BinWorker(self._store, params, generation)
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
        if self._bin_restart_pending and self._store is not None:
            self._bin_restart_pending = False
            self._start_bin_worker(self._bin_generation)

    def _on_binned(self, payload: object, elapsed: float) -> None:
        stack, generation = payload  # type: ignore[misc]
        if generation != self._bin_generation:
            return
        net = stack_for_network(stack)
        self.preview.setImage(net, autoLevels=True, axes={"t": 0, "y": 1, "x": 2})
        self.publisher.set_stack(net, push=True)
        self.push_btn.setEnabled(True)
        if self._blitz_clients <= 0:
            self.led.set_state("wait", "Pictures ready — waiting for BLITZ to connect…")
        else:
            self.led.set_state("wait", "Sending to BLITZ…")
        self.statusBar().showMessage(
            f"{tuple(net.shape)} in {elapsed * 1000:.0f} ms"
        )

    def _repush(self) -> None:
        self.publisher.push()
        if self._blitz_clients <= 0:
            self.led.set_state("wait", "Waiting for BLITZ to connect…")
        else:
            self.led.set_state("wait", "Sending to BLITZ…")

    def _on_blitz_downloaded(self, nbytes: int) -> None:
        mb = nbytes / (1024 * 1024)
        self.led.set_state("ok", f"BLITZ received the stack ({mb:.1f} MB)")

    def _on_client_count(self, n: int) -> None:
        self._blitz_clients = n
        if n <= 0 and self.led.label.text().startswith("BLITZ received"):
            return
        if n > 0 and "waiting for BLITZ" in self.led.label.text().lower():
            self.led.set_state("wait", "BLITZ connected — sending…")

    def _on_bin_failed(self, message: str) -> None:
        self.led.set_state("err", "Binning failed")
        QMessageBox.critical(self, "Binning failed", message)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._bin_restart_pending = False
        self._debounce.stop()
        for thread in (self._bin_thread, self._load_thread):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(10_000)
        super().closeEvent(event)
