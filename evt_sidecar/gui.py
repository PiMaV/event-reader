"""PyQt6 sidecar UI: load EVT3 RAW, tweak bin params, push stacks to BLITZ."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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

from .binning import AccumMode, BinParams, PolarityMode, bin_events, stack_for_network
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
            self.status.emit(f"Decoding {self.path.name} (Numba JIT on first run may take a bit)…")
            t0 = time.perf_counter()
            store = load_evt3_raw(self.path)
            elapsed = time.perf_counter() - t0
            log.info(
                "decoded %s: %s events in %.2fs",
                self.path.name,
                f"{len(store):,}",
                elapsed,
            )
            self.status.emit(f"Decoded {len(store):,} events in {elapsed:.1f}s")
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
        self.resize(520, 480)

        self._store: EventStore | None = None
        # Keep QObject workers alive (local-only refs get GC'd → stuck on Decoding…)
        self._load_thread: QThread | None = None
        self._load_worker: _LoadWorker | None = None
        self._bin_thread: QThread | None = None
        self._bin_worker: _BinWorker | None = None
        self._bin_generation = 0
        self._bin_restart_pending = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self._apply_bin)

        self.publisher = StackPublisher(host=host, port=port, token=token)
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
        self.path_edit.setPlaceholderText("IDS / Prophesee EVT3 .raw …")
        self.path_edit.setReadOnly(True)
        browse = QPushButton("Open RAW…")
        browse.clicked.connect(self._browse)
        file_row.addWidget(self.path_edit, stretch=1)
        file_row.addWidget(browse)
        layout.addLayout(file_row)

        self.meta_label = QLabel("No file loaded.")
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)

        params = QGroupBox("Binning")
        form = QFormLayout(params)

        self.dt_ms = QDoubleSpinBox()
        self.dt_ms.setRange(0.001, 1000.0)
        self.dt_ms.setDecimals(3)
        self.dt_ms.setSingleStep(0.1)
        self.dt_ms.setValue(1.0)
        self.dt_ms.setSuffix(" ms")
        self.dt_ms.valueChanged.connect(self._on_param_changed)
        form.addRow("Δt", self.dt_ms)

        self.polarity = QComboBox()
        for mode in PolarityMode:
            self.polarity.addItem(mode.value, mode)
        self.polarity.setCurrentIndex(2)  # both
        self.polarity.currentIndexChanged.connect(self._on_param_changed)
        form.addRow("Polarity", self.polarity)

        self.t0_ms = QDoubleSpinBox()
        self.t0_ms.setRange(0.0, 1e9)
        self.t0_ms.setDecimals(3)
        self.t0_ms.setSuffix(" ms")
        self.t0_ms.valueChanged.connect(self._on_param_changed)
        form.addRow("Window start (rel)", self.t0_ms)

        self.t1_ms = QDoubleSpinBox()
        self.t1_ms.setRange(0.0, 1e9)
        self.t1_ms.setDecimals(3)
        self.t1_ms.setSuffix(" ms")
        self.t1_ms.valueChanged.connect(self._on_param_changed)
        form.addRow("Window end (rel)", self.t1_ms)

        self.max_frames = QSpinBox()
        self.max_frames.setRange(1, 50_000)
        # 720×1280 uint8 ≈ 0.9 MB/frame after network compaction
        self.max_frames.setValue(40)
        self.max_frames.valueChanged.connect(self._on_param_changed)
        form.addRow("Max frames", self.max_frames)

        self.live_apply = QCheckBox("Live apply (debounce)")
        self.live_apply.setChecked(True)
        form.addRow(self.live_apply)

        layout.addWidget(params)

        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply → BLITZ")
        self.apply_btn.clicked.connect(self._apply_bin)
        self.apply_btn.setEnabled(False)
        self.push_btn = QPushButton("Re-push")
        self.push_btn.clicked.connect(self.publisher.push)
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
        note = QLabel(
            "Server starts with the app. Restart the sidecar to change host/port/token."
        )
        note.setWordWrap(True)
        net_form.addRow(note)
        layout.addWidget(net)

        layout.addStretch(1)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"Serving {self.publisher.base_url}")

    def _update_connect_hint(self) -> None:
        url = self.publisher.base_url
        token = self.publisher.token
        self.connect_hint.setText(
            f"In BLITZ → Network: address <b>{url}</b>, token <b>{token}</b>"
        )

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open EVT3 RAW",
            "",
            "RAW recordings (*.raw);;All files (*)",
        )
        if path:
            self._load_path(Path(path))

    def _load_path(self, path: Path) -> None:
        if self._load_thread is not None and self._load_thread.isRunning():
            self.statusBar().showMessage("Already decoding…")
            return

        self.path_edit.setText(str(path))
        self.meta_label.setText("Decoding…")
        self.statusBar().showMessage(f"Decoding {path.name}…")
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
        # Strong refs — required so CPython does not GC the worker mid-run
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
            f"{dur_ms:.1f} ms  |  {store.header.format_name} "
            f"(evt {store.header.evt_version})"
        )
        self.t0_ms.blockSignals(True)
        self.t1_ms.blockSignals(True)
        self.t0_ms.setValue(0.0)
        self.t1_ms.setValue(max(dur_ms, 0.001))
        self.t0_ms.blockSignals(False)
        self.t1_ms.blockSignals(False)
        self.apply_btn.setEnabled(True)
        self.statusBar().showMessage(f"Loaded {len(store):,} events — binning…")
        if self.live_apply.isChecked():
            self._apply_bin()

    def _on_load_failed(self, message: str) -> None:
        self.meta_label.setText("Load failed.")
        QMessageBox.critical(self, "Load failed", message)
        self.statusBar().showMessage("Load failed")

    def _on_param_changed(self) -> None:
        if self._store is None:
            return
        if self.live_apply.isChecked():
            self._debounce.start()

    def _current_params(self) -> BinParams:
        assert self._store is not None
        t_base = self._store.t_min
        t0 = t_base + int(self.t0_ms.value() * 1000.0)
        t1 = t_base + int(self.t1_ms.value() * 1000.0)
        return BinParams(
            dt_us=max(1, int(round(self.dt_ms.value() * 1000.0))),
            polarity=self.polarity.currentData(),
            accum=AccumMode.COUNT,
            t0_us=t0,
            t1_us=t1,
            max_frames=int(self.max_frames.value()),
        )

    def _apply_bin(self) -> None:
        if self._store is None:
            return
        self._bin_generation += 1
        # One bin worker at a time — orphaned QThreads abort the process
        # ("QThread: Destroyed while thread is still running").
        if self._bin_thread is not None and self._bin_thread.isRunning():
            self._bin_restart_pending = True
            self.statusBar().showMessage("Binning busy — will re-run with latest params…")
            return
        self._bin_restart_pending = False
        self._start_bin_worker(self._bin_generation)

    def _start_bin_worker(self, generation: int) -> None:
        assert self._store is not None
        params = self._current_params()
        mb = (
            params.max_frames
            * self._store.height
            * self._store.width
            / (1024 * 1024)
        )
        self.statusBar().showMessage(
            f"Binning Δt={params.dt_us} µs ≤{params.max_frames} frames "
            f"(~{mb:.0f} MB uint8)…"
        )

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
            # Use latest generation / UI params
            self._start_bin_worker(self._bin_generation)

    def _on_binned(self, payload: object, elapsed: float) -> None:
        stack, generation = payload  # type: ignore[misc]
        if generation != self._bin_generation:
            return  # stale (superseded while running)
        net = stack_for_network(stack)
        nbytes = int(net.nbytes)
        nz = int(np.count_nonzero(net))
        self.publisher.set_stack(net, push=True)
        self.push_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Pushed {tuple(net.shape)} {net.dtype} "
            f"({nbytes / (1024*1024):.1f} MB, nonzero={nz:,}) "
            f"in {elapsed*1000:.0f} ms → BLITZ"
        )

    def _on_bin_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Binning failed", message)
        self.statusBar().showMessage("Binning failed")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._bin_restart_pending = False
        self._debounce.stop()
        for thread in (self._bin_thread, self._load_thread):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(10_000)
        super().closeEvent(event)
