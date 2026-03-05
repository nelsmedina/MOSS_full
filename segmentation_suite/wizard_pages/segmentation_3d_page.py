#!/usr/bin/env python3
"""
3D Segmentation page - integrates em-pipeline LSD/Ensemble strategies into MOSS GUI.
"""

import os
import psutil
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QGroupBox, QFileDialog, QMessageBox, QCheckBox, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont

from ..dpi_scaling import scaled


class ConversionWorker(QThread):
    """Worker thread for TIFF to Zarr conversion."""

    progress = pyqtSignal(int, int, str)  # completed, total, message
    finished = pyqtSignal(bool, str)  # success, message
    log = pyqtSignal(str)

    def __init__(self, source: str, dest: str, chunk_size: tuple, workers: int):
        super().__init__()
        self.source = source
        self.dest = dest
        self.chunk_size = chunk_size
        self.workers = workers
        self._stop = False

    def run(self):
        try:
            from em_pipeline.data.convert import convert as do_convert

            def progress_callback(completed, total, msg):
                if not self._stop:
                    self.progress.emit(completed, total, msg)

            self.log.emit(f"Converting {self.source} to Zarr...")
            do_convert(
                self.source,
                self.dest,
                chunk_size=self.chunk_size,
                progress_callback=progress_callback,
                num_workers=self.workers,
            )
            self.finished.emit(True, f"Created {self.dest}")
        except Exception as e:
            self.finished.emit(False, str(e))

    def stop(self):
        self._stop = True


class SegmentationWorker(QThread):
    """Worker thread for 3D segmentation."""

    progress = pyqtSignal(str, int)  # stage, percent
    finished = pyqtSignal(bool, str)  # success, message
    log = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        strategy: str,
        quality: str,
        device: str,
        model_path: Optional[str] = None
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.strategy = strategy
        self.quality = quality
        self.device = device
        self.model_path = model_path
        self._stop = False

    def run(self):
        try:
            from em_pipeline.pipeline import SegmentationPipeline, PipelineConfig

            self.log.emit(f"Starting {self.strategy} segmentation...")
            self.log.emit(f"  Input: {self.input_path}")
            self.log.emit(f"  Output: {self.output_path}")
            self.log.emit(f"  Device: {self.device}")

            config = PipelineConfig(
                strategy=self.strategy,
                quality=self.quality,
                device=self.device,
                resume=True,
            )

            pipeline = SegmentationPipeline(config)

            # Run segmentation
            result = pipeline.run(
                self.input_path,
                self.output_path,
                model_path=self.model_path,
            )

            self.log.emit(f"Segmentation complete!")
            self.log.emit(f"  Segments: {result.num_segments}")
            self.log.emit(f"  Time: {result.total_time:.1f}s")

            self.finished.emit(True, f"Created {result.num_segments} segments")

        except Exception as e:
            import traceback
            self.log.emit(f"Error: {e}")
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, str(e))

    def stop(self):
        self._stop = True


class Segmentation3DPage(QWidget):
    """Page for running 3D segmentation (LSD/Ensemble pipeline)."""

    # Signals
    segmentation_complete = pyqtSignal(str)  # Output path
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_dir = None
        self.conversion_worker = None
        self.segmentation_worker = None
        self._is_busy = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(scaled(15))

        # Title
        title = QLabel("3D Segmentation Pipeline")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)

        desc = QLabel(
            "Run LSD or Ensemble segmentation on your EM volume. "
            "This uses the em-pipeline backend with GPU acceleration."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888888;")
        layout.addWidget(desc)

        # Input section
        input_group = QGroupBox("Input Data")
        input_group.setStyleSheet(self._group_style())
        input_layout = QVBoxLayout(input_group)

        # TIFF source
        tiff_row = QHBoxLayout()
        tiff_row.addWidget(QLabel("TIFF Directory:"))
        self.tiff_path_label = QLabel("Not selected")
        self.tiff_path_label.setStyleSheet("color: #888888;")
        tiff_row.addWidget(self.tiff_path_label, 1)
        self.browse_tiff_btn = QPushButton("Browse...")
        self.browse_tiff_btn.clicked.connect(self._browse_tiff)
        tiff_row.addWidget(self.browse_tiff_btn)
        input_layout.addLayout(tiff_row)

        # Zarr volume
        zarr_row = QHBoxLayout()
        zarr_row.addWidget(QLabel("Zarr Volume:"))
        self.zarr_path_label = QLabel("Not found")
        self.zarr_path_label.setStyleSheet("color: #888888;")
        zarr_row.addWidget(self.zarr_path_label, 1)
        self.convert_btn = QPushButton("Convert TIFF → Zarr")
        self.convert_btn.clicked.connect(self._start_conversion)
        self.convert_btn.setEnabled(False)
        zarr_row.addWidget(self.convert_btn)
        input_layout.addLayout(zarr_row)

        layout.addWidget(input_group)

        # Segmentation settings
        settings_group = QGroupBox("Segmentation Settings")
        settings_group.setStyleSheet(self._group_style())
        settings_layout = QVBoxLayout(settings_group)

        # Strategy selection
        strat_row = QHBoxLayout()
        strat_row.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["lsd", "ensemble", "joint"])
        self.strategy_combo.setToolTip(
            "LSD: Local Shape Descriptors (fast, good baseline)\n"
            "Ensemble: LSD + FFN voting (more accurate)\n"
            "Joint: Multi-task model (best quality, slower)"
        )
        strat_row.addWidget(self.strategy_combo)

        strat_row.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["fast", "balanced", "accurate"])
        strat_row.addWidget(self.quality_combo)
        strat_row.addStretch()
        settings_layout.addLayout(strat_row)

        # Device selection
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self._populate_devices()
        device_row.addWidget(self.device_combo)

        # Memory info
        mem = psutil.virtual_memory()
        mem_str = f"RAM: {mem.available / 1e9:.1f} / {mem.total / 1e9:.1f} GB"
        self.mem_label = QLabel(mem_str)
        self.mem_label.setStyleSheet("color: #888888;")
        device_row.addWidget(self.mem_label)
        device_row.addStretch()
        settings_layout.addLayout(device_row)

        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_path_label = QLabel("Default (untrained)")
        self.model_path_label.setStyleSheet("color: #888888;")
        model_row.addWidget(self.model_path_label, 1)
        self.browse_model_btn = QPushButton("Select Model...")
        self.browse_model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.browse_model_btn)
        settings_layout.addLayout(model_row)

        layout.addWidget(settings_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_group.setStyleSheet(self._group_style())
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888;")
        progress_layout.addWidget(self.status_label)

        # Log output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(scaled(120))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        progress_layout.addWidget(self.log_text)

        layout.addWidget(progress_group)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.run_btn = QPushButton("▶ Run Segmentation")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:disabled { background-color: #1B5E20; color: #888888; }
        """)
        self.run_btn.clicked.connect(self._start_segmentation)
        self.run_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        layout.addStretch()

    def _group_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """

    def _populate_devices(self):
        """Populate device dropdown with available devices."""
        self.device_combo.clear()

        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                    self.device_combo.addItem(f"cuda:{i} ({name}, {mem:.0f}GB)", f"cuda:{i}")
        except ImportError:
            pass

        self.device_combo.addItem("CPU", "cpu")

    def set_config(self, config: dict):
        """Set configuration from project."""
        self.project_dir = Path(config.get('project_dir', ''))
        self._scan_for_data()

    def _scan_for_data(self):
        """Scan project directory for data files."""
        if not self.project_dir or not self.project_dir.exists():
            return

        # Look for TIFF data
        data_dirs = ['data', 'raw', 'images', 'tiff', '.']
        for subdir in data_dirs:
            check_dir = self.project_dir / subdir
            if check_dir.exists():
                tiff_files = list(check_dir.glob("*.tif")) + list(check_dir.glob("*.tiff"))
                if tiff_files:
                    self.tiff_path_label.setText(str(check_dir))
                    self.tiff_path_label.setStyleSheet("color: #4CAF50;")
                    self.convert_btn.setEnabled(True)
                    break

        # Look for Zarr data
        zarr_dirs = list(self.project_dir.glob("*.zarr")) + list(self.project_dir.glob("**/*.zarr"))
        for zarr_dir in zarr_dirs:
            if (zarr_dir / '.zarray').exists():
                self.zarr_path_label.setText(str(zarr_dir))
                self.zarr_path_label.setStyleSheet("color: #4CAF50;")
                self.run_btn.setEnabled(True)
                break

        # Look for model
        model_files = list(self.project_dir.glob("**/*.pt")) + list(self.project_dir.glob("**/*.pth"))
        if model_files:
            self.model_path_label.setText(str(model_files[0]))
            self.model_path_label.setStyleSheet("color: #4CAF50;")

    def _browse_tiff(self):
        """Browse for TIFF directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select TIFF Directory",
            str(self.project_dir) if self.project_dir else ""
        )
        if path:
            tiff_files = list(Path(path).glob("*.tif")) + list(Path(path).glob("*.tiff"))
            if tiff_files:
                self.tiff_path_label.setText(path)
                self.tiff_path_label.setStyleSheet("color: #4CAF50;")
                self.convert_btn.setEnabled(True)
                self._log(f"Found {len(tiff_files)} TIFF files")
            else:
                QMessageBox.warning(self, "No TIFF Files", "No TIFF files found in selected directory")

    def _browse_model(self):
        """Browse for model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint",
            str(self.project_dir) if self.project_dir else "",
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.model_path_label.setText(path)
            self.model_path_label.setStyleSheet("color: #4CAF50;")

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _start_conversion(self):
        """Start TIFF to Zarr conversion."""
        tiff_path = self.tiff_path_label.text()
        if tiff_path == "Not selected":
            return

        # Determine output path
        tiff_dir = Path(tiff_path)
        zarr_path = tiff_dir.parent / f"{tiff_dir.name}.zarr"

        # Calculate smart chunk size and workers based on memory
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1e9

        # Estimate image size from first file
        tiff_files = list(tiff_dir.glob("*.tif")) + list(tiff_dir.glob("*.tiff"))
        if tiff_files:
            file_size_mb = tiff_files[0].stat().st_size / 1e6
            # If files are large (>50MB), use fewer workers
            if file_size_mb > 100:
                workers = 1
                chunk_z = 1
            elif file_size_mb > 50:
                workers = min(2, int(available_gb / 4))
                chunk_z = 8
            else:
                workers = min(4, int(available_gb / 2))
                chunk_z = 32

            self._log(f"Auto-configured: {workers} workers, chunk_z={chunk_z} (images ~{file_size_mb:.0f}MB)")
        else:
            workers = 2
            chunk_z = 32

        chunk_size = (chunk_z, 512, 512)

        self._set_busy(True)
        self.status_label.setText("Converting TIFF to Zarr...")

        self.conversion_worker = ConversionWorker(
            str(tiff_dir), str(zarr_path), chunk_size, workers
        )
        self.conversion_worker.progress.connect(self._on_conversion_progress)
        self.conversion_worker.finished.connect(self._on_conversion_finished)
        self.conversion_worker.log.connect(self._log)
        self.conversion_worker.start()

    def _on_conversion_progress(self, completed: int, total: int, msg: str):
        """Handle conversion progress update."""
        if total > 0:
            pct = int(100 * completed / total)
            self.progress_bar.setValue(pct)
            self.status_label.setText(f"Converting: {completed}/{total} ({pct}%)")

    def _on_conversion_finished(self, success: bool, message: str):
        """Handle conversion completion."""
        self._set_busy(False)

        if success:
            self._log(f"Conversion complete: {message}")
            self.status_label.setText("Conversion complete!")
            self.zarr_path_label.setText(message.replace("Created ", ""))
            self.zarr_path_label.setStyleSheet("color: #4CAF50;")
            self.run_btn.setEnabled(True)
            self.progress_bar.setValue(100)
        else:
            self._log(f"Conversion failed: {message}")
            self.status_label.setText(f"Error: {message}")
            QMessageBox.critical(self, "Conversion Failed", message)

    def _start_segmentation(self):
        """Start 3D segmentation."""
        zarr_path = self.zarr_path_label.text()
        if zarr_path in ("Not found", ""):
            QMessageBox.warning(self, "No Input", "Please convert TIFF to Zarr first")
            return

        # Determine output path
        input_path = Path(zarr_path)
        output_path = input_path.parent / f"segmentation_{self.strategy_combo.currentText()}.zarr"

        # Get model path if set
        model_path = self.model_path_label.text()
        if model_path in ("Default (untrained)", ""):
            model_path = None
            self._log("Warning: Using untrained model - results will not be meaningful")
            self._log("Train a model first or select a pretrained checkpoint")

        device = self.device_combo.currentData() or "cpu"

        self._set_busy(True)
        self.status_label.setText("Starting segmentation...")
        self.progress_bar.setValue(0)

        self.segmentation_worker = SegmentationWorker(
            str(zarr_path),
            str(output_path),
            self.strategy_combo.currentText(),
            self.quality_combo.currentText(),
            device,
            model_path
        )
        self.segmentation_worker.progress.connect(self._on_seg_progress)
        self.segmentation_worker.finished.connect(self._on_seg_finished)
        self.segmentation_worker.log.connect(self._log)
        self.segmentation_worker.start()

    def _on_seg_progress(self, stage: str, percent: int):
        """Handle segmentation progress."""
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"{stage}: {percent}%")

    def _on_seg_finished(self, success: bool, message: str):
        """Handle segmentation completion."""
        self._set_busy(False)

        if success:
            self._log(f"Segmentation complete: {message}")
            self.status_label.setText("Segmentation complete!")
            self.progress_bar.setValue(100)

            # Get output path
            zarr_path = self.zarr_path_label.text()
            input_path = Path(zarr_path)
            output_path = input_path.parent / f"segmentation_{self.strategy_combo.currentText()}.zarr"

            self.segmentation_complete.emit(str(output_path))
        else:
            self._log(f"Segmentation failed: {message}")
            self.status_label.setText(f"Error: {message}")
            QMessageBox.critical(self, "Segmentation Failed", message)

    def _cancel(self):
        """Cancel current operation."""
        if self.conversion_worker and self.conversion_worker.isRunning():
            self.conversion_worker.stop()
            self.conversion_worker.wait()
            self._log("Conversion cancelled")

        if self.segmentation_worker and self.segmentation_worker.isRunning():
            self.segmentation_worker.stop()
            self.segmentation_worker.wait()
            self._log("Segmentation cancelled")

        self._set_busy(False)
        self.status_label.setText("Cancelled")

    def _set_busy(self, busy: bool):
        """Set busy state."""
        self._is_busy = busy
        self.browse_tiff_btn.setEnabled(not busy)
        self.convert_btn.setEnabled(not busy and self.tiff_path_label.text() != "Not selected")
        self.run_btn.setEnabled(not busy and self.zarr_path_label.text() not in ("Not found", ""))
        self.browse_model_btn.setEnabled(not busy)
        self.strategy_combo.setEnabled(not busy)
        self.quality_combo.setEnabled(not busy)
        self.device_combo.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.busy_changed.emit(busy)

    def is_busy(self) -> bool:
        return self._is_busy
