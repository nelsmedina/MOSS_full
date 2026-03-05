"""
Proofreading page for MOSS wizard.

Integrates with Neuroglancer for 3D visualization and proofreading
of segmentation results.
"""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QTextEdit,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QSplitter,
)
from PyQt6.QtGui import QFont, QColor


class ProofreadingWorker(QThread):
    """Background worker for proofreading task generation."""

    started = pyqtSignal()
    progress = pyqtSignal(int, int, str)  # current, total, message
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, dict)  # success, result

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._should_stop = False

    def stop(self):
        """Signal the worker to stop."""
        self._should_stop = True

    def run(self):
        """Generate proofreading tasks from segmentation."""
        self.started.emit()

        try:
            # Import em_pipeline proofreading module
            from em_pipeline.proofreading import MOSSBridge

            project_dir = self.config.get('project_dir', '')
            num_samples = self.config.get('num_samples', 50)

            self.log.emit(f"Initializing proofreading bridge for {project_dir}")

            # Create bridge
            bridge = MOSSBridge(
                project_dir=project_dir,
                config={
                    'neuroglancer_port': self.config.get('port', 8080),
                    'resolution': self.config.get('resolution', (4.0, 4.0, 40.0)),
                }
            )

            # Generate tasks
            self.log.emit(f"Generating {num_samples} quality sampling tasks...")
            self.progress.emit(0, 100, "Analyzing segmentation...")

            if self._should_stop:
                self.finished.emit(False, {'error': 'Cancelled'})
                return

            tasks = bridge.generate_tasks_from_predictions(
                num_samples=num_samples,
            )

            self.progress.emit(100, 100, "Done!")
            self.log.emit(f"Generated {len(tasks)} proofreading tasks")

            # Save task queue
            bridge.save_progress()

            result = {
                'project_dir': project_dir,
                'num_tasks': len(tasks),
                'task_queue_path': str(Path(project_dir) / 'proofreading_tasks.json'),
            }

            self.finished.emit(True, result)

        except ImportError as e:
            self.log.emit(f"Error: em_pipeline not installed. {e}")
            self.log.emit("Please install with: pip install em-pipeline")
            self.finished.emit(False, {'error': str(e)})

        except Exception as e:
            self.log.emit(f"Error: {e}")
            self.finished.emit(False, {'error': str(e)})


class ProofreadingPage(QWidget):
    """
    MOSS wizard page for Neuroglancer-based proofreading.

    This page:
    1. Generates proofreading tasks from segmentation results
    2. Displays a task queue with priorities
    3. Launches Neuroglancer for each task
    4. Tracks proofreading progress
    """

    # Signals for wizard integration
    proofread_complete = pyqtSignal(dict)
    busy_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.config: Dict[str, Any] = {}
        self.worker: Optional[ProofreadingWorker] = None
        self._bridge = None
        self._server_running = False
        self.init_ui()

    def init_ui(self):
        """Build the page UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("Proofreading")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        description = QLabel(
            "Review your segmentation results in Neuroglancer. "
            "Tasks are prioritized to help you find and fix errors efficiently."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #888;")
        layout.addWidget(description)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Task generation and settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Number of samples
        samples_row = QHBoxLayout()
        samples_row.addWidget(QLabel("Quality samples:"))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(10, 500)
        self.samples_spin.setValue(50)
        self.samples_spin.setToolTip("Number of random locations to sample for quality review")
        samples_row.addWidget(self.samples_spin)
        samples_row.addStretch()
        settings_layout.addLayout(samples_row)

        # Port
        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Server port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(8080)
        self.port_spin.setToolTip("Port for local data server")
        port_row.addWidget(self.port_spin)
        port_row.addStretch()
        settings_layout.addLayout(port_row)

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution (nm):"))
        self.res_x = QLineEdit("4.0")
        self.res_x.setFixedWidth(50)
        self.res_y = QLineEdit("4.0")
        self.res_y.setFixedWidth(50)
        self.res_z = QLineEdit("40.0")
        self.res_z.setFixedWidth(50)
        res_row.addWidget(self.res_x)
        res_row.addWidget(QLabel("x"))
        res_row.addWidget(self.res_y)
        res_row.addWidget(QLabel("x"))
        res_row.addWidget(self.res_z)
        res_row.addStretch()
        settings_layout.addLayout(res_row)

        left_layout.addWidget(settings_group)

        # Generate button
        self.generate_btn = QPushButton("Generate Tasks")
        self.generate_btn.clicked.connect(self._on_generate)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #666;
            }
        """)
        left_layout.addWidget(self.generate_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        left_layout.addWidget(self.status_label)

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("background-color: #1a1a1a; font-family: monospace;")
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Right panel: Task list and viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Task queue group
        task_group = QGroupBox("Task Queue")
        task_layout = QVBoxLayout(task_group)

        # Progress summary
        self.progress_summary = QLabel("No tasks generated yet")
        self.progress_summary.setStyleSheet("font-weight: bold;")
        task_layout.addWidget(self.progress_summary)

        # Task list
        self.task_list = QListWidget()
        self.task_list.setAlternatingRowColors(True)
        self.task_list.itemDoubleClicked.connect(self._on_task_double_click)
        task_layout.addWidget(self.task_list)

        # Task actions
        task_actions = QHBoxLayout()

        self.open_btn = QPushButton("Open in Neuroglancer")
        self.open_btn.clicked.connect(self._on_open_task)
        self.open_btn.setEnabled(False)
        self.open_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 6px 12px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #666;
            }
        """)
        task_actions.addWidget(self.open_btn)

        self.complete_btn = QPushButton("Mark Complete")
        self.complete_btn.clicked.connect(self._on_complete_task)
        self.complete_btn.setEnabled(False)
        task_actions.addWidget(self.complete_btn)

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self._on_skip_task)
        self.skip_btn.setEnabled(False)
        task_actions.addWidget(self.skip_btn)

        task_actions.addStretch()
        task_layout.addLayout(task_actions)

        right_layout.addWidget(task_group)

        # Browse mode button
        self.browse_btn = QPushButton("Browse Full Volume")
        self.browse_btn.clicked.connect(self._on_browse)
        self.browse_btn.setToolTip("Open Neuroglancer to freely explore the volume")
        right_layout.addWidget(self.browse_btn)

        splitter.addWidget(right_panel)

        # Set splitter sizes
        splitter.setSizes([300, 400])
        layout.addWidget(splitter)

        # Connect task list selection
        self.task_list.itemSelectionChanged.connect(self._on_task_selection_changed)

    def set_config(self, config: dict):
        """Receive configuration from wizard."""
        self.config = config
        self._log(f"Received config: project_dir={config.get('project_dir', 'N/A')}")

        # Try to load existing task queue
        project_dir = config.get('project_dir', '')
        if project_dir:
            queue_path = Path(project_dir) / 'proofreading_tasks.json'
            if queue_path.exists():
                self._log(f"Found existing task queue at {queue_path}")
                self._load_task_queue()

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)

    def _get_bridge(self):
        """Get or create the MOSS bridge."""
        if self._bridge is None:
            try:
                from em_pipeline.proofreading import MOSSBridge

                project_dir = self.config.get('project_dir', '')
                if not project_dir:
                    raise ValueError("No project directory configured")

                resolution = (
                    float(self.res_x.text()),
                    float(self.res_y.text()),
                    float(self.res_z.text()),
                )

                self._bridge = MOSSBridge(
                    project_dir=project_dir,
                    config={
                        'neuroglancer_port': self.port_spin.value(),
                        'resolution': resolution,
                    }
                )

            except ImportError:
                QMessageBox.critical(
                    self,
                    "Module Not Found",
                    "em_pipeline is not installed.\n\n"
                    "Please install with: pip install em-pipeline"
                )
                return None

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to initialize proofreading: {e}"
                )
                return None

        return self._bridge

    def _on_generate(self):
        """Start task generation."""
        project_dir = self.config.get('project_dir', '')
        if not project_dir:
            QMessageBox.warning(
                self,
                "No Project",
                "Please load a project first."
            )
            return

        # Prepare config for worker
        try:
            resolution = (
                float(self.res_x.text()),
                float(self.res_y.text()),
                float(self.res_z.text()),
            )
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Resolution",
                "Please enter valid numbers for resolution."
            )
            return

        worker_config = {
            'project_dir': project_dir,
            'num_samples': self.samples_spin.value(),
            'port': self.port_spin.value(),
            'resolution': resolution,
        }

        # Start worker
        self.worker = ProofreadingWorker(worker_config)
        self.worker.started.connect(self._on_worker_started)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_started(self):
        """Handle worker start."""
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating tasks...")
        self.busy_changed.emit(True)

    def _on_worker_progress(self, current: int, total: int, message: str):
        """Handle worker progress."""
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))
        self.status_label.setText(message)

    def _on_worker_finished(self, success: bool, result: dict):
        """Handle worker completion."""
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        self.worker = None

        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.busy_changed.emit(False)

        if success:
            self.status_label.setText(f"Generated {result.get('num_tasks', 0)} tasks")
            self._load_task_queue()
        else:
            self.status_label.setText(f"Error: {result.get('error', 'Unknown')}")

    def _load_task_queue(self):
        """Load and display task queue."""
        bridge = self._get_bridge()
        if not bridge:
            return

        try:
            bridge.load_progress()
        except FileNotFoundError:
            self._log("No saved task queue found")
            return

        # Clear and repopulate list
        self.task_list.clear()

        for task in bridge.task_queue.tasks:
            item = QListWidgetItem()

            # Format display text
            status_icon = {
                'pending': '⏳',
                'in_progress': '🔄',
                'completed': '✅',
                'skipped': '⏭️',
            }.get(task.status.value, '❓')

            type_label = {
                'review': 'Review',
                'merge': 'Merge Check',
                'split': 'Split Check',
                'sample': 'Sample',
                'browse': 'Browse',
            }.get(task.task_type.value, task.task_type.value)

            text = f"{status_icon} [{type_label}] {task.description}"
            item.setText(text)
            item.setData(Qt.ItemDataRole.UserRole, task.task_id)

            # Color based on status
            if task.status.value == 'completed':
                item.setForeground(QColor('#4CAF50'))
            elif task.status.value == 'skipped':
                item.setForeground(QColor('#888'))
            elif task.status.value == 'in_progress':
                item.setForeground(QColor('#2196F3'))

            self.task_list.addItem(item)

        # Update progress summary
        summary = bridge.get_summary()
        self.progress_summary.setText(
            f"Tasks: {summary['completed']}/{summary['total_tasks']} completed "
            f"({summary['progress']*100:.0f}%)"
        )

    def _on_task_selection_changed(self):
        """Handle task selection change."""
        has_selection = len(self.task_list.selectedItems()) > 0
        self.open_btn.setEnabled(has_selection)
        self.complete_btn.setEnabled(has_selection)
        self.skip_btn.setEnabled(has_selection)

    def _on_task_double_click(self, item: QListWidgetItem):
        """Handle double-click on task."""
        self._on_open_task()

    def _on_open_task(self):
        """Open selected task in Neuroglancer."""
        items = self.task_list.selectedItems()
        if not items:
            return

        task_id = items[0].data(Qt.ItemDataRole.UserRole)
        bridge = self._get_bridge()
        if not bridge:
            return

        task = bridge.task_queue.get_task(task_id)
        if not task:
            self._log(f"Task {task_id} not found")
            return

        try:
            # Start server if needed
            if not self._server_running:
                self._log("Starting data server...")
                bridge.start_server()
                self._server_running = True

            # Launch task
            self._log(f"Opening task: {task.description}")
            url = bridge.launch_task(task)
            self._log(f"Neuroglancer URL: {url[:60]}...")

            # Refresh list to show updated status
            self._load_task_queue()

        except Exception as e:
            self._log(f"Error launching task: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open Neuroglancer: {e}"
            )

    def _on_complete_task(self):
        """Mark selected task as complete."""
        items = self.task_list.selectedItems()
        if not items:
            return

        task_id = items[0].data(Qt.ItemDataRole.UserRole)
        bridge = self._get_bridge()
        if not bridge:
            return

        bridge.task_queue.complete_task(task_id, {'verified': True})
        bridge.save_progress()
        self._load_task_queue()
        self._log(f"Task {task_id} marked complete")

    def _on_skip_task(self):
        """Skip selected task."""
        items = self.task_list.selectedItems()
        if not items:
            return

        task_id = items[0].data(Qt.ItemDataRole.UserRole)
        bridge = self._get_bridge()
        if not bridge:
            return

        bridge.task_queue.skip_task(task_id)
        bridge.save_progress()
        self._load_task_queue()
        self._log(f"Task {task_id} skipped")

    def _on_browse(self):
        """Open Neuroglancer in browse mode."""
        bridge = self._get_bridge()
        if not bridge:
            return

        try:
            # Start server if needed
            if not self._server_running:
                self._log("Starting data server...")
                bridge.start_server()
                self._server_running = True

            self._log("Opening Neuroglancer in browse mode...")
            url = bridge.launch_browse()
            self._log(f"Neuroglancer URL: {url[:60]}...")

        except Exception as e:
            self._log(f"Error: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open Neuroglancer: {e}"
            )

    def cleanup(self):
        """Clean up resources when leaving page."""
        # Stop worker if running
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

        # Stop server
        if self._bridge and self._server_running:
            try:
                self._bridge.stop_server()
            except Exception:
                pass
            self._server_running = False

    def get_result(self) -> dict:
        """Get proofreading results for wizard."""
        bridge = self._get_bridge()
        if bridge:
            return bridge.get_summary()
        return {}
