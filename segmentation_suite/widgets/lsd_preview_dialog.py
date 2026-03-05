#!/usr/bin/env python3
"""
LSD Preview Dialog - Interactive LSD predictions with navigation.

Simple preview without modifying OptimizedCanvas or Ground Truth tab.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QGroupBox, QWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QWheelEvent
from scipy.ndimage import distance_transform_edt, label as scipy_label, maximum_filter, median_filter
from skimage.segmentation import watershed
from skimage.exposure import equalize_adapthist
from skimage.transform import resize


class LSDPredictionWorker(QThread):
    """Background worker for LSD prediction on single slice."""

    prediction_ready = pyqtSignal(np.ndarray, np.ndarray)  # affinities, boundary_map
    error = pyqtSignal(str)

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.pending_slice = None
        self.pending_params = None
        self.running = True

    def predict_slice(self, image_slice, amplification, gamma, clahe_clip_limit=0.01):
        """Queue a prediction request."""
        self.pending_slice = image_slice
        self.pending_params = (amplification, gamma, clahe_clip_limit)

    def run(self):
        """Worker loop - processes pending predictions."""
        while self.running:
            if self.pending_slice is not None:
                try:
                    img = self.pending_slice
                    amp, gamma, clahe_clip = self.pending_params
                    self.pending_slice = None

                    print(f"[LSD Worker] Processing image: shape={img.shape}, dtype={img.dtype}, "
                          f"min={img.min()}, max={img.max()}")

                    # 1. CLAHE preprocessing
                    img_norm = img.astype(np.float32) / 255.0
                    img_clahe = equalize_adapthist(img_norm, clip_limit=clahe_clip)
                    img_preprocessed = (img_clahe * 255).astype(np.uint8)
                    print(f"[LSD Worker] After CLAHE: min={img_preprocessed.min()}, max={img_preprocessed.max()}")

                    # 2. Predict affinities
                    affinities = self._predict_affinity(img_preprocessed)
                    print(f"[LSD Worker] Affinities: shape={affinities.shape}, "
                          f"min={affinities.min():.3f}, max={affinities.max():.3f}")

                    # 3. Compute boundaries
                    boundary = 1.0 - affinities.mean(axis=0)
                    print(f"[LSD Worker] Boundary (before amp/gamma): min={boundary.min():.3f}, "
                          f"max={boundary.max():.3f}, mean={boundary.mean():.3f}")

                    # 4. Apply amplification + gamma
                    boundary = np.clip(boundary * amp, 0, 1)
                    boundary = np.power(boundary, gamma)
                    print(f"[LSD Worker] Boundary (after amp={amp}, gamma={gamma}): "
                          f"min={boundary.min():.3f}, max={boundary.max():.3f}, mean={boundary.mean():.3f}")

                    # 5. Median filter
                    boundary = median_filter(boundary, size=3)

                    self.prediction_ready.emit(affinities, boundary)

                except Exception as e:
                    self.error.emit(f"Prediction error: {e}")

            self.msleep(50)  # 50ms polling

    def _predict_affinity(self, img):
        """Predict affinities for single slice."""
        H, W = img.shape
        img_norm = img.astype(np.float32) / 255.0

        # Pad to multiple of 16
        H_pad = ((H + 15) // 16) * 16
        W_pad = ((W + 15) // 16) * 16
        img_padded = np.zeros((H_pad, W_pad), dtype=np.float32)
        img_padded[:H, :W] = img_norm

        # Predict
        tensor = torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            aff, _ = self.model(tensor)

        # Extract affinities
        aff_np = aff[0].cpu().numpy()[:, :H, :W]  # (2, H, W)
        return aff_np

    def stop(self):
        """Stop worker thread."""
        self.running = False


class LightweightCanvas(QWidget):
    """Lightweight canvas with pyramid support for performance."""

    slice_changed = pyqtSignal(int)  # Emitted when slice changes
    viewport_changed = pyqtSignal()  # Emitted when pan/zoom changes

    def __init__(self, zarr_source, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self.zarr_source = zarr_source
        self.current_slice = 0
        self.overlay = None  # RGB overlay
        self.overlay_pyramid_level = 2  # What pyramid level the overlay is at
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.panning = False
        self.last_pos = None

        # Cache
        self._cached_pixmap = None
        self._cache_valid = False

    def set_slice(self, slice_idx):
        """Change slice."""
        self.current_slice = slice_idx
        self._cache_valid = False
        self.update()

    def set_overlay(self, overlay, pyramid_level=2):
        """Set RGB overlay (H, W, 3) uint8."""
        self.overlay = overlay
        self.overlay_pyramid_level = pyramid_level
        self._cache_valid = False
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        """Zoom with ctrl+scroll, change slice with regular scroll."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Get mouse position
            mouse_x = event.position().x()
            mouse_y = event.position().y()

            # Calculate zoom factor
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 0.9

            old_zoom = self.zoom
            self.zoom *= factor
            self.zoom = max(0.1, min(10.0, self.zoom))
            zoom_ratio = self.zoom / old_zoom

            # Adjust offset to keep point under mouse stationary
            # The point under the mouse in canvas coordinates should map to the same
            # point in image coordinates before and after zoom
            self.offset_x = mouse_x - (mouse_x - self.offset_x) * zoom_ratio
            self.offset_y = mouse_y - (mouse_y - self.offset_y) * zoom_ratio

            self._cache_valid = False
            self.update()
            self.viewport_changed.emit()  # Trigger re-prediction on new viewport
            event.accept()
        else:
            # Change slice
            delta = event.angleDelta().y()
            step = 1 if delta > 0 else -1
            new_slice = self.current_slice + step
            new_slice = max(0, min(self.zarr_source.num_slices - 1, new_slice))

            if new_slice != self.current_slice:
                self.current_slice = new_slice
                self._cache_valid = False
                self.update()
                self.slice_changed.emit(new_slice)

            event.accept()

    def mousePressEvent(self, event):
        """Start panning."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.last_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Pan the view."""
        if self.panning and self.last_pos:
            delta = event.pos() - self.last_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """Stop panning."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.viewport_changed.emit()  # Trigger re-prediction after pan

    def paintEvent(self, event):
        """Render with pyramid optimization."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if not self._cache_valid:
            self._update_cache()

        if self._cached_pixmap is not None:
            # Draw cached pixmap at current pan position
            center_x = self.width() // 2
            center_y = self.height() // 2
            x = int(center_x - self._cached_pixmap.width() // 2 + self.offset_x)
            y = int(center_y - self._cached_pixmap.height() // 2 + self.offset_y)
            painter.drawPixmap(x, y, self._cached_pixmap)

    def _update_cache(self):
        """Update cached pixmap using appropriate pyramid level."""
        # Use pyramid level 0 (full resolution) since this Zarr only has 1 level
        level = 0
        downsample = 1

        # Load image at chosen pyramid level
        img = self.zarr_source.get_slice(self.current_slice, pyramid_level=level)

        # Ensure uint8
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        H, W = img.shape

        # Create RGB
        img_rgb = np.stack([img] * 3, axis=-1)

        # Apply overlay only where non-zero (don't darken the whole image)
        if self.overlay is not None:
            # Overlay and image should match dimensions (both at level 2)
            oh, ow = self.overlay.shape[:2]
            if oh == H and ow == W:
                # Perfect match - blend only where overlay is non-zero
                overlay_mask = self.overlay.sum(axis=2) > 0
                img_rgb[overlay_mask] = (img_rgb[overlay_mask] * 0.3 + self.overlay[overlay_mask] * 0.7).astype(np.uint8)
            else:
                # Sizes don't match - crop/pad overlay
                overlay_resized = np.zeros((H, W, 3), dtype=np.uint8)
                h_min = min(H, oh)
                w_min = min(W, ow)
                overlay_resized[:h_min, :w_min] = self.overlay[:h_min, :w_min]
                overlay_mask = overlay_resized.sum(axis=2) > 0
                img_rgb[overlay_mask] = (img_rgb[overlay_mask] * 0.3 + overlay_resized[overlay_mask] * 0.7).astype(np.uint8)

        # Convert to QPixmap
        qimg = QImage(img_rgb.data, W, H, W * 3, QImage.Format.Format_RGB888)

        # Scale by zoom (at level 0, so just scale by zoom)
        target_w = int(W * self.zoom)
        target_h = int(H * self.zoom)

        self._cached_pixmap = QPixmap.fromImage(qimg).scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self._cache_valid = True

    def get_viewport_bounds(self):
        """Get current viewport bounds in full-resolution coordinates.

        Returns (z, y0, y1, x0, x1) where coordinates are at pyramid level 0.
        """
        # We display at pyramid level 0 (full resolution)
        # Get canvas dimensions
        canvas_w = self.width()
        canvas_h = self.height()

        # Get full image dimensions at level 0
        img_full = self.zarr_source.get_slice(self.current_slice, pyramid_level=0)
        H_full, W_full = img_full.shape

        # Pixmap dimensions after zoom scaling
        pixmap_w = int(W_full * self.zoom)
        pixmap_h = int(H_full * self.zoom)

        # Pixmap position on canvas
        center_x = canvas_w // 2
        center_y = canvas_h // 2
        pixmap_x = int(center_x - pixmap_w // 2 + self.offset_x)
        pixmap_y = int(center_y - pixmap_h // 2 + self.offset_y)

        # Visible region of pixmap (in pixmap coordinates)
        visible_x0 = max(0, -pixmap_x)
        visible_y0 = max(0, -pixmap_y)
        visible_x1 = min(pixmap_w, canvas_w - pixmap_x)
        visible_y1 = min(pixmap_h, canvas_h - pixmap_y)

        # Ensure we have a valid region
        if visible_x1 <= visible_x0 or visible_y1 <= visible_y0:
            # No visible region - return centered region
            y0_full = max(0, H_full // 2 - 512)
            y1_full = min(H_full, H_full // 2 + 512)
            x0_full = max(0, W_full // 2 - 512)
            x1_full = min(W_full, W_full // 2 + 512)
            return self.current_slice, y0_full, y1_full, x0_full, x1_full

        # Convert pixmap coordinates to image coordinates
        scale = self.zoom
        img_x0 = int(visible_x0 / scale)
        img_y0 = int(visible_y0 / scale)
        img_x1 = int(visible_x1 / scale)
        img_y1 = int(visible_y1 / scale)

        # Clamp to valid bounds
        x0_full = max(0, min(img_x0, W_full - 1))
        y0_full = max(0, min(img_y0, H_full - 1))
        x1_full = max(x0_full + 1, min(img_x1, W_full))
        y1_full = max(y0_full + 1, min(img_y1, H_full))

        return self.current_slice, y0_full, y1_full, x0_full, x1_full


class LSDPreviewDialog(QDialog):
    """Interactive LSD preview."""

    def __init__(self, zarr_source, initial_slice, model_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LSD Preview - Interactive")
        self.setMinimumSize(1000, 800)

        self.zarr_source = zarr_source
        self.current_slice = initial_slice
        self.model_path = Path(model_path)

        # Add scripts to path
        scripts_path = Path(__file__).parent.parent.parent.parent / 'scripts'
        if str(scripts_path) not in sys.path:
            sys.path.insert(0, str(scripts_path))

        # Load model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()

        # Parameters
        self.amplification = 2.0
        self.gamma = 4.0
        self.seed_threshold = 0.7
        self.min_distance = 15
        self.min_distance_value = 7
        self.raw_avg_window = 3
        self.clahe_clip_limit = 0.01
        self.min_segment_size = 100

        # Current data
        self.current_image = None
        self.boundary_map = None
        self.segmentation = None
        self.viewport_bounds = None  # (y0, y1, x0, x1) in full-res coords
        self.show_boundaries = True
        self.show_segments = True

        # Background worker
        self.worker = LSDPredictionWorker(self.model, self.device)
        self.worker.prediction_ready.connect(self._on_prediction_ready)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self._init_ui()
        self._load_and_predict_slice(initial_slice)

    def _load_model(self):
        """Load MtLSD model."""
        from ..mtlsd_model_compat import MtLsdModel
        print(f"Loading LSD model...")
        self.model = MtLsdModel.from_pretrained(str(self.model_path), self.device)
        self.model.eval()

    def _init_ui(self):
        """Build UI."""
        layout = QVBoxLayout()

        # Canvas
        self.canvas = LightweightCanvas(self.zarr_source)
        self.canvas.set_slice(self.current_slice)
        self.canvas.slice_changed.connect(self._on_slice_changed)
        self.canvas.viewport_changed.connect(self._on_viewport_changed)
        layout.addWidget(self.canvas, stretch=3)

        # Controls
        controls = self._create_controls()
        layout.addWidget(controls, stretch=1)

        # Status bar
        status_layout = QHBoxLayout()
        self.slice_label = QLabel(f"Slice: {self.current_slice} / {self.zarr_source.num_slices - 1}")
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888;")
        help_label = QLabel("Mouse wheel=slice | Ctrl+scroll=zoom | Middle-drag=pan")
        help_label.setStyleSheet("color: #888888; font-size: 11px;")

        status_layout.addWidget(self.slice_label)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(help_label)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        status_layout.addWidget(close_btn)

        layout.addLayout(status_layout)
        self.setLayout(layout)

    def _create_controls(self):
        """Create controls."""
        group = QGroupBox("LSD Parameters")
        layout = QVBoxLayout()

        # Overlays
        overlay_layout = QHBoxLayout()
        self.boundaries_check = QCheckBox("Show Boundaries")
        self.boundaries_check.setChecked(True)
        self.boundaries_check.stateChanged.connect(self._update_display)
        self.segments_check = QCheckBox("Show Segments")
        self.segments_check.setChecked(True)
        self.segments_check.stateChanged.connect(self._update_display)
        overlay_layout.addWidget(self.boundaries_check)
        overlay_layout.addWidget(self.segments_check)
        overlay_layout.addStretch()
        layout.addLayout(overlay_layout)

        # Amplification
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(QLabel("Boundary Amplification:"))
        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setRange(5, 50)
        self.amp_slider.setValue(20)
        self.amp_value = QLabel("2.0")
        self.amp_slider.valueChanged.connect(self._on_amp_changed)
        amp_layout.addWidget(self.amp_slider, 3)
        amp_layout.addWidget(self.amp_value)
        layout.addLayout(amp_layout)

        # Gamma
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(10, 100)
        self.gamma_slider.setValue(40)
        self.gamma_value = QLabel("4.0")
        self.gamma_slider.valueChanged.connect(self._on_gamma_changed)
        gamma_layout.addWidget(self.gamma_slider, 3)
        gamma_layout.addWidget(self.gamma_value)
        layout.addLayout(gamma_layout)

        # Threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Seed Threshold:"))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(30, 90)
        self.thresh_slider.setValue(70)
        self.thresh_value = QLabel("0.70")
        self.thresh_slider.valueChanged.connect(self._on_thresh_changed)
        thresh_layout.addWidget(self.thresh_slider, 3)
        thresh_layout.addWidget(self.thresh_value)
        layout.addLayout(thresh_layout)

        # Distance
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Min Seed Distance:"))
        self.dist_slider = QSlider(Qt.Orientation.Horizontal)
        self.dist_slider.setRange(5, 30)
        self.dist_slider.setValue(15)
        self.dist_value = QLabel("15")
        self.dist_slider.valueChanged.connect(self._on_dist_changed)
        dist_layout.addWidget(self.dist_slider, 3)
        dist_layout.addWidget(self.dist_value)
        layout.addLayout(dist_layout)

        # Min Distance Value
        min_dist_val_layout = QHBoxLayout()
        min_dist_val_layout.addWidget(QLabel("Min Distance Value:"))
        self.min_dist_val_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_dist_val_slider.setRange(1, 20)
        self.min_dist_val_slider.setValue(7)
        self.min_dist_val_value = QLabel("7")
        self.min_dist_val_slider.valueChanged.connect(self._on_min_dist_val_changed)
        min_dist_val_layout.addWidget(self.min_dist_val_slider, 3)
        min_dist_val_layout.addWidget(self.min_dist_val_value)
        layout.addLayout(min_dist_val_layout)

        # Raw Avg Window
        raw_avg_layout = QHBoxLayout()
        raw_avg_layout.addWidget(QLabel("Raw Avg Window:"))
        self.raw_avg_slider = QSlider(Qt.Orientation.Horizontal)
        self.raw_avg_slider.setRange(1, 11)
        self.raw_avg_slider.setValue(3)
        self.raw_avg_value = QLabel("3")
        self.raw_avg_slider.valueChanged.connect(self._on_raw_avg_changed)
        raw_avg_layout.addWidget(self.raw_avg_slider, 3)
        raw_avg_layout.addWidget(self.raw_avg_value)
        layout.addLayout(raw_avg_layout)

        # CLAHE Clip Limit
        clahe_layout = QHBoxLayout()
        clahe_layout.addWidget(QLabel("CLAHE Clip Limit:"))
        self.clahe_slider = QSlider(Qt.Orientation.Horizontal)
        self.clahe_slider.setRange(1, 100)
        self.clahe_slider.setValue(10)
        self.clahe_value = QLabel("0.010")
        self.clahe_slider.valueChanged.connect(self._on_clahe_changed)
        clahe_layout.addWidget(self.clahe_slider, 3)
        clahe_layout.addWidget(self.clahe_value)
        layout.addLayout(clahe_layout)

        # Min Segment Size
        min_seg_layout = QHBoxLayout()
        min_seg_layout.addWidget(QLabel("Min Segment Size:"))
        self.min_seg_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_seg_slider.setRange(10, 1000)
        self.min_seg_slider.setValue(100)
        self.min_seg_value = QLabel("100")
        self.min_seg_slider.valueChanged.connect(self._on_min_seg_changed)
        min_seg_layout.addWidget(self.min_seg_slider, 3)
        min_seg_layout.addWidget(self.min_seg_value)
        layout.addLayout(min_seg_layout)

        group.setLayout(layout)
        return group

    def wheelEvent(self, event):
        """Handle mouse wheel for slice navigation."""
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            delta = event.angleDelta().y()
            if delta > 0:
                self.current_slice = min(self.current_slice + 1, self.zarr_source.num_slices - 1)
            else:
                self.current_slice = max(self.current_slice - 1, 0)
            self._load_and_predict_slice(self.current_slice)
            event.accept()

    def _on_amp_changed(self, value):
        self.amplification = value / 10.0
        self.amp_value.setText(f"{self.amplification:.1f}")
        if self.current_image is not None:
            self.worker.predict_slice(self.current_image, self.amplification, self.gamma, self.clahe_clip_limit)

    def _on_gamma_changed(self, value):
        self.gamma = value / 10.0
        self.gamma_value.setText(f"{self.gamma:.1f}")
        if self.current_image is not None:
            self.worker.predict_slice(self.current_image, self.amplification, self.gamma, self.clahe_clip_limit)

    def _on_thresh_changed(self, value):
        self.seed_threshold = value / 100.0
        self.thresh_value.setText(f"{self.seed_threshold:.2f}")
        if self.boundary_map is not None:
            self._recompute_watershed()
            self._update_display()

    def _on_dist_changed(self, value):
        self.min_distance = value
        self.dist_value.setText(f"{self.min_distance}")
        if self.boundary_map is not None:
            self._recompute_watershed()
            self._update_display()

    def _on_min_dist_val_changed(self, value):
        self.min_distance_value = value
        self.min_dist_val_value.setText(f"{self.min_distance_value}")
        if self.boundary_map is not None:
            self._recompute_watershed()
            self._update_display()

    def _on_raw_avg_changed(self, value):
        # Ensure odd values
        if value % 2 == 0:
            value = value + 1
            self.raw_avg_slider.setValue(value)
        self.raw_avg_window = value
        self.raw_avg_value.setText(f"{self.raw_avg_window}")
        # Raw avg affects preprocessing, so we need to reload and re-predict
        if self.current_image is not None:
            self._load_and_predict_slice(self.current_slice)

    def _on_clahe_changed(self, value):
        self.clahe_clip_limit = value / 1000.0
        self.clahe_value.setText(f"{self.clahe_clip_limit:.3f}")
        # CLAHE affects preprocessing, so we need to reload and re-predict
        if self.current_image is not None:
            self._load_and_predict_slice(self.current_slice)

    def _on_min_seg_changed(self, value):
        self.min_segment_size = value
        self.min_seg_value.setText(f"{self.min_segment_size}")
        if self.boundary_map is not None:
            self._recompute_watershed()
            self._update_display()

    def _load_and_predict_slice(self, slice_idx):
        """Load and predict only the visible viewport."""
        try:
            self.slice_label.setText(f"Slice: {slice_idx} / {self.zarr_source.num_slices - 1}")
            self.status_label.setText("Loading...")

            # Update canvas (canvas handles its own pyramid level selection)
            self.canvas.set_slice(slice_idx)

            # Get viewport bounds in full-resolution coordinates
            z, y0, y1, x0, x1 = self.canvas.get_viewport_bounds()

            # Clamp to valid bounds
            img_full = self.zarr_source.get_slice(z, pyramid_level=0)
            max_h, max_w = img_full.shape
            y0 = max(0, min(y0, max_h - 1))
            y1 = max(y0 + 1, min(y1, max_h))
            x0 = max(0, min(x0, max_w - 1))
            x1 = max(x0 + 1, min(x1, max_w))

            # Extract viewport crop at full resolution
            img_crop = img_full[y0:y1, x0:x1].copy()

            # Validate crop size - only fall back if completely invalid
            if img_crop.size == 0 or img_crop.shape[0] < 10 or img_crop.shape[1] < 10:
                print(f"[LSD Preview] Invalid crop! bounds=({y0},{y1},{x0},{x1}), shape={img_full.shape}")
                # Fallback to small centered region
                min_size = 256
                y0 = max(0, max_h // 2 - min_size // 2)
                y1 = min(max_h, y0 + min_size)
                x0 = max(0, max_w // 2 - min_size // 2)
                x1 = min(max_w, x0 + min_size)
                img_crop = img_full[y0:y1, x0:x1].copy()
                print(f"[LSD Preview] Fallback to centered region: ({y0},{y1},{x0},{x1})")
            else:
                print(f"[LSD Preview] Processing viewport: ({y0},{y1},{x0},{x1}), size={img_crop.shape}")

            # Convert to uint8
            if img_crop.dtype != np.uint8:
                crop_max = img_crop.max() if img_crop.size > 0 else 1.0
                if crop_max <= 1.0:
                    img_crop = (img_crop * 255).astype(np.uint8)
                else:
                    img_crop = img_crop.astype(np.uint8)

            # Store crop bounds for later overlay placement
            self.viewport_bounds = (y0, y1, x0, x1)
            self.current_image = img_crop
            self.worker.predict_slice(img_crop, self.amplification, self.gamma, self.clahe_clip_limit)

        except Exception as e:
            print(f"[LSD Preview] Error loading slice: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Error: {e}")

    def _on_slice_changed(self, new_slice):
        """Handle slice change from canvas."""
        self.current_slice = new_slice
        self._load_and_predict_slice(new_slice)

    def _on_viewport_changed(self):
        """Handle viewport change (pan/zoom) - re-predict on new viewport."""
        if self.current_slice is not None:
            self._load_and_predict_slice(self.current_slice)

    def _on_prediction_ready(self, affinities, boundary_map):
        """Prediction ready."""
        print(f"[LSD Preview] Prediction ready: boundary_map shape={boundary_map.shape}, "
              f"min={boundary_map.min():.3f}, max={boundary_map.max():.3f}, "
              f"mean={boundary_map.mean():.3f}")
        self.boundary_map = boundary_map
        self._recompute_watershed()
        self._update_display()
        n_segs = self.segmentation.max() if self.segmentation is not None and self.segmentation.size > 0 else 0
        self.status_label.setText(f"Ready - {int(n_segs)} segments")

    def _on_error(self, error_msg):
        """Error."""
        self.status_label.setText(f"Error: {error_msg}")

    def _recompute_watershed(self):
        """Run 2D watershed."""
        if self.boundary_map is None:
            return

        interior = self.boundary_map < self.seed_threshold
        interior_frac = interior.sum() / interior.size
        print(f"[LSD Preview] Interior: {interior.sum()} / {interior.size} pixels ({interior_frac*100:.1f}%)")

        distances = distance_transform_edt(interior)
        dist_max = distances.max() if distances.size > 0 else 0
        print(f"[LSD Preview] Distance transform: max={dist_max:.1f}")

        footprint = np.ones((self.min_distance, self.min_distance))
        local_max = (maximum_filter(distances, footprint=footprint) == distances)
        local_max &= distances > self.min_distance_value
        n_peaks = local_max.sum()
        print(f"[LSD Preview] Found {n_peaks} local maxima (min_distance={self.min_distance}, min_value={self.min_distance_value})")

        seeds, n_seeds = scipy_label(local_max)
        print(f"[LSD Preview] Watershed: {n_seeds} seeds")

        if n_seeds == 0:
            self.segmentation = np.zeros_like(self.boundary_map, dtype=np.uint32)
        else:
            self.segmentation = watershed(self.boundary_map, seeds, mask=interior).astype(np.uint32)
            n_segments_before = int(self.segmentation.max())
            print(f"[LSD Preview] Watershed produced {n_segments_before} segments")

            # Filter out small segments
            if self.min_segment_size > 0:
                for label_id in np.unique(self.segmentation):
                    if label_id == 0:
                        continue
                    if np.sum(self.segmentation == label_id) < self.min_segment_size:
                        self.segmentation[self.segmentation == label_id] = 0
                n_segments_after = int(self.segmentation.max())
                print(f"[LSD Preview] After filtering: {n_segments_after} segments (removed {n_segments_before - n_segments_after})")

    def _update_display(self):
        """Update display - create overlay for viewport crop."""
        if self.boundary_map is None or self.viewport_bounds is None:
            return

        try:
            self.show_boundaries = self.boundaries_check.isChecked()
            self.show_segments = self.segments_check.isChecked()

            # Create overlay for the crop at full resolution
            H, W = self.boundary_map.shape
            if H == 0 or W == 0:
                print(f"[LSD Preview] Warning: Empty boundary map! shape={self.boundary_map.shape}")
                return

            # Start with black background
            overlay_crop = np.zeros((H, W, 3), dtype=np.uint8)

            # Draw segments first (as base layer)
            if self.show_segments and self.segmentation is not None:
                seg_max = self.segmentation.max() if self.segmentation.size > 0 else 0
                n = int(seg_max) + 1
                if n > 1:
                    np.random.seed(42)
                    colors = (np.random.rand(n, 3) * 200 + 55).astype(np.uint8)  # Brighter colors
                    colors[0] = [0, 0, 0]  # Background stays black
                    segment_rgb = colors[self.segmentation]
                    overlay_crop = segment_rgb

            # Draw boundaries on top (bright red, full opacity)
            if self.show_boundaries:
                boundary_mask = self.boundary_map > 0.3  # Threshold for visibility
                overlay_crop[boundary_mask] = [255, 0, 0]  # Bright red

            # No downsampling - work at full resolution (level 0)
            overlay_crop_fullres = overlay_crop

            # Get viewport position (already at full resolution)
            y0, y1, x0, x1 = self.viewport_bounds

            # Create full-sized overlay at level 0
            img_level0 = self.zarr_source.get_slice(self.current_slice, pyramid_level=0)
            full_h, full_w = img_level0.shape
            overlay_full = np.zeros((full_h, full_w, 3), dtype=np.uint8)

            # Place crop overlay in correct position (with bounds checking)
            crop_h, crop_w = overlay_crop_fullres.shape[:2]
            y0_clamped = max(0, min(y0, full_h - 1))
            x0_clamped = max(0, min(x0, full_w - 1))
            y1_place = min(y0_clamped + crop_h, full_h)
            x1_place = min(x0_clamped + crop_w, full_w)

            # Only place if we have valid region
            if y1_place > y0_clamped and x1_place > x0_clamped:
                crop_h_valid = y1_place - y0_clamped
                crop_w_valid = x1_place - x0_clamped
                overlay_full[y0_clamped:y1_place, x0_clamped:x1_place] = overlay_crop_fullres[:crop_h_valid, :crop_w_valid]

            print(f"[LSD Preview] Overlay: crop={H}x{W}, fullres={crop_h}x{crop_w}, "
                  f"placed at ({y0_clamped},{x0_clamped}) in {full_h}x{full_w} canvas, "
                  f"non-zero pixels={np.count_nonzero(overlay_full)}")

            self.canvas.set_overlay(overlay_full, pyramid_level=0)

        except Exception as e:
            print(f"[LSD Preview] Error updating display: {e}")
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        """Cleanup."""
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(event)
