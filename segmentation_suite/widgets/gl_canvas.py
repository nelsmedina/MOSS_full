#!/usr/bin/env python3
"""
OpenGL-accelerated canvas for smooth tile-based rendering.
Uses GPU textures for fast panning and zooming.
"""

import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QWheelEvent, QMouseEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLTexture
from OpenGL.GL import *
from OpenGL.GLU import *


class GLCanvas(QOpenGLWidget):
    """OpenGL-accelerated canvas with tile caching."""

    viewport_changed = pyqtSignal()
    edit_made = pyqtSignal()
    brush_size_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # Zarr source
        self._zarr_source = None
        self._slice_idx = 0
        self._global_min = 0
        self._global_max = 255

        # Image dimensions (full res)
        self._img_width = 0
        self._img_height = 0

        # View state
        self.zoom_level = 1.0
        self.offset = QPoint(0, 0)

        # Mask (full resolution, edited by user)
        self.mask = None
        self.mask_alpha = 0.5

        # Suggestion overlay
        self.suggestion = None
        self.show_suggestion = True

        # Brush
        self.brush_size = 10
        self.erasing = False
        self.drawing = False
        self.cursor_pos = None

        # Pan state
        self._panning = False
        self._last_mouse_pos = None

        # Tile cache: {(level, tile_y, tile_x): QOpenGLTexture}
        self._tile_textures = {}
        self._tile_size = 512  # Tile size in pixels at that pyramid level
        self._max_cached_tiles = 64

        # Mask texture (updated on edit)
        self._mask_texture = None
        self._mask_texture_dirty = True

        # For drawing
        self.mask_before_edit = None

    def set_zarr_source(self, zarr_source, slice_index: int):
        """Set Zarr source for rendering."""
        if self._slice_idx != slice_index:
            self._clear_tile_cache()

        self._zarr_source = zarr_source
        self._slice_idx = slice_index
        self._global_min = zarr_source.global_min or 0
        self._global_max = zarr_source.global_max or 255
        self._img_width = zarr_source.width
        self._img_height = zarr_source.height

        self.update()

    def set_mask(self, mask: np.ndarray):
        """Set the segmentation mask."""
        self.mask = mask
        self._mask_texture_dirty = True
        self.update()

    def set_suggestion(self, suggestion: np.ndarray):
        """Set the AI suggestion overlay."""
        self.suggestion = suggestion
        self.update()

    def set_mask_alpha(self, alpha: float):
        """Set mask overlay alpha."""
        self.mask_alpha = max(0.0, min(1.0, alpha))
        self.update()

    def toggle_suggestion_visibility(self, visible: bool):
        """Toggle suggestion visibility."""
        self.show_suggestion = visible
        self.update()

    def _clear_tile_cache(self):
        """Clear all cached tiles."""
        # Delete OpenGL textures
        if self._tile_textures:
            for tex in self._tile_textures.values():
                if tex is not None:
                    tex.destroy()
            self._tile_textures.clear()

    def initializeGL(self):
        """Initialize OpenGL state."""
        glClearColor(0.16, 0.16, 0.16, 1.0)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        """Handle resize."""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)  # Top-left origin
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        """Render the scene."""
        glClear(GL_COLOR_BUFFER_BIT)

        if self._zarr_source is None or self._img_width == 0:
            return

        # Calculate image position on screen
        scaled_w = int(self._img_width * self.zoom_level)
        scaled_h = int(self._img_height * self.zoom_level)
        img_x = (self.width() - scaled_w) // 2 + self.offset.x()
        img_y = (self.height() - scaled_h) // 2 + self.offset.y()

        # Select pyramid level
        level_idx, level_path, downsample = self._zarr_source.select_pyramid_level(self.zoom_level)

        # Calculate visible region in full-res coordinates
        vis_left = max(0, int(-img_x / self.zoom_level))
        vis_top = max(0, int(-img_y / self.zoom_level))
        vis_right = min(self._img_width, int((self.width() - img_x) / self.zoom_level) + 1)
        vis_bottom = min(self._img_height, int((self.height() - img_y) / self.zoom_level) + 1)

        if vis_right <= vis_left or vis_bottom <= vis_top:
            return

        # Tile size in full-res coordinates
        tile_size_fullres = self._tile_size * downsample

        # Calculate which tiles are visible
        tile_y_start = vis_top // tile_size_fullres
        tile_y_end = (vis_bottom + tile_size_fullres - 1) // tile_size_fullres
        tile_x_start = vis_left // tile_size_fullres
        tile_x_end = (vis_right + tile_size_fullres - 1) // tile_size_fullres

        # Render visible tiles
        for tile_y in range(tile_y_start, tile_y_end):
            for tile_x in range(tile_x_start, tile_x_end):
                self._render_tile(level_idx, downsample, tile_y, tile_x, img_x, img_y)

        # Render mask overlay
        if self.mask is not None and self.mask_alpha > 0:
            self._render_mask_overlay(img_x, img_y, vis_left, vis_top, vis_right, vis_bottom, downsample)

        # Render brush cursor
        if self.cursor_pos is not None:
            self._render_brush_cursor()

    def _render_tile(self, level_idx, downsample, tile_y, tile_x, img_x, img_y):
        """Render a single tile."""
        cache_key = (self._slice_idx, level_idx, tile_y, tile_x)

        # Get or create texture
        if cache_key not in self._tile_textures:
            # Load tile from Zarr
            tile_size_fullres = self._tile_size * downsample
            y1 = tile_y * tile_size_fullres
            y2 = min((tile_y + 1) * tile_size_fullres, self._img_height)
            x1 = tile_x * tile_size_fullres
            x2 = min((tile_x + 1) * tile_size_fullres, self._img_width)

            tile_data, _ = self._zarr_source.get_tile_native(
                self._slice_idx, y1, y2, x1, x2, level_idx
            )

            if tile_data.size == 0:
                return

            # Normalize
            scale = 255.0 / max(self._global_max - self._global_min, 1)
            tile_data = ((tile_data - self._global_min) * scale).clip(0, 255).astype(np.uint8)

            # Create texture
            tex = self._create_texture(tile_data)

            # Cache eviction
            if len(self._tile_textures) >= self._max_cached_tiles:
                oldest_key = next(iter(self._tile_textures))
                old_tex = self._tile_textures.pop(oldest_key)
                if old_tex:
                    old_tex.destroy()

            self._tile_textures[cache_key] = tex
        else:
            tex = self._tile_textures[cache_key]

        if tex is None:
            return

        # Calculate screen coordinates
        tile_size_fullres = self._tile_size * downsample
        screen_x = img_x + tile_x * tile_size_fullres * self.zoom_level
        screen_y = img_y + tile_y * tile_size_fullres * self.zoom_level
        screen_w = tex.width() * downsample * self.zoom_level
        screen_h = tex.height() * downsample * self.zoom_level

        # Draw textured quad
        tex.bind()
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(screen_x, screen_y)
        glTexCoord2f(1, 0); glVertex2f(screen_x + screen_w, screen_y)
        glTexCoord2f(1, 1); glVertex2f(screen_x + screen_w, screen_y + screen_h)
        glTexCoord2f(0, 1); glVertex2f(screen_x, screen_y + screen_h)
        glEnd()
        tex.release()

    def _create_texture(self, data: np.ndarray) -> QOpenGLTexture:
        """Create an OpenGL texture from grayscale data."""
        h, w = data.shape

        # Convert to RGB for texture
        rgb = np.stack([data, data, data], axis=-1)
        rgb = np.ascontiguousarray(rgb)

        tex = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        tex.setFormat(QOpenGLTexture.TextureFormat.RGB8_UNorm)
        tex.setSize(w, h)
        tex.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
        tex.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        tex.allocateStorage()
        tex.setData(QOpenGLTexture.PixelFormat.RGB, QOpenGLTexture.PixelType.UInt8, rgb.tobytes())

        return tex

    def _render_mask_overlay(self, img_x, img_y, vis_left, vis_top, vis_right, vis_bottom, downsample):
        """Render mask as red overlay."""
        if self.mask is None:
            return

        # Get visible portion of mask
        mask_region = self.mask[vis_top:vis_bottom, vis_left:vis_right]
        if mask_region.size == 0:
            return

        # Downsample for display
        if downsample > 1:
            mask_region = mask_region[::downsample, ::downsample]

        h, w = mask_region.shape
        if h == 0 or w == 0:
            return

        # Create RGBA with red channel for mask
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_bool = mask_region > 127
        rgba[mask_bool, 0] = 255  # Red
        rgba[mask_bool, 3] = int(255 * self.mask_alpha)  # Alpha
        rgba = np.ascontiguousarray(rgba)

        # Create temporary texture
        tex = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        tex.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        tex.setSize(w, h)
        tex.setMinMagFilters(QOpenGLTexture.Filter.Nearest, QOpenGLTexture.Filter.Nearest)
        tex.allocateStorage()
        tex.setData(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8, rgba.tobytes())

        # Calculate screen position
        screen_x = img_x + vis_left * self.zoom_level
        screen_y = img_y + vis_top * self.zoom_level
        screen_w = w * downsample * self.zoom_level
        screen_h = h * downsample * self.zoom_level

        # Draw
        tex.bind()
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(screen_x, screen_y)
        glTexCoord2f(1, 0); glVertex2f(screen_x + screen_w, screen_y)
        glTexCoord2f(1, 1); glVertex2f(screen_x + screen_w, screen_y + screen_h)
        glTexCoord2f(0, 1); glVertex2f(screen_x, screen_y + screen_h)
        glEnd()
        tex.release()
        tex.destroy()

    def _render_brush_cursor(self):
        """Render brush cursor circle."""
        if self.cursor_pos is None:
            return

        # Draw circle at cursor
        glDisable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 0.7)
        glLineWidth(2)

        cx, cy = self.cursor_pos.x(), self.cursor_pos.y()
        radius = self.brush_size * self.zoom_level

        glBegin(GL_LINE_LOOP)
        segments = 32
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            glVertex2f(x, y)
        glEnd()

        glEnable(GL_TEXTURE_2D)

    # Mouse handling
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_mouse_pos = event.pos()
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.mask is not None:
                self.drawing = True
                self.mask_before_edit = self.mask.copy()
                self._draw_at(event.pos())
        elif event.button() == Qt.MouseButton.RightButton:
            if self.mask is not None:
                self.drawing = True
                self.erasing = True
                self.mask_before_edit = self.mask.copy()
                self._draw_at(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        self.cursor_pos = event.pos()

        if self._panning and self._last_mouse_pos:
            delta = event.pos() - self._last_mouse_pos
            self.offset += delta
            self._last_mouse_pos = event.pos()
            self.viewport_changed.emit()
        elif self.drawing:
            self._draw_at(event.pos())

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._last_mouse_pos = None
        elif event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            if self.drawing:
                self.drawing = False
                self.erasing = False
                self.edit_made.emit()

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Brush size
            delta = event.angleDelta().y()
            if delta > 0:
                self.brush_size = min(100, self.brush_size + 2)
            else:
                self.brush_size = max(1, self.brush_size - 2)
            self.brush_size_changed.emit(self.brush_size)
        else:
            # Zoom
            old_zoom = self.zoom_level
            factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
            self.zoom_level = max(0.05, self.zoom_level * factor)

            # Zoom toward cursor
            if self.cursor_pos:
                # Adjust offset to keep point under cursor stationary
                cx, cy = self.cursor_pos.x(), self.cursor_pos.y()
                self.offset.setX(int(cx - (cx - self.offset.x()) * self.zoom_level / old_zoom))
                self.offset.setY(int(cy - (cy - self.offset.y()) * self.zoom_level / old_zoom))

            self._clear_tile_cache()  # Clear cache on zoom change
            self.viewport_changed.emit()

        self.update()

    def _draw_at(self, pos: QPoint):
        """Draw on the mask at screen position."""
        if self.mask is None:
            return

        # Convert screen to image coordinates
        scaled_w = int(self._img_width * self.zoom_level)
        scaled_h = int(self._img_height * self.zoom_level)
        img_x = (self.width() - scaled_w) // 2 + self.offset.x()
        img_y = (self.height() - scaled_h) // 2 + self.offset.y()

        px = int((pos.x() - img_x) / self.zoom_level)
        py = int((pos.y() - img_y) / self.zoom_level)

        if px < 0 or py < 0 or px >= self._img_width or py >= self._img_height:
            return

        # Draw circle on mask
        h, w = self.mask.shape
        value = 0 if self.erasing else 255
        radius = self.brush_size

        y_min = max(0, py - radius)
        y_max = min(h, py + radius + 1)
        x_min = max(0, px - radius)
        x_max = min(w, px + radius + 1)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        circle = (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2
        self.mask[y_min:y_max, x_min:x_max][circle] = value

        self._mask_texture_dirty = True
        self.update()
