#!/usr/bin/env python3
"""
Zarr-backed canvas that loads pyramid levels based on zoom level.
Extends OptimizedCanvas to support efficient rendering of large OME-Zarr volumes.
"""

import numpy as np
from PyQt6.QtGui import QPixmap, QImage
from scipy.ndimage import zoom as scipy_zoom

from .optimized_canvas import OptimizedCanvas


class ZarrCanvas(OptimizedCanvas):
    """Canvas that renders from Zarr pyramid levels based on zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Zarr state
        self.zarr_group = None  # Root zarr group
        self.zarr_slice_idx = 0  # Current slice index
        self.zarr_pyramid_paths = []  # List of pyramid level paths (e.g., ['0', 's1', 's2', 's3'])
        self.zarr_downsample_factors = []  # Downsample factor for each level (e.g., [1, 2, 4, 8])

    def set_zarr_store(self, zarr_group, slice_idx: int):
        """
        Set the Zarr store to use for rendering.

        Args:
            zarr_group: Zarr group containing the pyramid levels
            slice_idx: Index of the slice to display
        """
        self.zarr_group = zarr_group
        self.zarr_slice_idx = slice_idx

        # Discover pyramid levels from .zattrs
        self._discover_pyramid_levels()

        # Load full resolution image metadata (for dimensions)
        if self.zarr_pyramid_paths:
            level_0 = self.zarr_group[self.zarr_pyramid_paths[0]]
            # Assume shape is (slices, height, width)
            if len(level_0.shape) == 3:
                self.raw_image = np.zeros(level_0.shape[1:], dtype=np.float32)
            else:
                # Fallback for 2D
                self.raw_image = np.zeros(level_0.shape, dtype=np.float32)

        self._cache_valid = False
        self.update()

    def _discover_pyramid_levels(self):
        """Discover available pyramid levels from Zarr metadata."""
        if self.zarr_group is None:
            return

        # Check for OME-Zarr multiscales metadata
        if '.zattrs' in self.zarr_group:
            attrs = dict(self.zarr_group.attrs)
            if 'multiscales' in attrs:
                multiscales = attrs['multiscales'][0]  # First multiscale
                datasets = multiscales.get('datasets', [])

                self.zarr_pyramid_paths = []
                self.zarr_downsample_factors = []

                for i, ds in enumerate(datasets):
                    path = ds.get('path', str(i))

                    # FIX: Check if the path actually exists, otherwise try common alternatives
                    if path not in self.zarr_group:
                        # Try common naming patterns
                        alternatives = [f's{i}', str(i), f'level_{i}']
                        path_found = False
                        for alt_path in alternatives:
                            if alt_path in self.zarr_group:
                                path = alt_path
                                path_found = True
                                break

                        if not path_found:
                            print(f"[ZarrCanvas] Warning: Path '{ds.get('path')}' not found in zarr group, skipping")
                            continue

                    self.zarr_pyramid_paths.append(path)

                    # Extract downsample factor from coordinateTransformations or assume power of 2
                    transforms = ds.get('coordinateTransformations', [])
                    scale_factor = 1
                    for transform in transforms:
                        if transform.get('type') == 'scale':
                            scale = transform.get('scale', [1, 1, 1])
                            # For 3D volumes, scale is [z, y, x], we use y or x
                            scale_factor = int(scale[-1]) if len(scale) > 1 else 1
                            break

                    if scale_factor == 1 and i > 0:
                        # Fallback: assume power of 2 downsampling
                        scale_factor = 2 ** i

                    self.zarr_downsample_factors.append(scale_factor)

                print(f"[ZarrCanvas] Discovered {len(self.zarr_pyramid_paths)} pyramid levels:")
                for path, factor in zip(self.zarr_pyramid_paths, self.zarr_downsample_factors):
                    print(f"  {path}: {factor}x downsample")
            else:
                # No multiscales - use single level
                self.zarr_pyramid_paths = ['0']
                self.zarr_downsample_factors = [1]
        else:
            # Fallback: check for common pyramid level names
            common_paths = ['0', 's1', 's2', 's3', 's4']
            self.zarr_pyramid_paths = []
            self.zarr_downsample_factors = []

            for i, path in enumerate(common_paths):
                if path in self.zarr_group:
                    self.zarr_pyramid_paths.append(path)
                    self.zarr_downsample_factors.append(2 ** i)

            if not self.zarr_pyramid_paths:
                # Last resort: use '0' if it exists
                if '0' in self.zarr_group:
                    self.zarr_pyramid_paths = ['0']
                    self.zarr_downsample_factors = [1]

    def _select_pyramid_level(self, zoom_level: float):
        """
        Select the appropriate pyramid level based on zoom level.

        Args:
            zoom_level: Current canvas zoom level

        Returns:
            tuple: (level_index, level_path, downsample_factor)
        """
        if not self.zarr_pyramid_paths:
            return 0, '0', 1

        # Selection logic based on zoom level:
        # zoom >= 0.5: use level 0 (full res)
        # zoom >= 0.25: use level s1 (1/2 res)
        # zoom >= 0.125: use level s2 (1/4 res)
        # zoom < 0.125: use level s3 (1/8 res)

        # Find the best level where downsample_factor <= 1/zoom_level
        # This ensures we have enough resolution for the current zoom
        target_downsample = 1.0 / zoom_level

        best_level = 0
        for i in range(len(self.zarr_pyramid_paths) - 1, -1, -1):
            if self.zarr_downsample_factors[i] <= target_downsample:
                best_level = i
                break

        # For very high zoom (>1.0), always use level 0
        if zoom_level >= 1.0:
            best_level = 0

        return best_level, self.zarr_pyramid_paths[best_level], self.zarr_downsample_factors[best_level]

    def _render_region(self, x1: int, y1: int, x2: int, y2: int) -> QPixmap:
        """
        Render a specific region using the appropriate Zarr pyramid level.

        This method overrides OptimizedCanvas._render_region to load from Zarr
        instead of the in-memory raw_image array.
        """
        if self.zarr_group is None:
            # Fall back to parent implementation
            return super()._render_region(x1, y1, x2, y2)

        # Select pyramid level based on zoom
        level_idx, level_path, downsample = self._select_pyramid_level(self.zoom_level)

        # Load the Zarr array for this level
        try:
            zarr_array = self.zarr_group[level_path]
        except KeyError:
            print(f"[ZarrCanvas] Warning: Level {level_path} not found, using level 0")
            zarr_array = self.zarr_group[self.zarr_pyramid_paths[0]]
            downsample = 1

        # Calculate coordinates in the downsampled space
        # Region coords (x1, y1, x2, y2) are in full-resolution space
        ds_x1 = x1 // downsample
        ds_y1 = y1 // downsample
        ds_x2 = (x2 + downsample - 1) // downsample  # Ceiling division
        ds_y2 = (y2 + downsample - 1) // downsample

        # Load the region from Zarr
        # Handle both 2D and 3D arrays
        if len(zarr_array.shape) == 3:
            # 3D: (slices, height, width)
            slice_idx = min(self.zarr_slice_idx, zarr_array.shape[0] - 1)
            region = zarr_array[slice_idx, ds_y1:ds_y2, ds_x1:ds_x2]
        else:
            # 2D: (height, width)
            region = zarr_array[ds_y1:ds_y2, ds_x1:ds_x2]

        # Convert to numpy array (in case it's still a zarr array)
        region = np.array(region, dtype=np.float32)

        # If we loaded from a downsampled level, upscale to match expected size
        if downsample > 1:
            # Calculate target size in full resolution
            target_h = y2 - y1
            target_w = x2 - x1

            # Upscale the region
            zoom_factors = (target_h / region.shape[0], target_w / region.shape[1])
            region = scipy_zoom(region, zoom_factors, order=1)  # Bilinear interpolation

            # Ensure exact size (zoom might be slightly off due to rounding)
            if region.shape[0] != target_h or region.shape[1] != target_w:
                region = region[:target_h, :target_w]

        h, w = region.shape

        # Normalize for display (use global min/max if available, otherwise local)
        if hasattr(self, '_zarr_global_min') and hasattr(self, '_zarr_global_max'):
            img_min = self._zarr_global_min
            img_max = self._zarr_global_max
        else:
            img_min, img_max = region.min(), region.max()

        if img_max > img_min:
            region = (region - img_min) / (img_max - img_min)
        region = (region * 255 * self.image_alpha).astype(np.uint8)

        # Create RGB
        img_rgb = np.stack([region, region, region], axis=-1)

        # Overlay mask (red channel) - mask is always full resolution
        if self.mask is not None:
            mask_region = self.mask[y1:y2, x1:x2]
            mask_overlay = (mask_region > 127).astype(np.float32)
            img_rgb[:, :, 0] = np.clip(
                img_rgb[:, :, 0] * (1 - mask_overlay * self.mask_alpha) +
                255 * mask_overlay * self.mask_alpha, 0, 255
            ).astype(np.uint8)

        # Overlay suggestion (green channel) - suggestion is also full resolution
        if self.show_suggestion and self.suggestion is not None:
            sugg_region = self.suggestion[y1:y2, x1:x2]
            sugg_overlay = (sugg_region > 127).astype(np.float32)
            img_rgb[:, :, 1] = np.clip(
                img_rgb[:, :, 1] * (1 - sugg_overlay * self.mask_alpha) +
                255 * sugg_overlay * self.mask_alpha, 0, 255
            ).astype(np.uint8)

        # Highlight hovered component (same as parent)
        if self._hovered_component is not None:
            hover_region = self._hovered_component[y1:y2, x1:x2]
            if hover_region.any():
                # Create checkerboard pattern
                yy, xx = np.ogrid[:h, :w]
                checker = ((yy + y1 + xx + x1) % 3 < 1)
                red_pixels = hover_region & checker
                img_rgb[red_pixels, 0] = np.clip(
                    img_rgb[red_pixels, 0] * (1 - self.mask_alpha) +
                    255 * self.mask_alpha, 0, 255
                ).astype(np.uint8)
                img_rgb[red_pixels, 1] = (img_rgb[red_pixels, 1] * (1 - self.mask_alpha)).astype(np.uint8)
                img_rgb[red_pixels, 2] = (img_rgb[red_pixels, 2] * (1 - self.mask_alpha)).astype(np.uint8)

        # Convert to QPixmap
        img_rgb = np.ascontiguousarray(img_rgb)
        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        # Return tuple (pixmap, downsample_factor) to match OptimizedCanvas signature
        return QPixmap.fromImage(qimg.copy()), 1

    def set_image(self, image: np.ndarray):
        """
        Set a full-resolution image (fallback for non-Zarr mode).
        When using Zarr, this is typically not called.
        """
        super().set_image(image)
        # Clear Zarr mode if setting a regular image
        self.zarr_group = None

    def compute_global_stats(self):
        """
        Compute global min/max from the full Zarr volume for consistent normalization.
        This is optional but recommended for better visualization.
        """
        if self.zarr_group is None or not self.zarr_pyramid_paths:
            return

        # Use the most downsampled level for fast stats computation
        stats_level = self.zarr_pyramid_paths[-1]
        zarr_array = self.zarr_group[stats_level]

        # Sample a few slices to estimate global stats
        if len(zarr_array.shape) == 3:
            num_slices = zarr_array.shape[0]
            sample_indices = np.linspace(0, num_slices - 1, min(10, num_slices), dtype=int)

            global_min = float('inf')
            global_max = float('-inf')

            for idx in sample_indices:
                slice_data = np.array(zarr_array[idx])
                global_min = min(global_min, slice_data.min())
                global_max = max(global_max, slice_data.max())

            self._zarr_global_min = global_min
            self._zarr_global_max = global_max
            print(f"[ZarrCanvas] Global stats: min={global_min:.2f}, max={global_max:.2f}")
        else:
            # 2D array
            data = np.array(zarr_array)
            self._zarr_global_min = data.min()
            self._zarr_global_max = data.max()
