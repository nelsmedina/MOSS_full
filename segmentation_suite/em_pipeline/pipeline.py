"""
End-to-end segmentation pipeline.

Orchestrates the full segmentation workflow:
1. Load input volume
2. Run inference (affinities/LSDs)
3. Post-processing (watershed, agglomeration)
4. Save output

Supports:
- Quality presets (fast, balanced, accurate)
- Resume from interruption
- Progress tracking
- Multiple output formats
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for the segmentation pipeline.

    Parameters
    ----------
    strategy : str
        Segmentation strategy ('lsd', 'ensemble', 'joint').
    quality : str
        Quality preset ('fast', 'balanced', 'accurate').
    device : str
        Device for inference.
    chunk_size : tuple
        Processing chunk size (D, H, W).
    context : tuple
        Context/overlap for chunks (D, H, W).
    batch_size : int
        Batch size for inference.
    num_workers : int
        Number of parallel workers.
    use_amp : bool
        Use mixed precision inference.
    resume : bool
        Resume from previous run if possible.
    output_format : str
        Output format ('zarr', 'precomputed').
    """

    strategy: str = 'lsd'
    quality: str = 'balanced'
    device: str = 'cuda'
    chunk_size: Tuple[int, int, int] = (64, 256, 256)
    context: Tuple[int, int, int] = (8, 32, 32)
    batch_size: int = 1
    num_workers: int = 4
    use_amp: bool = True
    resume: bool = True
    output_format: str = 'zarr'

    # Quality preset overrides
    _quality_presets: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'fast': {
            'chunk_size': (16, 128, 128),  # ~260K voxels - very memory efficient
            'context': (2, 8, 8),
            'batch_size': 1,
        },
        'balanced': {
            'chunk_size': (32, 192, 192),  # ~1.2M voxels - moderate memory
            'context': (4, 16, 16),
            'batch_size': 1,
        },
        'accurate': {
            'chunk_size': (64, 256, 256),  # ~4M voxels - higher quality
            'context': (8, 32, 32),
            'batch_size': 1,
        },
    }, repr=False)

    def apply_quality_preset(self):
        """Apply quality preset settings."""
        if self.quality in self._quality_presets:
            preset = self._quality_presets[self.quality]
            for key, value in preset.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy': self.strategy,
            'quality': self.quality,
            'device': self.device,
            'chunk_size': self.chunk_size,
            'context': self.context,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'use_amp': self.use_amp,
            'resume': self.resume,
            'output_format': self.output_format,
        }


@dataclass
class PipelineProgress:
    """Progress tracking for pipeline execution."""

    total_chunks: int = 0
    completed_chunks: int = 0
    current_stage: str = 'initializing'
    start_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time."""
        return time.time() - self.start_time if self.start_time > 0 else 0

    @property
    def progress_fraction(self) -> float:
        """Progress as fraction [0, 1]."""
        if self.total_chunks == 0:
            return 0.0
        return self.completed_chunks / self.total_chunks

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining."""
        if self.completed_chunks == 0 or self.elapsed_time == 0:
            return None
        rate = self.completed_chunks / self.elapsed_time
        remaining = self.total_chunks - self.completed_chunks
        return remaining / rate if rate > 0 else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_chunks': self.total_chunks,
            'completed_chunks': self.completed_chunks,
            'current_stage': self.current_stage,
            'elapsed_time': self.elapsed_time,
            'progress_fraction': self.progress_fraction,
            'eta_seconds': self.eta_seconds,
        }


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    success: bool
    output_path: str
    num_segments: int = 0
    total_time: float = 0.0
    config: Optional[Dict[str, Any]] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'output_path': self.output_path,
            'num_segments': self.num_segments,
            'total_time': self.total_time,
            'config': self.config,
            'statistics': self.statistics,
            'error': self.error,
        }


class SegmentationPipeline:
    """
    End-to-end segmentation pipeline.

    Handles the full workflow from raw EM data to instance segmentation.

    Example
    -------
    >>> pipeline = SegmentationPipeline(config)
    >>> result = pipeline.run("input.zarr", "output.zarr")
    >>> print(f"Segmented {result.num_segments} objects in {result.total_time:.1f}s")
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        config : PipelineConfig, optional
            Pipeline configuration.
        progress_callback : callable, optional
            Callback for progress updates.
        """
        self.config = config or PipelineConfig()
        self.config.apply_quality_preset()
        self.progress_callback = progress_callback
        self.progress = PipelineProgress()

        # Automatically adjust chunk size for available GPU memory
        if 'cuda' in self.config.device:
            self._adjust_chunk_size_for_gpu()

        # Components (lazy loaded)
        self._strategy = None
        self._model = None

    def _adjust_chunk_size_for_gpu(self):
        """Adjust chunk size based on available GPU memory."""
        try:
            import torch
            if not torch.cuda.is_available():
                return

            # Get device index
            if ':' in self.config.device:
                device_idx = int(self.config.device.split(':')[1])
            else:
                device_idx = 0

            # Get available memory (leave 20% headroom for safety)
            torch.cuda.set_device(device_idx)
            free_memory = torch.cuda.mem_get_info(device_idx)[0]
            usable_memory = free_memory * 0.8

            # Estimate memory needed for current chunk size
            # Model input + output + intermediate activations
            # Rough estimate: ~150 bytes per voxel for full LSD model with AMP
            cz, cy, cx = self.config.chunk_size
            oz, oy, ox = self.config.context
            total_z, total_y, total_x = cz + 2*oz, cy + 2*oy, cx + 2*ox
            voxels = total_z * total_y * total_x
            bytes_per_voxel = 150 if self.config.use_amp else 300
            estimated_memory = voxels * bytes_per_voxel

            if estimated_memory > usable_memory:
                # Need to reduce chunk size
                logger.warning(
                    f"Chunk size {self.config.chunk_size} requires ~{estimated_memory/1e9:.1f} GB, "
                    f"but only {usable_memory/1e9:.1f} GB available. Reducing chunk size."
                )

                # Calculate maximum safe voxels
                max_voxels = usable_memory / bytes_per_voxel

                # Scale down proportionally, keeping aspect ratio
                scale = (max_voxels / voxels) ** (1/3)
                new_cz = max(16, int(cz * scale) // 16 * 16)  # Round to multiple of 16
                new_cy = max(64, int(cy * scale) // 32 * 32)
                new_cx = max(64, int(cx * scale) // 32 * 32)

                # Also scale context proportionally
                context_scale = min(1.0, scale)
                new_oz = max(2, int(oz * context_scale))
                new_oy = max(8, int(oy * context_scale))
                new_ox = max(8, int(ox * context_scale))

                self.config.chunk_size = (new_cz, new_cy, new_cx)
                self.config.context = (new_oz, new_oy, new_ox)

                logger.info(
                    f"Adjusted chunk size to {self.config.chunk_size} with context {self.config.context}"
                )
            else:
                logger.info(
                    f"Chunk size {self.config.chunk_size} fits in available GPU memory "
                    f"({usable_memory/1e9:.1f} GB available)"
                )

        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}. Using default chunk size.")

    def _update_progress(self, stage: str = None, chunks_done: int = None):
        """Update and report progress."""
        if stage:
            self.progress.current_stage = stage
        if chunks_done is not None:
            self.progress.completed_chunks = chunks_done
        if self.progress_callback:
            self.progress_callback(self.progress)

    def _load_strategy(self):
        """Load segmentation strategy."""
        from segmentation_suite.em_pipeline.strategies import get_strategy

        self._strategy = get_strategy(self.config.strategy)
        self._strategy.config.device = self.config.device
        self._strategy.config.use_amp = self.config.use_amp
        self._strategy.config.batch_size = self.config.batch_size

    def _load_volume(self, input_path: str) -> Tuple[Any, Dict[str, Any]]:
        """Load input volume."""
        from segmentation_suite.em_pipeline.data.volume import open_volume

        volume = open_volume(input_path)
        metadata = {
            'shape': volume.shape,
            'dtype': str(volume.dtype),
            'resolution': volume.resolution,
        }
        return volume, metadata

    def _create_output(
        self,
        output_path: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        resolution: Tuple[float, ...],
    ):
        """Create output volume."""
        import zarr

        output_path = Path(output_path)

        if self.config.output_format == 'zarr':
            # Create Zarr store
            store = zarr.DirectoryStore(str(output_path))
            root = zarr.group(store, overwrite=not self.config.resume)

            # Create segmentation array
            if 'segmentation' not in root or not self.config.resume:
                root.create_dataset(
                    'segmentation',
                    shape=shape,
                    dtype=dtype,
                    chunks=self.config.chunk_size,
                    compressor=zarr.Blosc(cname='zstd', clevel=3),
                    overwrite=True,
                )

            # Add metadata
            root.attrs['resolution'] = resolution
            root.attrs['pipeline_config'] = self.config.to_dict()

            return root['segmentation']

        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")

    def _compute_chunks(
        self,
        shape: Tuple[int, ...],
    ) -> List[Tuple[slice, slice, slice]]:
        """Compute processing chunks."""
        chunks = []
        cz, cy, cx = self.config.chunk_size
        oz, oy, ox = self.config.context

        for z in range(0, shape[0], cz):
            for y in range(0, shape[1], cy):
                for x in range(0, shape[2], cx):
                    # Chunk with context
                    z_start = max(0, z - oz)
                    y_start = max(0, y - oy)
                    x_start = max(0, x - ox)

                    z_end = min(shape[0], z + cz + oz)
                    y_end = min(shape[1], y + cy + oy)
                    x_end = min(shape[2], x + cx + ox)

                    chunks.append((
                        slice(z_start, z_end),
                        slice(y_start, y_end),
                        slice(x_start, x_end),
                    ))

        return chunks

    def _load_checkpoint(self, output_path: str) -> set:
        """Load checkpoint of completed chunks."""
        checkpoint_path = Path(output_path) / '.pipeline_checkpoint.json'
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                data = json.load(f)
                return set(tuple(c) for c in data.get('completed_chunks', []))
        return set()

    def _save_checkpoint(self, output_path: str, completed: set):
        """Save checkpoint of completed chunks."""
        checkpoint_path = Path(output_path) / '.pipeline_checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'completed_chunks': [list(c) for c in completed],
                'config': self.config.to_dict(),
            }, f)

    def run(
        self,
        input_path: str,
        output_path: str,
        model_path: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run the full segmentation pipeline.

        Parameters
        ----------
        input_path : str
            Path to input volume.
        output_path : str
            Path for output segmentation.
        model_path : str, optional
            Path to trained model.

        Returns
        -------
        PipelineResult
            Execution result.
        """
        self.progress = PipelineProgress()
        self.progress.start_time = time.time()
        start_time = time.time()

        try:
            # Stage 1: Load input
            self._update_progress('loading')
            logger.info(f"Loading input: {input_path}")
            volume, metadata = self._load_volume(input_path)

            # Stage 2: Initialize strategy
            self._update_progress('initializing')
            logger.info(f"Initializing strategy: {self.config.strategy}")
            self._load_strategy()

            if model_path:
                self._strategy.load_model(model_path)

            # Stage 3: Create output
            self._update_progress('preparing')
            logger.info(f"Creating output: {output_path}")
            output = self._create_output(
                output_path,
                shape=metadata['shape'],
                dtype=np.uint64,
                resolution=metadata['resolution'],
            )

            # Stage 4: Compute chunks
            chunks = self._compute_chunks(metadata['shape'])
            self.progress.total_chunks = len(chunks)

            # Load checkpoint if resuming
            completed_chunks = set()
            if self.config.resume:
                completed_chunks = self._load_checkpoint(output_path)
                self.progress.completed_chunks = len(completed_chunks)
                logger.info(f"Resuming: {len(completed_chunks)}/{len(chunks)} chunks done")

            # Stage 5: Process chunks
            self._update_progress('segmenting')
            logger.info(f"Processing {len(chunks)} chunks")

            with self._strategy:
                for i, chunk_slices in enumerate(chunks):
                    # Skip if already done
                    chunk_key = tuple(
                        (s.start, s.stop) for s in chunk_slices
                    )
                    if chunk_key in completed_chunks:
                        continue

                    # Extract chunk
                    raw_chunk = np.asarray(volume[chunk_slices])

                    # Normalize if needed
                    if raw_chunk.max() > 1:
                        raw_chunk = raw_chunk.astype(np.float32) / 255.0

                    # Segment
                    result = self._strategy.segment(raw_chunk)

                    # Compute output region (without context)
                    oz, oy, ox = self.config.context
                    cz, cy, cx = self.config.chunk_size

                    # Determine valid region
                    z_start = oz if chunk_slices[0].start > 0 else 0
                    y_start = oy if chunk_slices[1].start > 0 else 0
                    x_start = ox if chunk_slices[2].start > 0 else 0

                    z_end = result.segmentation.shape[0] - oz if chunk_slices[0].stop < metadata['shape'][0] else result.segmentation.shape[0]
                    y_end = result.segmentation.shape[1] - oy if chunk_slices[1].stop < metadata['shape'][1] else result.segmentation.shape[1]
                    x_end = result.segmentation.shape[2] - ox if chunk_slices[2].stop < metadata['shape'][2] else result.segmentation.shape[2]

                    # Write to output
                    out_z = slice(chunk_slices[0].start + z_start, chunk_slices[0].start + z_end)
                    out_y = slice(chunk_slices[1].start + y_start, chunk_slices[1].start + y_end)
                    out_x = slice(chunk_slices[2].start + x_start, chunk_slices[2].start + x_end)

                    seg_crop = result.segmentation[z_start:z_end, y_start:y_end, x_start:x_end]

                    # Relabel to avoid conflicts
                    if seg_crop.max() > 0:
                        # Add offset for unique labels
                        max_label = int(output[out_z, out_y, out_x].max()) if self.progress.completed_chunks > 0 else 0
                        seg_crop = np.where(seg_crop > 0, seg_crop + max_label, 0)

                    output[out_z, out_y, out_x] = seg_crop.astype(np.uint64)

                    # Update progress
                    completed_chunks.add(chunk_key)
                    self._save_checkpoint(output_path, completed_chunks)
                    self._update_progress(chunks_done=len(completed_chunks))

            # Stage 6: Finalize
            self._update_progress('finalizing')
            total_time = time.time() - start_time

            # Count unique segments
            num_segments = len(np.unique(output[:])) - 1  # Exclude background

            # Statistics
            statistics = {
                'input_shape': metadata['shape'],
                'num_chunks': len(chunks),
                'chunks_per_second': len(chunks) / total_time if total_time > 0 else 0,
            }

            return PipelineResult(
                success=True,
                output_path=output_path,
                num_segments=num_segments,
                total_time=total_time,
                config=self.config.to_dict(),
                statistics=statistics,
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return PipelineResult(
                success=False,
                output_path=output_path,
                error=str(e),
                total_time=time.time() - start_time,
                config=self.config.to_dict(),
            )


def run_segmentation(
    input_path: str,
    output_path: str,
    strategy: str = 'lsd',
    quality: str = 'balanced',
    device: str = 'cuda',
    model_path: Optional[str] = None,
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
) -> PipelineResult:
    """
    Convenience function to run segmentation.

    Parameters
    ----------
    input_path : str
        Path to input volume.
    output_path : str
        Path for output segmentation.
    strategy : str
        Segmentation strategy.
    quality : str
        Quality preset.
    device : str
        Device for inference.
    model_path : str, optional
        Path to trained model.
    progress_callback : callable, optional
        Callback for progress updates.

    Returns
    -------
    PipelineResult
        Execution result.
    """
    config = PipelineConfig(
        strategy=strategy,
        quality=quality,
        device=device,
    )

    pipeline = SegmentationPipeline(config, progress_callback)
    return pipeline.run(input_path, output_path, model_path)


__all__ = [
    'PipelineConfig',
    'PipelineProgress',
    'PipelineResult',
    'SegmentationPipeline',
    'run_segmentation',
]
