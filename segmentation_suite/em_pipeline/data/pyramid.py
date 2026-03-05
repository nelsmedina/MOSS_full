"""
Multi-resolution pyramid generation for EM volumes.

Generates image pyramids for efficient visualization at multiple scales.
Supports both in-place (add to existing volume) and output-to-new-location modes.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import tensorstore as ts
    TENSORSTORE_AVAILABLE = True
except ImportError:
    TENSORSTORE_AVAILABLE = False
    ts = None

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ndimage = None


def _downsample_chunk(
    data: np.ndarray,
    factors: Tuple[int, ...],
    method: str = 'mean',
) -> np.ndarray:
    """
    Downsample a chunk by given factors.

    Parameters
    ----------
    data : ndarray
        Input data chunk.
    factors : tuple of int
        Downsampling factors per dimension.
    method : str
        Downsampling method: 'mean', 'max', 'mode', 'nearest'.

    Returns
    -------
    ndarray
        Downsampled data.
    """
    # Ensure factors don't exceed dimensions
    factors = tuple(min(f, s) for f, s in zip(factors, data.shape))

    if method == 'nearest':
        slices = tuple(slice(None, None, f) for f in factors)
        return data[slices].copy()

    # For other methods, use block reduction
    # Pad to make divisible
    pad_width = []
    for s, f in zip(data.shape, factors):
        remainder = s % f
        if remainder:
            pad_width.append((0, f - remainder))
        else:
            pad_width.append((0, 0))

    padded = np.pad(data, pad_width, mode='edge')

    # Reshape for block operations
    new_shape = []
    for s, f in zip(padded.shape, factors):
        new_shape.extend([s // f, f])
    reshaped = padded.reshape(new_shape)

    # Determine axes to reduce
    reduce_axes = tuple(range(1, len(new_shape), 2))

    if method == 'mean':
        result = reshaped.mean(axis=reduce_axes)
    elif method == 'max':
        result = reshaped.max(axis=reduce_axes)
    elif method == 'mode':
        # Mode is expensive - use for segmentation labels
        from scipy import stats
        # Flatten blocks and compute mode
        flat_shape = list(result.shape for result in [reshaped.shape[::2]])
        result = stats.mode(reshaped.reshape(*flat_shape[0], -1), axis=-1, keepdims=False).mode
    else:
        raise ValueError(f"Unknown downsampling method: {method}")

    return result.astype(data.dtype)


def generate_pyramid(
    source_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    num_levels: int = 4,
    factors: Tuple[int, ...] = (2, 2, 2),
    method: str = 'mean',
    chunk_size: Optional[Tuple[int, ...]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    num_workers: int = 4,
) -> List[str]:
    """
    Generate a multi-resolution pyramid from a volume.

    Parameters
    ----------
    source_path : str or Path
        Path to source Zarr volume.
    output_path : str or Path, optional
        Output path. If None, adds pyramid levels to source.
    num_levels : int
        Number of pyramid levels to generate (including base level).
    factors : tuple of int
        Downsampling factors per level (z, y, x).
    method : str
        Downsampling method: 'mean' for images, 'mode' for labels.
    chunk_size : tuple of int, optional
        Chunk size for output levels.
    progress_callback : callable, optional
        Progress callback.
    num_workers : int
        Number of parallel workers.

    Returns
    -------
    list of str
        Paths to all pyramid levels.
    """
    if not TENSORSTORE_AVAILABLE:
        raise ImportError("tensorstore required")

    source_path = Path(source_path)
    if output_path is None:
        output_path = source_path.parent
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect Zarr version and open source (level 0)
    # Check for zarr v3 (has zarr.json) vs v2 (has .zarray)
    zarr_v3 = (source_path / 'zarr.json').exists()
    driver = 'zarr3' if zarr_v3 else 'zarr'

    src_spec = {
        'driver': driver,
        'kvstore': {'driver': 'file', 'path': str(source_path)},
        'open': True,
    }
    src = ts.open(src_spec).result()

    base_shape = tuple(src.shape)
    dtype = src.dtype.numpy_dtype

    if chunk_size is None:
        chunk_size = (64, 256, 256)

    # Get source metadata for v3 sharding config
    source_metadata = None
    if zarr_v3:
        with open(source_path / 'zarr.json', 'r') as f:
            source_metadata = json.load(f)

    level_paths = [str(source_path)]
    current_src = src
    current_shape = base_shape

    # Total chunks across all levels for progress
    total_chunks = 0
    for level in range(1, num_levels):
        level_shape = tuple(max(1, s // (factors[i] ** level)) for i, s in enumerate(base_shape))
        n_chunks = math.prod(math.ceil(s / c) for s, c in zip(level_shape, chunk_size))
        total_chunks += n_chunks

    completed = 0

    for level in range(1, num_levels):
        # Calculate shape for this level
        prev_shape = current_shape
        current_shape = tuple(max(1, s // f) for s, f in zip(prev_shape, factors))

        if all(s <= 1 for s in current_shape):
            break  # Too small to continue

        # Create output for this level (matching source zarr version)
        level_path = output_path / f"s{level}"
        level_paths.append(str(level_path))

        adj_chunk_size = tuple(min(c, s) for c, s in zip(chunk_size, current_shape))

        # Build metadata matching source format
        if zarr_v3 and source_metadata:
            # Copy codec configuration from source
            codecs = source_metadata.get('codecs', [])

            # Convert numpy dtype to Zarr v3 data type name
            dtype_np = np.dtype(dtype)
            dtype_map = {
                np.dtype('uint8'): 'uint8',
                np.dtype('uint16'): 'uint16',
                np.dtype('uint32'): 'uint32',
                np.dtype('uint64'): 'uint64',
                np.dtype('int8'): 'int8',
                np.dtype('int16'): 'int16',
                np.dtype('int32'): 'int32',
                np.dtype('int64'): 'int64',
                np.dtype('float32'): 'float32',
                np.dtype('float64'): 'float64',
            }
            zarr_dtype = dtype_map.get(dtype_np, dtype_np.name)

            metadata = {
                'shape': list(current_shape),
                'chunk_grid': {
                    'name': 'regular',
                    'configuration': {
                        'chunk_shape': list(adj_chunk_size)
                    }
                },
                'chunk_key_encoding': {
                    'name': 'default',
                    'configuration': {'separator': '/'}
                },
                'codecs': codecs,
                'data_type': zarr_dtype,
                'zarr_format': 3,
                'node_type': 'array',
            }
        else:
            # Zarr v2 format
            metadata = {
                'shape': list(current_shape),
                'chunks': list(adj_chunk_size),
                'dtype': np.dtype(dtype).str,
            }

        dst_spec = {
            'driver': driver,
            'kvstore': {'driver': 'file', 'path': str(level_path)},
            'metadata': metadata,
            'create': True,
            'delete_existing': True,
        }
        dst = ts.open(dst_spec).result()

        # Compute chunk ranges for destination
        def compute_ranges():
            n_chunks_per_dim = [math.ceil(s / c) for s, c in zip(current_shape, adj_chunk_size)]
            for z in range(n_chunks_per_dim[0]):
                for y in range(n_chunks_per_dim[1]):
                    for x in range(n_chunks_per_dim[2]):
                        dst_slice = (
                            slice(z * adj_chunk_size[0], min((z + 1) * adj_chunk_size[0], current_shape[0])),
                            slice(y * adj_chunk_size[1], min((y + 1) * adj_chunk_size[1], current_shape[1])),
                            slice(x * adj_chunk_size[2], min((x + 1) * adj_chunk_size[2], current_shape[2])),
                        )
                        # Source slice is factors larger
                        src_slice = tuple(
                            slice(s.start * f, min(s.stop * f, ps))
                            for s, f, ps in zip(dst_slice, factors, prev_shape)
                        )
                        yield src_slice, dst_slice

        def process_chunk(src_slice_dst_slice):
            src_slice, dst_slice = src_slice_dst_slice
            # Read from previous level
            chunk_data = current_src[src_slice].read().result()
            # Downsample
            downsampled = _downsample_chunk(chunk_data, factors, method)
            # Write to current level
            # Ensure shapes match
            expected_shape = tuple(s.stop - s.start for s in dst_slice)
            if downsampled.shape != expected_shape:
                # Trim if needed
                trim_slices = tuple(slice(0, e) for e in expected_shape)
                downsampled = downsampled[trim_slices]
            dst[dst_slice].write(downsampled).result()

        chunk_list = list(compute_ranges())
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_chunk, cr) for cr in chunk_list]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_chunks, f"Level {level}")

        # Use this level as source for next
        current_src = dst

    # Create root-level group metadata for Zarr v3
    if zarr_v3:
        # Create Zarr v3 group metadata
        group_metadata = {
            'zarr_format': 3,
            'node_type': 'group',
            'attributes': {}
        }
        with open(output_path / 'zarr.json', 'w') as f:
            json.dump(group_metadata, f, indent=2)

    # Create OME-Zarr multiscales metadata for both v2 and v3
    # This allows viewers like napari, neuroglancer to discover pyramid levels
    datasets = []
    for i in range(len(level_paths)):
        level_name = '0' if i == 0 else f's{i}'
        datasets.append({
            'path': level_name,
            'coordinateTransformations': [{
                'type': 'scale',
                'scale': [float(factors[j] ** i) for j in range(3)]
            }]
        })

    multiscales = [{
        'version': '0.4',
        'name': output_path.name,
        'axes': [
            {'name': 'z', 'type': 'space', 'unit': 'pixel'},
            {'name': 'y', 'type': 'space', 'unit': 'pixel'},
            {'name': 'x', 'type': 'space', 'unit': 'pixel'},
        ],
        'datasets': datasets,
        'type': method,
    }]

    zattrs = {'multiscales': multiscales}
    with open(output_path / '.zattrs', 'w') as f:
        json.dump(zattrs, f, indent=2)

    # Create .zgroup for v2 compatibility
    if not zarr_v3:
        with open(output_path / '.zgroup', 'w') as f:
            json.dump({'zarr_format': 2}, f)

    return level_paths


def generate_ome_zarr_pyramid(
    source_path: Union[str, Path],
    output_path: Union[str, Path],
    num_levels: int = 4,
    factors: Tuple[int, ...] = (2, 2, 2),
    method: str = 'mean',
    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    unit: str = 'nanometer',
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    num_workers: int = 4,
) -> str:
    """
    Generate an OME-Zarr pyramid with proper metadata.

    Creates an OME-Zarr v0.4 compliant multiscale pyramid that can be
    opened directly in Neuroglancer, napari, and other viewers.

    Parameters
    ----------
    source_path : str or Path
        Path to source volume (Zarr or TIFF).
    output_path : str or Path
        Output path for OME-Zarr.
    num_levels : int
        Number of pyramid levels.
    factors : tuple of int
        Downsampling factors (z, y, x).
    method : str
        Downsampling method.
    resolution : tuple of float
        Base resolution (z, y, x) in the given unit.
    unit : str
        Resolution unit ('nanometer', 'micrometer', etc.).
    progress_callback : callable, optional
        Progress callback.
    num_workers : int
        Parallel workers.

    Returns
    -------
    str
        Path to created OME-Zarr.
    """
    if not TENSORSTORE_AVAILABLE:
        raise ImportError("tensorstore required")

    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine source type and get shape/dtype
    if source_path.suffix in ('.tiff', '.tif'):
        import tifffile
        tiff = tifffile.TiffFile(str(source_path))
        shape = tiff.series[0].shape
        dtype = tiff.series[0].dtype
        tiff.close()

        # Convert to Zarr first
        from .convert import tiff_to_zarr
        level0_path = output_path / "0"
        tiff_to_zarr(
            source_path, level0_path,
            chunk_size=(64, 256, 256),
            progress_callback=lambda c, t, m: progress_callback(c // (num_levels + 1), t * num_levels, m) if progress_callback else None,
            num_workers=num_workers,
        )
    else:
        # Copy or symlink level 0
        import shutil
        level0_path = output_path / "0"
        if level0_path.exists():
            shutil.rmtree(level0_path)
        shutil.copytree(source_path, level0_path)

        # Auto-detect Zarr version
        zarr_v3 = (source_path / 'zarr.json').exists()
        driver = 'zarr3' if zarr_v3 else 'zarr'

        src_spec = {
            'driver': driver,
            'kvstore': {'driver': 'file', 'path': str(source_path)},
            'open': True,
        }
        src = ts.open(src_spec).result()
        shape = tuple(src.shape)
        dtype = src.dtype.numpy_dtype

    # Generate additional pyramid levels
    level_paths = generate_pyramid(
        level0_path,
        output_path,
        num_levels=num_levels,
        factors=factors,
        method=method,
        progress_callback=lambda c, t, m: progress_callback(c + (t if source_path.suffix in ('.tiff', '.tif') else 0), t * num_levels, m) if progress_callback else None,
        num_workers=num_workers,
    )

    # Create OME-Zarr metadata
    datasets = []
    for level in range(num_levels):
        level_resolution = [r * (f ** level) for r, f in zip(resolution, factors)]
        datasets.append({
            'path': str(level),
            'coordinateTransformations': [{
                'type': 'scale',
                'scale': [1.0] + level_resolution,  # Add time/channel dimension
            }]
        })

    multiscales = [{
        'version': '0.4',
        'name': output_path.stem,
        'axes': [
            {'name': 'z', 'type': 'space', 'unit': unit},
            {'name': 'y', 'type': 'space', 'unit': unit},
            {'name': 'x', 'type': 'space', 'unit': unit},
        ],
        'datasets': datasets,
        'coordinateTransformations': [{
            'type': 'scale',
            'scale': [1.0] + list(resolution),
        }],
        'type': 'mean' if method == 'mean' else method,
    }]

    zattrs = {'multiscales': multiscales}

    with open(output_path / '.zattrs', 'w') as f:
        json.dump(zattrs, f, indent=2)

    # Create .zgroup
    with open(output_path / '.zgroup', 'w') as f:
        json.dump({'zarr_format': 2}, f)

    return str(output_path)


__all__ = [
    'generate_pyramid',
    'generate_ome_zarr_pyramid',
]
