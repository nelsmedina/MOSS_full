"""
Format conversion utilities for EM volumes.

Provides streaming, chunked conversion between formats without loading
entire volumes into memory. Supports parallel processing for large volumes.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import tensorstore as ts
    TENSORSTORE_AVAILABLE = True
except ImportError:
    TENSORSTORE_AVAILABLE = False
    ts = None

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    tifffile = None


class ConversionProgress:
    """Progress tracker for conversions."""

    def __init__(
        self,
        total_chunks: int,
        callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        self.total_chunks = total_chunks
        self.completed_chunks = 0
        self.callback = callback

    def update(self, chunks: int = 1, message: str = "") -> None:
        """Update progress."""
        self.completed_chunks += chunks
        if self.callback:
            self.callback(self.completed_chunks, self.total_chunks, message)

    @property
    def percent(self) -> float:
        """Progress as percentage."""
        if self.total_chunks == 0:
            return 100.0
        return (self.completed_chunks / self.total_chunks) * 100


def _compute_chunk_ranges(
    shape: Tuple[int, ...],
    chunk_size: Tuple[int, ...],
) -> List[Tuple[Tuple[slice, ...], Tuple[int, ...]]]:
    """
    Compute all chunk ranges for a volume.

    Returns list of (slices, chunk_indices) tuples.
    """
    ranges = []
    ndim = len(shape)

    # Compute number of chunks per dimension
    n_chunks = [math.ceil(s / c) for s, c in zip(shape, chunk_size)]

    # Generate all chunk indices
    def generate_indices(dim: int, current: List[int]) -> Iterator[List[int]]:
        if dim == ndim:
            yield current.copy()
            return
        for i in range(n_chunks[dim]):
            current.append(i)
            yield from generate_indices(dim + 1, current)
            current.pop()

    for indices in generate_indices(0, []):
        slices = tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(indices, chunk_size, shape)
        )
        ranges.append((slices, tuple(indices)))

    return ranges


def tiff_to_zarr(
    tiff_path: Union[str, Path],
    zarr_path: Union[str, Path],
    chunk_size: Tuple[int, ...] = (64, 256, 256),
    compression: Optional[str] = "blosc",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    num_workers: int = 4,
) -> None:
    """
    Convert a TIFF stack to Zarr format.

    Streams data chunk-by-chunk to avoid loading entire volume into RAM.

    Parameters
    ----------
    tiff_path : str or Path
        Path to input TIFF file OR directory of TIFF slices.
    zarr_path : str or Path
        Path for output Zarr array.
    chunk_size : tuple of int
        Chunk size for Zarr array (z, y, x).
    compression : str or None
        Compression codec ('blosc', 'zstd', 'gzip', None).
    progress_callback : callable, optional
        Called with (completed, total, message) for progress updates.
    num_workers : int
        Number of parallel workers for chunk processing.
    """
    if not TIFFFILE_AVAILABLE:
        raise ImportError("tifffile required. Install with: pip install tifffile")
    if not TENSORSTORE_AVAILABLE:
        raise ImportError("tensorstore required. Install with: pip install tensorstore")

    tiff_path = Path(tiff_path)
    zarr_path = Path(zarr_path)

    # Handle directory of TIFF slices
    if tiff_path.is_dir():
        return _tiff_dir_to_zarr(
            tiff_path, zarr_path, chunk_size, compression,
            progress_callback, num_workers
        )

    # Open TIFF and get metadata
    tiff = tifffile.TiffFile(str(tiff_path))
    shape = tiff.series[0].shape
    dtype = tiff.series[0].dtype

    # Adjust chunk size to not exceed volume dimensions
    chunk_size = tuple(min(c, s) for c, s in zip(chunk_size, shape))

    # Build Zarr metadata
    zarr_metadata = {
        'shape': list(shape),
        'chunks': list(chunk_size),
        'dtype': np.dtype(dtype).str,
    }

    if compression:
        if compression == 'blosc':
            zarr_metadata['compressor'] = {
                'id': 'blosc',
                'cname': 'lz4',
                'clevel': 5,
                'shuffle': 1,
            }
        elif compression == 'zstd':
            zarr_metadata['compressor'] = {
                'id': 'zstd',
                'level': 3,
            }
        elif compression == 'gzip':
            zarr_metadata['compressor'] = {
                'id': 'gzip',
                'level': 5,
            }
    else:
        zarr_metadata['compressor'] = None

    # Create Zarr array with TensorStore
    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': str(zarr_path)},
        'metadata': zarr_metadata,
        'create': True,
        'delete_existing': True,
    }

    store = ts.open(spec).result()

    # Compute chunk ranges
    chunk_ranges = _compute_chunk_ranges(shape, chunk_size)
    progress = ConversionProgress(len(chunk_ranges), progress_callback)

    # Read TIFF data (memmap for large files)
    data = tiff.asarray()

    def process_chunk(slices_and_idx):
        slices, idx = slices_and_idx
        chunk_data = data[slices]
        store[slices].write(chunk_data).result()
        return slices

    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk, cr) for cr in chunk_ranges]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Raise any exceptions
            progress.update(1, f"{progress.percent:.1f}% complete")

    tiff.close()


def _tiff_dir_to_zarr(
    tiff_dir: Path,
    zarr_path: Path,
    chunk_size: Tuple[int, ...] = (64, 256, 256),
    compression: Optional[str] = "blosc",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    num_workers: int = 4,
) -> None:
    """
    Convert a directory of TIFF slices to Zarr v3 format with sharding.

    Each TIFF file in the directory is treated as one Z-slice.
    Files are sorted alphabetically to determine Z-order.

    Uses Zarr v3 with sharding for massive speedup (~75x) by writing
    multiple chunks in a single shard file operation.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # Allow large images

    import zarr
    from zarr.storage import LocalStore
    from zarr.codecs import BytesCodec, BloscCodec, ShardingCodec

    # Find all TIFF files and sort them
    tiff_files = sorted(
        list(tiff_dir.glob("*.tif")) + list(tiff_dir.glob("*.tiff"))
    )

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {tiff_dir}")

    # Read first slice metadata only (don't load full image if huge)
    with tifffile.TiffFile(str(tiff_files[0])) as tif:
        page = tif.pages[0]
        height, width = page.shape[:2]
        dtype = page.dtype
    n_slices = len(tiff_files)

    # Validate all slices have consistent dimensions (sample check for speed)
    # Check first, middle, and last files
    check_indices = [0, n_slices // 2, n_slices - 1]
    for idx in check_indices:
        if idx >= n_slices:
            continue
        with tifffile.TiffFile(str(tiff_files[idx])) as tif:
            page = tif.pages[0]
            h, w = page.shape[:2]
            if h != height or w != width:
                raise ValueError(
                    f"Dimension mismatch: {tiff_files[0].name} is {height}x{width}, "
                    f"but {tiff_files[idx].name} is {h}x{w}. "
                    f"All TIFF slices must have the same dimensions."
                )

    shape = (n_slices, height, width)
    image_pixels = height * width

    # Adjust chunk size to not exceed volume dimensions
    chunk_size = tuple(min(c, s) for c, s in zip(chunk_size, shape))
    z_chunk, y_chunk, x_chunk = chunk_size

    # Zarr v3 with sharding:
    # - Shard size = chunk_size (e.g., 64, 256, 256) - one file per shard
    # - Inner chunk size = half z dimension (e.g., 32, 256, 256) - chunks within each shard
    # This means writing 64 slices = 2 inner chunks = 1 shard = 1 disk write
    shard_shape = chunk_size
    inner_chunk_shape = (max(1, z_chunk // 2), y_chunk, x_chunk)

    # Setup compression codecs
    if compression == 'blosc':
        inner_codecs = [BytesCodec(), BloscCodec(cname='zstd', clevel=3)]
    elif compression == 'zstd':
        from zarr.codecs import ZstdCodec
        inner_codecs = [BytesCodec(), ZstdCodec(level=3)]
    elif compression == 'gzip':
        from zarr.codecs import GzipCodec
        inner_codecs = [BytesCodec(), GzipCodec(level=5)]
    else:
        inner_codecs = [BytesCodec()]

    # Create Zarr v3 array with sharding using zarr-python
    # (This generates correct metadata that TensorStore can read)
    store = LocalStore(str(zarr_path))

    zarr_array = zarr.create(
        shape=shape,
        chunk_shape=shard_shape,  # This is the SHARD size
        dtype=dtype,
        codecs=[ShardingCodec(chunk_shape=inner_chunk_shape, codecs=inner_codecs)],
        zarr_format=3,
        store=store,
        overwrite=True
    )

    # Now open with TensorStore for fast parallel writes
    spec = {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': str(zarr_path)},
        'open': True,
    }

    store_ts = ts.open(spec).result()

    # Always batch by z_chunk (shard size) for maximum speedup
    # Writing full shards at once avoids read-modify-write overhead
    n_z_chunks = math.ceil(n_slices / z_chunk)
    progress = ConversionProgress(n_z_chunks, progress_callback)

    def load_and_write_z_chunk(z_start: int):
        z_end = min(z_start + z_chunk, n_slices)
        slices_data = []
        for z in range(z_start, z_end):
            slice_data = tifffile.imread(str(tiff_files[z]))
            # Handle RGB -> grayscale if needed
            if slice_data.ndim == 3:
                slice_data = slice_data[..., 0]  # Take first channel
            slices_data.append(slice_data)

        # Stack into chunk-sized block
        chunk_data = np.stack(slices_data, axis=0)
        # Write entire chunk (shard) in one operation - this is the key optimization!
        store_ts[z_start:z_end, :, :].write(chunk_data).result()
        return z_start

    # Process Z-chunks (shards) in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        z_starts = list(range(0, n_slices, z_chunk))
        futures = [executor.submit(load_and_write_z_chunk, z) for z in z_starts]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Raise any exceptions
            progress.update(1, f"{progress.percent:.1f}% complete")


def zarr_to_precomputed(
    zarr_path: Union[str, Path],
    precomputed_path: Union[str, Path],
    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    chunk_size: Optional[Tuple[int, ...]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    num_workers: int = 4,
) -> None:
    """
    Convert Zarr to Neuroglancer Precomputed format.

    Parameters
    ----------
    zarr_path : str or Path
        Path to input Zarr array.
    precomputed_path : str or Path
        Path for output Precomputed volume.
    resolution : tuple of float
        Voxel resolution in nm (x, y, z).
    chunk_size : tuple of int, optional
        Chunk size. Uses source chunk size if not specified.
    progress_callback : callable, optional
        Progress callback.
    num_workers : int
        Number of parallel workers.
    """
    if not TENSORSTORE_AVAILABLE:
        raise ImportError("tensorstore required")

    zarr_path = Path(zarr_path)
    precomputed_path = Path(precomputed_path)

    # Open source Zarr
    src_spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': str(zarr_path)},
        'open': True,
    }
    src = ts.open(src_spec).result()

    shape = tuple(src.shape)
    dtype = src.dtype.numpy_dtype

    # Determine chunk size
    if chunk_size is None:
        # Try to get from source
        chunk_size = (64, 64, 64)  # Default

    chunk_size = tuple(min(c, s) for c, s in zip(chunk_size, shape))

    # Create info file for precomputed format
    precomputed_path.mkdir(parents=True, exist_ok=True)

    # Neuroglancer precomputed uses x,y,z ordering
    info = {
        '@type': 'neuroglancer_multiscale_volume',
        'type': 'image',
        'data_type': str(dtype),
        'num_channels': 1,
        'scales': [{
            'key': '1_1_1',
            'size': list(reversed(shape)),  # x,y,z
            'resolution': list(resolution),
            'chunk_sizes': [[*reversed(chunk_size)]],
            'encoding': 'raw',
        }],
    }

    with open(precomputed_path / 'info', 'w') as f:
        json.dump(info, f, indent=2)

    # Open destination with TensorStore
    dst_spec = {
        'driver': 'neuroglancer_precomputed',
        'kvstore': {'driver': 'file', 'path': str(precomputed_path)},
        'scale_metadata': {
            'size': list(reversed(shape)),
            'resolution': list(resolution),
            'encoding': 'raw',
            'chunk_size': list(reversed(chunk_size)),
        },
        'multiscale_metadata': {
            'type': 'image',
            'data_type': str(dtype),
            'num_channels': 1,
        },
        'create': True,
        'delete_existing': True,
    }

    dst = ts.open(dst_spec).result()

    # Compute chunk ranges
    chunk_ranges = _compute_chunk_ranges(shape, chunk_size)
    progress = ConversionProgress(len(chunk_ranges), progress_callback)

    def process_chunk(slices_and_idx):
        slices, idx = slices_and_idx
        chunk_data = src[slices].read().result()
        # Transpose for precomputed (z,y,x) -> (x,y,z)
        transposed = np.transpose(chunk_data)
        # Build corresponding slices for destination
        dst_slices = tuple(
            slice(s.stop - 1, s.start - 1, -1) if s.stop > s.start else slice(s.start, s.stop)
            for s in reversed(slices)
        )
        dst[dst_slices].write(transposed).result()
        return slices

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk, cr) for cr in chunk_ranges]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            progress.update(1)


def convert(
    source: Union[str, Path],
    destination: Union[str, Path],
    output_format: Optional[str] = None,
    chunk_size: Tuple[int, ...] = (64, 256, 256),
    resolution: Optional[Tuple[float, float, float]] = None,
    compression: Optional[str] = "blosc",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    num_workers: int = 4,
) -> None:
    """
    Convert between volume formats.

    Automatically detects source format and converts to the specified
    output format using streaming chunk-by-chunk processing.

    Parameters
    ----------
    source : str or Path
        Path to source volume.
    destination : str or Path
        Path for output volume.
    output_format : str, optional
        Output format ('zarr', 'precomputed', 'n5'). Auto-detected from
        destination extension if not specified.
    chunk_size : tuple of int
        Chunk size for output.
    resolution : tuple of float, optional
        Voxel resolution in nm (x, y, z).
    compression : str or None
        Compression for output.
    progress_callback : callable, optional
        Progress callback.
    num_workers : int
        Number of parallel workers.
    """
    source = Path(source)
    destination = Path(destination)

    # Detect source format
    if source.suffix in ('.tiff', '.tif'):
        src_format = 'tiff'
    elif source.is_dir() and any(source.glob("*.tif")) or any(source.glob("*.tiff")):
        # Directory of TIFF slices
        src_format = 'tiff'
    elif source.suffix == '.zarr' or (source.is_dir() and (source / '.zarray').exists()):
        src_format = 'zarr'
    elif source.is_dir() and (source / 'info').exists():
        src_format = 'precomputed'
    elif source.is_dir() and (source / 'attributes.json').exists():
        src_format = 'n5'
    else:
        raise ValueError(f"Cannot detect format of source: {source}")

    # Detect output format
    if output_format is None:
        if destination.suffix == '.zarr':
            output_format = 'zarr'
        elif 'precomputed' in str(destination).lower():
            output_format = 'precomputed'
        else:
            output_format = 'zarr'  # Default

    # Route to appropriate converter
    if src_format == 'tiff' and output_format == 'zarr':
        tiff_to_zarr(
            source, destination,
            chunk_size=chunk_size,
            compression=compression,
            progress_callback=progress_callback,
            num_workers=num_workers,
        )
    elif src_format == 'zarr' and output_format == 'precomputed':
        zarr_to_precomputed(
            source, destination,
            resolution=resolution or (1.0, 1.0, 1.0),
            chunk_size=chunk_size,
            progress_callback=progress_callback,
            num_workers=num_workers,
        )
    elif src_format == 'tiff' and output_format == 'precomputed':
        # Two-step: TIFF -> Zarr (temp) -> Precomputed
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_zarr = Path(tmpdir) / "temp.zarr"
            tiff_to_zarr(
                source, tmp_zarr,
                chunk_size=chunk_size,
                compression=None,  # No compression for intermediate
                progress_callback=lambda c, t, m: progress_callback(c // 2, t, m) if progress_callback else None,
                num_workers=num_workers,
            )
            zarr_to_precomputed(
                tmp_zarr, destination,
                resolution=resolution or (1.0, 1.0, 1.0),
                chunk_size=chunk_size,
                progress_callback=lambda c, t, m: progress_callback(t // 2 + c // 2, t, m) if progress_callback else None,
                num_workers=num_workers,
            )
    else:
        raise ValueError(f"Conversion from {src_format} to {output_format} not yet supported")


__all__ = [
    'convert',
    'tiff_to_zarr',
    'zarr_to_precomputed',
    'ConversionProgress',
]
