"""
Proofreading viewer management.

Provides local Neuroglancer server management and browser integration
for viewing EM data and segmentations.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .neuroglancer_state import NeuroglancerState


@dataclass
class ViewerConfig:
    """Configuration for the proofreading viewer."""

    host: str = "localhost"
    port: int = 8080
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    neuroglancer_url: str = "https://neuroglancer-demo.appspot.com"
    auto_open_browser: bool = True
    data_directory: Optional[Path] = None


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler with CORS support for Neuroglancer."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.zarr': 'application/octet-stream',
        '.json': 'application/json',
    }

    def __init__(self, *args, cors_origins: List[str] = None, **kwargs):
        self.cors_origins = cors_origins or ["*"]
        super().__init__(*args, **kwargs)

    def end_headers(self):
        """Add CORS headers."""
        origin = self.headers.get('Origin', '*')
        if '*' in self.cors_origins or origin in self.cors_origins:
            self.send_header('Access-Control-Allow-Origin', origin)
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')
        self.send_header('Access-Control-Expose-Headers', 'Content-Range, Content-Length')
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress logging for cleaner output."""
        pass


def create_cors_handler(cors_origins: List[str], directory: Path):
    """Create a CORS-enabled request handler for a directory."""

    class Handler(CORSRequestHandler):
        def __init__(self, *args, **kwargs):
            self.cors_origins = cors_origins
            super().__init__(*args, directory=str(directory), **kwargs)

    return Handler


class DataServer:
    """Simple HTTP server for serving local data to Neuroglancer."""

    def __init__(
        self,
        directory: Path,
        host: str = "localhost",
        port: int = 8080,
        cors_origins: List[str] = None,
    ):
        """Initialize data server.

        Args:
            directory: Root directory to serve
            host: Server host
            port: Server port
            cors_origins: Allowed CORS origins
        """
        self.directory = Path(directory)
        self.host = host
        self.port = port
        self.cors_origins = cors_origins or ["*"]
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def base_url(self) -> str:
        """Get base URL for the server."""
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        """Start the server in a background thread."""
        if self._running:
            return

        handler = create_cors_handler(self.cors_origins, self.directory)
        self._server = HTTPServer((self.host, self.port), handler)
        self._running = True

        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

        # Register cleanup
        atexit.register(self.stop)

    def _serve(self) -> None:
        """Server loop."""
        while self._running:
            self._server.handle_request()

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        self._thread = None

    def get_layer_url(self, relative_path: str, layer_type: str = "zarr") -> str:
        """Get Neuroglancer-compatible layer URL.

        Args:
            relative_path: Path relative to server root
            layer_type: Data format (zarr, precomputed, n5)

        Returns:
            Neuroglancer source URL
        """
        if layer_type == "zarr":
            return f"zarr://{self.base_url}/{relative_path}"
        elif layer_type == "precomputed":
            return f"precomputed://{self.base_url}/{relative_path}"
        elif layer_type == "n5":
            return f"n5://{self.base_url}/{relative_path}"
        else:
            return f"{self.base_url}/{relative_path}"


class ProofreadingViewer:
    """Manages Neuroglancer viewer for proofreading workflows.

    This class handles:
    - Local data server for serving volumes
    - Opening Neuroglancer with pre-configured states
    - Managing multiple data sources
    """

    def __init__(self, config: Optional[ViewerConfig] = None):
        """Initialize viewer.

        Args:
            config: Viewer configuration
        """
        self.config = config or ViewerConfig()
        self._data_server: Optional[DataServer] = None
        self._registered_volumes: Dict[str, str] = {}  # name -> path

    @property
    def is_serving(self) -> bool:
        """Check if data server is running."""
        return self._data_server is not None and self._data_server._running

    @property
    def server_url(self) -> Optional[str]:
        """Get data server URL if running."""
        if self._data_server:
            return self._data_server.base_url
        return None

    def start_server(self, directory: Optional[Path] = None) -> str:
        """Start local data server.

        Args:
            directory: Directory to serve (defaults to config.data_directory)

        Returns:
            Server base URL
        """
        if self._data_server and self._data_server._running:
            return self._data_server.base_url

        serve_dir = directory or self.config.data_directory
        if serve_dir is None:
            raise ValueError("No directory specified for data server")

        serve_dir = Path(serve_dir)
        if not serve_dir.exists():
            raise FileNotFoundError(f"Directory not found: {serve_dir}")

        self._data_server = DataServer(
            directory=serve_dir,
            host=self.config.host,
            port=self.config.port,
            cors_origins=self.config.cors_origins,
        )
        self._data_server.start()

        return self._data_server.base_url

    def stop_server(self) -> None:
        """Stop the data server."""
        if self._data_server:
            self._data_server.stop()
            self._data_server = None

    def register_volume(
        self,
        name: str,
        path: Union[str, Path],
        format: str = "zarr",
    ) -> str:
        """Register a volume for serving.

        Args:
            name: Volume name (used in layer URLs)
            path: Path to volume data
            format: Data format (zarr, precomputed, n5)

        Returns:
            Neuroglancer source URL for the volume
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Volume not found: {path}")

        self._registered_volumes[name] = str(path)

        if self._data_server:
            # Path relative to server root
            try:
                rel_path = path.relative_to(self._data_server.directory)
                return self._data_server.get_layer_url(str(rel_path), format)
            except ValueError:
                # Path not under server root - would need symlink or separate server
                raise ValueError(
                    f"Volume {path} is not under server directory "
                    f"{self._data_server.directory}"
                )

        return f"{format}://{path}"

    def open_state(
        self,
        state: NeuroglancerState,
        open_browser: bool = None,
    ) -> str:
        """Open Neuroglancer with the given state.

        Args:
            state: Neuroglancer state to open
            open_browser: Whether to open browser (defaults to config)

        Returns:
            Full Neuroglancer URL
        """
        url = state.to_url(self.config.neuroglancer_url)

        should_open = open_browser if open_browser is not None else self.config.auto_open_browser
        if should_open:
            webbrowser.open(url)

        return url

    def open_url(self, url: str) -> None:
        """Open a URL in the default browser.

        Args:
            url: URL to open
        """
        webbrowser.open(url)

    def create_url(
        self,
        state: NeuroglancerState,
        base_url: Optional[str] = None,
    ) -> str:
        """Create Neuroglancer URL without opening browser.

        Args:
            state: Neuroglancer state
            base_url: Optional custom base URL

        Returns:
            Full Neuroglancer URL
        """
        base = base_url or self.config.neuroglancer_url
        return state.to_url(base)

    def get_local_source_url(
        self,
        path: Union[str, Path],
        format: str = "zarr",
    ) -> str:
        """Get source URL for local data (requires server running).

        Args:
            path: Path to data
            format: Data format

        Returns:
            Neuroglancer source URL
        """
        if not self._data_server:
            raise RuntimeError("Data server not running. Call start_server() first.")

        path = Path(path)
        try:
            rel_path = path.relative_to(self._data_server.directory)
            return self._data_server.get_layer_url(str(rel_path), format)
        except ValueError:
            raise ValueError(
                f"Path {path} is not under server directory "
                f"{self._data_server.directory}"
            )


def launch_neuroglancer_viewer(
    raw_path: Union[str, Path],
    segmentation_path: Optional[Union[str, Path]] = None,
    position: Optional[tuple] = None,
    resolution: tuple = (4.0, 4.0, 40.0),
    server_port: int = 8080,
) -> str:
    """Convenience function to quickly launch viewer with local data.

    Args:
        raw_path: Path to raw image data (zarr)
        segmentation_path: Optional path to segmentation (zarr)
        position: Optional (x, y, z) center position
        resolution: (x, y, z) resolution in nm
        server_port: Port for local data server

    Returns:
        Neuroglancer URL
    """
    from .neuroglancer_state import NeuroglancerStateBuilder

    raw_path = Path(raw_path)
    data_dir = raw_path.parent

    # Start viewer
    viewer = ProofreadingViewer(ViewerConfig(
        port=server_port,
        data_directory=data_dir,
    ))
    viewer.start_server()

    # Build state
    builder = NeuroglancerStateBuilder()
    builder.with_resolution(resolution)

    # Add raw layer
    raw_url = viewer.get_local_source_url(raw_path, "zarr")
    builder.with_raw_layer(raw_url)

    # Add segmentation if provided
    if segmentation_path:
        seg_path = Path(segmentation_path)
        seg_url = viewer.get_local_source_url(seg_path, "zarr")
        builder.with_segmentation_layer(seg_url)

    # Set position if provided
    if position:
        builder.center_on(*position)

    state = builder.build()
    return viewer.open_state(state)
