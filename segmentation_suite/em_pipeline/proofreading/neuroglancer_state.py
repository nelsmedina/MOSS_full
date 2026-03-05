"""
Neuroglancer state builder for proofreading workflows.

Provides a fluent API for building Neuroglancer JSON state configurations
for different proofreading tasks (segment review, merge/split error detection).
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class LayerType(Enum):
    """Neuroglancer layer types."""
    IMAGE = "image"
    SEGMENTATION = "segmentation"
    ANNOTATION = "annotation"


@dataclass
class LayerConfig:
    """Configuration for a Neuroglancer layer."""

    name: str
    source: str
    layer_type: LayerType
    visible: bool = True
    opacity: float = 1.0

    # Image layer options
    shader: Optional[str] = None

    # Segmentation layer options
    selected_alpha: float = 0.5
    not_selected_alpha: float = 0.1
    segments: List[int] = field(default_factory=list)
    hide_segment_zero: bool = True

    # Annotation layer options
    annotation_color: str = "#ffff00"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Neuroglancer layer dict."""
        layer = {
            "type": self.layer_type.value,
            "source": self.source,
            "name": self.name,
            "visible": self.visible,
        }

        if self.layer_type == LayerType.IMAGE:
            layer["opacity"] = self.opacity
            if self.shader:
                layer["shader"] = self.shader

        elif self.layer_type == LayerType.SEGMENTATION:
            layer["selectedAlpha"] = self.selected_alpha
            layer["notSelectedAlpha"] = self.not_selected_alpha
            if self.segments:
                layer["segments"] = [str(s) for s in self.segments]
            layer["hideSegmentZero"] = self.hide_segment_zero

        elif self.layer_type == LayerType.ANNOTATION:
            layer["annotationColor"] = self.annotation_color

        return layer


@dataclass
class NeuroglancerState:
    """Complete Neuroglancer viewer state."""

    dimensions: Dict[str, Tuple[float, str]]
    position: List[float]
    cross_section_scale: float
    projection_scale: float
    layers: List[LayerConfig]
    selected_layer: Optional[str] = None
    layout: str = "4panel"
    show_axis_lines: bool = True
    show_scale_bar: bool = True
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Neuroglancer state dict."""
        state = {
            "dimensions": {
                k: [v[0], v[1]] for k, v in self.dimensions.items()
            },
            "position": self.position,
            "crossSectionScale": self.cross_section_scale,
            "projectionScale": self.projection_scale,
            "layers": [layer.to_dict() for layer in self.layers],
            "layout": self.layout,
            "showAxisLines": self.show_axis_lines,
            "showScaleBar": self.show_scale_bar,
        }

        if self.selected_layer:
            state["selectedLayer"] = {"layer": self.selected_layer}

        if self.title:
            state["title"] = self.title

        return state

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_url(self, base_url: str = "https://neuroglancer-demo.appspot.com") -> str:
        """Convert to Neuroglancer URL."""
        state_json = self.to_json()
        encoded = urllib.parse.quote(state_json, safe='')
        return f"{base_url}/#!{encoded}"

    def to_url_fragment(self) -> str:
        """Get just the URL fragment (for local servers)."""
        state_json = self.to_json()
        return urllib.parse.quote(state_json, safe='')


class NeuroglancerStateBuilder:
    """Fluent API for building Neuroglancer states."""

    def __init__(self):
        """Initialize builder with defaults."""
        self._dimensions: Dict[str, Tuple[float, str]] = {
            "x": (1.0, "nm"),
            "y": (1.0, "nm"),
            "z": (1.0, "nm"),
        }
        self._position: List[float] = [0.0, 0.0, 0.0]
        self._cross_section_scale: float = 1.0
        self._projection_scale: float = 1000.0
        self._layers: List[LayerConfig] = []
        self._selected_layer: Optional[str] = None
        self._layout: str = "4panel"
        self._show_axis_lines: bool = True
        self._show_scale_bar: bool = True
        self._title: Optional[str] = None

    def with_dimensions(
        self,
        x: Tuple[float, str] = (1.0, "nm"),
        y: Tuple[float, str] = (1.0, "nm"),
        z: Tuple[float, str] = (1.0, "nm"),
    ) -> NeuroglancerStateBuilder:
        """Set coordinate dimensions (resolution and units)."""
        self._dimensions = {"x": x, "y": y, "z": z}
        return self

    def with_resolution(
        self,
        resolution: Tuple[float, float, float],
        unit: str = "nm",
    ) -> NeuroglancerStateBuilder:
        """Set resolution as (x, y, z) tuple."""
        self._dimensions = {
            "x": (resolution[0], unit),
            "y": (resolution[1], unit),
            "z": (resolution[2], unit),
        }
        return self

    def center_on(
        self,
        x: float,
        y: float,
        z: float,
    ) -> NeuroglancerStateBuilder:
        """Set viewer position (center point)."""
        self._position = [x, y, z]
        return self

    def with_zoom(
        self,
        cross_section_scale: float = 1.0,
        projection_scale: float = 1000.0,
    ) -> NeuroglancerStateBuilder:
        """Set zoom levels."""
        self._cross_section_scale = cross_section_scale
        self._projection_scale = projection_scale
        return self

    def with_layout(self, layout: str) -> NeuroglancerStateBuilder:
        """Set layout mode: 4panel, xy, xz, yz, 3d, xy-3d, etc."""
        self._layout = layout
        return self

    def with_title(self, title: str) -> NeuroglancerStateBuilder:
        """Set viewer title."""
        self._title = title
        return self

    def add_layer(self, layer: LayerConfig) -> NeuroglancerStateBuilder:
        """Add a layer configuration."""
        self._layers.append(layer)
        return self

    def with_raw_layer(
        self,
        source: str,
        name: str = "raw",
        shader: Optional[str] = None,
    ) -> NeuroglancerStateBuilder:
        """Add a raw image layer."""
        layer = LayerConfig(
            name=name,
            source=source,
            layer_type=LayerType.IMAGE,
            shader=shader,
        )
        self._layers.append(layer)
        return self

    def with_segmentation_layer(
        self,
        source: str,
        name: str = "segmentation",
        segments: Optional[List[int]] = None,
        selected_alpha: float = 0.5,
        not_selected_alpha: float = 0.1,
    ) -> NeuroglancerStateBuilder:
        """Add a segmentation layer."""
        layer = LayerConfig(
            name=name,
            source=source,
            layer_type=LayerType.SEGMENTATION,
            segments=segments or [],
            selected_alpha=selected_alpha,
            not_selected_alpha=not_selected_alpha,
        )
        self._layers.append(layer)
        return self

    def with_annotation_layer(
        self,
        name: str = "annotations",
        color: str = "#ffff00",
    ) -> NeuroglancerStateBuilder:
        """Add an annotation layer."""
        layer = LayerConfig(
            name=name,
            source="local://annotations",
            layer_type=LayerType.ANNOTATION,
            annotation_color=color,
        )
        self._layers.append(layer)
        return self

    def highlight_segment(
        self,
        segment_id: int,
        layer_name: str = "segmentation",
    ) -> NeuroglancerStateBuilder:
        """Highlight a specific segment in a segmentation layer."""
        for layer in self._layers:
            if layer.name == layer_name and layer.layer_type == LayerType.SEGMENTATION:
                if segment_id not in layer.segments:
                    layer.segments.append(segment_id)
        return self

    def highlight_segments(
        self,
        segment_ids: List[int],
        layer_name: str = "segmentation",
    ) -> NeuroglancerStateBuilder:
        """Highlight multiple segments."""
        for seg_id in segment_ids:
            self.highlight_segment(seg_id, layer_name)
        return self

    def select_layer(self, layer_name: str) -> NeuroglancerStateBuilder:
        """Set the selected/active layer."""
        self._selected_layer = layer_name
        return self

    def show_axis_lines(self, show: bool = True) -> NeuroglancerStateBuilder:
        """Toggle axis lines visibility."""
        self._show_axis_lines = show
        return self

    def show_scale_bar(self, show: bool = True) -> NeuroglancerStateBuilder:
        """Toggle scale bar visibility."""
        self._show_scale_bar = show
        return self

    def build(self) -> NeuroglancerState:
        """Build the final state object."""
        return NeuroglancerState(
            dimensions=self._dimensions,
            position=self._position,
            cross_section_scale=self._cross_section_scale,
            projection_scale=self._projection_scale,
            layers=self._layers,
            selected_layer=self._selected_layer,
            layout=self._layout,
            show_axis_lines=self._show_axis_lines,
            show_scale_bar=self._show_scale_bar,
            title=self._title,
        )


# Preset state builders for common proofreading tasks

def review_segment_state(
    raw_source: str,
    seg_source: str,
    segment_id: int,
    location: Tuple[float, float, float],
    resolution: Tuple[float, float, float] = (4.0, 4.0, 40.0),
) -> NeuroglancerState:
    """Create state for reviewing a single segment.

    Args:
        raw_source: Source URL for raw image data
        seg_source: Source URL for segmentation
        segment_id: ID of segment to review
        location: (x, y, z) center location
        resolution: (x, y, z) voxel resolution in nm

    Returns:
        Configured NeuroglancerState
    """
    return (
        NeuroglancerStateBuilder()
        .with_resolution(resolution)
        .with_raw_layer(raw_source)
        .with_segmentation_layer(seg_source, segments=[segment_id])
        .center_on(*location)
        .with_zoom(cross_section_scale=2.0, projection_scale=500.0)
        .with_title(f"Review Segment {segment_id}")
        .build()
    )


def merge_error_review_state(
    raw_source: str,
    seg_source: str,
    segment_ids: List[int],
    location: Tuple[float, float, float],
    resolution: Tuple[float, float, float] = (4.0, 4.0, 40.0),
) -> NeuroglancerState:
    """Create state for reviewing potential merge error.

    Args:
        raw_source: Source URL for raw image data
        seg_source: Source URL for segmentation
        segment_ids: IDs of potentially incorrectly merged segments
        location: (x, y, z) center at merge boundary
        resolution: (x, y, z) voxel resolution in nm

    Returns:
        Configured NeuroglancerState
    """
    return (
        NeuroglancerStateBuilder()
        .with_resolution(resolution)
        .with_raw_layer(raw_source)
        .with_segmentation_layer(
            seg_source,
            segments=segment_ids,
            selected_alpha=0.7,
            not_selected_alpha=0.05,
        )
        .with_annotation_layer("merge_boundary", color="#ff0000")
        .center_on(*location)
        .with_zoom(cross_section_scale=1.5, projection_scale=300.0)
        .with_title(f"Merge Error Review: Segments {segment_ids}")
        .build()
    )


def split_error_review_state(
    raw_source: str,
    seg_source: str,
    segment_id: int,
    location: Tuple[float, float, float],
    resolution: Tuple[float, float, float] = (4.0, 4.0, 40.0),
) -> NeuroglancerState:
    """Create state for reviewing potential split error.

    Args:
        raw_source: Source URL for raw image data
        seg_source: Source URL for segmentation
        segment_id: ID of segment that may need to be split
        location: (x, y, z) center at potential split location
        resolution: (x, y, z) voxel resolution in nm

    Returns:
        Configured NeuroglancerState
    """
    return (
        NeuroglancerStateBuilder()
        .with_resolution(resolution)
        .with_raw_layer(raw_source)
        .with_segmentation_layer(
            seg_source,
            segments=[segment_id],
            selected_alpha=0.8,
        )
        .with_annotation_layer("split_points", color="#00ff00")
        .center_on(*location)
        .with_zoom(cross_section_scale=1.0, projection_scale=200.0)
        .with_title(f"Split Error Review: Segment {segment_id}")
        .build()
    )


def full_volume_browse_state(
    raw_source: str,
    seg_source: str,
    volume_center: Tuple[float, float, float],
    resolution: Tuple[float, float, float] = (4.0, 4.0, 40.0),
) -> NeuroglancerState:
    """Create state for browsing full volume.

    Args:
        raw_source: Source URL for raw image data
        seg_source: Source URL for segmentation
        volume_center: (x, y, z) center of volume
        resolution: (x, y, z) voxel resolution in nm

    Returns:
        Configured NeuroglancerState
    """
    return (
        NeuroglancerStateBuilder()
        .with_resolution(resolution)
        .with_raw_layer(raw_source)
        .with_segmentation_layer(seg_source, selected_alpha=0.3)
        .with_annotation_layer()
        .center_on(*volume_center)
        .with_zoom(cross_section_scale=4.0, projection_scale=2000.0)
        .with_layout("4panel")
        .with_title("Volume Browser")
        .build()
    )
