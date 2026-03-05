"""
MOSS-Neuroglancer bridge for proofreading workflows.

Provides integration between MOSS (PyQt6 desktop app) and Neuroglancer
for seamless proofreading experience.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .neuroglancer_state import (
    NeuroglancerState,
    NeuroglancerStateBuilder,
    full_volume_browse_state,
    merge_error_review_state,
    review_segment_state,
    split_error_review_state,
)
from .viewer import ProofreadingViewer, ViewerConfig


class TaskType(Enum):
    """Types of proofreading tasks."""
    REVIEW = "review"           # Review a single segment
    MERGE_ERROR = "merge"       # Check potential merge error
    SPLIT_ERROR = "split"       # Check potential split error
    QUALITY_SAMPLE = "sample"   # Random quality sampling
    BROWSE = "browse"           # Free browsing


class TaskStatus(Enum):
    """Status of a proofreading task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class ProofreadingTask:
    """A single proofreading task for the queue."""

    task_id: str
    task_type: TaskType
    location: Tuple[int, int, int]  # (x, y, z) voxel coordinates
    segment_ids: List[int]
    priority: float  # 0-1, higher = more important
    description: str
    status: TaskStatus = TaskStatus.PENDING

    # Results
    result: Optional[Dict[str, Any]] = None
    completed_at: Optional[datetime] = None
    reviewer: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "location": list(self.location),
            "segment_ids": self.segment_ids,
            "priority": self.priority,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "reviewer": self.reviewer,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProofreadingTask:
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            location=tuple(data["location"]),
            segment_ids=data["segment_ids"],
            priority=data["priority"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            result=data.get("result"),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            reviewer=data.get("reviewer"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class TaskQueue:
    """Queue of proofreading tasks with persistence."""

    def __init__(self, save_path: Optional[Path] = None):
        """Initialize task queue.

        Args:
            save_path: Optional path to save/load queue state
        """
        self.tasks: List[ProofreadingTask] = []
        self.save_path = save_path
        self._current_index = 0

    def add_task(self, task: ProofreadingTask) -> None:
        """Add a task to the queue."""
        self.tasks.append(task)
        self._sort_by_priority()

    def add_tasks(self, tasks: List[ProofreadingTask]) -> None:
        """Add multiple tasks."""
        self.tasks.extend(tasks)
        self._sort_by_priority()

    def _sort_by_priority(self) -> None:
        """Sort tasks by priority (highest first), then by status."""
        status_order = {
            TaskStatus.IN_PROGRESS: 0,
            TaskStatus.PENDING: 1,
            TaskStatus.COMPLETED: 2,
            TaskStatus.SKIPPED: 3,
        }
        self.tasks.sort(
            key=lambda t: (status_order[t.status], -t.priority)
        )

    def get_next_task(self) -> Optional[ProofreadingTask]:
        """Get the next pending task."""
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                return task
        return None

    def get_task(self, task_id: str) -> Optional[ProofreadingTask]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any],
        reviewer: Optional[str] = None,
    ) -> None:
        """Mark a task as completed."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            task.reviewer = reviewer
            self._sort_by_priority()

    def skip_task(self, task_id: str) -> None:
        """Skip a task."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.SKIPPED
            self._sort_by_priority()

    @property
    def pending_count(self) -> int:
        """Number of pending tasks."""
        return sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)

    @property
    def completed_count(self) -> int:
        """Number of completed tasks."""
        return sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)

    @property
    def progress(self) -> float:
        """Progress as fraction (0-1)."""
        total = len(self.tasks)
        if total == 0:
            return 1.0
        done = self.completed_count + sum(1 for t in self.tasks if t.status == TaskStatus.SKIPPED)
        return done / total

    def save(self, path: Optional[Path] = None) -> None:
        """Save queue state to JSON."""
        save_path = path or self.save_path
        if save_path is None:
            raise ValueError("No save path specified")

        data = {
            "tasks": [t.to_dict() for t in self.tasks],
            "saved_at": datetime.now().isoformat(),
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[Path] = None) -> None:
        """Load queue state from JSON."""
        load_path = path or self.save_path
        if load_path is None:
            raise ValueError("No load path specified")

        with open(load_path, 'r') as f:
            data = json.load(f)

        self.tasks = [ProofreadingTask.from_dict(t) for t in data["tasks"]]
        self._sort_by_priority()


class MOSSBridge:
    """Bridge between MOSS and Neuroglancer for proofreading.

    This class is designed to be imported and used directly by MOSS.
    It handles:
    - Task queue generation from segmentation results
    - Neuroglancer state building for different task types
    - Viewer management and browser launching
    """

    def __init__(
        self,
        project_dir: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize MOSS bridge.

        Args:
            project_dir: MOSS project directory
            config: Optional configuration overrides
        """
        self.project_dir = Path(project_dir)
        self.config = config or {}

        # Default paths
        self.raw_dir = self.project_dir / "raw_images"
        self.masks_dir = self.project_dir / "masks"
        self.predictions_dir = self.project_dir / "predictions"

        # Proofreading state
        self.task_queue = TaskQueue(
            save_path=self.project_dir / "proofreading_tasks.json"
        )

        # Viewer
        self._viewer: Optional[ProofreadingViewer] = None

        # Resolution (default for EM)
        self.resolution = self.config.get("resolution", (4.0, 4.0, 40.0))

    def get_viewer(self) -> ProofreadingViewer:
        """Get or create the viewer instance."""
        if self._viewer is None:
            viewer_config = ViewerConfig(
                port=self.config.get("neuroglancer_port", 8080),
                data_directory=self.project_dir,
            )
            self._viewer = ProofreadingViewer(viewer_config)
        return self._viewer

    def start_server(self) -> str:
        """Start the data server.

        Returns:
            Server URL
        """
        viewer = self.get_viewer()
        return viewer.start_server()

    def stop_server(self) -> None:
        """Stop the data server."""
        if self._viewer:
            self._viewer.stop_server()

    def generate_tasks_from_predictions(
        self,
        predictions_path: Optional[Path] = None,
        num_samples: int = 50,
        include_merge_errors: bool = True,
        include_split_errors: bool = True,
    ) -> List[ProofreadingTask]:
        """Generate proofreading tasks from prediction results.

        Args:
            predictions_path: Path to predictions (zarr/folder of masks)
            num_samples: Number of random samples to include
            include_merge_errors: Include potential merge error tasks
            include_split_errors: Include potential split error tasks

        Returns:
            List of generated tasks
        """
        pred_path = predictions_path or self.predictions_dir
        tasks = []

        # Try to load predictions as zarr or folder of masks
        try:
            import zarr
            if (pred_path / '.zarray').exists() or (pred_path / '.zgroup').exists():
                predictions = zarr.open(str(pred_path), mode='r')
                if isinstance(predictions, zarr.Group):
                    # Look for segmentation array
                    for key in ['segmentation', 'labels', 'masks', 's0']:
                        if key in predictions:
                            predictions = predictions[key]
                            break
                volume_shape = predictions.shape
            else:
                # Folder of mask images
                mask_files = sorted(pred_path.glob("*.tif")) + sorted(pred_path.glob("*.png"))
                if mask_files:
                    from PIL import Image
                    first = np.array(Image.open(mask_files[0]))
                    volume_shape = (len(mask_files), *first.shape)
                    predictions = None  # Will load on demand
                else:
                    raise FileNotFoundError(f"No predictions found in {pred_path}")

        except ImportError:
            # No zarr, try as folder
            mask_files = sorted(pred_path.glob("*.tif")) + sorted(pred_path.glob("*.png"))
            if mask_files:
                from PIL import Image
                first = np.array(Image.open(mask_files[0]))
                volume_shape = (len(mask_files), *first.shape)
                predictions = None
            else:
                raise FileNotFoundError(f"No predictions found in {pred_path}")

        # Generate quality sampling tasks
        tasks.extend(self._generate_sample_tasks(volume_shape, num_samples))

        # Generate error detection tasks (heuristic-based)
        if include_merge_errors:
            tasks.extend(self._generate_merge_error_tasks(volume_shape))

        if include_split_errors:
            tasks.extend(self._generate_split_error_tasks(volume_shape))

        # Add to queue
        self.task_queue.add_tasks(tasks)

        return tasks

    def _generate_sample_tasks(
        self,
        volume_shape: Tuple[int, ...],
        num_samples: int,
    ) -> List[ProofreadingTask]:
        """Generate random sampling tasks."""
        tasks = []
        z_size, y_size, x_size = volume_shape[:3]

        for i in range(num_samples):
            # Random location (avoid edges)
            x = int(np.random.uniform(x_size * 0.1, x_size * 0.9))
            y = int(np.random.uniform(y_size * 0.1, y_size * 0.9))
            z = int(np.random.uniform(z_size * 0.1, z_size * 0.9))

            task = ProofreadingTask(
                task_id=str(uuid.uuid4())[:8],
                task_type=TaskType.QUALITY_SAMPLE,
                location=(x, y, z),
                segment_ids=[],
                priority=0.5,  # Medium priority
                description=f"Quality sample at ({x}, {y}, {z})",
            )
            tasks.append(task)

        return tasks

    def _generate_merge_error_tasks(
        self,
        volume_shape: Tuple[int, ...],
    ) -> List[ProofreadingTask]:
        """Generate potential merge error tasks.

        TODO: Implement actual merge error detection using:
        - Boundary uncertainty
        - Segment size anomalies
        - Connectivity analysis
        """
        # Placeholder - return empty for now
        # Real implementation would analyze the segmentation
        return []

    def _generate_split_error_tasks(
        self,
        volume_shape: Tuple[int, ...],
    ) -> List[ProofreadingTask]:
        """Generate potential split error tasks.

        TODO: Implement actual split error detection using:
        - Segment continuity analysis
        - Size/shape anomalies
        - Multi-view disagreement
        """
        # Placeholder - return empty for now
        return []

    def get_task_state(self, task: ProofreadingTask) -> NeuroglancerState:
        """Get Neuroglancer state for a task.

        Args:
            task: Proofreading task

        Returns:
            Configured Neuroglancer state
        """
        # Get data source URLs
        viewer = self.get_viewer()
        if not viewer.is_serving:
            viewer.start_server()

        # Determine raw source
        raw_source = self._get_raw_source()
        seg_source = self._get_segmentation_source()

        # Build state based on task type
        if task.task_type == TaskType.REVIEW:
            return review_segment_state(
                raw_source=raw_source,
                seg_source=seg_source,
                segment_id=task.segment_ids[0] if task.segment_ids else 0,
                location=task.location,
                resolution=self.resolution,
            )

        elif task.task_type == TaskType.MERGE_ERROR:
            return merge_error_review_state(
                raw_source=raw_source,
                seg_source=seg_source,
                segment_ids=task.segment_ids,
                location=task.location,
                resolution=self.resolution,
            )

        elif task.task_type == TaskType.SPLIT_ERROR:
            return split_error_review_state(
                raw_source=raw_source,
                seg_source=seg_source,
                segment_id=task.segment_ids[0] if task.segment_ids else 0,
                location=task.location,
                resolution=self.resolution,
            )

        else:  # QUALITY_SAMPLE or BROWSE
            return full_volume_browse_state(
                raw_source=raw_source,
                seg_source=seg_source,
                volume_center=task.location,
                resolution=self.resolution,
            )

    def _get_raw_source(self) -> str:
        """Get Neuroglancer source URL for raw data."""
        viewer = self.get_viewer()

        # Check for zarr
        zarr_path = self.project_dir / "raw.zarr"
        if zarr_path.exists():
            return viewer.get_local_source_url(zarr_path, "zarr")

        # Check for precomputed
        precomputed_path = self.project_dir / "raw_precomputed"
        if precomputed_path.exists():
            return viewer.get_local_source_url(precomputed_path, "precomputed")

        # Fall back to raw_images folder (would need conversion)
        if self.raw_dir.exists():
            # Return placeholder - real implementation would convert on the fly
            return f"precomputed://{viewer.server_url}/raw_images"

        raise FileNotFoundError("No raw data source found")

    def _get_segmentation_source(self) -> str:
        """Get Neuroglancer source URL for segmentation."""
        viewer = self.get_viewer()

        # Check for zarr
        zarr_path = self.project_dir / "segmentation.zarr"
        if zarr_path.exists():
            return viewer.get_local_source_url(zarr_path, "zarr")

        # Check for predictions zarr
        pred_zarr = self.predictions_dir / "segmentation.zarr"
        if pred_zarr.exists():
            return viewer.get_local_source_url(pred_zarr, "zarr")

        # Check for masks folder
        if self.masks_dir.exists():
            return f"precomputed://{viewer.server_url}/masks"

        # Check predictions folder
        if self.predictions_dir.exists():
            return f"precomputed://{viewer.server_url}/predictions"

        raise FileNotFoundError("No segmentation source found")

    def launch_task(self, task: ProofreadingTask) -> str:
        """Launch Neuroglancer for a specific task.

        Args:
            task: Task to launch

        Returns:
            Neuroglancer URL
        """
        task.status = TaskStatus.IN_PROGRESS
        state = self.get_task_state(task)
        viewer = self.get_viewer()
        return viewer.open_state(state)

    def launch_next_task(self) -> Optional[str]:
        """Launch Neuroglancer for the next pending task.

        Returns:
            Neuroglancer URL or None if no pending tasks
        """
        task = self.task_queue.get_next_task()
        if task:
            return self.launch_task(task)
        return None

    def launch_browse(self) -> str:
        """Launch Neuroglancer in browse mode (no specific task).

        Returns:
            Neuroglancer URL
        """
        task = ProofreadingTask(
            task_id="browse",
            task_type=TaskType.BROWSE,
            location=(0, 0, 0),
            segment_ids=[],
            priority=0,
            description="Browse volume",
        )
        state = self.get_task_state(task)
        viewer = self.get_viewer()
        return viewer.open_state(state)

    def get_task_url(self, task: ProofreadingTask) -> str:
        """Get Neuroglancer URL for a task without opening browser.

        Args:
            task: Task to get URL for

        Returns:
            Neuroglancer URL
        """
        state = self.get_task_state(task)
        viewer = self.get_viewer()
        return viewer.create_url(state)

    def save_progress(self) -> None:
        """Save task queue progress."""
        self.task_queue.save()

    def load_progress(self) -> None:
        """Load task queue progress."""
        if self.task_queue.save_path and self.task_queue.save_path.exists():
            self.task_queue.load()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of proofreading progress.

        Returns:
            Summary dictionary
        """
        return {
            "total_tasks": len(self.task_queue.tasks),
            "pending": self.task_queue.pending_count,
            "completed": self.task_queue.completed_count,
            "skipped": sum(1 for t in self.task_queue.tasks if t.status == TaskStatus.SKIPPED),
            "progress": self.task_queue.progress,
        }
