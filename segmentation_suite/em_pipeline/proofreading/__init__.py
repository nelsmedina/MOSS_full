"""Proofreading module with Neuroglancer integration."""
from .neuroglancer_state import (
    NeuroglancerStateBuilder,
    NeuroglancerState,
    LayerConfig,
)
from .viewer import (
    ProofreadingViewer,
    ViewerConfig,
)
from .moss_bridge import (
    MOSSBridge,
    ProofreadingTask,
    TaskType,
    TaskStatus,
)
