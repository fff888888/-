"""Video search pipeline utilities."""

from .runtime import apply_macos_omp_fixes

# Apply macOS OpenMP guards as early as possible so downstream imports share
# the same stable environment.
apply_macos_omp_fixes()

from .frames import extract_keyframes
from .embeddings import ensure_metadata_embeddings, metadata_has_embeddings
from .features import OnnxClipEncoder, build_frame_feature_cache
from .metadata import FrameRecord, VideoMetadata, load_metadata, save_metadata
from .index import FaissIndexer
from .pipeline import build_or_update_index, process_video_to_embeddings
from .webapp import WebAppConfig, create_app
from .jobs import JobStore, JobRecord

__all__ = [
    "extract_keyframes",
    "OnnxClipEncoder",
    "build_frame_feature_cache",
    "FrameRecord",
    "VideoMetadata",
    "load_metadata",
    "save_metadata",
    "FaissIndexer",
    "process_video_to_embeddings",
    "build_or_update_index",
    "ensure_metadata_embeddings",
    "metadata_has_embeddings",
    "WebAppConfig",
    "create_app",
    "JobStore",
    "JobRecord",
]
