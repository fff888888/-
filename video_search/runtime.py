"""Runtime guards and environment defaults for stability.

This module centralizes process-level tweaks that improve stability on
macOS/Apple Silicon where OpenMP/libomp can be loaded multiple times by
OpenCV, FFmpeg, and onnxruntime. By applying them once at package import
time, downstream scripts (CLI/Web) don't need to remember to export a long
list of env vars.
"""

from __future__ import annotations

import os
import platform


def apply_macos_omp_fixes() -> None:
    """Set conservative defaults to avoid libomp duplication crashes.

    These defaults are intentionally minimal and only applied when running on
    macOS. They keep threading to a single worker, favor a non-spinning wait
    policy, and opt out of OpenMP inside onnxruntime where possible. Users can
    still override the values by exporting the variables before launch.
    """

    if platform.system() != "Darwin":
        return

    defaults = {
        "OMP_NUM_THREADS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        # Prefer pthreads/eigen over OpenMP in ORT to avoid double-loading
        # libomp from both onnxruntime-silicon and OpenCV/FFmpeg wheels.
        "ORT_USE_OPENMP": "0",
        # Avoid fork safety crashes when uvicorn/fastapi spawn worker threads.
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
    }

    for key, value in defaults.items():
        os.environ.setdefault(key, value)


__all__ = ["apply_macos_omp_fixes"]
