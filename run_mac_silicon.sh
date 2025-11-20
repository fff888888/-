#!/usr/bin/env bash
# 在 macOS Apple Silicon 上一键启动 Web UI，同时注入稳定性相关的环境变量。

set -euo pipefail

# 仅在未提前设置时添加默认值，避免覆盖用户的显式配置。
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export ORT_USE_OPENMP=${ORT_USE_OPENMP:-0}
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}

# 避免 uvicorn 在本地 CPU 场景下拉起多个 worker。
export UVICORN_WORKERS=${UVICORN_WORKERS:-1}

# 默认使用仓库内的 start_app 入口，可通过参数覆盖索引/模型路径。
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHONPATH=. python start_app.py "$@"
