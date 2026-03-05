#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FANUC_COMPOSE="${ROOT_DIR}/GIGAS-CODE/ros2-fanuc-robot-interface/docker-compose.yml"
PIPELINE_COMPOSE="${ROOT_DIR}/tomato_pipeline/docker-compose.yml"
ENV_FILE="${ROOT_DIR}/env/runtime.env"
IK_BIN="${ROOT_DIR}/GIGAS-CODE/ros2-fanuc-robot-interface/workspaces/fanuc_interface/src/fanuc_interface/fanuc_interface/ik"

FANUC_COMPOSE_ARGS=(-f "${FANUC_COMPOSE}")
PIPELINE_COMPOSE_ARGS=(-f "${PIPELINE_COMPOSE}")

if [ -f "${ENV_FILE}" ]; then
  FANUC_COMPOSE_ARGS+=(--env-file "${ENV_FILE}")
  PIPELINE_COMPOSE_ARGS+=(--env-file "${ENV_FILE}")
fi

# Keep IKFast binary executable in source-mounted workflows.
if [ -f "${IK_BIN}" ]; then
  chmod +x "${IK_BIN}"
fi

# Build shared robot interfaces image first (service: interfaces).
docker compose "${FANUC_COMPOSE_ARGS[@]}" build interfaces
echo "Built image: gigas/interfaces"

# Build Fanuc interface
docker compose "${FANUC_COMPOSE_ARGS[@]}" build ros2-fanuc-interface
echo "Built service: ros2-fanuc-interface"

# Build tomato pipeline
# Use --no-cache to avoid reusing an old layer that installed a too-new setuptools version.
docker compose "${PIPELINE_COMPOSE_ARGS[@]}" build --no-cache tomato-pipeline
echo "Built service: tomato-pipeline"
