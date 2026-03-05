#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FANUC_COMPOSE="${ROOT_DIR}/GIGAS-CODE/ros2-fanuc-robot-interface/docker-compose.yml"
PIPELINE_COMPOSE="${ROOT_DIR}/tomato_pipeline/docker-compose.yml"
ENV_FILE="${ROOT_DIR}/env/runtime.env"

FANUC_COMPOSE_ARGS=(-f "${FANUC_COMPOSE}")
PIPELINE_COMPOSE_ARGS=(-f "${PIPELINE_COMPOSE}")

if [ -f "${ENV_FILE}" ]; then
  FANUC_COMPOSE_ARGS+=(--env-file "${ENV_FILE}")
  PIPELINE_COMPOSE_ARGS+=(--env-file "${ENV_FILE}")
fi

# Stop in reverse runtime order: pipeline -> fanuc -> interfaces.

docker compose "${PIPELINE_COMPOSE_ARGS[@]}" stop \
  tomato-pipeline || true

docker compose "${PIPELINE_COMPOSE_ARGS[@]}" rm -f \
  tomato-pipeline || true

echo "Stopped service: tomato-pipeline"

docker compose "${FANUC_COMPOSE_ARGS[@]}" stop \
  ros2-fanuc-interface || true

docker compose "${FANUC_COMPOSE_ARGS[@]}" rm -f \
  ros2-fanuc-interface || true

echo "Stopped service: ros2-fanuc-interface"

docker compose "${FANUC_COMPOSE_ARGS[@]}" stop \
  interfaces || true

docker compose "${FANUC_COMPOSE_ARGS[@]}" rm -f \
  interfaces || true

echo "Stopped service: interfaces"
