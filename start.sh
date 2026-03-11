#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_ENABLED="${LOGS_ENABLED:-true}"
[[ "${1:-}" == "--no-logs" ]] && LOGS_ENABLED=false

FANUC_COMPOSE="${ROOT_DIR}/GIGAS-CODE/ros2-fanuc-robot-interface/docker-compose.yml"
PIPELINE_COMPOSE="${ROOT_DIR}/tomato_pipeline/docker-compose.yml"
ENV_FILE="${ROOT_DIR}/env/runtime.env"
ROS2_NETWORK="ros2_humble_net"

FANUC_COMPOSE_ARGS=(-f "${FANUC_COMPOSE}")
PIPELINE_COMPOSE_ARGS=(-f "${PIPELINE_COMPOSE}")

if [ -f "${ENV_FILE}" ]; then
  FANUC_COMPOSE_ARGS+=(--env-file "${ENV_FILE}")
  PIPELINE_COMPOSE_ARGS+=(--env-file "${ENV_FILE}")
  # Export runtime values for shell-level interpolation consistency.
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# Allow the container to open OpenCV GUI windows on the local X server.
if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null 2>&1; then
  xhost +local:root >/dev/null 2>&1 || true
fi

# Ensure the shared ROS2 bridge network exists (used by both compose projects).
if ! docker network inspect "${ROS2_NETWORK}" >/dev/null 2>&1; then
  docker network create "${ROS2_NETWORK}" >/dev/null
fi

# 1) Start shared robot interfaces container.
docker compose "${FANUC_COMPOSE_ARGS[@]}" up -d --no-build --no-deps \
  interfaces
echo "Started service: interfaces"

# 2) Start Fanuc interface (creates ros2_humble_net used by tomato-pipeline).
docker compose "${FANUC_COMPOSE_ARGS[@]}" up -d --no-build --no-deps \
  ros2-fanuc-interface
echo "Started service: ros2-fanuc-interface"

# 3) Start tomato pipeline (joins ros2_humble_net)
docker compose "${PIPELINE_COMPOSE_ARGS[@]}" up -d --no-build \
  tomato-pipeline

echo "Started service: tomato-pipeline"
echo "Pipeline artifacts path pattern: /data/<date>/<trial>"
echo "Current default trial: ${PIPELINE_TRIAL_NUM:-1}"

# Open tabs for fanuc logs + pipeline logs + keyboard trigger utility.
if [ "${LOGS_ENABLED}" = true ]; then
  if command -v gnome-terminal >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
    LOG_TAB_FAILED=false

    # Follow the superproject style: launch each tab with its own gnome-terminal invocation.
    for service in ros2-fanuc-interface tomato-pipeline; do
      if [ "${service}" = "ros2-fanuc-interface" ]; then
        compose_file="${FANUC_COMPOSE}"
      else
        compose_file="${PIPELINE_COMPOSE}"
      fi

      if ! gnome-terminal --tab --title="${service}" -- bash -lc "
        echo 'Service: ${service}'
        echo '--------------------------------------'
        docker compose -f '${compose_file}' logs --tail 200 -f ${service} || true
        exec bash
      "; then
        LOG_TAB_FAILED=true
      fi

      # Keep launch order predictable and avoid DBus tab race issues.
      sleep 0.1
    done

    # Dedicated interactive trigger tab (Enter => run pipeline, q => quit utility).
    if ! gnome-terminal --tab --title="pipeline-trigger" -- bash -lc "
      echo 'Service: pipeline-trigger'
      echo '--------------------------------------'
      docker exec -it tomato-pipeline bash -lc \"source /workspace/install/setup.bash && ros2 run tomato_pipeline pipeline_trigger_keyboard\" || true
      exec bash
    "; then
      LOG_TAB_FAILED=true
    fi

    if [ "${LOG_TAB_FAILED}" = true ]; then
      echo "Warning: could not open one or more gnome-terminal log tabs."
      echo "Run logs manually:"
      echo "  docker compose ${FANUC_COMPOSE_ARGS[*]} logs --tail 200 -f ros2-fanuc-interface"
      echo "  docker compose ${PIPELINE_COMPOSE_ARGS[*]} logs --tail 200 -f tomato-pipeline"
      echo "  docker exec -it tomato-pipeline bash -lc \"source /workspace/install/setup.bash && ros2 run tomato_pipeline pipeline_trigger_keyboard\""
    fi
  else
    echo "Skipping auto log tabs: no graphical terminal session detected."
    echo "Run logs manually:"
    echo "  docker compose ${FANUC_COMPOSE_ARGS[*]} logs --tail 200 -f ros2-fanuc-interface"
    echo "  docker compose ${PIPELINE_COMPOSE_ARGS[*]} logs --tail 200 -f tomato-pipeline"
    echo "  docker exec -it tomato-pipeline bash -lc \"source /workspace/install/setup.bash && ros2 run tomato_pipeline pipeline_trigger_keyboard\""
  fi
fi
