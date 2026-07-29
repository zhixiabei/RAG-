#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="${RAG_PROJECT_DIR:-$HOME/startwork/RAG}"
SERVICES_DIR="${RAG_SERVICES_DIR:-/root/autodl-tmp/rag-services}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rag}"
QDRANT_DIR="$SERVICES_DIR/qdrant"
MINIO_DIR="$SERVICES_DIR/minio"
FRONTEND_DIR="$PROJECT_DIR/frontend"

BACKEND_PID=""
FRONTEND_PID=""
USE_PROCESS_GROUPS=false

fail() {
  printf 'Startup failed: %s\n' "$*" >&2
  exit 1
}

port_is_open() {
  local port="$1"
  (exec 3<>"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1
}

wait_for_port() {
  local service_name="$1"
  local port="$2"
  local timeout_seconds="${3:-30}"
  local elapsed=0

  until port_is_open "$port"; do
    if ((elapsed >= timeout_seconds)); then
      fail "$service_name did not listen on 127.0.0.1:$port within ${timeout_seconds}s"
    fi
    sleep 1
    ((elapsed += 1))
  done
  printf '%s is ready on 127.0.0.1:%s\n' "$service_name" "$port"
}

start_daemon() {
  local service_name="$1"
  local process_name="$2"
  local port="$3"
  local working_directory="$4"
  local pid_file="$5"
  local log_file="$6"
  shift 6

  if port_is_open "$port"; then
    printf '%s is already listening on 127.0.0.1:%s\n' "$service_name" "$port"
    return
  fi

  if [[ -f "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(<"$pid_file")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      printf '%s is starting with PID %s\n' "$service_name" "$existing_pid"
      wait_for_port "$service_name" "$port"
      return
    fi
  fi

  if command -v pgrep >/dev/null 2>&1 && pgrep -x "$process_name" >/dev/null 2>&1; then
    printf '%s process already exists; waiting for its port\n' "$service_name"
    wait_for_port "$service_name" "$port"
    return
  fi

  (
    cd "$working_directory"
    nohup "$@" >"$log_file" 2>&1 &
    printf '%s\n' "$!" >"$pid_file"
  )
  wait_for_port "$service_name" "$port"
}

stop_managed_process() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0

  if [[ "$USE_PROCESS_GROUPS" == true ]]; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
}

cleanup_apps() {
  trap - EXIT INT TERM
  stop_managed_process "$FRONTEND_PID"
  stop_managed_process "$BACKEND_PID"
  [[ -z "$FRONTEND_PID" ]] || wait "$FRONTEND_PID" 2>/dev/null || true
  [[ -z "$BACKEND_PID" ]] || wait "$BACKEND_PID" 2>/dev/null || true
}

start_managed_process() {
  if [[ "$USE_PROCESS_GROUPS" == true ]]; then
    setsid "$@" &
  else
    "$@" &
  fi
  MANAGED_PID="$!"
}

[[ -d "$PROJECT_DIR" ]] || fail "project directory not found: $PROJECT_DIR"
[[ -x "$QDRANT_DIR/qdrant" ]] || fail "Qdrant executable not found: $QDRANT_DIR/qdrant"
[[ -f "$QDRANT_DIR/config.yaml" ]] || fail "Qdrant config not found: $QDRANT_DIR/config.yaml"
[[ -x "$MINIO_DIR/minio" ]] || fail "MinIO executable not found: $MINIO_DIR/minio"
[[ -f "$FRONTEND_DIR/package.json" ]] || fail "frontend package.json not found: $FRONTEND_DIR/package.json"
command -v service >/dev/null 2>&1 || fail "service command is unavailable"
command -v conda >/dev/null 2>&1 || fail "conda is unavailable in PATH"

if command -v setsid >/dev/null 2>&1; then
  USE_PROCESS_GROUPS=true
fi

printf 'Starting PostgreSQL...\n'
if ((EUID == 0)); then
  service postgresql start
elif command -v sudo >/dev/null 2>&1; then
  sudo service postgresql start
else
  fail "PostgreSQL requires root privileges and sudo is unavailable"
fi
wait_for_port "PostgreSQL" 5432

start_daemon \
  "Qdrant" "qdrant" 6333 "$QDRANT_DIR" \
  "$QDRANT_DIR/qdrant.pid" "$QDRANT_DIR/qdrant.log" \
  ./qdrant --config-path config.yaml

mkdir -p "$MINIO_DIR/data"
start_daemon \
  "MinIO" "minio" 9000 "$MINIO_DIR" \
  "$MINIO_DIR/minio.pid" "$MINIO_DIR/minio.log" \
  env MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
  ./minio server "$MINIO_DIR/data" \
  --address 127.0.0.1:9000 --console-address 127.0.0.1:9001

deactivate 2>/dev/null || true
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
command -v python >/dev/null 2>&1 || fail "python is unavailable in Conda environment: $CONDA_ENV_NAME"
command -v npm >/dev/null 2>&1 || fail "npm is unavailable in PATH"

export CORS_ORIGINS="${CORS_ORIGINS:-http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:6008,http://localhost:6008}"

port_is_open 8080 && fail "backend port 8080 is already in use"
port_is_open 6008 && fail "frontend port 6008 is already in use"

trap cleanup_apps EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$PROJECT_DIR"
start_managed_process \
  python -m uvicorn rag_app.main:app \
  --app-dir backend/src \
  --host 127.0.0.1 \
  --port 8080
BACKEND_PID="$MANAGED_PID"

cd "$FRONTEND_DIR"
start_managed_process npm run dev -- --host 127.0.0.1 --port 6008
FRONTEND_PID="$MANAGED_PID"
cd "$PROJECT_DIR"

wait_for_port "Backend" 8080
wait_for_port "Frontend" 6008

printf '\nRAG server is running.\n'
printf 'Frontend: http://127.0.0.1:6008\n'
printf 'Backend:  http://127.0.0.1:8080/docs\n'
printf 'Press Ctrl+C to stop the frontend and backend. PostgreSQL, Qdrant, and MinIO remain running.\n\n'

set +e
wait -n "$BACKEND_PID" "$FRONTEND_PID"
exit_code="$?"
set -e

if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
  printf 'Backend exited unexpectedly (status %s).\n' "$exit_code" >&2
else
  printf 'Frontend exited unexpectedly (status %s).\n' "$exit_code" >&2
fi
exit "$exit_code"
