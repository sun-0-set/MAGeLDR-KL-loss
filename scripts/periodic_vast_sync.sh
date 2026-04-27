#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  RUN_ID=vast_2stage_T5_b38a1_diskfix_20260425_140015 \
  SSH_HOST=210.157.233.86 \
  SSH_PORT=54845 \
  scripts/periodic_vast_sync.sh

Periodically rsyncs a Vast run back to local storage. This script never destroys
an instance; pair it with watch_vast_run_sync_destroy.sh for final cleanup.
USAGE
}

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    log "missing required environment variable: ${name}"
    usage
    exit 2
  fi
}

require_env RUN_ID
require_env SSH_HOST
require_env SSH_PORT

SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/workspace/results/k6_promptcv}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/workspace/run_logs}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-../results/k6_promptcv}"
LOCAL_LOG_ROOT="${LOCAL_LOG_ROOT:-../results/vast_run_logs}"
SYNC_SECONDS="${SYNC_SECONDS:-300}"
SSH_MISS_LIMIT="${SSH_MISS_LIMIT:-20}"

REMOTE_RESULTS_DIR="${REMOTE_RESULTS_ROOT%/}/${RUN_ID}"
REMOTE_LOG_FILE="${REMOTE_LOG_ROOT%/}/${RUN_ID}.log"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_ROOT%/}/${RUN_ID}"
LOCAL_LOG_FILE="${LOCAL_LOG_ROOT%/}/${RUN_ID}.log"
LOCAL_FINAL_MARKER="${LOCAL_RESULTS_DIR}/SYNC_COMPLETE.txt"
LOCAL_PERIODIC_MARKER="${LOCAL_RESULTS_DIR}/PERIODIC_SYNC.txt"

SSH_OPTS=(
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -p "$SSH_PORT"
)
SSH_TARGET="${SSH_USER}@${SSH_HOST}"

mkdir -p "$LOCAL_RESULTS_DIR" "$LOCAL_LOG_ROOT"

ssh_misses=0
log "periodic sync starting for ${SSH_TARGET}:${REMOTE_RESULTS_DIR}"
while true; do
  if [[ -f "$LOCAL_FINAL_MARKER" ]]; then
    log "final sync marker exists; exiting periodic sync"
    exit 0
  fi

  set +e
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "test -d '$REMOTE_RESULTS_DIR'" >/dev/null 2>&1
  ssh_rc=$?
  set -e

  if (( ssh_rc != 0 )); then
    ssh_misses=$((ssh_misses + 1))
    log "remote unavailable or result dir missing (rc=${ssh_rc}); miss ${ssh_misses}/${SSH_MISS_LIMIT}"
    if (( ssh_misses >= SSH_MISS_LIMIT )); then
      log "miss limit reached; exiting periodic sync without cleanup"
      exit 1
    fi
    sleep "$SYNC_SECONDS"
    continue
  fi

  ssh_misses=0
  log "syncing results with ssh+tar"
  tmp_results_dir="$(mktemp -d "${LOCAL_RESULTS_ROOT%/}/.${RUN_ID}.periodic.XXXXXX")"
  set +e
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "tar --warning=no-file-changed --ignore-failed-read --exclude='*/best.pt' -C '$REMOTE_RESULTS_DIR' -cf - .; rc=\$?; if [ \"\$rc\" -le 1 ]; then exit 0; else exit \"\$rc\"; fi" | tar -C "$tmp_results_dir" -xf -
  pipe_status=("${PIPESTATUS[@]}")
  tar_rc=${pipe_status[0]}
  extract_rc=${pipe_status[1]}
  set -e
  if (( tar_rc != 0 || extract_rc != 0 )); then
    rm -rf "$tmp_results_dir"
    log "result sync failed (remote_tar=${tar_rc}, local_tar=${extract_rc}); continuing"
    sleep "$SYNC_SECONDS"
    continue
  fi
  rm -rf "$LOCAL_RESULTS_DIR"
  mv "$tmp_results_dir" "$LOCAL_RESULTS_DIR"

  log "syncing run log with ssh+cat"
  set +e
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cat '$REMOTE_LOG_FILE'" > "${LOCAL_LOG_FILE}.tmp"
  log_rc=$?
  set -e
  if (( log_rc != 0 )); then
    rm -f "${LOCAL_LOG_FILE}.tmp"
    log "run-log sync failed with rc=${log_rc}; continuing"
  else
    mv "${LOCAL_LOG_FILE}.tmp" "$LOCAL_LOG_FILE"
  fi

  {
    echo "run_id=${RUN_ID}"
    echo "synced_at=$(date -Is)"
    echo "local_results=${LOCAL_RESULTS_DIR}"
    echo "local_log=${LOCAL_LOG_FILE}"
    echo "file_count=$(find "$LOCAL_RESULTS_DIR" -type f | wc -l)"
  } > "$LOCAL_PERIODIC_MARKER"

  log "periodic sync complete; sleeping ${SYNC_SECONDS}s"
  sleep "$SYNC_SECONDS"
done
