#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  INSTANCE_ID=35574169 \
  RUN_ID=vast_2stage_T5_b38a1_diskfix_20260425_140015 \
  SSH_HOST=210.157.233.86 \
  SSH_PORT=54845 \
  SESSION_NAME=jager_2stage_vast_2stage_T5_b38a1_diskfix_20260425_140015 \
  scripts/watch_vast_run_sync_destroy.sh

Environment:
  INSTANCE_ID          Vast instance id to destroy after successful sync.
  RUN_ID               Run id under remote results root.
  SSH_HOST             SSH hostname or IP for the Vast instance.
  SSH_PORT             SSH port for the Vast instance.
  SESSION_NAME         Remote tmux session to wait for.

Optional:
  SSH_USER             Default: root
  SSH_KEY              Default: ~/.ssh/id_ed25519
  REMOTE_RESULTS_ROOT  Default: /workspace/results/k6_promptcv
  REMOTE_LOG_ROOT      Default: /workspace/run_logs
  LOCAL_RESULTS_ROOT   Default: ../results/k6_promptcv
  LOCAL_LOG_ROOT       Default: ../results/vast_run_logs
  POLL_SECONDS         Default: 120
  SSH_MISS_LIMIT       Default: 10
  VASTAI_BIN           Default: vastai
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

require_env INSTANCE_ID
require_env RUN_ID
require_env SSH_HOST
require_env SSH_PORT
require_env SESSION_NAME

SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/workspace/results/k6_promptcv}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/workspace/run_logs}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-../results/k6_promptcv}"
LOCAL_LOG_ROOT="${LOCAL_LOG_ROOT:-../results/vast_run_logs}"
POLL_SECONDS="${POLL_SECONDS:-120}"
SSH_MISS_LIMIT="${SSH_MISS_LIMIT:-10}"
VASTAI_BIN="${VASTAI_BIN:-vastai}"

REMOTE_RESULTS_DIR="${REMOTE_RESULTS_ROOT%/}/${RUN_ID}"
REMOTE_LOG_FILE="${REMOTE_LOG_ROOT%/}/${RUN_ID}.log"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_ROOT%/}/${RUN_ID}"
LOCAL_LOG_FILE="${LOCAL_LOG_ROOT%/}/${RUN_ID}.log"
LOCAL_MARKER="${LOCAL_RESULTS_DIR}/SYNC_COMPLETE.txt"

SSH_OPTS=(
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -p "$SSH_PORT"
)
SSH_TARGET="${SSH_USER}@${SSH_HOST}"

mkdir -p "$LOCAL_RESULTS_ROOT" "$LOCAL_LOG_ROOT"

log "watching instance=${INSTANCE_ID} run=${RUN_ID} session=${SESSION_NAME}"
log "remote results: ${SSH_TARGET}:${REMOTE_RESULTS_DIR}"
log "local results: ${LOCAL_RESULTS_DIR}"

ssh_misses=0
while true; do
  set +e
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "tmux has-session -t '$SESSION_NAME' >/dev/null 2>&1"
  session_rc=$?
  set -e

  if (( session_rc == 0 )); then
    ssh_misses=0
    log "remote tmux session is still running; sleeping ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
    continue
  fi

  set +e
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "true" >/dev/null 2>&1
  ssh_rc=$?
  set -e

  if (( ssh_rc == 0 )); then
    log "remote tmux session is gone; starting final sync"
    break
  fi

  ssh_misses=$((ssh_misses + 1))
  log "ssh unavailable while checking session (session_rc=${session_rc}, ssh_rc=${ssh_rc}); miss ${ssh_misses}/${SSH_MISS_LIMIT}"
  if (( ssh_misses >= SSH_MISS_LIMIT )); then
    log "ssh stayed unavailable; refusing to destroy instance"
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

log "remote summary before sync"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
  "date; du -sh '$REMOTE_RESULTS_DIR' '$REMOTE_LOG_FILE' 2>/dev/null || true; find '$REMOTE_RESULTS_DIR' -maxdepth 4 -type f | wc -l" || true

log "syncing results with ssh+tar"
tmp_results_dir="$(mktemp -d "${LOCAL_RESULTS_ROOT%/}/.${RUN_ID}.final.XXXXXX")"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "tar --warning=no-file-changed --ignore-failed-read -C '$REMOTE_RESULTS_DIR' -cf - .; rc=\$?; if [ \"\$rc\" -le 1 ]; then exit 0; else exit \"\$rc\"; fi" | tar -C "$tmp_results_dir" -xf -
rm -rf "$LOCAL_RESULTS_DIR"
mv "$tmp_results_dir" "$LOCAL_RESULTS_DIR"

log "syncing run log with ssh+cat"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cat '$REMOTE_LOG_FILE'" > "${LOCAL_LOG_FILE}.tmp"
mv "${LOCAL_LOG_FILE}.tmp" "$LOCAL_LOG_FILE"

file_count="$(find "$LOCAL_RESULTS_DIR" -type f | wc -l)"
if [[ "$file_count" -lt 1 ]]; then
  log "local result dir is empty after rsync; refusing to destroy instance"
  exit 1
fi

{
  echo "run_id=${RUN_ID}"
  echo "instance_id=${INSTANCE_ID}"
  echo "synced_at=$(date -Is)"
  echo "local_results=${LOCAL_RESULTS_DIR}"
  echo "local_log=${LOCAL_LOG_FILE}"
  echo "file_count=${file_count}"
} > "$LOCAL_MARKER"

log "sync verified with ${file_count} files; destroying Vast instance ${INSTANCE_ID}"
"$VASTAI_BIN" destroy instance "$INSTANCE_ID" -y

log "done"
