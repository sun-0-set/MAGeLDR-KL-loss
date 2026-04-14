#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/sun-0-set/MAGeLDR-KL-loss.git}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_ROOT/MAGeLDR-KL-loss}"
BRANCH="${BRANCH:-main}"

mkdir -p \
  "$WORKSPACE_ROOT/data" \
  "$WORKSPACE_ROOT/models" \
  "$WORKSPACE_ROOT/splits" \
  "$WORKSPACE_ROOT/cache" \
  "$WORKSPACE_ROOT/results"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git remote set-url origin "$REPO_URL"
git fetch origin --prune

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git checkout "$BRANCH"
elif git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git checkout -b "$BRANCH" --track "origin/$BRANCH"
else
  git checkout -b "$BRANCH" origin/main
fi

echo "[info] repo:   $REPO_DIR"
echo "[info] branch: $(git branch --show-current)"
echo "[info] root:   $WORKSPACE_ROOT"
echo "[info] next:   cd $REPO_DIR"
