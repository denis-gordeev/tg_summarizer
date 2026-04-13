#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
CONTROLLER_ROOT="${AUTOWORK_CONTROLLER_ROOT:-/Users/denis/programming/autowork/repo-autowork}"
PYTHON_BIN="${AUTOWORK_PYTHON_BIN:-python3}"

cd "$CONTROLLER_ROOT"
PYTHONPATH="$CONTROLLER_ROOT/src" "$PYTHON_BIN" -m repo_autowork.cli project-run --repo "$REPO_DIR" "$@"
