#!/bin/bash
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR" || exit 10
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export FBCCA_QUICK_ENTRY="${FBCCA_QUICK_ENTRY:-1}"
PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_EXE" ]]; then
    printf '%s\n' '[ENVIRONMENT] Project .venv is missing. Run setup_macos.command explicitly first.' >&2
    exit 10
fi
exec "$PYTHON_EXE" "$SCRIPT_DIR/run_keyboard.py" "$@"
