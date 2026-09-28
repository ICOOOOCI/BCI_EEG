#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if ! command -v python3.10 >/dev/null 2>&1; then
    printf '%s\n' '[ENVIRONMENT] Install 64-bit CPython 3.10.11 first; see README.' >&2
    exit 10
fi
python3.10 "$SCRIPT_DIR/setup_environment.py"
EXIT_CODE=$?
printf 'Setup exit code: %s\n' "$EXIT_CODE"
if [[ -t 0 && "${FBCCA_NO_PAUSE:-0}" != 1 ]]; then
    read -r -p 'Press Enter to close...'
fi
exit "$EXIT_CODE"
