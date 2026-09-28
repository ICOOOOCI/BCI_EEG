#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
/bin/bash "$SCRIPT_DIR/start_keyboard_dual_mode.sh" "$@"
EXIT_CODE=$?
printf 'Exit code: %s\n' "$EXIT_CODE"
if [[ -t 0 && "${FBCCA_NO_PAUSE:-0}" != 1 ]]; then
    read -r -p 'Press Enter to close...'
fi
exit "$EXIT_CODE"
