#!/bin/bash
# 离线：./start_keyboard_dual_mode.command --offline-session 会话.zip --offline-output 输出目录
# 历史陷波复测：./start_keyboard_optimized.command
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR" || exit 10
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/fbcca_keyboard_dual_mode.py" "$@"
elif command -v python3.10 >/dev/null 2>&1; then
    python3.10 "$SCRIPT_DIR/fbcca_keyboard_dual_mode.py" "$@"
else
    python3 "$SCRIPT_DIR/fbcca_keyboard_dual_mode.py" "$@"
fi
EXIT_CODE=$?
printf 'Exit code: %s\n' "$EXIT_CODE"
if [[ -t 0 && "${FBCCA_NO_PAUSE:-0}" != 1 ]]; then
    read -r -p 'Press Enter to close...'
fi
exit "$EXIT_CODE"
