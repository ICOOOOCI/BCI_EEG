#!/bin/bash
# 依据 session_20261001T083940Z_9210ab5d 的离线分析生成，供新会话提示测试。
# 固定方案同场 42/120；包含方案选择的嵌套验证 36/120，尚未完成新会话验证。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
printf '%s\n' '前提：每通道有效输出支持90Hz频带；Wi-Fi可支持250Hz，串口Daisy125Hz限制不适用于所有连接。'
printf '%s\n' '复测方案：M3 + 4秒历史陷波；2秒分析窗；140毫秒起点；新目标顺序。'
printf '%s\n' '离线准确率仍较低，请先改善电极接触和工频干扰，再做模式2提示测试。'
exec "$SCRIPT_DIR/start_keyboard_dual_mode.command" \
    --fbcca-profile m3 --notch-mode history --protocol standard_2s \
    --response-delay 0.14 --seed 20261002 "$@"
