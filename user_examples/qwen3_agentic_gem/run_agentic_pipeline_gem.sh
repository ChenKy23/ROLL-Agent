#!/bin/bash
set -x

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# 设置PYTHONPATH以便Python能找到roll模块
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# # 设置Ray日志环境变量
# # export RAY_LOG_TO_DRIVER=1
# # export RAY_DISABLE_IMPORT_WARNING=1

# 检查并清理残留Ray进程和临时文件
echo "Cleaning up existing Ray processes and temporary files..."
ray stop || echo "No existing Ray processes to stop."
rm -rf /tmp/ray || echo "No existing Ray temporary files to remove."

# 检查并等待Ray服务
echo "Checking Ray status..."
if ! ray status &>/dev/null; then
    echo "Starting Ray cluster..."
    ray start --head --port=6379 --dashboard-host=0.0.0.0 --temp-dir=/tmp/ray --disable-usage-stats
    sleep 2
fi

# 验证Ray集群状态
MAX_RETRIES=10
RETRY_COUNT=0
while ! ray status &>/dev/null; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to start Ray cluster after $MAX_RETRIES retries. Exiting..."
        exit 1
    fi
    echo "Waiting for Ray cluster to start... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

echo "Ray cluster started successfully."

CONFIG_PATH=$(basename $(dirname $0))
python user_examples/start_agentic_pipeline.py --config_path $CONFIG_PATH --config_name mix_tool_rl