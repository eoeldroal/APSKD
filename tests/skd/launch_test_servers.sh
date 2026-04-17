#!/usr/bin/env bash
# SKD 통합 테스트용 vLLM 서버 2개를 띄우는 스크립트.
# 서버가 뜬 후 test_skd_integration.py를 실행하면 됨.
#
# Usage:
#   bash tests/skd/launch_test_servers.sh
#   # 서버 Ready 후 다른 터미널에서:
#   python tests/skd/test_skd_integration.py

set -euo pipefail

STUDENT_MODEL="/home/work/DDAI_revised/OSworld/verl/checkpoints/Qwen3-1.7B"
TEACHER_MODEL="/home/work/DDAI_revised/OSworld/verl/checkpoints/Qwen3-8B"

STUDENT_PORT=8001
TEACHER_PORT=8002

echo "[SKD Test] Starting Student vLLM (Qwen3-1.7B) on GPU 0, port ${STUDENT_PORT}..."
CUDA_VISIBLE_DEVICES=0 vllm serve "${STUDENT_MODEL}" \
    --port "${STUDENT_PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --enable-prefix-caching \
    --max-model-len 9000 \
    --disable-log-requests \
    &
STUDENT_PID=$!

echo "[SKD Test] Starting Teacher vLLM (Qwen3-8B) on GPU 1, port ${TEACHER_PORT}..."
CUDA_VISIBLE_DEVICES=1 vllm serve "${TEACHER_MODEL}" \
    --port "${TEACHER_PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --enable-prefix-caching \
    --max-model-len 9000 \
    --disable-log-requests \
    &
TEACHER_PID=$!

echo "[SKD Test] Waiting for servers to be ready..."

wait_for_server() {
    local url=$1
    local name=$2
    for i in $(seq 1 120); do
        if curl -s "${url}/health" > /dev/null 2>&1; then
            echo "[SKD Test] ${name} ready!"
            return 0
        fi
        sleep 2
    done
    echo "[SKD Test] ERROR: ${name} failed to start within 240s"
    return 1
}

wait_for_server "http://127.0.0.1:${STUDENT_PORT}" "Student"
wait_for_server "http://127.0.0.1:${TEACHER_PORT}" "Teacher"

echo ""
echo "============================================"
echo "Both servers ready!"
echo "  Student: http://127.0.0.1:${STUDENT_PORT}"
echo "  Teacher: http://127.0.0.1:${TEACHER_PORT}"
echo ""
echo "Run test:"
echo "  python tests/skd/test_skd_integration.py"
echo ""
echo "Press Ctrl+C to stop both servers."
echo "============================================"

trap "kill ${STUDENT_PID} ${TEACHER_PID} 2>/dev/null; echo 'Servers stopped.'" EXIT
wait
