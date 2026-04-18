#!/usr/bin/env bash
# SKD chunk_size sweep: verify_top_k=3 고정, chunk_size를 변화시키며
# 총 시간, accept rate, Teacher 시간 증가율을 측정한다.
#
# 사전 조건: launch_test_servers.sh로 서버가 이미 떠 있어야 함.
#
# Usage:
#   bash tests/skd/manual/sweep_chunk_size.sh

set -euo pipefail

cd /home/work/DDAI_revised/OSworld/verl

PYTHON="/home/work/DDAI_revised/miniconda3/envs/kd/bin/python"
TEST_SCRIPT="tests/skd/manual/skd_integration_manual.py"

VERIFY_TOP_K=3
MAX_RESPONSE=4096

CHUNK_SIZES=(64 128 256 512 1024 2048)

echo "============================================================"
echo "SKD Chunk Size Sweep"
echo "  verify_top_k=${VERIFY_TOP_K}, max_response=${MAX_RESPONSE}"
echo "  chunk_sizes: ${CHUNK_SIZES[*]}"
echo "============================================================"
echo ""

for CHUNK in "${CHUNK_SIZES[@]}"; do
    echo "------------------------------------------------------------"
    echo "  chunk_size=${CHUNK}"
    echo "------------------------------------------------------------"
    ${PYTHON} ${TEST_SCRIPT} \
        --chunk-size ${CHUNK} \
        --verify-top-k ${VERIFY_TOP_K} \
        --max-response ${MAX_RESPONSE} \
        2>/dev/null
    echo ""
done

echo "============================================================"
echo "Sweep 완료!"
echo "============================================================"
