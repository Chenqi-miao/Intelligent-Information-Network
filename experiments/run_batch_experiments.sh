#!/bin/bash
# run_batch_experiments.sh
# 批量跑多种超参数组合
# 每组实验完成后自动记录到 experiments/results/

set -e
cd "$(dirname "$0")/.."
PYTHONPATH=.

echo "=========================================="
echo "Batch experiments started: $(date)"
echo "=========================================="

# ── 固定基础配置 ──
BASE="--train-ratio 0.35 --val-ratio 0.05"
GROUP="--groups institutions"

TIME_START=$(date +%s)

# ========== 窗口对比实验 ==========

# LSTM (168, 24)
echo ""
echo "[1/10] LSTM (168, 24)"
uv run python experiments/run_dl.py --model LSTM --tw 168 --pw 24 $BASE $GROUP

# LSTM (168, 168)
echo ""
echo "[2/10] LSTM (168, 168)"
uv run python experiments/run_dl.py --model LSTM --tw 168 --pw 168 $BASE $GROUP

# LSTM (744, 168)
echo ""
echo "[3/10] LSTM (744, 168)"
uv run python experiments/run_dl.py --model LSTM --tw 744 --pw 168 $BASE $GROUP

# GRU (168, 24)
echo ""
echo "[4/10] GRU (168, 24)"
uv run python experiments/run_dl.py --model GRU --tw 168 --pw 24 $BASE $GROUP

# GRU (168, 168)
echo ""
echo "[5/10] GRU (168, 168)"
uv run python experiments/run_dl.py --model GRU --tw 168 --pw 168 $BASE $GROUP

# GRU (744, 168)
echo ""
echo "[6/10] GRU (744, 168)"
uv run python experiments/run_dl.py --model GRU --tw 744 --pw 168 $BASE $GROUP

# ========== 学习率对比实验 ==========

# LSTM lr=0.001
echo ""
echo "[7/10] LSTM (24,24) lr=0.001"
uv run python experiments/run_dl.py --model LSTM --lr 0.001 $BASE $GROUP

# GRU lr=0.001
echo ""
echo "[8/10] GRU (24,24) lr=0.001"
uv run python experiments/run_dl.py --model GRU --lr 0.001 $BASE $GROUP

# ========== 子网级数据 ==========

# LSTM (24,24) on institution_subnets
echo ""
echo "[9/10] LSTM (24,24) subnets"
uv run python experiments/run_dl.py --model LSTM --groups institution_subnets $BASE

# GRU (24,24) on institution_subnets
echo ""
echo "[10/10] GRU (24,24) subnets"
uv run python experiments/run_dl.py --model GRU --groups institution_subnets $BASE

TIME_END=$(date +%s)
DURATION=$((TIME_END - TIME_START))

echo ""
echo "=========================================="
echo "All experiments done!"
echo "Duration: $((DURATION / 60)) min $((DURATION % 60)) sec"
echo "=========================================="
