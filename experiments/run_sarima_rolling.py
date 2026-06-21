"""
run_sarima_rolling.py — SARIMA 滚动预测

每条序列：在测试集的前 N 个窗口上做滚动预测。
训练集数据不断累积，每次重新拟合。

用法：
    uv run python experiments/run_sarima_rolling.py --max-files 5
    uv run python experiments/run_sarima_rolling.py --n-windows 10 --groups institutions
"""

import argparse, logging, sys, time, warnings
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import Config, DATA_DIR
from src.preprocessing import load_and_align, impute_missing, create_sliding_windows, temporal_split
from src.evaluate import compute_metrics, record_results, EXPERIMENTS_DIR
from src.models import MeanBaseline

LOG_DIR = EXPERIMENTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "run_sarima.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

SARIMA_ORDER = (1, 0, 1)
SARIMA_SEASONAL = (1, 0, 1, 24)

def scan_files(group, aggregation="agg_1_hour"):
    data_dir = DATA_DIR / group / aggregation
    return sorted(int(f.stem) for f in data_dir.glob("*.csv")
                  if f.name not in ("times.csv", "identifiers.csv"))

def run_single(file_id, group, n_windows, output_csv):
    # ── 加载 + 填充 ──
    try:
        df = load_and_align(file_id, "n_bytes", group)
        df = impute_missing(df, "zeros", "n_bytes")
        values = df["n_bytes"].values
    except Exception as e:
        logger.warning("  skip file=%d (load: %s)", file_id, e)
        return

    # ── 滑窗 + 分割 ──
    X, y = create_sliding_windows(df, 24, 24)
    n_total = len(X)
    train_end = int(n_total * 0.35)
    val_end = train_end + int(n_total * 0.05)
    y_test = y[val_end:]
    n_test = min(n_windows, len(y_test))

    if n_test < 1:
        return

    # ── Mean 基线 ──
    test_start_idx = val_end * 24 + 24
    train_vals = values[:test_start_idx]
    mean_pred = np.tile(train_vals.mean(), (n_test, 24))
    mean_metrics = compute_metrics(y_test[:n_test].flatten(), mean_pred.flatten())

    # ── 记录 Mean ──
    record_results(output_csv, file_id, group, 24, 24, "Mean", "n_bytes", mean_metrics, n_samples=n_test)

    # ── SARIMA 滚动 ──
    sarima_preds = []
    t0 = time.time()
    coefs = None

    for i in range(n_test):
        forecast_start = test_start_idx + i * 24
        history = values[:forecast_start]

        try:
            model = SARIMAX(history, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL,
                            enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(maxiter=100, disp=False)
            forecast = fitted.forecast(steps=24)
            sarima_preds.append(forecast)

            # 保存最后一次拟合的系数
            if i == n_test - 1:
                coefs = fitted.params.to_dict()
        except:
            sarima_preds.append(np.full(24, np.nan))

    elapsed = time.time() - t0
    y_test_sub = y_test[:n_test]
    sarima_preds = np.array(sarima_preds)
    if len(sarima_preds) != len(y_test_sub):
        sarima_preds = sarima_preds[:len(y_test_sub)]

    mask = ~np.isnan(sarima_preds).any(axis=1)
    if mask.sum() > 0:
        sarima_metrics = compute_metrics(y_test_sub[mask].flatten(), sarima_preds[mask].flatten())
        record_results(output_csv, file_id, group, 24, 24, "SARIMA", "n_bytes", sarima_metrics,
                       training_time=elapsed, n_samples=int(mask.sum()))

    # ── 打印系数（前 5 条详细看）──
    if file_id <= 5 and coefs:
        ar_coefs = {k: v for k, v in coefs.items() if 'ar' in k.lower() and 'ma' not in k.lower()}
        ma_coefs = {k: v for k, v in coefs.items() if 'ma' in k.lower() and 'ar' not in k.lower()}
        sar_coefs = {k: v for k, v in coefs.items() if 'ar' in k.lower() and 'seasonal' in k.lower()}
        logger.info("  SARIMA coefs | file=%d | AR=%s | MA=%s | SeasonalAR=%s | Time=%.1fs",
                     file_id, ar_coefs, ma_coefs, sar_coefs, elapsed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=["institutions"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--n-windows", type=int, default=10, help="每条序列跑多少个测试窗口")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = EXPERIMENTS_DIR / "results" / f"sarima_rolling_{timestamp}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("SARIMA rolling: %d windows per series", args.n_windows)
    logger.info("  groups: %s", args.groups)
    logger.info("  output: %s", output_csv)

    total = 0
    for group in args.groups:
        file_ids = scan_files(group)
        if args.max_files:
            file_ids = file_ids[:args.max_files]

        logger.info("Processing %s (%d files)", group, len(file_ids))
        t0 = time.time()
        for idx, file_id in enumerate(file_ids, 1):
            logger.info("[%s %3d/%d] file=%d", group, idx, len(file_ids), file_id)
            run_single(file_id, group, args.n_windows, output_csv)
            total += 1
            if idx % 50 == 0:
                logger.info("  %d/%d done, %.1fs elapsed", idx, len(file_ids), time.time()-t0)

        logger.info("Done: %s (%.1fs)", group, time.time()-t0)

    logger.info("All done! %d series", total)


if __name__ == "__main__":
    main()
