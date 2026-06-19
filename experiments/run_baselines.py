"""
run_baselines.py — 批量运行基线模型（Mean + SARIMA）

在指定层级的所有时间序列上，逐一运行预处理 → 预测 → 评估 → 记录。
输出结果 CSV 到 experiments/results/ 目录。

用法：
    uv run python experiments/run_baselines.py                     # 全量运行
    uv run python experiments/run_baselines.py --max-files 10      # 试跑 10 条
    uv run python experiments/run_baselines.py --groups institutions  # 只跑机构级

日志输出到 experiments/logs/。
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, WINDOW_PRESETS, DATA_DIR
from src.preprocessing import preprocess_pipeline, load_and_align
from src.models import create_model
from src.evaluate import compute_metrics, predict_model, record_results, EXPERIMENTS_DIR

# ── 日志 ──
LOG_DIR = EXPERIMENTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "run_baselines.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════ 辅助函数 ═══════════════════════

def scan_files(group: str, aggregation: str = "agg_1_hour") -> list[int]:
    """
    扫描指定层级下的所有 CSV 文件，按文件 ID 排序返回。

    排除 times.csv / identifiers.csv 等非数据文件。
    """
    data_dir = DATA_DIR / group / aggregation
    csv_files = sorted(
        int(f.stem) for f in data_dir.glob("*.csv")
        if f.name not in ("times.csv", "identifiers.csv")
    )
    logger.info("Scan %s/%s: %d files", group, aggregation, len(csv_files))
    return csv_files


def count_missing_original(file_id: int, group: str, aggregation: str) -> int:
    """统计原始数据的缺失点数（right merge 前）"""
    df = load_and_align(file_id, "n_bytes", group, aggregation)
    return int(df["n_bytes"].isna().sum())


# ═══════════════════════ 主流程 ═══════════════════════

def run_single_series(
    file_id: int,
    group: str,
    cfg: Config,
    output_csv: Path,
    run_sarima: bool = False,
):
    """
    单条时间序列：预处理 → Mean 基线 → 记录结果。

    如果 run_sarima=True，额外跑 SARIMA。
    """
    # ── 预处理 ──
    try:
        n_missing = count_missing_original(file_id, group, cfg.aggregation)
        result = preprocess_pipeline(
            file_id=file_id,
            ts_attribute=cfg.ts_attributes[0],
            group=group,
            aggregation=cfg.aggregation,
            impute_method=cfg.impute_method,
            training_window=cfg.training_window,
            prediction_window=cfg.prediction_window,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
        )
    except Exception as e:
        logger.warning("  skip file=%d (preprocess failed: %s)", file_id, e)
        return

    X_test = result["X_test"]
    y_test = result["y_test"]
    scaler = result["scaler"]
    n_samples = len(X_test)

    # 反标准化用于评估
    y_test_orig = scaler.inverse_transform(y_test)
    ts_attribute = cfg.ts_attributes[0]

    # ── Mean 基线 ──
    t0 = time.time()
    mean_model = create_model("Mean", prediction_window=cfg.prediction_window)
    mean_pred = predict_model(mean_model, X_test, scaler)
    mean_time = time.time() - t0
    mean_metrics = compute_metrics(y_test_orig, mean_pred)

    record_results(
        output_path=output_csv,
        ts_id=file_id,
        ts_group=group,
        training_window=cfg.training_window,
        prediction_window=cfg.prediction_window,
        model_name="Mean",
        ts_metric=ts_attribute,
        metrics=mean_metrics,
        prediction_time=mean_time,
        n_samples=n_samples,
        n_missing=n_missing,
    )

    # ── SARIMA 基线（可选）──
    if run_sarima:
        try:
            y_train = scaler.inverse_transform(result["y_train"])
            y_val = scaler.inverse_transform(result["y_val"])
            y_train_full = np.concatenate([y_train.flatten(), y_val.flatten()])

            t0 = time.time()
            sarima = create_model(
                "SARIMA",
                prediction_window=cfg.prediction_window,
                order=cfg.sarima_order,
                seasonal_order=cfg.sarima_seasonal_order,
            )
            sarima.fit(None, y_train_full.reshape(-1, 1))
            sarima_pred = predict_model(sarima, X_test, scaler)
            sarima_time = time.time() - t0

            sarima_metrics = compute_metrics(y_test_orig, sarima_pred)
            record_results(
                output_path=output_csv,
                ts_id=file_id,
                ts_group=group,
                training_window=cfg.training_window,
                prediction_window=cfg.prediction_window,
                model_name="SARIMA",
                ts_metric=ts_attribute,
                metrics=sarima_metrics,
                training_time=sarima_time,
                n_samples=n_samples,
                n_missing=n_missing,
            )
        except Exception as e:
            logger.warning("  SARIMA file=%d failed: %s", file_id, e)


def summarize_results(csv_path: Path):
    """打印结果汇总：各模型在各层级的平均指标"""
    if not csv_path.exists():
        logger.warning("Results file not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    print("\n" + "=" * 70)
    print("  Summary (by TS_GROUP x MODEL)")
    print("=" * 70)

    summary = df.groupby(["TS_GROUP", "MODEL"])[["RMSE", "SMAPE", "R2_SCORE"]].mean().round(4)
    print(summary.to_string())
    print("=" * 70)
    print(f"  {len(df)} records | {df['TS_ID'].nunique()} series | {df['MODEL'].nunique()} models\n")


# ═══════════════════════ CLI ═══════════════════════

def main():
    parser = argparse.ArgumentParser(description="批量运行基线模型")
    parser.add_argument("--max-files", type=int, default=None, help="每个层级最多处理文件数（试跑用）")
    parser.add_argument(
        "--groups", nargs="+",
        default=["institutions", "institution_subnets", "ip_addresses_sample"],
        help="要跑的层级（默认全跑）",
    )
    parser.add_argument("--sarima", action="store_true", help="同时运行 SARIMA（默认只跑 Mean）")
    parser.add_argument("--tw", type=int, default=24, help="训练窗口大小（默认 24）")
    parser.add_argument("--pw", type=int, default=24, help="预测窗口大小（默认 24）")
    args = parser.parse_args()

    # ── 配置 ──
    cfg = Config(
        training_window=args.tw,
        prediction_window=args.pw,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = EXPERIMENTS_DIR / "results" / f"baseline_results_{timestamp}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Baseline batch run")
    logger.info("  groups: %s", args.groups)
    logger.info("  window: (%d, %d)", cfg.training_window, cfg.prediction_window)
    logger.info("  models: Mean" + (" + SARIMA" if args.sarima else ""))
    logger.info("  output: %s", output_csv)
    logger.info("=" * 60)

    total_series = 0
    for group in args.groups:
        file_ids = scan_files(group, cfg.aggregation)
        if args.max_files:
            file_ids = file_ids[:args.max_files]
            logger.info("  test mode: limit %d files", args.max_files)

        logger.info("Processing %s (%d files)", group, len(file_ids))
        for idx, file_id in enumerate(file_ids, 1):
            logger.info(
                "[%s %3d/%d] file=%d", group, idx, len(file_ids), file_id
            )
            run_single_series(
                file_id, group, cfg, output_csv, run_sarima=args.sarima,
            )
            total_series += 1

        logger.info("Done: %s", group)

    # ── 汇总 ──
    logger.info("All done! Processed %d series total", total_series)
    summarize_results(output_csv)


if __name__ == "__main__":
    main()
