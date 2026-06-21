"""
run_dl.py — 批量训练 LSTM/GRU 模型

顺序处理每条时间序列：预处理 → 训练 → 评估 → 记录结果。
支持 GPU 加速（--device cuda）。

用法：
    uv run python experiments/run_dl.py                          # 默认 LSTM
    uv run python experiments/run_dl.py --model GRU              # 切换 GRU
    uv run python experiments/run_dl.py --max-files 10           # 试跑
    uv run python experiments/run_dl.py --device cuda            # GPU
    uv run python experiments/run_dl.py --epochs 50 --lr 0.001   # 调参
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, DATA_DIR
from src.preprocessing import preprocess_pipeline, load_and_align
from src.models import create_model
from src.train import train_model, make_loader, set_seed
from src.evaluate import compute_metrics, predict_model, record_results, EXPERIMENTS_DIR

LOG_DIR = EXPERIMENTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "run_dl.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def scan_files(group: str, aggregation: str = "agg_1_hour") -> list[int]:
    data_dir = DATA_DIR / group / aggregation
    return sorted(
        int(f.stem) for f in data_dir.glob("*.csv")
        if f.name not in ("times.csv", "identifiers.csv")
    )


def run_single_series(
    file_id: int,
    group: str,
    cfg: Config,
    output_csv: Path,
    device: str = "cpu",
    cut_date: str | None = None,
):
    # ── 预处理 ──
    try:
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
            max_date=cut_date,
        )
    except Exception as e:
        logger.warning("  skip file=%d (preprocess failed: %s)", file_id, e)
        return

    X_train, X_val, X_test = result["X_train"], result["X_val"], result["X_test"]
    y_train, y_val, y_test = result["y_train"], result["y_val"], result["y_test"]
    scaler = result["scaler"]
    n_samples = len(X_test)

    # ── 创建 DataLoader ──
    train_loader = make_loader(X_train, y_train, batch_size=cfg.batch_size)
    val_loader = make_loader(X_val, y_val, batch_size=cfg.batch_size, shuffle=False)

    # ── 创建模型 ──
    set_seed(cfg.seed)
    model = create_model(
        cfg.model_name,
        prediction_window=cfg.prediction_window,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.n_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    )

    # ── 训练 ──
    try:
        model, train_time, history = train_model(
            model, train_loader, val_loader,
            epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            patience=cfg.patience,
            device=device,
            seed=cfg.seed,
        )
    except Exception as e:
        logger.warning("  skip file=%d (train failed: %s)", file_id, e)
        return

    # ── 评估 ──
    y_pred = predict_model(model, X_test, scaler, device=device)
    y_test_orig = scaler.inverse_transform(y_test)

    metrics = compute_metrics(y_test_orig, y_pred)

    record_results(
        output_path=output_csv,
        ts_id=file_id,
        ts_group=group,
        training_window=cfg.training_window,
        prediction_window=cfg.prediction_window,
        model_name=cfg.model_name,
        ts_metric=cfg.ts_attributes[0],
        metrics=metrics,
        training_time=train_time,
        n_samples=n_samples,
    )


def main():
    parser = argparse.ArgumentParser(description="批量训练 LSTM/GRU")
    parser.add_argument("--model", default="LSTM", choices=["LSTM", "GRU"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--groups", nargs="+", default=["institutions"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tw", type=int, default=24)
    parser.add_argument("--pw", type=int, default=24)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cut-date", type=str, default=None,
                        help="只保留此日期前数据，如 2024-06-01")
    args = parser.parse_args()

    cfg = Config(
        model_name=args.model,
        training_window=args.tw,
        prediction_window=args.pw,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        batch_size=args.batch_size,
        bidirectional=args.bidirectional,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        patience=args.patience,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = EXPERIMENTS_DIR / "results" / f"dl_results_{args.model}_{timestamp}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info("=" * 60)
    logger.info("DL batch run: %s", args.model)
    logger.info("  groups: %s", args.groups)
    logger.info("  window: (%d, %d)", cfg.training_window, cfg.prediction_window)
    logger.info("  device: %s", device)
    logger.info("  model params: hidden=%d layers=%d lr=%.4f epochs=%d batch=%d",
                args.hidden_size, args.layers, args.lr, args.epochs, args.batch_size)
    logger.info("  output: %s", output_csv)
    logger.info("=" * 60)

    total = 0
    for group in args.groups:
        file_ids = scan_files(group, cfg.aggregation)
        if args.max_files:
            file_ids = file_ids[:args.max_files]
            logger.info("  test mode: limit %d files", args.max_files)

        logger.info("Processing %s (%d files)", group, len(file_ids))
        for idx, file_id in enumerate(file_ids, 1):
            logger.info("[%s %3d/%d] file=%d", group, idx, len(file_ids), file_id)
            run_single_series(file_id, group, cfg, output_csv, device=device, cut_date=args.cut_date)
            total += 1

        logger.info("Done: %s", group)

    logger.info("All done! Processed %d series", total)


if __name__ == "__main__":
    main()
