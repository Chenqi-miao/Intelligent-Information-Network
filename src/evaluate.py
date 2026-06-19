"""
evaluate.py — 评估指标与预测入口

指标（对齐论文 Section V Evaluation）：
  - RMSE：均方根误差（√(1/N ∑(y-ŷ)²)）
  - R²：决定系数（1 - SS_res / SS_tot）
  - sMAPE：对称平均绝对百分比误差

用法：
    from src.evaluate import compute_metrics, predict_model, record_results
    metrics = compute_metrics(y_true, y_pred)
    record_results(metrics, ...)  # 追加写入 CSV
"""

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# CSV 结果列名（对齐参考代码格式）
RESULT_COLUMNS = [
    "TS_ID", "TS_GROUP", "TRAINING_WINDOW", "PREDICTION_WINDOW",
    "MODEL", "TS_METRIC",
    "RMSE", "SMAPE", "R2_SCORE",
    "TRAINING_TIME", "PREDICTION_TIME",
    "N_SAMPLES", "N_MISSING",
]

# 输出路径
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """决定系数 R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """对称平均绝对百分比误差（%）"""
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 1e-10
    # y_true=y_pred=0 时 sMAPE=0，跳过避免除零
    smape_values = np.divide(
        2 * np.abs(y_true - y_pred), denominator,
        where=mask, out=np.zeros_like(denominator, dtype=float),
    )
    return float(np.mean(smape_values) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    计算全部评估指标。

    Parameters
    ----------
    y_true : np.ndarray
        真实值（已展平或形状一致）
    y_pred : np.ndarray
        预测值

    Returns
    -------
    dict : {"rmse": float, "r2": float, "smape": float}
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    metrics = {
        "rmse": compute_rmse(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
        "smape": compute_smape(y_true, y_pred),
    }
    logger.info("Metrics: RMSE=%.4f  R2=%.4f  sMAPE=%.2f%%", metrics["rmse"], metrics["r2"], metrics["smape"])
    return metrics


def predict_model(
    model,
    X_test: np.ndarray,
    scaler=None,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """
    统一预测接口：处理 PyTorch 模型和基线模型。

    Parameters
    ----------
    model : nn.Module 或 BaseModel
    X_test : np.ndarray, shape (N, tw, 1)
    scaler : ZScoreScaler, optional
        若有 scaler，将预测结果反标准化回原始尺度
    device : str
    batch_size : int

    Returns
    -------
    y_pred : np.ndarray, shape (N, prediction_window)
        预测值（若提供了 scaler，为原始尺度；否则为标准化后尺度）
    """
    # 基线模型（Mean / SARIMA 等）
    if not isinstance(model, torch.nn.Module):
        y_pred = model.predict(X_test)
        if scaler:
            y_pred = scaler.inverse_transform(y_pred)
        return y_pred

    # PyTorch 模型
    model.eval()
    model = model.to(device)

    loader = DataLoader(
        torch.tensor(X_test, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )

    all_preds = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            all_preds.append(pred.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)

    if scaler:
        y_pred = scaler.inverse_transform(y_pred)

    return y_pred


# ═══════════════════════ 结果记录 ═══════════════════════

def record_results(
    output_path: str | Path,
    ts_id: int,
    ts_group: str,
    training_window: int,
    prediction_window: int,
    model_name: str,
    ts_metric: str,
    metrics: dict[str, float],
    training_time: float = 0.0,
    prediction_time: float = 0.0,
    n_samples: int = 0,
    n_missing: int = 0,
):
    """
    将一条实验结果追加到 CSV 文件。

    若 CSV 不存在，自动创建表头。
    格式对齐参考代码 `create_record()`，便于后续分析。

    Parameters
    ----------
    output_path : str | Path
        CSV 输出路径
    ts_id : int
        时间序列文件编号
    ts_group : str
        数据集层级（institutions / institution_subnets / ip_addresses_sample）
    training_window : int
    prediction_window : int
    model_name : str
        模型名称（Mean / SARIMA / LSTM / GRU 等）
    ts_metric : str
        建模的流量指标（n_bytes / n_flows 等）
    metrics : dict
        compute_metrics() 的返回值，含 rmse / r2 / smape
    training_time : float
        训练耗时（秒）
    prediction_time : float
        预测耗时（秒）
    n_samples : int
        测试集样本数（滑动窗口数）
    n_missing : int
        原始数据缺失点数
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "TS_ID": ts_id,
        "TS_GROUP": ts_group,
        "TRAINING_WINDOW": training_window,
        "PREDICTION_WINDOW": prediction_window,
        "MODEL": model_name,
        "TS_METRIC": ts_metric,
        "RMSE": metrics.get("rmse"),
        "SMAPE": metrics.get("smape"),
        "R2_SCORE": metrics.get("r2"),
        "TRAINING_TIME": training_time,
        "PREDICTION_TIME": prediction_time,
        "N_SAMPLES": n_samples,
        "N_MISSING": n_missing,
    }

    file_exists = output_path.exists()
    with open(output_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.debug("结果已记录：%s | %s | %s | RMSE=%.4f", ts_group, model_name, ts_metric, metrics.get("rmse", -1))
