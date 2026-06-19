"""
evaluate.py — 评估指标与预测入口

指标（对齐论文 Section V Evaluation）：
  - RMSE：均方根误差（√(1/N ∑(y-ŷ)²)）
  - R²：决定系数（1 - SS_res / SS_tot）
  - sMAPE：对称平均绝对百分比误差

用法：
    from src.evaluate import evaluate_model, compute_metrics
    metrics = compute_metrics(y_true, y_pred)
"""

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


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
    mask = denominator > 1e-10  # 避免除零
    return float(np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100)


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
    logger.info("指标：RMSE=%.4f  R²=%.4f  sMAPE=%.2f%%", metrics["rmse"], metrics["r2"], metrics["smape"])
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
