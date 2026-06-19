"""
models.py — 模型定义

包含：
  - MeanBaseline：预测训练窗口均值（最简基线）
  - PersistenceBaseline：预测最后一个观测值（持久性基线）
  - SARIMAModel：statsmodels 实现的 SARIMA 基线
  - LSTM / GRU：纯 PyTorch 实现（预留接口）

用法：
    from src.models import MeanBaseline
    model = MeanBaseline(prediction_window=24)
    y_pred = model.predict(X_test)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════ 基类 ═══════════════════════

class BaseModel:
    """所有模型的统一基类（鸭子类型，无需强制继承）"""

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ═══════════════════════ 基线模型 ═══════════════════════

class MeanBaseline(BaseModel):
    """
    预测窗口内各时间步 = 训练窗口的均值。

    这是最简基线：假设未来值等于过去一段时间的平均值。
    论文预期表现：最差基线。
    """

    def __init__(self, prediction_window: int = 24):
        self.prediction_window = prediction_window

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Mean 模型不需要训练，接口保留以保持统一"""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (N, training_window, 1)

        Returns
        -------
        y_pred : np.ndarray, shape (N, prediction_window)
        """
        means = X.mean(axis=1)                         # (N, 1)
        return np.repeat(means, self.prediction_window, axis=1)   # (N, pw)


class PersistenceBaseline(BaseModel):
    """
    预测窗口内各时间步 = 训练窗口的最后一个观测值。

    即"明天和今天一样"，对强周期性数据效果意外好。
    """

    def __init__(self, prediction_window: int = 24):
        self.prediction_window = prediction_window

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        last_val = X[:, -1, :]                                # (N, 1)
        return np.repeat(last_val, self.prediction_window, axis=1)


class SARIMAModel(BaseModel):
    """
    基于 statsmodels 的 SARIMA 基线。

    注意：SARIMA 不使用滑动窗口输入，而是直接在原始序列上
    fit → forecast。所以它的接口与其他模型不同——需要原始序列值。
    此处保留统一接口，但内部忽略 X，直接用传入的原始 y。

    Parameters
    ----------
    order : tuple
        (p, d, q) 非季节性阶数
    seasonal_order : tuple
        (P, D, Q, s) 季节性阶数，s=24 表示日周期
    prediction_window : int
    """

    def __init__(
        self,
        order: tuple = (1, 0, 1),
        seasonal_order: tuple = (1, 0, 1, 24),
        prediction_window: int = 24,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.prediction_window = prediction_window
        self.model_fit_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        在训练集 y 上拟合 SARIMA。

        注意：这里 y 是预处理后的原始值（标准化后），
        需要提供原始尺度以便 SARIMA 内部处理。
        """
        y_series = y.flatten()
        logger.info(
            "SARIMA 拟合：order=%s seasonal=%s | %d 个时间点",
            self.order, self.seasonal_order, len(y_series),
        )
        model = SARIMAX(
            y_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit_ = model.fit(maxiter=200, disp=False)
        logger.info("SARIMA 拟合完成：AIC=%.2f", self.model_fit_.aic)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对每个样本，用最近 training_window 个值作为历史，
        预测未来 prediction_window 步。

        注意：这里的 X 是滑动窗口格式 (N, tw, 1)，但 SARIMA 是
        自回归模型，只需要从序列末尾往前推。简化实现：
        对整个序列一次性 forecast。

        Returns
        -------
        y_pred : np.ndarray, shape (N, prediction_window)
        """
        if self.model_fit_ is None:
            raise RuntimeError("请先调用 fit()")
        forecast = self.model_fit_.forecast(steps=self.prediction_window)
        # 复制给每个样本（SARIMA 不能逐样本预测）
        y_pred = np.tile(forecast, (len(X), 1))
        return y_pred


# ═══════════════════════ 深度学习模型（PyTorch） ═══════════════════════

class LSTM(nn.Module):
    """
    单层 LSTM + Linear 输出层。

    结构：
      input (batch, tw, 1)
        → LSTM(hidden_size, num_layers=1, batch_first=True)
        → 取最后时间步 hidden state
        → Linear(hidden_size → prediction_window)
        → Sigmoid 激活（输出范围 [0,1]，与标准化后数据匹配）
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 100,
        num_layers: int = 1,
        prediction_window: int = 24,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, prediction_window)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, tw, 1)
        lstm_out, (h_n, _) = self.lstm(x)
        # lstm_out: (batch, tw, hidden) → 取最后时间步
        last_out = lstm_out[:, -1, :]     # (batch, hidden)
        out = self.fc(last_out)            # (batch, pw)
        return self.sigmoid(out)


class GRU(nn.Module):
    """
    单层 GRU + Linear 输出层。

    结构与 LSTM 一致，只是将 LSTM 替换为 GRU。
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 100,
        num_layers: int = 1,
        prediction_window: int = 24,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, prediction_window)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, h_n = self.gru(x)
        last_out = gru_out[:, -1, :]
        out = self.fc(last_out)
        return self.sigmoid(out)


# ═══════════════════════ 模型工厂 ═══════════════════════

def create_model(
    model_name: str,
    prediction_window: int = 24,
    **kwargs,
) -> BaseModel | nn.Module:
    """
    根据名称返回模型实例。

    Parameters
    ----------
    model_name : str
        "Mean" | "Persistence" | "SARIMA" | "LSTM" | "GRU"
    prediction_window : int
    **kwargs : 传递给具体模型的参数

    Returns
    -------
    model : BaseModel | nn.Module
    """
    registry = {
        "Mean": MeanBaseline,
        "Persistence": PersistenceBaseline,
        "SARIMA": SARIMAModel,
        "LSTM": LSTM,
        "GRU": GRU,
    }
    if model_name not in registry:
        raise ValueError(f"未知模型: {model_name}，可选: {list(registry.keys())}")

    model_class = registry[model_name]

    if model_name in ("Mean", "Persistence"):
        return model_class(prediction_window=prediction_window)
    elif model_name == "SARIMA":
        return model_class(
            order=kwargs.get("order", (1, 0, 1)),
            seasonal_order=kwargs.get("seasonal_order", (1, 0, 1, 24)),
            prediction_window=prediction_window,
        )
    else:  # LSTM / GRU
        return model_class(
            input_size=1,
            hidden_size=kwargs.get("hidden_size", 100),
            num_layers=kwargs.get("num_layers", 1),
            prediction_window=prediction_window,
            dropout=kwargs.get("dropout", 0.0),
        )
