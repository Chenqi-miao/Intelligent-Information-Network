"""
train.py — 模型训练循环

负责 PyTorch 模型的训练流程：
  - 训练循环（for epoch → for batch）
  - 验证集评估
  - Early Stopping
  - 学习率调度
  - 训练日志记录

用法：
    from src.train import train_model
    model, train_time, history = train_model(model, train_loader, val_loader, epochs=100)
"""

import time
import logging
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """固定随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """将 numpy 数组包装成 PyTorch DataLoader"""
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.01,
    patience: int = 5,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[nn.Module, float, dict[str, list]]:
    """
    训练 PyTorch 模型。

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader : DataLoader
    epochs : int
    learning_rate : float
    patience : int
    device : str
    seed : int
        随机种子

    Returns
    -------
    model, training_time, history
    """

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    start = time.time()

    for epoch in range(epochs):
        # ── 训练阶段 ──
        model.train()
        train_loss_sum = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)

        avg_train_loss = train_loss_sum / len(train_loader.dataset)

        # ── 验证阶段 ──
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss_sum += loss.item() * batch_x.size(0)

        avg_val_loss = val_loss_sum / len(val_loader.dataset)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # ── 日志 ──
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Epoch %3d/%d | train loss: %.6f | val loss: %.6f",
                epoch + 1, epochs, avg_train_loss, avg_val_loss,
            )

        # ── Early Stopping ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    "Early Stopping @ epoch %d (val loss did not improve for %d epochs)",
                    epoch + 1, patience,
                )
                break

    training_time = time.time() - start
    logger.info("Training done: %d epochs, %.2f sec", epoch + 1, training_time)

    return model, training_time, history
