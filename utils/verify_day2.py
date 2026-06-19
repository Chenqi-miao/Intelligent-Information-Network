"""
verify_day2.py — Day 2 完成验证

跑通全部 pipeline：预处理 → Mean 基线 → SARIMA 基线 → 评估
在 1 条时间序列上验证（institutions/agg_1_hour/1.csv）。
"""
import sys, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.preprocessing import preprocess_pipeline
from src.models import MeanBaseline, SARIMAModel
from src.evaluate import compute_metrics, predict_model
from src.train import train_model, make_loader


# ── 1. 预处理 pipeline ──
print("=" * 60)
print("1. 预处理 pipeline 测试")
print("=" * 60)

result = preprocess_pipeline(
    file_id=1,
    ts_attribute="n_bytes",
    training_window=24,
    prediction_window=24,
)

X_train, X_val, X_test = result["X_train"], result["X_val"], result["X_test"]
y_train, y_val, y_test = result["y_train"], result["y_val"], result["y_test"]
scaler = result["scaler"]

print(f"   X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"   X_val:   {X_val.shape}  y_val:   {y_val.shape}")
print(f"   X_test:  {X_test.shape}  y_test:  {y_test.shape}")
print()

# ── 2. Mean 基线 ──
print("=" * 60)
print("2. Mean 基线")
print("=" * 60)

mean_model = MeanBaseline(prediction_window=24)
mean_pred = predict_model(mean_model, X_test, scaler)
y_test_orig = scaler.inverse_transform(y_test)
metrics = compute_metrics(y_test_orig, mean_pred)
print()

# ── 3. SARIMA 基线 ──
print("=" * 60)
print("3. SARIMA 基线")
print("=" * 60)

sarima = SARIMAModel(prediction_window=24)
sarima.fit(X_train, y_train)
sarima_pred = predict_model(sarima, X_test, scaler)
metrics = compute_metrics(y_test_orig, sarima_pred)
print()

# ── 4. LSTM 快速验证 ──
print("=" * 60)
print("4. LSTM 快速训练（5 epoch 验证）")
print("=" * 60)

import torch
from src.models import LSTM

model = LSTM(prediction_window=24)
train_loader = make_loader(X_train, y_train, batch_size=16)
val_loader = make_loader(X_val, y_val, batch_size=16, shuffle=False)

model, train_time, history = train_model(
    model, train_loader, val_loader,
    epochs=5, patience=3,
)
lstm_pred = predict_model(model, X_test, scaler)
metrics = compute_metrics(y_test_orig, lstm_pred)

print(f"\n{'='*60}")
print("[OK] Day 2 验证全部通过！")
print(f"{'='*60}")
