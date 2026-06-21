"""
SARIMA 滚动预测：对每个测试窗口重新拟合 → 预测 → 对比。
小范围测试，只跑 1-2 条序列。
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.preprocessing import load_and_align, impute_missing, create_sliding_windows
from src.evaluate import compute_metrics
from src.models import MeanBaseline

# ── 配置 ──
FILE_ID = 0
TW, PW = 24, 24
ORDER = (1, 0, 1)
SEASONAL = (1, 0, 1, 24)

# ── 加载数据 ──
df = load_and_align(FILE_ID)
df = impute_missing(df, 'zeros')
values = df['n_bytes'].values
print(f'File {FILE_ID}: {len(values)} points')

# ── 获取滑窗和分割 ──
X, y = create_sliding_windows(df, TW, PW)
n_total = len(X)
train_end = int(n_total * 0.35)
val_end = train_end + int(n_total * 0.05)

y_test = y[val_end:]
print(f'Test windows: {len(y_test)} (val_end={val_end})')

# ── 找测试窗口对应的时间索引 ──
# 滑动窗口: X[i] = values[i*PW : i*PW + TW], y[i] = values[i*PW + TW : i*PW + TW + PW]
# 所以第 i 个窗口的预测目标从 i*PW + TW 开始
test_start_idx = val_end * PW + TW
print(f'Test starts at data index: {test_start_idx}')

# ── Mean 基线（用全部训练+验证数据做均值）──
train_vals = values[:test_start_idx]
mean_val = train_vals.mean()
mean_pred = np.tile(mean_val, (len(y_test), PW))
mean_metrics = compute_metrics(y_test.flatten(), mean_pred.flatten())
print(f'Mean: R2={mean_metrics["r2"]:.4f}')

# ── SARIMA 滚动预测 ──
# 对每个测试窗口: 用历史数据拟合 → 预测24h → 和y_test[i]对比
sarima_preds = []
times = []
t0 = time.time()

n_test = min(len(y_test), 20)  # 只跑 20 个窗口
print(f'Rolling SARIMA on {n_test} windows...')

for i in range(n_test):
    # 决定训练数据截止点
    # 这个窗口要预测从 test_start_idx + i*PW 开始的 24 个点
    forecast_start = test_start_idx + i * PW

    history = values[:forecast_start]

    # 拟合 SARIMA
    model = SARIMAX(history, order=ORDER, seasonal_order=SEASONAL,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(maxiter=200, disp=False)

    # 预测 24 步
    forecast = fitted.forecast(steps=PW)
    sarima_preds.append(forecast)

    if (i+1) % 5 == 0:
        print(f'  window {i+1}/{n_test} ({time.time()-t0:.1f}s)')

elapsed = time.time() - t0
sarima_preds = np.array(sarima_preds)
sarima_metrics = compute_metrics(y_test[:n_test].flatten(), sarima_preds.flatten())

print(f'\nSARIMA rolling ({n_test} windows): time={elapsed:.0f}s, R2={sarima_metrics["r2"]:.4f}')
print(f'SARIMA per window: {elapsed/n_test:.2f}s')
