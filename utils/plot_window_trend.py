import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'experiments/02_analysis/figures'

# 数据：不同窗口下的 R² 中位数
windows = ['(24,24)\nShort', '(168,24)\nMed', '(168,168)\nLong', '(744,168)\nVLong']
lstm_r2 = [0.098, 0.095, 0.046, 0.032]
gru_r2 = [0.097, 0.099, 0.063, 0.047]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(windows))
w = 0.3

bars1 = ax.bar(x - w/2, lstm_r2, w, label='LSTM', color='#1f77b4', alpha=0.85)
bars2 = ax.bar(x + w/2, gru_r2, w, label='GRU', color='#ff7f0e', alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(windows, fontsize=11)
ax.set_ylabel('R2 median', fontsize=12)
ax.set_title('R2 median by window configuration', fontsize=13)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.14)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_window_trend.png', bbox_inches='tight', dpi=150)
print(f'Saved: {OUTPUT_DIR}/04_window_trend.png')
plt.close()

# 第二张图：窗口配置的 RMSE 趋势
fig, ax = plt.subplots(figsize=(10, 5))
# 用 RMSE 比值（相对(24,24)归一化）来看衰减趋势
lstm_ratio = [1.0, 1.05, 2.13, 2.56]
gru_ratio = [1.0, 1.02, 1.60, 2.11]

ax.plot(x, lstm_ratio, 'o-', label='LSTM', color='#1f77b4', linewidth=2, markersize=8)
ax.plot(x, gru_ratio, 's-', label='GRU', color='#ff7f0e', linewidth=2, markersize=8)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(windows, fontsize=11)
ax.set_ylabel('RMSE ratio (relative to (24,24))', fontsize=12)
ax.set_title('RMSE degradation as prediction window grows', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_window_rmse_degradation.png', bbox_inches='tight', dpi=150)
print(f'Saved: {OUTPUT_DIR}/05_window_rmse_degradation.png')
plt.close()
