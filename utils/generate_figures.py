"""
Generate explanatory figures for report:
1. Preprocessing pipeline flowchart
2. Sliding window mechanism
3. Temporal split visualization
4. LSTM architecture
5. SARIMA decomposition
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'experiments/02_analysis/figures'


# ═══════════════════ 1. Preprocessing Pipeline ═══════════════════

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    steps = [
        ('Raw CSV\n+ times.csv', 1),
        ('Right Merge\n→ Time-aligned', 3),
        ('Impute\nMissing (zeros)', 5),
        ('Sliding Window\n(tw, pw)', 7),
        ('Temporal Split\n35:5:60', 9),
        ('Z-score\nNormalize', 11),
        ('Train / Eval', 13),
    ]

    for label, x in steps:
        box = FancyBboxPatch((x-0.7, 1.2), 1.4, 1.6, boxstyle="round,pad=0.1",
                              facecolor='#e8f4f8', edgecolor='#2b7baa', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 2.0, label, ha='center', va='center', fontsize=8, linespacing=1.5)

    for i in range(len(steps)-1):
        x1 = steps[i][1] + 0.7
        x2 = steps[i+1][1] - 0.7
        ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

    # Data flow indicators
    ax.annotate('', xy=(1.7, 2.8), xytext=(1.7, 3.5),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1))
    ax.text(2, 3.5, '原始CSV缺失时间点→NaN', fontsize=7, color='#666', ha='center')

    ax.text(7, 0.5, '图 1：预处理流程 Pipeline', ha='center', fontsize=11, fontweight='bold')
    plt.savefig(f'{OUTPUT_DIR}/fig01_pipeline.png', bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print('Saved: fig01_pipeline.png')

draw_pipeline()


# ═══════════════════ 2. Sliding Window ═══════════════════

def draw_sliding_window():
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={'height_ratios': [1, 1.5]})
    fig.subplots_adjust(hspace=0.4)

    # Top: time series
    ax = axes[0]
    np.random.seed(42)
    t = np.arange(200)
    data = 50 + 10*np.sin(t/24*2*np.pi) + np.random.randn(200)*3 + t*0.02
    ax.plot(t, data, 'k-', linewidth=0.8)
    ax.set_ylabel('n_bytes')
    ax.set_title('Time Series (hourly)', fontsize=11)
    ax.set_xlim(0, 200)

    # Highlight first sliding window
    tw, pw = 24, 24
    colors = ['#ff7f0e', '#2ca02c']
    labels = [f'training_window={tw}h', f'prediction_window={pw}h']
    for start, color, label in [(10, colors[0], labels[0]), (10+tw, colors[1], labels[1])]:
        ax.axvspan(start, start+24, alpha=0.2, color=color, label=label)
    ax.legend(fontsize=8, loc='upper right')

    # Bottom: window pairs
    ax = axes[1]
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 6)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Sample index')
    ax.set_title('Sliding window pairs (stride = prediction_window)', fontsize=11)

    n_windows = 6
    for i in range(n_windows):
        start = i * pw
        y = n_windows - i - 1  # stack from bottom
        # training part
        ax.barh(y, tw, left=start, height=0.6, color='#ff7f0e', alpha=0.7, edgecolor='white')
        ax.barh(y, pw, left=start+tw, height=0.6, color='#2ca02c', alpha=0.7, edgecolor='white')
        ax.text(start + tw/2, y, f'X{i}', ha='center', va='center', fontsize=7, color='white')
        ax.text(start + tw + pw/2, y, f'y{i}', ha='center', va='center', fontsize=7, color='white')

    ax.set_yticks(range(n_windows))
    ax.set_yticklabels([f'Sample {i}' for i in range(n_windows-1, -1, -1)], fontsize=8)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff7f0e', alpha=0.7, label=f'Input (past {tw}h)'),
                       Patch(facecolor='#2ca02c', alpha=0.7, label=f'Target (next {pw}h)')]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

    fig.suptitle('图 2：滑动窗口机制', fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(f'{OUTPUT_DIR}/fig02_sliding_window.png', bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print('Saved: fig02_sliding_window.png')

draw_sliding_window()


# ═══════════════════ 3. Temporal Split ═══════════════════

def draw_temporal_split():
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(hspace=0.5)

    colors_split = {'Train': '#1f77b4', 'Val': '#ff7f0e', 'Test': '#2ca02c'}

    # Top: 70:15:15
    ax = axes[0]
    ax.barh(0, 0.7, left=0, height=0.4, color=colors_split['Train'], edgecolor='white', label='Train')
    ax.barh(0, 0.15, left=0.7, height=0.4, color=colors_split['Val'], edgecolor='white', label='Val')
    ax.barh(0, 0.15, left=0.85, height=0.4, color=colors_split['Test'], edgecolor='white', label='Test')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_title('70:15:15 split — test set concentrated in summer', fontsize=10)
    ax.text(0.35, -0.3, 'Oct~Apr (active)', ha='center', fontsize=7, color='#1f77b4')
    ax.text(0.92, -0.3, 'Jun~Jul (summer slump)', ha='center', fontsize=7, color='#2ca02c')
    ax.axis('off')

    # Bottom: 35:5:60
    ax = axes[1]
    ax.barh(0, 0.35, left=0, height=0.4, color=colors_split['Train'], edgecolor='white', label='Train')
    ax.barh(0, 0.05, left=0.35, height=0.4, color=colors_split['Val'], edgecolor='white', label='Val')
    ax.barh(0, 0.60, left=0.40, height=0.4, color=colors_split['Test'], edgecolor='white', label='Test')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_title('35:5:60 split — test set covers full range', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.axis('off')

    fig.suptitle('图 3：时间分割策略对比', fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(f'{OUTPUT_DIR}/fig03_temporal_split.png', bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print('Saved: fig03_temporal_split.png')

draw_temporal_split()


# ═══════════════════ 4. LSTM Architecture ═══════════════════

def draw_lstm_arch():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Input
    ax.text(1, 4.0, 'Input\n(seq_len=24, dim=1)', ha='center', va='center',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='#e0e0e0', edgecolor='#666'))

    # LSTM layer
    lstm_box = FancyBboxPatch((2.5, 2.5), 3, 2, boxstyle="round,pad=0.15",
                               facecolor='#d4e6f1', edgecolor='#2b7baa', linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(4, 4.2, 'Bidirectional LSTM', ha='center', fontsize=9, fontweight='bold')
    ax.text(4, 3.7, 'hidden_size=100', ha='center', fontsize=8, color='#555')
    ax.text(4, 3.2, 'num_layers=1', ha='center', fontsize=8, color='#555')

    # LSTM cell visual
    cell_positions = [(4.5, 2.8)]
    for cx, cy in cell_positions:
        circle = plt.Circle((cx, cy), 0.3, color='#2b7baa', alpha=0.5)
        ax.add_patch(circle)

    # Hidden state output
    h_box = FancyBboxPatch((6.5, 2.5), 1.5, 2, boxstyle="round,pad=0.1",
                            facecolor='#f0e6d4', edgecolor='#b8860b', linewidth=1.5)
    ax.add_patch(h_box)
    ax.text(7.25, 3.5, 'Last hidden\nstate', ha='center', fontsize=8, fontweight='bold')
    ax.text(7.25, 2.9, '(200-dim\nbidir)', ha='center', fontsize=7, color='#555')

    # Output
    out_box = FancyBboxPatch((9, 2.5), 1.5, 2, boxstyle="round,pad=0.1",
                              facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=1.5)
    ax.add_patch(out_box)
    ax.text(9.75, 3.5, 'Output\n24h pred', ha='center', fontsize=8, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(2.5, 4.0), xytext=(1.7, 4.0),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    ax.annotate('', xy=(5.5, 3.5), xytext=(5.0, 3.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    ax.annotate('', xy=(6.5, 3.5), xytext=(6.0, 3.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    ax.annotate('', xy=(9, 3.5), xytext=(8.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

    # Linear layer annotation
    ax.annotate('Linear\n(hidden→pw)', ha='center', fontsize=7, color='#888',
                xy=(8.25, 1.8), xytext=(8.25, 1.2),
                arrowprops=dict(arrowstyle='->', color='#888', lw=0.8))

    # Gate details below
    ax.text(4, 1.5, 'Forget Gate: $f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$',
            ha='center', fontsize=7, color='#444')
    ax.text(4, 1.0, 'Input Gate:  $i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$',
            ha='center', fontsize=7, color='#444')
    ax.text(4, 0.5, 'Output Gate: $o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$',
            ha='center', fontsize=7, color='#444')

    fig.suptitle('图 4：LSTM 模型结构', fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(f'{OUTPUT_DIR}/fig04_lstm_arch.png', bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print('Saved: fig04_lstm_arch.png')

draw_lstm_arch()


# ═══════════════════ 5. SARIMA Decomposition ═══════════════════

def draw_sarima_decomp():
    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    np.random.seed(42)
    t = np.arange(336)

    # Original: trend + seasonality + noise
    trend = t * 0.02
    seasonal = 10 * np.sin(t/24*2*np.pi)
    noise = np.random.randn(336) * 2
    original = 50 + trend + seasonal + noise

    # Components
    components = [
        ('Original $y_t$', original, '#1f77b4'),
        ('Trend $T_t$', 50 + trend, '#ff7f0e'),
        ('Seasonal $S_t$ (period=24h)', seasonal + 50, '#2ca02c'),
        ('Residual $\\varepsilon_t$', noise + 50, '#d62728'),
    ]

    for ax, (title, data, color) in zip(axes, components):
        ax.plot(t, data, linewidth=0.7, color=color)
        ax.set_ylabel(title, fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 336)
        ax.axvline(0, color='gray', linewidth=0.5)

    axes[-1].set_xlabel('Time (hours, 2 weeks)', fontsize=10)
    fig.suptitle('图 5：SARIMA 时间序列分解', fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(f'{OUTPUT_DIR}/fig05_sarima_decomp.png', bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print('Saved: fig05_sarima_decomp.png')

draw_sarima_decomp()


# ═══════════════════ 6. Data alignment illustration ═══════════════════

def draw_alignment():
    fig, axes = plt.subplots(3, 1, figsize=(10, 4))
    fig.subplots_adjust(hspace=0.6)

    # Raw: missing rows
    ax = axes[0]
    data1 = [(0, 100), (1, 120), (2, None), (3, None), (4, 90), (5, 110)]
    times1 = [d[0] for d in data1]
    vals1 = [d[1] for d in data1]
    for t_, v in data1:
        marker = 'o' if v is not None else 'x'
        color = '#1f77b4' if v is not None else '#d62728'
        ax.plot(t_, v if v else 0, marker, color=color, markersize=8)
    ax.set_title('Raw CSV (missing rows = gaps)', fontsize=9)
    ax.set_ylabel('n_bytes')
    ax.set_xlim(-0.5, 5.5)

    # Right merge
    ax = axes[1]
    data2 = [(0, 100), (1, 120), (2, 0), (3, 0), (4, 90), (5, 110)]
    for t_, v in data2:
        ax.plot(t_, v, 'o', color='#ff7f0e', markersize=8)
    ax.set_title('After right merge with times.csv (NaN → detect gaps)', fontsize=9)
    ax.set_ylabel('n_bytes')
    ax.set_xlim(-0.5, 5.5)

    # After impute
    ax = axes[2]
    for t_, v in data2:
        ax.plot(t_, v, 'o', color='#2ca02c', markersize=8)
    ax.set_title('After imputation (zeros/mean/interpolate)', fontsize=9)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('n_bytes')
    ax.set_xlim(-0.5, 5.5)

    fig.suptitle('图 6：时间轴对齐与缺失值填充示意图', fontsize=11, fontweight='bold', y=1.02)
    plt.savefig(f'{OUTPUT_DIR}/fig06_alignment.png', bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print('Saved: fig06_alignment.png')

draw_alignment()

print(f'\nAll figures saved to {OUTPUT_DIR}/')
