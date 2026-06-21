"""
Show imputation methods visually - zeros vs mean vs interpolate
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = 'experiments/02_analysis/figures'

# Create sample time series with a gap in the middle
np.random.seed(42)
t = np.arange(48)
signal = 10 + 5*np.sin(t/24*2*np.pi) + np.random.randn(48)*0.5
signal[20:28] = np.nan  # 8-hour gap

fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
fig.subplots_adjust(hspace=0.35)

colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c']
labels = ['Original (with missing values)', 'Zeros fill', 'Mean fill', 'Interpolate']
methods = [
    ('original', None),
    ('zeros', 0),
    ('mean', np.nanmean(signal)),
    ('interpolate', None),
]

for ax, (method, fill_val), color, label in zip(axes, methods, colors, labels):
    data = signal.copy()
    if method == 'original':
        ax.plot(t, data, 'o-', color=color, markersize=3, linewidth=0.8)
        # Mark NaN region
        ax.axvspan(19.5, 28.5, alpha=0.1, color='gray', label='gap')
    elif method == 'zeros':
        data[np.isnan(data)] = 0
        ax.plot(t, data, 'o-', color=color, markersize=3, linewidth=0.8)
        ax.plot(t[20:28], data[20:28], 'o', color=color, markersize=5, markerfacecolor='none', markeredgewidth=2)
    elif method == 'mean':
        data[np.isnan(data)] = fill_val
        ax.plot(t, data, 'o-', color=color, markersize=3, linewidth=0.8)
        ax.plot(t[20:28], data[20:28], 'o', color=color, markersize=5, markerfacecolor='none', markeredgewidth=2)
        ax.axhline(fill_val, color=color, linestyle=':', alpha=0.5, label=f'mean={fill_val:.1f}')
    elif method == 'interpolate':
        # linear interpolation
        mask = np.isnan(data)
        data[mask] = np.interp(t[mask], t[~mask], data[~mask])
        ax.plot(t, data, 'o-', color=color, markersize=3, linewidth=0.8)
        ax.plot(t[20:28], data[20:28], 'o', color=color, markersize=5, markerfacecolor='none', markeredgewidth=2)
        # Draw interpolation lines
        ax.plot([t[19], t[20]], [data[19], data[20]], '--', color=color, linewidth=1, alpha=0.5)
        ax.plot([t[27], t[28]], [data[27], data[28]], '--', color=color, linewidth=1, alpha=0.5)

    ax.set_ylabel('n_bytes')
    ax.set_title(label, fontsize=10, color=color, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.5, 47.5)
    if method == 'interpolate':
        ax.set_xlabel('Time (hours)')

    if method == 'original':
        ax.legend(fontsize=8)

plt.savefig(f'{OUTPUT_DIR}/fig07_imputation_compare.png', bbox_inches='tight', dpi=150, facecolor='white')
plt.close()
print('Saved: fig07_imputation_compare.png')
