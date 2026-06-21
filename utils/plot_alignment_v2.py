"""
Redraw alignment figure - show right merge clearly
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = 'experiments/02_analysis/figures'

fig = plt.figure(figsize=(12, 5.5))

# ── Raw CSV ──
ax1 = fig.add_axes([0.03, 0.52, 0.30, 0.43])
ax1.axis('off')
ax1.set_xlim(0, 2.5)
ax1.set_ylim(0, 8)

ax1.text(0.5, 7.3, '1) Raw CSV', fontsize=11, fontweight='bold', color='#2b7baa')
ax1.text(0.5, 6.6, '(data rows only)', fontsize=8, color='#888')

times = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00']
has_data = [True, True, False, False, True, True, False]
n_bytes = ['100', '120', '-', '-', '90', '110', '-']

y = 5.6
ax1.text(0.1, y, 'time', fontsize=7, fontweight='bold', color='#333')
ax1.text(1.0, y, 'n_bytes', fontsize=7, fontweight='bold', color='#333')
y -= 0.7
for t, has, val in zip(times, has_data, n_bytes):
    if has:
        ax1.text(0.1, y, t, fontsize=7, color='#111')
        ax1.text(1.0, y, val, fontsize=7, color='#111')
        y -= 0.7

ax1.text(0.1, y-0.2, '03:00, 04:00:', fontsize=7, color='#d62728', fontweight='bold')
ax1.text(0.1, y-0.6, 'row missing in CSV', fontsize=7, color='#d62728')

# ── times.csv ──
ax2 = fig.add_axes([0.36, 0.52, 0.22, 0.43])
ax2.axis('off')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 8)

ax2.text(0.5, 7.3, '2) times.csv', fontsize=11, fontweight='bold', color='#b8860b')
ax2.text(0.5, 6.6, '(all timestamps)', fontsize=8, color='#888')

y = 5.6
ax2.text(0.1, y, 'id_time', fontsize=7, fontweight='bold', color='#333')
ax2.text(1.0, y, 'time', fontsize=7, fontweight='bold', color='#333')
y -= 0.7
for i, t in enumerate(times):
    ax2.text(0.1, y, str(i), fontsize=7, color='#111')
    ax2.text(1.0, y, t, fontsize=7, color='#111')
    y -= 0.7

# ── right merge arrow ──
ax2.annotate('', xy=(2.3, 4.5), xytext=(2.3, 3.5),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.5))
ax2.text(2.4, 4.0, 'right merge\non id_time', fontsize=7, fontweight='bold', color='#d62728')

# ── After merge ──
ax3 = fig.add_axes([0.68, 0.52, 0.30, 0.43])
ax3.axis('off')
ax3.set_xlim(0, 2.5)
ax3.set_ylim(0, 8)

ax3.text(0.5, 7.3, '3) After merge', fontsize=11, fontweight='bold', color='#2ca02c')
ax3.text(0.5, 6.6, '(NaN marks missing)', fontsize=8, color='#888')

y = 5.6
ax3.text(0.1, y, 'time', fontsize=7, fontweight='bold', color='#333')
ax3.text(1.2, y, 'n_bytes', fontsize=7, fontweight='bold', color='#333')
y -= 0.7
for t, has, val in zip(times, has_data, n_bytes):
    ax3.text(0.1, y, t, fontsize=7, color='#111')
    if has:
        ax3.text(1.2, y, val, fontsize=7, color='#2ca02c')
    else:
        ax3.text(1.2, y, 'NaN', fontsize=7, color='#d62728', fontweight='bold')
    y -= 0.7

ax3.text(0.1, y-0.2, '03:00, 04:00:', fontsize=7, color='#d62728', fontweight='bold')
ax3.text(0.1, y-0.6, 'now have rows,', fontsize=7, color='#d62728')
ax3.text(0.1, y-1.0, 'values = NaN', fontsize=7, color='#d62728')

# ── Imputation methods ──
ax4 = fig.add_axes([0.03, 0.05, 0.94, 0.38])
ax4.axis('off')
ax4.set_xlim(0, 12)
ax4.set_ylim(0, 3.5)

ax4.text(0, 3.0, '4) Imputation methods', fontsize=11, fontweight='bold', color='#2b7baa')

# zeros
plt.Rectangle((0, 0.2), 3.5, 2.2, facecolor='#e8f4f8', edgecolor='#2b7baa', linewidth=1.5)
ax4.add_patch(plt.Rectangle((0, 0.2), 3.5, 2.2, facecolor='#e8f4f8', edgecolor='#2b7baa', linewidth=1.5))
ax4.text(1.75, 1.8, 'zeros', fontsize=10, fontweight='bold', ha='center')
ax4.text(1.75, 1.3, 'NaN -> 0', fontsize=9, color='#2b7baa', ha='center')
ax4.text(1.75, 0.75, 'Missing rate ~1%\nDefault for institutions', fontsize=7, color='#666', ha='center')

# mean
ax4.add_patch(plt.Rectangle((4, 0.2), 3.5, 2.2, facecolor='#fef0d9', edgecolor='#ff7f0e', linewidth=1.5))
ax4.text(5.75, 1.8, 'mean', fontsize=10, fontweight='bold', ha='center')
ax4.text(5.75, 1.3, 'NaN -> mean(df)', fontsize=9, color='#ff7f0e', ha='center')
ax4.text(5.75, 0.75, 'High missing rate\nStable distribution', fontsize=7, color='#666', ha='center')

# interpolate
ax4.add_patch(plt.Rectangle((8, 0.2), 3.5, 2.2, facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=1.5))
ax4.text(9.75, 1.8, 'interpolate', fontsize=10, fontweight='bold', ha='center')
ax4.text(9.75, 1.3, 'linear interpolation', fontsize=9, color='#27ae60', ha='center')
ax4.text(9.75, 0.75, 'Short gaps good\nLong gaps -> zeros', fontsize=7, color='#666', ha='center')

plt.savefig(f'{OUTPUT_DIR}/fig06_alignment_v2.png', bbox_inches='tight', dpi=150, facecolor='white')
plt.close()
print('Saved: fig06_alignment_v2.png')
