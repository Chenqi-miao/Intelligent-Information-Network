"""
Compute normalized RMSE and Harmonic Score from existing raw-scale results.
Correct approach: normalize y_test and y_pred to [0,1], then compute RMSE.
"""
import sys, os, glob, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.preprocessing import load_and_align, impute_missing, create_sliding_windows
from src.evaluate import compute_rmse

RESULTS_DIR = 'experiments/results'

# ── Load all results ──
all_results = []
for f in sorted(glob.glob(f'{RESULTS_DIR}/dl_results_*.csv')):
    all_results.append(pd.read_csv(f))
for f in sorted(glob.glob(f'{RESULTS_DIR}/baseline_results_*.csv')):
    all_results.append(pd.read_csv(f))
for f in sorted(glob.glob(f'{RESULTS_DIR}/sarima_rolling_*.csv')):
    all_results.append(pd.read_csv(f))

df = pd.concat(all_results, ignore_index=True)

# Filter: use all windows, focus on institutions + subnets
# Reconstruct normalized RMSE for each row
def compute_norm_rmse_for_row(row):
    """For a given result row, load the original file and compute normalized RMSE"""
    try:
        file_id = int(row['TS_ID'])
        group = row['TS_GROUP']
        tw = int(row['TRAINING_WINDOW'])
        pw = int(row['PREDICTION_WINDOW'])
        raw_pred_rmse = row['RMSE']  # raw-scale RMSE from our results

        # Load data and split
        ts_df = load_and_align(file_id, 'n_bytes', group, 'agg_1_hour')
        ts_df = impute_missing(ts_df, 'zeros', 'n_bytes')
        values = ts_df['n_bytes'].values

        # Get training set values for MinMax fit
        X, y = create_sliding_windows(ts_df, tw, pw)
        n_total = len(X)
        train_end = int(n_total * 0.35)
        val_end = train_end + int(n_total * 0.05)

        # Training data: first 35% of sliding windows
        y_train_flat = y[:train_end].flatten()
        train_min = y_train_flat.min()
        train_max = y_train_flat.max()

        if train_max - train_min < 1e-10:
            return 0.0, 0.0

        # The raw RMSE is on the original scale
        # Normalize it: since RMSE is in same units as data,
        # normalized RMSE = raw_RMSE / (train_max - train_min)
        norm_rmse = raw_pred_rmse / (train_max - train_min)

        # Clip to reasonable range [0, 10]
        norm_rmse = min(norm_rmse, 10.0)

        return norm_rmse, train_max - train_min

    except Exception as e:
        return np.nan, np.nan

# Only process institutions and subnets (ip_addresses_sample has extreme sparsity)
df_sub = df[df.TS_GROUP.isin(['institutions', 'institution_subnets'])].copy()

print(f'Computing normalized metrics for {len(df_sub)} records...')
norm_data = df_sub.apply(compute_norm_rmse_for_row, axis=1, result_type='expand')
df_sub['NORM_RMSE'] = norm_data[0]
df_sub['SCALE'] = norm_data[1]

# Harmonic Score
def harmonic(norm_rmse, r2):
    if np.isnan(norm_rmse) or np.isnan(r2):
        return np.nan
    r2t = abs(r2 - 1)
    denom = norm_rmse + r2t
    if denom < 1e-10:
        return 0.0
    return 2 * (norm_rmse * r2t) / denom

df_sub['HARMONIC'] = df_sub.apply(lambda r: harmonic(r['NORM_RMSE'], r['R2_SCORE']), axis=1)

# ════════ PRINT TABLES ════════

output_lines = []

def p(text=""):
    output_lines.append(text)
    print(text)

# ── Table IV: Normalized RMSE ──
p("=" * 80)
p("TABLE IV: Overall Mean Normalized RMSE (lower is better)")
p("=" * 80)
pivot_rmse = df_sub.pivot_table(
    values='NORM_RMSE', index='MODEL', columns='TS_GROUP', aggfunc='mean'
).round(4)
p(pivot_rmse.to_string())
p()

# ── Table V: R2-score ──
p("=" * 80)
p("TABLE V: Overall Mean R2-score (higher is better)")
p("=" * 80)
pivot_r2 = df_sub.pivot_table(
    values='R2_SCORE', index='MODEL', columns='TS_GROUP', aggfunc='mean'
).round(4)
p(pivot_r2.to_string())
p()

# ── Table VI: Harmonic Score ──
p("=" * 80)
p("TABLE VI: Overall Mean Harmonic Score (lower is better)")
p("=" * 80)
pivot_hs = df_sub.pivot_table(
    values='HARMONIC', index='MODEL', columns='TS_GROUP', aggfunc='mean'
).round(4)
p(pivot_hs.to_string())
p()

# ── By window (paper format) ──
p("=" * 80)
p("TABLE IV-VI detail: By Model x Window x Group (NormRMSE | R2 | HS)")
p("=" * 80)
detail = df_sub.pivot_table(
    values=['NORM_RMSE', 'R2_SCORE', 'HARMONIC'],
    index=['MODEL', 'TRAINING_WINDOW', 'PREDICTION_WINDOW'],
    columns='TS_GROUP',
    aggfunc='mean'
).round(4)
p(detail.to_string())
p()

# Save
with open(f'{RESULTS_DIR}/normalized_tables.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))
print(f'\nSaved: {RESULTS_DIR}/normalized_tables.txt')
