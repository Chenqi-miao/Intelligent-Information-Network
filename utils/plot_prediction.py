import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.preprocessing import preprocess_pipeline
from src.models import MeanBaseline, LSTM, GRU, create_model
from src.train import train_model, make_loader
from src.evaluate import predict_model, compute_metrics

def train_and_pred(model_name, X_train, y_train, X_val, y_val, X_test, scaler):
    model = create_model(model_name, prediction_window=24, hidden_size=100, bidirectional=True)
    train_loader = make_loader(X_train, y_train, batch_size=16)
    val_loader = make_loader(X_val, y_val, batch_size=16, shuffle=False)
    model, _, _ = train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.01, patience=5, device='cuda')
    pred = predict_model(model, X_test, scaler, device='cuda')
    return pred

# ── 选文件 ──
for file_id, label in [(0, 'Good case: file=0'), (243, 'Bad case: file=243')]:
    result = preprocess_pipeline(file_id=file_id, ts_attribute='n_bytes',
                                  training_window=24, prediction_window=24,
                                  train_ratio=0.35, val_ratio=0.05)
    scaler = result['scaler']
    X_test, y_test = result['X_test'], result['y_test']
    y_test_orig = scaler.inverse_transform(y_test)

    # ── 三种预测 ──
    mean_pred = predict_model(MeanBaseline(prediction_window=24), X_test, scaler)
    lstm_pred = train_and_pred('LSTM', result['X_train'], result['y_train'],
                                result['X_val'], result['y_val'], X_test, scaler)
    gru_pred = train_and_pred('GRU', result['X_train'], result['y_train'],
                               result['X_val'], result['y_val'], X_test, scaler)

    mean_r2 = compute_metrics(y_test_orig, mean_pred)['r2']
    lstm_r2 = compute_metrics(y_test_orig, lstm_pred)['r2']
    gru_r2 = compute_metrics(y_test_orig, gru_pred)['r2']

    # ── 画图 ──
    n_plot = min(4, len(X_test))
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot), sharex=True)

    for i in range(n_plot):
        ax = axes[i] if n_plot > 1 else axes
        t = np.arange(24)
        ax.plot(t, y_test_orig[i], 'k-', linewidth=1.5, label='Actual')
        ax.plot(t, mean_pred[i], '--', color='orange', linewidth=1, alpha=0.7, label='Mean')
        ax.plot(t, lstm_pred[i], '--', color='blue', linewidth=1, alpha=0.7, label='LSTM')
        ax.plot(t, gru_pred[i], ':', color='green', linewidth=1.5, label='GRU')
        ax.set_ylabel('n_bytes')
        ax.set_title(f'Window {i+1}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{label}  |  Mean R2={mean_r2:.4f}  LSTM R2={lstm_r2:.4f}  GRU R2={gru_r2:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'experiments/02_analysis/figures/prediction_file{file_id}_all.png', bbox_inches='tight', dpi=150)
    print(f'Saved: prediction_file{file_id}_all.png')
    plt.close()
