# ISP Traffic Forecasting — 复现实验

复现论文 "Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting"（Koumar et al., 2025）的核心实验，在 CESNET-TimeSeries24 数据集上对比 Mean / SARIMA / LSTM / GRU 四种模型的流量预测性能。

## 项目结构

```
├── src/                          # 核心源码
│   ├── config.py                 # 配置管理（超参数、路径、窗口预设）
│   ├── preprocessing.py          # 预处理 pipeline（对齐→填充→滑窗→分割→标准化）
│   ├── models.py                 # 模型定义（Mean / SARIMA / LSTM / GRU）
│   ├── train.py                  # PyTorch 训练循环 + Early Stopping
│   └── evaluate.py               # 评估指标（RMSE / R² / sMAPE）+ 结果记录
├── experiments/                  # 实验输出
│   ├── run_baselines.py          # 批量运行基线（Mean / SARIMA）
│   ├── run_dl.py                 # 批量运行 LSTM / GRU
│   ├── run_sarima_rolling.py     # SARIMA 滚动预测
│   ├── run_batch_experiments.sh  # 批量超参数实验
│   └── results/                  # 结果 CSV
├── notebooks/                    # Jupyter 分析
│   ├── 01_eda.ipynb              # 数据探索性分析
│   └── 02_initial_analysis.ipynb # 初步实验结果分析
├── report/                       # 实验报告
│   └── report_draft.md           # 报告正文
├── data/                         # 数据集（需自行下载）
│   └── institutions/agg_1_hour/
├── utils/                        # 工具脚本
│   └── verify_day2.py            # pipeline 验证
├── memory/                       # 项目决策记录
└── pyproject.toml                # 依赖管理（uv）
```

## 环境要求

- Python >= 3.11
- GPU（推荐，LSTM/GRU 训练用）或 CPU
- uv（环境管理）

## 安装

```bash
uv sync
```

CUDA GPU 加速（可选，RTX 5060 需 torch ≥ 2.7.0）：

```bash
uv pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu128
```

## 数据准备

从 Zenodo 下载 [CESNET-TimeSeries24](https://zenodo.org/records/13382427) 数据集，将 `institutions/agg_1_hour/` 目录放入 `data/institutions/`。

数据格式：每条序列一个 CSV 文件，包含列 `id_time, n_bytes, n_flows, ...` 等 18 个流量指标，配套 `times.csv` 提供时间戳映射。

## 运行实验

### 1. 探索性数据分析

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
```

### 2. 基线模型（Mean）

```bash
uv run python experiments/run_baselines.py --train-ratio 0.35 --val-ratio 0.05 --groups institutions
```

### 3. LSTM / GRU

```bash
# LSTM，283 条机构级，35:5:60 分割
uv run python experiments/run_dl.py --model LSTM --train-ratio 0.35 --val-ratio 0.05 --groups institutions

# GRU
uv run python experiments/run_dl.py --model GRU --train-ratio 0.35 --val-ratio 0.05 --groups institutions

# 不同窗口配置（如 168→24）
uv run python experiments/run_dl.py --model GRU --tw 168 --pw 24 --train-ratio 0.35 --val-ratio 0.05
```

### 4. SARIMA 滚动预测（较慢）

```bash
uv run python experiments/run_sarima_rolling.py --n-windows 10 --groups institutions
```

### 5. 批量超参数实验

```bash
bash experiments/run_batch_experiments.sh
```

## 结果文件

实验结果输出到 `experiments/results/` 目录，CSV 格式：

| 文件 | 内容 |
|------|------|
| `baseline_results_*.csv` | Mean / SARIMA 基线结果 |
| `dl_results_*.csv` | LSTM / GRU 结果 |
| `sarima_rolling_*.csv` | SARIMA 滚动预测结果 |

每行包含：`TS_ID, TS_GROUP, TRAINING_WINDOW, PREDICTION_WINDOW, MODEL, TS_METRIC, RMSE, SMAPE, R2_SCORE, TRAINING_TIME`

## 核心结果

| 模型 | R² 中位数 | R²>0 占比 |
|------|-----------|-----------|
| Mean | -0.004 | 46.7% |
| LSTM | 0.116 | 83.5% |
| GRU | 0.145 | 81.1% |
| SARIMA（滚动预测） | 0.183 | 91.5% |

详细分析与图表见 `report/report_draft.md`。

## 技术栈

- PyTorch 2.11（纯 torch.nn，无 Lightning/tsai 等高封装库）
- statsmodels（SARIMA）
- numpy / pandas / matplotlib
- 环境管理：uv
