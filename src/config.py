"""
config.py — 实验配置管理

所有可调参数集中管理，不硬编码在脚本中。
每次实验前修改这里的参数，实现实验留痕。

论文参考：Table II（窗口设置）、Table III（超参数）
"""

from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path


# ───────────────────────── 项目路径 ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class Config:
    # ── 数据来源 ──
    data_group: str = "institutions"              # institutions | institution_subnets | ip_addresses_sample
    aggregation: str = "agg_1_hour"               # agg_1_hour | agg_10_minutes | agg_1_day

    # ── 预处理 ──
    impute_method: Literal["zeros", "mean", "interpolate"] = "zeros"
      # zeros:       缺失填 0（参考代码默认）
      # mean:        缺失填均值
      # interpolate: 线性插值
    normalize_method: Literal["zscore", "minmax"] = "zscore"
      # zscore: (x - μ) / σ（论文标准）
      # minmax: MinMaxScaler(0,1)（参考代码做法）

    # ── 滑动窗口（论文 Table II）──
    training_window: int = 24                     # 用过去多少小时
    prediction_window: int = 24                   # 预测未来多少小时

    # ── 数据集分割（按时间顺序，不随机打乱）──
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── 模型选择 ──
    model_name: str = "Mean"                      # Mean | SARIMA | LSTM | GRU | LSTM_FCN | GRU_FCN

    # ── LSTM / GRU 超参数（论文 Table III 参考）──
    hidden_size: int = 100
    n_layers: int = 1
    bidirectional: bool = False                   # 论文未明确用双向，默认 False
    dropout: float = 0.0

    # ── 训练超参数 ──
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.01
    patience: int = 5                             # Early Stopping 等待轮数
    seed: int = 42

    # ── 实验标识 ──
    experiment_name: str = ""                     # 留空会自动生成

    # ── 要建模的指标（空列表 = 使用默认指标）──
    ts_attributes: list[str] = field(default_factory=lambda: ["n_bytes"])

    # ── SARIMA 参数（由模型内部自动搜索，这里留占位）──
    sarima_order: tuple = (1, 0, 1)
    sarima_seasonal_order: tuple = (1, 0, 1, 24)  # 24 小时周期


# ───────────────────────── 快捷键配置 ─────────────────────────

# 论文 Table II 的 5 种窗口配置，快速切换
WINDOW_PRESETS = {
    "24_24":    (24, 24),      # 过去 1 天 → 未来 1 天
    "168_24":   (168, 24),     # 过去 1 周 → 未来 1 天
    "168_168":  (168, 168),    # 过去 1 周 → 未来 1 周
    "720_24":   (720, 24),     # 过去 1 月 → 未来 1 天
    "720_168":  (720, 168),    # 过去 1 月 → 未来 1 周
}

# 默认使用的 18 个指标（1h 聚合）
ALL_TS_ATTRIBUTES = [
    "n_bytes", "n_flows", "n_packets",
    "sum_n_dest_asn", "average_n_dest_asn", "std_n_dest_asn",
    "sum_n_dest_ports", "average_n_dest_ports", "std_n_dest_ports",
    "sum_n_dest_ip", "average_n_dest_ip", "std_n_dest_ip",
    "tcp_udp_ratio_packets", "tcp_udp_ratio_bytes",
    "dir_ratio_packets", "dir_ratio_bytes",
    "avg_duration", "avg_ttl",
]
