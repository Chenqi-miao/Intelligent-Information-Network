"""
preprocessing.py — 数据预处理完整 pipeline

流程（对齐论文 Fig.3 方法论）：
  1. 加载 CSV + times.csv → right merge 识别缺失时间点（产生 NaN）
  2. 缺失值填充（zeros / mean / interpolate）
  3. 创建滑动窗口样本 (X, y)
  4. 按时间顺序分割 train / val / test（70% / 15% / 15%）
  5. Z-score 标准化（仅用训练集 fit，再 transform 全部）

用法：
    from src.preprocessing import preprocess_pipeline

    result = preprocess_pipeline(
        file_id=1,
        ts_attribute="n_bytes",
        training_window=24,
        prediction_window=24,
    )
    X_train, X_val, X_test = result["X_train"], result["X_val"], result["X_test"]
    y_train, y_val, y_test = result["y_train"], result["y_val"], result["y_test"]

输出 shape：
    X: (样本数, training_window, 1)          # batch_first 格式，供 LSTM/GRU 直接使用
    y: (样本数, prediction_window)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════ 第一步：加载与对齐 ═══════════════════════

def load_and_align(
    file_id: int,
    ts_attribute: str = "n_bytes",
    group: str = "institutions",
    aggregation: str = "agg_1_hour",
) -> pd.DataFrame:
    """
    加载一条时间序列，与完整时间戳做 right merge。

    目的：原始 CSV 只包含有数据的行（数据缺口直接缺失行），
          right merge 后全时间轴对齐，缺失的时刻以 NaN 标记。

    Parameters
    ----------
    file_id : int
        文件编号（如 1, 2, ... 283）
    ts_attribute : str
        要加载的指标列名（如 "n_bytes", "n_flows"）
    group : str
        层级：institutions / institution_subnets / ip_addresses_sample
    aggregation : str
        聚合粒度：agg_1_hour / agg_10_minutes / agg_1_day

    Returns
    -------
    pd.DataFrame
        index=时间, 列=[ts_attribute], 缺失时刻为 NaN
    """
    data_dir = DATA_DIR / group / aggregation

    # 加载原始数据
    df = pd.read_csv(data_dir / f"{file_id}.csv")
    times = pd.read_csv(data_dir / "times.csv")

    # right merge：以 times.csv 为基准，缺失时间窗口变为 NaN
    df = df.merge(times, on="id_time", how="right")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df = df[[ts_attribute]]

    n_total = len(df)
    n_missing = df[ts_attribute].isna().sum()
    missing_rate = n_missing / n_total * 100
    logger.info(
        "加载文件 %d/%s/%s/%s | 总时间点 %d | 缺失 %.1f%% (%d/%d)",
        file_id, group, aggregation, ts_attribute,
        n_total, missing_rate, n_missing, n_total,
    )
    return df


# ═══════════════════════ 第二步：缺失值填充 ═══════════════════════

def impute_missing(
    df: pd.DataFrame,
    method: str = "zeros",
    ts_attribute: str = "n_bytes",
) -> pd.DataFrame:
    """
    填充缺失值（数据加载后已对齐时间轴，缺失位置为 NaN）。

    Parameters
    ----------
    df : pd.DataFrame
        经 load_and_align() 处理后的数据
    method : str
        "zeros"      — 缺失填 0（参考代码默认做法）
        "mean"       — 缺失填该列均值
        "interpolate"— 线性插值（最多连续填充 4 个缺失点）
    ts_attribute : str
        需要填充的列名

    Returns
    -------
    pd.DataFrame
        填充完成的数据
    """
    df = df.copy()
    missing_count = df[ts_attribute].isna().sum()

    if missing_count == 0:
        logger.info("无需填充，无缺失值")
        return df

    if method == "zeros":
        df[ts_attribute] = df[ts_attribute].fillna(0)
        logger.info("缺失值填充：zeros（共 %d 个）", missing_count)

    elif method == "mean":
        fill_value = df[ts_attribute].mean()
        df[ts_attribute] = df[ts_attribute].fillna(fill_value)
        logger.info("缺失值填充：mean = %.4f（共 %d 个）", fill_value, missing_count)

    elif method == "interpolate":
        df[ts_attribute] = df[ts_attribute].interpolate(
            method="linear", limit=4, limit_direction="forward"
        )
        # 如果首部仍有 NaN（interpolate 无法前向外推），用零填充
        remaining = df[ts_attribute].isna().sum()
        if remaining > 0:
            df[ts_attribute] = df[ts_attribute].fillna(0)
            logger.info(
                "缺失值填充：interpolate（共 %d，首部 %d 个补零）",
                missing_count, remaining,
            )
        else:
            logger.info("缺失值填充：interpolate（共 %d 个）", missing_count)
    else:
        raise ValueError(f"不支持的填充方法: {method}，可选 zeros/mean/interpolate")

    return df


# ═══════════════════════ 第三步：滑动窗口 ═══════════════════════

def create_sliding_windows(
    df: pd.DataFrame,
    training_window: int = 24,
    prediction_window: int = 24,
    ts_attribute: str = "n_bytes",
) -> tuple[np.ndarray, np.ndarray]:
    """
    将连续时间序列切成 (training_window, prediction_window) 的样本对。

    参考：论文 Table II + 开源代码 runner_component.py create_input_data()
    步长 = prediction_window，不重叠预测窗口。

    Parameters
    ----------
    df : pd.DataFrame
        填充完成的数据
    training_window : int
        训练窗口长度（用过去多少个时间步）
    prediction_window : int
        预测窗口长度（预测未来多少个时间步）
    ts_attribute : str
        目标列名

    Returns
    -------
    X : np.ndarray, shape (样本数, training_window, 1)
    y : np.ndarray, shape (样本数, prediction_window)
    """
    values = df[ts_attribute].values
    X, y = [], []

    for i in range(training_window, len(values) - prediction_window, prediction_window):
        x_sample = values[i - training_window : i]
        y_sample = values[i : i + prediction_window]

        # 跳过含 NaN 的样本（理论不会发生，保险检查）
        if np.isnan(x_sample).any() or np.isnan(y_sample).any():
            continue

        X.append(x_sample.reshape(-1, 1))       # (training_window, 1)
        y.append(y_sample)                       # (prediction_window,)

    X = np.array(X)
    y = np.array(y)

    logger.info(
        "滑动窗口：tw=%d pw=%d | 样本数 %d",
        training_window, prediction_window, len(X),
    )
    if len(X) == 0:
        raise ValueError(
            f"滑动窗口未产生任何样本。检查数据长度 ({len(values)}) "
            f"是否大于 training_window + prediction_window ({training_window + prediction_window})"
        )
    return X, y


# ═══════════════════════ 第四步：按时间顺序分割 ═══════════════════════

def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    按时间顺序分割数据集（不随机打乱）。

    时间序列预测必须按时间顺序分割，防止未来信息泄露。

    Parameters
    ----------
    X : np.ndarray
        特征，shape (N, training_window, 1)
    y : np.ndarray
        目标，shape (N, prediction_window)
    train_ratio : float
        训练集比例（默认 0.7）
    val_ratio : float
        验证集比例（默认 0.15）

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    logger.info(
        "时间分割：train=%d val=%d test=%d | 比例 %.0f/%.0f/%.0f",
        len(X_train), len(X_val), len(X_test),
        train_ratio * 100, val_ratio * 100, (1 - train_ratio - val_ratio) * 100,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════ 第五步：标准化 ═══════════════════════

class ZScoreScaler:
    """
    Z-score 标准化（均值为 0，标准差为 1）。

    用法：
        scaler = ZScoreScaler()
        scaler.fit(X_train)          # 仅用训练集计算 μ, σ
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled  = scaler.transform(X_test)   # 用训练集的 μ, σ
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        """仅从训练数据计算 μ 和 σ"""
        self.mean_ = X.mean()
        self.std_ = X.std()
        if self.std_ < 1e-10:
            self.std_ = 1.0  # 防止除零
        logger.info("Z-score：μ=%.4f σ=%.4f", self.mean_, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """用已计算的 μ, σ 做标准化"""
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """还原回原始尺度"""
        return X * self.std_ + self.mean_


# ═══════════════════════ 完整 Pipeline ═══════════════════════

def preprocess_pipeline(
    file_id: int,
    ts_attribute: str = "n_bytes",
    group: str = "institutions",
    aggregation: str = "agg_1_hour",
    impute_method: str = "zeros",
    training_window: int = 24,
    prediction_window: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict:
    """
    完整预处理流程：加载 → 填充 → 滑窗 → 分割 → 标准化。

    Parameters
    ----------
    file_id : int
    ts_attribute : str
    group : str
    aggregation : str
    impute_method : str
        zeros / mean / interpolate
    training_window : int
    prediction_window : int
    train_ratio : float
    val_ratio : float

    Returns
    -------
    dict 包含：
        X_train, X_val, X_test  — shape (N, training_window, 1)
        y_train, y_val, y_test  — shape (N, prediction_window)
        scaler                  — ZScoreScaler 对象（用于反标准化）
        df                      — 填充后的完整 DataFrame
    """
    # Step 1 & 2: 加载 + 填充
    df = load_and_align(file_id, ts_attribute, group, aggregation)
    df = impute_missing(df, impute_method, ts_attribute)

    # Step 3: 滑窗
    X, y = create_sliding_windows(df, training_window, prediction_window, ts_attribute)

    # Step 4: 时间分割
    splits = temporal_split(X, y, train_ratio, val_ratio)
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    # Step 5: Z-score 标准化（仅 fit 训练集）
    scaler = ZScoreScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # y 也需要标准化（用同样的 scaler，因为 X 和 y 同尺度）
    y_train = scaler.transform(y_train)
    y_val   = scaler.transform(y_val)
    y_test  = scaler.transform(y_test)

    logger.info("预处理完成：%s file=%d %s=%s", group, file_id, ts_attribute, "✓")
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler,
        "df": df,
    }
