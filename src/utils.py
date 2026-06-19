"""
utils.py — 可复用的工具函数（EDA + 数据加载）

用法：
    from src.utils import load_timeseries, compute_missing_rate
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_timeseries(
    group: str,
    file_id: int = 1,
    agg: str = "agg_1_hour",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    加载一条时间序列，合并时间戳索引。

    Parameters
    ----------
    group : str
        'institutions' | 'institution_subnets' | 'ip_addresses_sample'
    file_id : int
        文件编号（如 1, 2, 3...）
    agg : str
        'agg_1_hour' | 'agg_10_minutes' | 'agg_1_day'
    columns : list[str] | None
        指定加载的列，None 表示加载全部

    Returns
    -------
    pd.DataFrame
        index=时间, columns=指标
    """
    data_dir = DATA_DIR / group / agg
    df = pd.read_csv(data_dir / f"{file_id}.csv")
    times = pd.read_csv(data_dir / "times.csv")
    df = df.merge(times, on="id_time", how="left")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    if columns:
        df = df[columns]
    return df


def compute_missing_rate(
    group: str,
    agg: str = "agg_1_hour",
) -> dict:
    """
    计算某个层级下所有序列的缺失率。
    缺失 = 某些小时整行不存在（gap），不是 NaN。

    Parameters
    ----------
    group : str
        'institutions' | 'institution_subnets' | 'ip_addresses_sample'
    agg : str
        'agg_1_hour' | 'agg_10_minutes' | 'agg_1_day'

    Returns
    -------
    dict
        {"mean": float, "median": float, "rates": np.array, "n": int}
    """
    data_dir = DATA_DIR / group / agg
    csv_files = sorted(
        [
            f
            for f in data_dir.glob("*.csv")
            if f.name not in ("times.csv", "identifiers.csv")
        ]
    )
    times = pd.read_csv(data_dir / "times.csv")
    expected_count = len(times)

    rates = []
    for f in csv_files:
        df = pd.read_csv(f, usecols=["id_time"])
        rates.append((expected_count - len(df)) / expected_count * 100)

    rates = np.array(rates)
    return {
        "mean": float(rates.mean()),
        "median": float(np.median(rates)),
        "rates": rates,
        "n": len(rates),
    }


def list_available_files(
    group: str,
    agg: str = "agg_1_hour",
) -> list[int]:
    """列出某个层级下所有可用的文件 ID"""
    data_dir = DATA_DIR / group / agg
    files = []
    for f in data_dir.glob("*.csv"):
        if f.name not in ("times.csv", "identifiers.csv"):
            files.append(int(f.stem))
    return sorted(files)
