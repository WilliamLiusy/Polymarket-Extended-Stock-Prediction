"""特征矩阵的构建与缓存。

输入契约：qlib 二进制库 + instruments 股票池文件。
输出契约：`MultiIndex[datetime, instrument]` 的 DataFrame，
即方案里的 `[date, instruments, features]`；`LABEL0` 列是待预测的 5 日收益。

## 为什么要缓存

216 个 Alpha158 表达式 × 1500 只股票 × 4000 天，qlib 求值一次要几分钟到几十分钟。
walk-forward 会反复用同一份特征（只是切片不同），所以求值一次、落 parquet、
后续全部读缓存。缓存的 key 里带上股票池、窗口集合、标签表达式和日期范围，
**任一项变了就自动重算**，避免拿旧缓存配新配置。

## label 的横截面标准化为什么可以先做

`CSZScoreNorm` 是**每天在横截面内**减均值除标准差，只用到当天的信息，
不跨日期，所以不存在前视偏差，可以在切分之前一次做完。
特征的 `RobustZScoreNorm` 就不行——它要估计全局的 median/MAD，
**必须只在训练段上 fit**，所以放在 walk-forward 里逐窗口做。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 215 个因子：3/5/10/20/30/60/120 日窗口（3日~半年，覆盖各主要频段）
DEFAULT_WINDOWS = [3, 5, 10, 20, 30, 60, 120]

# Alpha158 默认是 ["OPEN","HIGH","LOW","VWAP"]。这里**去掉 VWAP**：
# 主数据源（新浪）不提供成交额，$vwap 无法计算。必须与
# `pmsp.build.to_qlib.QLIB_FIELDS` 保持一致，否则 qlib 求值时会因字段
# 不存在而整列 NaN——那比少一个因子更糟（模型会拿到一列常量）。
PRICE_FEATURES = ["OPEN", "HIGH", "LOW"]


def feature_config(windows: list[int] | None = None) -> tuple[list[str], list[str]]:
    """Alpha158 因子配置（窗口可扩）。返回 (表达式列表, 列名列表)。"""
    from qlib.contrib.data.loader import Alpha158DL

    cfg = {
        "kbar": {},
        "price": {"windows": [0], "feature": list(PRICE_FEATURES)},
        "rolling": {"windows": list(windows or DEFAULT_WINDOWS)},
    }
    return Alpha158DL.get_feature_config(cfg)


def ensure_qlib_init(qlib_dir: Path) -> None:
    """幂等地初始化 qlib。重复 init 会打一堆日志并重置缓存配置，所以先查状态。"""
    import qlib
    from qlib.config import C

    if not getattr(C, "registered", False):
        qlib.init(provider_uri=str(qlib_dir), region="cn")


def _cache_key(**kwargs) -> str:
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def cs_zscore_label(label: pd.Series) -> pd.Series:
    """标签的每日横截面 z-score（`CSZScoreNorm`）。

    这一步让 `objective=mse` 的训练严格等价于最大化 IC：
    标签满足 `E[y]=0, E[y²]=1` 后，`min MSE = 1 - IC²`。
    不做这步而直接拟合原始 5 日收益，模型会被高波动日主导，
    去拟合"大盘整体涨跌"——那是横截面常数项，对 IC 的贡献恰为零。
    """
    grp = label.groupby(level="datetime", group_keys=False)
    std = grp.transform("std").replace(0.0, np.nan)
    return (label - grp.transform("mean")) / std


def build_feature_matrix(
    qlib_dir: Path,
    universe: str,
    label_expr: str,
    start_date: str,
    end_date: str,
    windows: list[int] | None = None,
    extra_fields: list[str] | None = None,
    extra_names: list[str] | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """求值全部因子 + 标签，返回 (DataFrame, 特征列名)。带磁盘缓存。

    Parameters
    ----------
    universe : str
        qlib instruments 名（如 `top1500`），对应 `instruments/top1500.txt`。
    extra_fields, extra_names : list | None
        额外因子表达式与列名。**Polymarket 暴露度因子就从这里进来**
        （见 `pmsp.extensions.external_series.pm_feature_config`），
        下游代码完全不需要改。

    Returns
    -------
    (df, feat_cols)
        `df` 含所有特征列 + `LABEL0`（已横截面 z-score）+ `LABEL_RAW`（原始 5 日收益，回测用）。
    """
    from qlib.data import D

    fields, names = feature_config(windows)
    if extra_fields:
        if not extra_names or len(extra_names) != len(extra_fields):
            raise ValueError("extra_fields 与 extra_names 必须一一对应")
        fields, names = fields + list(extra_fields), names + list(extra_names)

    key = _cache_key(
        universe=universe, label=label_expr, start=start_date, end=end_date,
        windows=windows or DEFAULT_WINDOWS, extra=extra_names or [],
    )
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"features_{universe}_{key}.parquet"
        if cache_path.exists() and not force:
            df = pd.read_parquet(cache_path)
            return df, [c for c in df.columns if not c.startswith("LABEL")]

    ensure_qlib_init(qlib_dir)
    instruments = D.instruments(universe)
    raw = D.features(
        instruments, fields + [label_expr], start_time=start_date, end_time=end_date
    )
    raw.columns = [*names, "LABEL_RAW"]
    # 统一成 [date, instruments] 的契约（qlib 原生是 instrument 在前）
    if list(raw.index.names) == ["instrument", "datetime"]:
        raw = raw.swaplevel(0, 1).sort_index()

    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.dropna(subset=["LABEL_RAW"])
    raw["LABEL0"] = cs_zscore_label(raw["LABEL_RAW"])
    raw = raw.dropna(subset=["LABEL0"])

    feat_cols = [c for c in names if raw[c].notna().any()]
    dropped = set(names) - set(feat_cols)
    if dropped:
        print(f"[提示] {len(dropped)} 个因子全为 NaN 已剔除：{sorted(dropped)[:5]} ...")
    out = raw[[*feat_cols, "LABEL0", "LABEL_RAW"]].astype(
        {c: "float32" for c in feat_cols}
    )
    if cache_path is not None:
        out.to_parquet(cache_path)
        print(f"[缓存] 特征矩阵已写入 {cache_path}")
    return out, feat_cols


def daily_returns(qlib_dir: Path, universe: str, start_date: str, end_date: str) -> pd.Series:
    """回测用的单日收益率序列（后复权 close 的 pct_change）。"""
    from qlib.data import D

    ensure_qlib_init(qlib_dir)
    close = D.features(
        D.instruments(universe), ["$close"], start_time=start_date, end_time=end_date
    )["$close"]
    ret = close.groupby(level="instrument", group_keys=False).pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan)
    if list(ret.index.names) == ["instrument", "datetime"]:
        ret = ret.swaplevel(0, 1).sort_index()
    return ret.rename("ret_1d")
