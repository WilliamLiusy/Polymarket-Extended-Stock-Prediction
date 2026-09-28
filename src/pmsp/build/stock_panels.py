"""把 `data/raw/` 里的日线拼成本分析要的收益面板与行业指数。

三件事，每件都有一个容易做错的地方：

**收益率从哪个价格算。**A 股这边 `close` 已经是复权价（`02_build_qlib_data.py`
里过了 `adjust_panel_factor`，`factor` 列只是留档），直接取 `pct_change` 即可。
美股这边是**未复权**收盘价，除息日会多出一个约 −0.5% 的跳空；这件事的方向是
保守的（噪声进分母，相关系数向零偏），理由见 `pmsp.datasource.sina_global`
的模块文档，这里不再处理。

**行业指数怎么加权。**用**等权**，不是市值权。市值权会让中国国航一只股票
主导整个"航空指数"，而本分析要检验的是"航空业整体对油价的暴露"，不是
"最大那只航空股的走势"。等权的代价是小盘股噪声占比更高，这个代价可以接受——
噪声让结论向零偏，不会造假。

**指数的日收益不等于成分股收盘价之和的变化率。**必须先算个股日收益再按日
横截面平均。用"成分股价格等权平均"再求变化率是错的：那等于按价格高低隐式
加权，还会在成分股停牌/缺失时产生跳空。这个坑在
`scripts/05_verify.py` 的基准检验里已经踩过一次（缺口假收益把全市场等权
基准抬高了 19pp/年）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")

#: 单日收益绝对值超过这个数就当异常剔除。A 股有 10% 涨跌停（创业板 20%），
#: 美股航空股单日 25% 在样本期内只出现在 2020 年，本分析窗口内没有；
#: 期货连续合约换月跳空可以到 8% 以上（见 sina_global 的说明）。
RET_CLIP = {"cn": 0.22, "us": 0.30, "futures": 0.08}


def _returns(df: pd.DataFrame, key: str, kind: str) -> pd.DataFrame:
    """长表 -> (date × key) 的日收益宽表。"""
    d = df.sort_values([key, "date"]).copy()
    wide = d.pivot_table(index="date", columns=key, values="close", aggfunc="last")
    # `fill_method=None` 是关键：pandas 的默认 `'pad'` 会把缺失收盘价前向填充
    # 之后再算变化率，于是停牌/缺数据的那一天变成一个假的 0 收益，复牌当天
    # 变成一个假的大收益。这正是 `scripts/05_verify.py` 里修过的"缺口假收益"
    # （当时把全市场等权基准抬高了 19pp/年）。缺就是缺，留 NaN。
    ret = wide.pct_change(fill_method=None)
    cap = RET_CLIP[kind]
    # 超限值置 NaN 而不是截断到 ±cap：截断会保留一个假的大收益，
    # 置 NaN 只是少用一天，后者才是"不知道"的正确表示。
    return ret.where(ret.abs() <= cap)


def cn_returns() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """A 股：(航空股收益面板, 能源股收益面板, 名称映射)。列名为股票代码。"""
    air = pd.read_parquet(RAW / "cn_airlines.parquet")
    eng = pd.read_parquet(RAW / "cn_energy.parquet")
    names = pd.concat([air[["code", "name"]], eng[["code", "name"]]]).drop_duplicates()
    return (
        _returns(air, "code", "cn"),
        _returns(eng, "code", "cn"),
        names.set_index("code").name,
    )


def us_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """美股：(航空股收益面板, 对照组收益面板)。列名为 ticker。"""
    air = pd.read_parquet(RAW / "us_airlines.parquet")
    ctl = pd.read_parquet(RAW / "us_controls.parquet")
    return _returns(air, "symbol", "us"), _returns(ctl, "symbol", "us")


def oil_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """原油：(收盘价宽表, 日收益宽表)。列名为 CL / OIL。"""
    fut = pd.read_parquet(RAW / "oil_futures.parquet")
    px = fut.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    return px, _returns(fut, "symbol", "futures")


def equal_weight_index(ret_panel: pd.DataFrame, min_names: int = 3) -> pd.Series:
    """等权行业指数的**日收益**序列（先算个股收益，再按日横截面平均）。

    `min_names`：当日有效成分不足这个数就置 NaN。一只股票撑起来的
    "行业指数"不是行业指数。
    """
    n = ret_panel.notna().sum(axis=1)
    return ret_panel.mean(axis=1).where(n >= min_names)


def cum_index(ret: pd.Series, base: float = 100.0) -> pd.Series:
    """日收益 -> 累计指数（基期 = base）。用于时序图里跨量纲的同轴对比。"""
    r = ret.fillna(0.0)
    return base * (1.0 + r).cumprod()


def trading_calendar(*panels: pd.DataFrame) -> pd.DatetimeIndex:
    """多个面板的交易日**交集**。

    用交集而不是并集：并集会引入某一市场休市的日子，那些日子在该市场的
    收益序列里是 NaN，ffill 之后会变成假的 0 收益。这个坑
    `pmsp.extensions.external_series` 的文档里记过（`dump_bin` 的日历并集问题）。
    """
    idx = None
    for p in panels:
        cur = pd.DatetimeIndex(p.index)
        idx = cur if idx is None else idx.intersection(cur)
    return pd.DatetimeIndex(sorted(idx)) if idx is not None else pd.DatetimeIndex([])


def winsorize(s: pd.Series, q: float = 0.005) -> pd.Series:
    """双侧缩尾。只用于**相关性**的稳健性对照，主结果不缩尾。

    缩尾会削掉极端值，而本分析里的信号恰恰集中在极端日（地缘冲突那几天）。
    所以它的角色是反向检验：如果主结果在缩尾后消失，说明结论完全由几个
    极端点撑着，那就该报告"效应只存在于极端事件中"，而不是报告一个全样本相关。
    """
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def zscore(s: pd.Series, window: int = 250, min_periods: int = 120) -> pd.Series:
    """滚动 z-score，只用过去（shift(1) 后再滚动）。"""
    m = s.shift(1).rolling(window, min_periods=min_periods).mean()
    sd = s.shift(1).rolling(window, min_periods=min_periods).std()
    return (s - m) / sd.where(sd > 0)


def residualize(y: pd.Series, x: pd.Series, window: int = 250,
                min_periods: int = 120) -> pd.Series:
    """把 `y` 对 `x` 做滚动回归后取残差（剔除大盘/行业共同成分）。

    为什么必须做这一步：航空股与大盘的相关系数在 0.6 以上，而地缘风险概率
    同时也是**全市场**的风险指标。不剔掉大盘成分，"风险概率上升 ↔ 航空股跌"
    很可能只是"风险上升 ↔ 大盘跌"的影子，与燃油成本无关。

    滚动 β 而不是全样本 β：全样本 β 用到了未来数据。
    """
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    cov = df.y.rolling(window, min_periods=min_periods).cov(df.x)
    var = df.x.rolling(window, min_periods=min_periods).var()
    beta = (cov / var.where(var > 0)).shift(1)
    return (df.y - beta * df.x).reindex(y.index)


def describe_panel(ret_panel: pd.DataFrame, names: pd.Series | None = None) -> pd.DataFrame:
    """面板的基础统计，用于报告里"数据长什么样"那一节。"""
    rows = []
    for col in ret_panel.columns:
        s = ret_panel[col].dropna()
        if s.empty:
            continue
        rows.append({
            "代码": col,
            "名称": (names.get(col, "") if names is not None else ""),
            "天数": len(s),
            "起": s.index.min().date(),
            "止": s.index.max().date(),
            "年化收益": float((1 + s).prod() ** (252 / len(s)) - 1),
            "年化波动": float(s.std(ddof=1) * np.sqrt(252)),
        })
    return pd.DataFrame(rows)
