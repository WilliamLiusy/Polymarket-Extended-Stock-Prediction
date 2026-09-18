"""复权还原 + 可交易状态标注。

## 为什么不用买 `adj_factor`

tushare `daily` 的 `pre_close` 是**除权后**的昨收盘价，因此
`pct_chg = (close - pre_close) / pre_close * 100` 已经是剔除了送转、分红影响的
**真实收益率**。把它累乘就得到一条总收益指数：

    后复权价_t = 首日收盘价 × Π(1 + pct_chg_i / 100)

所以免费档完全够用，不需要 2000 积分的 `adj_factor`。这条等式的正确性由
`scripts/05_verify.py` 用东方财富 `fqt=2` 序列独立交叉校验。

## 为什么成交量也要复权

`adj_volume = vol / factor`，这样 `adj_close × adj_volume == close × vol`，
即**复权后的成交额恒等于真实成交额**。若不调整，一次 10 送 10 会让成交量凭空
翻倍，把所有滚动量比因子（VMA/VSTD/WVMA 等）在除权日污染掉。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 涨跌幅限制的生效日期（用于近似判定涨跌停；免费档拿不到真实涨跌停价）
_STAR_MARKET_20PCT_FROM = pd.Timestamp("2019-07-22")  # 科创板设立即 ±20%
_CHINEXT_20PCT_FROM = pd.Timestamp("2020-08-24")  # 创业板注册制改革后 ±20%


def _limit_pct_parts(code6: str, market: str, date: pd.Timestamp) -> float:
    """该股票在该日的涨跌幅限制（百分数）。"""
    if market.upper() == "BJ" or code6.startswith(("8", "4", "92")):
        return 30.0
    if code6.startswith("688"):
        return 20.0 if date >= _STAR_MARKET_20PCT_FROM else 10.0
    if code6.startswith("30"):
        return 20.0 if date >= _CHINEXT_20PCT_FROM else 10.0
    return 10.0


def _limit_pct(ts_code: str, date: pd.Timestamp) -> float:
    """tushare 口径代码（`600000.SH`）的涨跌幅限制。"""
    code, _, market = ts_code.partition(".")
    return _limit_pct_parts(code, market, date)


def _limit_pct_qlib(code: str, date: pd.Timestamp) -> float:
    """qlib 口径代码（`SH600000`）的涨跌幅限制。"""
    return _limit_pct_parts(code[2:], code[:2], date)


def to_qlib_code(ts_code: str) -> str:
    """`600000.SH` -> `SH600000`，对齐 qlib 的 instrument 命名。"""
    code, _, market = ts_code.partition(".")
    return f"{market.upper()}{code}"


def adjust_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """把原始面板转成后复权面板。

    Parameters
    ----------
    raw : pd.DataFrame
        `pmsp.datasource.base.RAW_COLUMNS` 的原始数据（tushare 口径：
        `vol` 单位手、`amount` 单位千元）。

    Returns
    -------
    pd.DataFrame
        列：`code date open high low close volume vwap amount_yuan factor
        pct_chg limit_hit suspended_prev`，按 `code, date` 排序。
        价量均为后复权，`amount_yuan` 为真实成交额（元，不复权）。
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["code"] = df["ts_code"].map(to_qlib_code)
    for col in ("open", "high", "low", "close", "pct_chg", "vol", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["code", "date"], ignore_index=True)
    df = df.drop_duplicates(subset=["code", "date"], keep="last", ignore_index=True)

    grp = df.groupby("code", sort=False)

    # 每只股票在本窗口内的首个观测：其 pct_chg 参照的 pre_close 在窗口之外，
    # 计入会让整条复权序列平移，所以置 0（复权因子的绝对水平无意义，只看比值）。
    ret = (df["pct_chg"] / 100.0).fillna(0.0)
    is_first = ~grp.cumcount().astype(bool)
    ret = ret.where(~is_first, 0.0)

    tri = (1.0 + ret).groupby(df["code"], sort=False).cumprod()
    base_close = grp["close"].transform("first")
    adj_close = base_close * tri

    # factor 把不复权价换算成后复权价；对成交量则是反向缩放
    factor = adj_close / df["close"].replace(0.0, np.nan)

    out = pd.DataFrame(
        {
            "code": df["code"],
            "date": df["date"],
            "open": df["open"] * factor,
            "high": df["high"] * factor,
            "low": df["low"] * factor,
            "close": adj_close,
            # vol 单位手 -> 股；除以 factor 保证 close×volume 恒等于真实成交额
            "volume": df["vol"] * 100.0 / factor,
            "factor": factor,
            "pct_chg": df["pct_chg"],
            # amount 单位千元 -> 元。这是真实成交额，不复权，股票池排名用它
            "amount_yuan": df["amount"] * 1000.0,
        }
    )
    # vwap 用真实成交额/真实股数算出真实均价，再乘 factor 保持与其他价格同口径
    real_shares = df["vol"] * 100.0
    out["vwap"] = (out["amount_yuan"] / real_shares.replace(0.0, np.nan)) * factor
    # 全天停牌或零成交时 vwap 无定义，退回收盘价，避免 Alpha158 里成片 NaN
    out["vwap"] = out["vwap"].fillna(out["close"])

    # 涨跌停近似（免费档无真实涨跌停价，用阈值判定；留 0.2pct 余量给四舍五入）
    limits = np.array(
        [_limit_pct(c, d) for c, d in zip(df["ts_code"], out["date"], strict=True)],
        dtype=float,
    )
    out["limit_hit"] = (out["pct_chg"].abs() >= limits - 0.2).fillna(False)

    # 停牌复牌：与上一条观测间隔超过 5 个自然日，视为期间停过牌
    gap_days = out.groupby("code", sort=False)["date"].diff().dt.days
    out["suspended_prev"] = (gap_days > 5).fillna(False)

    return out.sort_values(["code", "date"], ignore_index=True)


def adjust_panel_factor(raw: pd.DataFrame) -> pd.DataFrame:
    """把「未复权价 + 显式复权因子」的面板转成后复权面板（**新浪源走这条**）。

    与 `adjust_panel` 的区别只在复权因子的来源：那边从 `pct_chg` 累乘反推，
    这边直接用数据源给的累计因子。后者更硬——除权日是官方给的，不用从
    收益率序列里猜；两条路的输出 schema 完全一致，下游不用区分。

    Parameters
    ----------
    raw : pd.DataFrame
        `pmsp.datasource.sina_daily.SYMBOL_COLUMNS`：
        `code date open high low close volume adj_factor`。
        价为**未复权**（元），`volume` 单位**股**。

    Returns
    -------
    pd.DataFrame
        与 `adjust_panel` 相同的列，但 **没有 `vwap`**——新浪不提供成交额，
        算不出真实均价。见 `pmsp.build.to_qlib.QLIB_FIELDS` 的说明。
        `amount_yuan` 是 `close × volume` 的**代理**（真实成交额是
        `vwap × volume`，两者只差 `vwap/close`，对流动性**排名**影响可忽略）。
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume", "adj_factor"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["code", "date"], ignore_index=True)
    df = df.drop_duplicates(subset=["code", "date"], keep="last", ignore_index=True)
    df = df.dropna(subset=["close", "adj_factor"])
    # 四个价必须同时为正。只卡 close 不够：新浪对**北交所股票的新三板时期**
    # 会返回 open=high=low=close=0（成交量非 0）的行——协议转让没有连续
    # 竞价价格。这类行留下来会让 KMID/KLEN 之类的 K 线因子除零。
    pos = (df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df["adj_factor"] > 0)
    n_bad = int((~pos).sum())
    df = df[pos].reset_index(drop=True)
    if n_bad:
        print(f"[复权] 剔除 {n_bad:,} 行非正价格（多为北交所股票的新三板时期）")

    factor = df["adj_factor"]
    out = pd.DataFrame(
        {
            "code": df["code"],
            "date": df["date"],
            "open": df["open"] * factor,
            "high": df["high"] * factor,
            "low": df["low"] * factor,
            "close": df["close"] * factor,
            # 除以 factor 保证 close×volume 恒等于真实成交额（复权不变量）
            "volume": df["volume"] / factor,
            "factor": factor,
            # 真实成交额的代理，不复权。股票池排名用它
            "amount_yuan": df["close"] * df["volume"],
        }
    )

    # 真实涨跌幅 = 后复权价的日收益（除权影响已在 factor 里剔除）
    out["pct_chg"] = out.groupby("code", sort=False)["close"].pct_change() * 100.0

    limits = np.array(
        [_limit_pct_qlib(c, d) for c, d in zip(out["code"], out["date"], strict=True)],
        dtype=float,
    )
    out["limit_hit"] = (out["pct_chg"].abs() >= limits - 0.2).fillna(False)

    gap_days = out.groupby("code", sort=False)["date"].diff().dt.days
    out["suspended_prev"] = (gap_days > 5).fillna(False)

    return out.sort_values(["code", "date"], ignore_index=True)


def compare_adjustment(ours: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """把我们还原的后复权序列与外部参照（东方财富 `fqt=2`）对比。

    比的是**日收益率序列的相关性**而非价格水平——复权因子的绝对水平是任意的
    （取决于起点），只有收益率序列才有可比性。

    Returns
    -------
    dict
        `n_overlap` 重叠交易日数、`ret_corr` 收益率相关系数、
        `max_abs_ret_diff` 单日收益率最大绝对偏差。
    """
    a = ours[["date", "close"]].dropna().set_index("date")["close"].sort_index()
    b = reference.copy()
    b["date"] = pd.to_datetime(b["trade_date"], format="%Y%m%d")
    b = b[["date", "close"]].dropna().set_index("date")["close"].sort_index()

    ra, rb = a.pct_change().dropna(), b.pct_change().dropna()
    common = ra.index.intersection(rb.index)
    if len(common) < 30:
        return {"n_overlap": len(common), "ret_corr": np.nan, "max_abs_ret_diff": np.nan}
    ra, rb = ra.loc[common], rb.loc[common]
    return {
        "n_overlap": int(len(common)),
        "ret_corr": float(ra.corr(rb)),
        "max_abs_ret_diff": float((ra - rb).abs().max()),
    }
