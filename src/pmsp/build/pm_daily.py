"""把 Polymarket 的小时面板对齐成**某个股票市场口径下的**日频因子。

## 为什么必须显式指定"截断时刻"

Polymarket 24/7 交易，股票市场不是。如果直接拿"某个自然日的全部成交"当成
当日因子，再去跟当日股票收益算相关，就会用到收盘之后才到达的信息——
那不是关联性，那是前视偏差。

所以对齐必须围绕一个**明确的时钟**展开。两个口径：

    A 股   截到 07:00 UTC   = 北京时间 15:00 收盘
    美股   截到 20:00 UTC   = 美东 16:00 收盘（夏令时；冬令时为 21:00 UTC）

美股这里刻意**不做夏/冬令时切换**，统一用 20:00 UTC。理由：冬令时下 20:00 UTC
= 美东 15:00，比收盘早一小时，因此是**更保守**的截断——少用一小时信息，
绝不会多用。反过来（统一 21:00 UTC）在夏令时就会越过收盘线，那是不能接受的。

## 因子窗口 = 两次"可行动时刻"之间

日期 d 的因子，用的是 `(d-1 的截断时刻, d 的截断时刻]` 这个区间内到达的信息。
这样定义的好处是**信息不重不漏**：每一笔成交恰好被算进一个交易日，且都在
该交易日可行动时刻之前。周末与节假日的成交自动并入下一个交易日，这正是
"周一开盘要消化整个周末的消息"的现实。

## 概率水平用 ffill，但有失效上限

一个市场可能几天没有任何成交。此时"最后一个可得报价"仍然是它当下的最优
估计（预测市场的报价不会因为没人交易就失效），所以概率水位用 ffill。
但 ffill 不能无限：一个 30 天没人碰的市场，它的报价已经不含当下信息了，
继续 ffill 会制造一条平坦的假序列，还会让 Δp 在下一次成交时爆出一个假的
大跳变。所以 `max_stale_days` 之外置 NaN。

成交额 / 笔数 / OFI 这些**流量**量不能 ffill —— 没成交就是 0，ffill 会把
一天的成交额复制成一周的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 各市场的收盘截断时刻（UTC 整点）。见模块文档对美股为何不切夏令时的说明。
CUTOFF_UTC_HOUR = {"cn": 7, "us": 20}


def _as_of_index(calendar: pd.DatetimeIndex, cutoff_hour: int) -> pd.DatetimeIndex:
    """交易日 -> 该日的"可行动时刻"（UTC）。"""
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).tz_localize(None).normalize().unique()
    return (cal + pd.Timedelta(hours=cutoff_hour)).tz_localize("UTC")


def assign_trading_day(
    hourly: pd.DataFrame, calendar: pd.DatetimeIndex, market: str = "cn"
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """给每个小时打上它所属的交易日标签，并返回交易日索引。

    这是整条链路上**最容易埋前视**的一步，所以只允许有一份实现：
    `to_daily` 和 `pmsp.build.oil_ladder.daily_curves` 都调它。写成两份的话，
    某天有人改了其中一份的边界约定，另一份悄悄跟着错，而且两边跑出来的
    结果各自看着都正常。

    每小时落到哪个交易日：hour 属于 `(as_of[i-1], as_of[i]]` -> 交易日 i。
    `searchsorted` 用 `"left"` 使得恰好等于 `as_of[i]` 的小时归入交易日 i
    （小时标签 H 代表 `[H, H+1h)` 这根 K 线……注意标签恰为 as_of 的那根
    其实横跨收盘，但它的成交额大头在收盘前，且这里取的是该窗口的 `last`，
    相差一小时的口径误差远小于隔夜，保留在同一日）。

    用真实交易日历而不是 `floor("D")`：后者会造出周末和节假日这些
    **股票根本没法交易**的"交易日"，而 Polymarket 是 7×24 的，那些日子上
    会长出一堆无法对齐的概率变动。
    """
    cutoff = CUTOFF_UTC_HOUR[market]
    as_of = _as_of_index(calendar, cutoff)
    if len(as_of) < 2:
        raise ValueError("交易日历至少需要两天")
    h = hourly.copy()
    h["hour"] = pd.to_datetime(h.hour, utc=True)
    h = h.sort_values("hour")
    pos = np.searchsorted(as_of.values, h.hour.values, side="left")
    h = h[pos < len(as_of)].copy()
    pos = pos[pos < len(as_of)]
    h["date"] = as_of[pos].tz_convert(None).normalize()
    return h, pd.DatetimeIndex(as_of.tz_convert(None).normalize())


def to_daily(
    hourly: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    market: str = "cn",
    max_stale_days: int = 10,
    min_window_vol: float = 0.0,
) -> pd.DataFrame:
    """小时面板 -> (交易日 × condition_id) 的日频因子面板。

    Parameters
    ----------
    hourly : pd.DataFrame
        `pmsp.datasource.hf_polymarket.build_hourly_panel` 的产物。
    calendar : pd.DatetimeIndex
        目标股票市场的交易日历（只用它的日期部分）。
    market : {"cn", "us"}
        决定截断时刻。
    max_stale_days : int
        概率水位允许 ffill 的最长自然日数，超过置 NaN。
    min_window_vol : float
        窗口内成交额低于此值时，把**流量类**因子保留但把 `dp` 置 NaN——
        几十美元的成交撑起来的"概率变动"是噪声，不是信息。默认 0（不过滤），
        由调用方按市场规模决定阈值。

    Returns
    -------
    pd.DataFrame
        列：date, condition_id, p_level, dp, dp_norm, vol_usdc, ofi_usdc,
        ofi_share, n_trades, rvol, stale_days
    """
    h, dates = assign_trading_day(hourly, calendar, market)

    grp = h.groupby(["date", "condition_id"], sort=True)
    flow = grp.agg(
        vol_usdc=("vol_usdc", "sum"),
        ofi_usdc=("ofi_usdc", "sum"),
        n_trades=("n_trades", "sum"),
        p_last=("p_close", "last"),
        p_first=("p_open", "first"),
        p_high=("p_high", "max"),
        p_low=("p_low", "min"),
        n_hours=("p_close", "size"),
    ).reset_index()

    # 日内实现波动：Parkinson 型，用小时级 high/low 的极差。概率有界于 [0,1]，
    # 所以这里用**绝对**极差而不是对数极差——0.02→0.04 的对数极差看着很大，
    # 实际只是 2 个百分点的消息量。
    flow["rvol"] = (flow.p_high - flow.p_low) / np.sqrt(4.0 * np.log(2.0))
    # OFI 的量纲是美元，跨市场不可比；除以成交额得到 [-1, 1] 的"单边压力占比"
    flow["ofi_share"] = flow.ofi_usdc / flow.vol_usdc.where(flow.vol_usdc > 0)

    # 补全 (交易日 × 市场) 的完整网格，才能 ffill 概率水位、给流量填 0
    # （`dates` 由 assign_trading_day 一并返回，与 `h["date"]` 同源）
    out_parts = []
    for cid, g in flow.groupby("condition_id", sort=False):
        g = g.set_index("date").reindex(dates)
        g["condition_id"] = cid
        # 流量：没成交 = 0
        for col in ["vol_usdc", "ofi_usdc", "n_trades"]:
            g[col] = g[col].fillna(0.0)
        traded = g.n_trades > 0
        # 概率水位：只向后填充。绝不 bfill/插值——那是把未来搬到过去。
        g["p_level"] = g.p_last.ffill()
        # 距离最后一次成交多少天（用于失效判断）
        last_traded = pd.Series(np.where(traded, np.arange(len(g)), np.nan), index=g.index).ffill()
        g["stale_days"] = np.arange(len(g)) - last_traded
        g.loc[g.stale_days > max_stale_days, "p_level"] = np.nan
        # 截掉该市场首次成交之前的整段（reindex 出来的前缀全是 NaN）
        if traded.any():
            g = g.loc[traded.idxmax() :]
        out_parts.append(g.reset_index(names="date"))

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values(["condition_id", "date"])

    # Δp：相邻交易日的概率水位之差。用**绝对差**而非相对变化——概率从 0.02
    # 涨到 0.04 相对变化 +100%，但实际只是 2 个百分点的消息量。这与
    # `pmsp.extensions.external_series` 里的口径一致。
    out["dp"] = out.groupby("condition_id").p_level.diff()
    # 归一化版本：除以该市场 Δp 的滚动标准差，让不同活跃度的市场可比。
    # 滚动窗口只用过去（min_periods 保证前期不出数），不引入未来信息。
    roll = out.groupby("condition_id").dp.transform(
        lambda s: s.shift(1).rolling(60, min_periods=20).std()
    )
    out["dp_norm"] = out.dp / roll.where(roll > 0)

    if min_window_vol > 0:
        thin = out.vol_usdc < min_window_vol
        out.loc[thin, ["dp", "dp_norm"]] = np.nan

    cols = [
        "date", "condition_id", "p_level", "dp", "dp_norm", "vol_usdc",
        "ofi_usdc", "ofi_share", "n_trades", "rvol", "stale_days",
    ]
    return out[cols].reset_index(drop=True)


def aggregate_theme(
    daily: pd.DataFrame, weight: str = "vol_usdc", min_markets: int = 1
) -> pd.DataFrame:
    """把多个同主题市场合成一条"主题指数"序列。

    为什么需要这一步：Polymarket 的市场是**按到期日切碎**的
    （`...-before-july`、`...-in-2025`、`...-before-september` 是三个市场，
    问的是同一件事）。逐个市场算相关会把样本切成一堆几十天的碎片，每一段
    都没有统计功效；而且同一天有多个市场同时活跃时，逐个算等于重复计数。

    合成用**成交额加权**而不是等权：`will-iran-close-the-strait-of-hormuz-before-july`
    的成交额是 73.9 万美元，`will-opec-hike-production-by-next-meeting` 是 410 美元。
    等权会让后者的噪声与前者的信息同权。

    Δp 一侧的加权用**当日两个市场都有成交**的成交额；概率水位一侧同理。
    """
    d = daily.copy()
    w = d[weight].astype(float).clip(lower=0.0)

    def wmean(col: str) -> pd.Series:
        v = d[col].astype(float)
        ok = v.notna() & (w > 0)
        num = (v.where(ok, 0.0) * w.where(ok, 0.0)).groupby(d.date).sum()
        den = w.where(ok, 0.0).groupby(d.date).sum()
        return num / den.where(den > 0)

    out = pd.DataFrame(
        {
            "p_level": wmean("p_level"),
            "dp": wmean("dp"),
            "dp_norm": wmean("dp_norm"),
            "ofi_share": wmean("ofi_share"),
            "rvol": wmean("rvol"),
        }
    )
    out["vol_usdc"] = d.groupby("date")[weight].sum()
    out["ofi_usdc"] = d.groupby("date").ofi_usdc.sum()
    out["n_trades"] = d.groupby("date").n_trades.sum()
    out["n_markets"] = d[d.n_trades > 0].groupby("date").condition_id.nunique().reindex(out.index).fillna(0)
    out = out[out.n_markets >= min_markets] if min_markets > 1 else out
    return out.reset_index(names="date")
