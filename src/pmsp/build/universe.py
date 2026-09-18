"""动态股票池：每月末按过去 N 个交易日的日均成交额取前 `top_n`。

## 为什么不用沪深300 / 中证500

免费档拿不到指数成分股历史（`index_weight` / `index_member` 需 2000 积分），
而 qlib 自带的 `csi300` 文件在下不来的官方数据包里。若拿**当前**成分股回溯套到
历史，那是**幸存者偏差 + 前视偏差**双重污染——股票被调入沪深300 很大程度上
正因为它前面涨了，会系统性虚高 baseline。

"流动性前 N" 纯由 `amount` 算出，point-in-time、零偏差、免费档可得，
是可行的替代口径。`top_n=300` 即一个干净的大盘流动性池。

## 前视偏差的防线

排名只用**截至调仓日（含）**的成交额，成分从**下一个交易日**开始生效。
调仓日当天不能用当天的排名交易——那是前视。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def month_end_rebalance_dates(trade_dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """每个自然月的最后一个交易日。"""
    s = pd.Series(trade_dates, index=trade_dates)
    return list(s.groupby([trade_dates.year, trade_dates.month]).last().sort_values())


def build_universe(
    panel: pd.DataFrame,
    top_n: int,
    lookback_days: int = 60,
    min_listed_days: int = 250,
) -> pd.DataFrame:
    """生成成分股明细表。

    Parameters
    ----------
    panel : pd.DataFrame
        `adjust_panel` 的输出，至少含 `code date amount_yuan`。
    top_n : int
        每期取前多少只。
    lookback_days : int
        成交额均值的回看交易日数。
    min_listed_days : int
        剔除累计交易日数不足此值的次新股。

    Returns
    -------
    pd.DataFrame
        列 `code effective_from effective_to`，其中区间为**闭区间**、
        `effective_from` 为调仓日的下一个交易日。
    """
    df = panel[["code", "date", "amount_yuan"]].dropna(subset=["date"]).copy()
    trade_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    date_pos = {d: i for i, d in enumerate(trade_dates)}

    amt = df.pivot_table(index="date", columns="code", values="amount_yuan", aggfunc="last")
    amt = amt.reindex(trade_dates).sort_index()

    # 累计交易日数（用于剔次新）；停牌日不计入，这正是我们想要的口径
    listed_days = amt.notna().cumsum()
    # 日均成交额：min_periods 取一半，避免刚好卡在回看窗口边缘的股票被整段剔掉
    avg_amt = amt.rolling(lookback_days, min_periods=lookback_days // 2).mean()

    records: list[dict] = []
    rebal_dates = month_end_rebalance_dates(trade_dates)
    for i, rd in enumerate(rebal_dates):
        pos = date_pos[rd]
        if pos + 1 >= len(trade_dates):
            break  # 最后一个调仓日之后没有可交易日，成分无处生效
        eligible = avg_amt.loc[rd].dropna()
        eligible = eligible[listed_days.loc[rd].reindex(eligible.index) >= min_listed_days]
        if eligible.empty:
            continue
        picked = eligible.nlargest(top_n).index

        effective_from = trade_dates[pos + 1]
        if i + 1 < len(rebal_dates):
            effective_to = rebal_dates[i + 1]  # 下个调仓日当天仍属本期成分
        else:
            effective_to = trade_dates[-1]
        records.extend(
            {"code": c, "effective_from": effective_from, "effective_to": effective_to}
            for c in picked
        )

    return pd.DataFrame.from_records(records, columns=["code", "effective_from", "effective_to"])


def to_qlib_instruments(membership: pd.DataFrame) -> pd.DataFrame:
    """把逐期成分合并成连续在池区间。

    qlib 的 instruments 文件允许同一 code 出现多行（多段在池），所以只需把
    **相邻且连续**的月份合并，中断处开新段。不合并也能跑，但文件会大几十倍。
    """
    rows: list[dict] = []
    for code, grp in membership.sort_values(["code", "effective_from"]).groupby("code", sort=True):
        start = end = None
        for frm, to in zip(grp["effective_from"], grp["effective_to"], strict=True):
            if start is None:
                start, end = frm, to
            elif frm <= end + pd.Timedelta(days=7):
                end = max(end, to)  # 与上一段衔接，延长
            else:
                rows.append({"code": code, "start": start, "end": end})
                start, end = frm, to
        if start is not None:
            rows.append({"code": code, "start": start, "end": end})
    return pd.DataFrame.from_records(rows, columns=["code", "start", "end"])


def write_instruments_file(segments: pd.DataFrame, qlib_dir: str | Path, name: str) -> Path:
    """写 qlib `instruments/{name}.txt`（TAB 分隔，无表头）。"""
    out_dir = Path(qlib_dir) / "instruments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.txt"
    lines = [
        f"{r.code}\t{r.start:%Y-%m-%d}\t{r.end:%Y-%m-%d}"
        for r in segments.itertuples(index=False)
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
