"""分层回测 / 多空组合 / 换手与交易成本。

## 5 日持仓怎么每天都有仓位

信号每天都产生，但每笔持有 5 天，所以用**重叠分批**（overlapping sleeves）：
每天只调 1/5 的仓位，整个组合等于 5 个错开一天的子组合的平均。这是重叠预测
期的标准做法，比"每 5 天全量换一次"更贴近实盘，换手率也更真实。

## 时间线必须和标签口径严格对齐

标签是 `Ref($close,-6)/Ref($close,-1)-1`：T 日收盘出信号 → T+1 收盘成交 →
T+6 收盘卖出。所以 T 日的信号赚的是 **T+2 到 T+6** 这 5 个交易日的日收益
（T+1 到 T+2 是第一段盈亏）。

于是 d 日的持仓来自 T ∈ [d-6, d-2] 的信号，即 `W.shift(2).rolling(5).mean()`。
**错一天就是前视偏差**，会把结果显著做高。

## 关于多空组合

多空是**因子有效性的诊断工具，不是可实盘的策略**：A 股融券券源少、成本高、
小盘股基本借不到。所以多空曲线只用来看因子的横截面区分度；能落地的看
`top_group`（最高分组相对全市场等权的超额）。

成本上，多空两条腿的调仓费用是**相加**的（不是相减）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 243  # A 股年均交易日


def as_panel(s: pd.Series) -> pd.Series:
    """把 MultiIndex 统一成 `(datetime, instrument)` 顺序。

    qlib `D.features` 返回的是 `(instrument, datetime)`，而本项目的数据契约是
    `[date, instruments]`。不统一就会按位置取错轴——症状是把股票代码当日期解析。
    """
    if not isinstance(s.index, pd.MultiIndex) or s.index.nlevels != 2:
        raise ValueError(f"需要两层 MultiIndex(datetime, instrument)，收到 {s.index}")
    names = list(s.index.names)
    if names == ["instrument", "datetime"]:
        s = s.swaplevel(0, 1)
    elif names != ["datetime", "instrument"]:
        # 索引没有标准命名时，用"哪一层是时间类型"来判断
        lvl0_is_time = pd.api.types.is_datetime64_any_dtype(s.index.get_level_values(0))
        lvl1_is_time = pd.api.types.is_datetime64_any_dtype(s.index.get_level_values(1))
        if lvl1_is_time and not lvl0_is_time:
            s = s.swaplevel(0, 1)
        elif not lvl0_is_time:
            raise ValueError(f"两层索引都不像日期，无法判断顺序：names={names}")
        s = s.rename_axis(["datetime", "instrument"])
    return s.sort_index()


def _target_weights(pred: pd.Series, n_groups: int, group: int) -> pd.DataFrame:
    """某一分层的每日目标权重矩阵（date × code），组内等权、每日和为 1。"""
    df = pred.dropna().rename("pred").reset_index()
    date_col, inst_col = "datetime", "instrument"
    # rank(pct=True) 后切等分位；group 0 = 预测最低组，n_groups-1 = 最高组
    pct = df.groupby(date_col)["pred"].rank(pct=True, method="first")
    bucket = np.minimum((pct * n_groups).astype(int), n_groups - 1)
    sel = df[bucket == group]
    if sel.empty:
        return pd.DataFrame()
    w = sel.assign(w=1.0).pivot_table(index=date_col, columns=inst_col, values="w", aggfunc="last")
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)


def _hold_from_target(
    weights: pd.DataFrame, calendar: pd.DatetimeIndex, horizon: int = 5, exec_lag: int = 2
) -> pd.DataFrame:
    """目标权重 -> 实际持仓（重叠分批）。见模块 docstring 的时间线说明。"""
    w = weights.reindex(calendar).fillna(0.0)
    return w.shift(exec_lag).rolling(horizon, min_periods=1).mean().fillna(0.0)


def _perf_stats(ret: pd.Series) -> dict:
    """年化收益 / 波动 / Sharpe / 最大回撤 / Calmar。"""
    r = pd.Series(ret).dropna()
    if r.empty:
        return dict.fromkeys(["ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar"], float("nan"))
    nav = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS
    ann_ret = float(nav.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and nav.iloc[-1] > 0 else float("nan")
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    mdd = float((nav / nav.cummax() - 1.0).min())
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 and np.isfinite(ann_ret) else float("nan"),
        "max_drawdown": mdd,
        "calmar": float(ann_ret / abs(mdd)) if mdd < 0 and np.isfinite(ann_ret) else float("nan"),
    }


def quantile_backtest(
    pred: pd.Series,
    ret_1d: pd.Series,
    n_groups: int = 10,
    horizon: int = 5,
    cost: dict | None = None,
) -> dict:
    """分层回测。

    Parameters
    ----------
    pred : pd.Series
        `MultiIndex[datetime, instrument]` 的模型打分。
    ret_1d : pd.Series
        同索引的**单日**收益率（后复权 close 的 pct_change）。
    n_groups : int
        分层数，默认 10（十分组）。
    cost : dict | None
        `commission_bps` / `stamp_duty_bps` / `impact_bps`。None 则只出税前结果。

    Returns
    -------
    dict
        `group_ann_return` 各层年化（看**单调性**）、`long_short` 多空组合指标、
        `top_group` 最高分组指标、`daily` 逐日收益与换手明细。
    """
    pred = as_panel(pred)
    ret_1d = as_panel(ret_1d)

    ret_mat = ret_1d.dropna().unstack(level="instrument")
    calendar = pd.DatetimeIndex(sorted(ret_mat.index))
    ret_mat = ret_mat.reindex(calendar)
    # 「当天不在池/无数据」与「当天收平」都会是 0，必须在填充**之前**留下掩码，
    # 否则算等权基准时只能用 `!= 0` 区分，会把真实收平的日子也当成空缺剔除，
    # 分母缩小 → 基准被系统性抬高（实测约 +1.2 个百分点/年）。
    present = ret_mat.notna()
    ret_mat = ret_mat.fillna(0.0)

    cost = cost or {}
    buy_bps = cost.get("commission_bps", 0.0) + cost.get("impact_bps", 0.0)
    sell_bps = buy_bps + cost.get("stamp_duty_bps", 0.0)

    holds: dict[int, pd.DataFrame] = {}
    gross: dict[int, pd.Series] = {}
    fees: dict[int, pd.Series] = {}
    net: dict[int, pd.Series] = {}
    turnover: dict[int, pd.Series] = {}

    for g in range(n_groups):
        w = _target_weights(pred, n_groups, g)
        if w.empty:
            continue
        h = _hold_from_target(w, calendar, horizon=horizon).reindex(columns=ret_mat.columns).fillna(0.0)
        holds[g] = h
        gross[g] = (h * ret_mat).sum(axis=1)

        delta = h.diff().fillna(h)
        buys = delta.clip(lower=0.0).sum(axis=1)
        sells = (-delta.clip(upper=0.0)).sum(axis=1)
        turnover[g] = buys  # 单边换手
        fees[g] = (buys * buy_bps + sells * sell_bps) / 1e4
        net[g] = gross[g] - fees[g]

    if not gross:
        raise ValueError("没有任何分层产生持仓，检查 pred 与 ret_1d 的索引是否对得上")

    lo, hi = min(gross), max(gross)
    ls_gross = gross[hi] - gross[lo]
    # 多空组合的成本是两条腿**相加**：空头腿的调仓一样要付费。
    # 写成 net[hi] - net[lo] 会把空头腿的成本变成收益，使净值高于税前。
    ls_net = ls_gross - fees[hi] - fees[lo]
    # 全市场等权基准，用于算 Top 组超额。
    # **必须和分层组用同一套执行口径**（T+1 买、T+6 卖、5 日重叠分批），否则比的
    # 是两件事：基准每日全额再平衡，而组合每天只调 1/5 仓。口径不一致时，基准会
    # 白拿一份再平衡收益，Top 组超额随之被系统性低估。
    w_uni = present.astype(float)
    w_uni = w_uni.div(w_uni.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    h_uni = _hold_from_target(w_uni, calendar, horizon=horizon)
    eq_weight = (h_uni * ret_mat).sum(axis=1)

    return {
        "n_groups": n_groups,
        "group_ann_return": {g: _perf_stats(gross[g])["ann_return"] for g in sorted(gross)},
        "monotonicity_spearman": float(
            pd.Series({g: _perf_stats(gross[g])["ann_return"] for g in sorted(gross)})
            .rank()
            .corr(pd.Series(sorted(gross), index=sorted(gross)).rank(), method="spearman")
        ),
        "long_short": {"gross": _perf_stats(ls_gross), "net": _perf_stats(ls_net)},
        "top_group": {
            "gross": _perf_stats(gross[hi]),
            "net": _perf_stats(net[hi]),
            "excess_vs_eqw": _perf_stats(gross[hi] - eq_weight),
            # 年换手 34 倍，税前超额没有决策意义。基准按惯例不计费（等权买入持有
            # 的换手远低于组合），所以「税后组合 − 税前基准」就是可落地的超额。
            "excess_vs_eqw_net": _perf_stats(net[hi] - eq_weight),
        },
        "avg_daily_turnover_one_way": float(turnover[hi].mean()),
        "daily": pd.DataFrame(
            {
                "top_gross": gross[hi],
                "top_net": net[hi],
                "ls_gross": ls_gross,
                "ls_net": ls_net,
                "eq_weight": eq_weight,
                "top_turnover": turnover[hi],
            }
        ),
    }


def cost_sensitivity(
    pred: pd.Series, ret_1d: pd.Series, cost: dict, multipliers=(0.0, 0.5, 1.0, 2.0), **kwargs
) -> pd.DataFrame:
    """成本敏感性：把费率整体缩放，看多空 Sharpe 什么时候归零。

    5 日持仓的日均单边换手约 1/5，一年换手约 50 倍，成本很容易吃掉全部超额，
    所以这张表比单一的"扣费后"数字更有信息量。
    """
    rows = []
    for m in multipliers:
        scaled = {k: v * m for k, v in cost.items()}
        res = quantile_backtest(pred, ret_1d, cost=scaled, **kwargs)
        rows.append(
            {
                "cost_multiplier": m,
                "ls_sharpe_net": res["long_short"]["net"]["sharpe"],
                "ls_ann_return_net": res["long_short"]["net"]["ann_return"],
                "top_ann_return_net": res["top_group"]["net"]["ann_return"],
            }
        )
    return pd.DataFrame(rows)
