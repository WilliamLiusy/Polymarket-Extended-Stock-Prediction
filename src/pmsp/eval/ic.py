"""IC 评估。口径与 qlib `qlib/contrib/eva/alpha.py` 的 `calc_ic` 一致。

## IC 到底怎么算

就是"**按日期分组，在横截面上算相关系数**"，得到一条日度序列。qlib 原文：

    ic  = df.groupby(date_col).apply(lambda d: d["pred"].corr(d["label"]))
    ric = df.groupby(date_col).apply(lambda d: d["pred"].corr(d["label"], method="spearman"))

IC = Pearson（本项目主指标，衡量对 return 本身的预测力），
RankIC = Spearman（副指标，只看排序，抗极端值）。

## 5 日标签的重叠必须修正

相邻交易日的 5 日标签共用 4 天，日度 IC 高度自相关，直接按 N 天算 t 值会
**系统性高估显著性**。这里给两种修正：

* `t_naive`   : ICIR × √(N/h)，把有效样本粗略折算成 N/h。直观，但偏粗。
* `t_nw`      : Newey-West HAC，Bartlett 核、滞后 h-1 阶。**报告以此为准**。

（方案里的功效测算用的是 N/h 近似，所以实际 t 值会与那组数字略有差异。）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATE_LEVEL = "datetime"


def calc_ic_series(
    pred: pd.Series, label: pd.Series, date_col: str = DATE_LEVEL
) -> tuple[pd.Series, pd.Series]:
    """日度 IC / RankIC 序列。

    Parameters
    ----------
    pred, label : pd.Series
        `MultiIndex[datetime, instrument]`。
    """
    df = pd.DataFrame({"pred": pred, "label": label}).dropna()
    grp = df.groupby(level=date_col, group_keys=False)
    ic = grp.apply(lambda d: d["pred"].corr(d["label"]))
    ric = grp.apply(lambda d: d["pred"].corr(d["label"], method="spearman"))
    return ic.dropna(), ric.dropna()


def newey_west_tstat(x: pd.Series | np.ndarray, lags: int) -> tuple[float, float]:
    """序列均值的 Newey-West HAC t 值。

    Returns
    -------
    (t, se) : 均值的 t 统计量与 HAC 标准误。
    """
    a = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = a.size
    if n < 3:
        return float("nan"), float("nan")
    dev = a - a.mean()
    naive_var = float(dev @ dev) / n  # γ_0

    # 序列（数值上）恒定时 t 值没有意义。不拦住的话浮点残差会让 se≈1e-17、
    # t≈1e17，报告里就出现一个看着极显著、实则毫无内容的数字。
    if naive_var <= (1e-10 * max(1.0, abs(float(a.mean())))) ** 2:
        return float("nan"), float("nan")

    var = naive_var
    lags = max(0, min(int(lags), n - 2))
    for j in range(1, lags + 1):
        gamma = float(dev[j:] @ dev[:-j]) / n
        var += 2.0 * (1.0 - j / (lags + 1.0)) * gamma  # Bartlett 权
    # HAC 估计可能为负，也可能因为各滞后项几近抵消而塌到接近 0——
    # 后者同样会把 t 值放大到无意义的量级，所以两种情况都退回朴素方差。
    if var <= 1e-6 * naive_var:
        var = naive_var
    se = np.sqrt(var / n)
    return (float(a.mean() / se) if se > 0 else float("nan")), float(se)


def ic_summary(ic: pd.Series, ric: pd.Series | None = None, horizon: int = 5) -> dict:
    """把日度 IC 序列汇总成报告指标。

    Returns
    -------
    dict
        `ic_mean` 平均 IC —— 最核心的数字，>0.03 在 A 股日频已算不错；
        `ic_std`  IC 的波动，反映稳定性；
        `icir`    = ic_mean / ic_std，信息比率，衡量"每单位不稳定换来多少预测力"；
        `t_nw`    Newey-West t 值，|t| > 2 才算统计显著（**看这个**）；
        `t_naive` ICIR × √(N/h)，粗略对照；
        `ic_win_rate` IC > 0 的天数占比，0.55 以上说明不是靠少数几天撑起来的。
    """
    ic = pd.Series(ic).dropna()
    t_nw, se_nw = newey_west_tstat(ic, lags=horizon - 1)
    n = len(ic)
    icir = float(ic.mean() / ic.std(ddof=1)) if n > 1 and ic.std(ddof=1) > 0 else float("nan")
    out = {
        "n_days": n,
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std(ddof=1)) if n > 1 else float("nan"),
        "icir": icir,
        "t_nw": t_nw,
        "se_nw": se_nw,
        "t_naive": icir * np.sqrt(n / horizon) if np.isfinite(icir) else float("nan"),
        "ic_win_rate": float((ic > 0).mean()),
    }
    if ric is not None:
        ric = pd.Series(ric).dropna()
        out["ric_mean"] = float(ric.mean())
        out["ric_icir"] = (
            float(ric.mean() / ric.std(ddof=1)) if len(ric) > 1 and ric.std(ddof=1) > 0 else float("nan")
        )
        out["ric_t_nw"] = newey_west_tstat(ric, lags=horizon - 1)[0]
        # Pearson 与 Spearman 差太多 = 少数极端收益个股在主导 Pearson IC
        out["ic_ric_gap_warn"] = bool(
            np.isfinite(out["ic_mean"])
            and np.isfinite(out["ric_mean"])
            and abs(out["ic_mean"] - out["ric_mean"]) > 0.5 * max(abs(out["ric_mean"]), 1e-9)
        )
    return out


def paired_ic_test(ic_a: pd.Series, ic_b: pd.Series, horizon: int = 5) -> dict:
    """配对日度 IC 差值检验——**这是判断 Polymarket 有没有增量的那个检验**。

    必须配对：两个模型在同一天面对同一个市场，日度 IC 的共同波动（regime）能
    在差值里抵消掉。分别报两个 IC 均值再肉眼比大小会丢掉这部分方差削减，
    白扔掉一半统计功效。

    Parameters
    ----------
    ic_a : pd.Series
        待检验模型（加了新因子的）的日度 IC。
    ic_b : pd.Series
        基准模型的日度 IC。
    """
    a, b = pd.Series(ic_a).dropna(), pd.Series(ic_b).dropna()
    common = a.index.intersection(b.index)
    if len(common) < 10:
        raise ValueError(f"两条 IC 序列的共同交易日只有 {len(common)} 天，无法检验")
    diff = (a.loc[common] - b.loc[common]).astype(float)
    t_nw, se_nw = newey_west_tstat(diff, lags=horizon - 1)
    from scipy import stats

    n_eff = len(diff) / horizon
    return {
        "n_days": int(len(diff)),
        "n_eff": float(n_eff),
        "ic_a": float(a.loc[common].mean()),
        "ic_b": float(b.loc[common].mean()),
        "delta_ic": float(diff.mean()),
        "se_nw": se_nw,
        "t_nw": t_nw,
        "p_nw": float(2 * stats.norm.sf(abs(t_nw))) if np.isfinite(t_nw) else float("nan"),
        "win_rate": float((diff > 0).mean()),
    }


def detectable_delta(n_days: int, diff_daily_std: float = 0.025, horizon: int = 5,
                     power: float = 0.80, alpha: float = 0.05) -> float:
    """给定 OOS 长度，能检出的最小 ΔIC（功效分析）。

    用来在**动手之前**判断某个测试窗值不值得跑。Polymarket 的可用历史只有
    2024 年起，这个函数量化了由此带来的硬上限。
    """
    from scipy import stats

    se = diff_daily_std / np.sqrt(n_days / horizon)
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)
