"""关联性检验：Polymarket 概率序列 ↔ 航空股日频收益。

## 为什么不能只报一个相关系数

"有没有关联"这个问题在日频金融数据上有四种问法，答案可以互相矛盾，
而只报一个数就等于替读者挑了一个最好看的：

1. **同期相关**（`lead_lag`，k=0）——两者当天一起动吗？这只能说"同时反应
   同一批消息"，**不含任何预测含义**，而且在 A 股口径下它天然受限：
   北京 15:00 收盘之后到达的油价消息，明天才能体现在股价上。
2. **领先滞后相关**（`lead_lag`，k≠0）——谁先动？k>0 是"概率变动领先股价"
   （有预测价值），k<0 是"股价领先概率"（Polymarket 在跟随股市，没有增量）。
   这一条是本分析里信息量最大的。
3. **横截面 IC**（`cross_sectional_ic`）——同一天，油价敏感度高的航空股是不是
   真的跌得更多？这是本项目的主指标口径，也是唯一一个**宏观序列能对
   横截面排序产生贡献**的问法（见 `pmsp.extensions.external_series` 的论证：
   一个对所有股票都相同的宏观数，对截面相关的贡献恰为零，必须乘上因股而异
   的暴露度）。
4. **事件研究**（`event_study`）——只在概率大幅跳变的那些天，股价有没有
   系统性反应？日频相关系数会被 95% 的"什么都没发生"的日子稀释掉，
   而这类信号本来就集中在少数几天。

## 三角验证是必须的

直接检验"Polymarket 油价概率 → 航空股收益"有一个致命的解释困难：结果为零时
分不清是**Polymarket 没信息**还是**燃油成本这个传导渠道不通**。所以必须
把链路拆成两段分别验（`triangle`）：

    Δp  →  原油期货收益      Polymarket 的概率到底含不含油价信息？
    原油收益 → 航空股收益    燃油成本渠道在样本期内通不通？
    Δp  →  航空股收益        端到端

只有在前两段都显著的前提下，第三段的零结果才能解读为"预测市场的信息已经被
油价本身完全吸收，没有额外的股票层面增量"。若第一段就不显著，那整件事在
数据层面就到此为止了——这个结论同样有价值，但话要说对。

## 所有 t 值都用 Newey-West

日频重叠、事件聚集（一场中东冲突里连续十几天都在动）都会让残差强烈自相关。
朴素 t 值在这种数据上高估显著性能到两三倍。滞后阶数默认取 5（一周），
复用 `pmsp.eval.ic.newey_west_tstat`。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .ic import newey_west_tstat


def _hac_regression(
    y: np.ndarray, X: np.ndarray, lags: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """带 Newey-West HAC 标准误的 OLS。X 需已含常数列。

    Returns
    -------
    (beta, se, tstat, r2, n)
    """
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if n <= k + 2:
        nan = np.full(k, np.nan)
        return nan, nan, nan, float("nan"), n
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    # HAC 的"三明治"中层：S = Σ_j w_j (Γ_j + Γ_j')，Bartlett 权
    u = X * resid[:, None]
    S = u.T @ u
    lags = max(0, min(int(lags), n - k - 1))
    for j in range(1, lags + 1):
        G = u[j:].T @ u[:-j]
        S += (1.0 - j / (lags + 1.0)) * (G + G.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / np.where(se > 0, se, np.nan)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / tss if tss > 0 else float("nan")
    return beta, se, t, r2, n


def corr_with_t(x: pd.Series, y: pd.Series, lags: int = 5) -> dict:
    """两条日频序列的相关系数 + HAC t 值 + Spearman 对照。

    相关系数的 t 值不用教科书的 `r√(n-2)/√(1-r²)`——那个公式假设 iid。
    这里改成回归形式：把两列都标准化后做 `y ~ x`，斜率就是相关系数，
    而斜率的 HAC 标准误自动处理了自相关。
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n < 20:
        return {"n": n, "pearson": float("nan"), "t_nw": float("nan"),
                "p_nw": float("nan"), "spearman": float("nan")}
    xs = (df.x - df.x.mean()) / df.x.std(ddof=1)
    ys = (df.y - df.y.mean()) / df.y.std(ddof=1)
    X = np.column_stack([np.ones(n), xs.to_numpy()])
    beta, se, t, _, _ = _hac_regression(ys.to_numpy(), X, lags)
    return {
        "n": n,
        "pearson": float(beta[1]),
        "se_nw": float(se[1]),
        "t_nw": float(t[1]),
        "p_nw": float(2 * stats.norm.sf(abs(t[1]))) if np.isfinite(t[1]) else float("nan"),
        "spearman": float(stats.spearmanr(df.x, df.y).statistic),
    }


def lead_lag(
    signal: pd.Series, ret: pd.Series, max_lag: int = 5, lags: int = 5
) -> pd.DataFrame:
    """领先滞后相关扫描：corr(signal_t, ret_{t+k})，k = -max_lag..max_lag。

    符号约定（读表时最容易搞反的地方）：

        k > 0   signal 领先 ret    -> 有预测价值
        k = 0   同期
        k < 0   ret 领先 signal    -> 预测市场在跟随股市，无增量

    注意多重比较：扫 11 个 k 值，即使真实关联为零，最大的那个 |t| 的期望也
    在 2 附近。所以**不要**挑最显著的那个 k 报告；先看 k>0 整体的形态，
    再看是否与 k<0 对称（对称 = 同期共动，不是领先）。返回表里给出了
    Bonferroni 校正后的阈值供参照。
    """
    rows = []
    for k in range(-max_lag, max_lag + 1):
        res = corr_with_t(signal, ret.shift(-k), lags=lags)
        res["k"] = k
        rows.append(res)
    out = pd.DataFrame(rows).set_index("k")
    n_tests = 2 * max_lag + 1
    out["p_bonferroni"] = (out.p_nw * n_tests).clip(upper=1.0)
    return out


def predictive_regression(
    ret: pd.Series,
    signal: pd.Series,
    controls: pd.DataFrame | None = None,
    horizon: int = 1,
    lags: int = 5,
) -> dict:
    """`ret_{t+1..t+h}` 对 `signal_t` 的预测回归（HAC 标准误）。

    加 `controls` 是为了回答"这个关联是不是只是大盘的影子"。航空股与大盘
    相关系数通常在 0.6 以上，而 Polymarket 的地缘风险概率同时也是全市场的
    风险指标——不控制大盘，很容易把"风险偏好"误读成"燃油成本"。

    `horizon > 1` 时因变量是未来 h 日的累计收益，样本重叠 h-1 天，
    所以 HAC 滞后阶至少要取 h（默认 5 已覆盖 h≤5）。
    """
    fwd = ret.shift(-1).rolling(horizon).sum().shift(-(horizon - 1)) if horizon > 1 else ret.shift(-1)
    parts = {"y": fwd, "signal": signal}
    if controls is not None:
        for c in controls.columns:
            parts[f"ctl_{c}"] = controls[c]
    df = pd.DataFrame(parts).dropna()
    if len(df) < 30:
        return {"n": len(df), "beta": float("nan"), "t_nw": float("nan")}
    xcols = [c for c in df.columns if c != "y"]
    X = np.column_stack([np.ones(len(df)), df[xcols].to_numpy()])
    beta, se, t, r2, n = _hac_regression(df.y.to_numpy(), X, max(lags, horizon))
    return {
        "n": n,
        "horizon": horizon,
        "beta": float(beta[1]),
        "se_nw": float(se[1]),
        "t_nw": float(t[1]),
        "p_nw": float(2 * stats.norm.sf(abs(t[1]))) if np.isfinite(t[1]) else float("nan"),
        "r2": r2,
        "controls": xcols[1:],
        "beta_all": dict(zip(["const"] + xcols, beta)),
        "t_all": dict(zip(["const"] + xcols, t)),
    }


def cross_sectional_ic(
    signal_panel: pd.DataFrame, ret_panel: pd.DataFrame, lags: int = 5
) -> tuple[pd.Series, dict]:
    """横截面 IC：本项目主指标口径，但这里横截面只有 6–8 只股票。

    Parameters
    ----------
    signal_panel, ret_panel : pd.DataFrame
        `index=date, columns=股票代码`。signal 必须是**因股而异**的
        （典型构造：个股油价暴露度 β_i × 当日 Δp），否则截面 IC 恒为 NaN——
        对所有股票相同的一列，其截面标准差为 0。

    横截面只有 8 只时，单日 IC 的标准误约 1/√(8-1) ≈ 0.38，也就是说单日
    IC 基本是噪声。但日度 IC 序列的**均值**仍然是无偏的，t 值会随天数收敛，
    ~900 天下能检出的 IC 量级在 0.03 上下。这个限制必须在报告里写明，
    不能拿 8 只股票的 IC 去跟全市场 1500 只的 IC 直接比大小。
    """
    common_dates = signal_panel.index.intersection(ret_panel.index)
    common_cols = signal_panel.columns.intersection(ret_panel.columns)
    s = signal_panel.loc[common_dates, common_cols]
    r = ret_panel.loc[common_dates, common_cols]
    ic, ric, n_eff = [], [], []
    for d in common_dates:
        a, b = s.loc[d], r.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() < 3 or a[ok].std(ddof=1) == 0 or b[ok].std(ddof=1) == 0:
            ic.append(np.nan); ric.append(np.nan); n_eff.append(int(ok.sum()))
            continue
        ic.append(float(a[ok].corr(b[ok])))
        ric.append(float(a[ok].corr(b[ok], method="spearman")))
        n_eff.append(int(ok.sum()))
    ic_s = pd.Series(ic, index=common_dates).dropna()
    ric_s = pd.Series(ric, index=common_dates).dropna()
    t_nw, se = newey_west_tstat(ic_s, lags=lags)
    # `mean_xs_width` 只在**真正进了 IC 序列的那些天**上取均值。
    # 原来对全部 common_dates 取均值，把信号缺失日的 n_eff=0 也算进去，
    # 于是报出 0.158 这种数——读者会以为截面几乎是空的，而实际上进检验的
    # 每一天都有 6–8 只。这个字段的用途是回答"每天有几只股票参与排序"，
    # 没参与排序的日子本就不该进这个均值。
    used = pd.Series(n_eff, index=common_dates).reindex(ic_s.index)
    summary = {
        "n_days": len(ic_s),
        "mean_xs_width": float(used.mean()) if len(used) else float("nan"),
        "ic_mean": float(ic_s.mean()) if len(ic_s) else float("nan"),
        "ic_std": float(ic_s.std(ddof=1)) if len(ic_s) > 1 else float("nan"),
        "icir": float(ic_s.mean() / ic_s.std(ddof=1)) if len(ic_s) > 1 and ic_s.std(ddof=1) > 0 else float("nan"),
        "t_nw": t_nw,
        "p_nw": float(2 * stats.norm.sf(abs(t_nw))) if np.isfinite(t_nw) else float("nan"),
        "ic_win_rate": float((ic_s > 0).mean()) if len(ic_s) else float("nan"),
        "ric_mean": float(ric_s.mean()) if len(ric_s) else float("nan"),
    }
    return ic_s, summary


def rolling_exposure(
    ret_panel: pd.DataFrame, driver: pd.Series, window: int = 120, min_periods: int = 60
) -> pd.DataFrame:
    """个股对某个驱动变量（如原油收益）的滚动暴露度 β_i。

    只用**过去**的数据：`shift(1)` 后再滚动，所以 t 日的 β 完全由 t-1 及之前
    决定，不含未来信息。这是把"一个宏观序列"变成"有横截面区分度的因子"
    的标准做法，与 `pmsp.extensions.external_series.exposure_expr` 同源。
    """
    d = driver.reindex(ret_panel.index)
    out = {}
    for col in ret_panel.columns:
        y = ret_panel[col]
        cov = y.rolling(window, min_periods=min_periods).cov(d)
        var = d.rolling(window, min_periods=min_periods).var()
        out[col] = (cov / var.where(var > 0)).shift(1)
    return pd.DataFrame(out, index=ret_panel.index)


def event_study(
    ret: pd.Series,
    signal: pd.Series,
    quantile: float = 0.90,
    pre: int = 5,
    post: int = 10,
    direction: str = "abs",
) -> tuple[pd.DataFrame, dict]:
    """概率大幅跳变日前后的累计收益。

    Parameters
    ----------
    direction : {"abs", "up", "down"}
        `abs` 取 |signal| 的高分位（"有大消息"），`up`/`down` 取带方向的
        高/低分位（"油价风险上升/下降"）。

    为什么要做事件研究：日频相关系数把信息日和非信息日同权平均，而这类
    地缘风险市场 95% 的日子概率纹丝不动。如果真实效应集中在 5% 的日子里，
    全样本相关系数会被稀释到看不见，但事件窗口里应当清晰可见。

    统计上的坑：事件日会**成串出现**（一场冲突连着十几天），事件窗口互相
    重叠，所以横截面平均后的 t 值不能按独立样本算。这里报的 t 值用的是
    "每个事件的窗口累计收益"这一组数的 Newey-West t 值，把事件按时间排序后
    处理串联自相关。
    """
    df = pd.DataFrame({"ret": ret, "sig": signal}).dropna()
    if direction == "abs":
        score = df.sig.abs()
        thr = score.quantile(quantile)
        mask = score >= thr
    elif direction == "up":
        thr = df.sig.quantile(quantile)
        mask = df.sig >= thr
    else:
        thr = df.sig.quantile(1 - quantile)
        mask = df.sig <= thr
    idx = np.flatnonzero(mask.to_numpy())
    rets = df.ret.to_numpy()
    offsets = np.arange(-pre, post + 1)
    rows = []
    for i in idx:
        lo, hi = i - pre, i + post
        if lo < 0 or hi >= len(rets):
            continue
        rows.append(rets[lo : hi + 1])
    if not rows:
        return pd.DataFrame(), {"n_events": 0}
    M = np.array(rows)
    mean_ar = M.mean(axis=0)
    # CAR 锚定在**事件日前一天 = 0**，而不是从窗口第一天直接累加。
    # 不锚定的话窗口第一天的值就等于那天的平均收益（可能是 +0.7%），
    # 曲线一开头就悬在半空，读者没法判断"事件日之后到底涨了多少"，
    # 也没法把几条不同标的的曲线放在一起比——它们各自的起点都不一样。
    # 锚定后：负半轴读的是"冲进事件的那几天发生了什么"（有漂移就是警号），
    # 0 那一点读的是事件日本身的反应，正半轴读的是事后累计效应。
    car = np.cumsum(mean_ar)
    car = car - car[pre - 1] if pre >= 1 else car - 0.0
    # 事件后累计收益（不含事件日之前）用于显著性检验
    post_car = M[:, pre : pre + post + 1].sum(axis=1)
    t_nw, _ = newey_west_tstat(pd.Series(post_car), lags=5)
    table = pd.DataFrame(
        {"offset": offsets, "mean_ret": mean_ar, "car": car,
         "hit_rate": (M > 0).mean(axis=0)}
    )
    info = {
        "n_events": int(M.shape[0]),
        "threshold": float(thr),
        "direction": direction,
        "car_event_to_post": float(post_car.mean()),
        "t_nw": t_nw,
        "p_nw": float(2 * stats.norm.sf(abs(t_nw))) if np.isfinite(t_nw) else float("nan"),
    }
    return table, info


def triangle(
    signal: pd.Series, oil_ret: pd.Series, stock_ret: pd.Series, lags: int = 5
) -> pd.DataFrame:
    """三段链路各自的关联强度。见模块文档为什么必须拆开验。

    每一段都报同期与 +1 日两种口径：A 股口径下同期是"收盘前的概率变动 vs
    当日收益"（合法但受截断限制），+1 日是纯预测。
    """
    rows = []
    legs = [
        ("Δp → 原油收益", signal, oil_ret),
        ("原油收益 → 航空股收益", oil_ret, stock_ret),
        ("Δp → 航空股收益", signal, stock_ret),
    ]
    for name, x, y in legs:
        for k, tag in [(0, "同期"), (1, "+1日")]:
            res = corr_with_t(x, y.shift(-k), lags=lags)
            rows.append({"链路": name, "口径": tag, **res})
    return pd.DataFrame(rows)
