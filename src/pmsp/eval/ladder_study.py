"""原油行权价阶梯 ↔ 航空股：2026 年 3 月油价冲击窗口的专项检验。

这是主分析的**补充而非替代**。主分析（`oil_airline_study`）用的是长历史上的
"供给中断概率"，与油价之间隔着一层传导；本模块用的是
`pmsp.build.oil_ladder` 从行权价阶梯还原出的**隐含油价分布**，标的直接就是
油价，但只有 21 个交易日。两者的取舍正好相反：一个样本长、信号间接，
一个信号直接、样本短。**都报，不挑**。

## 为什么这 21 天值得单独拿出来

它不是随便一段 21 天。2026-02-27 → 2026-03-30 期间 WTI 从 67 涨到 105
（+56%），而阶梯在这段时间里每天有 7–9 个行权价在密集报价（每个行权价日均
上千笔成交）。也就是说：样本最短的那段，恰好是**信噪比最高**的那段——
油价冲击的幅度大到即使 21 天也有可能看出东西来。

同期的资产表现（2026-03-02 → 2026-03-30 累计）：

    WTI        +56.0%        航空等权   −22.2%
    XLE        +10.8%        JETS       −16.9%
    SPY         −7.9%

能源涨、航空跌、大盘小跌，这是**油价冲击**的特征形态，而不是"什么都在跌"
的普跌行情。这一条是本分析最重要的一道伪发现防线：如果 XLE 和 JETS 一起跌，
那就只是风险偏好恶化，跟油价没关系。

## 推断方法：为什么这里不用 HAC t 值

`assoc.corr_with_t` 在 `n < 20` 时直接返回 NaN，那个下限是对的——Newey-West
标准误在二十来个观测上本身就不可靠，硬算出来的 t 值会**看起来**很漂亮，
但它的名义显著性水平是假的。所以本模块**不降低那个门槛**，改用两件事：

1. **置换检验**。把信号与收益的配对随机打乱 `n_perm` 次，看实测 |r| 在
   零分布里的位置。这个检验在任何样本量下都是精确的，代价是它的零假设
   包含"两列都 iid"。
2. **自相关诊断**（`autocorr_check`）。上面那个代价是否要紧，取决于两列
   还剩多少自相关。两列都是**变动量**（`diff` / 收益率）而不是水位，
   一阶自相关本就该接近零；把它测出来放在结论旁边，读者才能判断置换
   检验的零分布是不是站得住。测出来明显不为零时，结论要打折扣，
   这一点在输出里写明。

## 领先滞后：预期它是零

同期相关高不等于有预测价值。预测市场与股市可能只是**同时**在消化同一条
新闻，那样 `k=0` 很强而 `k=±1` 都接近零。本模块把 `k=-3..3` 全扫出来并
要求读者先看形态、再看显著性：`k>0` 与 `k<0` 大致对称就是同期共动，
不是领先。扫 7 个 k 值的多重比较用 Bonferroni 标注。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 置换次数。经验 p 值的分辨率是 1/(n_perm+1)，两万次够报到 1e-4。
N_PERM = 20000

#: 固定随机种子。结论必须可复现，"再跑一次数字变了"是不能接受的。
SEED = 20260928


def _perm_p(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM,
            seed: int = SEED) -> tuple[float, np.ndarray]:
    """配对打乱的置换检验，返回 (双侧经验 p, 零分布)。

    经验 p 用 `(1 + #{|r_perm| >= |r_obs|}) / (1 + n_perm)`：加一是为了
    让 p 永远严格大于 0——报 "p = 0" 是在宣称零假设下这件事绝不可能发生，
    而置换检验只做了有限次抽样，支撑不了那个说法。

    实现上一次算完所有置换，不写 Python 循环：两列都先标准化，于是相关系数
    就是内积除以 n，两万次置换退化成一个 (n_perm × n) @ (n,) 的矩阵乘法。
    这不只是快，它决定了这个检验能不能放进主脚本——逐次 `np.corrcoef`
    的版本跑满一轮（两个市场 × 三个信号 × 七个标的 × 七个 k）要一分多钟。
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()
    r_obs = float(xs @ ys / n)
    # rng.permuted(..., axis=1) 对每一行独立打乱；`np.tile` 先把 y 铺成
    # (n_perm, n)。内存 n_perm × n 个 float64，两万 × 二十来天 = 几 MB。
    perm = rng.permuted(np.tile(ys, (n_perm, 1)), axis=1)
    null = perm @ xs / n
    p = float((1 + np.sum(np.abs(null) >= abs(r_obs))) / (1 + n_perm))
    return p, null


def corr_perm(x: pd.Series, y: pd.Series, min_n: int = 8,
              n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """两条短序列的相关系数 + 置换 p 值 + 该样本量下的可检出下限。

    `detectable` 一并返回，因为在 n≈18 的样本上，"r=0.2、p=0.4" 这种结果
    **既不是**"没有关联"也不是"有关联"——它只是"测不出来"，而这两种
    情形在报告里常被混为一谈。
    """
    from pmsp.eval.oil_airline_study import detectable

    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    out = {"n": n, "pearson": float("nan"), "p_perm": float("nan"),
           "spearman": float("nan"), "detectable_r": detectable(n)}
    if n < min_n:
        return out
    xv = df.x.to_numpy(float)
    yv = df.y.to_numpy(float)
    if xv.std() == 0 or yv.std() == 0:
        return out
    p, _ = _perm_p(xv, yv, n_perm=n_perm, seed=seed)
    out.update(pearson=float(np.corrcoef(xv, yv)[0, 1]), p_perm=p,
               spearman=float(df.x.rank().corr(df.y.rank())))
    return out


def autocorr_check(series: dict[str, pd.Series], max_lag: int = 3) -> pd.DataFrame:
    """各序列的一阶及高阶自相关，用来判断置换检验的零分布站不站得住。

    置换检验把配对打乱，其零假设里含"观测可交换"。两列都是变动量时这大致
    成立；若某列自相关明显不为零，经验 p 会偏小（偏向报出显著），结论必须
    据此打折扣。所以这张表是和结论一起读的，不是附录。
    """
    rows = []
    for name, s in series.items():
        s = s.dropna()
        rec = {"序列": name, "n": len(s)}
        for k in range(1, max_lag + 1):
            rec[f"rho_{k}"] = (float(s.autocorr(k))
                               if len(s) > k + 2 else float("nan"))
        # Ljung-Box 的 Q 统计量（前 max_lag 阶联合）——单看 rho_1 会漏掉
        # 只在高阶上有结构的情形。
        n = len(s)
        q = float(n * (n + 2) * sum(
            (rec[f"rho_{k}"] ** 2) / (n - k)
            for k in range(1, max_lag + 1)
            if np.isfinite(rec.get(f"rho_{k}", np.nan))))
        rec["LB_Q"] = q
        from scipy import stats
        rec["LB_p"] = float(stats.chi2.sf(q, max_lag)) if np.isfinite(q) else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows).set_index("序列")


def lead_lag_perm(signal: pd.Series, ret: pd.Series, max_lag: int = 3,
                  n_perm: int = N_PERM, seed: int = SEED) -> pd.DataFrame:
    """领先滞后扫描（置换推断版）。符号约定与 `assoc.lead_lag` 一致：

        k > 0   signal 领先 ret    -> 有预测价值
        k = 0   同期
        k < 0   ret 领先 signal    -> 预测市场在跟随股市，无增量

    **不要挑最显著的 k 报告。**扫 7 个 k，即使真实关联为零，最大 |r| 的
    期望也明显大于零。先看整条曲线的形态：`k>0` 与 `k<0` 大致对称就是
    同期共动而非领先。
    """
    rows = []
    for k in range(-max_lag, max_lag + 1):
        # 每个 k 换一个种子，否则所有 k 共用同一套置换，零分布之间完全
        # 相关，"7 个 k 里有几个显著"这种判断就失真了。
        res = corr_perm(signal, ret.shift(-k), n_perm=n_perm, seed=seed + k)
        res["k"] = k
        rows.append(res)
    out = pd.DataFrame(rows).set_index("k")
    out["p_bonferroni"] = (out.p_perm * (2 * max_lag + 1)).clip(upper=1.0)
    return out


#: `divergence` 默认认的对照列（美股口径）。A 股那一遍没有 XLE/JETS/SPY，
#: 调用方传自己的 {列名: 类别} 映射进来，检验逻辑一个字不用改。
US_CONTROLS = {"XLE": "能源", "JETS": "航空ETF", "SPY": "大盘"}


def divergence(controls: pd.DataFrame, airlines: pd.DataFrame,
               start: str, end: str,
               labels: dict[str, str] | None = None) -> pd.DataFrame:
    """窗口内的累计收益对照表：能源 / 航空 / 大盘。

    这是本分析的**主要伪发现防线**，不是背景信息。油价冲击的特征是
    能源涨、航空跌；若三者同向下跌，那就只是风险偏好恶化，跟油价无关，
    此时再高的相关系数也不能解读成"油价预期影响航空股"。
    """
    rows = []

    def cum(s: pd.Series) -> float:
        s = s.loc[start:end].dropna()
        return float((1 + s).prod() - 1) * 100 if len(s) else float("nan")

    rows.append({"资产": f"航空等权（{airlines.shape[1]} 只）",
                 "类别": "航空", "累计收益%": cum(airlines.mean(axis=1))})
    for sym in airlines.columns:
        rows.append({"资产": sym, "类别": "航空", "累计收益%": cum(airlines[sym])})
    for sym, lab in (labels or US_CONTROLS).items():
        if sym in controls.columns:
            rows.append({"资产": sym, "类别": lab, "累计收益%": cum(controls[sym])})
    return pd.DataFrame(rows)


def run(sig: pd.DataFrame, airlines: pd.DataFrame, oil_ret: pd.Series,
        controls: pd.DataFrame | None = None,
        signal_cols: tuple[str, ...] = ("d_p_at_K", "d_K50", "d_exp_exceed"),
        max_lag: int = 3, n_perm: int = N_PERM,
        control_labels: dict[str, str] | None = None) -> dict:
    """跑完整的阶梯专项检验。

    Returns
    -------
    dict
        `contemp`   各信号 vs 航空等权 / 各只航空股 / 油价，的同期相关
        `leadlag`   各信号的领先滞后扫描
        `autocorr`  自相关诊断（判断置换零分布是否站得住）
        `validate`  信号 vs 已实现油价收益的同期相关——这是**提取正确性**的
                    检验，不是发现：阶梯是油价的衍生品，本来就该高度相关，
                    不高才说明还原错了。
    """
    ew = airlines.mean(axis=1).rename("航空等权")
    out: dict = {}

    contemp, validate = [], []
    for c in signal_cols:
        if c not in sig.columns:
            continue
        s = sig[c]
        r = corr_perm(s, ew, n_perm=n_perm)
        r["信号"], r["标的"] = c, "航空等权"
        contemp.append(r)
        for sym in airlines.columns:
            r2 = corr_perm(s, airlines[sym], n_perm=n_perm)
            r2["信号"], r2["标的"] = c, sym
            contemp.append(r2)
        v = corr_perm(s, oil_ret, n_perm=n_perm)
        v["信号"], v["标的"] = c, "WTI 收益（提取正确性检验）"
        validate.append(v)

    cols = ["信号", "标的", "n", "pearson", "spearman", "p_perm", "detectable_r"]
    out["contemp"] = (pd.DataFrame(contemp)[cols] if contemp else pd.DataFrame())
    out["validate"] = (pd.DataFrame(validate)[cols] if validate else pd.DataFrame())
    out["leadlag"] = {c: lead_lag_perm(sig[c], ew, max_lag=max_lag, n_perm=n_perm)
                      for c in signal_cols if c in sig.columns}
    # 自相关必须在**本窗口**上测，不能用全历史：这张表的用途是判断置换检验
    # 的可交换性假设在这 21 天里站不站得住。喂全历史的航空收益进去会得到
    # n=11774 的诊断，那测的是另一个样本，对本窗口的推断一句话也没说。
    lo, hi = (sig.index.min(), sig.index.max()) if len(sig) else (None, None)
    clip = (lambda s: s.loc[lo:hi]) if lo is not None else (lambda s: s)
    out["autocorr"] = autocorr_check(
        {c: sig[c] for c in signal_cols if c in sig.columns}
        | {"航空等权收益": clip(ew), "WTI 收益": clip(oil_ret)})
    if controls is not None and len(sig):
        out["divergence"] = divergence(
            controls, airlines,
            str(sig.index.min().date()), str(sig.index.max().date()),
            labels=control_labels)
    return out
