"""把"原油概率 ↔ 航空股"这套检验固化成一个可对两个市场分别跑的流程。

为什么要单独一个模块而不是写在脚本里：**同一套检验要在 A 股与美股上各跑
一遍，而且必须逐字相同**。美股那一遍的作用是阳性对照——它与 Polymarket
同时区、同市场时段，如果连美股都测不出关联，A 股的零结果才能解读为
"信号本身弱"而不是"时区错配把信号磨掉了"。对照要成立，两边的口径必须
一模一样，任何一处手工差异都会让对照失效。所以流程写成函数，参数化掉
市场差异（截断时刻、油种、成分股），而不是复制两段代码。

## 参数化掉的三处市场差异

===============  ==================  ==========================================
             A 股                美股
===============  ==================  ==========================================
截断时刻         07:00 UTC           20:00 UTC
油种基准         布伦特（OIL）       WTI（CL）
航空标的         8 只 A 股等权       6 只美股等权 + JETS（现成 ETF，无加权自由度）
能源对照         中石油 + 中石化     XLE
大盘控制         全市场等权          SPY
===============  ==================  ==========================================

油种不是随便选的：美股航空的燃油成本贴 WTI，A 股航油采购贴布伦特。混用
会引入一个纯粹由地域价差（WTI-Brent spread）造成的噪声。

## 信号用哪一个

四个候选，各自回答的问题不同，所以全报而不是挑一个：

    dp          概率的绝对变动。**主信号**。"消息量"的自然度量。
    dp_norm     除以该市场自身 Δp 滚动标准差。跨市场可比，但滚动标准差
                在事件期本身就会放大，有把大消息缩小的副作用。
    ofi_share   单边成交压力占比（OFI / 成交额）。这是本分析里最接近
                "盘口"的量——真实的买卖盘挂单深度这份数据里没有
                （只有 64 天的订单簿，太短），成交流向是可得的最佳代理。
    p_level     概率水位本身。注意它是高度自相关的（近似随机游走），
                拿水位去跟收益率算相关是伪回归的经典形式，所以它只用于
                画图和事件定义，**不进显著性表**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..build import stock_panels as sp
from . import assoc

#: 进显著性表的信号。`p_level` 刻意不在其中（见模块文档：水位对收益的回归
#: 是伪回归）。
SIGNALS = ["dp", "dp_norm", "ofi_share"]


@dataclass
class MarketSpec:
    """一个市场的全部口径参数。"""

    key: str                      # "cn" / "us"，决定 pm_daily 的截断时刻
    label: str                    # 报告里的显示名
    airlines: pd.DataFrame        # 航空股日收益面板 (date × code)
    energy: pd.Series             # 能源对照日收益
    market: pd.Series             # 大盘日收益（控制变量）
    oil_ret: pd.Series            # 对应油种的日收益
    oil_px: pd.Series             # 对应油种的收盘价（画图用）
    oil_label: str
    etf: pd.Series | None = None  # 现成航空 ETF（美股有 JETS，A 股无）
    etf_label: str = ""
    names: pd.Series | None = None
    extra: dict = field(default_factory=dict)

    @property
    def airline_index(self) -> pd.Series:
        """等权航空指数日收益。有现成 ETF 时仍然自己合成一条，两条都报——
        ETF 少了加权自由度但含费用与折溢价，等权指数反过来。两条结论一致
        才算稳。"""
        return sp.equal_weight_index(self.airlines)


def build_signals(theme: pd.DataFrame) -> pd.DataFrame:
    """主题日频面板 -> 信号表（`index=date`）。"""
    t = theme.set_index("date") if "date" in theme.columns else theme.copy()
    out = t[[c for c in ["p_level", "dp", "dp_norm", "ofi_share", "rvol",
                         "vol_usdc", "n_trades", "n_markets"] if c in t.columns]].copy()
    # Δp 缺失 ≠ Δp 为零：缺失是"那天没有可用报价"，填 0 会在零附近
    # 堆出一大堆假的"无消息日"，把相关系数往零方向稀释。保持 NaN，
    # 由各检验自己 dropna。
    return out


def targets(spec: MarketSpec) -> dict[str, pd.Series]:
    """要检验的收益序列。键名直接进图例与报告表格。"""
    out = {
        f"{spec.label}航空（等权）": spec.airline_index,
        f"{spec.label}能源": spec.energy,
        f"{spec.label}大盘": spec.market,
    }
    if spec.etf is not None:
        out[spec.etf_label] = spec.etf
    out[spec.oil_label] = spec.oil_ret
    return out


def residual_targets(spec: MarketSpec) -> dict[str, pd.Series]:
    """剔除大盘成分后的残差收益。

    这一组是判别"燃油成本渠道"与"风险偏好渠道"的关键。地缘风险概率上升时
    大盘本身会跌，航空股跟着跌**不需要**任何油的逻辑。只有在剔除大盘之后
    航空股仍然跌、且能源股仍然涨，才能说抓到了油这个渠道。
    """
    out = {}
    for nm, s in targets(spec).items():
        if "大盘" in nm or nm == spec.oil_label:
            continue
        out[f"{nm}（剔大盘）"] = sp.residualize(s, spec.market)
    return out


def _ic_placebo(spec: MarketSpec, dp_raw: pd.Series, beta: pd.DataFrame,
                main_ic: pd.Series | None, lags: int = 5,
                n_perm: int = 2000, seed: int = 20260924) -> dict:
    """横截面 IC 的安慰剂对照。**这一步不能省。**

    打分是 `−β_i × Δp_t`。同一天里 Δp_t 对所有股票是同一个数，于是有一条很
    容易混进来的假阳性路径：若样本期内"低油价暴露的航空股跑得更好"本身成立
    （一个与预测市场无关的截面异象），而 Δp 又碰巧多数为正，IC 就会显著为正，
    但功劳一点也不属于 Polymarket。

    ## 一个必须写进报告的恒等式

    Pearson 相关对其中一个变量的正仿射变换不变，而 Δp_t 对当天所有股票是
    **标量**，所以

        IC_t = corr(−β·Δp_t, r) = sign(−Δp_t) · corr(β, r)

    也就是说，**截面 IC 只用到 Δp 的符号，|Δp| 被完全丢掉了**。Δp = +0.5 和
    Δp = +0.03 给出一模一样的 IC。这不是实现瑕疵，是"宏观序列 × 暴露度"这种
    构造的固有性质；它同时意味着这个 IC 检验的其实是一个很窄的命题：
    「供给风险上升的日子里，高油价暴露的航空股是否跑得更差」。

    有了这个恒等式，置换检验就不必重算 2000 次截面相关——只要置换符号向量。
    但**捷径必须自证**：函数里用主口径已算出的 IC 序列逐点核对一遍，
    误差超过 1e-9 就抛错。哪天打分构造改了（比如改成非线性），
    这个断言会立刻炸掉，而不是悄悄给出错的 p 值。

    Returns
    -------
    dict
        `ic_placebo`：`beta_only` 对照的 summary（与主口径逐格同样本）；
        `ic_permutation`：符号置换检验的经验 p 值等。
    """
    rng = np.random.default_rng(seed)
    dp = dp_raw.reindex(spec.airlines.index)
    # 对照必须在与主口径**逐格相同**的样本上评。不掩码的话 `beta_only`
    # 会一路摊到 Polymarket 还没有这个市场的年份上（样本从 288 天涨到
    # 2361 天），"不显著"就分不清是真没信号还是被稀释掉了。
    base_valid = (-beta.mul(dp, axis=0)).notna()
    b_ic, b_sum = assoc.cross_sectional_ic((-beta).where(base_valid),
                                           spec.airlines, lags=lags)
    placebo = {"beta_only（打分 = −β，不含 Polymarket 信息）": b_sum}

    perm: dict = {}
    if main_ic is not None and len(b_ic):
        # sign(0) = 0 会让打分整列为 0、截面标准差为 0，那天主口径本身就是
        # NaN，所以这些天不进检验（而不是当成"符号为正"）。
        sgn = np.sign(dp.reindex(b_ic.index)).replace(0.0, np.nan)
        implied = (b_ic * sgn).dropna()
        chk = main_ic.reindex(implied.index)
        err = float((implied - chk).abs().max())
        if not (err < 1e-9):
            raise AssertionError(
                f"IC = sign(−Δp)·corr(β,r) 这个恒等式不成立（最大偏差 {err:.3g}）；"
                "打分构造变了，符号置换的捷径失效，置换检验必须改回逐次重算"
            )
        s = sgn.reindex(implied.index).to_numpy()
        base = b_ic.reindex(implied.index).to_numpy()
        t_main = assoc.newey_west_tstat(implied, lags=lags)[0]
        t_null, m_null = [], []
        for _ in range(n_perm):
            ic_p = pd.Series(base * rng.permutation(s), index=implied.index)
            t_null.append(assoc.newey_west_tstat(ic_p, lags=lags)[0])
            m_null.append(float(ic_p.mean()))
        t_null = np.asarray(t_null, dtype=float)
        m_null = np.asarray(m_null, dtype=float)
        # 经验 p 值加 1：置换检验的标准做法，把观测值本身算作一个可能的排列，
        # 这样 p 永远不会是 0（"0 次超过"不等于"概率为 0"）。
        perm = {
            "n_perm": n_perm,
            "n_days": len(implied),
            "ic_mean_实测": float(implied.mean()),
            "t_NW_实测": float(t_main),
            "ic_mean_置换均值": float(m_null.mean()),
            "t_NW_置换_p5": float(np.nanpercentile(t_null, 5)),
            "t_NW_置换_p95": float(np.nanpercentile(t_null, 95)),
            "p_置换_双侧": float((1 + np.sum(np.abs(t_null) >= abs(t_main)))
                                 / (1 + n_perm)),
        }
    return {"ic_placebo": placebo, "ic_permutation": perm}


def run(spec: MarketSpec, signals: pd.DataFrame, max_lag: int = 5,
        lags: int = 5, event_q: float = 0.90) -> dict:
    """跑完一个市场的全套检验，返回一个可直接落盘/出图的结果字典。

    不在这里做任何"挑最好看的那个"的动作：所有信号 × 所有标的的结果都留在
    返回值里，取舍留给报告，且报告里要把没选的也列出来。这是本项目
    一贯的做法——筛选过程本身必须可复核。
    """
    tg = targets(spec)
    res_tg = residual_targets(spec)
    out: dict = {"spec": spec, "signals": signals}

    # ---- 1. 领先滞后扫描：每个信号 × 每个标的
    ll: dict[str, dict[str, pd.DataFrame]] = {}
    for sig in SIGNALS:
        if sig not in signals:
            continue
        ll[sig] = {
            nm: assoc.lead_lag(signals[sig], r, max_lag=max_lag, lags=lags)
            for nm, r in tg.items()
        }
    out["lead_lag"] = ll

    # ---- 2. 剔大盘后的领先滞后（主信号一个就够，避免多重比较膨胀）
    out["lead_lag_resid"] = {
        nm: assoc.lead_lag(signals["dp"], r, max_lag=max_lag, lags=lags)
        for nm, r in res_tg.items()
    } if "dp" in signals else {}

    # ---- 3. 三角验证
    out["triangle"] = assoc.triangle(
        signals["dp"], spec.oil_ret, spec.airline_index, lags=lags
    ) if "dp" in signals else pd.DataFrame()

    # ---- 4. 事件研究：|Δp| 高分位日；三个方向都做
    #        abs  = "有大消息"；up = "供给风险上升"；down = "风险缓解"
    #        只看 abs 会把方向相反的两类事件平均掉，那是最容易得出
    #        "没有效应"这种假结论的地方。
    ev: dict[str, dict] = {}
    for direction in ["abs", "up", "down"]:
        ev[direction] = {
            nm: assoc.event_study(r, signals["dp"], quantile=event_q,
                                  direction=direction)
            for nm, r in tg.items()
        } if "dp" in signals else {}
    out["events"] = ev

    # ---- 5. 横截面 IC：必须先把宏观序列乘上因股而异的暴露度
    #        一个对所有股票都相同的数，其横截面标准差为 0，对截面排序的
    #        贡献恰为零（`pmsp.extensions.external_series` 里有证明）。
    beta = assoc.rolling_exposure(spec.airlines, spec.oil_ret)
    ic_out, ic_sum = {}, {}
    for sig in SIGNALS:
        if sig not in signals:
            continue
        s = signals[sig].reindex(spec.airlines.index)
        # 预测方向：油价风险上升（Δp>0）对高油价暴露（β 大）的股票更不利，
        # 所以打分取 -β·Δp，让"分高 = 预期收益高"。符号搞反的后果是
        # IC 整体反号，而不是不显著——所以这一行必须能被单独复核。
        score = -beta.mul(s, axis=0)
        ic, summary = assoc.cross_sectional_ic(score, spec.airlines, lags=lags)
        ic_out[sig], ic_sum[sig] = ic, summary
    out["beta"] = beta
    out["ic"], out["ic_summary"] = ic_out, ic_sum

    # ---- 5b. IC 的安慰剂对照。**这一步不能省。**
    #
    # 打分是 `−β_i × Δp_t`。同一天里 Δp_t 对所有股票都是同一个数，所以它只
    # 决定排序的**符号**，排序的**形状**完全由 β_i 决定。于是有一个很容易
    # 混进来的假阳性：如果样本期内"低油价暴露的航空股跑得更好"本身成立
    # （一个与 Polymarket 毫无关系的截面异象），而 Δp 又碰巧多数为正，
    # 那么 IC 会显著为正，但功劳一点也不属于预测市场。
    #
    # 两个对照把这件事切开：
    #   beta_only        打分 = −β_i，完全不含 Polymarket 信息。
    #                    它显著 = IC 来自 β 异象，与预测市场无关。
    #   dp_shuffled      Δp 在日期上随机打乱（固定种子），β 不动。
    #                    它显著 = 同上；它不显著而主口径显著 = Δp 的
    #                    **时序对齐**真的有贡献。
    #
    # 主口径必须显著**且**两个对照都不显著，IC 结论才成立。
    if "dp" in signals:
        out.update(_ic_placebo(spec, signals["dp"], beta, ic_out.get("dp"),
                               lags=lags))
    else:
        out["ic_placebo"], out["ic_permutation"] = {}, {}

    # ---- 6. 预测回归：加大盘控制，看关联是不是大盘的影子
    reg = []
    for sig in SIGNALS:
        if sig not in signals:
            continue
        for nm, r in tg.items():
            if nm == spec.oil_label:
                continue
            ctl = pd.DataFrame({"mkt": spec.market})
            plain = assoc.predictive_regression(r, signals[sig], horizon=1, lags=lags)
            withc = assoc.predictive_regression(r, signals[sig], controls=ctl,
                                                horizon=1, lags=lags)
            reg.append({
                "信号": sig, "标的": nm, "n": plain["n"],
                "beta": plain["beta"], "t_NW": plain["t_nw"],
                "beta_控制大盘": withc["beta"], "t_NW_控制大盘": withc["t_nw"],
            })
    out["regression"] = pd.DataFrame(reg)

    # ---- 7. 分化检验：能源 − 航空 的收益差对 Δp 的回归。
    #        这是主要的伪发现防线。真实油价冲击 -> 能源涨、航空跌，价差走阔；
    #        若两者只是同涨同跌，价差对 Δp 的系数应当不显著。
    spread = (spec.energy - spec.airline_index).rename("energy_minus_airline")
    out["divergence"] = {
        "同期": assoc.corr_with_t(signals["dp"], spread, lags=lags),
        "+1日": assoc.corr_with_t(signals["dp"], spread.shift(-1), lags=lags),
    } if "dp" in signals else {}
    out["spread"] = spread

    # ---- 8. 稳健性：缩尾后重算主口径。若主结果在缩尾后消失，
    #        结论要改写成"效应只存在于极端事件日"，而不是报一个全样本相关。
    if "dp" in signals:
        out["robust_winsor"] = {
            nm: assoc.corr_with_t(sp.winsorize(signals["dp"]),
                                  sp.winsorize(r).shift(-1), lags=lags)
            for nm, r in tg.items()
        }
    return out


def headline(out: dict) -> pd.DataFrame:
    """把一个市场的结果压成报告开头的一张小表：主信号、主标的、k=0 与 k=+1。"""
    spec: MarketSpec = out["spec"]
    rows = []
    for nm, tb in out["lead_lag"].get("dp", {}).items():
        for k in [0, 1]:
            if k not in tb.index:
                continue
            rows.append({
                "市场": spec.label, "标的": nm, "k": k,
                "n": int(tb.loc[k, "n"]),
                "pearson": tb.loc[k, "pearson"],
                "t_NW": tb.loc[k, "t_nw"],
                "p_NW": tb.loc[k, "p_nw"],
                "p_Bonf": tb.loc[k, "p_bonferroni"],
                "spearman": tb.loc[k, "spearman"],
            })
    return pd.DataFrame(rows)


def detectable(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """给定样本量，双侧检验能以 `power` 的功效检出的最小 |相关系数|。

    这个数必须和每个零结果一起报。"相关系数 0.02、p=0.6" 单独看毫无信息，
    配上"这个样本量下只能检出 |r|>0.11"才说得清是"没有关联"还是"测不出"。
    """
    from scipy import stats

    if n < 10:
        return float("nan")
    z = stats.norm.isf(alpha / 2) + stats.norm.isf(1 - power)
    return float(np.tanh(z / np.sqrt(n - 3)))
