"""把 Polymarket 的原油行权价阶梯还原成一条隐含分布。

这是本分析里**唯一一个真正以油价本身为标的**的信号。其余原油市场问的都是
"某个地缘事件会不会发生"，概率与油价之间还隔着一层传导；阶梯问的是
"WTI 会不会摸到 100 美元"，答案直接以美元/桶计价。

## 为什么阶梯不能当成普通主题来聚合

`pmsp.build.pm_daily.aggregate_theme` 的做法是把同主题下各市场的概率做
成交额加权平均。对阶梯这样做会算出一个**不对应任何事件的数**：把
P(max≥100)=0.30 与 P(max≥200)=0.05 平均成 0.175，这个 0.175 不是任何
命题的概率。阶梯的正确用法是横跨行权价读出一条生存函数

    S(K) = P( max_{t≤T} WTI_t ≥ K )

然后从 S 上取有经济含义的统计量。所以 `oil_markets.THEMES` 里
`price_ladder` 是单列的主题，且不在 `OIL_THEMES` 里。

## 数据实况（全量目录实测，决定了下面的取舍）

四个到期月的阶梯质量差了一个数量级：

    到期            天数  每日报价 K 数  每(日,K)成交笔数中位  单调性违反
    end-of-march     33       9 (7–12)            1283          3/33
    march-13          6       6 (5–6)              313          0/6
    end-of-june      95       5 (1–10)              78         20/75
    end-of-february  25       —                     —            —

`end-of-june` 虽然横跨 95 天，但每日只有 5 个行权价在报价、每个行权价
一天才几十笔成交，四分之一的交易日出现 S(K) 随 K 上升——那不是市场观点，
是稀薄成交下的隔夜残价。**拿它拟合分布会得到一条主要由噪声构成的曲线。**
所以隐含分布只用 `end-of-march`（33 天），并且这个 33 天的样本长度必须
和结论一起报：n=33 时 80% 功效下只能检出 |r| > 0.46。

## 两种插值，一种合法一种不合法

* **跨行权价插值（合法）。**同一天里 K=110 有报价、K=115 没有，用相邻
  行权价插出 S(115)，用到的全部是当天的信息。而且 S 在 K 上单调，
  插值误差有界。
* **跨时间插值（不合法）。**某一天整条曲线缺失，用前后两天插出来——
  那是把明天的信息搬到今天，是前视。缺就是缺，留 NaN。

## 单调性怎么处理

S(K) 必须随 K 非增，这是无套利约束（"摸到 110" 蕴含 "摸到 100"）。实测
违反的那几天，违反幅度就是该日报价的噪声下限。处理方式是**保序回归**
（相邻违反对取加权均值，PAVA），而不是直接 `cummin`：`cummin` 会把
违反的锅全甩给高行权价那一侧，而实际上噪声两边都有。违反幅度一并输出，
超过阈值的日子在报告里标出来，不偷偷抹平。

## 障碍吸收：这是本模块最容易算错的地方

障碍型合约一旦被触碰就**立即**结算，Polymarket 把已结算的市场钉在 0.999。
这个数字长得和概率一样，但它不是预测：已经触碰的 K 概率恒为 1，对未来
油价零信息量，而它进入 `diff` 会造出一条假跳变。

所以 `p >= PINNED_HI` 的格子一律剔除。用钉价而不是"拿已实现油价判断有没有
触碰"，是因为阶梯参照的是哪一条价格序列（结算价？盘中高点？哪个交割月？）
并不明确，而 Polymarket 的钉价是**直接观测得到**的。
`diagnose_absorption` 拿两个外部口径来验这个阈值。

### 下侧**不能**用同样的阈值

这一条是反直觉的、也是最容易写错的：障碍合约在到期前**不会**结算为 No
（还没到期，就还有时间触碰），所以到期前的低价是**真实的深度虚值报价**，
不是结算。实测 spot≈90 时 P(摸到 200)=0.004——这是一个正常的尾部概率。
一旦按 `p <= 0.01` 把它当"已结算"剔掉，K=180/200 两列的覆盖率被打穿，
`fixed_grid` 从 9 个行权价塌成 3 个（100/110/120），隐含超额期望的积分
区间随之从 100 美元宽缩到 20 美元宽——**而这个缩窄看起来像油价预期下降**。

### 到期日必须按 `close_at` 切掉，不能靠阈值

到期当天整条曲线塌成 0.999/0.001，那是**结算**而不是观点：`end-of-march`
的 K=105 在 2026-03-31 从 0.526 掉到 0.001，看着像 −52pp 的重大消息，
实际是合约算账。既然目录里有 `close_at`，就按它切（`< close_at` 的日子
才保留），而不是靠"价格很极端"去猜哪天是到期日。

### 同一个 K 会有多个市场

`hit-high-90-by-end-of-march` 在目录里有**三个** condition_id，成交窗口
分别是 02-28→03-07、03-11→03-13、03-24→03-26：油价每次回落到 90 下方，
Polymarket 就重挂一次。三个在时间上不重叠，所以"按 K 分组取最后一笔"
恰好是对的——**但那是巧合**，靠的是分组内只有一个市场在报价。这里显式
按 `(hour, vol_usdc)` 排序后取末笔，让结果不依赖 groupby 的隐式顺序，
并把"同一天里同一个 K 有几个市场在报价"计数输出。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from pmsp.build import pm_daily

#: 阶梯 slug 的三种命名。`hit-high` / `hit-low` 是**障碍**型（触碰即结算），
#: `settle-at` 是**到期落区间**型。两者的含义不同：障碍型读出的是
#: "期间最高/最低价"的分布，落区间型读出的是"到期价"的分布。
#: 混用会把两条不同的分布叠在一起，所以 `kind` 一路带到最后。
_PAT_BARRIER = re.compile(r"crude-oil-cl-hit-(high|low)-(\d+)", re.I)
_PAT_SETTLE = re.compile(r"crude-oil-cl-settle-at-(\d+)(?:-(\d+))?", re.I)
_PAT_EXP = re.compile(r"by-(end-of-[a-z]+|[a-z]+-\d+)", re.I)

#: numpy 2.0 把 `trapz` 改名成 `trapezoid`。两个版本都要能跑。
_trapz = getattr(np, "trapezoid", None) or np.trapz

#: 做隐含分布用哪个到期。见模块文档的质量表——不是随便选的。
PRIMARY_EXPIRY = "end-of-march"

#: 纳入固定行权价网格的最低覆盖率。低于这个覆盖率的行权价不进网格：
#: 一个只在某几天出现的行权价会让跨时间可比的统计量（如隐含超额期望）
#: 在它出现/消失的那天跳一下，而那个跳动与油价无关。
MIN_K_COVERAGE = 0.80

#: 视为"障碍已触碰、已结算"的概率上界。**没有对应的下界**——见模块文档
#: 「下侧不能用同样的阈值」一节，加下界会把正常的深度虚值报价当成结算剔掉。
PINNED_HI = 0.99


def parse_ladder(catalog: pd.DataFrame) -> pd.DataFrame:
    """市场目录 -> 阶梯元数据（`condition_id` / kind / side / K / expiry）。

    无法解析的行直接丢掉并不作声是不行的——那样"某个阶梯市场为什么没进来"
    就查不到了。所以返回值里保留 `kind="unparsed"` 的行。
    """
    close_at = (pd.to_datetime(catalog.close_at, utc=True, errors="coerce")
                if "close_at" in catalog.columns
                else pd.Series(pd.NaT, index=catalog.index))
    won = (catalog.winning_outcome_label if "winning_outcome_label" in catalog.columns
           else pd.Series("", index=catalog.index))
    rows = []
    for cid, slug, ca, wo in zip(catalog.condition_id, catalog.market_slug.fillna(""),
                                 close_at, won):
        exp_m = _PAT_EXP.search(slug)
        exp = exp_m.group(1).lower() if exp_m else ""
        b = _PAT_BARRIER.search(slug)
        if b:
            rows.append({"condition_id": cid, "market_slug": slug, "kind": "barrier",
                         "side": b.group(1).lower(), "K": float(b.group(2)),
                         "expiry": exp, "close_at": ca, "won": wo})
            continue
        s = _PAT_SETTLE.search(slug)
        if s:
            lo = float(s.group(1))
            hi = float(s.group(2)) if s.group(2) else lo
            rows.append({"condition_id": cid, "market_slug": slug, "kind": "settle",
                         "side": "range", "K": (lo + hi) / 2.0, "expiry": exp,
                         "close_at": ca, "won": wo})
            continue
        if slug.startswith("crude-oil-all-time-high"):
            rows.append({"condition_id": cid, "market_slug": slug, "kind": "ath",
                         "side": "high", "K": np.nan, "expiry": exp,
                         "close_at": ca, "won": wo})
            continue
        rows.append({"condition_id": cid, "market_slug": slug, "kind": "unparsed",
                     "side": "", "K": np.nan, "expiry": exp,
                     "close_at": ca, "won": wo})
    return pd.DataFrame(rows)


def daily_curves(hourly: pd.DataFrame, meta: pd.DataFrame,
                 calendar: pd.DatetimeIndex, market: str = "us",
                 kind: str = "barrier", side: str = "high",
                 expiry: str = PRIMARY_EXPIRY) -> tuple[pd.DataFrame, pd.DataFrame]:
    """小时面板 -> 每日 `S(K)` 曲线（行=交易日，列=行权价）。

    截断口径直接复用 `pm_daily.assign_trading_day`，不在这里另写一份：
    第 d 天用的是 `(d-1 日 cutoff, d 日 cutoff]` 窗口里的最后一笔成交，
    且交易日来自**股票市场的真实日历**（不是 `floor("D")`——那会造出
    周末这种股票没法交易的"交易日"）。

    Returns
    -------
    (curves, weights)
        `curves` 是概率，`weights` 是同形状的成交额——后者用来在报告里
        说明"这条曲线是几笔成交撑起来的"，稀薄的格子不能和密实的格子
        一样对待，也用于保序回归的加权。
    """
    sel = meta[(meta.kind == kind) & (meta.side == side) & (meta.expiry == expiry)]
    h = hourly[hourly.condition_id.isin(set(sel.condition_id))]
    if not len(h) or not len(sel):
        return pd.DataFrame(), pd.DataFrame()
    h, _ = pm_daily.assign_trading_day(h, calendar, market)
    h = h.merge(sel[["condition_id", "K"]], on="condition_id")

    # 到期日及以后整个切掉：那几天的曲线是**结算**而不是观点。用目录里的
    # `close_at` 而不是"价格看着很极端"这种事后判断。
    close_at = pd.to_datetime(sel.close_at, utc=True, errors="coerce").dropna()
    if len(close_at):
        cutoff_day = close_at.min().tz_convert(None).normalize()
        h = h[h.date < cutoff_day]
        if not len(h):
            return pd.DataFrame(), pd.DataFrame()

    # 同一个 (date, K) 可能对应多个 condition_id（重挂）。显式按
    # (hour, vol_usdc) 排序后取末笔，不依赖 groupby 的隐式顺序。
    h = h.sort_values(["hour", "vol_usdc"])
    g = h.groupby(["date", "K"], sort=True)
    curves = g.p_close.last().unstack("K").sort_index()
    weights = g.vol_usdc.sum().unstack("K").reindex_like(curves)
    return curves, weights


def duplicate_listings(hourly: pd.DataFrame, meta: pd.DataFrame,
                       calendar: pd.DatetimeIndex, market: str = "us",
                       kind: str = "barrier", side: str = "high",
                       expiry: str = PRIMARY_EXPIRY) -> pd.DataFrame:
    """统计同一天里同一个行权价有几个市场在报价（重挂的重叠程度）。

    重挂本身正常（油价回落到障碍下方就重挂一次），只要在时间上不重叠，
    "按 K 取末笔"就是唯一解。**一旦重叠**，末笔取的是哪一个就取决于排序，
    报告里必须知道这件事有没有发生，而不是假设它没发生。
    """
    sel = meta[(meta.kind == kind) & (meta.side == side) & (meta.expiry == expiry)]
    h = hourly[hourly.condition_id.isin(set(sel.condition_id))]
    if not len(h):
        return pd.DataFrame()
    h, _ = pm_daily.assign_trading_day(h, calendar, market)
    h = h.merge(sel[["condition_id", "K"]], on="condition_id")
    n = h.groupby(["date", "K"]).condition_id.nunique()
    return n[n > 1].rename("n_markets").reset_index()


def fixed_grid(curves: pd.DataFrame, min_coverage: float = MIN_K_COVERAGE) -> list[float]:
    """挑出覆盖率足够、可跨时间比较的行权价网格。"""
    if not len(curves):
        return []
    cov = curves.notna().mean()
    return sorted(float(k) for k in cov[cov >= min_coverage].index)


def _isotonic_decreasing(k: np.ndarray, p: np.ndarray,
                         w: np.ndarray | None = None) -> np.ndarray:
    """把 `p` 投影成随 `k` 非增的序列（PAVA，成交额加权）。

    用保序回归而不是 `np.minimum.accumulate`：后者把违反全部归给高行权价
    一侧（低行权价的报价被当成绝对正确），而稀薄成交下两侧都可能是残价。
    PAVA 给出的是在加权最小二乘意义下改动最小的那个单调序列。
    """
    order = np.argsort(k)
    y = p[order].astype(float)
    ww = (np.ones_like(y) if w is None else np.asarray(w, float)[order]).copy()
    ww = np.where(np.isfinite(ww) & (ww > 0), ww, 1.0)
    # 对 -y 做非减保序回归，等价于对 y 做非增。
    vals = (-y).tolist()
    wts = ww.tolist()
    cnt = [1] * len(vals)
    i = 0
    while i < len(vals) - 1:
        if vals[i] <= vals[i + 1] + 1e-15:
            i += 1
            continue
        tw = wts[i] + wts[i + 1]
        vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tw
        wts[i] = tw
        cnt[i] += cnt[i + 1]
        del vals[i + 1], wts[i + 1], cnt[i + 1]
        while i > 0 and vals[i - 1] > vals[i] + 1e-15:
            tw = wts[i - 1] + wts[i]
            vals[i - 1] = (vals[i - 1] * wts[i - 1] + vals[i] * wts[i]) / tw
            wts[i - 1] = tw
            cnt[i - 1] += cnt[i]
            del vals[i], wts[i], cnt[i]
            i -= 1
    out = np.repeat(np.array(vals), np.array(cnt))
    res = np.empty_like(y)
    res[:] = -out
    back = np.empty_like(res)
    back[order] = res
    return np.clip(back, 0.0, 1.0)


def drop_pinned(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """剔除已结算/已吸收的格子。见模块文档的"障碍吸收"一节。

    Returns
    -------
    (live, pinned)
        `live` 是只剩"仍在双边报价"的概率矩阵，`pinned` 是同形状的布尔掩码
        （True = 被剔除的钉住格子）。掩码要返回，因为"哪个障碍在哪天被触碰"
        本身就是结果的一部分——它是市场对油价路径的**已实现**记录，
        报告里要画出来。
    """
    if not len(curves):
        return curves, curves
    pinned = curves.notna() & (curves >= PINNED_HI)
    return curves.mask(pinned), pinned


def clean_curves(curves: pd.DataFrame, weights: pd.DataFrame | None = None,
                 grid: list[float] | None = None,
                 min_points: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐日做保序修正 + 跨行权价插值。

    输出在**两套行权价列**上都给值：`grid` 里的固定网格（供跨日可比的积分
    使用），以及当日报价覆盖到的其他行权价（供 K50 这类"读水平"的统计量
    使用）。只用固定网格会白扔信息——`end-of-march` 的网格从 K=100 起，
    而 3 月初 spot 才 67 美元、S(90)=0.38，中位障碍价落在 88 附近，
    用 100 起步的网格根本读不到 K50（实测 23 天里有 9 天读不出来）。

    Returns
    -------
    (clean, diag)
        `diag` 每天一行：报价个数、被钉住剔除的个数、单调性违反的最大幅度、
        插值补了几个格子。违反幅度是该日报价噪声的下限，报告里要能看到。
    """
    if not len(curves):
        return pd.DataFrame(), pd.DataFrame()
    live, pinned = drop_pinned(curves)
    grid = grid if grid is not None else fixed_grid(live)
    cols = sorted({float(c) for c in live.columns} | {float(g) for g in grid})
    rows, diag = {}, []
    for day, row in live.iterrows():
        obs = row.dropna()
        rec = {"day": day, "n_quotes": len(obs),
               "n_pinned": int(pinned.loc[day].sum()),
               "max_violation": np.nan, "n_interp": 0, "used": False}
        if len(obs) < min_points:
            diag.append(rec)
            continue
        k = obs.index.to_numpy(dtype=float)
        p = obs.to_numpy(dtype=float)
        w = (weights.loc[day, obs.index].to_numpy(dtype=float)
             if weights is not None else None)
        order = np.argsort(k)
        viol = float(np.max(np.diff(p[order]))) if len(p) > 1 else 0.0
        fitted = _isotonic_decreasing(k, p, w)
        ko, po = k[order], fitted[order]
        # 只在报价的行权价**区间内**插值，绝不外推：外推到 K=200 而当天最高
        # 报价只有 K=140，等于凭空发明一个尾部概率。
        inside = [x for x in cols if ko[0] <= x <= ko[-1]]
        rows[day] = pd.Series(np.interp(inside, ko, po), index=inside)
        rec.update(max_violation=max(viol, 0.0),
                   n_interp=int(len(inside) - np.isin(inside, ko).sum()),
                   used=True)
        diag.append(rec)
    clean = pd.DataFrame(rows).T.reindex(columns=cols).sort_index()
    return clean, pd.DataFrame(diag).set_index("day")


def diagnose_absorption(curves: pd.DataFrame, oil_close: pd.Series) -> pd.DataFrame:
    """核对"钉价识别出的吸收"与"已实现油价真的触碰了障碍"是否一致。

    这是对 `PINNED_HI/LO` 这个阈值口径的**外部验证**，不是装饰：阈值法
    只看 Polymarket 自己的报价，如果它识别出的吸收与已实现 WTI 走势对不上，
    说明阈值选错了、或者阶梯参照的根本不是 WTI 收盘，那必须知道。

    检验的不变式是

        被钉住的最高 K  ≤  收盘历史最大值

    即"被判为已结算的障碍，已实现油价真的越过去了"。**只看当天真的有报价
    的行权价**：完全没挂过的 K 既不算钉住也不算存活。

    反方向（"仍在报价的最低 K 一定高于已实现最大值"）**不能**当成不变式：
    已实现口径本身有歧义——本仓库的 `CL` 序列是连续合约，阶梯参照的是
    具体交割月的官方结算价，两者在 76 美元附近差得出 0.5–1 美元，实测
    2026-03-04（收盘 76.09）时 K=75 仍报 0.909。这类边界日单独计数报出来，
    不当作错误。

    `ok=False` 的日子在报告里逐条列出，不做静默处理。
    """
    if not len(curves):
        return pd.DataFrame()
    live, pinned = drop_pinned(curves)
    ks = np.array([float(c) for c in curves.columns])
    run_max = oil_close.reindex(curves.index).cummax()
    out = []
    for day in curves.index:
        pin = pinned.loc[day].to_numpy(dtype=bool)
        liv = live.loc[day].notna().to_numpy()
        k_pin_max = float(ks[pin].max()) if pin.any() else np.nan
        k_live_min = float(ks[liv].min()) if liv.any() else np.nan
        rm = float(run_max.get(day, np.nan))
        why, borderline = "", ""
        if np.isfinite(rm):
            if np.isfinite(k_pin_max) and k_pin_max > rm:
                why = f"已结算的 K={k_pin_max:g} 高于收盘最大值 {rm:.2f}"
            if np.isfinite(k_live_min) and k_live_min <= rm:
                borderline = f"仍报价的 K={k_live_min:g} 不高于收盘最大值 {rm:.2f}"
        out.append({"day": day, "oil_run_max": rm, "K_pinned_max": k_pin_max,
                    "K_live_min": k_live_min, "n_pinned": int(pin.sum()),
                    "ok": not why, "note": why, "borderline": borderline})
    return pd.DataFrame(out).set_index("day")


def diagnose_resolution(curves: pd.DataFrame, meta: pd.DataFrame,
                        kind: str = "barrier", side: str = "high",
                        expiry: str = PRIMARY_EXPIRY) -> pd.DataFrame:
    """拿目录里的**实际结算结果**验证吸收口径。

    这是比 `diagnose_absorption` 更干净的一道验证：它不依赖任何外部价格
    序列，用的是 Polymarket 自己记录的 `winning_outcome_label`。不变式：

    * 结算为 **Yes** 的障碍，在样本期内**一定**出现过钉价（触碰后立即结算）。
    * 结算为 **No** 的障碍，**绝不**应该出现钉价——若出现，说明 `PINNED_HI`
      把一个正常的高概率报价误判成了结算。

    反过来说，若某个 Yes 的 K 从没被判为钉住，那它是在**到期日**才结算的
    （障碍在最后一天被触碰），而到期日已按 `close_at` 切掉——这种情况合理，
    单独标为 `resolved_at_expiry` 而不算失败。
    """
    if not len(curves):
        return pd.DataFrame()
    sel = meta[(meta.kind == kind) & (meta.side == side) & (meta.expiry == expiry)]
    won = sel.groupby("K").won.agg(
        lambda s: "Yes" if (s.astype(str) == "Yes").any() else
                  ("No" if (s.astype(str) == "No").any() else ""))
    _, pinned = drop_pinned(curves)
    ever = pinned.any()
    out = []
    for k in curves.columns:
        w = str(won.get(float(k), ""))
        p = bool(ever.get(k, False))
        if w == "Yes":
            ok, note = True, ("" if p else "resolved_at_expiry（到期日才触碰，该日已被切掉）")
        elif w == "No":
            ok = not p
            note = "" if ok else "结算为 No 却出现过钉价 —— PINNED_HI 误判"
        else:
            ok, note = True, "无结算记录"
        out.append({"K": float(k), "won": w or "?", "ever_pinned": p,
                    "ok": ok, "note": note})
    return pd.DataFrame(out).set_index("K")


def implied_stats(clean: pd.DataFrame, grid: list[float] | None = None,
                  at: float = 100.0) -> pd.DataFrame:
    """从每日 `S(K)` 曲线读出有经济含义的统计量。

    三个量，各自回答不同的问题，所以全出而不是挑一个：

        p_at_K        P(WTI 期间摸到 K 美元)。最直观、成交最密的那个格子。
        K50           S(K)=0.5 的行权价，单位是**美元/桶**：市场认为
                      "一半一半会摸到"的价位。概率量纲换成美元量纲，
                      跟油价直接可比，这是阶梯相对于单个概率市场的主要好处。
        exp_exceed    ∫S(K)dK = E[min(max, K_hi)] − K_lo 的非负部分，即
                      "期间最高价超出网格下界的期望美元数，在网格上界截断"。
                      把整条曲线压成一个数。

    ## 两类统计量用两套行权价，这是有意的

    `exp_exceed` 是**在一个定义域上的积分**：积分区间变了，数就变了，
    而那个变化与油价无关。所以它只在 `grid`（固定网格）上算，且要求
    当天把网格**整个**覆盖住，缺一个格子就返回 NaN——用"当天有的那部分
    网格"去积分，等于每天换一个定义域，画出来是一条主要反映报价覆盖范围
    的曲线。

    `K50` 与 `p_at_K` 是**在某一点上读水平**，没有定义域问题，所以用当天
    所有还在报价的行权价，把信息用满。
    """
    if not len(clean):
        return pd.DataFrame()
    cols = np.array([float(c) for c in clean.columns], dtype=float)
    grid_arr = np.array(sorted(float(g) for g in grid), dtype=float) if grid else None
    out = []
    for day, row in clean.iterrows():
        s = row.to_numpy(dtype=float)
        ok = np.isfinite(s)
        if ok.sum() < 2:
            continue
        k, sv = cols[ok], s[ok]

        # K50：S 非增，所以在 S 上做反向插值。曲线整体落在 0.5 同侧时 K50
        # 无定义，返回 NaN 而不是夹到端点上——夹端点会造出一条贴着网格边界
        # 的假序列，而且那条假序列看起来很像"市场观点稳定在 100 美元"。
        k50 = np.nan
        if sv.min() <= 0.5 <= sv.max():
            k50 = float(np.interp(0.5, sv[::-1], k[::-1]))

        ee = np.nan
        if grid_arr is not None and np.isin(grid_arr, k).all():
            g_s = np.interp(grid_arr, k, sv)
            # numpy 1.x 只有 trapz，2.0 改名 trapezoid 且把旧名标了弃用。
            ee = float(_trapz(g_s, grid_arr))

        out.append({
            "day": day,
            "p_at_K": float(np.interp(at, k, sv)) if k[0] <= at <= k[-1] else np.nan,
            "K50": k50,
            "exp_exceed": ee,
            "K_lo": float(k[0]), "K_hi": float(k[-1]), "n_K": int(ok.sum()),
        })
    return pd.DataFrame(out).set_index("day")


def signals(stats: pd.DataFrame,
            calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """隐含分布统计量 -> 可进检验的信号（日变动）。

    用**变动**而不是水位，理由与主分析一致：水位高度自相关（近似随机游走），
    拿水位对收益率回归是伪回归的经典形式。`d_K50` 与 `d_exp_exceed` 的单位
    都是美元/桶，可以和油价自身的日变动直接比大小——这是本分析里唯一
    做得到这一点的信号。
    """
    if not len(stats):
        return pd.DataFrame()
    cols = ["p_at_K", "K50", "exp_exceed"]
    # 先把索引摊回**完整的交易日历**，再做 diff。
    #
    # 变动一律用 diff 而不是 pct_change：概率从 0.02 到 0.04 相对变动 +100%，
    # 但只含 2 个百分点的消息量。美元量纲的 K50 同理，绝对变动才是"涨了几块钱"。
    #
    # 摊回日历这一步不是整洁癖：`stats` 的索引来自 `clean_curves`，报价太稀的
    # 日子已被整行剔除，所以索引**有洞**。直接 `diff()` 会跨过洞相减，把两三天
    # 的累计变动记成一天的、并落在洞后那一天上——那天的"消息量"就被凭空放大，
    # 而事件研究恰好是挑 |Δ| 最大的日子，等于专门去挑这些假跳变。
    # 摊回日历后，洞两侧的 diff 自动是 NaN。
    if calendar is not None:
        cal = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
        idx = cal[(cal >= stats.index.min()) & (cal <= stats.index.max())]
        base = stats[cols].reindex(idx)
    else:
        base = stats[cols]
    out = base.copy()
    for c in cols:
        out[f"d_{c}"] = base[c].diff()
    return out.dropna(how="all")
