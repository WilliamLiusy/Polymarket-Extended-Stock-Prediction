"""原油概率 ↔ 航空股 这组对比分析的成图。

每个函数画一张图、返回 `(fig, table)`，由 `style.save()` 同时落 PNG 与 CSV。
返回表不是顺手加的：图上读不出精确数值，而这份分析的结论全靠数值的量级与
符号，所以表格视图是结论的一部分。

## 图形选型的理由（为什么不是别的画法）

**时序对齐图 = 上下分面，不是双轴。**要对比的三样东西量纲完全不同：
概率 ∈ [0,1]、油价 ∈ [50, 90] 美元、股价指数。放进一张双 Y 轴图里，两个刻度
怎么对齐是画图的人随手定的，而不同的对齐方式能让同一份数据看起来"高度同步"
或"完全无关"。本分析要检验的恰恰是"有没有关联"，用一个能凭手感造出相关性的
图去展示它，等于先污染了证据。所以：共享 x 轴的三个分面，各自保留自己的
真实量纲；需要放在同一轴上比较的两条（航空 vs 能源）先归一化到基期 = 100。

**领先滞后用热图，不用 11 条折线。**要读的是"哪个 k 上相关最强、k>0 与 k<0
是否对称"，也就是一张二维表。11 个 k × 若干标的画成折线会互相压住，而且
折线会诱导读者把相邻 k 之间连起来看趋势——k 是离散的滞后阶，"趋势"没有意义。

**事件研究用累计收益折线。**这里横轴是事件相对日，读的是"累计效应在事件日
前后怎么积累"，路径形状本身就是结论：事件日**之前**就开始漂移说明信息提前
泄漏（或者事件定义本身用到了未来），事件日之后才走说明是真的反应。

**IC 用累计曲线，不用逐日柱。**8 只股票的单日 IC 标准误约 0.38，逐日柱就是
一片噪声，什么也读不出。累计 IC 曲线能回答唯一重要的问题：这点 IC 是长期
稳定攒出来的，还是某几天一次性给的。
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style as st


def timeline(
    p_level: pd.Series,
    dp: pd.Series,
    oil: pd.Series,
    indexed: pd.DataFrame,
    event_dates: pd.DatetimeIndex | None = None,
    oil_name: str = "WTI 原油连续合约",
    clip_to_signal: bool = True,
    figsize: tuple[float, float] = (11.5, 8.2),
):
    """三分面时序对齐图：概率水位 / 原油价格 / 航空 vs 能源（基期=100）。

    Parameters
    ----------
    p_level, dp : pd.Series
        主题概率水位与日度变动（`index=date`）。
    oil : pd.Series
        原油期货收盘价。
    indexed : pd.DataFrame
        已归一化到 100 的股票指数，列名即图例名。最多 4 列（分类色前 4 槽）。
    event_dates : pd.DatetimeIndex | None
        要竖线标出的事件日（大 |Δp| 日）。竖线画在**所有**分面上，
        这样"概率跳变那天，油和股票在干什么"是一眼对齐的。
    clip_to_signal : bool
        把 x 轴裁到概率序列真正有数的区间。默认 True。

        为什么必须裁：股票与期货的日线一直更新到今天，Polymarket 主题概率
        却在某个市场全部到期后就断了（冒烟测试里股票到 2026-09-18，概率
        只到 2026-02-24）。不裁的话右侧有一大片只有下面两个分面有线、
        第一个分面空白的区域。三分面图唯一的作用是**对齐**，而那片区域
        里没有任何东西可对齐；把它留着，读者会误以为"概率降到 0 了"或者
        "这段时间没有事件"，而真相是这段时间**没有数据**。
    """
    if indexed.shape[1] > 4:
        raise ValueError("同一轴上最多 4 条序列；再多请拆分面或并入“其他”")

    clip_note = ""
    if clip_to_signal:
        valid = [s.dropna().index for s in (p_level, dp) if s.notna().any()]
        if valid:
            lo = min(ix[0] for ix in valid)
            hi = max(ix[-1] for ix in valid)
            # 裁掉了多少必须写在图上。悄悄裁窗口和悄悄选样本是同一种毛病。
            full = indexed.dropna(how="all").index
            if len(full) and full[-1] > hi:
                clip_note = (f"x 轴裁到概率有数的区间 {lo:%Y-%m-%d} → {hi:%Y-%m-%d}；"
                             f"股票数据本身到 {full[-1]:%Y-%m-%d}，"
                             "之后概率序列无数据，不是概率归零")
            p_level = p_level.loc[lo:hi]
            dp = dp.loc[lo:hi]
            oil = oil.loc[lo:hi]
            indexed = indexed.loc[lo:hi]
            # 重新归一化：截断后基期变了，还按老基期画的话第一个点不在 100 上，
            # y 轴标签"基期 = 100"就成了假话。
            indexed = indexed.apply(st.index_to_100)
            if event_dates is not None and len(event_dates):
                ed = pd.DatetimeIndex(event_dates)
                event_dates = ed[(ed >= lo) & (ed <= hi)]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             height_ratios=[1.0, 0.85, 1.15])

    # --- 分面 1：概率水位（单序列，slot 1）。Δp 不另开一条线——它是水位的
    # 一阶差分，画两条会让读者以为是两个独立的量。改用底部的短竖线标出跳变日。
    ax = axes[0]
    ax.plot(p_level.index, p_level.values, color=st.SERIES[0], linewidth=1.6)
    ax.fill_between(p_level.index, 0, p_level.values, color=st.SERIES[0], alpha=0.10,
                    linewidth=0)
    ax.set_ylabel("事件概率")
    ax.set_ylim(bottom=0)
    last = p_level.dropna()
    if len(last):
        st.end_label(ax, last.index[-1], last.iloc[-1], f"{last.iloc[-1]:.0%}",
                     st.SERIES[0])
    st.title(ax, "① Polymarket 原油供给中断主题概率（成交额加权）",
             "注意量纲：这是概率，不是油价。油价水平市场只有 21 个密集报价日，单独在报告 §5 里分析。")

    # --- 分面 2：油价（单序列，slot 2）
    ax = axes[1]
    ax.plot(oil.index, oil.values, color=st.SERIES[1], linewidth=1.6)
    ax.set_ylabel("美元 / 桶")
    o = oil.dropna()
    if len(o):
        st.end_label(ax, o.index[-1], o.iloc[-1], f"${o.iloc[-1]:,.0f}", st.SERIES[1])
    st.title(ax, f"② {oil_name}收盘价", "传导链条的中间一环：概率若真含油价信息，应先体现在这里。")

    # --- 分面 3：航空 vs 能源，同一轴，因为都已归一化到基期 = 100
    ax = axes[2]
    # 末端标注用 `end_labels` 一次性画完，不在循环里逐条调 `end_label`：三条线
    # 归一化到同一基期后末端常常收敛到几个点之内（141/137/135），逐条画会叠成
    # 一团。`end_labels` 在显示坐标里把它们撑开。
    tips = []
    for i, col in enumerate(indexed.columns):
        s = indexed[col].dropna()
        c = st.SERIES[i]
        ax.plot(s.index, s.values, color=c, linewidth=1.6, label=col)
        if len(s):
            tips.append((s.index[-1], s.iloc[-1], f"{col} {s.iloc[-1]:,.0f}", c))
    ax.axhline(100, color=st.AXIS, linewidth=1.0, zorder=1)
    ax.set_ylabel("指数（基期 = 100）")
    ax.legend(loc="upper left", ncol=min(4, indexed.shape[1]))
    # 放在 ylim 定下来之后：`end_labels` 要把 y 换算成像素才能按点错开，
    # 先画标注、后 `axhline` 的话换算用的是还会变的坐标范围。
    st.end_labels(ax, tips)
    st.title(ax, "③ 航空股 vs 能源股（同一轴：都归一化到基期 = 100）",
             "真实油价冲击的教科书特征是分化——能源涨、航空跌；同涨同跌说明测到的是风险偏好。")

    notes = []
    if event_dates is not None and len(event_dates):
        for a in axes:
            for d in event_dates:
                a.axvline(d, color=st.MUTED, linewidth=0.7, alpha=0.55, zorder=0)
        # 图例式说明放**图脚**，不放分面里：竖线画在三个分面上，说明它的文字
        # 放进任何一个分面都是错位的；而且第一版放在分面 1 右下角，概率曲线
        # 在 2026-01 附近正好压过去，文字读不出来。
        notes.append(f"竖线 = 概率跳变日（|Δp| 前 10%），共 {len(event_dates)} 天")
    if clip_note:
        notes.append(clip_note)

    axes[-1].set_xlabel("")
    if notes:
        st.foot(fig, "　·　".join(notes))
    table = pd.DataFrame({"p_level": p_level, "dp": dp, "oil_close": oil}).join(
        indexed.add_suffix("_idx100")
    )
    table.insert(0, "date", table.index)
    if event_dates is not None:
        table["is_event"] = table.date.isin(pd.DatetimeIndex(event_dates))
    return fig, table.reset_index(drop=True)


def lead_lag_heatmap(
    tables: dict[str, pd.DataFrame],
    signal_name: str = "Δp（供给中断概率日变动）",
    figsize: tuple[float, float] = (10.6, 0.62),
):
    """领先滞后相关热图。行 = 标的，列 = 滞后阶 k。

    `tables` 的每个值是 `pmsp.eval.assoc.lead_lag` 的返回（index=k）。

    颜色是**双向**的：相关系数有符号，蓝↔灰↔红，vmin/vmax 强制对称，
    这样灰色恰好落在 0 上。不对称的色标会把"弱正相关"画成中性色，
    那是用颜色撒谎。

    显著性不靠颜色表达——颜色已经被"强度+符号"占满了。Bonferroni 校正后
    p<0.05 的格子在数字后加 `*`，是文字标注，色觉障碍下同样可读。
    """
    names = list(tables)
    ks = sorted(set().union(*[set(t.index) for t in tables.values()]))
    M = np.full((len(names), len(ks)), np.nan)
    P = np.full((len(names), len(ks)), np.nan)
    for i, nm in enumerate(names):
        t = tables[nm]
        for j, k in enumerate(ks):
            if k in t.index:
                M[i, j] = t.loc[k, "pearson"]
                P[i, j] = t.loc[k, "p_bonferroni"]

    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.1
    vmax = max(vmax, 0.05)
    fig, ax = plt.subplots(figsize=(figsize[0], 2.4 + figsize[1] * len(names)))
    ax.grid(False)
    # edgecolor = 底色、2px：格子之间的"分隔"用底色留白，不画描边线。
    mesh = ax.pcolormesh(
        np.arange(len(ks) + 1), np.arange(len(names) + 1), M,
        cmap=st.DIVERGING, vmin=-vmax, vmax=vmax,
        edgecolors=st.SURFACE, linewidth=2,
    )
    for i in range(len(names)):
        for j in range(len(ks)):
            if not np.isfinite(M[i, j]):
                continue
            # 格子底色深的地方用白字，浅的地方用墨色——文字对比度不能靠运气
            deep = abs(M[i, j]) > 0.62 * vmax
            mark = "*" if np.isfinite(P[i, j]) and P[i, j] < 0.05 else ""
            ax.text(j + 0.5, i + 0.5, f"{M[i, j]:+.2f}{mark}",
                    ha="center", va="center", fontsize=8.8,
                    color=st.SURFACE if deep else st.INK)
    ax.set_xticks(np.arange(len(ks)) + 0.5)
    ax.set_xticklabels([f"{k:+d}" if k else "0" for k in ks])
    ax.set_yticks(np.arange(len(names)) + 0.5)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("滞后阶 k　　k>0：概率领先股价（有预测价值）　　k<0：股价领先概率（无增量）")
    cb = fig.colorbar(mesh, ax=ax, pad=0.015, aspect=14)
    cb.set_label("Pearson 相关系数", color=st.INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=st.MUTED, labelcolor=st.INK_2, labelsize=8.5)
    st.title(
        ax, f"{signal_name} 与各标的日收益的领先滞后相关",
        "格内 * = Bonferroni 校正（11 次检验）后 p<0.05。不要只挑最显著的 k 看——"
        "零假设下 11 个 k 里最大 |t| 的期望本就在 2 附近。",
    )
    table = pd.DataFrame(M, index=names, columns=[f"k={k}" for k in ks])
    table.insert(0, "标的", table.index)
    return fig, table.reset_index(drop=True)


def event_curves(
    studies: dict[str, tuple[pd.DataFrame, dict]],
    head: str = "概率跳变日前后的累计平均收益",
    sub: str | None = None,
    figsize: tuple[float, float] = (10.2, 5.4),
):
    """事件研究：多个标的的 CAR 路径画在同一轴上（单位都是累计收益率，可比）。

    `studies` 的每个值是 `assoc.event_study` 的返回 `(table, info)`。
    """
    if len(studies) > 4:
        raise ValueError("同一轴上最多 4 条序列；再多请拆分面")
    fig, ax = plt.subplots(figsize=figsize)
    rows = []
    tips = []
    for i, (nm, (tb, info)) in enumerate(studies.items()):
        if tb.empty:
            continue
        c = st.SERIES[i]
        ax.plot(tb.offset, tb.car * 100, color=c, linewidth=1.8, label=nm,
                marker="o", markersize=4.0, markeredgecolor=st.SURFACE,
                markeredgewidth=1.2)
        tips.append((tb.offset.iloc[-1], tb.car.iloc[-1] * 100,
                     f"{nm} {tb.car.iloc[-1] * 100:+.2f}%", c))
        t = info.get("t_nw", float("nan"))
        rows.append({"标的": nm, "事件数": info.get("n_events"),
                     "阈值": info.get("threshold"),
                     "事件后累计收益": info.get("car_event_to_post"),
                     "t_NW": t, "p_NW": info.get("p_nw")})
    st.zero_line(ax)
    ax.axvline(0, color=st.AXIS, linewidth=1.0, zorder=1)
    ax.text(0, ax.get_ylim()[1], " 事件日", color=st.MUTED, fontsize=8.5,
            va="top", ha="left")
    ax.set_xlabel("相对事件日的交易日")
    ax.set_ylabel("累计平均收益（%）")
    # 图例放画布外下方：轴内任何一个固定位置都可能被某条曲线压住，而曲线
    # 形状是数据决定的、每次跑都不一样。放外面是唯一确定不会撞的选择。
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center",
               ncol=min(4, len(studies)))
    # 末端直接标注要伸到轴外，右侧留出余量
    ax.set_xlim(right=float(max(tb.offset.max() for tb, _ in studies.values()
                                if not tb.empty)) + 0.4)
    # 末端标注放在 xlim/ylim 都定下来之后，且用 `end_labels` 而不是逐条：
    # 两条 CAR 曲线的末值是数据决定的，撞不撞不该靠运气。
    st.end_labels(ax, tips)
    n = max((info.get("n_events", 0) for _, info in studies.values()), default=0)
    st.title(ax, head, sub or (
        f"共 {n} 个事件；累计收益锚定在事件日前一天 = 0。窗口会重叠（一场冲突连着十几天），"
        "所以 t 值用 Newey-West。事件日之前就出现漂移是警号——那说明事件定义里混进了未来信息。"))
    return fig, pd.DataFrame(rows)


def ic_cumulative(
    ic_series: dict[str, pd.Series],
    summaries: dict[str, dict],
    figsize: tuple[float, float] = (10.2, 5.0),
):
    """累计横截面 IC 曲线。

    读法：斜率 = 平均 IC。一条稳定向上的直线说明信号一直有效；
    一条在某个区间陡升、其余时间走平的曲线说明"IC 均值为正"只是
    某几天的一次性贡献，那不是可用的信号。
    """
    fig, ax = plt.subplots(figsize=figsize)
    rows = []
    tips = []
    for i, (nm, s) in enumerate(ic_series.items()):
        if s is None or not len(s):
            continue
        c = st.SERIES[i]
        cum = s.cumsum()
        ax.plot(cum.index, cum.values, color=c, linewidth=1.7, label=nm)
        tips.append((cum.index[-1], cum.iloc[-1], f"{nm}", c))
        sm = summaries.get(nm, {})
        rows.append({"口径": nm, "天数": sm.get("n_days"),
                     "IC均值": sm.get("ic_mean"), "IC标准差": sm.get("ic_std"),
                     "ICIR": sm.get("icir"), "t_NW": sm.get("t_nw"),
                     "p_NW": sm.get("p_nw"), "IC胜率": sm.get("ic_win_rate"),
                     "截面宽度": sm.get("mean_xs_width")})
    st.zero_line(ax)
    ax.set_ylabel("累计日度 IC")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center",
               ncol=min(4, len(ic_series)))
    st.end_labels(ax, tips)
    st.title(ax, "累计横截面 IC（曲线的斜率就是平均 IC）",
             "横截面只有 6–8 只股票，单日 IC 标准误约 1/√7 ≈ 0.38——所以看累计曲线的"
             "形状，不要看任何单日的值，也不要把它和全市场 1500 只的 IC 比大小。")
    return fig, pd.DataFrame(rows)


def triangle_chart(tri: pd.DataFrame, figsize: tuple[float, float] = (10.2, 4.6)):
    """三角验证：三段链路的相关系数 + HAC 95% 置信区间。

    误差棒用的是 Newey-West 标准误的 ±1.96 倍。画出区间而不是只标星号，
    是因为这份分析里"效应小到测不出"和"效应确实为零"必须能被区分开——
    一个跨越 0 但很窄的区间是真的零，一个跨越 0 但很宽的区间只是没功效。
    """
    legs = list(dict.fromkeys(tri["链路"]))
    horizons = list(dict.fromkeys(tri["口径"]))
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", visible=False)
    h = 0.34
    for j, hz in enumerate(horizons):
        c = st.SERIES[j]
        sub = tri[tri["口径"] == hz].set_index("链路")
        ys, xs, err = [], [], []
        for i, leg in enumerate(legs):
            if leg not in sub.index:
                continue
            ys.append(i + (j - (len(horizons) - 1) / 2) * h)
            xs.append(sub.loc[leg, "pearson"])
            err.append(1.96 * sub.loc[leg, "se_nw"])
        ax.barh(ys, xs, height=h * 0.86, color=c, label=hz, zorder=3)
        ax.errorbar(xs, ys, xerr=err, fmt="none", ecolor=st.INK_2,
                    elinewidth=1.0, capsize=3.0, zorder=4)
        for y, x, e in zip(ys, xs, err):
            off = 0.012 * (1 if x >= 0 else -1)
            ax.text(x + e * np.sign(x if x else 1) + off, y, f"{x:+.3f}",
                    va="center", ha="left" if x >= 0 else "right",
                    fontsize=8.8, color=st.INK_2)
    st.zero_line(ax, axis="x")
    ax.set_yticks(range(len(legs)))
    ax.set_yticklabels(legs)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson 相关系数（误差棒 = Newey-West 95% 置信区间）")
    ax.legend(loc="lower right", ncol=len(horizons))
    st.title(ax, "三角验证：把「概率 → 股价」拆成两段分别检验",
             "只有前两段都显著时，第三段的零结果才能读作"
             "“信息已被油价吸收”；第一段就不显著的话，链条在起点就断了。")
    return fig, tri.copy()


def market_universe(audit: pd.DataFrame, top: int = 14,
                    figsize: tuple[float, float] = (10.6, 5.6)):
    """原油关键词命中市场的成交额排序条形图：本分析最重要的一张背景图。

    它回答的是"分析对象到底是什么"。用两种颜色区分纳入/排除，
    并把排除理由直接写在条形末端——这张图的全部意义就是让人看见
    "Polymarket 上没有油价市场，最大的油相关市场是霍尔木兹海峡封锁"，
    以及"关键词命中里成交额最高的几个是 NHL 油人队和加拿大政治人物"。
    """
    d = audit.head(top).copy()
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", visible=False)
    colors = [st.SERIES[0] if inc else st.MUTED for inc in d.included]
    ax.barh(range(len(d)), d.vol_usdc, color=colors, height=0.72, zorder=3)
    ax.set_xscale("symlog", linthresh=100)
    for i, (v, inc, why) in enumerate(zip(d.vol_usdc, d.included, d.reason)):
        txt = f"  ${v:,.0f}" + ("" if inc else f"　{why}")
        ax.text(v, i, txt, va="center", ha="left", fontsize=8.5, color=st.INK_2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([s[:52] for s in d.market_slug], fontsize=8.5)
    ax.set_xlabel("累计成交额（USDC，symlog 轴）")
    ax.set_xlim(right=float(d.vol_usdc.max()) * 12)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=st.SERIES[0]),
        plt.Rectangle((0, 0), 1, 1, color=st.MUTED),
    ]
    ax.legend(handles, ["纳入原油主题", "排除（理由标在条形右侧）"],
              loc="lower right")
    st.title(ax, "关键词命中的“原油相关”市场，按成交额排序",
             "Polymarket 上没有流动性足够的油价水平市场；油相关的钱几乎全在"
             "地缘供给中断风险上。灰条是子串假阳性——它们的成交额并不小。")
    cols = [c for c in ["market_slug", "vol_usdc", "included", "theme", "reason"]
            if c in audit.columns]
    return fig, audit[cols].copy()


def ladder_surface(
    clean: pd.DataFrame,
    pinned: pd.DataFrame,
    oil_close: pd.Series,
    k50: pd.Series | None = None,
    figsize: tuple[float, float] = (11.0, 5.6),
):
    """行权价阶梯的隐含生存函数曲面：x = 交易日，y = 行权价（美元/桶），
    颜色 = P(期间最高价 ≥ K)。

    ## 为什么这张图可以把已实现油价直接画在热图上

    通常"热图 + 折线"是要挨骂的，因为折线的 y 轴与热图的行索引没有共同单位，
    叠在一起是两套坐标硬凑。这里是个例外：**纵轴两边都是美元/桶**。行权价
    是美元，已实现 WTI 收盘价也是美元，所以已实现价格穿过哪一档行权价是有
    确切含义的——它正好画出"哪些障碍被触碰了"。这不是双轴，是同一根轴。

    ## 颜色用单向色标

    概率是"多与少"，不是"正与负"，所以一个色相由浅到深（`style.SEQUENTIAL`），
    不用双向标。这里刻意**不**把 0.5 设成中性色：0.5 在生存函数上不是分界点，
    没有"以下为负"的含义，给它一个中点会凭空造出一条语义边界。

    ## 已吸收的格子画成斜纹，不画成白色

    留白等于"没有数据"，而已吸收的格子是**有数据而且数据等于 1**——障碍已被
    触碰、合约已结算。两件事在图上必须能分开：斜纹 = 已结算，留白 = 没报价。
    """
    if not len(clean):
        raise ValueError("clean 为空，无法出图")
    ks = np.array([float(c) for c in clean.columns], dtype=float)
    dates = pd.DatetimeIndex(clean.index)

    # pcolormesh 要的是**格子边界**，比中心点多一个。行权价网格是不等距的
    # （100,105,110,120,...,200），所以边界取相邻中点，不能用固定步长——
    # 用固定步长会让 180→200 那一档在图上和 100→105 一样宽，读者会以为
    # 行权价是等距的。
    def edges(v: np.ndarray) -> np.ndarray:
        mid = (v[:-1] + v[1:]) / 2.0
        return np.concatenate([[v[0] - (mid[0] - v[0])], mid,
                               [v[-1] + (v[-1] - mid[-1])]])

    xnum = mpl.dates.date2num(dates)
    xe = edges(xnum) if len(xnum) > 1 else np.array([xnum[0] - 0.5, xnum[0] + 0.5])
    ye = edges(ks)

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    # 格子描边用 GRID 而不是 SURFACE。规范要求相邻填充之间有底色留白，
    # 但在这张图上"有格子"本身是信息：概率真的接近 0 的格子填色极淡，
    # 若描边也是底色，它与"当天没有报价"在画布上完全一样。一层很淡的
    # 灰描边同时做到分隔与"此处有数据"。NaN 格子不会被画出来，自然无边。
    mesh = ax.pcolormesh(xe, ye, clean.to_numpy(dtype=float).T,
                         cmap=st.SEQUENTIAL, vmin=0.0, vmax=1.0,
                         edgecolors=st.GRID, linewidth=1.0)

    # 已吸收（障碍已触碰、合约已结算）的格子打斜纹
    pin = pinned.reindex(index=clean.index, columns=clean.columns).fillna(False)
    n_pin = 0
    for j, k in enumerate(ks):
        for i in range(len(dates)):
            if not bool(pin.iloc[i, j]):
                continue
            n_pin += 1
            ax.add_patch(plt.Rectangle(
                (xe[i], ye[j]), xe[i + 1] - xe[i], ye[j + 1] - ye[j],
                facecolor=st.GRID, edgecolor=st.MUTED, linewidth=0.6,
                hatch="///", zorder=2))

    ax.plot(dates, oil_close.reindex(dates), color=st.SERIES[1], linewidth=2.2,
            zorder=4, label="已实现 WTI 收盘", solid_capstyle="round")
    if k50 is not None and k50.notna().any():
        ax.plot(dates, k50.reindex(dates), color=st.INK, linewidth=2.0,
                linestyle=(0, (4, 2)), zorder=4,
                label="隐含 K50（市场认为五成机会摸到的价位）")

    ax.set_ylabel("行权价 / 油价（美元每桶）", color=st.INK_2)
    ax.set_xlabel("")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.09), ncol=2,
                    frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(st.INK_2)
    cb = fig.colorbar(mesh, ax=ax, pad=0.015, aspect=22)
    cb.set_label("P(期间最高价 ≥ 行权价)", color=st.INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=st.MUTED, labelcolor=st.INK_2, labelsize=8.5)
    st.title(
        ax, "Polymarket 原油阶梯还原出的隐含分布（到期 2026-03-31）",
        f"纵轴两边都是美元每桶，所以已实现价格线与行权价格子在同一根轴上可比。"
        f"斜纹格 = 障碍已触碰、合约已结算（{n_pin} 格），留白 = 当日无报价。",
    )
    table = clean.copy()
    table.insert(0, "date", table.index.strftime("%Y-%m-%d"))
    table.insert(1, "WTI收盘", oil_close.reindex(dates).to_numpy())
    return fig, table.reset_index(drop=True)


def ladder_scatter(
    sig: pd.Series,
    ret: pd.Series,
    stats_by_k: pd.DataFrame,
    ks: tuple[int, ...] = (0, 1),
    signal_name: str = "隐含 K50 日变动（美元/桶）",
    figsize: tuple[float, float] = (10.4, 4.6),
):
    """同期 vs 领先一日的散点对照：一眼看出"同期强、领先为零"。

    ## 为什么用散点而不是又一张热图

    热图（`lead_lag_heatmap`）答的是"哪个 k 强"，已经有了。这里要答的是
    另一个问题：**这个 r = −0.86 是一条真实的线性关系，还是被一两个极端点
    撬出来的**。只有散点能答——21 个点的相关系数完全可能由单点杠杆造成，
    而热图里的 −0.86 和散点里"18 个点排成一条线"的 −0.86 是完全不同的证据。

    两个分面共享 x/y 轴范围，否则 k=1 那张会被自动缩放拉满，视觉上和 k=0
    一样"有结构"——而它实际上是一团噪声。
    """
    ks = tuple(ks)
    pairs = {}
    for k in ks:
        d = pd.DataFrame({"x": sig, "y": ret.shift(-k)}).dropna()
        pairs[k] = d
    allx = pd.concat([d.x for d in pairs.values()])
    ally = pd.concat([d.y for d in pairs.values()])

    def pad(v: pd.Series) -> tuple[float, float]:
        lo, hi = float(v.min()), float(v.max())
        m = (hi - lo) * 0.10 or 1.0
        return lo - m, hi + m

    xlim, ylim = pad(allx), pad(ally * 100)

    fig, axes = plt.subplots(1, len(ks), figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, k in zip(axes, ks):
        d = pairs[k]
        y = d.y * 100
        st.zero_line(ax, "y")
        st.zero_line(ax, "x")
        col = st.SERIES[0] if k == 0 else st.SERIES[3]
        ax.plot(d.x, y, "o", color=col, markersize=8, alpha=0.85,
                markeredgecolor=st.SURFACE, markeredgewidth=1.4, zorder=3)
        r = stats_by_k.loc[k, "pearson"] if k in stats_by_k.index else np.nan
        p = stats_by_k.loc[k, "p_perm"] if k in stats_by_k.index else np.nan
        if len(d) > 2 and np.isfinite(r):
            b = np.polyfit(d.x, y, 1)
            xs = np.linspace(*xlim, 50)
            ax.plot(xs, np.polyval(b, xs), color=col, linewidth=1.6,
                    alpha=0.55, zorder=2)
        lab = "同期（k=0）" if k == 0 else f"信号领先 {k} 日（k=+{k}）"
        st.title(ax, lab,
                 f"r = {r:+.2f}　置换 {st.fmt_p(p)}　n = {len(d)}"
                 if np.isfinite(r) else "样本不足")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(signal_name, color=st.INK_2)
        for _, rr in d.iterrows():
            rows.append({"k": k, "signal": rr.x, "airline_ret_pct": rr.y * 100})
    axes[0].set_ylabel("航空股等权日收益（%）", color=st.INK_2)
    st.foot(fig, _scatter_foot(stats_by_k, ks), width=52)
    return fig, pd.DataFrame(rows)


def _scatter_foot(stats_by_k: pd.DataFrame, ks: tuple[int, ...],
                  alpha: float = 0.05) -> str:
    """散点图脚的措辞，**从统计表里读出来**，不写死。

    这一条是踩出来的：第一版把"同期有关系、领先一日没有"直接写进图脚，那是
    拿美股那一遍的形态（k=0 的 p < 0.001、k=+1 的 p = 0.79）当成两个市场的
    共同结论。同一个函数画 A 股时，k=0 的置换 p 是 0.052 —— 图上印着"同期
    有关系"，而它旁边自己的副标题写着 p = 0.052。图脚和副标题打架，读者只能
    二选一信一个，那这张图就不能用了。
    """
    def g(k, col):
        return (float(stats_by_k.loc[k, col])
                if k in stats_by_k.index else float("nan"))

    parts = []
    r0, p0 = g(0, "pearson"), g(0, "p_perm")
    if 0 in ks and np.isfinite(p0):
        parts.append(f"同期{'有关系' if p0 < alpha else '未达显著'}"
                     f"（r = {r0:+.2f}，{st.fmt_p(p0)}）")
    leads = [k for k in ks if k > 0 and np.isfinite(g(k, "p_perm"))]
    if leads:
        sig = [k for k in leads if g(k, "p_perm") < alpha]
        if not sig:
            parts.append("领先各期都没有")
        else:
            parts.append("领先期里 k=" + "、".join(f"+{k}" for k in sig) + " 也显著")
    if parts:
        tail = ("——两个市场在同时消化同一条新闻，预测市场没有提前于股价。"
                if (np.isfinite(p0) and p0 < alpha and leads
                    and not [k for k in leads if g(k, "p_perm") < alpha])
                else "。")
        head = "、".join(parts) + tail
    else:
        head = "样本不足，未作推断。"
    # 可检出下限直接取表里的值，不写死 0.6：它随 n 变，而 n 两个市场不一样。
    floors = [g(k, "detectable_r") for k in ks]
    floors = [f for f in floors if np.isfinite(f)]
    if floors:
        head += (f"注意本窗口只有二十来天，只能排除很大的效应"
                 f"（八成把握的可检出下限 |r| > {max(floors):.2f}），排不掉小的。")
    return head
