"""Polymarket 原油市场 ↔ 航空股日频数据的关联性分析（出图 + 出报告）。

    python scripts/11_oil_airline_analysis.py
    python scripts/11_oil_airline_analysis.py --min-volume 20000   # 只用大市场
    python scripts/11_oil_airline_analysis.py --start 2024-01-01

依赖 `scripts/10_fetch_polymarket.py` 先把小时面板拉好。

这个脚本只做三件事：装数据、调 `pmsp.eval.oil_airline_study.run`、把结果渲染成
图与 Markdown。所有口径与统计逻辑都在模块里（`pmsp.build.pm_daily`、
`pmsp.build.oil_markets`、`pmsp.eval.assoc`、`pmsp.eval.oil_airline_study`、
`pmsp.viz.oil_airline`），这样换个主题（比如「美联储议息概率 ↔ 银行股」）
只需要换 `oil_markets` 的正则和 `MarketSpec` 的成分股，检验和出图不用动。

**A 股与美股各跑一遍，口径逐字相同。**美股那一遍是阳性对照：它与 Polymarket
同时区，如果连它都测不出关联，A 股的零结果才能读作「信号弱」而不是「时区
错配把信号磨没了」。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from pmsp.build import oil_ladder, oil_markets, pm_daily
from pmsp.build import stock_panels as sp
from pmsp.eval import ladder_study
from pmsp.eval import oil_airline_study as study
from pmsp.viz import oil_airline as viz
from pmsp.viz import style

PM_DIR = Path("data/raw/polymarket")
OUT_DIR = Path("reports")
FIG_DIR = OUT_DIR / "figs"
REPORT = OUT_DIR / "report_polymarket_oil_airline.md"


# --------------------------------------------------------------- Markdown 渲染

def md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}", max_rows: int = 40) -> str:
    """DataFrame -> GitHub Markdown 表。只格式化，不做任何筛选或排序。"""
    if df is None or df.empty:
        return "_（无数据）_\n"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else floatfmt.format(v))
        else:
            d[c] = d[c].astype(str)
    sep = " | "
    head = "| " + sep.join(str(c) for c in d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    body = "\n".join(
        "| " + sep.join(str(v) for v in row) + " |"
        for row in d.itertuples(index=False)
    )
    extra = ""
    if len(df) > max_rows:
        extra = (f"\n\n_（共 {len(df)} 行，此处显示前 {max_rows} 行；"
                 f"完整结果见同名 CSV）_\n")
    return "\n".join([head, rule, body]) + extra + "\n"


def verdict(r: float, t: float, p_bonf: float, n: int) -> str:
    """把一个 (相关系数, t, 校正 p, n) 判成一句人话，并带上可检出下限。

    零结果必须配上「这个样本量下能检出多少」，否则「不显著」是句空话。
    n<20 时 `corr_with_t` 直接返回 NaN，不会给出一个靠几个点算出来的相关。
    """
    mde = study.detectable(n)
    if not np.isfinite(t):
        return f"样本不足（n={n}）"
    if p_bonf < 0.05:
        return f"**显著**（r={r:+.3f}, t={t:+.2f}, Bonf p={p_bonf:.3f}）"
    if p_bonf < 0.20:
        return f"边缘（r={r:+.3f}, t={t:+.2f}, Bonf p={p_bonf:.2f}）"
    tail = "真的接近零" if abs(r) < mde / 2 else "功效不足，无法区分零与小效应"
    return (f"不显著（r={r:+.3f}, t={t:+.2f}）；n={n} 下 80% 功效的可检出下限 "
            f"|r|≈{mde:.3f}，所以这是{tail}")


# ------------------------------------------------------------------- 数据装配

def load_polymarket(
    min_volume: float, pm_dir: Path = PM_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    hourly = pd.read_parquet(pm_dir / "hourly_panel.parquet")
    catalog = pd.read_parquet(pm_dir / "market_catalog.parquet")
    vol = hourly.groupby("condition_id").vol_usdc.sum()
    audit = oil_markets.audit_table(catalog, volume=vol, min_volume=min_volume)
    sel = oil_markets.selected_ids(audit, oil_markets.OIL_THEMES)
    return hourly, catalog, audit, sel


def run_ladder(hourly: pd.DataFrame, catalog: pd.DataFrame,
               spec: study.MarketSpec, calendar: pd.DatetimeIndex,
               wti_px: pd.Series, wti_ret: pd.Series,
               controls: pd.DataFrame, control_labels: dict[str, str],
               expiry: str, n_perm: int) -> dict:
    """行权价阶梯这一路：还原隐含分布 -> 三道提取校验 -> 关联检验。

    这一路与 §2/§3 的主分析**标的不同**：主分析用的是"地缘供给中断概率"，
    离油价隔着一层传导；这里的标的直接就是油价水平。代价是样本只有阶梯密集
    报价的那二十来天，所以两路都报、都不挑。

    诊断一律拿 **WTI（CL）** 核对，与 `spec.oil_px` 无关：阶梯的标的合约是
    CL，A 股那一遍的燃油基准虽然贴布伦特，但"障碍有没有被触碰"这件事只能
    用 CL 去验。
    """
    # 先把目录缩到原油市场再解析：`parse_ladder` 是逐行正则，对着九十万行
    # 全量目录跑纯属浪费，而它本来也只认 `crude-oil-*` 这些 slug。
    slug = catalog.market_slug.fillna("")
    cat = catalog[slug.str.contains("crude-oil", case=False, na=False)]
    cat = cat.drop_duplicates("condition_id")
    meta = oil_ladder.parse_ladder(cat)

    curves, weights = oil_ladder.daily_curves(
        hourly, meta, calendar, market=spec.key, expiry=expiry)
    if not len(curves):
        return {}
    live, pinned = oil_ladder.drop_pinned(curves)
    grid = oil_ladder.fixed_grid(live)
    clean, diag = oil_ladder.clean_curves(curves, weights, grid)
    stats = oil_ladder.implied_stats(clean, grid=grid)
    sig = oil_ladder.signals(stats, calendar=calendar)

    res = ladder_study.run(sig, spec.airlines, wti_ret, controls=controls,
                           control_labels=control_labels, n_perm=n_perm)
    res.update(
        spec=spec, meta=meta, curves=curves, pinned=pinned, clean=clean,
        diag=diag, stats=stats, sig=sig, grid=grid, expiry=expiry,
        wti_px=wti_px,
        # 三道相互独立的提取校验，缺一不可：结算结果核对吸收口径、
        # 已实现油价核对障碍不变式、重挂检查"按 K 取末笔"是否唯一。
        resolution=oil_ladder.diagnose_resolution(curves, meta, expiry=expiry),
        absorption=oil_ladder.diagnose_absorption(curves, wti_px),
        duplicates=oil_ladder.duplicate_listings(
            hourly, meta, calendar, market=spec.key, expiry=expiry),
    )
    return res


def build_specs() -> tuple[dict[str, study.MarketSpec], dict[str, tuple],
                           pd.Series, pd.Series]:
    """两个市场的口径 + 各自的对照资产（给阶梯那一路的分化检验用）。"""
    cn_air, cn_eng, names = sp.cn_returns()
    us_air, us_ctl = sp.us_returns()
    oil_px, oil_ret = sp.oil_returns()
    cn_mkt = pd.read_parquet("data/raw/cn_market_ret.parquet").mkt_ret

    cn_ctl = pd.DataFrame({"能源等权": sp.equal_weight_index(cn_eng, min_names=2),
                           "沪深全A": cn_mkt})
    controls = {
        "cn": (cn_ctl, {"能源等权": "能源", "沪深全A": "大盘"}),
        "us": (us_ctl, ladder_study.US_CONTROLS),
    }
    specs = {
        "cn": study.MarketSpec(
            key="cn", label="A股", airlines=cn_air,
            energy=sp.equal_weight_index(cn_eng, min_names=2),
            market=cn_mkt,
            # A 股航油采购贴布伦特，不贴 WTI
            oil_ret=oil_ret["OIL"], oil_px=oil_px["OIL"], oil_label="布伦特原油",
            names=names,
        ),
        "us": study.MarketSpec(
            key="us", label="美股", airlines=us_air,
            energy=us_ctl["XLE"], market=us_ctl["SPY"],
            oil_ret=oil_ret["CL"], oil_px=oil_px["CL"], oil_label="WTI原油",
            etf=us_ctl["JETS"], etf_label="JETS航空ETF",
        ),
    }
    # WTI 单独返回：阶梯的标的合约是 CL，A 股那一遍的燃油基准虽然贴布伦特，
    # "障碍有没有被触碰"这件事只能拿 CL 去核对。
    return specs, controls, oil_px["CL"], oil_ret["CL"]


# ----------------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-volume", type=float, default=5_000.0,
                    help="纳入原油主题的单市场最低累计成交额（USDC）")
    ap.add_argument("--event-quantile", type=float, default=0.90)
    # 行权价阶梯那一路（§6）。默认跟着主分析一起跑；到期日可换（阶梯按到期
    # 分组，只有 end-of-march 那一组在样本里密集报价，见 `oil_ladder` 文档）。
    ap.add_argument("--ladder-expiry", default=oil_ladder.PRIMARY_EXPIRY)
    ap.add_argument("--ladder-nperm", type=int, default=ladder_study.N_PERM,
                    help="阶梯检验的置换次数；经验 p 的分辨率是 1/(n+1)")
    ap.add_argument("--no-ladder", action="store_true",
                    help="跳过阶梯那一路（只要主分析时用）")
    # 面板目录可换：全量下载还在跑的时候，可以先用部分文件聚合出来的面板
    # 把整条链路跑通（口径、出图、报告），不必等 16.8 GB 下完。
    ap.add_argument("--pm-dir", default=str(PM_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    report = out_dir / "report_polymarket_oil_airline.md"

    style.use()
    fig_dir.mkdir(parents=True, exist_ok=True)
    # 清掉上一轮的图。**这不是洁癖，是正确性问题。**报告按固定文件名引用图，
    # 一旦编号或纳入的市场变了，上一轮的 PNG 会原地留下，报告里那个
    # `![...](figs/09_us_event_up.png)` 就指向一张**用旧口径算出来的图**，
    # 而且看起来完全正常。冒烟测试里就同时躺着两套编号的 12 张陈图。
    # 只删本脚本自己产的 PNG/CSV，不动目录里的其他东西。
    stale = sorted(p for p in fig_dir.iterdir()
                   if p.suffix in {".png", ".csv"} and p.name[:2].isdigit())
    for p in stale:
        p.unlink()
    if stale:
        print(f"  清掉上一轮 {len(stale)} 个图/表文件", flush=True)

    print("装载 Polymarket 面板 ...", flush=True)
    hourly, catalog, audit, sel = load_polymarket(args.min_volume, Path(args.pm_dir))
    print(f"  关键词命中 {len(audit)} 个市场，纳入 {len(sel)} 个", flush=True)

    fig, tb = viz.market_universe(audit)
    style.save(fig, fig_dir / "01_market_universe.png", tb)

    hourly_sel = hourly[hourly.condition_id.isin(sel)]
    print(f"  纳入市场的小时记录 {len(hourly_sel):,} 行", flush=True)

    specs, controls, wti_px, wti_ret = build_specs()
    results: dict[str, dict] = {}
    signals: dict[str, pd.DataFrame] = {}
    ladders: dict[str, dict] = {}

    for key, spec in specs.items():
        print(f"\n=== {spec.label} ===", flush=True)
        cal = pd.DatetimeIndex(spec.airlines.index)
        cal = cal[cal >= pd.Timestamp(args.start)]
        if args.end:
            cal = cal[cal <= pd.Timestamp(args.end)]
        daily = pm_daily.to_daily(hourly_sel, cal, market=spec.key)
        theme = pm_daily.aggregate_theme(daily)
        sig = study.build_signals(theme)
        signals[key] = sig
        print(f"  日频信号 {len(sig)} 天，Δp 非缺 {sig.dp.notna().sum()} 天，"
              f"活跃市场中位数 {sig.n_markets.median():.0f}", flush=True)
        results[key] = study.run(spec, sig, event_q=args.event_quantile)

        if not args.no_ladder:
            ctl, ctl_labels = controls[key]
            lad = run_ladder(hourly, catalog, spec, cal, wti_px, wti_ret,
                             ctl, ctl_labels, args.ladder_expiry, args.ladder_nperm)
            if lad:
                ladders[key] = lad
                print(f"  阶梯：{len(lad['sig'])} 个交易日，网格 {lad['grid']}，"
                      f"结算口径不一致 {int((~lad['resolution'].ok).sum())} 例",
                      flush=True)
            else:
                print(f"  阶梯：到期 {args.ladder_expiry} 在本窗口内无报价，跳过",
                      flush=True)

    render(args, audit, sel, specs, signals, results, ladders, fig_dir, report)
    print(f"\n报告：{report}\n图：{fig_dir}/*.png（每张图旁边有同名 CSV）")


def render_ladder(A, ladders: dict[str, dict], args) -> None:
    """§5：行权价阶梯那一路的报告正文。`A` 是 `parts.append`。"""
    if not ladders:
        A("## 5. 行权价阶梯：直接的「油价预期」\n")
        A("_本轮没有可用的阶梯样本（`--no-ladder`，或所选到期在窗口内无报价）。_\n")
        return

    n_days = max(len(l["sig"]) for l in ladders.values())
    # 提取质量（校验、曲面）只按一个口径报：美股截断与阶梯同时区，是最干净的
    # 那一遍。各市场自己的网格与样本量在 5.4 / 5.6 各自的抬头行里。
    ref = ladders.get("us") or next(iter(ladders.values()))
    diag = ref["diag"]
    A(f"## 5. 行权价阶梯：直接的「油价预期」（{n_days} 个交易日）\n")
    A(f"到期 `{ref['expiry']}`　|　提取质量按 {ref['spec'].label} 口径报　|　"
      f"可用 {int(diag.used.sum())}/{len(diag)} 个交易日\n")

    A("### 5.1 阶梯在问什么（读图之前必须先弄清的一件事）\n")
    A("`will-crude-oil-cl-hit-high-K-by-T` 的赔付条件是「到 T 之前 WTI **触碰过** "
      "K」，所以整条曲线 `S(K) = P(期间最高价 ≥ K)` 还原的是**期间最高价**的"
      "分布，不是到期价的分布。这不是措辞讲究：`K50`（市场认为五成机会摸到的"
      "价位）因此系统性地**高于**当时的现货价，图上那条虚线一直压在油价线上方，"
      "是对的，不是错位。\n")
    A("由无套利，`S(K)` 必须随 K 非增（能摸到 110 就一定摸到过 100）。"
      "报价违反这一条时用**成交额加权的保序回归（PAVA）**修正，不用 `cummin`——"
      "后者把全部违反量都摊给高行权价那一侧，而违反究竟出在哪一侧是未知的。"
      f"本轮最大违反量 {float(diag.max_violation.max()):.4f}。\n")
    A("![阶梯隐含分布](figs/13_ladder_surface.png)\n")
    A("斜纹格 = **障碍已触碰、合约已结算**。障碍一旦触碰就立即结算并被钉在 "
      "0.999，此后不再含任何前瞻信息，它的日变动是假跳变，必须剔除。"
      "**只有上侧有这个阈值（p ≥ 0.99），下侧没有**：障碍型合约在到期前"
      "不可能裁定 No，所以 p ≤ 0.01 是一个货真价实的深度虚值报价"
      "（现货 90 时「摸到 200」的确该报 0.004），加个下界会把正常报价当成"
      "结算剔掉——这个坑本轮踩过一次，后果是把网格从 9 个行权价打到 3 个，"
      "积分区间从 100 美元宽缩到 20 美元宽，而那种缩窄读起来像是"
      "「市场预期的油价在下降」。\n")

    A("### 5.2 三道提取校验（这一节是在验代码，不是在报结论）\n")
    A("还原出来的分布对不对，有三个**相互独立**的验法，都必须过：\n")
    res = ref["resolution"]
    ok = int((~res.ok).sum())
    A(f"**（a）拿实际结算结果验吸收口径。**目录里有每个市场的 "
      f"`winning_outcome_label`：结算为 Yes 的行权价必须在到期前被钉住过，"
      f"结算为 No 的必须从未被钉住。本轮 **{ok}/{len(res)} 例不一致**"
      f"{'——阈值 0.99 就是这条分界本身' if ok == 0 else '（逐条见下表）'}。\n")
    A(md_table(res.reset_index(names="K"), max_rows=20))
    ab = ref["absorption"]
    A(f"**（b）拿已实现油价验障碍不变式。**被钉住的最高行权价必须 ≤ 期间"
      f"已实现最高价。本轮硬违反 {int((~ab.ok).sum())} 天、边界 "
      f"{int((ab.borderline != '').sum())} 天。反方向（已实现价越过了某个"
      "行权价、而它没被钉住）只记作 `borderline` 不判错：CL 连续合约不是"
      "阶梯的参照合约，两者差一个换月基差。\n")
    dup = ref["duplicates"]
    A(f"**（c）重挂检查。**同一个行权价会被重新挂牌（油价回落到障碍下方时），"
      f"于是同一天同一个 K 可能有多个 `condition_id`。本轮同日重叠 "
      f"**{len(dup)} 例**，"
      f"{'所以「按 K 取末笔」是唯一解' if not len(dup) else '取末笔依赖排序，已显式按 (hour, vol) 排序'}。\n")
    val = ref["validate"]
    if not val.empty:
        A("**（d）信号 vs 已实现 WTI 收益。**这一条同样是**校验而不是发现**："
          "阶梯是油价的衍生品，隐含分布的日变动与当日油价收益本来就该高度相关，"
          "**不高才说明还原错了**。\n")
        A(md_table(val))

    A("### 5.3 为什么这里不用 Newey-West t 值，改用置换检验\n")
    A("`assoc.corr_with_t` 在 `n < 20` 时直接返回 NaN，这个下限是对的，"
      "**本节不降低它**：HAC 标准误在二十来个观测上不可靠，硬算出来的 t 值"
      "名义显著性是假的。替代方案是两件事配套使用——\n")
    A(f"1. **置换检验**（{args.ladder_nperm:,} 次，配对随机打乱）。经验 p 用 "
      "`(1 + #{|r_perm| ≥ |r_obs|}) / (1 + n_perm)`，分子上那个 +1 保证 "
      "p 严格大于 0：只跑了有限次置换，支撑不了「零假设下绝不可能」。\n")
    A("2. **自相关诊断**。置换检验的零假设含「观测可交换」，两列都是变动量时"
      "这大致成立，但必须测出来放在结论旁边。这张表测的是**本窗口内**的"
      "自相关，不是全历史——喂全历史进去会得到 n 上万的诊断，那说的是"
      "另一个样本。\n")
    A(md_table(ref["autocorr"].reset_index()))

    for i, (key, lad) in enumerate(ladders.items()):
        spec = lad["spec"]
        A(f"### 5.{4 + 2 * i} {spec.label}：同期相关\n")
        A(f"截断 {pm_daily.CUTOFF_UTC_HOUR[spec.key]:02d}:00 UTC　|　"
          f"信号 {len(lad['sig'])} 个交易日　|　固定网格 "
          f"{'、'.join(f'{k:.0f}' for k in lad['grid'])} 美元/桶\n")
        if spec.key == "cn":
            A("_两件与 A 股口径有关的事，读数之前先说清。_\n")
            A("_其一，**网格更窄**。行权价能不能进固定网格看的是覆盖率 ≥ 80%，"
              "而覆盖率是在该市场自己的截断口径下算的：A 股截到 07:00 UTC，"
              "落在这个窗口里的末笔报价更少，于是只有 4 个行权价够格。"
              "`exp_exceed` 的积分域因此与美股那一遍不同，两边的该列**不可比大小**，"
              "只能各自看符号与显著性。_\n")
            A("_其二，A 股的「同期」本身就含一段真实的时滞：第 d 个交易日的信号"
              "窗口是 `(d-1 日 07:00 UTC, d 日 07:00 UTC]`，它**覆盖了前一个"
              "美国交易时段的全部消息**，而截断点正是北京时间 15:00 收盘。"
              "所以这不是前视，但它也不是美股那种严格意义上的同时刻共动。_\n")
        A(md_table(lad["contemp"], max_rows=30))
        ew_rows = lad["contemp"][lad["contemp"].标的 == "航空等权"]
        if not ew_rows.empty:
            best = ew_rows.loc[ew_rows.pearson.abs().idxmax()]
            m = len(ew_rows)               # 信号列数 = 这一族里做了几次检验
            p_adj = min(1.0, float(best.p_perm) * m)
            A(f"**{spec.label}最强的一列是 `{best.信号}`**："
              f"r={best.pearson:+.3f}，置换 p={best.p_perm:.4g}，n={int(best.n)}；"
              f"按 {m} 个信号列做 Bonferroni 校正后 p={p_adj:.4g}"
              f"（{'仍然显著' if p_adj < 0.05 else '**不再显著**'}）。"
              "校正是必须的：三列信号都是从同一条隐含分布上读出来的，"
              "报最强那一列而不校正就是在挑显著。逐只个股那些行是**支持性证据**"
              "而不是另外 6 个独立命题，所以不并入这里的校正。\n")
            if abs(float(best.pearson)) < float(best.detectable_r):
                A(f"_注意 |r|={abs(float(best.pearson)):.3f} 落在该样本量的"
                  f"可检出下限 {best.detectable_r:.3f} **之下**。这不矛盾："
                  f"可检出下限说的是「效应真有这么大时能有 80% 概率测出来」，"
                  f"而 p 值说的是「零假设下出现这么大的 r 有多罕见」。"
                  f"两句话同时成立的读法是：这个关联报出来了，但功效不足，"
                  f"复现率不会高。_\n")
        ll = lad["leadlag"].get("d_K50")
        if ll is not None:
            A(f"![阶梯散点](figs/{14 + i:02d}_{key}_ladder_scatter.png)\n")
            A("散点不是装饰：n≈18 的相关系数完全可能由一两个极端点撬出来，"
              "而「十几个点排成一条线」的 r 和「一个远点把回归线拽过去」的 r "
              "是完全不同的证据。两个分面共享坐标范围，否则 `k=+1` 那张会被"
              "自动缩放拉满，看起来和 `k=0` 一样有结构。\n")
            # 这一句不能省。图上画的是 d_K50（与下面的扫描同一列），而上面那段
            # 结论用的是三列里最强的那一列——两个 r 不是同一个数，不说明白就是
            # 让读者以为其中一个印错了。
            best_col = str(best.信号) if not ew_rows.empty else None
            if best_col and best_col != "d_K50":
                A(f"_图上画的是 `d_K50`（与下面的领先滞后扫描同一列，选它是因为"
                  f"量纲最直白），**不是**上面那段结论里的 `{best_col}`。"
                  f"所以图上的 r 与正文的 "
                  f"{float(best.pearson):+.3f} 本来就不该相等。_\n")
            A("领先滞后扫描（`k>0` = 信号领先收益，有预测价值；"
              "`k<0` = 收益领先信号，无增量）。**只扫 `d_K50` 这一列**，"
              "标的固定为航空等权：三列信号各扫一遍就是 21 次检验，"
              "校正后什么都不会剩，而扫哪一列必须在看结果之前定下来。"
              "选 `d_K50` 的理由是它量纲最直白（美元/桶），"
              "不是因为它最显著。\n")
            A(md_table(ll.reset_index()))
            A(ladder_leadlag_verdict(ll))
        div = lad.get("divergence")
        if div is not None and not div.empty:
            A(f"### 5.{5 + 2 * i} {spec.label}：窗口累计收益（主要的伪发现防线）\n")
            A("油价冲击的特征形态是**分化**：能源涨、航空跌。若三者同向下跌，"
              "那就只是风险偏好恶化，此时再高的相关系数也不能解读成"
              "「油价预期影响航空股」。\n")
            A(md_table(div, max_rows=20))
            cat = div.set_index("资产")["累计收益%"]
            eng = [v for a, v in cat.items()
                   if a in set(div[div.类别 == "能源"].资产)]
            air = float(cat.iloc[0])
            if eng and np.isfinite(eng[0]) and np.isfinite(air):
                verd = ("**分化成立**" if eng[0] > 0 > air else
                        "**分化不成立**")
                A(f"{verd}：能源 {eng[0]:+.1f}%、航空等权 {air:+.1f}%。"
                  + ("能源涨而航空跌，是油价冲击的特征形态，不是普跌行情。\n"
                     if eng[0] > 0 > air else
                     "两者同向，本节的相关系数不能解读成油价渠道。\n"))


def ladder_leadlag_verdict(ll: pd.DataFrame) -> str:
    """把一张领先滞后扫描表判成一段话。**不预先写死结论**。

    这个函数存在的理由：第一版把「k=0 显著、|k|≥1 是噪声」直接写在报告模板里，
    那是拿美股那一遍的结果当成两个市场的共同结论，而 A 股跑出来的形态并不
    一样（`k=-1` 的原始 p 到了 0.02）。模板里写死结论，等于让报告替数据说话。
    """
    if 0 not in ll.index:
        return "_扫描表缺 k=0，无法判读。_\n"
    k0 = ll.loc[0]
    sig_ks = [int(k) for k in ll.index if ll.loc[k, "p_bonferroni"] < 0.05]
    raw_ks = [int(k) for k in ll.index
              if ll.loc[k, "p_perm"] < 0.05 and int(k) not in sig_ks]
    lines = [f"**读法**：Bonferroni 校正后显著的 k = "
             f"{sig_ks if sig_ks else '（无）'}"]
    if raw_ks:
        lines.append(f"；仅原始 p<0.05、校正后不显著的 k = {raw_ks}"
                     f"（扫 {len(ll)} 个 k，零假设下出现这种情形本属常事，"
                     f"**不能**拿来当领先性的证据）")
    lines.append(f"。`k=0` 处 r={k0.pearson:+.3f}、Bonferroni p="
                 f"{k0.p_bonferroni:.3f}。")
    if sig_ks == [0]:
        lines.append("唯一显著的是同期项，且 |k|≥1 大致关于 0 对称——这是"
                     "**同期共动**，不是领先：两个市场在同时消化同一条新闻，"
                     "预测市场没有跑在股价前面。")
    elif not sig_ks:
        lines.append("没有任何 k 通过校正，包括同期项。")
    else:
        lines.append("显著项不止同期，形态要逐个 k 看，不能只报最显著的那个。")
    lines.append(f"注意 n={int(k0.n)} 下 80% 功效的可检出下限是 "
                 f"|r|≈{k0.detectable_r:.3f}，所以这只排除了**大的**"
                 f"领先效应，排不掉小的。\n")
    return "".join(lines)


def render(args, audit, sel, specs, signals, results, ladders,
           fig_dir: Path, report: Path) -> None:
    """出图 + 写报告。图和报告用同一批数字，不重算。"""
    parts: list[str] = []
    A = parts.append

    # ================================================================== 图
    for key, out in results.items():
        spec: study.MarketSpec = out["spec"]
        sig = signals[key]
        n = 2 if key == "cn" else 7

        # --- 时序对齐
        idx = pd.DataFrame({
            f"{spec.label}航空": sp.cum_index(spec.airline_index.reindex(sig.index)),
            f"{spec.label}能源": sp.cum_index(spec.energy.reindex(sig.index)),
        })
        if spec.etf is not None:
            idx[spec.etf_label] = sp.cum_index(spec.etf.reindex(sig.index))
        ev_abs = sig.dp.abs().dropna()
        ev_dates = ev_abs[ev_abs >= ev_abs.quantile(args.event_quantile)].index
        fig, tb = viz.timeline(
            sig.p_level, sig.dp, spec.oil_px.reindex(sig.index).ffill(), idx,
            event_dates=ev_dates, oil_name=spec.oil_label,
        )
        style.save(fig, fig_dir / f"{n:02d}_{key}_timeline.png", tb)

        # --- 领先滞后热图（主信号 dp）
        fig, tb = viz.lead_lag_heatmap(out["lead_lag"]["dp"])
        style.save(fig, fig_dir / f"{n+1:02d}_{key}_leadlag.png", tb)

        # --- 剔除大盘后的领先滞后
        if out["lead_lag_resid"]:
            fig, tb = viz.lead_lag_heatmap(
                out["lead_lag_resid"],
                signal_name="Δp（标的为剔除大盘成分后的残差收益）")
            style.save(fig, fig_dir / f"{n+2:02d}_{key}_leadlag_resid.png", tb)

        # --- 事件研究：风险上升方向（最有解释力的那一个）
        ev = out["events"].get("up", {})
        if ev:
            fig, tb = viz.event_curves(
                {k: v for k, v in ev.items() if spec.oil_label not in k},
                head=f"{spec.label}：原油供给风险概率上升日前后的累计平均收益",
            )
            style.save(fig, fig_dir / f"{n+3:02d}_{key}_event_up.png", tb)

        # --- 三角验证
        if not out["triangle"].empty:
            fig, tb = viz.triangle_chart(out["triangle"])
            style.save(fig, fig_dir / f"{n+4:02d}_{key}_triangle.png", tb)

    # --- 累计 IC（两个市场画一张，便于直接比）
    ic_series, ic_sum = {}, {}
    for key, out in results.items():
        label = out["spec"].label
        if "dp" in out["ic"]:
            ic_series[f"{label}（信号 Δp）"] = out["ic"]["dp"]
            ic_sum[f"{label}（信号 Δp）"] = out["ic_summary"]["dp"]
    if ic_series:
        fig, tb = viz.ic_cumulative(ic_series, ic_sum)
        style.save(fig, fig_dir / "12_cross_sectional_ic.png", tb)

    # --- 行权价阶梯：隐含分布曲面只画一张（美股截断口径）。两个市场的曲面
    #     只差一个截断时刻，几乎逐格相同，画两张是同一张图占两个编号。
    if "us" in ladders:
        lad = ladders["us"]
        fig, tb = viz.ladder_surface(lad["clean"], lad["pinned"], lad["wti_px"],
                                     k50=lad["stats"].K50)
        style.save(fig, fig_dir / "13_ladder_surface.png", tb)
    for i, (key, lad) in enumerate(ladders.items()):
        ll = lad["leadlag"].get("d_K50")
        if ll is None:
            continue
        fig, tb = viz.ladder_scatter(lad["sig"]["d_K50"],
                                     lad["spec"].airlines.mean(axis=1), ll)
        style.save(fig, fig_dir / f"{14 + i:02d}_{key}_ladder_scatter.png", tb)

    # =============================================================== 报告
    A("# Polymarket 原油市场 ↔ 航空股：关联性分析\n")
    A(f"数据窗口 {args.start} → {args.end or '数据末端'}　|　"
      f"生成于 `scripts/11_oil_airline_analysis.py`\n")

    A("## 0. 先说三件必须先知道的事\n")
    A("**（1）「油价水平」市场存在，但只密集报价了二十来个交易日；"
      "长样本上能用的只有「供给中断概率」。**\n")
    A("成交额最大的那一批原油市场问的是**地缘供给中断风险**"
      "（霍尔木兹海峡封锁、伊朗石油设施被袭），而 OPEC 产量决议这种直接的"
      "油价驱动市场累计成交额只有三位数，统计上等于不存在。所以 §2/§3 的"
      "主分析检验的是 **「原油供给中断概率」↔ 航空股**，不是「油价预期」↔ 航空股。"
      "两者不是一回事：供给风险概率上升同时意味着油价上行压力**与**"
      "全局风险偏好下降，后者对航空股也是利空，所以三角验证与剔除大盘"
      "不是可选项。\n")
    A("但 Polymarket 上另有一套 **`will-crude-oil-cl-hit-high-K-by-T`** 的"
      "行权价阶梯（K = 70…200 美元/桶），它给出的是油价的**隐含分布**，"
      "标的直接就是油价水平。代价是它只在 2026 年 3 月那波油价冲击里密集报价，"
      "算下来 21 个交易日。这一路单独放在 **§5**，与主分析并列呈现——"
      "一个样本长、信号间接，一个信号直接、样本短，**都报，不挑**。\n")
    A("**（2）真正的「盘口」（挂单深度）数据只有 64 天，本分析用的是成交流。**\n")
    A("原始需求里的「盘口数据」指的是订单簿。这份归档里带订单簿的层只覆盖 "
      "2026-07-23 → 2026-09-24 共 64 天，做日频相关分析样本太短。"
      "因此本分析用成交口径的微观结构代理：成交额加权概率、成交笔数、"
      "日内实现波动，以及 **OFI（订单流不平衡 = Σ 方向 × 成交额）**"
      "作为方向性压力的代理。这是可得数据下最接近盘口压力的量，"
      "但它不是挂单深度，结论表述上不能混。\n")
    A("**（3）横截面只有 6–8 只股票。**\n")
    A("单日横截面 IC 的标准误约 1/√7 ≈ 0.38，单日 IC 基本是纯噪声。"
      "日度 IC 序列的**均值**仍是无偏的，但这个 IC 不能与本项目"
      "全市场 1500 只股票的 IC 直接比大小。\n")

    A("## 1. 分析对象：哪些市场进来了，为什么\n")
    A("![市场全景](figs/01_market_universe.png)\n")
    A(f"三步筛选（宽召回 → 硬排除 → 人工归类），命中 {len(audit)} 个市场，"
      f"纳入 {len(sel)} 个（单市场累计成交额 ≥ {args.min_volume:,.0f} USDC）。"
      "灰条是被排除的子串假阳性，它们的成交额并不小——"
      "NHL 埃德蒙顿油人队（**Oil**ers）、加拿大政治人物 P**oil**ievre、"
      "英超 **Brent**ford、餐饮公司 Cracker **Barrel**。"
      "排除理由逐条写在图上与下表里，可复核。\n")
    A(md_table(audit.head(25), max_rows=25))

    for key, out in results.items():
        spec: study.MarketSpec = out["spec"]
        sig = signals[key]
        n = 2 if key == "cn" else 7
        sec = 2 if key == "cn" else 3
        A(f"## {sec}. {spec.label}\n")
        A(f"截断时刻 {pm_daily.CUTOFF_UTC_HOUR[spec.key]:02d}:00 UTC　|　"
          f"油种基准 {spec.oil_label}　|　"
          f"信号 {len(sig)} 天，Δp 非缺 {int(sig.dp.notna().sum())} 天\n")

        A(f"### {sec}.1 时序对齐\n")
        A(f"![时序](figs/{n:02d}_{key}_timeline.png)\n")
        A("三个分面共享 x 轴、各自保留真实量纲——**刻意不画双 Y 轴**："
          "概率 ∈ [0,1]、油价 ∈ 美元/桶、股价指数三者量纲不同，"
          "双轴图的刻度对齐方式由画图的人随手决定，能让同一份数据看起来"
          "「高度同步」或「完全无关」。本分析要检验的恰恰是有没有关联，"
          "不能用一个会凭手感造出相关性的图去展示它。"
          "第三个分面里航空与能源都归一化到基期 = 100，那是同一个单位，"
          "所以可以同轴。\n")

        A(f"### {sec}.2 领先滞后相关\n")
        A(f"![领先滞后](figs/{n+1:02d}_{key}_leadlag.png)\n")
        A("`k>0` 是概率领先股价（有预测价值），`k<0` 是股价领先概率"
          "（Polymarket 在跟随股市，无增量）。`*` 是 Bonferroni 校正后 p<0.05。"
          "**读法提醒**：扫 11 个 k，零假设下最大 |t| 的期望本就在 2 附近，"
          "所以不要挑最显著的 k 报告；要看 k>0 整体形态，以及是否与 k<0 对称"
          "（对称 = 同期共动，不是领先）。\n")
        A(md_table(study.headline(out)))
        dp_air = out["lead_lag"]["dp"].get(f"{spec.label}航空（等权）")
        if dp_air is not None and 1 in dp_air.index:
            r1 = dp_air.loc[1]
            A(f"**主口径结论（Δp → 次日航空股收益）**："
              f"{verdict(r1.pearson, r1.t_nw, r1.p_bonferroni, int(r1.n))}\n")

        if out["lead_lag_resid"]:
            A("剔除大盘成分后重算：\n")
            A(f"![剔大盘](figs/{n+2:02d}_{key}_leadlag_resid.png)\n")
            A("这一张是判别渠道的关键。地缘风险概率上升时大盘本身会跌，"
              "航空股跟着跌**不需要**任何油的逻辑。只有剔掉大盘之后航空股仍跌、"
              "能源股仍涨，才能说抓到了燃油成本这个渠道。\n")

        A(f"### {sec}.3 事件研究\n")
        A(f"![事件研究](figs/{n+3:02d}_{key}_event_up.png)\n")
        A("日频相关系数把有消息的日子和 95% 什么都没发生的日子同权平均，"
          "这类地缘风险信号本来就集中在少数几天，全样本相关会被稀释到看不见。"
          "事件研究只看概率大幅跳变的那些天。\n")
        rows = []
        for direction, label in [("up", "风险上升"), ("down", "风险缓解"),
                                 ("abs", "有大消息（不分方向）")]:
            for nm, (tb, info) in out["events"].get(direction, {}).items():
                if not info.get("n_events"):
                    continue
                rows.append({"方向": label, "标的": nm,
                             "事件数": info["n_events"],
                             "阈值Δp": info["threshold"],
                             "事件后累计收益": info["car_event_to_post"],
                             "t_NW": info["t_nw"], "p_NW": info["p_nw"]})
        A(md_table(pd.DataFrame(rows), max_rows=60))
        A("_事件窗口会重叠（一场冲突连着十几天），所以 t 值用 Newey-West。"
          "若曲线在事件日**之前**就出现漂移，那是警号——说明事件定义里"
          "混进了未来信息。_\n")

        A(f"### {sec}.4 三角验证\n")
        A(f"![三角](figs/{n+4:02d}_{key}_triangle.png)\n")
        A("直接检验「概率 → 航空股」有个致命的解释困难：结果为零时分不清是"
          "**Polymarket 没信息**还是**燃油成本渠道不通**。所以拆成两段分别验。\n")
        A(md_table(out["triangle"]))

        A(f"### {sec}.5 分化检验（主要的伪发现防线）\n")
        A("真实油价冲击的教科书特征是**分化**：能源涨、航空跌，价差走阔。"
          "如果 Δp 上升时两者只是同涨同跌，说明测到的是全局风险偏好，不是油价。"
          "下表是 `能源收益 − 航空收益` 对 Δp 的相关。\n")
        div = out.get("divergence", {})
        A(md_table(pd.DataFrame([{"口径": k, **v} for k, v in div.items()])))

        A(f"### {sec}.6 预测回归（控制大盘）\n")
        A(md_table(out["regression"], max_rows=50))

        A(f"### {sec}.7 稳健性：缩尾\n")
        A("主结果**不缩尾**。这里把 Δp 与收益各双侧缩尾 0.5% 后重算："
          "若关联在缩尾后消失，结论要改写成「效应只存在于极端事件日」，"
          "而不是报一个全样本相关。\n")
        rob = out.get("robust_winsor", {})
        A(md_table(pd.DataFrame([{"标的": k, **v} for k, v in rob.items()])))

    A("## 4. 横截面 IC（本项目主指标口径）\n")
    A("![累计IC](figs/12_cross_sectional_ic.png)\n")
    A("一个对所有股票都相同的宏观数，其横截面标准差为 0，对截面排序的贡献"
      "**恰为零**。所以打分必须是 `−β_i × Δp`：β_i 是个股对原油收益的滚动暴露度"
      "（只用过去 120 天，`shift(1)` 后再滚动，不含未来信息），"
      "负号让「分高 = 预期收益高」。\n")
    A("曲线的**斜率**就是平均 IC。一条稳定向上的直线说明信号一直有效；"
      "某个区间陡升、其余走平，说明 IC 均值为正只是某几天的一次性贡献，"
      "那不是可用的信号。\n")
    rows = []
    for key, out in results.items():
        for sname, s in out["ic_summary"].items():
            rows.append({"市场": out["spec"].label, "信号": sname, **s})
    A(md_table(pd.DataFrame(rows), max_rows=30))

    A("### 4.1 这个 IC 只用到 Δp 的符号（一个必须先知道的恒等式）\n")
    A("Pearson 相关对其中一个变量的正仿射变换不变，而 Δp_t 对当天所有股票是"
      "**标量**，所以\n")
    A("> IC_t = corr(−β·Δp_t, r_t) = sign(−Δp_t) × corr(β, r_t)\n")
    A("也就是说 **|Δp| 被完全丢掉了**：Δp = +0.50 与 Δp = +0.03 给出"
      "一模一样的 IC。这不是实现瑕疵，是「宏观序列 × 个股暴露度」这种构造的"
      "固有性质，但它把 §4 检验的命题缩窄成了一句很具体的话——"
      "**「供给风险上升的日子里，高油价暴露的航空股是否跑得更差」**。"
      "消息的**大小**不在这个口径里，那部分证据在 §2/§3 的时序相关与"
      "事件研究里（那两处用的是 Δp 的数值本身）。\n")
    A("代码里对这个恒等式做了运行时断言（偏差 > 1e-9 直接抛错），"
      "所以下面的符号置换检验不是凭推导写的捷径，是有自检的。\n")

    A("### 4.2 安慰剂对照：IC 里到底有没有 Polymarket 的功劳\n")
    A("既然排序的**形状**完全由 β_i 决定、Δp 只定**符号**，就有一条很容易"
      "混进来的假阳性路径：若样本期内「低油价暴露的航空股跑得更好」本身成立"
      "（一个与预测市场无关的截面异象），而 Δp 又碰巧多数为正，"
      "IC 就会显著为正，但功劳一点也不属于 Polymarket。\n")
    A("**对照一：beta_only。**打分 = `−β_i`，完全不含 Polymarket 信息。"
      "它显著就说明 IC 来自 β 异象。它与主口径在**逐格相同**的样本上评——"
      "两行的 `mean_xs_width` 必须一致；不掩码的话它会一路摊到 Polymarket "
      "还没有这个市场的年份上（样本从 288 天涨到 2361 天），"
      "「不显著」就分不清是真没信号还是被稀释掉了。\n")
    prows = []
    for key, out in results.items():
        base = out["ic_summary"].get("dp")
        if base is not None:
            prows.append({"市场": out["spec"].label,
                          "打分": "主口径 −β·Δp", **base})
        for pname, s in out.get("ic_placebo", {}).items():
            prows.append({"市场": out["spec"].label, "打分": pname, **s})
    A(md_table(pd.DataFrame(prows), max_rows=30))

    A("**对照二：符号置换检验。**把 Δp 的符号在**同一批交易日之间**随机打乱"
      "（β 不动，样本一天不变），重复 2000 次，看实测 t 值在置换零分布里的位置。"
      "这比单次打乱可靠得多：单次打乱只是零分布里的一个抽样点，"
      "小样本下它自己就可能偶然显著。\n")
    A("注意打乱必须**限定在进入检验的那批日子内部**。直接对整条 Δp 做置换会把"
      "有效值扔到 β 还没滚满 120 天的早期日子上，样本从 288 天掉到 23 天，"
      "然后在 23 天上给出一个 t = 2.9 的假显著——这个坑本轮踩过一次。\n")
    perm_rows = [{"市场": out["spec"].label, **out["ic_permutation"]}
                 for out in results.values() if out.get("ic_permutation")]
    A(md_table(pd.DataFrame(perm_rows), max_rows=10))
    A("判据：主口径显著、**且** beta_only 不显著、**且** 置换 p 值显著，"
      "§4 的结论才成立。任一条不满足，结论要降级成"
      "「IC 由横截面 β 异象驱动，与预测市场无关」。\n")

    render_ladder(A, ladders, args)

    A("## 6. 口径与已知局限（逐条）\n")
    A("| 项 | 处理 | 偏差方向 |")
    A("|---|---|---|")
    A("| 盘口深度数据仅 64 天 | 改用成交口径的 OFI 作方向压力代理 | "
      "代理不如真深度，效应向零偏 |")
    A("| 美股日线**未复权** | DAL/LUV 每季一个约 −0.5% 的除息跳空 | "
      "与油价概率无关的噪声进分母，相关系数向零偏（保守） |")
    A("| 原油为**连续合约** | 换月跳空按 \\|r\\|>8% 剔除 | 少用约每月 1 天 |")
    A("| A 股截断 07:00 UTC | = 北京 15:00 收盘 | 无前视 |")
    A("| 美股截断 20:00 UTC | 冬令时 = 美东 15:00，比收盘早 1 小时 | "
      "少用 1 小时信息（保守），绝不越过收盘线 |")
    A("| 概率水位 ffill | 上限 10 自然日，超过置 NaN | "
      "不 bfill、不插值——那是把未来搬到过去 |")
    A("| 流量类因子（成交额/笔数/OFI） | 无成交 = 0，**不** ffill | "
      "ffill 会把一天的成交额复制成一周 |")
    A("| 横截面 6–8 只 | IC 照报，但标注单日 SE≈0.38 | "
      "不可与全市场 1500 只的 IC 比大小 |")
    A("| 截面 IC 只用到 sign(Δp) | 恒等式见 §4.1，配 beta_only 与 2000 次"
      "符号置换两重对照 | 丢掉了消息大小，该部分证据看 §2/§3 |")
    A("| 11 次滞后扫描 | 报 Bonferroni 校正 p | 防挑最显著的 k |")
    A("| 阶梯样本仅 21 个交易日 | 不用 HAC t 值，改置换检验 + 自相关诊断 | "
      "n<20 的 HAC 名义显著性是假的 |")
    A("| 阶梯还原的是**期间最高价**分布 | K50 系统性高于现货，不作修正 | "
      "不是错位；把它当到期价读才是错 |")
    A("| 障碍触碰后被钉在 0.999 | p≥0.99 判为已结算并剔除；**下侧无对应阈值** | "
      "加下界会把深度虚值报价当结算剔掉 |")
    A("| 阶梯曲线的跨行权价插值 | 只在当日已观测的行权价区间**内**插，不外推 | "
      "跨时间插值是前视，禁用 |")
    A("| `exp_exceed` 是个积分 | 只在当日覆盖**整个**固定网格时才出数 | "
      "否则积分域一变，数字会因非经济原因移动 |")
    A("| `p_level` 高度自相关 | **不进**显著性表，只用于画图与事件定义 | "
      "水位对收益回归是伪回归的经典形式 |")
    A("\n每张图旁边都有同名 CSV（表格视图）。这是强制的：调色板里 aqua/yellow "
      "两槽对浅色底的对比度低于 3:1，规范要求有等价的非颜色读法；"
      "而「图里那个峰值到底是多少」这种问题，图本身永远答不了。\n")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
