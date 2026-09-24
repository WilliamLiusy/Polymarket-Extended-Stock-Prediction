#!/usr/bin/env python
"""出报告。读 03 存档的预测，算 IC / 分层 / 多空 / 换手与成本，写 markdown。

分两块，回答的是两个不同的问题：

* **块 (a) baseline 长 OOS（2018→2026）** —— baseline 本身行不行。窗口长、功效足。
* **块 (b) PM 对比窗（2025-01→2026-09，锁死）** —— 这块的数字**不用来评价 baseline
  好坏**，它的唯一用途是作为将来 Polymarket 版本的配对对照基线。窗口只有 400 多天，
  单看它的 IC 噪声很大。

用法：
    python scripts/04_report.py
    python scripts/04_report.py --universe top300 --model ridge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pmsp.config import abs_path, load_config, load_model_config
from pmsp.eval.backtest import cost_sensitivity, quantile_backtest
from pmsp.eval.ic import calc_ic_series, detectable_delta, ic_summary


def _fmt_pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.2%}"


def _fmt(x: float, n: int = 4) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{n}f}"


def report_window(
    tag: str, pred_dir: Path, ret1d: pd.Series, cfg, mcfg, horizon: int
) -> list[str]:
    pred_path = pred_dir / f"pred_{tag}.parquet"
    if not pred_path.exists():
        return [f"> 缺 `{pred_path.name}`，跳过。先跑 `scripts/03_run_workflow.py`。", ""]

    pred = pd.read_parquet(pred_path)["score"]
    ic = pd.read_parquet(pred_dir / f"ic_{tag}.parquet")
    summ = ic_summary(ic["ic"], ic["rank_ic"], horizon=horizon)

    lines = [
        "#### IC",
        "",
        "| 指标 | 值 | 含义 |",
        "|---|---|---|",
        f"| 交易日数 | {summ['n_days']} | 有效样本约 {summ['n_days'] // horizon}（5 日标签重叠） |",
        f"| **IC 均值** | **{_fmt(summ['ic_mean'])}** | 主指标。每日横截面 Pearson 相关的均值 |",
        f"| IC 标准差 | {_fmt(summ['ic_std'])} | 日度波动，多半由市场 regime 驱动 |",
        f"| **ICIR** | **{_fmt(summ['icir'], 3)}** | IC均值/IC标准差，信号的稳定性 |",
        f"| t 值 (Newey-West) | {_fmt(summ['t_nw'], 2)} | **以此为准**，已修正标签重叠导致的自相关 |",
        f"| t 值 (朴素 N/h) | {_fmt(summ['t_naive'], 2)} | 粗略折算，通常比 NW 略乐观 |",
        f"| IC 胜率 | {_fmt_pct(summ['ic_win_rate'])} | IC>0 的天数占比 |",
        f"| RankIC 均值 | {_fmt(summ['ric_mean'])} | 副指标（Spearman），抗极端值 |",
        f"| RankIC ICIR | {_fmt(summ['ric_icir'], 3)} | |",
        "",
    ]
    if summ["ic_ric_gap_warn"]:
        lines += ["> **告警**：IC 与 RankIC 差距偏大，说明 Pearson IC 被少数极端收益个股"
                  "主导，信号的稳健性存疑。", ""]

    # 分层回测
    common = pred.index.intersection(ret1d.index)
    if len(common) < 100:
        lines += [f"> 预测与收益率的公共索引只有 {len(common)} 条，跳过回测。", ""]
        return lines

    cost = dict(cfg.cost)
    bt = quantile_backtest(pred, ret1d.reindex(common), n_groups=mcfg.eval["n_groups"],
                           horizon=horizon, cost=cost)
    groups = bt["group_ann_return"]
    lines += [
        f"#### 分层回测（{bt['n_groups']} 组，扣费前年化）",
        "",
        "| 组 | " + " | ".join(str(g) for g in sorted(groups)) + " |",
        "|---|" + "---|" * len(groups),
        "| 年化 | " + " | ".join(_fmt_pct(groups[g]) for g in sorted(groups)) + " |",
        "",
        f"组号越大 = 预测收益越高。**单调性 Spearman = {_fmt(bt['monotonicity_spearman'], 3)}**"
        f"（接近 1 说明各层收益随预测分单调排列，这比多空收益本身更能说明因子有效）。",
        "",
        "#### 组合表现",
        "",
        "| 组合 | 年化 | 波动 | Sharpe | 最大回撤 | Calmar |",
        "|---|---|---|---|---|---|",
    ]
    for name, st in [
        ("多空（税前）", bt["long_short"]["gross"]),
        ("多空（税后）", bt["long_short"]["net"]),
        ("Top 组（税前）", bt["top_group"]["gross"]),
        ("Top 组（税后）", bt["top_group"]["net"]),
        ("Top 组超额 vs 全市场等权", bt["top_group"]["excess_vs_eqw"]),
    ]:
        lines.append(
            f"| {name} | {_fmt_pct(st['ann_return'])} | {_fmt_pct(st['ann_vol'])} | "
            f"{_fmt(st['sharpe'], 2)} | {_fmt_pct(st['max_drawdown'])} | {_fmt(st['calmar'], 2)} |"
        )
    lines += [
        "",
        f"日均单边换手 **{_fmt_pct(bt['avg_daily_turnover_one_way'])}**，"
        f"年化约 {bt['avg_daily_turnover_one_way'] * 243:.0f} 倍。",
        "",
        "> 多空组合是**因子诊断工具，不是可实盘策略**：A 股融券券源少、成本高、"
        "小盘股基本借不到。能落地的看「Top 组超额」那一行。",
        "",
        "#### 成本敏感性",
        "",
    ]
    cs = cost_sensitivity(pred, ret1d.reindex(common), cost=cost,
                          multipliers=tuple(mcfg.eval["cost_multipliers"]),
                          n_groups=mcfg.eval["n_groups"], horizon=horizon)
    lines += ["| 成本倍数 | 多空 Sharpe(税后) | 多空年化(税后) | Top组年化(税后) |",
              "|---|---|---|---|"]
    for _, r in cs.iterrows():
        lines.append(f"| {r['cost_multiplier']:.1f}× | {_fmt(r['ls_sharpe_net'], 2)} | "
                     f"{_fmt_pct(r['ls_ann_return_net'])} | {_fmt_pct(r['top_ann_return_net'])} |")
    lines += ["", "> 换手这么高，成本是生死线。看 Sharpe 在几倍成本下归零，"
              "比单看一个「扣费后」数字有信息量得多。", ""]

    imp_path = pred_dir / f"importance_{tag}.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0).iloc[:, 0].head(15)
        lines += ["#### 特征重要性 Top 15", "", "| 因子 | 平均 gain |", "|---|---|"]
        lines += [f"| {k} | {v:,.0f} |" for k, v in imp.items()]
        lines.append("")

    log_path = pred_dir / f"log_{tag}.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        lines += [
            f"#### 滚动窗口（{len(log)} 个）",
            "",
            f"训练集规模 {log['n_train'].min():,} → {log['n_train'].max():,} 行"
            f"（{log['n_train_days'].min()} → {log['n_train_days'].max()} 个交易日，扩张窗口）。",
            f"验证集 IC 均值 {_fmt(log['best_valid_ic'].mean())}，"
            f"区间 [{_fmt(log['best_valid_ic'].min())}, {_fmt(log['best_valid_ic'].max())}]。",
            "",
        ]
        if "best_iteration" in log:
            lines.append(f"LightGBM 最优轮数中位数 {log['best_iteration'].median():.0f}"
                         f"（早停依据是**验证集日度 IC**，不是 MSE）。")
            lines.append("")
    return lines


def main() -> int:
    cfg, mcfg = load_config(), load_model_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=f"top{cfg.universe['top_n_list'][0]}")
    ap.add_argument("--model", default="lgb", choices=["lgb", "ridge"])
    args = ap.parse_args()

    pred_dir = abs_path(mcfg.output["pred_dir"])
    report_dir = abs_path(mcfg.output["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    horizon = cfg.label["horizon"]

    ret_path = pred_dir / f"ret1d_{args.universe}.parquet"
    if not ret_path.exists():
        raise SystemExit(f"缺 {ret_path}，请先跑 scripts/03_run_workflow.py")
    ret1d = pd.read_parquet(ret_path)["ret_1d"]

    pm = cfg.split["pm_compare"]
    n_oos = len(pd.bdate_range(pm["oos_start"], pm["oos_end"]))
    md = [
        f"# A 股 5 日收益 baseline 报告 — {args.universe} / {args.model}",
        "",
        f"股票池 `{args.universe}`（每月末按过去 {cfg.universe['lookback_days']} 日"
        f"日均成交额取前 N）；因子 Alpha158 扩窗 {mcfg.features['windows']}；"
        f"标签 `{cfg.label['expr']}`（T+1 收盘买、T+6 收盘卖）。",
        "",
        "标签做了每日横截面 z-score，所以 `objective=mse` 的训练严格等价于最大化 IC"
        "（`min MSE = 1 - IC²`）；早停指标用的是**验证集日度 IC**。",
        "",
        "---",
        "",
        "## (a) baseline 长 OOS —— 评 baseline 本身",
        "",
        f"{cfg.split['baseline']['oos_start']} 起，每 "
        f"{cfg.split['baseline']['retrain_freq_months']} 个月滚动重训。窗口长、统计功效足，"
        "**判断 baseline 好坏看这一块**。",
        "",
    ]
    md += report_window(f"{args.universe}_{args.model}_baseline", pred_dir, ret1d, cfg, mcfg, horizon)
    md += [
        "---",
        "",
        "## (b) Polymarket 对比窗（锁死）—— 仅作为将来的配对对照基线",
        "",
        f"{pm['oos_start']} → {pm['oos_end']}，每 {pm['retrain_freq_months']} 个月重训。"
        "这块的数字**不用来评价 baseline 好坏**——窗口太短，单看 IC 噪声很大。"
        "它唯一的用途是给将来的 Polymarket 版本做**逐日配对差值检验**的对照，",
        "日度 IC 序列已存档在 `data/predictions/ic_*.parquet`，"
        "到时候直接读、不需要重训 baseline。",
        "",
    ]
    md += report_window(f"{args.universe}_{args.model}_pm_compare", pred_dir, ret1d, cfg, mcfg, horizon)

    mde = detectable_delta(n_oos, horizon=horizon)
    md += [
        "### 这个窗口能检出多大的 Polymarket 增益",
        "",
        f"约 {n_oos} 个交易日、有效样本约 {n_oos // horizon} 个独立 5 日区间，"
        f"双边 α=0.05、power=80% 下**可检出的最小 ΔIC ≈ {_fmt(mde)}**。",
        "",
        "> 这是 Polymarket 历史长度的物理上限，不是方法问题："
        "若真实增益只有 0.003，这个窗口无论怎么做都证不出来。事前就该认下来。",
        "",
        "---",
        "",
        "## 已知局限",
        "",
        "主数据源为新浪财经的公开日线接口（不需要 token）。它只给 OHLCV，"
        "以下几项因此无法做。前四项买 tushare 2000 积分（200 元/年）可全部解决，"
        "schema 已预留字段：",
        "",
        "1. **无市值、无行业分类** → 没做市值/行业中性化，模型可能部分在赚小盘股溢价",
        "2. **无历史股票名称** → 无法精确剔除历史 ST",
        "3. **无指数成分股历史** → 股票池是「流动性前 N」而非真正的沪深300/中证500"
        "（高度重叠但不等同；用当前成分股回溯会引入幸存者偏差+前视偏差，比近似更糟）",
        "4. **无成交额** → 流动性排名用 `close × volume` 代理（真实值是 "
        "`vwap × volume`，两者只差 `vwap/close`，对**排名**影响可忽略）",
        "5. **无 `$vwap`** → Alpha158 的 `VWAP0` 因子拿不到，因子数 215 而非 216。"
        "不拿「典型价」`(H+L+C)/3` 冒充——那是 HLC 的线性组合，与 KMID/KLEN/KSFT "
        "高度重复，等于凭空造一个假因子",
        "6. 涨跌停用 `pct_chg` 阈值近似；停牌由「日期序列有缺口」推导",
        "",
        "另：复权用新浪的官方后复权累计因子（`hfq.js`），除权日是官方口径、"
        "不从收益率序列里猜。正确性由 `scripts/05_verify.py` 第 1 项的**自洽检验**"
        "保证（因子恒为正、除权日的未复权跳空被抹平、非除权日逐点不变），"
        "并另有东财第三方源交叉核对。注意因子**不是**非递减的：缩股（破产重整里的"
        "出资人权益调整）会让它合法地下降，全市场只有 2 只出现过，检验改为卡"
        "「缩股日复权后收益落在涨跌停带内」。",
        "",
        "## 多重比较声明",
        "",
        f"股票池 `top{cfg.universe['top_n_list']}` 两个值是**实施前预注册**的"
        "（1500 为主检验、300 为第二窗口），不是事后挑选。两个都报。",
        "",
    ]

    out = report_dir / f"report_{args.universe}_{args.model}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
