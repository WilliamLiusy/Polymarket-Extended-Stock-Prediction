#!/usr/bin/env python
"""真实数据上的验证项。**baseline 的数字在这些项通过之前不可信。**

    python scripts/05_verify.py                 # 跑不需要网络、不需要重训的项（1,3,2/6）
    python scripts/05_verify.py --only adjust   # 复权自洽检验 + 外部交叉校验（后者要联网）
    python scripts/05_verify.py --only benchmark  # qlib 官方基准量级对照（要训练，约几十分钟）
    python scripts/05_verify.py --only embargo    # embargo 有效性（要训练两次）

## 与方案的一处修正

方案的验证第 3 条写的是「把 label 改成**过去** 5 日收益，IC 应 ≈ 0」。
**这条测不出它想测的东西**：Alpha158 里本来就含 ROC5（过去 5 日收益本身），
模型学到反转后，预测与过去 5 日收益必然强负相关。IC 显著非零是**预期行为**，
不是前视偏差的证据，所以这条会给出假警报。

换成两条真正能证伪的：

* **3a 标签口径直算核对** —— 从复权面板直接算 `close[t+6]/close[t+1]-1`，
  与 qlib 表达式求值的结果逐点比对。这是全项目最高风险的一行配置，值得单独验。
* **3b point-in-time 截断一致性** —— 把数据在日期 T 处截断后重新求值特征，
  T 之前的特征值必须与用全量历史求值的结果**完全一致**。
  若任何因子偷看了未来，砍掉未来数据就会让它变化。这条是决定性的。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pmsp.build.adjust import compare_adjustment, to_qlib_code
from pmsp.config import abs_path, load_config, load_model_config
from pmsp.eval.ic import calc_ic_series, ic_summary

# qlib 官方 examples/benchmarks/README.md 的 Alpha158 表（csi300、1 日标签、
# 测试期 2017-01-01→2020-08-01，20 个随机种子的均值）
QLIB_BENCH = {
    "lgb": {"ic": 0.0448, "icir": 0.3660, "rank_ic": 0.0469, "rank_icir": 0.3877},
    "ridge": {"ic": 0.0397, "icir": 0.3000, "rank_ic": 0.0472, "rank_icir": 0.3531},  # 表里的 Linear
}

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool | None, detail: str = "") -> None:
    _results.append((name, ok, detail))
    mark = "[跳过]" if ok is None else ("[通过]" if ok else "[失败]")
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------- 1. 复权正确性
def _high_dividend_codes(panel: pd.DataFrame, n: int) -> pd.Series:
    """复权因子跨度最大的 n 只（= 期间送转/分红最多 = 最能暴露复权错误）。"""
    span = panel.groupby("code")["factor"].agg(lambda s: s.max() / max(s.min(), 1e-12))
    n_obs = panel.groupby("code").size()
    return span[n_obs > 1000].sort_values(ascending=False).head(n)


def verify_adjust_internal(cfg) -> None:
    """复权正确性的**自洽**检验：不依赖任何第二个数据源，因此永远可跑。

    为什么不能只靠外部交叉校验：本机东方财富/网易的批量接口都打不通
    （东财连发约 6 次即封 IP），tushare 这个 token 的 `daily` 无权限。
    把复权的正确性押在一个时通时不通的外部源上，等于这条验证形同虚设。

    下面三条合起来其实比"和某家对一遍"更有力，因为它们直接检验复权**要
    干的事**——把除权造成的价格跳空移走，且只移走这个：

    * **因子形状**：后复权累计因子必须恒为正。**不能卡"非递减"**——分红送股
      只会让它上台阶，但**缩股**（破产重整里的出资人权益调整）会让它合法地
      下降：股数变少、单价机械性跳涨，后复权必须把后面的价格乘回去。
      全市场实测只有 2 只出现过：SH600381（未复权价跨 440 天停牌 2.08→17.59
      即 ×8.46，因子 8.193→1.0175 即 ×1/8.05）和 SZ200054（×4.20 / ×1/4.00），
      两只复牌当天复权后收益都恰好 +5.0%——它们是 ST，当年涨跌停正是 5%。
      所以改成卡**下降当日复权后收益必须落在涨跌停带内**：因子若取反或对齐
      错了，抵消不成立，那天会出现一个几百个百分点的荒谬收益。
    * **除权日跳空**：在因子变化日上，**未复权**收益应显著为负（除权除息
      当天价格机械性下跌），复权应把这个跳空**抹掉绝大部分**。
      看的是**抹掉的比例**而不是残差的绝对水平：残差里含真实的贴权漂移
      （A 股除权日本身确实小幅下跌，实测约 −0.6%，t≈−4），卡绝对值会给
      假警报。抹掉比例能抓住真正的错误——完全没复权是 0%，因子取反是负数。
    * **非除权日不动**：在因子不变的日子里，复权前后的收益率必须逐点相等。
      这是**最能抓 bug 的一条**：它同时证明了因子没有对齐错一天（差一天的
      话，除权日的邻日就会出现差异）。实测最大偏差在 1e-16 量级。
    """
    print("\n[1] 复权正确性（自洽检验，不依赖外部源）")
    panel = pd.read_parquet(cfg.path("data.panel_path"))
    panel = panel.sort_values(["code", "date"], ignore_index=True)
    g = panel.groupby("code", sort=False)

    # 未复权价 = 复权价 / 因子。两条收益率序列的差别应**只**来自除权日
    raw_close = panel["close"] / panel["factor"]
    ret_adj = g["close"].pct_change()
    ret_raw = raw_close.groupby(panel["code"], sort=False).pct_change()
    is_exdiv = g["factor"].pct_change().abs() > 1e-9
    ok = ret_adj.notna() & ret_raw.notna()

    # --- 1a 因子形状：恒为正，且因子下降（缩股）当日复权后收益不荒谬
    f_min = g["factor"].min()
    f_chg = g["factor"].diff()
    shrink = ok & (f_chg < -1e-9)          # 缩股日
    n_shrink = int(shrink.sum())
    # 0.11 = 主板 10% 涨跌停留一点余量；复牌日、创业板 20% 也都在 0.25 以内
    worst = float(ret_adj[shrink].abs().max()) if n_shrink else 0.0
    check("复权因子恒为正；缩股日的跳涨被因子下降精确抵消（复权后收益不荒谬）",
          bool(f_min.min() > 0 and n_shrink <= 20 and worst < 0.25),
          f"最小因子 {f_min.min():.6f}，{n_shrink} 个缩股日，"
          f"复权后 |收益| 最大 {worst:.2%}（未复权那天是 "
          f"{float(ret_raw[shrink].abs().max()) if n_shrink else 0:.0%}）")

    # --- 1b 除权日跳空
    ex = ok & is_exdiv
    other = ok & ~is_exdiv
    n_ex = int(ex.sum())
    raw_mean, adj_mean = ret_raw[ex].mean(), ret_adj[ex].mean()
    raw_med, adj_med = ret_raw[ex].median(), ret_adj[ex].median()
    rm_mean = 1.0 - abs(adj_mean) / max(abs(raw_mean), 1e-12)
    rm_med = 1.0 - abs(adj_med) / max(abs(raw_med), 1e-12)
    check("除权日的机械性跳空被抹掉 ≥75%（按均值和中位数各算一次）",
          bool(n_ex > 100 and raw_mean < -0.005 and rm_mean > 0.75 and rm_med > 0.75),
          f"{n_ex:,} 个除权日：未复权均值 {raw_mean*100:+.3f}% → 复权后 "
          f"{adj_mean*100:+.3f}%（抹掉 {rm_mean*100:.1f}%）；"
          f"中位 {raw_med*100:+.3f}% → {adj_med*100:+.3f}%（抹掉 {rm_med*100:.1f}%）；"
          f"普通日均值 {ret_adj[other].mean()*100:+.3f}%")

    # --- 1c 非除权日不动
    d = (ret_adj[other] - ret_raw[other]).abs()
    check("非除权日复权前后收益率逐点相等（复权没动不该动的日子）",
          bool(d.max() < 1e-9),
          f"{len(d):,} 个非除权日，最大偏差 {d.max():.2e}")

    cands = _high_dividend_codes(panel, 5)
    print("      分红送转最多的 5 只（复权因子跨度）："
          + ", ".join(f"{c}({v:.1f}×)" for c, v in cands.items()))


def verify_adjust_external(cfg, n_stocks: int = 3) -> None:
    """可选的外部交叉校验。打不通就**跳过**，不算失败。

    比的是**收益率序列相关性**而不是价格水平：复权因子的绝对水位是任意的
    （取决于以哪天为基准），只有收益率序列才有唯一正确答案。
    """
    print("\n[1'] 复权交叉校验（外部源，打不通则跳过）")
    from pmsp.datasource.eastmoney import EastmoneyDaily

    panel = pd.read_parquet(cfg.path("data.panel_path"))
    cands = _high_dividend_codes(panel, n_stocks)
    ds = EastmoneyDaily()
    for code in cands.index:
        # compare_adjustment 收的是 DataFrame（自己认 date / trade_date 列），不是 Series
        ours = panel[panel["code"] == code][["date", "close"]].sort_values("date")
        ts_code = f"{code[2:]}.{code[:2]}"
        try:
            # 起止日期跟我们自己的面板对齐；东财这个签名要求显式传，不能省
            ref = ds.fetch_one_symbol(
                ts_code,
                start_date=f"{ours['date'].min():%Y%m%d}",
                end_date=f"{ours['date'].max():%Y%m%d}",
                adjust=2,
            )
        except Exception as exc:
            check(f"{code} 复权交叉校验", None, f"东财取数失败：{exc}")
            time.sleep(8)
            continue
        if ref is None or ref.empty:
            check(f"{code} 复权交叉校验", None, "东财返回空（限流）")
            time.sleep(8)
            continue
        cmp = compare_adjustment(ours, ref)
        check(f"{code} 复权后收益率序列与东财一致",
              cmp["ret_corr"] > 0.999 and cmp["n_overlap"] > 200,
              f"重叠 {cmp['n_overlap']} 日，收益率相关 {cmp['ret_corr']:.6f}，"
              f"单日最大偏差 {cmp['max_abs_ret_diff']:.2e}")
        time.sleep(8)  # 东财连发约 6 次即封 IP，必须慢


# --------------------------------------------------- 2. 随机因子 & 6. IC/RankIC
def verify_from_predictions(cfg, mcfg, universe: str, model: str) -> None:
    print("\n[2] 随机因子对照 + [6] IC vs RankIC 一致性")
    pred_dir = abs_path(mcfg.output["pred_dir"])
    tag = f"{universe}_{model}_baseline"
    p = pred_dir / f"pred_{tag}.parquet"
    if not p.exists():
        check("随机因子对照", None, f"缺 {p.name}，先跑 scripts/03_run_workflow.py")
        return

    pred = pd.read_parquet(p)["score"]
    cache = abs_path(mcfg.output["cache_dir"])
    feats = sorted(cache.glob(f"features_{universe}_*.parquet"))
    if not feats:
        check("随机因子对照", None, "缺特征缓存，无法取标签")
        return
    label = pd.read_parquet(feats[-1], columns=["LABEL0"])["LABEL0"].reindex(pred.index)

    rng = np.random.default_rng(0)
    rand = pd.Series(rng.standard_normal(len(pred)), index=pred.index)
    ric, rric = calc_ic_series(rand, label)
    rs = ic_summary(ric, rric, horizon=cfg.label["horizon"])
    check("随机因子 IC ≈ 0 且 |t_NW| < 2（评估代码不自造信号）",
          abs(rs["ic_mean"]) < 0.005 and abs(rs["t_nw"]) < 2.0,
          f"IC={rs['ic_mean']:.5f}  t_NW={rs['t_nw']:.2f}")

    ic, rrc = calc_ic_series(pred, label)
    ss = ic_summary(ic, rrc, horizon=cfg.label["horizon"])
    same_sign = np.sign(ss["ic_mean"]) == np.sign(ss["ric_mean"])
    ratio = abs(ss["ic_mean"]) / max(abs(ss["ric_mean"]), 1e-12)
    check("IC 与 RankIC 同号且量级接近",
          bool(same_sign and 0.5 < ratio < 2.0),
          f"IC={ss['ic_mean']:.4f}  RankIC={ss['ric_mean']:.4f}  比值 {ratio:.2f}"
          + ("（差距过大 → Pearson IC 被少数极端收益个股主导）" if not (0.5 < ratio < 2.0) else ""))


# ----------------------------------------- 3a 标签直算核对 / 3b 截断一致性
def verify_no_lookahead(cfg, mcfg, universe: str, n_check_stocks: int = 30) -> None:
    print("\n[3] 无前视：标签口径直算核对 + point-in-time 截断一致性")
    from qlib.data import D

    from pmsp.model.dataset import ensure_qlib_init, feature_config

    qlib_dir = cfg.path("data.qlib_dir")
    ensure_qlib_init(qlib_dir)

    codes = list(D.list_instruments(D.instruments(universe), as_list=True))[:n_check_stocks]
    end = cfg.data["end_date"]

    # --- 3a：qlib 表达式求出的标签 vs 从后复权 close 直接算的 close[t+6]/close[t+1]-1
    got = D.features(codes, ["$close", cfg.label["expr"]], start_time="2018-01-01", end_time=end)
    got.columns = ["close", "label"]
    if list(got.index.names) == ["instrument", "datetime"]:
        got = got.swaplevel(0, 1).sort_index()
    by_code = got.groupby(level="instrument", group_keys=False)
    manual = by_code["close"].shift(-6) / by_code["close"].shift(-1) - 1.0
    both = pd.DataFrame({"expr": got["label"], "manual": manual}).dropna()
    dev = (both["expr"] - both["manual"]).abs().max()
    check("标签 == close[t+6]/close[t+1]-1（T+1 买、T+6 卖）",
          bool(dev < 1e-6) and len(both) > 1000,
          f"{len(both):,} 个样本，最大偏差 {dev:.2e}")

    # --- 3b：在 T 处截断数据重算特征，T 之前的值必须完全不变
    fields, names = feature_config(mcfg.features["windows"])
    cut = "2023-06-30"
    full = D.features(codes, fields, start_time="2022-01-01", end_time=end)
    trunc = D.features(codes, fields, start_time="2022-01-01", end_time=cut)
    full.columns = trunc.columns = names
    for f in (full, trunc):
        if list(f.index.names) == ["instrument", "datetime"]:
            f.sort_index(inplace=True)
    common = full.index.intersection(trunc.index)
    a = full.loc[common].to_numpy(dtype=float)
    b = trunc.loc[common].to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
    # 逐因子看，这样能直接点出是哪个因子偷看了未来
    per_factor = pd.Series(np.nanmax(diff, axis=0), index=names)
    bad = per_factor[per_factor > 1e-6]
    check(f"截断到 {cut} 后，之前的 {len(names)} 个因子值完全不变（无未来函数）",
          bad.empty,
          f"{len(common):,} 个样本全部一致" if bad.empty
          else f"{len(bad)} 个因子发生变化：{list(bad.index[:5])}（最大偏差 {bad.max():.2e}）")


# ----------------------------------------------- 4. qlib 官方基准量级对照
def verify_benchmark(cfg, mcfg, model: str = "lgb") -> None:
    """用 qlib 的口径复跑一遍：top300 + **1 日标签** + Alpha158 默认 158 因子 + 官方超参。

    这是最强的端到端验证——如果数据采集或 dump_bin 转换有问题，这里的 IC 会明显对不上。

    两点必须说清楚：
      * **只是量级对照，不是逐项复现**。免费档拿不到沪深300 成分股历史，
        `top300` 只是它的近似；测试期我们也用 2017-01→2020-08 对齐官方，但股票池不同。
      * 官方数字是 20 个随机种子的均值，我们只跑 1 个种子。
    """
    print("\n[4] qlib 官方基准量级对照（top300 + 1 日标签 + Alpha158 默认 158 因子）")
    from pmsp.model.dataset import build_feature_matrix
    from pmsp.model.walkforward import generate_splits, run_walkforward

    bench = QLIB_BENCH[model]
    qlib_dir = cfg.path("data.qlib_dir")
    # qlib 的 1 日标签，同样是 T+1 买 T+2 卖
    df, feat_cols = build_feature_matrix(
        qlib_dir=qlib_dir, universe="top300",
        label_expr="Ref($close, -2)/Ref($close, -1) - 1",
        start_date="2008-01-01", end_date="2020-08-01",
        windows=[5, 10, 20, 30, 60],  # 官方默认，158 个因子
        cache_dir=abs_path(mcfg.output["cache_dir"]),
    )
    check("Alpha158 默认配置为 158 个因子", len(feat_cols) == 158, f"{len(feat_cols)} 个")

    dates = pd.DatetimeIndex(df.index.get_level_values("datetime").unique()).sort_values()
    splits = generate_splits(dates, "2017-01-01", "2020-08-01", retrain_months=12,
                             embargo_days=1, valid_days=240, min_train_days=500)
    lgb_params = dict(mcfg.models["lgb"])
    lgb_params.pop("num_boost_round", None)
    lgb_params.pop("early_stopping_rounds", None)
    res = run_walkforward(df, feat_cols, splits, model=model,
                          lgb_params=lgb_params if model == "lgb" else None)
    ic, ric = calc_ic_series(res.pred, res.label)
    s = ic_summary(ic, ric, horizon=1)

    # 量级对照：允许 2 倍以内的差异（股票池近似 + 单种子 + 测试期股票池不同）
    lo, hi = bench["ic"] / 2.5, bench["ic"] * 2.5
    check(f"IC 落在官方 {bench['ic']:.4f} 的量级内（{lo:.4f}–{hi:.4f}）",
          bool(lo <= s["ic_mean"] <= hi),
          f"我们 IC={s['ic_mean']:.4f}  ICIR={s['icir']:.3f}  "
          f"RankIC={s['ric_mean']:.4f} | 官方 IC={bench['ic']:.4f} "
          f"ICIR={bench['icir']:.3f} RankIC={bench['rank_ic']:.4f}")
    print("      注：官方是真沪深300 成分股 + 20 种子均值；我们是 top300 近似 + 1 种子。"
          "差一个数量级才算有问题，小偏差是预期的。")


# ------------------------------------------------------ 5. embargo 有效性
def verify_embargo(cfg, mcfg, universe: str) -> None:
    """去掉 embargo 重跑，**验证集** IC 应明显虚高——反证 embargo 起了作用。

    注意看的是验证集 IC 而不是测试集 IC：embargo 的作用是防止早停偷看，
    症状就出现在验证集分数上。
    """
    print("\n[5] embargo 有效性（去掉 embargo，验证集 IC 应虚高）")
    from pmsp.model.dataset import build_feature_matrix
    from pmsp.model.walkforward import generate_splits, run_walkforward

    df, feat_cols = build_feature_matrix(
        qlib_dir=cfg.path("data.qlib_dir"), universe=universe,
        label_expr=cfg.label["expr"],
        start_date=cfg.data["start_date"], end_date=cfg.data["end_date"],
        windows=mcfg.features["windows"], cache_dir=abs_path(mcfg.output["cache_dir"]),
    )
    dates = pd.DatetimeIndex(df.index.get_level_values("datetime").unique()).sort_values()
    lgb_params = dict(mcfg.models["lgb"])
    lgb_params.pop("num_boost_round", None)
    lgb_params.pop("early_stopping_rounds", None)

    vals = {}
    for emb in (cfg.split["embargo_days"], 0):
        splits = generate_splits(dates, "2022-01-01", "2024-01-01", retrain_months=12,
                                 embargo_days=emb, valid_days=240,
                                 min_train_days=mcfg.walkforward["min_train_days"])
        res = run_walkforward(df, feat_cols, splits, model="lgb",
                              lgb_params=lgb_params, verbose=False)
        vals[emb] = float(res.log["best_valid_ic"].mean())
        print(f"      embargo={emb} 天 → 验证集 IC {vals[emb]:.4f}")

    emb = cfg.split["embargo_days"]
    check(f"去掉 embargo 后验证集 IC 虚高（{emb} 天 vs 0 天）",
          vals[0] > vals[emb],
          f"embargo={emb}: {vals[emb]:.4f}  →  embargo=0: {vals[0]:.4f}  "
          f"（虚高 {(vals[0] - vals[emb]) / max(abs(vals[emb]), 1e-9):+.1%}）")


def main() -> int:
    cfg, mcfg = load_config(), load_model_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=f"top{cfg.universe['top_n_list'][0]}")
    ap.add_argument("--model", default="lgb", choices=["lgb", "ridge"])
    ap.add_argument("--only", default="fast",
                    choices=["fast", "all", "adjust", "predictions", "lookahead",
                             "benchmark", "embargo"],
                    help="fast = 不联网不重训的项（3+2/6）；adjust 要联网；"
                         "benchmark/embargo 要训练")
    args = ap.parse_args()

    if not (cfg.path("data.qlib_dir") / "calendars" / "day.txt").exists():
        raise SystemExit("还没有 qlib 数据，请先跑 scripts/01_download.py 和 02_build_qlib_data.py")

    print("=" * 70)
    print(f"验证 — 股票池 {args.universe}，模型 {args.model}")
    print("=" * 70)

    want = args.only
    if want in ("fast", "all", "adjust"):
        verify_adjust_internal(cfg)
    if want in ("all", "adjust"):
        verify_adjust_external(cfg)
    if want in ("fast", "all", "lookahead"):
        verify_no_lookahead(cfg, mcfg, args.universe)
    if want in ("fast", "all", "predictions"):
        verify_from_predictions(cfg, mcfg, args.universe, args.model)
    if want in ("all", "benchmark"):
        verify_benchmark(cfg, mcfg, args.model)
    if want in ("all", "embargo"):
        verify_embargo(cfg, mcfg, args.universe)

    n_pass = sum(1 for _, ok, _ in _results if ok is True)
    n_fail = sum(1 for _, ok, _ in _results if ok is False)
    n_skip = sum(1 for _, ok, _ in _results if ok is None)
    print("\n" + "=" * 70)
    print(f"验证结果：{n_pass} 通过 / {n_fail} 失败 / {n_skip} 跳过")
    if n_fail:
        print("\n失败项（**在修好之前 baseline 的数字不可信**）：")
        for name, ok, detail in _results:
            if ok is False:
                print(f"  - {name}: {detail}")
    if n_skip:
        print("\n跳过项：")
        for name, ok, detail in _results:
            if ok is None:
                print(f"  - {name}: {detail}")
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
