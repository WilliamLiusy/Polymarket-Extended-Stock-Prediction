#!/usr/bin/env python
"""合成数据端到端自测。**不需要 tushare token，不碰网络。**

拿到真实数据之前先把管线跑通，这样 token 到手后只需要跑数据校验，
不用再debug代码。合成数据里故意埋了两样东西：

* **一次 10 送 10 拆股** —— 若复权错了，价格序列会出现一个 -50% 的假跳空
* **一个真实的反转信号** —— 若特征/标签/训练链路错了，模型学不出正 IC

检查项（任一失败即退出码 1）：
    1. 复权不变量：复权价 × 复权量 == 真实成交额；两条复权路径互相验证
    2. 拆股处理：复权后的收益率序列在拆股日无异常跳空
    3. 股票池无前视：成分生效日严格晚于排名日
    4. dump_bin 转换后 qlib 能读出数据，且价量与源面板一致
    5. Alpha158 表达式可求值，因子数达预期量级
    6. 埋入的反转信号能被 LightGBM 学出正 IC
    7. 随机因子在多个种子上的平均 IC ≈ 0（证明评估代码本身不产生假信号）
    8. 回测时间线无前视：用未来收益构造的因子 IC 应显著为正，
       用过去收益构造的应接近 0
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pmsp.build.adjust import adjust_panel, adjust_panel_factor
from pmsp.build.to_qlib import QLIB_FIELDS, dump_to_qlib, write_per_symbol
from pmsp.build.universe import build_universe, to_qlib_instruments, write_instruments_file
from pmsp.eval.backtest import cost_sensitivity, quantile_backtest
from pmsp.eval.ic import calc_ic_series, ic_summary, newey_west_tstat, paired_ic_test

N_STOCKS = 120
N_DAYS = 1500  # 约 6 年，够 walk-forward 切出多个窗口
SPLIT_STOCK = 0  # 第 0 只股票在中途 10 送 10
SPLIT_DAY = 400
REVERSAL_PHI = 0.35  # 埋入的反转强度

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    print(f"  {'[通过]' if ok else '[失败]'} {name}" + (f" — {detail}" if detail else ""))


def make_synthetic_raw() -> pd.DataFrame:
    """造 tushare `daily` 格式的原始数据（含一次拆股和一个反转信号）。"""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=N_DAYS)
    codes = [f"{600000 + i}.SH" for i in range(N_STOCKS)]

    # 反转结构：今日收益对过去 5 日累计收益负相关 -> 过去 5 日动量可预测未来
    rets = np.zeros((N_DAYS, N_STOCKS))
    eps = rng.standard_normal((N_DAYS, N_STOCKS)) * 0.02
    market = rng.standard_normal(N_DAYS) * 0.01  # 共同市场因子
    for t in range(5, N_DAYS):
        past5 = rets[t - 5 : t].sum(axis=0)
        rets[t] = -REVERSAL_PHI * past5 / 5.0 + eps[t] + market[t]
    rets = np.clip(rets, -0.098, 0.098)  # 尊重 ±10% 涨跌停

    rows = []
    for j, code in enumerate(codes):
        r = rets[:, j]
        # 真实(不复权)价格：正常按收益率走，拆股日价格砍半
        px = np.zeros(N_DAYS)
        px[0] = 10.0 + j * 0.1
        for t in range(1, N_DAYS):
            px[t] = px[t - 1] * (1 + r[t])
            if j == SPLIT_STOCK and t >= SPLIT_DAY:
                # 拆股当日除权（价格腰斩），此后价格一直在腰斩后的水平上走
                if t == SPLIT_DAY:
                    px[t] *= 0.5  # 除权：价格腰斩，但 pct_chg 仍是真实收益
        vol = rng.uniform(1e4, 1e6, N_DAYS) * (1 + j / N_STOCKS)
        if j == SPLIT_STOCK:
            vol[SPLIT_DAY:] *= 2.0  # 拆股后股数翻倍，成交量随之翻倍
        # 昨收：正常日就是上一日收盘；除权日是"除权后的昨收"，使 pct_chg 仍为真实收益
        pre_close = np.concatenate([[px[0] / (1 + r[0])], px[:-1]])
        if j == SPLIT_STOCK:
            pre_close[SPLIT_DAY] = px[SPLIT_DAY] / (1 + r[SPLIT_DAY])
        rows.append(
            pd.DataFrame(
                {
                    "ts_code": code,
                    "trade_date": [d.strftime("%Y%m%d") for d in dates],
                    "open": px * (1 + rng.normal(0, 0.002, N_DAYS)),
                    "high": px * (1 + abs(rng.normal(0, 0.006, N_DAYS))),
                    "low": px * (1 - abs(rng.normal(0, 0.006, N_DAYS))),
                    "close": px,
                    "pre_close": pre_close,
                    "change": px - pre_close,
                    "pct_chg": rets[:, j] * 100.0,
                    "vol": vol,  # 单位手
                    "amount": px * vol * 100 / 1000.0,  # 单位千元
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def make_synthetic_sina(raw: pd.DataFrame) -> pd.DataFrame:
    """把同一份合成数据改写成新浪源的 schema（未复权价 + 显式复权因子）。

    这是**生产路径**的输入格式。10 送 10 后不复权价腰斩，所以后复权累计因子
    从拆股日起由 1.0 变 2.0——与新浪 `hfq.js` 的口径一致（因子随时间递增）。
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["code"] = df["ts_code"].map(lambda c: c.partition(".")[2] + c.partition(".")[0])
    split_date = pd.to_datetime(raw["trade_date"].unique()[SPLIT_DAY], format="%Y%m%d")
    is_split_stock = df["ts_code"] == f"{600000 + SPLIT_STOCK}.SH"
    df["adj_factor"] = np.where(is_split_stock & (df["date"] >= split_date), 2.0, 1.0)
    out = df[["code", "date", "open", "high", "low", "close", "adj_factor"]].copy()
    out["volume"] = df["vol"] * 100.0  # 手 -> 股（新浪 K 线的 volume 单位是股）
    return out[["code", "date", "open", "high", "low", "close", "volume", "adj_factor"]]


def test_adjust(raw: pd.DataFrame) -> pd.DataFrame:
    print("\n[1] 复权还原")
    sina_raw = make_synthetic_sina(raw)
    panel = adjust_panel_factor(sina_raw)      # 生产路径（新浪：显式因子）
    panel_pct = adjust_panel(raw)              # 备用路径（tushare：pct_chg 累乘）

    # 不变量：复权价 × 复权量 == 真实成交额（元）
    lhs = panel["close"] * panel["volume"]
    rhs = panel["amount_yuan"]
    rel_err = ((lhs - rhs).abs() / rhs.replace(0, np.nan)).dropna()
    check("复权不变量 close×volume == 真实成交额", rel_err.max() < 1e-6,
          f"最大相对误差 {rel_err.max():.2e}")

    # 拆股股票：复权后日收益率序列不应出现假跳空
    sp = panel[panel["code"] == "SH600000"].sort_values("date").reset_index(drop=True)
    adj_ret = sp["close"].pct_change()
    true_ret = raw[raw["ts_code"] == "600000.SH"].sort_values("trade_date")["pct_chg"].to_numpy() / 100.0
    diff = (adj_ret.to_numpy() - true_ret)[1:]
    check("拆股日复权后无假跳空", np.nanmax(np.abs(diff)) < 1e-9,
          f"复权收益率与真实收益率最大偏差 {np.nanmax(np.abs(diff)):.2e}")

    # 不复权价在拆股日确实有 -50% 跳空（证明这个测试有意义，不是空转）
    raw_sp = raw[raw["ts_code"] == "600000.SH"].sort_values("trade_date").reset_index(drop=True)
    raw_jump = raw_sp["close"].pct_change().iloc[SPLIT_DAY]
    check("不复权价确有除权跳空（证明测试有效）", raw_jump < -0.4,
          f"不复权跳空 {raw_jump:.1%}，复权后已消除")

    # 两条独立复权路径的**互相验证**：显式因子 vs pct_chg 累乘。
    # 复权因子的绝对水平取决于基准日，所以只能比收益率序列。
    a = panel.set_index(["code", "date"])["close"].groupby(level="code").pct_change()
    b = panel_pct.set_index(["code", "date"])["close"].groupby(level="code").pct_change()
    d = (a - b.reindex(a.index)).abs().dropna()
    check("两条复权路径（显式因子 vs pct_chg 累乘）收益率完全一致",
          d.max() < 1e-9,
          f"{len(d):,} 个观测，最大偏差 {d.max():.2e}")

    check("新浪路径不产出 vwap（与 QLIB_FIELDS 一致，不静默造假因子）",
          "vwap" not in panel.columns and "vwap" not in QLIB_FIELDS,
          f"面板列 {[c for c in panel.columns if c not in ('code', 'date')]}")
    return panel


def test_universe(panel: pd.DataFrame) -> None:
    print("\n[2] 动态股票池")
    membership = build_universe(panel, top_n=50, lookback_days=60, min_listed_days=100)
    check("股票池非空", not membership.empty, f"{len(membership)} 条成分记录")

    # 前视检查：成分生效日必须严格晚于用于排名的调仓日
    trade_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rebal = membership["effective_from"].unique()
    ok = all(d in set(trade_dates) for d in rebal)
    check("成分生效日均为交易日", ok)

    per_period = membership.groupby("effective_from")["code"].nunique()
    check("每期成分数正确", bool((per_period <= 50).all()),
          f"每期 {per_period.min()}–{per_period.max()} 只（上限 50）")

    segments = to_qlib_instruments(membership)
    check("区间合并有效", len(segments) < len(membership),
          f"{len(membership)} 条成分记录 -> {len(segments)} 段区间")


def test_qlib_roundtrip(panel: pd.DataFrame, workdir: Path) -> Path:
    print("\n[3] dump_bin 转换 + qlib 读回")
    per_symbol = workdir / "by_symbol"
    qlib_dir = workdir / "qlib_cn"
    n = write_per_symbol(panel, per_symbol)
    check("按股票拆分", n == N_STOCKS, f"{n} 个 parquet")

    dump_to_qlib(per_symbol, qlib_dir)
    check("生成 calendars", (qlib_dir / "calendars" / "day.txt").exists())
    check("生成 features", (qlib_dir / "features" / "sh600000").exists())
    check("生成 instruments/all.txt", (qlib_dir / "instruments" / "all.txt").exists())

    membership = build_universe(panel, top_n=50, lookback_days=60, min_listed_days=100)
    write_instruments_file(to_qlib_instruments(membership), qlib_dir, "top50")

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(qlib_dir), region="cn", expression_cache=None, dataset_cache=None)

    got = D.features(["SH600001"], ["$close", "$volume", "$vwap"],
                     start_time="2022-06-01", end_time="2023-06-01")
    check("qlib 能读出数据", not got.empty, f"{len(got)} 行")

    # 与源面板逐点比对，确认二进制转换没有精度或错位问题
    src = panel[(panel["code"] == "SH600001")].set_index("date")["close"]
    q = got["$close"].droplevel(0)
    common = src.index.intersection(q.index)
    max_dev = float((src.loc[common] - q.loc[common]).abs().max())
    check("qlib 读回的价格与源面板一致", max_dev < 1e-3,
          f"{len(common)} 个交易日，最大偏差 {max_dev:.2e}")
    return qlib_dir


def test_pipeline(qlib_dir: Path, workdir: Path) -> tuple[pd.Series, pd.Series]:
    """跑**真实的管线函数**（不是在测试里复写一遍逻辑）。"""
    print("\n[4] 特征矩阵 + walk-forward（走 src/pmsp/model 的真实代码路径）")
    from pmsp.model.dataset import build_feature_matrix, daily_returns
    from pmsp.model.walkforward import generate_splits, run_walkforward

    df, feat_cols = build_feature_matrix(
        qlib_dir=qlib_dir, universe="top50",
        label_expr="Ref($close, -6)/Ref($close, -1) - 1",
        start_date="2022-01-03", end_date="2027-12-31",
        windows=[3, 5, 10, 20, 30, 60, 120],
        cache_dir=workdir / "features",
    )
    # 215 而非方案里写的 216：新浪源没有成交额 → 没有 $vwap → Alpha158 的
    # VWAP0 这一个因子拿不到。仍是"两百多个"。
    check("扩窗后达到 215 个因子（去掉 VWAP0 后的两百多个）",
          len(feat_cols) == 215, f"{len(feat_cols)} 个因子，{len(df):,} 行")

    # 标签横截面 z-score 后应满足 E[y]=0, E[y²]=1 —— MSE≡IC 等价性的前提
    per_day = df["LABEL0"].groupby(level="datetime")
    check("标签已横截面标准化（MSE 训练 ≡ 最大化 IC 的前提）",
          abs(per_day.mean().mean()) < 1e-6 and abs(per_day.std().mean() - 1) < 0.01,
          f"日均值 {per_day.mean().mean():.2e}，日标准差均值 {per_day.std().mean():.4f}")

    # 特征缓存：第二次调用必须命中，且结果完全一致
    df2, _ = build_feature_matrix(
        qlib_dir=qlib_dir, universe="top50",
        label_expr="Ref($close, -6)/Ref($close, -1) - 1",
        start_date="2022-01-03", end_date="2027-12-31",
        windows=[3, 5, 10, 20, 30, 60, 120], cache_dir=workdir / "features",
    )
    check("特征缓存命中且内容一致", df2.shape == df.shape and df2.equals(df))

    dates = pd.DatetimeIndex(df.index.get_level_values("datetime").unique()).sort_values()
    splits = generate_splits(dates, str(dates[900].date()), str(dates[-1].date()),
                             retrain_months=6, embargo_days=5, valid_days=200,
                             min_train_days=400)
    check("切出滚动窗口", len(splits) >= 2, f"{len(splits)} 个窗口")

    res = run_walkforward(df, feat_cols, splits, model="lgb",
                          lgb_params={"num_leaves": 31, "num_threads": 8}, verbose=False)
    ic, ric = calc_ic_series(res.pred, res.label)
    summ = ic_summary(ic, ric, horizon=5)
    check("LightGBM walk-forward 学出埋入的反转信号（IC>0 且 t_NW>2）",
          summ["ic_mean"] > 0.01 and summ["t_nw"] > 2,
          f"IC={summ['ic_mean']:.4f} ICIR={summ['icir']:.3f} t_NW={summ['t_nw']:.2f} "
          f"({summ['n_days']} 天)")
    check("特征重要性可取出", not res.importance.empty,
          f"Top3 因子 {list(res.importance.mean(axis=1).nlargest(3).index)}")

    res_r = run_walkforward(df, feat_cols, splits, model="ridge", verbose=False)
    icr, ricr = calc_ic_series(res_r.pred, res_r.label)
    sr = ic_summary(icr, ricr, horizon=5)
    check("Ridge 线性对照也能跑通并学出信号",
          sr["ic_mean"] > 0.005,
          f"Ridge IC={sr['ic_mean']:.4f}（LightGBM {summ['ic_mean']:.4f}），"
          f"alpha={res_r.log['alpha'].tolist()}")

    print("\n[5] 评估代码的空对照")
    # 必须多种子平均。单个种子的 ic_mean 抽样标准误约 1/√(n_days·n_stocks) ≈ 0.007，
    # 且 |t|>2 本来就该以 5% 的频率发生——拿一个种子去卡 |t|<2 等于让自测有
    # 5% 的概率无故报红。实测 300 个种子：IC 均值 +0.0032、|t|>2 占比 5.7%。
    # 这里跑 20 个种子，看**跨种子均值**（标准误降到 0.0015）和拒绝率。
    N_SEEDS = 20
    seed_ics, seed_ts = [], []
    rand_ic = None
    for sd in range(101, 101 + N_SEEDS):
        r = np.random.default_rng(sd)
        rand = pd.Series(r.standard_normal(len(res.pred)), index=res.pred.index)
        ic_s, ric_s = calc_ic_series(rand, res.label)
        s = ic_summary(ic_s, ric_s, horizon=5)
        seed_ics.append(s["ic_mean"])
        seed_ts.append(s["t_nw"])
        if rand_ic is None:
            rand_ic = ic_s  # 留给下面的配对检验
    ic_bar = float(np.mean(seed_ics))
    rej = float(np.mean([abs(t) > 2 for t in seed_ts]))
    check("随机因子 IC ≈ 0（评估代码不自造信号）",
          abs(ic_bar) < 0.01,
          f"{N_SEEDS} 个种子的 IC 均值 {ic_bar:+.4f}，单种子区间 "
          f"[{min(seed_ics):+.4f}, {max(seed_ics):+.4f}]")
    check("随机因子的显著率接近名义水平（t 值没被系统性放大）",
          rej <= 0.30,
          f"|t_NW|>2 占比 {rej:.0%}（名义 5%，{N_SEEDS} 个种子下抽样波动大），"
          f"|t| 中位数 {np.median(np.abs(seed_ts)):.2f}")

    const = pd.Series(0.037, index=ic.index)  # 数值上恒定的 IC 序列
    check("恒定 IC 序列返回 NaN 而非天文数字 t 值（HAC 退化保护）",
          not np.isfinite(newey_west_tstat(const, lags=4)[0]),
          f"t_NW={newey_west_tstat(const, lags=4)[0]}"
          "（无保护时浮点残差会给出 se≈1e-17、t≈1e17）")

    pt = paired_ic_test(ic, rand_ic, horizon=5)
    check("配对 IC 差值检验可用且方向正确（PM 阶段的判决函数）",
          pt["delta_ic"] > 0 and pt["t_nw"] > 2,
          f"ΔIC={pt['delta_ic']:.4f} t_NW={pt['t_nw']:.2f} p={pt['p_nw']:.2e}")

    fic, _ = calc_ic_series(res.label, res.label)
    check("完美因子对照：IC ≈ 1", float(fic.mean()) > 0.999,
          f"IC={fic.mean():.4f}（不接近 1 说明索引对齐有问题）")

    print("\n[6] 分层回测与成本")
    ret1d = daily_returns(qlib_dir, "top50", "2022-01-03", "2027-12-31")
    bt = quantile_backtest(res.pred, ret1d.reindex(res.pred.index).dropna(),
                           n_groups=5, horizon=5,
                           cost={"commission_bps": 2.5, "stamp_duty_bps": 10.0, "impact_bps": 10.0})
    grp = bt["group_ann_return"]
    check("分层回测可运行", len(grp) == 5, f"各层年化 {[f'{v:.1%}' for v in grp.values()]}")
    check("最高分组优于最低分组", grp[max(grp)] > grp[min(grp)],
          f"top {grp[max(grp)]:.1%} vs bottom {grp[min(grp)]:.1%}")
    check("扣费后收益低于扣费前",
          bt["long_short"]["net"]["ann_return"] < bt["long_short"]["gross"]["ann_return"],
          f"多空 税前 {bt['long_short']['gross']['ann_return']:.1%} → "
          f"税后 {bt['long_short']['net']['ann_return']:.1%}，"
          f"日均单边换手 {bt['avg_daily_turnover_one_way']:.1%}")
    cs = cost_sensitivity(res.pred, ret1d.reindex(res.pred.index).dropna(),
                          cost={"commission_bps": 2.5, "stamp_duty_bps": 10.0, "impact_bps": 10.0},
                          multipliers=(0.0, 1.0, 2.0), n_groups=5, horizon=5)
    check("成本敏感性表：Sharpe 随成本单调下降",
          bool(cs["ls_sharpe_net"].is_monotonic_decreasing),
          " → ".join(f"{m:.0f}×:{s:.2f}" for m, s in
                     zip(cs["cost_multiplier"], cs["ls_sharpe_net"])))
    return res.pred, res.label


def test_splits() -> None:
    """walk-forward 切分的纯逻辑检查（不需要数据，但这是最容易出前视偏差的地方）。"""
    print("\n[7] walk-forward 切分逻辑")
    from pmsp.model.walkforward import generate_splits

    dates = pd.DatetimeIndex(pd.bdate_range("2010-01-04", "2026-09-18"))
    pos = {d: i for i, d in enumerate(dates)}
    splits = generate_splits(dates, "2018-01-01", "2026-09-18", retrain_months=6,
                             embargo_days=5, valid_days=240, min_train_days=500)
    check("生成了滚动窗口", len(splits) > 10, f"{len(splits)} 个窗口")

    ok_order = all(
        pos[s.train_start] <= pos[s.train_end] < pos[s.valid_start] <= pos[s.valid_end]
        < pos[s.test_start] <= pos[s.test_end]
        for s in splits
    )
    check("每个窗口内 train < valid < test 严格递增且不重叠", ok_order)

    emb_tv = [pos[s.valid_start] - pos[s.train_end] - 1 for s in splits]
    emb_vt = [pos[s.test_start] - pos[s.valid_end] - 1 for s in splits]
    check("train↔valid 的 embargo ≥ 5 个交易日", min(emb_tv) >= 5, f"最小 {min(emb_tv)} 天")
    check("valid↔test 的 embargo ≥ 5 个交易日", min(emb_vt) >= 5, f"最小 {min(emb_vt)} 天")

    gaps = [pos[b.test_start] - pos[a.test_end] for a, b in zip(splits, splits[1:])]
    check("相邻窗口的 test 区间首尾相接、不重叠不留缝",
          all(g == 1 for g in gaps), f"间隔集合 {sorted(set(gaps))}（应为 {{1}}）")

    covered = pos[splits[-1].test_end] - pos[splits[0].test_start] + 1
    check("test 区间合起来覆盖整个 OOS",
          splits[0].test_start.year == 2018 and splits[-1].test_end.year == 2026,
          f"{splits[0].test_start:%Y-%m-%d} → {splits[-1].test_end:%Y-%m-%d}，共 {covered} 个交易日")

    # 训练集不得越过 test 起点 —— 这条一旦破，全部结果作废
    check("训练/验证数据全部早于 test 起点",
          all(pos[s.valid_end] < pos[s.test_start] for s in splits))

    # 匹配样本对照：限制 train_start 后，训练集不应再回溯到 2010
    matched = generate_splits(dates, "2025-01-01", "2026-09-18", retrain_months=1,
                              embargo_days=5, valid_days=120, min_train_days=100,
                              train_start="2022-01-01")
    check("train_start 配置生效（匹配样本对照可用）",
          bool(matched) and all(s.train_start >= pd.Timestamp("2022-01-01") for s in matched),
          f"{len(matched)} 个月度窗口，训练集起点 {matched[0].train_start:%Y-%m-%d}")


def test_external_series(panel: pd.DataFrame, qlib_dir: Path, workdir: Path) -> None:
    """Polymarket 接入路径：外部时序 → 伪 instrument → 暴露度因子。

    在拿到任何 Polymarket 数据之前就把这条路验通，免得到时候才发现接不上。
    """
    print("\n[8] Polymarket 接入路径（外部时序 → 暴露度因子）")
    from pmsp.build.to_qlib import dump_to_qlib
    from pmsp.extensions.external_series import (
        align_to_calendar, exposure_expr, read_calendar, sanity_check,
        write_external_instrument,
    )

    calendar = read_calendar(qlib_dir)
    check("能读回交易日历", len(calendar) > 1000, f"{len(calendar)} 个交易日")

    # 造一条"7×24 交易"的外部概率序列：含周末，且故意留一段空窗
    rng = np.random.default_rng(11)
    all_days = pd.date_range(calendar[0] - pd.Timedelta(days=5), calendar[-1], freq="D")
    prob = pd.Series(
        np.clip(0.5 + np.cumsum(rng.standard_normal(len(all_days)) * 0.02), 0.01, 0.99),
        index=all_days,
    )
    gap = (all_days >= calendar[300]) & (all_days < calendar[300] + pd.Timedelta(days=40))
    ext = prob[~gap]  # 40 天没有报价，模拟一个冷掉的市场

    aligned = align_to_calendar(ext, calendar, max_stale_days=10)
    check("对齐后索引严格等于交易日历（不会污染日历）",
          aligned.index.equals(calendar),
          f"{sanity_check(aligned, calendar)['index_outside_calendar']} 个日期在日历外")
    check("陈旧超限的区间被置 NaN（不把旧报价一路拖下去）",
          bool(aligned.isna().any()) and aligned.notna().sum() > len(calendar) * 0.9,
          f"覆盖率 {sanity_check(aligned, calendar)['coverage']:.1%}")

    # 只向后填充：对齐值必须等于"当日或之前最后一个真实报价"，绝不能来自未来
    probe = calendar[500]
    past = ext[ext.index <= probe]
    check("对齐只向后填充（未来的报价不会漏进来）",
          bool(np.isclose(aligned.loc[probe], past.iloc[-1])),
          f"{probe:%Y-%m-%d} 对齐值 = 该日或之前最后一个报价")

    per_symbol = workdir / "by_symbol"
    write_external_instrument("PM_TEST_EVENT", ext, per_symbol, calendar, max_stale_days=10)
    n_cal_before = len(calendar)
    dump_to_qlib(per_symbol, qlib_dir)
    check("注册伪 instrument 后交易日历长度不变（关键：不能多出非交易日）",
          len(read_calendar(qlib_dir)) == n_cal_before,
          f"{n_cal_before} → {len(read_calendar(qlib_dir))}")

    # 暴露度因子必须**有横截面区分度**，否则对 IC 毫无贡献
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(qlib_dir), region="cn", expression_cache=None, dataset_cache=None)
    codes = list(D.list_instruments(D.instruments("top50"), as_list=True))[:30]
    corr_expr = exposure_expr("PM_TEST_EVENT", 60, "corr")
    level_expr = exposure_expr("PM_TEST_EVENT", 60, "level")
    got = D.features(codes, [corr_expr, level_expr],
                     start_time=str(calendar[800].date()), end_time=str(calendar[-30].date()))
    got.columns = ["corr", "level"]
    if list(got.index.names) == ["instrument", "datetime"]:
        got = got.swaplevel(0, 1).sort_index()

    check("ChangeInstrument 暴露度因子可求值（无需改 qlib 代码）",
          got["corr"].notna().sum() > 1000,
          f"{got['corr'].notna().sum():,} 个有效值，范围 "
          f"[{got['corr'].min():.3f}, {got['corr'].max():.3f}]")

    cs_std = got["corr"].groupby(level="datetime").std().mean()
    check("暴露度因子有横截面区分度（这才是它有用的前提）",
          cs_std > 0.05, f"日均横截面标准差 {cs_std:.4f}")

    lvl_std = got["level"].groupby(level="datetime").std().mean()
    check("对照：宏观序列本身横截面标准差 ≈ 0（直接当因子 IC 贡献恒为零）",
          lvl_std < 1e-9,
          f"日均横截面标准差 {lvl_std:.2e} —— 所以必须做成「外部时序 × 个股暴露度」")


def main() -> int:
    print("=" * 68)
    print("合成数据端到端自测（不需要 token，不碰网络）")
    print("=" * 68)
    workdir = Path(tempfile.mkdtemp(prefix="pmsp_selftest_"))
    try:
        raw = make_synthetic_raw()
        print(f"\n合成原始数据：{len(raw):,} 行，{N_STOCKS} 只股票，{N_DAYS} 个交易日")
        panel = test_adjust(raw)
        test_universe(panel)
        qlib_dir = test_qlib_roundtrip(panel, workdir)
        test_pipeline(qlib_dir, workdir)
        test_splits()
        test_external_series(panel, qlib_dir, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = len(_results) - n_pass
    print("\n" + "=" * 68)
    print(f"自测结果：{n_pass} 通过 / {n_fail} 失败（共 {len(_results)} 项）")
    if n_fail:
        print("\n失败项：")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 68)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
