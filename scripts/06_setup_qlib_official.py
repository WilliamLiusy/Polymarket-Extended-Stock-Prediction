#!/usr/bin/env python
"""配置 qlib **官方**数据包，并体检它能不能补上我们缺的东西。

结论（2026-09-28 本机实测，细节见 README「数据源盘点」）：**补不上**。
官方包日历止于 2022-12-30、csi300 成分止于 2020-09-25，且和新浪一样没有
`amount`/`vwap`/市值/行业。它唯一的价值是当**离线的弱参照**：能不限流地全市场粗筛
复权差异，但它自己就有错（见下第 4 条），**不是真值**。真正的外部校验用 baostock
（`scripts/07_fetch_baostock.py --what crosscheck`）。

这个脚本做四件事：

1. 下载（断点续传）+ 解包官方包到 `reference.qlib_official_dir`；
2. 体检官方包与我们自己的 `data/qlib_cn`，并排打出来；
3. **复权交叉校验**：抽 N 只股票，比官方复权收益率与我们的收益率序列；
4. 汇报官方包自己的断点（实测 2020-09-28 有一处约 −86% 的假跳变，全市场普遍存在，
   这正是不能把它当"真值"、只能当参照的原因）。

用法：
    python scripts/06_setup_qlib_official.py                 # 全流程（首次约 40 分钟下载）
    python scripts/06_setup_qlib_official.py --skip-download # 已下载过，只体检
    python scripts/06_setup_qlib_official.py --n-stocks 50   # 交叉校验抽 50 只
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pmsp.config import load_config
from pmsp.datasource import qlib_official as qo

#: 官方包在这一天有一处全市场假跳变（数据是两个时间段拼起来的）。
#: 交叉校验时单独报告，不混进"我们错了"的证据里。
KNOWN_SPLICE_DATE = pd.Timestamp("2020-09-28")


def print_inspection(title: str, info: dict) -> None:
    cal = info["calendar"]
    print(f"\n{title}")
    print(f"  目录        {info['dir']}")
    print(f"  日历        {cal['n']} 个交易日，{cal['first']:%Y-%m-%d} → {cal['last']:%Y-%m-%d}")
    print(f"  股票数      {info['n_instruments']}")
    print(f"  字段        {', '.join(info['fields'])}")
    for name, f in info["instrument_files"].items():
        print(f"  {name + '.txt':16s} {f['rows']:5d} 段 / {f['codes']:5d} 只，"
              f"成分区间 {f['first_start']:%Y-%m-%d} → {f['last_end']:%Y-%m-%d}")


def cross_check_adjustment(
    official_dir: Path, our_by_symbol: Path, n_stocks: int, seed: int = 0
) -> pd.DataFrame:
    """抽 n 只股票，比官方复权收益率与我们的。

    比**收益率序列**而不是价格水平：官方把每只股票的价格归一化到首日 = 1.0
    （`SH600000` 首日 close 正好是 1.0），价格水平根本不可比。

    官方 `close` 已是复权价（它的 `pct_change` 与自带的 `change` 字段相关 0.999），
    所以直接对 `close` 求收益率，不要再乘 `factor`——`close/factor` 才是未复权价。
    """
    cal = qo.read_calendar(official_dir)
    codes = sorted(p.name.upper() for p in (official_dir / "features").iterdir() if p.is_dir())
    ours_avail = {p.stem.upper() for p in our_by_symbol.glob("*.parquet")}
    both = sorted(set(codes) & ours_avail)
    rng = np.random.default_rng(seed)
    picked = [both[i] for i in rng.choice(len(both), size=min(n_stocks, len(both)), replace=False)]

    rows = []
    for code in picked:
        off = qo.read_feature(official_dir, code, "close", cal)
        ours = pd.read_parquet(our_by_symbol / f"{code.lower()}.parquet", columns=["date", "close"])
        if off.empty or ours.empty:
            continue
        ours = ours.set_index(pd.to_datetime(ours["date"]))["close"].sort_index()
        r_off, r_our = off.pct_change().dropna(), ours.pct_change().dropna()
        common = r_off.index.intersection(r_our.index)
        if len(common) < 250:
            continue
        diff = (r_off.loc[common] - r_our.loc[common]).abs()
        # 剔掉官方那处已知拼接跳变后再算相关，否则一个点就能把相关从 0.99 打到 0.73
        keep = common[common != KNOWN_SPLICE_DATE]
        rows.append(
            {
                "code": code,
                "n_overlap": len(common),
                "ret_corr": float(r_off.loc[keep].corr(r_our.loc[keep])),
                "median_abs_diff": float(diff.median()),
                "n_diff_gt_1pct": int((diff > 0.01).sum()),
                "worst_day": diff.idxmax().strftime("%Y-%m-%d"),
                "worst_diff": float(diff.max()),
                "splice_diff": float(diff.get(KNOWN_SPLICE_DATE, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=cfg.reference["asset"], choices=list(qo.ASSETS))
    ap.add_argument("--skip-download", action="store_true", help="复用已下载/已解包的")
    ap.add_argument("--n-stocks", type=int, default=30, help="复权交叉校验抽样只数")
    args = ap.parse_args()

    official_dir = cfg.path("reference.qlib_official_dir")
    print("=" * 74)
    print("qlib 官方数据包 — 配置与体检")
    print("=" * 74)

    if not args.skip_download:
        zip_path = qo.download(args.asset, cfg.path("reference.download_dir"))
        qo.extract(zip_path, official_dir)
    if not (official_dir / "calendars" / "day.txt").exists():
        raise SystemExit(f"{official_dir} 里没有 qlib 数据，别加 --skip-download")

    off_info = qo.inspect_dump(official_dir)
    print_inspection("[官方包]", off_info)

    our_dir = cfg.path("data.qlib_dir")
    if (our_dir / "calendars" / "day.txt").exists():
        print_inspection("[我们自己灌的]", qo.inspect_dump(our_dir))

    print("\n[判定] 官方包能补上哪些「已知局限」")
    fields = set(off_info["fields"])
    verdicts = [
        ("成交额 amount", "amount" in fields),
        ("$vwap", "vwap" in fields),
        ("市值 / 行业 / 历史 ST", False),
        (f"覆盖到 {cfg.data['end_date']}（Polymarket 窗口）",
         off_info["calendar"]["last"] >= pd.Timestamp(cfg.data["end_date"])),
        ("指数成分股历史（到今天）",
         off_info["instrument_files"].get("csi300", {}).get("last_end", pd.Timestamp("1900-01-01"))
         >= pd.Timestamp(cfg.data["end_date"])),
    ]
    for name, ok in verdicts:
        print(f"  {'✅' if ok else '❌'} {name}")
    print("  → 一条都补不上。它的用途只有下面这项外部交叉校验，"
          "以及给 instruments 文件格式/量级做参照。")

    print(f"\n[交叉校验] 复权收益率 vs 我们的（抽 {args.n_stocks} 只，离线，不限流）")
    res = cross_check_adjustment(official_dir, cfg.path("data.per_symbol_dir"), args.n_stocks)
    if res.empty:
        print("  两边没有足够重叠的股票，跳过")
        return 0
    print(f"  可比股票 {len(res)} 只，重叠 {res['n_overlap'].median():.0f} 日（中位）")
    print(f"  收益率相关系数：中位 {res['ret_corr'].median():.6f}，"
          f"最低 {res['ret_corr'].min():.6f}，≥0.99 的占 "
          f"{(res['ret_corr'] >= 0.99).mean():.0%}")
    print(f"  单日偏差中位 {res['median_abs_diff'].median():.2e}，"
          f"偏差>1% 的天数中位 {res['n_diff_gt_1pct'].median():.0f}")
    n_splice = int((res["splice_diff"] > 0.5).sum())
    print(f"\n  官方包自己的问题：{KNOWN_SPLICE_DATE:%Y-%m-%d} 单日偏差 >50% 的有 "
          f"{n_splice}/{len(res)} 只 —— 这是官方包的拼接断点，不是我们的错。")
    worst = res.nsmallest(3, "ret_corr")
    print("  相关系数最低的 3 只（若 <0.97 值得去看具体哪天）：")
    for r in worst.itertuples():
        print(f"    {r.code} corr={r.ret_corr:.4f} 最差日 {r.worst_day} "
              f"偏差 {r.worst_diff:.3f}（>1% 的天数 {r.n_diff_gt_1pct}）")

    out = cfg.path("reference.download_dir") / "crosscheck_qlib_official.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\n  明细 -> {out}")
    print("\n下一步：python scripts/07_fetch_baostock.py --what index,industry"
          "（几分钟，真实指数成分 + 行业）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
