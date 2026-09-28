#!/usr/bin/env python
"""用 baostock 补齐新浪源拿不到的字段。免费、无 token、无注册。

四件事，按代价从小到大排：

    --what index       真实沪深300/中证500 成分历史 -> qlib instruments 文件   约 2 分钟
    --what industry    行业分类快照（证监会口径）                              约 5 秒
    --what crosscheck  复权外部交叉校验（原先没有可用的外部源）                约 1 分钟
    --what daily       全市场逐日 amount / turn / isST / tradestatus          十几小时，可续传

`index` / `industry` / `crosscheck` 立刻能跑完，`daily` 是长活。全都**不动
baseline 的任何输入**：写出来的是新文件（`instruments/hs300.txt`、
`data/raw/baostock_by_symbol/*.parquet`），README 里 baseline 的数字不受影响。

## daily 为什么必须并行，而且必须是独立进程

单只股票 16 年约 4000 行，逐行 `next()` 取回约 9–24 秒（服务端速度会变），
5489 只（沪深）单进程要 20 小时以上。而 baostock 在模块全局持有一个 socket，
`multiprocessing.Pool` fork 出来的子进程共用它，实测 12 只股票跑 12 分钟无任何输出
（挂死）。所以这里用 **subprocess 拉起独立的 worker 进程**，每个自己 login。

## 但并发不能开大：8 个 worker 会被拉黑

实测 `--workers 8` 跑 2.5 分钟就触发 `10001011 黑名单用户`：80 只里 48 只落盘、
22 只失败，之后连单进程都登不进去。拉黑按账号（免费用户共用匿名账号）+ IP 判，
重试无用。所以默认 `workers: 3`，且一旦被拉黑 worker 立刻停（不再硬撞），
下次重跑自动跳过已落盘的。全量分几次跑完是正常用法。

用法：
    python scripts/07_fetch_baostock.py --what index,industry
    python scripts/07_fetch_baostock.py --what crosscheck --n-stocks 5
    python scripts/07_fetch_baostock.py --what daily --limit 50        # 先试水
    python scripts/07_fetch_baostock.py --what daily                   # 全量（workers 取配置里的 3），可反复重跑续传
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pmsp.build.adjust import compare_adjustment
from pmsp.build.index_universe import (
    build_index_instruments,
    compare_with_reference,
    fetch_index_snapshots,
)
from pmsp.config import load_config
from pmsp.datasource import qlib_official as qo
from pmsp.datasource.baostock_daily import (
    BaostockBlacklisted,
    BaostockDaily,
    derive_fields,
    is_supported,
)

WHAT_CHOICES = ["index", "industry", "crosscheck", "daily", "all"]


# ------------------------------------------------------------------ 指数成分
def do_index(cfg, args) -> None:
    qlib_dir = cfg.path("data.qlib_dir")
    trade_dates = qo.read_calendar(qlib_dir)
    local_codes = set(qo.read_instruments(qlib_dir, "all")["code"])
    out_dir = cfg.path("reference.download_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    with BaostockDaily(rate_limit_per_min=cfg.supplement["rate_limit_per_min"]) as ds:
        for index in cfg.supplement["indexes"]:
            print(f"\n[指数成分] {index} —— 按月末交易日抓快照 "
                  f"({trade_dates[0]:%Y-%m} → {trade_dates[-1]:%Y-%m})")
            snaps = fetch_index_snapshots(ds, index, trade_dates)
            sizes = pd.Series({d: len(s) for d, s in snaps.items()})
            print(f"           {len(snaps)} 个快照，每期成分数 "
                  f"min={sizes.min()} / 中位={sizes.median():.0f} / max={sizes.max()}")

            rep = build_index_instruments(snaps, trade_dates, qlib_dir, index, local_codes)
            print(f"           -> {rep['path']}")
            print(f"           去重代码 {rep['n_codes_index']} 只，本地有数据 "
                  f"{rep['n_codes_written']} 只（覆盖率 {rep['coverage']:.2%}），"
                  f"{rep['n_segments']} 段")
            if rep["n_dropped"]:
                print(f"           [注意] 本地缺 {rep['n_dropped']} 只，样例："
                      f"{rep['dropped_sample']}")
                if rep["coverage"] < 0.99:
                    print("           覆盖率 <99%，去查这些代码为什么没下到——"
                          "指数成分缺股票会让整段截面静默变薄")

            pd.DataFrame(
                [{"date": d, "code": c} for d, s in snaps.items() for c in sorted(s)]
            ).to_parquet(out_dir / f"index_members_{index}.parquet", index=False)

            # 与官方 csi300 的重合率（只有 hs300 可比，且只有 2010→2020 重叠段）
            official = cfg.path("reference.qlib_official_dir")
            ref_name = {"hs300": "csi300", "zz500": "csi500"}.get(index)
            ref_path = official / "instruments" / f"{ref_name}.txt" if ref_name else None
            if ref_path is not None and ref_path.exists():
                ours = qo.read_instruments(qlib_dir, index)
                ref = qo.read_instruments(official, ref_name)
                check_dates = [d for d in trade_dates if d.day <= 3 and d.month in (1, 7)]
                cmp = compare_with_reference(ours, ref, check_dates)
                cmp = cmp[cmp["date"] <= ref["end"].max()]
                if not cmp.empty:
                    print(f"           与官方 {ref_name}.txt 的重合率（半年一个检查点，"
                          f"{len(cmp)} 个点）：Jaccard 中位 {cmp['jaccard'].median():.3f}、"
                          f"最低 {cmp['jaccard'].min():.3f}")
                    print("           （量级对照。官方那份也是重建的、且止于 "
                          f"{ref['end'].max():%Y-%m}，不是真值）")


# ------------------------------------------------------------------ 行业
def do_industry(cfg, args) -> None:
    out = cfg.path("reference.download_dir") / "industry_snapshot.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with BaostockDaily(rate_limit_per_min=cfg.supplement["rate_limit_per_min"]) as ds:
        ind = ds.fetch_industry()
    ind.to_parquet(out, index=False)
    n_missing = int(ind["industry"].isna().sum())
    print(f"\n[行业] {len(ind)} 只，{ind['industry'].nunique()} 个行业，"
          f"行业为空 {n_missing} 只")
    print(f"       快照日期取值：{sorted(ind['update_date'].dt.strftime('%Y-%m-%d').unique())}")
    print("       [限制] **只有一个快照**，拿它套到 2010 年有轻微前视偏差。"
          "做行业中性化时必须在报告里写明这一点。")
    print(f"       -> {out}")


# ------------------------------------------------------------------ 复权交叉校验
def do_crosscheck(cfg, args) -> None:
    """用 baostock 的后复权序列检验我们自己的复权。

    这是 README 里那条"外部交叉校验形同虚设"的正解：东财连发 6 次封 IP、
    tushare 无 `daily` 权限，而 baostock 免费且不限流。
    """
    per_symbol = cfg.path("data.per_symbol_dir")
    files = sorted(per_symbol.glob("*.parquet"))
    if not files:
        raise SystemExit(f"{per_symbol} 是空的，先跑 scripts/02_build_qlib_data.py")

    # 挑复权因子跨度最大的（= 期间送转/分红最多 = 最能暴露复权错误）。
    # 逐个读 factor 列很快（每个文件只几千行）。
    spans = []
    for f in files:
        if not is_supported(f.stem.upper()):
            continue
        d = pd.read_parquet(f, columns=["factor"])
        if len(d) < 1000:
            continue
        lo, hi = float(d["factor"].min()), float(d["factor"].max())
        if lo > 0:
            spans.append((hi / lo, f.stem.upper()))
    spans.sort(reverse=True)
    picked = [c for _, c in spans[: args.n_stocks]]
    print(f"\n[复权交叉校验] 抽复权因子跨度最大的 {len(picked)} 只：{picked}")

    rows = []
    with BaostockDaily(rate_limit_per_min=cfg.supplement["rate_limit_per_min"]) as ds:
        for code in picked:
            ours = pd.read_parquet(per_symbol / f"{code.lower()}.parquet",
                                   columns=["date", "close"])
            ref = ds.fetch_adjusted_close(
                code, f"{ours['date'].min():%Y-%m-%d}", f"{ours['date'].max():%Y-%m-%d}"
            )
            if ref.empty:
                print(f"  {code}: baostock 返回空，跳过")
                continue
            cmp = compare_adjustment(ours, ref)
            rows.append({"code": code, **cmp})
            print(f"  {code}: 重叠 {cmp['n_overlap']} 日，收益率相关 "
                  f"{cmp['ret_corr']:.6f}，单日最大偏差 {cmp['max_abs_ret_diff']:.2e}")

    res = pd.DataFrame(rows)
    if res.empty:
        return
    ok = (res["ret_corr"] > 0.999).mean()
    print(f"\n  相关系数 >0.999 的占 {ok:.0%}（中位 {res['ret_corr'].median():.6f}）")
    print("  比的是收益率序列而不是价格水平——复权因子的绝对水位取决于基准日，"
          "只有收益率有唯一正确答案。")
    out = cfg.path("reference.download_dir") / "crosscheck_baostock.csv"
    res.to_csv(out, index=False)
    print(f"  明细 -> {out}")


# ------------------------------------------------------------------ 逐日补充字段
def _shard_worker(cfg, shard: int, n_shard: int, start: str, end: str, limit: int = 0) -> None:
    """worker 进程：只处理自己那一份，落盘后立刻可续传。

    `limit` 必须在**分片之前**截断：先分片再截断的话，父进程算的总量
    （`limit` 只）和子进程实际抓的量（`limit × n_shard` 只）会不一致，
    试水一跑就变成全量。
    """
    out_dir = cfg.path("supplement.by_symbol_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = _todo_list(cfg, out_dir)
    if limit:
        todo = todo[:limit]
    todo = todo[shard::n_shard]
    log = out_dir / f"_worker{shard}.log"
    n_ok = n_empty = n_rows = 0
    # logout_on_exit=False：免费用户共用一个匿名账号，谁 logout 谁就把兄弟 worker 踢下线
    try:
        ds = BaostockDaily(rate_limit_per_min=cfg.supplement["rate_limit_per_min"],
                           logout_on_exit=False)
        ds.login()
    except BaostockBlacklisted as exc:
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"BLACKLISTED shard{shard} 登录即被拒：{exc}\n")
        print(f"shard{shard}: 被拉黑，未开工")
        return
    with ds, open(log, "a", encoding="utf-8") as lf:
        for code in todo:
            try:
                df = ds.fetch_one_symbol(code, start, end)
            except BaostockBlacklisted as exc:
                # 拉黑后继续跑只会刷一屏失败，并且很可能延长封禁。停在这里，
                # 已落盘的部分下次重跑自动续上。
                lf.write(f"BLACKLISTED shard{shard} {code} {exc}\n")
                lf.flush()
                print(f"shard{shard}: 被拉黑，停在 {code}（已完成 {n_ok} 只）")
                return
            except Exception as exc:  # noqa: BLE001  单只失败不该拖垮整个 shard
                lf.write(f"FAIL {code} {exc}\n")
                lf.flush()
                continue
            if df.empty:
                n_empty += 1
            else:
                n_ok += 1
                n_rows += len(df)
            # 先写临时文件再改名：中途被 kill 不会留下半个 parquet 骗过续传逻辑
            tmp = out_dir / f".{code.lower()}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            tmp.rename(out_dir / f"{code.lower()}.parquet")
        lf.write(f"DONE shard{shard} ok={n_ok} empty={n_empty} rows={n_rows}\n")
    print(f"shard{shard}: {n_ok} 只有数据 / {n_empty} 只空 / {n_rows} 行")


def _todo_list(cfg, out_dir: Path) -> list[str]:
    """待抓列表：股票列表里 baostock 支持的（沪深）、且还没落盘的。"""
    symbols = pd.read_parquet(cfg.path("data.symbol_list_path"))
    codes = [c for c in symbols["code"].astype(str) if is_supported(c)]
    return [c for c in codes if not (out_dir / f"{c.lower()}.parquet").exists()]


def do_daily(cfg, args) -> None:
    out_dir = cfg.path("supplement.by_symbol_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = pd.read_parquet(cfg.path("data.symbol_list_path"))
    n_all = len(symbols)
    n_bj = sum(1 for c in symbols["code"].astype(str) if not is_supported(c))
    todo = _todo_list(cfg, out_dir)
    done = n_all - n_bj - len(todo)
    print(f"\n[逐日补充] 股票列表 {n_all} 只，其中北交所 {n_bj} 只 baostock 不支持（跳过）")
    print(f"           已完成 {done} 只，待抓 {len(todo)} 只")
    if not todo:
        print("           全部完成。下一步：--what crosscheck，或把 amount/vwap 接进面板")
        return
    if args.limit:
        todo = todo[: args.limit]
        print(f"           [限制] 本次只跑前 {len(todo)} 只")

    n_shard = max(1, args.workers)
    if n_shard == 1:
        _shard_worker(cfg, 0, 1, args.start, args.end, args.limit)
        return

    # 用独立进程而不是 multiprocessing：baostock 的全局 socket 经 fork 会挂死
    print(f"           起 {n_shard} 个独立 worker 进程（baostock 的 socket 不能 fork）")
    t0 = time.time()
    procs = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--what", "daily",
             "--_shard", str(k), "--_nshard", str(n_shard),
             "--start", args.start, "--end", args.end]
            + (["--limit", str(args.limit)] if args.limit else []),
        )
        for k in range(n_shard)
    ]
    n_target = done + len(todo)
    while any(p.poll() is None for p in procs):
        time.sleep(30)
        have = len(list(out_dir.glob("*.parquet")))
        rate = (have - done) / max(time.time() - t0, 1) * 3600
        left = (n_target - have) / max(rate, 1e-9)
        print(f"           {have}/{n_target} 只（{rate:.0f} 只/小时，"
              f"预计还需 {left:.1f} 小时）", flush=True)
    for p in procs:
        p.wait()
    print(f"\n           本轮用时 {(time.time() - t0) / 60:.1f} 分钟，"
          f"落盘 {len(list(out_dir.glob('*.parquet')))} 只")
    lines = [
        line for f in out_dir.glob("_worker*.log")
        for line in f.read_text(encoding="utf-8").splitlines()
    ]
    fails = sum(1 for line in lines if line.startswith("FAIL"))
    banned = sum(1 for line in lines if line.startswith("BLACKLISTED"))
    if fails:
        print(f"           [失败] {fails} 条（见 {out_dir}/_worker*.log），重跑本脚本会自动重试")
    if banned:
        print(f"           [拉黑] {banned} 个 worker 撞上 10001011 黑名单用户。"
              f"并发太高了——baostock 按账号+IP 判，免费用户共用一个匿名账号。")
        print(f"           降到 --workers 2~3 并等一段时间再重跑（已落盘的会自动跳过）。")

    _report_daily(cfg, out_dir)


def _report_daily(cfg, out_dir: Path) -> None:
    """抽查已落盘的数据，把补上的字段兑现成几个数字。"""
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        return
    rng = np.random.default_rng(0)
    pick = [files[i] for i in rng.choice(len(files), size=min(200, len(files)), replace=False)]
    dfs = [pd.read_parquet(f) for f in pick]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        print("           抽查的文件全是空的（可能都是无数据的死代码）")
        return
    d = derive_fields(pd.concat(dfs, ignore_index=True))
    print(f"\n[抽查] {len(dfs)} 只 / {len(d):,} 行，"
          f"{d['date'].min():%Y-%m-%d} → {d['date'].max():%Y-%m-%d}")
    print(f"       amount 非空 {d['amount'].notna().mean():.2%}，"
          f"vwap 非空 {d['vwap'].notna().mean():.2%}，"
          f"流通市值非空 {d['float_mv'].notna().mean():.2%}")
    print(f"       vwap/close 中位 {(d['vwap'] / d['close']).median():.4f}"
          f"（应≈1，量纲对不上就是单位错了）")
    print(f"       停牌日 {int(d['suspended'].sum()):,} 行（{d['suspended'].mean():.2%}），"
          f"ST 日 {int(d['is_st'].eq(1).sum()):,} 行（{d['is_st'].eq(1).mean():.2%}）")
    # 用 close×volume 代理 vs 真 amount：README 说"对排名影响可忽略"，这里量化它
    proxy = d["close"] * d["volume"]
    rel = ((proxy - d["amount"]) / d["amount"]).abs().dropna()
    print(f"       close×volume 代理 vs 真 amount：相对偏差中位 {rel.median():.3%}、"
          f"99 分位 {rel.quantile(0.99):.3%}")


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="index,industry",
                    help=f"逗号分隔，可选 {WHAT_CHOICES}")
    ap.add_argument("--workers", type=int, default=cfg.supplement["workers"])
    ap.add_argument("--limit", type=int, default=0, help="daily：本次最多抓多少只")
    ap.add_argument("--n-stocks", type=int, default=5, help="crosscheck：抽样只数")
    ap.add_argument("--start", default=cfg.data["start_date"])
    ap.add_argument("--end", default=cfg.data["end_date"])
    # 内部参数：worker 子进程用，不写进 --help 的正经用法里
    ap.add_argument("--_shard", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_nshard", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._shard >= 0:  # noqa: SLF001  自己传给自己的
        _shard_worker(cfg, args._shard, args._nshard, args.start, args.end, args.limit)
        return 0

    wants = ["index", "industry", "crosscheck", "daily"] if "all" in args.what \
        else [w.strip() for w in args.what.split(",") if w.strip()]
    bad = [w for w in wants if w not in WHAT_CHOICES]
    if bad:
        raise SystemExit(f"未知的 --what：{bad}，可选 {WHAT_CHOICES}")

    print("=" * 74)
    print("baostock 补充字段 —— " + " / ".join(wants))
    print("=" * 74)
    if "index" in wants:
        do_index(cfg, args)
    if "industry" in wants:
        do_industry(cfg, args)
    if "crosscheck" in wants:
        do_crosscheck(cfg, args)
    if "daily" in wants:
        do_daily(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
