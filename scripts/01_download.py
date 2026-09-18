#!/usr/bin/env python
"""下载全市场日线（新浪源，按股票取数）。每股一个 parquet，可随时中断续传。

## 为什么是按股票而不是按日期

新浪一次请求返回**单只股票的全历史**，所以总请求数 ≈ 股票数 × 2
（一次 K 线 + 一次复权因子），约 5800 只 → 11600 次，而不是交易日数 × 翻页。
限速 40 次/分（1.5s 匀速）下约 5 小时。（tushare 的 `daily` 是按日期取全市场，
但本机 token 没有该接口权限，见 `pmsp.datasource.sina_daily` 的模块说明。）

## 股票池必须包含退市股票

只下在市股票 = 幸存者偏差。所以待下载列表 = 新浪在市名单 ∪ 交易所退市名单。
退市名单里只剔除"数据起点之前就已退市"的——一只 2012 年退市的股票在
2010–2012 年间是可交易的，属于当时的股票池。

## 空结果也要落盘

代码不存在、或全历史都在数据起点之前的股票，会落一个**空 parquet**。
不落的话每次重跑都要把这些死代码再探一遍。

用法：
    python scripts/01_download.py                    # 全量（可反复重跑续传）
    python scripts/01_download.py --limit 50         # 先小规模试水
    python scripts/01_download.py --refresh-symbols  # 强制重建股票列表
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tqdm import tqdm

from pmsp.config import load_config
from pmsp.datasource.exchange_lists import fetch_all_delisted
from pmsp.datasource.sina_daily import SYMBOL_COLUMNS, SinaDaily, market_prefix


def build_symbol_list(ds: SinaDaily, start_date: str, cache: Path,
                      refresh: bool = False) -> pd.DataFrame:
    """在市名单 ∪ 退市名单。结果缓存到 parquet，默认复用。"""
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        print(f"[股票列表] 复用缓存 {cache.name}：{len(df)} 只"
              f"（其中已退市 {int(df['delisted'].sum())} 只）")
        return df

    print("[股票列表] 拉取新浪在市名单 ...")
    listed = ds.list_listed_symbols()
    listed["delisted"] = False
    print(f"           在市 {len(listed)} 只：{listed['code'].str[:2].value_counts().to_dict()}")

    print("[股票列表] 拉取交易所退市名单 ...")
    delisted = fetch_all_delisted(min_delist_date=start_date)
    if not delisted.empty:
        # 交易所名单里的代码没有交易所前缀信息可直接信（B 股/老代码混杂），
        # 统一用代码段重新判交易所，避免 sz/sh 贴错导致整只股票拉不到数据
        delisted["sina_symbol"] = delisted["code"].map(market_prefix) + delisted["code"]
        add = pd.DataFrame(
            {
                "code": delisted["sina_symbol"].str.upper(),
                "sina_symbol": delisted["sina_symbol"],
                "name": delisted["name"],
                "delisted": True,
            }
        )
        print(f"           退市 {len(add)} 只（已剔除 {start_date} 之前就退市的）")
    else:
        add = pd.DataFrame(columns=["code", "sina_symbol", "name", "delisted"])

    df = pd.concat([listed, add], ignore_index=True)
    df = df.drop_duplicates(subset=["sina_symbol"], keep="first", ignore_index=True)
    df = df.sort_values("sina_symbol", ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"[股票列表] 合计 {len(df)} 只 -> {cache}")
    return df


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=cfg.data["start_date"])
    ap.add_argument("--end", default=cfg.data["end_date"])
    ap.add_argument("--limit", type=int, default=0, help="本次最多下载多少只（0=不限）")
    ap.add_argument("--refresh-symbols", action="store_true", help="强制重建股票列表")
    args = ap.parse_args()

    out_dir = cfg.path("data.raw_symbol_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    src = cfg.source
    ds = SinaDaily(
        rate_limit_per_min=src["rate_limit_per_min"],
        max_retries=src["max_retries"],
        retry_backoff=src["retry_backoff"],
        throttle_cooldown=src.get("throttle_cooldown", 330.0),
        max_throttles=src.get("max_throttles", 4),
    )

    symbols = build_symbol_list(
        ds, args.start, cfg.path("data.symbol_list_path"), refresh=args.refresh_symbols
    )

    todo = [
        (row.sina_symbol, row.code)
        for row in symbols.itertuples()
        if not (out_dir / f"{row.sina_symbol}.parquet").exists()
    ]
    done = len(symbols) - len(todo)
    print(f"\n待下载 {len(todo)} 只，已完成 {done} 只")
    if not todo:
        print("已全部下载完成。下一步：python scripts/02_build_qlib_data.py")
        return 0
    if args.limit:
        todo = todo[: args.limit]
        print(f"[限制] 本次只跑前 {len(todo)} 只")

    t0 = time.time()
    n_ok = n_empty = n_rows = 0
    failed: list[str] = []
    for sym, code in tqdm(todo, desc="下载", unit="只"):
        try:
            df = ds.fetch_one_symbol(sym, start_date=args.start, end_date=args.end)
        except Exception as exc:  # noqa: BLE001  单只失败不该中断整轮
            failed.append(f"{sym}: {exc}")
            continue
        if df.empty:
            n_empty += 1
            df = pd.DataFrame(columns=SYMBOL_COLUMNS)
        else:
            n_ok += 1
            n_rows += len(df)
        df.to_parquet(out_dir / f"{sym}.parquet", index=False)

    print(f"\n完成：有数据 {n_ok} 只、空 {n_empty} 只、共 {n_rows:,} 行，"
          f"用时 {(time.time() - t0) / 60:.1f} 分钟，请求 {ds.call_count} 次")
    if ds.throttle_count:
        print(f"[限流] 撞到 HTTP 456 共 {ds.throttle_count} 次（每次冷却 "
              f"{ds.throttle_cooldown:.0f}s）。若频繁出现，调低 "
              f"configs/data.yaml 的 source.rate_limit_per_min")
    if failed:
        print(f"[失败] {len(failed)} 只（重跑本脚本会自动重试这些）：")
        for line in failed[:10]:
            print(f"       {line}")
        if len(failed) > 10:
            print(f"       ... 另有 {len(failed) - 10} 只")

    remaining = sum(
        1 for s in symbols["sina_symbol"] if not (out_dir / f"{s}.parquet").exists()
    )
    if remaining:
        print(f"[未完成] 还剩 {remaining} 只，重跑本脚本继续")
    else:
        print("下一步：python scripts/02_build_qlib_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
