"""拉取 Polymarket 成交数据并聚合成 (UTC 小时 × 市场) 面板。

为什么是这个数据源、为什么聚合到小时而不是天，见
`pmsp.datasource.hf_polymarket` 的模块文档。一句话版本：官方 API 在本机被
SNI 阻断（换 IP 无效），只能走 HF 镜像的第三方归档；数据集按 UTC+8 切日文件，
直接按日聚合会把 A 股收盘后的成交混进当日因子，所以留到小时粒度。

    python scripts/10_fetch_polymarket.py                 # 全量，约 16.8 GB / 30 分钟（16 并发）
    python scripts/10_fetch_polymarket.py --start 2025-01-01
    python scripts/10_fetch_polymarket.py --no-download   # 只重跑聚合（原始文件已缓存）

原始日文件默认**保留**在 `data/raw/polymarket/`。16.8 GB 换来的是：以后要改
聚合口径（比如加 maker/taker 的钱包维度）不用再下一遍。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pmsp.datasource import hf_polymarket as hp

RAW_DIR = Path("data/raw/polymarket")
OUT_HOURLY = Path("data/raw/polymarket/hourly_panel.parquet")
OUT_CATALOG = Path("data/raw/polymarket/market_catalog.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="起始日期（含），如 2024-01-01")
    ap.add_argument("--end", default=None, help="结束日期（含）")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="跳过下载，只对已缓存的文件重跑聚合",
    )
    args = ap.parse_args()

    listings = []
    for sub in hp.TRADE_SUBDIRS:
        lst = hp.list_trade_files(sub)
        if args.start:
            lst = lst[lst.date >= pd.Timestamp(args.start)]
        if args.end:
            lst = lst[lst.date <= pd.Timestamp(args.end)]
        print(
            f"{sub}: {len(lst)} 个日文件, {lst['size'].sum() / 1e9:.2f} GB, "
            f"{lst.date.min():%Y-%m-%d} → {lst.date.max():%Y-%m-%d}",
            flush=True,
        )
        listings.append(lst)
    files = pd.concat(listings, ignore_index=True)

    if args.no_download:
        files["local"] = [str(RAW_DIR / p) for p in files.path]
        missing = [l for l in files.local if not Path(l).exists()]
        if missing:
            print(f"警告：{len(missing)} 个文件未缓存，将被跳过")
            files = files[[Path(l).exists() for l in files.local]]
    else:
        files = hp.download_many(files, RAW_DIR, workers=args.workers)

    hp.build_hourly_panel(files, OUT_HOURLY, OUT_CATALOG, workers=args.workers)


if __name__ == "__main__":
    main()
