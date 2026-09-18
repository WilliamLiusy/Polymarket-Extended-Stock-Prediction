#!/usr/bin/env python
"""原始日线 -> 复权面板 -> qlib 二进制数据 + 股票池文件。

流程：
    每股未复权 parquet → 合并 → 乘复权因子 → 每股复权 parquet → dump_bin → qlib_cn/
                                          ↘ 动态股票池 → instruments/top{N}.txt

用法：
    python scripts/02_build_qlib_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tqdm import tqdm

from pmsp.build.adjust import adjust_panel_factor
from pmsp.build.to_qlib import dump_to_qlib, write_per_symbol
from pmsp.build.universe import build_universe, to_qlib_instruments, write_instruments_file
from pmsp.config import load_config


def load_raw_panel(raw_symbol_dir: Path) -> pd.DataFrame:
    """把每股一个的未复权 parquet 合并成一张面板。

    空文件（代码不存在／全历史都在数据起点之前）直接跳过——下载脚本是故意
    落空文件的，为的是重跑时不再探这些死代码。
    """
    files = sorted(raw_symbol_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"{raw_symbol_dir} 下没有数据，请先跑 scripts/01_download.py")
    frames = []
    for path in tqdm(files, desc="读取每股文件", unit="只"):
        df = pd.read_parquet(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise SystemExit("所有文件都是空的，检查下载是否正常")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    cfg = load_config()
    raw_symbol_dir = cfg.path("data.raw_symbol_dir")
    qlib_dir = cfg.path("data.qlib_dir")

    print("[1/5] 合并原始日线 ...")
    raw = load_raw_panel(raw_symbol_dir)
    print(f"      {len(raw):,} 行，{raw['code'].nunique()} 只股票（含已退市）")

    print("[2/5] 复权（乘新浪的后复权累计因子）...")
    panel = adjust_panel_factor(raw)
    panel_path = cfg.path("data.panel_path")
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    n_limit = int(panel["limit_hit"].sum())
    print(f"      面板 {len(panel):,} 行 -> {panel_path}")
    print(f"      日期范围 {panel['date'].min():%Y-%m-%d} → {panel['date'].max():%Y-%m-%d}")
    print(f"      涨跌停(近似) {n_limit:,} 条，占 {n_limit / len(panel) * 100:.2f}%")

    print("[3/5] 按股票拆分 ...")
    per_symbol_dir = cfg.path("data.per_symbol_dir")
    n_files = write_per_symbol(panel, per_symbol_dir)
    print(f"      {n_files} 个 parquet -> {per_symbol_dir}")

    print("[4/5] 转 qlib 二进制（dump_bin）...")
    qlib_dir.mkdir(parents=True, exist_ok=True)
    dump_to_qlib(per_symbol_dir, qlib_dir)
    print(f"      -> {qlib_dir}")

    print("[5/5] 生成动态股票池 ...")
    uni = cfg.universe
    for top_n in uni["top_n_list"]:
        membership = build_universe(
            panel,
            top_n=top_n,
            lookback_days=uni["lookback_days"],
            min_listed_days=uni["min_listed_days"],
        )
        segments = to_qlib_instruments(membership)
        name = f"top{top_n}"
        path = write_instruments_file(segments, qlib_dir, name)
        n_periods = membership.groupby("effective_from").size()
        print(f"      {name}: {membership['code'].nunique()} 只股票出现过、"
              f"{len(segments)} 段在池区间、每期均 {n_periods.mean():.0f} 只 -> {path.name}")

    print("\n完成。下一步：python scripts/05_verify.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
