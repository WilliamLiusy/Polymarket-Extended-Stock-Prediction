"""面板 -> qlib 二进制格式。

qlib 官方 A 股数据包挂在 GitHub Releases，本机 github.com 不通，所以自己灌。
自己灌反而更好：能覆盖到 2026-09、能自己控制退市股与复权口径，官方快照给不了。

转换靠官方 `scripts/dump_bin.py`（已放在 `third_party/`，wheel 里不含此文件）。
它的三条约定必须对齐，错一条就查半天：

1. **symbol 取自文件名**（`get_symbol_from_file`），不是某一列
2. `instruments/*.txt` 里的 code 是**大写**（`SH600000`）
3. `features/` 下的目录名是**小写**（`sh600000/`）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# 喂给 qlib 的字段。factor 留着是为了以后需要还原不复权价时不用重算。
#
# **没有 $vwap**：主数据源（新浪）不提供成交额，算不出真实成交均价。
# 拿 (H+L+C)/3 之类的"典型价"冒充是错的——那是 HLC 的线性组合，
# 与 KMID/KLEN/KSFT 高度重复，等于凭空造一个假因子。所以老老实实去掉，
# Alpha158 的 `VWAP0` 因子随之消失，因子数 216 → 215。
# 一致性要求：`pmsp.model.dataset.PRICE_FEATURES` 必须与这里同步。
# 以后若买了 tushare 2000 积分（有 amount），两处一起加回 vwap 即可。
QLIB_FIELDS = ["open", "high", "low", "close", "volume", "factor"]


def write_per_symbol(panel: pd.DataFrame, out_dir: str | Path) -> int:
    """按股票拆分成一堆 parquet，文件名 = 小写 code。返回写出的文件数。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()  # 全量重灌，残留旧文件会被 dump_bin 当成还在市的股票

    cols = ["date", *QLIB_FIELDS]
    n = 0
    for code, grp in panel.groupby("code", sort=True):
        out = grp[cols].sort_values("date", ignore_index=True)
        out = out.dropna(subset=["close"])
        if out.empty:
            continue
        out.to_parquet(out_dir / f"{code.lower()}.parquet", index=False)
        n += 1
    return n


def dump_to_qlib(
    per_symbol_dir: str | Path,
    qlib_dir: str | Path,
    max_workers: int = 16,
) -> None:
    """调用官方 dump_bin 生成二进制数据、日历和 `instruments/all.txt`。

    注意这会写 `instruments/all.txt`（全市场）。我们自己的股票池文件
    （`top1500.txt` / `top300.txt`）由 `universe.write_instruments_file` 另写，
    互不覆盖。
    """
    third_party = Path(__file__).resolve().parents[3] / "third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from dump_bin import DumpDataAll  # noqa: PLC0415  第三方脚本，延迟导入

    DumpDataAll(
        data_path=str(per_symbol_dir),
        qlib_dir=str(qlib_dir),
        freq="day",
        max_workers=max_workers,
        date_field_name="date",
        file_suffix=".parquet",
        symbol_field_name="code",
        include_fields=",".join(QLIB_FIELDS),
    ).dump()
