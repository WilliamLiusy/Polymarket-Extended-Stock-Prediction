"""交易日历。

免费档没有 `trade_cal` 接口，所以日历不是"查"出来的，是"推"出来的：
遍历所有工作日去调 `daily`，**空响应即非交易日**。这样自成体系、无额外依赖，
代价是每年多约 10 次无效调用（法定节假日），完全可接受。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def candidate_business_days(start_date: str, end_date: str) -> list[str]:
    """待探测的工作日列表（`YYYYMMDD`），已排除周末。

    周末一定不开市，跳过它们能把调用次数从自然日的 ~6100 降到 ~4360。
    """
    days = pd.bdate_range(start=start_date, end=end_date, freq="C", weekmask="Mon Tue Wed Thu Fri")
    return [d.strftime("%Y%m%d") for d in days]


def derive_calendar(raw_day_dir: str | Path) -> list[str]:
    """从下载缓存反推交易日历。

    下载器给每个探测过的工作日都落一个 parquet；非交易日落的是空表。
    所以"非空的文件名"就是交易日。
    """
    raw_day_dir = Path(raw_day_dir)
    trade_dates = []
    for path in sorted(raw_day_dir.glob("*.parquet")):
        # 空文件（非交易日）只有表头，parquet 元数据里就能读到行数，不必读全表
        import pyarrow.parquet as pq

        if pq.ParquetFile(path).metadata.num_rows > 0:
            trade_dates.append(path.stem)
    return trade_dates


def write_calendar(trade_dates: list[str], qlib_dir: str | Path, freq: str = "day") -> Path:
    """写 qlib 的 `calendars/{freq}.txt`（每行一个 `YYYY-MM-DD`）。"""
    out_dir = Path(qlib_dir) / "calendars"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{freq}.txt"
    formatted = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sorted(trade_dates)]
    out_path.write_text("\n".join(formatted) + "\n", encoding="utf-8")
    return out_path
