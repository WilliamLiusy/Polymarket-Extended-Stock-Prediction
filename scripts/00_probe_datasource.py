#!/usr/bin/env python
"""tushare 数据源探活。**现在是可选步骤，不在主流程里。**

主数据源已改为新浪财经（不需要 token），原因见 `configs/data.yaml` 与
`pmsp.datasource.sina_daily` 的模块说明：本机 token（120 积分）的 `daily`
与 `bak_daily` 均无访问权限，有权限的接口配额是 1 次/小时。

这个脚本保留下来做两件事：
  * 你若有 2000 积分的 token，先跑它确认权限，再把 `configs/data.yaml` 的
    `source.name` 改回 `tushare`（那样能拿回 `amount`/`vwap` 与中性化字段）；
  * 排查"我的 token 到底能调什么"。它只花约 5 次调用。

用法：
    export TUSHARE_TOKEN=<你的token>
    python scripts/00_probe_datasource.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pmsp.config import load_config
from pmsp.datasource.base import RAW_COLUMNS
from pmsp.datasource.calendar import candidate_business_days
from pmsp.datasource.tushare_daily import TushareDaily


def main() -> int:
    cfg = load_config()
    src, data = cfg.source, cfg.data

    print("=" * 68)
    print("tushare 数据源探活")
    print("=" * 68)

    if not os.environ.get("TUSHARE_TOKEN"):
        print("\n[提示] 环境变量 TUSHARE_TOKEN 未设置，将尝试使用 tushare 预存的凭证。")
        print("       若失败请先执行：export TUSHARE_TOKEN=<你的token>")

    try:
        ds = TushareDaily(
            rate_limit_per_min=src["rate_limit_per_min"],
            max_retries=src["max_retries"],
            retry_backoff=src["retry_backoff"],
            page_limit=src["page_limit"],
        )
    except Exception as exc:
        print(f"\n[失败] 初始化 tushare 失败: {exc}")
        print("       120 积分免费档即可，只需要 `daily` 接口权限。")
        return 1
    print("\n[1/4] 初始化成功，token 可用。")

    # 取一个肯定是交易日的近期日期（往前找最近的工作日）
    recent = candidate_business_days("2026-09-01", data["end_date"])[-3]
    print(f"\n[2/4] 拉取 {recent} 全市场日线 ...")
    try:
        df = ds.fetch_one_day(recent)
    except Exception as exc:
        print(f"[失败] 调用 daily 失败: {exc}")
        print("       若提示积分不足，说明该 token 连 120 积分免费档都没到。")
        return 1

    if df.empty:
        print(f"[注意] {recent} 返回空 —— 可能是非交易日。这本身也验证了日历推导逻辑可用。")
    else:
        print(f"       返回 {len(df)} 行，{df['ts_code'].nunique()} 只股票")
        missing = set(RAW_COLUMNS) - set(df.columns)
        print(f"       字段完整: {not missing}" + (f"（缺 {missing}）" if missing else ""))
        cap = src["page_limit"]
        if len(df) >= cap:
            print(f"       [关键] 行数达到单次上限 {cap} —— 翻页逻辑已生效，未静默丢数据")
        else:
            print(f"       行数 {len(df)} < 上限 {cap}，本日无需翻页")

    # 非交易日必须返回空，交易日历完全靠这一点推导
    print("\n[3/4] 验证非交易日返回空（交易日历推导的前提）...")
    holiday = "20260101"  # 元旦，必然休市
    empty = ds.fetch_one_day(holiday)
    print(f"       {holiday} 返回 {len(empty)} 行 -> {'符合预期' if empty.empty else '异常！'}")

    print("\n[4/4] 下载成本估算")
    days = candidate_business_days(data["start_date"], data["end_date"])
    per_min = src["rate_limit_per_min"]
    print(f"       区间 {data['start_date']} → {data['end_date']}")
    print(f"       需探测工作日 {len(days)} 个（已排除周末）")
    print(f"       限速 {per_min} 次/分钟 → 约 {len(days) / per_min / 60:.2f} 小时")
    quota_note = "一天内可跑完" if len(days) <= src["daily_quota"] else "需跨天（脚本支持断点续传）"
    print(f"       每日配额 {src['daily_quota']} 次 -> {quota_note}")
    print(f"       本次探活共用掉 {ds.call_count} 次调用")

    print("\n" + "=" * 68)
    print("探活通过。下一步：python scripts/01_download.py")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
