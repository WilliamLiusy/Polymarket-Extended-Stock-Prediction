"""真实指数成分股历史 → qlib instruments 文件（补 README 已知局限第 3 条）。

## 它解决的问题

`pmsp.build.universe` 的注释里写：「免费档拿不到指数成分股历史，所以用流动性前 N
近似」。baostock 免费给出**按日查询的沪深300 / 中证500 / 上证50 成分**，实测
2010-01-08 → 2026-09-25 每个查询日都返回完整 300 / 500 只，`updateDate` 约每周一次。
所以这条局限可以真正补掉：`top300` 近似可以换成**真沪深300**，不是猜的。

两者都留着，不互相替换：

* `top300` / `top1500`：纯由成交额算出，口径 100% 自洽、完全 point-in-time；
* `hs300` / `zz500`：真实指数成分，可与 qlib 官方基准（用的就是 csi300）直接对话。

README 里 baseline 的数字是 `top1500` / `top300` 跑出来的，**不因为这个文件而变**。
换股票池是一次新实验，不是修 bug。

## point-in-time 的口径（与 `universe.py` 严格一致）

`fetch_index_members(index, date=d)` 返回的是"截至 d 的最近一次成分更新"。所以
在 d 这一天我们能知道的就是这份名单，成分从 **d 的下一个交易日**开始生效，
到下一个快照日（含）为止。查询日取每月最后一个交易日，与 `universe.py` 的
`month_end_rebalance_dates` 对齐——这样 `hs300` 和 `top300` 的调仓时点相同，
两个股票池的结果可以直接比。

用 `updateDate` 当生效日是**错的**：它可能早于查询日几天，但那几天里我们并不知道
这次调整（指数公司公告在前、生效在后，baostock 给的是已生效的名单快照）。
按月采样时这点差异被月末对齐吸收掉，写在这儿是防止以后有人"优化"成 updateDate。

## 一个必须做的交集

指数成分里可能有我们 `data/qlib_cn` 没有的股票（比如新浪那边代码对不上、
或下载失败）。写进 instruments 文件的代码如果在 `features/` 下没有目录，
qlib 取数时会静默给 NaN——整段截面少几只股票很难发现。所以这里强制与本地
`all.txt` 取交集，并把覆盖率打出来：**低于 99% 就该去查为什么**。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .universe import month_end_rebalance_dates, to_qlib_instruments, write_instruments_file

#: 支持的指数及其 qlib instruments 文件名。
INDEX_NAMES = {"hs300": "hs300", "zz500": "zz500", "sz50": "sz50"}


def membership_from_snapshots(
    snapshots: dict[pd.Timestamp, set[str]],
    trade_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """按日快照 -> 逐期成分明细（列 `code` / `effective_from` / `effective_to`）。

    Parameters
    ----------
    snapshots : dict
        `{查询日: {code, ...}}`。查询日必须是交易日。
    trade_dates : pd.DatetimeIndex
        本地交易日历，用来定位"下一个交易日"。

    Returns
    -------
    pd.DataFrame
        与 `universe.build_universe` 输出同构，可直接喂给 `to_qlib_instruments`。
    """
    pos = {d: i for i, d in enumerate(trade_dates)}
    keys = sorted(snapshots)
    records: list[dict] = []
    for i, d in enumerate(keys):
        if d not in pos:
            raise ValueError(f"快照日 {d:%Y-%m-%d} 不在本地交易日历里")
        if pos[d] + 1 >= len(trade_dates):
            break  # 最后一个快照日之后没有可交易日，成分无处生效
        effective_from = trade_dates[pos[d] + 1]
        effective_to = keys[i + 1] if i + 1 < len(keys) else trade_dates[-1]
        records.extend(
            {"code": c, "effective_from": effective_from, "effective_to": effective_to}
            for c in sorted(snapshots[d])
        )
    return pd.DataFrame.from_records(
        records, columns=["code", "effective_from", "effective_to"]
    )


def fetch_index_snapshots(
    ds,
    index: str,
    trade_dates: pd.DatetimeIndex,
    start: str | pd.Timestamp | None = None,
) -> dict[pd.Timestamp, set[str]]:
    """按月末交易日抓一遍成分快照。约 200 次查询，一两分钟。

    `ds` 是已登录的 `BaostockDaily`。空结果（早于指数发布日）会被跳过而不是报错——
    中证500 发布于 2007、上证50 于 2004，沪深300 于 2005，都早于本项目起点 2010，
    但留着这层容错，以后加别的指数不用改代码。
    """
    if index not in INDEX_NAMES:
        raise ValueError(f"未知指数 {index}，支持 {list(INDEX_NAMES)}")
    dates = month_end_rebalance_dates(trade_dates)
    if start is not None:
        start = pd.Timestamp(start)
        dates = [d for d in dates if d >= start]

    snapshots: dict[pd.Timestamp, set[str]] = {}
    for d in dates:
        members = ds.fetch_index_members(index, f"{d:%Y-%m-%d}")
        if members.empty:
            continue
        snapshots[d] = set(members["code"])
    return snapshots


def build_index_instruments(
    snapshots: dict[pd.Timestamp, set[str]],
    trade_dates: pd.DatetimeIndex,
    qlib_dir: str | Path,
    name: str,
    local_codes: set[str] | None = None,
) -> dict:
    """快照 -> instruments 文件。返回一份小体检报告。

    `local_codes` 给定时（通常是本地 `all.txt` 的代码集）强制取交集，
    并在报告里给出覆盖率与被丢掉的代码样例。
    """
    membership = membership_from_snapshots(snapshots, trade_dates)
    n_raw = membership["code"].nunique()
    dropped: list[str] = []
    if local_codes is not None:
        missing = sorted(set(membership["code"]) - local_codes)
        dropped = missing
        membership = membership[membership["code"].isin(local_codes)]

    segments = to_qlib_instruments(membership)
    path = write_instruments_file(segments, qlib_dir, name)
    return {
        "name": name,
        "path": str(path),
        "n_snapshots": len(snapshots),
        "snapshot_first": min(snapshots) if snapshots else None,
        "snapshot_last": max(snapshots) if snapshots else None,
        "n_codes_index": n_raw,
        "n_codes_written": int(membership["code"].nunique()),
        "coverage": (float(membership["code"].nunique()) / n_raw) if n_raw else float("nan"),
        "n_segments": int(len(segments)),
        "dropped_sample": dropped[:10],
        "n_dropped": len(dropped),
    }


def compare_with_reference(
    ours: pd.DataFrame, reference: pd.DataFrame, dates: list[pd.Timestamp]
) -> pd.DataFrame:
    """把我们生成的成分段与外部参照（qlib 官方 `csi300.txt`）逐日比重合率。

    两边都是 `code` / `start` / `end` 的区间表。返回每个检查日的
    `n_ours` / `n_ref` / `n_both` / `jaccard`。

    官方 csi300 止于 2020-09-25 且 2005 年之前的段一律写成 `2005-01-01`，
    所以只有 2010-01 → 2020-09 这段可比。重合率不会是 100%：官方那份是从
    Yahoo/中证公告重建的，本身有误差。**这项是量级对照，不是逐项复现**——
    能发现"我们把中证500 当成沪深300 了"这类错误，发现不了个别股票的出入。
    """
    rows = []
    for d in dates:
        a = set(ours.loc[(ours["start"] <= d) & (ours["end"] >= d), "code"])
        b = set(reference.loc[(reference["start"] <= d) & (reference["end"] >= d), "code"])
        if not a or not b:
            continue
        rows.append(
            {
                "date": d,
                "n_ours": len(a),
                "n_ref": len(b),
                "n_both": len(a & b),
                "jaccard": len(a & b) / len(a | b),
            }
        )
    return pd.DataFrame(rows)
