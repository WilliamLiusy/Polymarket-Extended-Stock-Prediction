"""Polymarket 成交数据源：HuggingFace 镜像上的 `TimeSeventeen/Polymarket-v1`。

## 为什么不直连 Polymarket 官方 API

本机网络实测（2026-09-24）：

    gamma-api.polymarket.com   系统 DNS 返回 2a03:2880::/32（Meta 的段）与
                               103.73.161.52 —— 都是投毒结果，不是真实地址
    真实地址（doh.pub 查得）    104.18.34.205 / 172.64.153.51（Cloudflare）
    用真实 IP + SNI 直连        TLS 握手阶段 **Connection reset by peer**
    WebFetch                   同样被拦（"Unable to verify if domain ... is safe"）

也就是说官方 CLOB / Gamma 接口在本机彻底不可达，**换 IP 没用，是 SNI 阻断**。
对照组：api.github.com 200/0.5s、hf-mirror.com 200/1.6s，所以不是没有出网。

结论：历史数据只能走**第三方镜像**。hf-mirror.com 实测单流 5.6 MB/s、支持
Range 请求（parquet 列裁剪可行），是唯一稳定的路。

## 为什么选 `daily_aligned/` 这一层

该数据集有四层，选 `daily_aligned/`（标准二元市场，`neg_risk=false`）与
`daily_aligned_multi/`（多结果市场，`neg_risk=true`）：

* `OrderFilled/` 是**名义**链上成交带，含平台 relayer/router 记录。直接拿它算
  成交额会虚高——数据集作者在 README 里专门警告过这一点。
* `daily_aligned/` 已经剔除 relayer、join 了市场元数据（关键是 `market_slug`，
  否则根本不知道哪个 `condition_id` 是油价市场）、并加了 `p_event`。

`p_event` 这一列是必须用的：`price` 是**被交易的那个 token** 的价格，同一个市场
里 Yes 腿成交在 0.30、No 腿成交在 0.70 是同一个信息，混在一起算均价会得到噪声。
`p_event` 统一折算到 `outcome_seq=1` 这条参考轴上（Yes 腿 `p_event=price`，
No 腿 `p_event=1-price`），才是"该事件发生的概率"。

## 这不是盘口数据，必须说清楚

数据集 README 原文：*"None of the layers include order-book snapshots, quote
updates, cancellations, or off-chain resting-order depth."* 所以本模块拿到的是
**成交带（trade tape）**，不是 bid/ask 挂单深度。

真盘口（`order_book_depth`）在另一个数据集 `trentmkelly/polymarket_historical_data`
里，但只覆盖 2026-07-23 起的 64 天，做不了日频关联分析（见
`pmsp.datasource.hf_polymarket_book`）。

因此本项目口径下的"盘口"指**由成交带构造的微观结构量**：成交额、笔数、
VWAP 概率、日内实现波动、以及 taker 方向净额（OFI）。OFI 用 `D` 列算——
`D=+1` 表示主动方在买入参考事件概率、`-1` 表示卖出，这是数据集已经归一化好
的"攻击方方向"，比自己从 `taker_direction` + `outcome_seq` 反推更不容易出错。

## 为什么聚合到**小时**而不是直接到天

数据集按 **UTC+8** 切日文件。若直接按文件聚合成"一天一个数"，再去跟 A 股
当日收益做相关，就用到了 A 股 15:00 收盘**之后**（北京时间 15:00–24:00）
才发生的 Polymarket 成交——那是前视偏差。

所以本层统一聚合到 **UTC 小时**，把"按哪个时钟截断"这件事留给下游：
A 股口径截到 07:00 UTC（= 北京 15:00 收盘），美股口径截到 20:00 UTC
（= 美东 16:00 收盘）。换口径不需要重新下载 13 GB。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

#: HF 镜像。官方 huggingface.co 在本机不可达（connect timeout），只能走镜像。
HF_MIRROR = "https://hf-mirror.com"

DATASET = "TimeSeventeen/Polymarket-v1"

#: 两层成交数据。二元市场与多结果市场**必须分开标记**再合并：
#: `neg_risk=true` 的市场里同一事件有多个候选 `condition_id`，`p_event` 不求和为 1。
TRADE_SUBDIRS = ("daily_aligned", "daily_aligned_multi")

#: 只读这些列。全 24 列里 `maker`/`taker`/`asset_id` 占 39% 的字节而本层分析
#: 用不到（钱包层面的行为研究是另一个课题），裁掉后网络量降到约 60%。
READ_COLUMNS = [
    "block_timestamp",
    "condition_id",
    "market_slug",
    "category",
    "category_refined",
    "neg_risk",
    "outcome_seq",
    "price",
    "p_event",
    "usdc_amount",
    "D",
    "close_at",
    "resolution_status",
    "winning_outcome_label",
]

#: `daily_aligned_multi/` 比 `daily_aligned/` 多这一列，`daily_aligned/` 里不存在，
#: 所以列清单必须按层区分——读不存在的列 pyarrow 直接抛，不会静默补 NaN。
#: 这一列很关键：油价市场里"2026 年油价触及多少"这种分桶市场就是多结果市场，
#: 各个价格桶是独立的 condition_id，只有它能把同一事件的桶归到一起。
MULTI_EXTRA_COLUMNS = ["neg_risk_market_id"]

_UA = "Mozilla/5.0 (X11; Linux x86_64) pmsp-research/0.1"


def _columns_for(subdir: str) -> list[str]:
    if subdir.endswith("_multi"):
        return READ_COLUMNS + MULTI_EXTRA_COLUMNS
    return list(READ_COLUMNS)


# --------------------------------------------------------------------- 文件清单


def list_trade_files(subdir: str = "daily_aligned", timeout: float = 60.0) -> pd.DataFrame:
    """列出某一层的全部日文件（含字节数）。

    HF 的 tree 接口分页返回，且 `Link: rel="next"` 里给的是 **huggingface.co**
    的绝对地址——本机不可达，必须把 host 换成镜像，否则第 1001 个文件起就拉不到。
    """
    session = requests.Session()
    session.headers.update({"User-Agent": _UA})
    url = f"{HF_MIRROR}/api/datasets/{DATASET}/tree/main/{subdir}?recursive=true"
    rows: list[dict] = []
    while url:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        rows.extend(resp.json())
        link = resp.headers.get("Link", "")
        url = ""
        if 'rel="next"' in link:
            url = link.split("<", 1)[1].split(">", 1)[0].replace(
                "huggingface.co", "hf-mirror.com"
            )
    df = pd.DataFrame([{"path": r["path"], "size": r.get("size", 0)} for r in rows])
    df = df[df.path.str.endswith(".parquet")].copy()
    df["date"] = pd.to_datetime(
        df.path.str.extract(r"(\d{4}_\d{2}_\d{2})\.parquet$")[0], format="%Y_%m_%d"
    )
    df["subdir"] = subdir
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------- 下载


def download(path: str, dest: Path, max_retries: int = 5, timeout: float = 120.0) -> Path:
    """下载单个文件到 `dest`，已存在且非空则跳过（断点续传靠整文件重下）。

    不做 HTTP Range 续传：单文件最大约 140 MB、5.6 MB/s 下 25 秒就下完，
    断点续传的复杂度换不回什么；重下更不容易留下半截文件。半截 parquet 的
    footer 缺失，pyarrow 会在**读取时**才报错，那时候已经很难定位是哪一步坏的，
    所以这里下完先落临时名，校验能打开 footer 再改名。
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_MIRROR}/datasets/{DATASET}/resolve/main/{path}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with requests.get(
                url, stream=True, timeout=timeout, headers={"User-Agent": _UA}
            ) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            pq.ParquetFile(tmp).metadata  # footer 校验：截断的文件在这里就炸
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - 网络层什么都可能抛
            last_err = exc
            tmp.unlink(missing_ok=True)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"下载失败 {path}: {last_err}")


def download_many(
    files: pd.DataFrame, cache_dir: Path, workers: int = 16, log_every: int = 25
) -> pd.DataFrame:
    """并发下载一批文件。返回带本地路径的清单。

    `workers=16`：hf-mirror 是**按连接**限速的，不是总带宽瓶颈。实测（同一批
    2025-11 之后的大文件，同一台机器、间隔几分钟）：

        4 并发    1.06 MB/s
        16 并发  10.92 MB/s      ← 约 10 倍

    一开始按"总带宽 6 MB/s 是瓶颈"设了 4 并发，结果 16.8 GB 要跑 4 小时。
    单连接被压到约 0.3 MB/s，加并发是线性叠加的。没有再往上试是因为 10 MB/s
    下全量只要半小时，而并发数越高越可能撞上镜像端的连接数限制——这个收益
    已经够了，不值得拿被封 IP 去换。

    幂等：已存在且非空的文件直接跳过，所以中断后原地重跑即可续传，也可以
    先用小并发起跑、再换大并发重启（本项目就是这么干的）。
    """
    cache_dir = Path(cache_dir)
    files = files.copy()
    files["local"] = [str(cache_dir / p) for p in files.path]
    todo = [
        (p, Path(l))
        for p, l in zip(files.path, files.local)
        if not (Path(l).exists() and Path(l).stat().st_size > 0)
    ]
    if not todo:
        return files
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(download, p, l): p for p, l in todo}
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % log_every == 0 or done == len(todo):
                mb = sum(
                    Path(l).stat().st_size for _, l in todo if Path(l).exists()
                ) / 1e6
                print(
                    f"  [{done}/{len(todo)}] {mb:,.0f} MB, "
                    f"{mb / max(1e-9, time.time() - t0):.1f} MB/s",
                    flush=True,
                )
    return files


# ------------------------------------------------------------------- 小时聚合


def _vwap(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float((values * weights).sum() / total)


def aggregate_hourly(path: str | Path, subdir: str = "daily_aligned") -> pd.DataFrame:
    """把一个日文件的逐笔成交聚合成 (UTC 小时 × condition_id) 的面板。

    产出的量与它们为什么这样定义：

    ``p_vwap``   成交额加权的事件概率。不用算术均值：Polymarket 上一笔 5 美元
                 的成交和一笔 50 万美元的成交在算术均值里同权，那是噪声主导。
    ``p_open`` / ``p_close``
                 小时内首笔 / 末笔的 `p_event`。跨小时算 Δp 时用 `p_close`
                 的差，比用 `p_vwap` 的差干净——后者混进了小时内成交时点的分布。
    ``p_high`` / ``p_low``
                 用于日内实现波动（Parkinson 型），概率区间天然有界，
                 比收益率波动更稳。
    ``vol_usdc`` 成交额（USDC.e）。这是流动性/关注度的主度量。
    ``ofi_usdc`` **订单流不平衡** = Σ D·usdc_amount。`D=+1` 为主动买入参考事件
                 概率。它带方向，是本项目里最接近"盘口压力"的量——盘口深度拿
                 不到，但主动方向净额拿得到。
    ``n_trades`` 成交笔数。与 `vol_usdc` 一起能分出"一笔大单"和"很多人在交易"。
    """
    tbl = pq.ParquetFile(path).read(columns=_columns_for(subdir))
    df = tbl.to_pandas()
    if df.empty:
        return pd.DataFrame()

    # block_timestamp 是**秒**（数据集 README 专门强调过，别当毫秒用）
    ts = pd.to_datetime(df.block_timestamp, unit="s", utc=True)
    df["hour"] = ts.dt.floor("h")
    df = df.sort_values("block_timestamp")

    w = df.usdc_amount.to_numpy(dtype=float)
    df["_pw"] = df.p_event.to_numpy(dtype=float) * w
    df["_dw"] = df.D.to_numpy(dtype=float) * w

    grp = df.groupby(["hour", "condition_id"], sort=False)
    out = grp.agg(
        n_trades=("usdc_amount", "size"),
        vol_usdc=("usdc_amount", "sum"),
        _pw=("_pw", "sum"),
        ofi_usdc=("_dw", "sum"),
        p_open=("p_event", "first"),
        p_close=("p_event", "last"),
        p_high=("p_event", "max"),
        p_low=("p_event", "min"),
    ).reset_index()
    out["p_vwap"] = out._pw / out.vol_usdc.where(out.vol_usdc > 0)
    out = out.drop(columns=["_pw"])
    return out


def market_catalog(path: str | Path, subdir: str = "daily_aligned") -> pd.DataFrame:
    """从一个日文件抽市场元数据（每个 condition_id 一行）。

    元数据在成交层里是**逐笔冗余存储**的，所以只要取 first 就够；跨日合并时
    再去重。留 `market_slug` 是为了做关键词筛选，留 `close_at`/`resolution_status`
    是为了把"还没到期的市场"和"已结算的市场"分开——已结算市场在临近结算时
    概率会被拉向 0/1，那不是新信息。
    """
    cols = [
        "condition_id",
        "market_slug",
        "category",
        "category_refined",
        "neg_risk",
        "close_at",
        "resolution_status",
        "winning_outcome_label",
    ]
    if subdir.endswith("_multi"):
        cols += MULTI_EXTRA_COLUMNS
    tbl = pq.ParquetFile(path).read(columns=cols)
    df = tbl.to_pandas()
    if df.empty:
        return pd.DataFrame()
    return df.groupby("condition_id", as_index=False).first()


def build_hourly_panel(
    files: pd.DataFrame,
    out_hourly: Path,
    out_catalog: Path,
    workers: int = 4,
    flush_every: int = 120,
) -> tuple[Path, Path]:
    """对已下载的文件批量聚合，写出小时面板与市场目录。

    分批 flush 而不是全量 concat：2026 年的单日文件有 60 万笔以上，
    1248 天全读进内存再 concat 会吃掉十几 GB。
    """
    out_hourly = Path(out_hourly)
    out_catalog = Path(out_catalog)
    out_hourly.parent.mkdir(parents=True, exist_ok=True)

    hourly_parts: list[pd.DataFrame] = []
    catalog_parts: list[pd.DataFrame] = []
    shards: list[pd.DataFrame] = []
    rows = list(zip(files.local, files.subdir))

    def one(item: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        local, subdir = item
        h = aggregate_hourly(local, subdir)
        c = market_catalog(local, subdir)
        if not h.empty:
            h["subdir"] = subdir
        if not c.empty:
            c["subdir"] = subdir
        return h, c

    with ThreadPoolExecutor(workers) as pool:
        for i, (h, c) in enumerate(pool.map(one, rows), start=1):
            if not h.empty:
                hourly_parts.append(h)
            if not c.empty:
                catalog_parts.append(c)
            if i % flush_every == 0:
                shards.append(pd.concat(hourly_parts, ignore_index=True))
                hourly_parts = []
                print(f"  聚合 {i}/{len(rows)}", flush=True)
    if hourly_parts:
        shards.append(pd.concat(hourly_parts, ignore_index=True))

    hourly = pd.concat(shards, ignore_index=True) if shards else pd.DataFrame()
    catalog = (
        pd.concat(catalog_parts, ignore_index=True)
        .sort_values("condition_id")
        .groupby("condition_id", as_index=False)
        .first()
        if catalog_parts
        else pd.DataFrame()
    )
    hourly.to_parquet(out_hourly, index=False)
    catalog.to_parquet(out_catalog, index=False)
    print(f"小时面板 {len(hourly):,} 行 -> {out_hourly}")
    print(f"市场目录 {len(catalog):,} 个市场 -> {out_catalog}")
    return out_hourly, out_catalog


def fetch(
    cache_dir: Path,
    out_hourly: Path,
    out_catalog: Path,
    subdirs: Iterable[str] = TRADE_SUBDIRS,
    start: str | None = None,
    end: str | None = None,
    workers: int = 4,
) -> tuple[Path, Path]:
    """端到端：列清单 -> 下载 -> 聚合。"""
    listings = []
    for sub in subdirs:
        lst = list_trade_files(sub)
        if start:
            lst = lst[lst.date >= pd.Timestamp(start)]
        if end:
            lst = lst[lst.date <= pd.Timestamp(end)]
        print(
            f"{sub}: {len(lst)} 个日文件, {lst['size'].sum() / 1e9:.2f} GB, "
            f"{lst.date.min():%Y-%m-%d} → {lst.date.max():%Y-%m-%d}"
        )
        listings.append(lst)
    files = pd.concat(listings, ignore_index=True)
    files = download_many(files, cache_dir, workers=workers)
    return build_hourly_panel(files, out_hourly, out_catalog, workers=workers)
