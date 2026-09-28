"""qlib 官方 A 股数据包（GitHub Releases）。**只能当参照集，不能当主源。**

## 结论先写在这儿

`to_qlib.py` 的模块注释原来写「官方数据包挂在 GitHub Releases，本机 github.com
不通，所以自己灌」。2026-09-28 复测：**github.com 的 HTTPS 通了**（约 100 KB/s，
240 MB 包下完约 40 分钟）。但把包拆开看完之后，结论没变——**还是得自己灌**，
理由从"下不下来"换成了"内容不够"：

| | 官方包 v3 | 我们自己灌的 `data/qlib_cn` |
|---|---|---|
| 日历 | 1999-11-10 → **2022-12-30** | 2010-01-04 → 2026-09-18 |
| 字段 | open high low close volume factor **change** | open high low close volume factor |
| `amount` / `vwap` | **没有** | 没有（同样缺） |
| instruments | all(4368) / csi300 / csi500 / csi100 | all / top1500 / top300 |
| csi300 成分止于 | **2020-09-25** | —— |
| csi500 成分止于 | **2022-03-21** | —— |
| 数据来源 | Yahoo Finance（官方 README 自己提示质量有限） | 新浪 K 线 + 官方后复权因子 |

也就是说：Polymarket 对比窗口（2025-01 → 2026-09）官方包**完全没有覆盖**，
它连 baseline 的样本外区间（2018→2026）都只覆盖到一半。补不了任何一条
「已知局限」——缺的 `amount`/`vwap`/市值/行业，它也一样没有。

## 那留着它干什么

三件它能做、而且只有它能做的事：

1. **复权交叉校验（弱参照）**。它是**离线**的，一次下载就能对 2010–2022 段全市场
   逐点比对，不受限流约束，口径也独立（Yahoo，不是新浪）。但实测它自己错得不少：
   抽 30 只，收益率相关中位 0.9898 却有一只低到 0.497，且 2020-09-28 这天
   9/30 只出现 >50% 的假跳变（数据拼接断点）。把三处最差的拿 baostock 当第三方
   仲裁，baostock 与**我们**相关 1.000000——错的是官方包（例：SZ002455 2018-03-08
   官方 +177.3%，我们和 baostock 都是 −0.34%）。
   所以：**主校验用 baostock**（`scripts/07_fetch_baostock.py --what crosscheck`），
   官方包只用来做离线全市场粗筛，它与我们不一致时不能默认是我们错。
2. **`instruments/csi300.txt` 的格式与量级参照**。我们自己的 `hs300.txt`
   （`pmsp.build.index_universe` 从 baostock 生成）可以和它在 2010–2020 重叠段
   逐日比成分重合率——这是唯一能验证"我们的指数成分对不对"的外部依据。
3. **`sh000300` / `sh000905` 指数本身的日线**，2005→2022。回测基准目前用全市场
   等权，想换成真沪深300 基准时这是免费来源（2022 之后的缺口需另补）。

## .bin 格式（自己读，不初始化 qlib）

我们进程里通常已经 `qlib.init` 到自己的 `data/qlib_cn` 了，再 init 到官方目录会
互相踩。所以这里直接读二进制，格式就是 `dump_bin.py` 写出来的那个：

    float32[0]   = 该股票首个数据点在 calendars/day.txt 里的**行号**
    float32[1:]  = 连续的值，缺失日期为 NaN（dump_bin 会把停牌日填 NaN）

`read_feature()` 按这个约定还原成以日期为索引的 Series，并用我们自己的
`data/qlib_cn` 做过一致性检验（见 `scripts/06_setup_qlib_official.py`）。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

#: Releases 根地址。qlib 0.9.7 的 `qlib.tests.data.GetData` 用的就是这个仓库。
RELEASE_BASE = "https://github.com/SunsetWolf/qlib_dataset/releases/download"

#: 实测可用的资产（2026-09-28）。`simple` 版只有 csi300 成分股，体积小，
#: 做格式验证够用；全量版才有 4368 只与退市股。
ASSETS = {
    "cn_1d": "v3/qlib_data_cn_1d_latest.zip",          # 240.7 MB，4368 只，止于 2022-12-30
    "cn_1d_simple": "v2/qlib_data_simple_cn_1d_latest.zip",  # 51.8 MB，csi300 子集
    "us_1d": "v3/qlib_data_us_1d_latest.zip",          # 450.1 MB，美股（美股对照组备用）
}


def download(asset: str, dest_dir: str | Path, chunk_mb: int = 1) -> Path:
    """下载数据包到 `dest_dir`，返回 zip 路径。**支持断点续传**。

    本机到 github.com 约 100 KB/s，240 MB 要 40 分钟，中途断掉重下一遍很亏，
    所以用 `.part` 临时文件 + `Range` 续传。已下载完成的 zip 直接复用。
    """
    if asset not in ASSETS:
        raise ValueError(f"未知资产 {asset}，可选：{list(ASSETS)}")
    url = f"{RELEASE_BASE}/{ASSETS[asset]}"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / Path(ASSETS[asset]).name
    part = out.with_suffix(out.suffix + ".part")

    if out.exists() and zipfile.is_zipfile(out):
        print(f"[官方包] 复用已下载的 {out.name}（{out.stat().st_size / 1e6:.1f} MB）")
        return out

    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, timeout=120, headers=headers) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) + have
        mode = "ab" if have and resp.status_code == 206 else "wb"
        if mode == "wb":
            have = 0  # 服务端不支持续传，只能从头来
        print(f"[官方包] 下载 {url}\n         已有 {have / 1e6:.1f} MB / 共 {total / 1e6:.1f} MB")
        with open(part, mode) as fh:
            for chunk in resp.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                fh.write(chunk)
    if not zipfile.is_zipfile(part):
        raise RuntimeError(f"下载完成但不是合法 zip：{part}（可能被截断，删掉重试）")
    part.rename(out)
    print(f"[官方包] 完成 {out}（{out.stat().st_size / 1e6:.1f} MB）")
    return out


def extract(zip_path: str | Path, target_dir: str | Path) -> Path:
    """解包成 qlib 目录（`calendars/` `instruments/` `features/`）。

    已解包过就跳过——判据是 `calendars/day.txt` 存在，而不是目录存在
    （半途中断留下的空目录会骗过后者）。
    """
    target_dir = Path(target_dir)
    if (target_dir / "calendars" / "day.txt").exists():
        print(f"[官方包] {target_dir} 已解包，跳过")
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    print(f"[官方包] 解包到 {target_dir}")
    return target_dir


def read_calendar(qlib_dir: str | Path) -> pd.DatetimeIndex:
    """读 `calendars/day.txt`。"""
    txt = (Path(qlib_dir) / "calendars" / "day.txt").read_text(encoding="utf-8")
    return pd.DatetimeIndex(pd.to_datetime(txt.split()))


def read_feature(
    qlib_dir: str | Path, code: str, field: str, calendar: pd.DatetimeIndex | None = None
) -> pd.Series:
    """读单只股票的单个字段，返回以交易日为索引的 Series（缺失日期不出现）。

    Parameters
    ----------
    code : str
        qlib 风格代码（`SH600000`，大小写都认）。
    field : str
        `close` / `factor` / `volume` / ...（不带 `$`）。
    """
    qlib_dir = Path(qlib_dir)
    cal = read_calendar(qlib_dir) if calendar is None else calendar
    path = qlib_dir / "features" / code.strip().lower() / f"{field}.day.bin"
    if not path.exists():
        return pd.Series(dtype="float64")
    arr = np.fromfile(path, dtype="<f4")
    if arr.size < 2:
        return pd.Series(dtype="float64")
    start = int(arr[0])
    values = arr[1:].astype("float64")
    idx = cal[start : start + len(values)]
    s = pd.Series(values, index=idx, name=field)
    return s.dropna()


def read_instruments(qlib_dir: str | Path, name: str) -> pd.DataFrame:
    """读 `instruments/{name}.txt`，返回列 `code` / `start` / `end`。"""
    path = Path(qlib_dir) / "instruments" / f"{name}.txt"
    df = pd.read_csv(path, sep="\t", header=None, names=["code", "start", "end"])
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def inspect_dump(qlib_dir: str | Path) -> dict:
    """体检一个 qlib 数据目录：日历范围、字段、instruments 覆盖。

    Returns
    -------
    dict
        `calendar`（n/first/last）、`n_instruments`、`fields`、
        `instrument_files`（每个文件的行数/去重代码数/成分起止）。
    """
    qlib_dir = Path(qlib_dir)
    cal = read_calendar(qlib_dir)
    feat_dir = qlib_dir / "features"
    inst_dirs = sorted(p for p in feat_dir.iterdir() if p.is_dir()) if feat_dir.exists() else []
    fields: set[str] = set()
    for p in inst_dirs[:50]:  # 抽样 50 只即可，全扫 4368 个目录纯属浪费
        fields |= {f.name.replace(".day.bin", "") for f in p.glob("*.day.bin")}

    files = {}
    inst_root = qlib_dir / "instruments"
    if inst_root.exists():
        for f in sorted(inst_root.glob("*.txt")):
            df = read_instruments(qlib_dir, f.stem)
            files[f.stem] = {
                "rows": int(len(df)),
                "codes": int(df["code"].nunique()),
                "first_start": df["start"].min(),
                "last_end": df["end"].max(),
            }
    return {
        "dir": str(qlib_dir),
        "calendar": {"n": len(cal), "first": cal[0], "last": cal[-1]},
        "n_instruments": len(inst_dirs),
        "fields": sorted(fields),
        "instrument_files": files,
    }
