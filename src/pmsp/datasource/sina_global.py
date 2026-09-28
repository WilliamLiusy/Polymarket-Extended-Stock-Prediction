"""新浪的**美股日线**与**外盘期货日线**（本项目 A 股之外的两块拼图）。

原油—航空这个对比分析需要三样东西，A 股日线仓库里已经有了，另两样在这里：

    美股航空股日线     US_MinKService.getDailyK            AAL / DAL / UAL / LUV ...
    外盘原油日线       GlobalFuturesService...DailyKLine   CL（NYMEX WTI）/ OIL（布伦特）

## 为什么不用别的源

* **stooq.com**：返回 JS 工作量证明挑战页（`crypto.subtle.digest` 循环），
  curl 拿不到 CSV。
* **Yahoo / EIA / FRED**：域名在本机不可达（与 polymarket.com 同样的阻断，
  见 `pmsp.datasource.hf_polymarket` 的实测记录）。
* **新浪**：已经是本项目 A 股的主源，限速行为已经摸清（见 `sina_daily`），
  再加两个接口不引入新的失败模式。

## 一处必须说明的口径损失：美股日线是**未复权**的

`US_MinKService.getDailyK` 返回的是原始收盘价，不含拆股/分红调整。
本分析窗口（2022-11 → 2026-09）内：

* **拆股**：AAL / DAL / UAL / LUV / ALK / JBLU 均无拆股，这一项损失为零。
* **分红**：只有 DAL 与 LUV 有现金分红，股息率约 1%–2.5%/年，单次除息日的
  价格跳空约 −0.3% 到 −0.7%。日频收益序列里这是**每季度一个**约 0.5% 的
  向下噪声点。

对本分析的影响：我们关心的是"原油概率变动 ↔ 航空股当日收益"的相关性，
除息跳空与 Polymarket 上的油价概率**无关**，所以它进入的是分母（噪声方差），
只会让相关系数**向零偏**，不会造出假的相关。也就是说这个偏差方向是保守的，
结论若为正则不受它影响。`fetch_us_daily(drop_ex_div_jumps=True)` 提供了把
疑似除息日剔除的开关，用于稳健性检验。

## 外盘期货是**连续合约**

`symbol=CL` 返回的是新浪拼接的连续主力合约，换月时会有跳空。用于本分析的是
日收益率与"油价水位"两件事：

* **日收益率**：换月跳空会污染换月当日（每月一次），做稳健性检验时按
  `|r| > 8%` 剔除即可（WTI 真实单日波动超过 8% 在样本内极罕见）。
* **油价水位**：跳空幅度相对水位很小（近月-次月价差通常 <2%），不影响水位判断。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_US_DAILY_URL = (
    "https://stock.finance.sina.com.cn/usstock/api/jsonp.php/x/"
    "US_MinKService.getDailyK"
)
_FUT_DAILY_URL = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/x/"
    "GlobalFuturesService.getGlobalFuturesDailyKLine"
)

#: 美股航空股。选这六只的理由：
#: 三大网络型（AAL / DAL / UAL）+ 最大低成本（LUV）+ 两只中小型（ALK / JBLU）。
#: 网络型有燃油对冲与燃油附加费转嫁能力，低成本型对油价更裸——这个横截面差异
#: 本身就是可检验的假设：若 Polymarket 油价信号真有用，对 LUV/JBLU 的
#: 敏感度应当高于 DAL/UAL。
US_AIRLINES = {
    "AAL": "American Airlines",
    "DAL": "Delta Air Lines",
    "UAL": "United Airlines",
    "LUV": "Southwest Airlines",
    "ALK": "Alaska Air Group",
    "JBLU": "JetBlue Airways",
}

#: A 股航空股（代码用本仓库的 qlib 风格）。600221 海航控股在样本期内经历过
#: 重整，601156 东航物流是货运（对油价敏感但需求端逻辑不同），两只都留着
#: 但在报告里单独标注。
CN_AIRLINES = {
    "SH601111": "中国国航",
    "SH600029": "南方航空",
    "SH600115": "中国东航",
    "SH601021": "春秋航空",
    "SH603885": "吉祥航空",
    "SZ002928": "华夏航空",
    "SH600221": "海航控股",
    "SH601156": "东航物流",
}

#: 美股对照组 —— 这三个是本分析里最有判别力的东西。
#:
#: `JETS` 航空 ETF：现成的美股航空指数，不必自己合成，也避开了"选哪几只、
#:        怎么加权"的自由度。
#: `XLE`  能源 ETF：油价冲击的**反向**受益方。真实的油价冲击有一个教科书式的
#:        特征——能源涨、航空跌。如果 Polymarket 的油价风险概率上升时
#:        XLE 与 JETS 只是**同涨同跌**，那说明它测到的是大盘风险偏好，
#:        不是油价；只有在两者**分化**时，才能说抓到了油价渠道。
#:        这一条比任何单边显著性都更难被偶然满足，所以它是主要的伪发现防线。
#: `SPY`  大盘：控制变量。航空股与大盘的相关系数在 0.6 以上，不控制它，
#:        "地缘风险概率 ↔ 航空股跌" 很可能只是 "地缘风险 ↔ 大盘跌" 的影子。
US_CONTROLS = {"JETS": "美国航空业 ETF", "XLE": "能源 ETF", "SPY": "标普500 ETF"}

#: A 股这一侧的能源对照。A 股没有可用的免费航空 ETF 历史，航空指数由
#: `CN_AIRLINES` 自行合成；能源侧用两大石油股，作用与 XLE 相同。
CN_ENERGY = {"SH601857": "中国石油", "SH600028": "中国石化"}

#: 外盘原油。CL = NYMEX WTI，OIL = 布伦特（新浪的命名，`hf_OIL` 中文名为
#: "英国布伦特原油"）。两条都取：WTI 是美国基准（美股航空的燃油成本更贴它），
#: 布伦特是全球基准（A 股航油采购更贴它），不该混为一谈。
OIL_FUTURES = {"CL": "NYMEX WTI 原油", "OIL": "布伦特原油"}


def _get_jsonp(url: str, params: dict, timeout: float = 30.0, retries: int = 4) -> list:
    """取新浪的 JSONP 响应并剥出 JSON 数组。

    新浪在响应体最前面塞了一段 `/*<script>location.href='//sina.com';</script>*/`
    的防盗链垫片，直接 `json.loads` 会炸，所以按第一个 `(` 与最后一个 `)` 截取。
    """
    session = requests.Session()
    session.headers.update({"User-Agent": _UA, "Referer": "https://finance.sina.com.cn"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            text = session.get(url, params=params, timeout=timeout).text
            body = text[text.index("(") + 1 : text.rindex(")")]
            data = json.loads(body)
            if not isinstance(data, list):
                raise ValueError(f"响应不是数组: {str(data)[:120]}")
            return data
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url} 取数失败: {last}")


def fetch_us_daily(symbol: str) -> pd.DataFrame:
    """单只美股的全历史日线（未复权）。列：date/open/high/low/close/volume。"""
    data = _get_jsonp(_US_DAILY_URL, {"symbol": symbol, "___qn": "3"})
    df = pd.DataFrame(data)
    df = df.rename(
        columns={"d": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df["date"] = pd.to_datetime(df.date)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = symbol
    return (
        df[["symbol", "date", "open", "high", "low", "close", "volume"]]
        .dropna(subset=["close"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_futures_daily(symbol: str) -> pd.DataFrame:
    """单个外盘期货的全历史日线（连续主力合约）。"""
    data = _get_jsonp(_FUT_DAILY_URL, {"symbol": symbol, "_": "1"})
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df.date)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = symbol
    keep = [c for c in ["symbol", "date", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def fetch_many(
    symbols: dict[str, str], kind: str, pause: float = 1.5, cache: Path | None = None
) -> pd.DataFrame:
    """批量取数。`kind` ∈ {"us", "futures"}。

    `pause=1.5s`：与 A 股源同一套限速经验（40 次/分）。这里只有十来个请求，
    远不到限流阈值，留着是为了不给新浪添乱。
    """
    fetcher = {"us": fetch_us_daily, "futures": fetch_futures_daily}[kind]
    parts = []
    for i, sym in enumerate(symbols):
        if i:
            time.sleep(pause)
        df = fetcher(sym)
        df["name"] = symbols[sym]
        print(
            f"  {sym:6s} {symbols[sym]:16s} {len(df):6,} 根  "
            f"{df.date.min():%Y-%m-%d} → {df.date.max():%Y-%m-%d}",
            flush=True,
        )
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
    return out
