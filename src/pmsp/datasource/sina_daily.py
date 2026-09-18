"""新浪财经日线数据源（本项目的**主数据源**）。

## 为什么不用 tushare

实测本机 token（120 积分档）的真实权限：

    daily        ❌ 无访问权限        ← 核心 OHLCV，直接堵死
    bak_daily    ❌ 无访问权限        ← 备用行情也堵死
    daily_basic  ✅ 但 **1 次/小时**   ← 只有 close，没有 OHLC，且配额无法批量
    stock_basic  ✅ 但 **1 次/小时**
    adj_factor   ✅ 但配额同上

也就是说 tushare 这一档拿不到日线。付 200 元/年升到 2000 积分能解决，
但新浪这条路免费且已验证完整，所以主数据源走新浪，tushare 只留作
一次性元数据的交叉校验（1 次/小时的配额刚好够抽样核对）。

## 新浪的三个接口

1. **K 线**：`CN_MarketData.getKLineData`，一次请求返回**单只股票的全历史**
   （`datalen` 上限约 6000 根，覆盖到 2001 年，本项目只要 2010+ 完全够）。
   **返回的是未复权价**，已验证：600000 收盘 19.89 → 9.07，除权日有 -4~-6% 跳空。
   已退市股票同样可取（`sh600087` 到 2014-06-04、`sz000033` 到 2017-07-06），
   所以**没有幸存者偏差**。
2. **后复权因子**：`realstock/company/{sym}/hfq.js`，给出全部除权节点与累计因子。
   ffill 后乘到价格上，除权日跳空被精确抹平（600000 六个年度除权日：
   -4.40% → +0.42%、-5.91% → -0.68% …）。比自己用 `pct_chg` 累乘更硬——
   除权日是官方给的，不用从收益率里反推。
3. **股票列表**：`Market_Center.getHQNodeData`，`node=hs_a` 已包含沪、深、北三所。

## 两处真实损失（新浪不提供成交额）

* **无 `amount`** → 流动性排名改用 `close × volume` 代理。真实成交额是
  `vwap × volume`，两者只差 `vwap/close`，日内通常在 ±1% 内，对**排名**几乎无影响。
* **无 `vwap`** → Alpha158 的 `VWAP0`（`$vwap/$close-1`）这一个因子做不了，
  因子数 216 → **215**。见 `pmsp.build.to_qlib.QLIB_FIELDS` 的说明。

## 限速

实测 1.2s 间隔连发 25 个请求全部成功。默认设 40 次/分钟（1.5s 间隔），
全市场约 6200 只（含退市）约需 2.5 小时，可断点续传。
东方财富实测连发几个就掐连接、网易返 502，都不适合批量。
"""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import requests

from .base import DataSource, RateLimiter

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_KLINE_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
_FACTOR_URL = "https://finance.sina.com.cn/realstock/company/{sym}/{kind}.js"
_LIST_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)

#: 单次 K 线请求能取回的最大根数（新浪的硬上限，实测约 6000）
MAX_KLINE_BARS = 6000

#: 每股一个 parquet 的列。价量均为**未复权**，`adj_factor` 是后复权累计因子。
SYMBOL_COLUMNS = ["code", "date", "open", "high", "low", "close", "volume", "adj_factor"]


def to_sina_symbol(code: str) -> str:
    """qlib 风格代码 -> 新浪符号：`SH600000` -> `sh600000`。"""
    return code.strip().lower()


def to_qlib_code(sina_symbol: str) -> str:
    """新浪符号 -> qlib 风格代码：`sh600000` -> `SH600000`。"""
    return sina_symbol.strip().upper()


def market_prefix(code6: str) -> str:
    """6 位数字代码 -> 交易所前缀（`sh` / `sz` / `bj`）。

    北交所的段比较散（原新三板精选层平移过来的 8/4 开头，加后来新发的 920），
    所以放在最后兜底，不靠"不是沪就是深"这种二分法。
    """
    c = str(code6).zfill(6)
    if c.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "sh"
    if c.startswith(("000", "001", "002", "003", "004", "300", "301", "200", "201")):
        return "sz"
    return "bj"


class SinaDaily(DataSource):
    """新浪财经日线源。按**股票**取数（不是按日期），一次拿全历史。"""

    name = "sina"

    def __init__(
        self,
        rate_limit_per_min: int = 40,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
        timeout: float = 30.0,
        datalen: int = MAX_KLINE_BARS,
        throttle_cooldown: float = 330.0,
        max_throttles: int = 4,
    ):
        self.limiter = RateLimiter(rate_limit_per_min)
        # 光靠滑动窗口限速器不够：它会在窗口开头**瞬间放行** rate_limit_per_min
        # 个请求，而新浪对突发比对平均速率敏感得多（实测 0.5s 间隔连发即触发
        # HTTP 456）。所以再叠一层最小间隔，把请求摊匀。
        self._min_interval = 60.0 / max(1, int(rate_limit_per_min))
        self._last_call = 0.0
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self.datalen = min(int(datalen), MAX_KLINE_BARS)
        self.throttle_cooldown = throttle_cooldown
        self.max_throttles = max_throttles
        self.call_count = 0
        self.throttle_count = 0
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": _UA, "Referer": "https://finance.sina.com.cn"}
        )

    # ------------------------------------------------------------------ 底层

    def _get(self, url: str, params: dict | None = None) -> str:
        """带限速与指数退避的 GET，返回响应文本。

        **HTTP 456 单独处理**。456 是新浪的"请求过频"码，且它封的是 IP、不是
        这一个请求——实测触发后约 4~5 分钟内所有请求一律 456。用常规的
        2/4/8s 退避去撞它，只会把 max_retries 全部烧掉然后判这只股票失败，
        而且期间的重试还在给封禁续期。所以遇到 456 就整体停 5.5 分钟，
        并且**不计入 max_retries**（另有 max_throttles 兜底防死循环）。
        """
        last: Exception | None = None
        attempt = 0
        throttles = 0
        while attempt < self.max_retries:
            self.limiter.acquire()
            gap = self._min_interval - (time.monotonic() - self._last_call)
            if gap > 0:
                time.sleep(gap)
            self._last_call = time.monotonic()
            self.call_count += 1
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code == 456:
                    self.throttle_count += 1
                    throttles += 1
                    last = RuntimeError("HTTP 456（新浪限流，IP 级）")
                    if throttles > self.max_throttles:
                        break
                    time.sleep(self.throttle_cooldown)
                    continue  # 不消耗 attempt
                last = RuntimeError(f"HTTP {resp.status_code}")
            except Exception as exc:  # noqa: BLE001  网络异常一律重试
                last = exc
            attempt += 1
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"{url} 重试 {attempt} 次仍失败: {last}")

    @staticmethod
    def _parse_jsonp(text: str) -> Any:
        """新浪的响应有时是 `var xxx={...};` 形式，剥掉外壳再解析。"""
        s = text.strip()
        if s.startswith(("[", "{")):
            return json.loads(s)
        lo, hi = s.find("{"), s.rfind("}")
        if lo < 0 or hi <= lo:
            lo, hi = s.find("["), s.rfind("]")
        if lo < 0 or hi <= lo:
            raise ValueError(f"无法解析新浪响应: {s[:120]!r}")
        return json.loads(s[lo : hi + 1])

    # ------------------------------------------------------------ 股票列表

    def list_listed_symbols(self, page_size: int = 100, max_pages: int = 200) -> pd.DataFrame:
        """全部**在市** A 股（`node=hs_a` 已含沪深北三所）。

        Returns
        -------
        pd.DataFrame
            列 `code`（qlib 风格，如 `SH600000`）、`sina_symbol`、`name`。
        """
        rows: list[dict] = []
        seen: set[str] = set()
        for page in range(1, max_pages + 1):
            text = self._get(
                _LIST_URL,
                {
                    "page": page,
                    "num": page_size,
                    "sort": "symbol",
                    "asc": 1,
                    "node": "hs_a",
                    "symbol": "",
                    "_s_r_a": "page",
                },
            )
            s = text.strip()
            if not s or s in ("null", "[]"):
                break
            try:
                data = self._parse_jsonp(s)
            except ValueError:
                break
            if not data:
                break
            new = 0
            for item in data:
                sym = str(item.get("symbol", "")).lower()
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                rows.append(
                    {
                        "code": to_qlib_code(sym),
                        "sina_symbol": sym,
                        "name": item.get("name", ""),
                    }
                )
                new += 1
            if new == 0:  # 翻到重复页说明已到底
                break
        return pd.DataFrame(rows, columns=["code", "sina_symbol", "name"])

    # -------------------------------------------------------------- 复权因子

    def fetch_adj_factors(self, sina_symbol: str, kind: str = "hfq") -> pd.Series:
        """后复权累计因子序列，索引为除权生效日。

        新浪把因子挂在一个 js 文件里，`data` 是 `{"d": 日期, "f": 累计因子}` 列表。
        第一个节点通常是 `1900-01-01, f=1.0`，所以对齐时 ffill 就够，
        **不需要 bfill**（bfill 会把后来的因子搬到更早的日期上）。
        """
        text = self._get(_FACTOR_URL.format(sym=sina_symbol, kind=kind))
        obj = self._parse_jsonp(text)
        data = obj.get("data") if isinstance(obj, dict) else None
        if not data:
            return pd.Series(dtype=float, name="adj_factor")
        df = pd.DataFrame(data)
        df["d"] = pd.to_datetime(df["d"], errors="coerce")
        df["f"] = pd.to_numeric(df["f"], errors="coerce")
        df = df.dropna(subset=["d", "f"])
        s = df.set_index("d")["f"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s.name = "adj_factor"
        return s

    # ------------------------------------------------------------------ K 线

    def fetch_one_symbol(
        self, ts_code: str, start_date: str = "", end_date: str = ""
    ) -> pd.DataFrame:
        """单只股票的**未复权** OHLCV，附带后复权因子列。

        Parameters
        ----------
        ts_code : str
            新浪符号（`sh600000`）或 qlib 风格代码（`SH600000`），两种都接受。
        start_date, end_date : str
            `YYYY-MM-DD` 或 `YYYYMMDD`，留空表示不裁剪。新浪不支持按区间请求，
            所以是取回全历史后在本地裁剪。

        Returns
        -------
        pd.DataFrame
            列为 `SYMBOL_COLUMNS`。**无数据时返回空 DataFrame**（新股/长期停牌/
            代码不存在都会走到这里）。`volume` 单位是**股**。
        """
        sym = to_sina_symbol(ts_code)
        text = self._get(
            _KLINE_URL, {"symbol": sym, "scale": 240, "ma": "no", "datalen": self.datalen}
        )
        s = text.strip()
        if not s or s in ("null", "[]"):
            return pd.DataFrame(columns=SYMBOL_COLUMNS)
        try:
            data = self._parse_jsonp(s)
        except ValueError:
            return pd.DataFrame(columns=SYMBOL_COLUMNS)
        if not data:
            return pd.DataFrame(columns=SYMBOL_COLUMNS)

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["day"], errors="coerce")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date", ignore_index=True)
        # 全天零成交（停牌挂着）留着没用，价格是上一日结算价的复制
        df = df[df["volume"] > 0]
        if df.empty:
            return pd.DataFrame(columns=SYMBOL_COLUMNS)

        fac = self.fetch_adj_factors(sym)
        if fac.empty:
            df["adj_factor"] = 1.0
        else:
            idx = df["date"]
            aligned = fac.reindex(fac.index.union(idx)).ffill().reindex(idx)
            # 最早的除权节点之后才有因子；更早的日期用最早节点的值（相当于把
            # 复权基准前推，只影响价格的绝对水平，不影响任何比值型因子）
            df["adj_factor"] = aligned.bfill().to_numpy()

        df["code"] = to_qlib_code(sym)
        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        return df[SYMBOL_COLUMNS].reset_index(drop=True)

    # ------------------------------------------------------------------ 不支持

    def fetch_one_day(self, trade_date: str) -> pd.DataFrame:
        raise NotImplementedError(
            "新浪没有『一次取全市场某一天』的接口，本源按股票取数。"
            "交易日历由所有股票的日期并集推出（见 pmsp.datasource.calendar）。"
        )
