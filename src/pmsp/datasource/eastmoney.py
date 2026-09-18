"""东方财富日线——**仅用于抽样交叉校验**，不做批量下载。

它直接给后复权价（`fqt=2`），所以是验证我们自己用 `pct_chg` 还原的复权序列
对不对的独立参照物。

三个已实测的坑，缺一个都拿不到数据：

1. **必须强制 IPv4**：`push2his.eastmoney.com` 的 DNS 会返回一个不可路由的
   IPv6（`240e:...`），直连报 "Cannot assign requested address"。
2. **限流极凶**：实测连发约 6 个请求后，服务器接受 TCP 但不返回任何内容，
   且 45 秒内不恢复。所以单线程 + 每请求间隔 + 指数退避，**继续探测只会延长封禁**。
3. **需要浏览器式请求头**（含 Referer），否则会被拒。
"""

from __future__ import annotations

import socket
import time

import pandas as pd
import requests

from .base import RAW_COLUMNS, DataSource

_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "*/*",
}


def _force_ipv4() -> None:
    """把 urllib3 的地址族锁到 IPv4。见模块 docstring 第 1 条。"""
    import urllib3.util.connection as urllib3_conn

    urllib3_conn.allowed_gai_family = lambda: socket.AF_INET


def _secid(ts_code: str) -> str:
    """`600000.SH` -> `1.600000`；`000001.SZ` -> `0.000001`。"""
    code, _, market = ts_code.partition(".")
    prefix = {"SH": "1", "SZ": "0", "BJ": "0"}.get(market.upper())
    if prefix is None:
        raise ValueError(f"无法识别的市场后缀: {ts_code}")
    return f"{prefix}.{code}"


class EastmoneyDaily(DataSource):
    """备用源 / 交叉校验源。`adjust` 参数：0 不复权、1 前复权、2 后复权。"""

    name = "eastmoney"

    def __init__(self, sleep_between: float = 3.0, max_retries: int = 3, retry_backoff: float = 5.0):
        _force_ipv4()
        self.sleep_between = sleep_between
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._last_call = 0.0

    def _get(self, params: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.sleep_between:
                time.sleep(self.sleep_between - elapsed)
            self._last_call = time.monotonic()
            try:
                resp = self._session.get(_KLINE_URL, params=params, timeout=20)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff ** (attempt + 1))
        raise RuntimeError(f"东方财富请求失败（可能已被限流，勿继续重试）: {last_exc}") from last_exc

    def fetch_one_day(self, trade_date: str) -> pd.DataFrame:
        raise NotImplementedError("东方财富按股票取数，不支持按日取全市场；批量下载请用 tushare")

    def fetch_one_symbol(
        self, ts_code: str, start_date: str, end_date: str, adjust: int = 2
    ) -> pd.DataFrame:
        payload = self._get(
            {
                "secid": _secid(ts_code),
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": 101,  # 日线
                "fqt": adjust,  # 2 = 后复权
                "beg": start_date,
                "end": end_date,
            }
        )
        klines = (payload.get("data") or {}).get("klines") or []
        rows = []
        for line in klines:
            parts = line.split(",")
            # 日期,开,收,高,低,成交量(手),成交额(元),振幅,涨跌幅,涨跌额,换手率
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": parts[0].replace("-", ""),
                    "open": float(parts[1]),
                    "close": float(parts[2]),
                    "high": float(parts[3]),
                    "low": float(parts[4]),
                    "vol": float(parts[5]),
                    "amount": float(parts[6]) / 1000.0,  # 元 -> 千元，对齐 tushare 口径
                    "pct_chg": float(parts[8]),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=RAW_COLUMNS)
        df["pre_close"] = pd.NA
        df["change"] = pd.NA
        return df.reindex(columns=RAW_COLUMNS).sort_values("trade_date", ignore_index=True)

    def trading_calendar(self, start_date: str, end_date: str) -> list[str]:
        """用上证指数的 K 线反推交易日历（1 个请求）。

        我们默认的日历是从 `daily` 的空响应推出来的（免费档没 `trade_cal`），
        这个函数用来独立核对那份日历对不对。
        """
        payload = self._get(
            {
                "secid": "1.000001",  # 上证综指
                "fields1": "f1",
                "fields2": "f51",
                "klt": 101,
                "fqt": 0,
                "beg": start_date,
                "end": end_date,
            }
        )
        klines = (payload.get("data") or {}).get("klines") or []
        return [line.split(",")[0].replace("-", "") for line in klines]
