"""行情数据源抽象。

加新数据源（或以后接 Polymarket）只要实现 `DataSource`，下游全部不用改。

统一的原始 schema 就是 tushare `daily` 的字段，一个字都不改名——
调试时能直接跟官方文档对照，这比起一套自创命名更省事：

    ts_code trade_date open high low close pre_close change pct_chg vol amount

单位（tushare 口径，务必记住，算 vwap 时会用到）：
    vol    成交量，单位 **手**（1 手 = 100 股）
    amount 成交额，单位 **千元**
"""

from __future__ import annotations

import abc
import time
from collections import deque

import pandas as pd

RAW_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]


class RateLimiter:
    """滑动窗口限速器。

    tushare 免费档 50 次/分钟，超了会直接抛异常而不是排队，所以必须自己卡住。
    """

    def __init__(self, max_calls: int, period_sec: float = 60.0):
        self.max_calls = max_calls
        self.period = period_sec
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] >= self.period:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            sleep_for = self.period - (now - self._calls[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)
            return self.acquire()
        self._calls.append(time.monotonic())


class DataSource(abc.ABC):
    """日线行情数据源。"""

    name: str = "base"

    @abc.abstractmethod
    def fetch_one_day(self, trade_date: str) -> pd.DataFrame:
        """取某一交易日的全市场日线。

        Parameters
        ----------
        trade_date : str
            `YYYYMMDD`。

        Returns
        -------
        pd.DataFrame
            列为 `RAW_COLUMNS`。**非交易日返回空 DataFrame** —— 交易日历就是
            靠这一点推出来的（免费档没有 `trade_cal` 接口）。
        """

    def fetch_one_symbol(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """取单只股票的历史日线。只用于抽样交叉校验，不用于批量下载。"""
        raise NotImplementedError(f"{self.name} 未实现按股票取数")
