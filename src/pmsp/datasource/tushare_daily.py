"""tushare 免费档（120 积分）日线下载器。

免费档只开放 `daily`（非复权日线）。这个类把它的三个坑都处理了：

1. **限速**：50 次/分钟。超了 tushare 直接抛异常，所以用 `RateLimiter` 主动卡住。
2. **单次 6000 行上限**：A 股现有 5917 只，已贴着上限。用 `offset` 翻页兜住，
   不然某天新股上市突破 6000 就会**静默丢数据**。
3. **偶发网络错误**：指数退避重试。

复权不在这里做（见 `pmsp.build.adjust`）：免费档没有 `adj_factor`，
但 `daily` 的 `pct_chg` 本身就是除权后收益率，cumprod 即可还原。
"""

from __future__ import annotations

import os
import time

import pandas as pd

from .base import RAW_COLUMNS, DataSource, RateLimiter


class TushareDaily(DataSource):
    name = "tushare"

    def __init__(
        self,
        token: str | None = None,
        rate_limit_per_min: int = 50,
        max_retries: int = 5,
        retry_backoff: float = 2.0,
        page_limit: int = 6000,
    ):
        import tushare as ts

        token = token or os.environ.get("TUSHARE_TOKEN")
        if token:
            ts.set_token(token)
        # 没显式给 token 时依赖 tushare 预存的凭证；不可用会在这里就报错
        self._api = ts.pro_api()
        self._limiter = RateLimiter(rate_limit_per_min)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.page_limit = page_limit
        self.call_count = 0

    def _query(self, api_name: str, **kwargs) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._limiter.acquire()
            self.call_count += 1
            try:
                return self._api.query(api_name, **kwargs)
            except Exception as exc:  # tushare 把配额/网络错误都抛成普通 Exception
                last_exc = exc
                msg = str(exc)
                # 触发每分钟限制时多等一会儿，别把退避浪费在立刻重试上
                wait = self.retry_backoff ** (attempt + 1)
                if "每分钟" in msg or "minute" in msg.lower():
                    wait = max(wait, 61.0)
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        raise RuntimeError(f"{api_name} 重试 {self.max_retries} 次仍失败: {last_exc}") from last_exc

    def fetch_one_day(self, trade_date: str) -> pd.DataFrame:
        """取某交易日全市场日线，自动翻页。空 DataFrame 表示非交易日。"""
        pages: list[pd.DataFrame] = []
        offset = 0
        while True:
            page = self._query(
                "daily",
                trade_date=trade_date,
                fields=",".join(RAW_COLUMNS),
                offset=offset,
                limit=self.page_limit,
            )
            if page is None or page.empty:
                break
            pages.append(page)
            if len(page) < self.page_limit:
                break
            offset += self.page_limit

        if not pages:
            return pd.DataFrame(columns=RAW_COLUMNS)
        df = pd.concat(pages, ignore_index=True)
        # 翻页边界上偶发重复，去重后才是干净的一天
        return df.drop_duplicates(subset=["ts_code", "trade_date"], ignore_index=True)

    def fetch_one_symbol(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        pages: list[pd.DataFrame] = []
        offset = 0
        while True:
            page = self._query(
                "daily",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields=",".join(RAW_COLUMNS),
                offset=offset,
                limit=self.page_limit,
            )
            if page is None or page.empty:
                break
            pages.append(page)
            if len(page) < self.page_limit:
                break
            offset += self.page_limit
        if not pages:
            return pd.DataFrame(columns=RAW_COLUMNS)
        out = pd.concat(pages, ignore_index=True)
        return out.sort_values("trade_date", ignore_index=True)
