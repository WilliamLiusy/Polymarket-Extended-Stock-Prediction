"""baostock 日线源：专门用来**补齐新浪源拿不到的字段**（不替换主源）。

## 它补上了什么

新浪只给 OHLCV，README「已知局限」里前六条有五条卡在缺字段上。baostock 免费、
无 token、无注册，实测（2026-09-28，本机）把其中四条补齐、两条部分补齐：

| 缺的东西 | baostock | 实测证据 |
|---|---|---|
| 成交额 `amount` | ✅ 逐日，单位**元** | SH600000 2010-01-04 = 14.20 亿元，与 `close×volume` 代理差 1.2% |
| `$vwap` | ✅ `amount / volume` | `vwap/close` 中位 0.9999、1%/99% 分位 0.9755/1.0213，量纲正确 |
| 历史 ST 标记 | ✅ 逐日 `isST` | SH600891 从 2010-01-04 起 877 天为 ST；SZ000668 首个 ST 日 2025-04-30 |
| 真实停牌标记 | ✅ 逐日 `tradestatus` | SZ000001 有 47 天 `tradestatus=0`（此前只能靠"日期序列有缺口"推） |
| 流通市值 | ⚠️ 可由 `turn` 反推 | `close×volume/(turn/100)`，SZ000001 2026-09-28 = 2193 亿元，量级对 |
| 行业分类 | ⚠️ **只有当期快照** | `query_stock_industry()` 的 `updateDate` 只有一个取值 → 回溯用有轻微前视 |
| 总市值 | ❌ | 只能从 `peTTM`/`pbMRQ` 间接凑，不如流通市值干净 |

指数成分股历史（README 局限第 3 条）走 `fetch_index_members`，见下。

## 五个必须知道的坑

1. **`ResultData.get_data()` 在 pandas ≥ 2.0 上直接崩**——baostock 0.9.4 里它用的是
   `DataFrame.append`，pandas 2.0 已删除该方法。所以本模块一律自己 `next()` 逐行取，
   不碰 `get_data()`。（本项目 pandas 是 2.x。）
2. **不含北交所**。`bj.*` 一律返回 `10004011 股票代码未标识sh或sz`。我们的下载列表里
   BJ 共 344 只拿不到补充字段——影响很小（BJ 成交额远低于主板，几乎进不了流动性前 N），
   但股票池若改成"按真实 amount 排名"，BJ 必须显式排除而不是当成 amount=NaN 静默丢掉。
3. **`multiprocessing.Pool` + fork 会挂死**。baostock 在模块级持有一个全局 socket，
   fork 出来的子进程共用同一个连接，实测 12 只股票跑 12 分钟没有任何输出。并行必须用
   **各自独立 login 的独立进程**（`subprocess` / `spawn`），见 `scripts/07_fetch_baostock.py`。
4. **并发高了会被拉黑**，`10001011 黑名单用户`，账号级、重试无用。实测 8 个 worker
   2.5 分钟触发。默认 `workers: 3`，见下「速度」。
5. **一个进程 `logout()` 会把同账号的其他进程踢下线**。免费用户都是同一个匿名账号，
   实测另一个进程调 `logout()` 后，本进程下一次查询立刻 `10001001 用户未登录`。
   两道防线：`_query` 遇到 10001001 自动重登（不消耗重试预算，见 `max_relogins`），
   并行 worker 传 `logout_on_exit=False`（宁可不登出，也别把兄弟进程弄死）。

## 没有幸存者偏差

退市股票同样取得到，这点与新浪一致，所以两边可以逐点对齐：
SH600087 到 2014-06-05（1069 行）、SZ000033 到 2017-07-07（1824 行）、
SH600806 到 2018-07-13（2073 行）。

## 单位（与 `base.RAW_COLUMNS` 的 tushare 口径**不同**，别混）

    volume   成交量，单位 **股**（tushare 的 `vol` 是手）
    amount   成交额，单位 **元**（tushare 的 `amount` 是千元）
    turn     换手率，单位 **%**，流通股口径

本模块输出的 `amount` 直接可与 `pmsp.build.adjust` 里的 `amount_yuan` 比较，
不需要再乘 100 或 1000。

## 速度，以及并发的**硬上限**

单进程约 450 行/s，一只股票 16 年约 4000 行 ≈ 9s（服务端忙时能掉到 24–95s），
全市场 5489 只（沪深）单进程要 20 小时以上。想靠并发压缩——但压不动：

**8 个独立进程跑 2.5 分钟就被 `10001011 黑名单用户` 拉黑。** 80 只里 48 只落盘、
22 只失败，之后**连单进程都登不进去**。拉黑是按账号（免费用户共用一个匿名账号）
+ IP 判的，重试、重登、重开进程全都没用，只能降并发 + 等——实测触发后近 1 小时单进程登录仍被拒，别指望几分钟就恢复。

所以：`configs/data.yaml` 的 `supplement.workers` 默认 **3**（不是 8），
`rate_limit_per_min` 降到 240，并且 `BaostockBlacklisted` 一抛出就**立刻停**
——继续撞只会刷一屏失败，还可能延长封禁。已落盘的部分可断点续传，
全量分几次跑完是正常用法。
"""

from __future__ import annotations

import contextlib
import io
import time
from typing import Any

import pandas as pd

from .base import DataSource, RateLimiter

#: 逐日字段。`peTTM`/`pbMRQ` 顺手带上——将来做市值/估值中性化时不用重拉一遍。
_KLINE_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,"
    "tradestatus,pctChg,isST,peTTM,pbMRQ"
)

#: 每股一个 parquet 的列。**不复权**价（`adjustflag=3`），与新浪源同口径，
#: 这样两边能逐点对齐；复权仍然由 `pmsp.build.adjust` 用新浪的官方因子做。
SUPPLEMENT_COLUMNS = [
    "code",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turn",
    "pct_chg",
    "trade_status",
    "is_st",
    "pe_ttm",
    "pb_mrq",
]

#: `adjustflag` 的取值。1 = 后复权，2 = 前复权，3 = 不复权。
ADJ_BACKWARD, ADJ_FORWARD, ADJ_NONE = "1", "2", "3"

#: 会话被别的进程 logout 踢掉。重登即可继续。
NOT_LOGGED_IN_CODE = "10001001"
#: **并发过高会被拉黑**，账号级、重试无用。实测 8 个 worker 几分钟就触发。
BLACKLIST_CODE = "10001011"


class BaostockBlacklisted(RuntimeError):
    """账号被 baostock 拉黑（`10001011`）。

    重试、重登、换进程都没用——它是按账号（免费用户共用匿名账号）+ IP 判的，
    只能**降并发 + 等**。所以这个异常要一路向上传，让 worker 立刻停下来，
    而不是继续把剩下几千只股票逐个撞成失败。
    """

_NUMERIC = ["open", "high", "low", "close", "volume", "amount", "turn", "pct_chg",
            "pe_ttm", "pb_mrq"]


def to_baostock_code(code: str) -> str:
    """qlib 风格代码 -> baostock 符号：`SH600000` -> `sh.600000`。"""
    c = code.strip().upper()
    return f"{c[:2].lower()}.{c[2:]}"


def to_qlib_code(bs_code: str) -> str:
    """baostock 符号 -> qlib 风格代码：`sh.600000` -> `SH600000`。"""
    return bs_code.strip().replace(".", "").upper()


def is_supported(code: str) -> bool:
    """baostock 只覆盖沪深两市；北交所（`BJ*`）一律不支持。"""
    return code.strip().upper()[:2] in ("SH", "SZ")


class BaostockDaily(DataSource):
    """baostock 日线源。按**股票**取数，一次拿全历史。

    用法（必须在 session 内，否则每次查询都会被拒）：

        with BaostockDaily() as ds:
            df = ds.fetch_one_symbol("SH600000", "2010-01-01", "2026-09-18")
    """

    name = "baostock"

    def __init__(
        self,
        rate_limit_per_min: int = 600,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        max_relogins: int = 5,
        logout_on_exit: bool = True,
    ):
        # baostock 没有公布速率上限，实测也没撞到拒绝。这里仍挂一个宽松的限速器：
        # 免费公共服务，出问题时能一个参数调下来，比出问题时才去改代码好。
        self.limiter = RateLimiter(rate_limit_per_min)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_relogins = max_relogins
        # 并发时置 False：logout 会把**同一个匿名账号的其他进程**一起踢下线（实测），
        # 长任务的 worker 宁可不登出，进程退出后服务端自己会回收。
        self.logout_on_exit = logout_on_exit
        self.call_count = 0
        self.relogin_count = 0
        self._logged_in = False
        self._bs: Any = None

    # ------------------------------------------------------------ session
    def login(self) -> None:
        """登录。baostock 把登录状态存在模块全局里，所以一个进程只能有一个 session。

        `10002007 网络接收错误` 是并发起多个进程时的常见开局失败（服务端还没给出
        连接就被下一个请求挤掉），退避重试就好；`10001011 黑名单用户` 重试无用，
        直接抛 `BaostockBlacklisted` 让调用方停下来等。
        """
        import baostock as bs  # noqa: PLC0415  可选依赖，延迟导入

        self._bs = bs
        last = ""
        for attempt in range(self.max_retries):
            # baostock 无条件往 stdout 打 "login success!"，会把进度条和报告冲烂
            with contextlib.redirect_stdout(io.StringIO()):
                res = bs.login()
            if res.error_code == "0":
                self._logged_in = True
                return
            last = f"{res.error_code} {res.error_msg}"
            if res.error_code == BLACKLIST_CODE:
                raise BaostockBlacklisted(f"baostock 登录被拒：{last}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_backoff * (attempt + 1))
        raise RuntimeError(f"baostock 登录失败（重试 {self.max_retries} 次）：{last}")

    def logout(self) -> None:
        if self._logged_in and self._bs is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                self._bs.logout()
            self._logged_in = False

    def __enter__(self) -> BaostockDaily:
        self.login()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.logout_on_exit:
            self.logout()

    def _require_session(self) -> Any:
        if not self._logged_in:
            raise RuntimeError("请先 login()（或用 `with BaostockDaily() as ds:`）")
        return self._bs

    # ------------------------------------------------------------ 底层取数
    def _rows(self, rs: Any) -> pd.DataFrame:
        """把 ResultData 逐行读成 DataFrame。

        **不能用 `rs.get_data()`**：baostock 0.9.4 里它调 `DataFrame.append`，
        该方法在 pandas 2.0 已被删除，直接 AttributeError。
        """
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            raise RuntimeError(f"baostock 查询失败：{rs.error_code} {rs.error_msg}")
        return pd.DataFrame(rows, columns=list(rs.fields))

    def _query(self, fn: str, **kwargs: Any) -> pd.DataFrame:
        """带重试的查询。失败重试对 baostock 是安全的——所有接口都是只读幂等。

        **会话是会被踢掉的**，报 `10001001 用户未登录`。实测触发条件：同一账号
        （免费档是匿名账号）另一个进程 `logout` 时，本进程的会话一起失效。
        并行抓数时几乎必然遇到，所以撞到这个错误码就**重新登录再重试**，
        而不是把它当普通网络错误干等——干等永远等不回来。
        """
        last: Exception | None = None
        attempt = relogin = 0
        while attempt < self.max_retries and relogin <= self.max_relogins:
            bs = self._require_session()
            self.limiter.acquire()
            self.call_count += 1
            try:
                return self._rows(getattr(bs, fn)(**kwargs))
            except Exception as exc:  # noqa: BLE001  网络抖动 / 服务端偶发错误码
                last = exc
                if BLACKLIST_CODE in str(exc):
                    # 拉黑是账号级的，重试只会把剩下几千只股票全撞成失败。立刻上抛。
                    raise BaostockBlacklisted(
                        f"{fn} 被拒：{exc}\n"
                        f"→ 已调用 {self.call_count} 次。降低 --workers（≤3）并等一段时间再试。"
                    ) from exc
                if NOT_LOGGED_IN_CODE in str(exc):
                    # 掉线单独计数：它不是"这次查询有问题"，而是"会话没了"，
                    # 和网络抖动混在一个预算里会让长任务在中途整批失败
                    relogin += 1
                    self.relogin_count += 1
                    self._logged_in = False
                    self.login()
                    continue
                attempt += 1
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
        raise RuntimeError(
            f"{fn} 失败（重试 {attempt} 次、重登 {relogin} 次）：{last}"
        ) from last

    # ------------------------------------------------------------ DataSource 接口
    def fetch_one_day(self, trade_date: str) -> pd.DataFrame:
        """baostock 没有"一次取全市场某日行情"的接口，本源只能按股票取。

        （`query_all_stock` 只返回 code / 名称 / 是否停牌，没有价量，
        所以连退化实现都给不了，硬报错比返回半个结果好。）
        """
        raise NotImplementedError(
            "baostock 只能按股票取数，请用 fetch_one_symbol()。"
            "按日期取全市场是 tushare `daily` 的能力，见 pmsp.datasource.tushare_daily"
        )

    def fetch_one_symbol(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str = ADJ_NONE,
    ) -> pd.DataFrame:
        """取单只股票的日线补充字段。

        Parameters
        ----------
        code : str
            qlib 风格代码（`SH600000`）或 baostock 符号（`sh.600000`），都认。
        start_date, end_date : str
            `YYYY-MM-DD` 或 `YYYYMMDD`。
        adjust : str
            `ADJ_NONE`（默认，不复权，与新浪源同口径）/ `ADJ_BACKWARD` / `ADJ_FORWARD`。

        Returns
        -------
        pd.DataFrame
            列为 `SUPPLEMENT_COLUMNS`。**北交所或无数据时返回空 DataFrame**
            （不抛异常——批量下载时一只股票没数据不该中断整轮）。
        """
        qcode = to_qlib_code(code) if "." in code else code.strip().upper()
        if not is_supported(qcode):
            return pd.DataFrame(columns=SUPPLEMENT_COLUMNS)

        raw = self._query(
            "query_history_k_data_plus",
            code=to_baostock_code(qcode),
            fields=_KLINE_FIELDS,
            start_date=_norm_date(start_date),
            end_date=_norm_date(end_date),
            frequency="d",
            adjustflag=adjust,
        )
        if raw.empty:
            return pd.DataFrame(columns=SUPPLEMENT_COLUMNS)

        out = pd.DataFrame(
            {
                "code": qcode,
                "date": pd.to_datetime(raw["date"]),
                "open": raw["open"],
                "high": raw["high"],
                "low": raw["low"],
                "close": raw["close"],
                "volume": raw["volume"],
                "amount": raw["amount"],
                "turn": raw["turn"],
                "pct_chg": raw["pctChg"],
                # 停牌日 baostock 给的是 tradestatus=0、价格沿用前收、volume=0
                "trade_status": pd.to_numeric(raw["tradestatus"], errors="coerce").astype("Int8"),
                "is_st": pd.to_numeric(raw["isST"], errors="coerce").astype("Int8"),
                "pe_ttm": raw["peTTM"],
                "pb_mrq": raw["pbMRQ"],
            }
        )
        # baostock 的空值是空字符串而不是 NaN；停牌日的 turn / peTTM 常为空
        for col in _NUMERIC:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out[SUPPLEMENT_COLUMNS].sort_values("date", ignore_index=True)

    def fetch_adjusted_close(
        self, code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """取**后复权**收盘价序列，专供复权交叉校验用（列：`trade_date` / `close`）。

        列名刻意对齐 `pmsp.build.adjust.compare_adjustment` 期望的参照格式
        （`trade_date` 为 `YYYYMMDD` 字符串），这样它能和东财参照走同一条比较路径。
        """
        qcode = to_qlib_code(code) if "." in code else code.strip().upper()
        if not is_supported(qcode):
            return pd.DataFrame(columns=["trade_date", "close"])
        raw = self._query(
            "query_history_k_data_plus",
            code=to_baostock_code(qcode),
            fields="date,close",
            start_date=_norm_date(start_date),
            end_date=_norm_date(end_date),
            frequency="d",
            adjustflag=ADJ_BACKWARD,
        )
        if raw.empty:
            return pd.DataFrame(columns=["trade_date", "close"])
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(raw["date"]).dt.strftime("%Y%m%d"),
                "close": pd.to_numeric(raw["close"], errors="coerce"),
            }
        )

    # ------------------------------------------------------------ 指数成分 / 行业
    def fetch_index_members(self, index: str, date: str) -> pd.DataFrame:
        """取某一天的指数成分股（沪深300 / 中证500 / 上证50）。

        返回列 `code`（qlib 风格）、`name`、`update_date`。

        **`update_date` 是关键**：baostock 返回的是"截至 `date` 的最近一次成分更新"，
        更新频率约每周一次。所以拿它做 point-in-time 股票池时，生效日必须用
        `date` 之后的交易日，而不是 `update_date`——我们在 `date` 这一天能知道的
        就是这份名单。
        """
        fn = {
            "hs300": "query_hs300_stocks",
            "zz500": "query_zz500_stocks",
            "sz50": "query_sz50_stocks",
        }.get(index.lower())
        if fn is None:
            raise ValueError(f"未知指数 {index}（支持 hs300 / zz500 / sz50）")
        raw = self._query(fn, date=_norm_date(date))
        if raw.empty:
            return pd.DataFrame(columns=["code", "name", "update_date"])
        return pd.DataFrame(
            {
                "code": raw["code"].map(to_qlib_code),
                "name": raw["code_name"],
                "update_date": pd.to_datetime(raw["updateDate"]),
            }
        )

    def fetch_industry(self) -> pd.DataFrame:
        """取行业分类（证监会口径）。返回列 `code` / `name` / `industry` / `update_date`。

        **只有一个快照**（实测 `update_date` 只有一个取值，5556 行里 334 行行业为空）。
        拿它做历史行业中性化 = 用今天的分类套到 2010 年，有轻微前视偏差。
        行业变更在 A 股属低频事件，多数研究容忍这一近似，但必须写在报告里，
        不能假装是 point-in-time 的。
        """
        raw = self._query("query_stock_industry")
        if raw.empty:
            return pd.DataFrame(columns=["code", "name", "industry", "update_date"])
        return pd.DataFrame(
            {
                "code": raw["code"].map(to_qlib_code),
                "name": raw["code_name"],
                "industry": raw["industry"].replace("", pd.NA),
                "update_date": pd.to_datetime(raw["updateDate"]),
            }
        )


def _norm_date(d: str) -> str:
    """`20100104` / `2010-01-04` -> `2010-01-04`（baostock 只认带横线的）。"""
    s = str(d).strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else s


def derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    """由原始补充字段推出 `vwap` / `float_mv` / `suspended`。

    * `vwap = amount / volume` —— 真实成交均价。停牌日 `volume=0`，置 NaN 而不是 0。
    * `float_mv = close × volume / (turn/100)` —— 流通市值（元）。`turn` 是流通股
      口径的换手率，所以 `volume/(turn/100)` 就是流通股本。停牌日 `turn` 为空 → NaN。
    * `suspended = trade_status == 0` —— 真实停牌标记，不再靠"日期序列有缺口"推断。
    """
    out = df.copy()
    vol = out["volume"].where(out["volume"] > 0)
    out["vwap"] = out["amount"] / vol
    turn = out["turn"].where(out["turn"] > 0)
    out["float_mv"] = out["close"] * vol / (turn / 100.0)
    out["suspended"] = out["trade_status"].eq(0).fillna(False)
    return out
