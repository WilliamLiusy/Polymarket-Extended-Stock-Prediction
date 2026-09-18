"""交易所官网的**退市股票名单**——消除幸存者偏差的关键一环。

新浪的 `Market_Center` 只列**在市**股票。若股票池只由在市股票构成，
那就是典型的幸存者偏差：2010 年买入并持有到退市的那些股票的亏损被整体抹掉，
baseline 的 IC 与回测收益都会系统性虚高。

tushare 的 `stock_basic(list_status='D')` 能给退市名单，但本机 token 是
**1 次/小时**的配额，且它本身也是个外部依赖。交易所官网的接口免费、无限速、
且是**权威来源**，所以走这条。（akshare 的 `stock_info_sh_delist` /
`stock_info_sz_delist` 也是这么做的。）

## 两个接口的字段口径完全不同，别混

上交所 `getStockListData2.do?stockType=5`（终止上市）：
    SECURITY_CODE_A  A 股代码      SECURITY_ABBR_A  简称
    LISTING_DATE     上市日期      CHANGE_DATE      终止上市日期（可能为 '-'）

深交所 `ShowReport/data?CATALOGID=1793_ssgs&TABKEY=tab2`（终止上市）：
    zqdm 证券代码   zqjc 证券简称   ssrq 上市日期   zzrq 终止上市日期
    —— 分页，`metadata.pagecount` 给总页数。

注意上交所名单里混着 B 股（`SECURITY_CODE_B` / `900xxx`）和并购退市的
正常公司（如中国重工吸收合并），这里只取 A 股代码，是否纳入股票池由
下游的流动性排名和上市天数过滤决定，这一层不做业务判断。
"""

from __future__ import annotations

import time

import pandas as pd
import requests

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_SSE_URL = "http://query.sse.com.cn/security/stock/getStockListData2.do"
_SZSE_URL = "http://www.szse.cn/api/report/ShowReport/data"

DELIST_COLUMNS = ["code", "name", "list_date", "delist_date", "exchange"]


def _session(referer: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Referer": referer})
    return s


def _norm_date(v: object) -> pd.Timestamp | None:
    """交易所把缺失写成 `'-'` 或空串，pd.to_datetime 会抛，先挡掉。"""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in {"-", "--", "null", "None"}:
        return None
    ts = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(ts) else ts


def fetch_sse_delisted(page_size: int = 100, max_pages: int = 20,
                       gap: float = 1.0) -> pd.DataFrame:
    """上交所终止上市公司名单。"""
    s = _session("http://www.sse.com.cn/")
    rows: list[dict] = []
    for page in range(1, max_pages + 1):
        r = s.get(
            _SSE_URL,
            params={
                "isPagination": "true",
                "stockType": 5,  # 5 = 终止上市
                "pageHelp.beginPage": page,
                "pageHelp.pageSize": page_size,
                "pageHelp.pageNo": page,
                "pageHelp.cacheSize": 1,
                "pageHelp.endPage": page,
            },
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        result = payload.get("result") or []
        if not result:
            break
        for item in result:
            code = str(item.get("SECURITY_CODE_A") or "").strip()
            if not code.isdigit() or len(code) != 6:
                continue  # 纯 B 股退市等，没有 A 股代码
            rows.append(
                {
                    "code": code,
                    "name": (item.get("SECURITY_ABBR_A") or "").strip(),
                    "list_date": _norm_date(item.get("LISTING_DATE")),
                    "delist_date": _norm_date(item.get("CHANGE_DATE")),
                    "exchange": "SH",
                }
            )
        total = (payload.get("pageHelp") or {}).get("total")
        if total is not None and page * page_size >= int(total):
            break
        time.sleep(gap)
    return pd.DataFrame(rows, columns=DELIST_COLUMNS)


def fetch_szse_delisted(max_pages: int = 30, gap: float = 1.0) -> pd.DataFrame:
    """深交所终止上市公司名单。"""
    s = _session("http://www.szse.cn/")
    rows: list[dict] = []
    pagecount = None
    for page in range(1, max_pages + 1):
        r = s.get(
            _SZSE_URL,
            params={
                "SHOWTYPE": "JSON",
                "CATALOGID": "1793_ssgs",
                "TABKEY": "tab2",  # tab2 = 终止上市
                "PAGENO": page,
                "random": f"0.{page:04d}",
            },
            timeout=30,
        )
        r.raise_for_status()
        blocks = r.json()
        blocks = blocks if isinstance(blocks, list) else [blocks]
        got = 0
        for blk in blocks:
            data = blk.get("data") or []
            if not data:
                continue
            if pagecount is None:
                pagecount = (blk.get("metadata") or {}).get("pagecount")
            for item in data:
                code = str(item.get("zqdm") or "").strip()
                if not code.isdigit() or len(code) != 6:
                    continue
                rows.append(
                    {
                        "code": code,
                        # 深交所的简称带全角字符（`PT金田Ａ`），原样保留
                        "name": (item.get("zqjc") or "").strip(),
                        "list_date": _norm_date(item.get("ssrq")),
                        "delist_date": _norm_date(item.get("zzrq")),
                        "exchange": "SZ",
                    }
                )
                got += 1
        if got == 0:
            break
        if pagecount is not None and page >= int(pagecount):
            break
        time.sleep(gap)
    df = pd.DataFrame(rows, columns=DELIST_COLUMNS)
    return df.drop_duplicates(subset=["code"], keep="first", ignore_index=True)


def fetch_all_delisted(min_delist_date: str | None = None) -> pd.DataFrame:
    """沪深两所的退市名单合并。

    Parameters
    ----------
    min_delist_date : str, optional
        只保留在此日期之后退市的股票。**注意默认不过滤**：一只 2012 年退市的
        股票在 2010–2012 年间是可交易的，属于当时的股票池，必须下载。
        只有早于数据起点就退市的才该剔除。

    Returns
    -------
    pd.DataFrame
        列 `code`（6 位）、`name`、`list_date`、`delist_date`、`exchange`、
        `sina_symbol`（如 `sh600087`）。
    """
    parts = []
    for fn in (fetch_sse_delisted, fetch_szse_delisted):
        try:
            parts.append(fn())
        except Exception as exc:  # noqa: BLE001  一所挂了不该拖垮另一所
            print(f"[警告] {fn.__name__} 失败：{exc}")
    if not parts:
        return pd.DataFrame(columns=[*DELIST_COLUMNS, "sina_symbol"])
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["code"], keep="first", ignore_index=True)
    if min_delist_date:
        cut = pd.Timestamp(min_delist_date)
        # delist_date 缺失的一律保留——宁可多下一只，也不要漏掉当时可交易的股票
        df = df[df["delist_date"].isna() | (df["delist_date"] >= cut)]
    df["sina_symbol"] = df["exchange"].str.lower() + df["code"]
    return df.reset_index(drop=True)
