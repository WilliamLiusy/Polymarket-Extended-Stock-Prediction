"""把外部日频时序注册成 qlib 伪 instrument —— Polymarket 因子的接入点。

## 为什么要走"伪 instrument"这条路

Polymarket 的市场都是**全市场级别的宏观事件**（降息概率、大选、关税）。
一个宏观概率序列在任一天对所有股票都是同一个数，横截面标准化之后**恒等于零**，
对 IC 的贡献严格为 0。所以不能直接当因子塞进去。

有意义的用法是「**外部时序 × 个股暴露度**」的交叉因子：个股收益对该宏观序列
的滚动 beta / 相关性，这个量是**因股而异**的，才有横截面区分度。

qlib 的 `ChangeInstrument` 算子原生支持在表达式里引用另一个 instrument，
所以只要把外部序列注册成一个 instrument，暴露度因子就是一行表达式，
**不需要改 qlib 一行代码**：

    Corr($close/Ref($close,1)-1,
         ChangeInstrument("PM_FED_CUT", $close/Ref($close,1)-1), 60)

## 两个必须守住的约束

1. **外部序列必须先对齐到已有交易日历。** `dump_bin` 的日历是所有输入文件日期的
   **并集**——若外部序列带进来一个非 A 股交易日（比如美国假期外的周末结算），
   整个日历会多出一天，所有股票在那天变成 NaN，全库错位。

2. **对齐只能向后填充（ffill），绝不能向前。** Polymarket 是 24/7 交易，A 股周末
   休市。周一该用的是"周一开盘前最后一个可得的报价"，用 `ffill`；若用了
   `bfill` 或插值，就是把未来信息搬到了过去。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 伪 instrument 的字段要和真股票一致，否则 dump_bin 的列集合不齐。
# volume 填 1 而非 0：0 会让任何含 volume 分母的表达式产出 inf。
_PSEUDO_DEFAULTS = {"volume": 1.0, "factor": 1.0}


def read_calendar(qlib_dir: Path, freq: str = "day") -> pd.DatetimeIndex:
    """读已建好的 qlib 交易日历。外部序列必须对齐到它。"""
    path = Path(qlib_dir) / "calendars" / f"{freq}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，请先跑 scripts/02_build_qlib_data.py")
    return pd.DatetimeIndex(pd.read_csv(path, header=None)[0].map(pd.Timestamp)).sort_values()


def align_to_calendar(
    series: pd.Series, calendar: pd.DatetimeIndex, max_stale_days: int | None = 10
) -> pd.Series:
    """把外部日频序列对齐到交易日历，只向后填充。

    Parameters
    ----------
    series : pd.Series
        DatetimeIndex 索引的外部序列（如某个 Polymarket 市场的收盘概率）。
    calendar : pd.DatetimeIndex
        目标交易日历（`read_calendar` 的输出）。
    max_stale_days : int | None
        允许的最大陈旧天数。超过这个天数没有新报价就置 NaN，避免一个早就
        没人交易的市场把最后一个价格一路拖到几个月后，制造出"信号还在更新"的假象。
        None 表示不限制。

    Returns
    -------
    pd.Series
        索引 == `calendar`，缺失处为 NaN。
    """
    s = pd.Series(series).dropna().sort_index()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        raise ValueError("外部序列为空")

    # 关键：reindex + ffill，取"当日或之前最后一个可得报价"。绝不 bfill。
    out = s.reindex(s.index.union(calendar)).ffill().reindex(calendar)

    if max_stale_days is not None:
        # 每个交易日距上一次真实报价有多久
        last_obs = pd.Series(s.index, index=s.index).reindex(
            s.index.union(calendar)
        ).ffill().reindex(calendar)
        stale = (pd.Series(calendar, index=calendar) - last_obs).dt.days
        out = out.where(stale <= max_stale_days)

    out.index.name = "date"
    return out.rename("value")


def write_external_instrument(
    name: str,
    series: pd.Series,
    per_symbol_dir: Path,
    calendar: pd.DatetimeIndex,
    max_stale_days: int | None = 10,
) -> Path:
    """把外部序列写成 `by_symbol/` 下的一个伪 instrument parquet。

    写完后重跑 `dump_to_qlib(per_symbol_dir, qlib_dir)` 即可生效。

    Parameters
    ----------
    name : str
        伪 instrument 名，建议统一加 `PM_` 前缀（如 `PM_FED_CUT`），
        这样在任何股票池里都一眼能认出它不是股票。

    Returns
    -------
    Path
        写出的 parquet 路径。
    """
    name = name.upper()
    if not name.startswith("PM_"):
        raise ValueError(f"伪 instrument 名建议以 PM_ 开头以便与股票区分，收到 {name!r}")

    aligned = align_to_calendar(series, calendar, max_stale_days=max_stale_days)
    df = pd.DataFrame({"code": name, "date": aligned.index})
    # 价格类字段全填同一个值：表达式里只会用到 $close，其余字段是为了列集合齐整
    for col in ("open", "high", "low", "close", "vwap"):
        df[col] = aligned.to_numpy()
    for col, val in _PSEUDO_DEFAULTS.items():
        df[col] = val
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{name} 对齐到交易日历后没有任何有效数据")

    per_symbol_dir = Path(per_symbol_dir)
    per_symbol_dir.mkdir(parents=True, exist_ok=True)
    out = per_symbol_dir / f"{name.lower()}.parquet"  # dump_bin 从文件名取代码
    df.to_parquet(out, index=False)
    return out


def exposure_expr(name: str, window: int = 60, kind: str = "corr") -> str:
    """生成「个股 vs 外部序列」的暴露度因子表达式。

    Parameters
    ----------
    name : str
        伪 instrument 名。
    window : int
        滚动窗口（交易日）。
    kind : {"corr", "beta", "level"}
        * `corr`  —— 滚动相关系数。无量纲、天然有界，横截面可比性最好，**推荐默认**。
        * `beta`  —— 滚动回归斜率。有量纲（受个股波动率影响），需另做标准化。
        * `level` —— 外部序列本身。**横截面上是常数，IC 贡献恒为 0**，
          仅用于对照实验，证明"直接塞宏观序列没用"。

    Returns
    -------
    str
        可直接放进 qlib 表达式列表的字符串。
    """
    stock_ret = "$close/Ref($close,1)-1"
    ext_ret = f'ChangeInstrument("{name}", $close/Ref($close,1)-1)'
    if kind == "corr":
        return f"Corr({stock_ret},{ext_ret},{window})"
    if kind == "beta":
        # Cov/Var；qlib 没有 Cov 算子，用 Corr×Std 比展开
        return (
            f"Corr({stock_ret},{ext_ret},{window})"
            f"*Std({stock_ret},{window})/Std({ext_ret},{window})"
        )
    if kind == "level":
        return f'ChangeInstrument("{name}", $close)'
    raise ValueError(f"kind 只能是 corr/beta/level，收到 {kind!r}")


def pm_feature_config(names: list[str], windows=(20, 60), kinds=("corr",)) -> tuple[list, list]:
    """批量生成 Polymarket 暴露度因子的 (fields, names)，格式同 `Alpha158DL`。

    直接和 Alpha158 的输出相加即可：

        f1, n1 = Alpha158DL.get_feature_config(cfg)
        f2, n2 = pm_feature_config(["PM_FED_CUT", "PM_TARIFF"])
        D.features(instruments, f1 + f2 + [label], ...)
    """
    fields, cols = [], []
    for nm in names:
        for kind in kinds:
            for w in windows:
                fields.append(exposure_expr(nm, w, kind))
                cols.append(f"{nm}_{kind.upper()}{w}")
    return fields, cols


def sanity_check(aligned: pd.Series, calendar: pd.DatetimeIndex) -> dict:
    """对齐结果的体检报告，注册前先看一眼。"""
    valid = aligned.dropna()
    return {
        "n_calendar_days": len(calendar),
        "n_valid": len(valid),
        "coverage": float(len(valid) / len(calendar)) if len(calendar) else float("nan"),
        "first_valid": valid.index.min() if len(valid) else None,
        "last_valid": valid.index.max() if len(valid) else None,
        "value_min": float(valid.min()) if len(valid) else float("nan"),
        "value_max": float(valid.max()) if len(valid) else float("nan"),
        # 概率序列若全程几乎不动，暴露度因子会退化成噪声
        "n_distinct": int(valid.nunique()),
        "daily_change_std": float(valid.diff().std()) if len(valid) > 2 else float("nan"),
        "index_outside_calendar": int(
            len(pd.DatetimeIndex(np.asarray(aligned.index)).difference(calendar))
        ),
    }
