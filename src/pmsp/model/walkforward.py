"""滚动重训（walk-forward）。**全项目最容易出前视偏差的地方，切分逻辑集中在这里。**

## 一次滚动窗口的时间轴

    ┌──────── train ────────┐ E ┌─ valid ─┐ E ┌──── test ────┐
    2010-01 ............. t1   t2 ..... t3   t4 ........... t5
                              (E = embargo，≥5 天)

* `test`  —— 本窗口的样本外预测区间，长度 = 重训频率（如 6 个月 / 1 个月）
* `valid` —— 紧贴 test 之前的一段，仅用于早停和选正则强度
* `train` —— 从数据起点一路到 valid 之前（**扩张窗口**，不是固定长度）

## 两处 embargo 都是必需的，少一处就虚高

5 日标签让相邻交易日的样本共用 4 天的收益，高度相关。

* `valid` 与 `test` 之间不留空档 → 早停会在"和测试集几乎同一批收益"上选轮数，
  等于偷看测试集
* `train` 与 `valid` 之间不留空档 → 验证集分数虚高，早停停得太晚（过拟合）

## 为什么用扩张窗口而不是固定长度窗口

216 个因子的模型吃数据。固定 3 年滚动窗口在 2010–2026 上只有约 730 个交易日、
折算约 146 个独立 5 日区间，撑不住这个维度。扩张窗口让后期窗口能用上全部历史。
代价是早期窗口和后期窗口的训练集长度不同，报告里按窗口分别记录训练集规模。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pmsp.model.models import RobustZScore, predict, train_lgb, train_ridge


@dataclass
class Split:
    """一个滚动窗口的日期边界（左闭右闭）。"""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> str:
        return (
            f"train {self.train_start:%Y-%m-%d}→{self.train_end:%Y-%m-%d} | "
            f"valid {self.valid_start:%Y-%m-%d}→{self.valid_end:%Y-%m-%d} | "
            f"test {self.test_start:%Y-%m-%d}→{self.test_end:%Y-%m-%d}"
        )


def generate_splits(
    dates: pd.DatetimeIndex,
    oos_start: str,
    oos_end: str,
    retrain_months: int,
    embargo_days: int = 5,
    valid_days: int = 240,
    min_train_days: int = 500,
    train_start: str | None = None,
) -> list[Split]:
    """按交易日历生成滚动窗口。

    Parameters
    ----------
    dates : pd.DatetimeIndex
        **交易日**序列（用日历日会让 embargo 天数算错）。
    retrain_months : int
        重训频率（月）。baseline 长 OOS 用 6，PM 对比窗用 1。
    embargo_days : int
        train/valid 与 valid/test 之间各留多少个**交易日**。5 日标签下取 5。
    valid_days : int
        验证集长度（交易日）。240 ≈ 1 年。
    min_train_days : int
        训练集不足这么多交易日就跳过该窗口，避免开头几个窗口在极少数据上训练。
    train_start : str | None
        训练集起点。None = 用数据起点（扩张窗口）。
        设成 `oos_start` 附近的日期即可做方案里的"匹配样本对照"。

    Returns
    -------
    list[Split]
        各窗口的 test 区间**首尾相接且互不重叠**，拼起来正好覆盖 [oos_start, oos_end]。
    """
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    oos_start, oos_end = pd.Timestamp(oos_start), pd.Timestamp(oos_end)
    lo = int(np.searchsorted(dates, oos_start, side="left"))
    hi = int(np.searchsorted(dates, oos_end, side="right")) - 1
    if lo > hi:
        raise ValueError(f"OOS 区间 {oos_start:%Y-%m-%d}→{oos_end:%Y-%m-%d} 内没有交易日")

    ts_start = pd.Timestamp(train_start) if train_start else dates[0]

    splits: list[Split] = []
    cur = lo
    while cur <= hi:
        test_start = dates[cur]
        # 本窗口 test 结束于 test_start + retrain_months 个月后的前一个交易日
        nxt = test_start + pd.DateOffset(months=retrain_months)
        nxt_pos = int(np.searchsorted(dates, nxt, side="left"))
        test_end_pos = min(nxt_pos - 1, hi)
        if test_end_pos < cur:  # 该月无交易日，直接前进
            cur = nxt_pos
            continue

        valid_end_pos = cur - 1 - embargo_days
        valid_start_pos = valid_end_pos - valid_days + 1
        train_end_pos = valid_start_pos - 1 - embargo_days
        train_start_pos = int(np.searchsorted(dates, ts_start, side="left"))

        n_train = train_end_pos - train_start_pos + 1
        if valid_start_pos <= train_start_pos or n_train < min_train_days:
            cur = test_end_pos + 1
            continue

        splits.append(
            Split(
                train_start=dates[train_start_pos],
                train_end=dates[train_end_pos],
                valid_start=dates[valid_start_pos],
                valid_end=dates[valid_end_pos],
                test_start=test_start,
                test_end=dates[test_end_pos],
            )
        )
        cur = test_end_pos + 1
    return splits


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    d = df.index.get_level_values("datetime")
    return df[(d >= start) & (d <= end)]


@dataclass
class WalkForwardResult:
    pred: pd.Series
    label: pd.Series
    log: pd.DataFrame
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_walkforward(
    df: pd.DataFrame,
    feat_cols: list[str],
    splits: list[Split],
    model: str = "lgb",
    label_col: str = "LABEL0",
    lgb_params: dict | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """跑完整条 walk-forward，拼出一条连续的样本外预测序列。

    Parameters
    ----------
    model : {"lgb", "ridge"}

    Returns
    -------
    WalkForwardResult
        `pred` / `label` 索引对齐，可直接喂 `pmsp.eval.ic.calc_ic_series`。
        `log` 每窗口一行，记录训练集规模、最优轮数、验证集 IC。
    """
    if model not in ("lgb", "ridge"):
        raise ValueError(f"model 只支持 lgb/ridge，收到 {model!r}")

    preds, logs, imps = [], [], []
    for i, sp in enumerate(splits, 1):
        tr = _slice(df, sp.train_start, sp.train_end)
        va = _slice(df, sp.valid_start, sp.valid_end)
        te = _slice(df, sp.test_start, sp.test_end)
        if tr.empty or va.empty or te.empty:
            if verbose:
                print(f"  [跳过] 窗口 {i}: 有区间为空 — {sp.describe()}")
            continue

        # 标准化统计量**只在训练段 fit**，再套用到 valid/test
        scaler = RobustZScore().fit(tr[feat_cols])
        xtr, xva, xte = (scaler.transform(d[feat_cols]) for d in (tr, va, te))

        if model == "lgb":
            m, info = train_lgb(xtr, tr[label_col], xva, va[label_col], params=lgb_params)
        else:
            m, info = train_ridge(xtr, tr[label_col], xva, va[label_col])

        p = predict(m, xte)
        preds.append(p)

        from pmsp.model.models import feature_importance

        imps.append(feature_importance(m, feat_cols).rename(f"w{i}"))
        logs.append(
            {
                "window": i,
                "train_start": sp.train_start, "train_end": sp.train_end,
                "valid_start": sp.valid_start, "valid_end": sp.valid_end,
                "test_start": sp.test_start, "test_end": sp.test_end,
                "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
                "n_train_days": tr.index.get_level_values("datetime").nunique(),
                **info,
            }
        )
        if verbose:
            print(f"  窗口 {i}/{len(splits)}  {sp.describe()}  "
                  f"n_train={len(tr):,}  valid_IC={info['best_valid_ic']:.4f}")

    if not preds:
        raise RuntimeError("没有任何窗口成功训练，检查 splits 与数据日期范围是否匹配")

    pred = pd.concat(preds).sort_index()
    # 窗口的 test 区间互不重叠，若这里有重复索引说明切分逻辑出了问题
    dup = int(pred.index.duplicated().sum())
    if dup:
        raise RuntimeError(f"样本外预测出现 {dup} 个重复 (date, code)，切分区间重叠了")

    return WalkForwardResult(
        pred=pred,
        label=df[label_col].reindex(pred.index),
        log=pd.DataFrame(logs),
        importance=pd.concat(imps, axis=1) if imps else pd.DataFrame(),
    )
