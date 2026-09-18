"""模型：LightGBM（主）与 Ridge（线性下限对照）。

## 为什么早停指标必须是「验证集 IC」而不是 MSE

标签做过每日横截面 z-score 后，`min MSE = 1 - IC²`，两者在**理论最优点**是等价的。
但训练过程中不是：MSE 还包含"预测幅度校准"这一项，模型可能靠把预测整体缩小来降
MSE 而 IC 毫无改善。既然我们只关心 IC，就直接用 IC 做早停，不要绕一层。

## 为什么要有 Ridge 这条线

Ridge 是**线性下限**。如果 LightGBM 赢不了 Ridge 多少，说明 216 个因子里的信号
基本是线性的，非线性容量没带来增益。这条对照在后面判断
「Polymarket 的增量到底来自新信息，还是只是把模型做大了」时是必需的——
没有它，一个更强的模型带来的提升会被误读成新因子有效。

## 特征标准化只在训练段 fit

`RobustZScore` 用 median/MAD（比 mean/std 抗极端值），**统计量只能在训练段上估计**。
用全样本估计就是前视偏差：测试期的极端值会影响到训练期样本的缩放。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MAD_TO_STD = 1.4826  # 正态分布下 std ≈ 1.4826 × MAD


class RobustZScore:
    """MAD 版 z-score + 截尾。`fit` 只能喂训练段。"""

    def __init__(self, clip: float = 3.0):
        self.clip = clip
        self.center_: pd.Series | None = None
        self.scale_: pd.Series | None = None

    def fit(self, x: pd.DataFrame) -> "RobustZScore":
        self.center_ = x.median()
        mad = (x - self.center_).abs().median()
        scale = mad * _MAD_TO_STD
        # 常数列（MAD=0）会除出 inf，置 1 等价于"该列标准化后恒为 0"
        self.scale_ = scale.replace(0.0, np.nan).fillna(1.0)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.center_ is None:
            raise RuntimeError("先 fit 再 transform")
        z = (x - self.center_) / self.scale_
        return z.clip(-self.clip, self.clip).fillna(0.0)

    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.fit(x).transform(x)


def _daily_ic(pred: np.ndarray, label: np.ndarray, day_codes: np.ndarray) -> float:
    """按日分组的平均 Pearson IC。给 LightGBM 的 feval 用，走 numpy 求速度。"""
    order = np.argsort(day_codes, kind="stable")
    p, y, d = pred[order], label[order], day_codes[order]
    bounds = np.flatnonzero(np.diff(d)) + 1
    ics = []
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, d.size]):
        pp, yy = p[lo:hi], y[lo:hi]
        if pp.size < 5:
            continue
        sp, sy = pp.std(), yy.std()
        if sp <= 0 or sy <= 0:
            continue
        ics.append(float(((pp - pp.mean()) * (yy - yy.mean())).mean() / (sp * sy)))
    return float(np.mean(ics)) if ics else 0.0


DEFAULT_LGB_PARAMS = {
    # 前 7 项照抄 qlib 自带的参考配置 `qlib/tests/config.py:GBDT_MODEL`
    # （其中 colsample_bytree=0.8879 / subsample=0.8789 是 feature_fraction /
    #  bagging_fraction 的别名）。注意 qlib 那份配置里**没有** bagging_freq，
    # 而原生 LightGBM 在 bagging_freq=0 时 bagging_fraction 完全不生效——
    # 也就是说 qlib 参考配置里的 subsample 实际是空转的。这里显式补上 bagging_freq=5
    # 让它真正起作用，属于我们的改动而非官方配置。
    "objective": "mse",
    "learning_rate": 0.0421,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "feature_fraction": 0.8879,
    "bagging_fraction": 0.8789,
    "bagging_freq": 5,       # 我们补的（见上）
    "min_data_in_leaf": 100,  # 我们补的：4000+ 天 × 1500 股，叶子太小会过拟合
    "num_threads": 20,
    "verbose": -1,
    "seed": 0,
}


def train_lgb(
    xtr: pd.DataFrame,
    ytr: pd.Series,
    xva: pd.DataFrame,
    yva: pd.Series,
    params: dict | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
):
    """训练 LightGBM，**以验证集日度 IC 早停**。

    Returns
    -------
    (model, info)
        `info` 含 `best_iteration` 与 `best_valid_ic`。
    """
    import lightgbm as lgb

    p = {**DEFAULT_LGB_PARAMS, **(params or {})}
    p["metric"] = "None"  # 关掉内置 l2，让 IC 成为唯一早停依据

    va_days = pd.factorize(xva.index.get_level_values("datetime"))[0]
    yva_np = yva.to_numpy(dtype=float)

    def feval(preds, _eval_data):
        return "daily_ic", _daily_ic(np.asarray(preds, dtype=float), yva_np, va_days), True

    dtr = lgb.Dataset(xtr, label=ytr)
    dva = lgb.Dataset(xva, label=yva, reference=dtr)
    evals: dict = {}
    model = lgb.train(
        p,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=[dva],
        valid_names=["valid"],
        feval=feval,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
            lgb.record_evaluation(evals),
        ],
    )
    hist = evals.get("valid", {}).get("daily_ic", [])
    return model, {
        "best_iteration": int(model.best_iteration or num_boost_round),
        "best_valid_ic": float(hist[model.best_iteration - 1]) if hist and model.best_iteration else float("nan"),
    }


RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def train_ridge(
    xtr: pd.DataFrame,
    ytr: pd.Series,
    xva: pd.DataFrame,
    yva: pd.Series,
    alphas=RIDGE_ALPHAS,
):
    """Ridge 线性对照，正则强度按**验证集 IC** 选（不是按 MSE 或 R²）。"""
    from sklearn.linear_model import Ridge

    va_days = pd.factorize(xva.index.get_level_values("datetime"))[0]
    yva_np = yva.to_numpy(dtype=float)

    best, best_ic, best_alpha = None, -np.inf, None
    for a in alphas:
        m = Ridge(alpha=a, fit_intercept=True)
        m.fit(xtr.to_numpy(dtype=np.float32), ytr.to_numpy(dtype=float))
        ic = _daily_ic(m.predict(xva.to_numpy(dtype=np.float32)), yva_np, va_days)
        if ic > best_ic:
            best, best_ic, best_alpha = m, ic, a
    return best, {"alpha": best_alpha, "best_valid_ic": float(best_ic)}


def predict(model, x: pd.DataFrame) -> pd.Series:
    """统一的预测出口，返回 `MultiIndex[datetime, instrument]` 的 Series。"""
    import lightgbm as lgb

    if isinstance(model, lgb.Booster):
        raw = model.predict(x, num_iteration=model.best_iteration or None)
    else:
        raw = model.predict(x.to_numpy(dtype=np.float32))
    return pd.Series(np.asarray(raw, dtype=float), index=x.index, name="score")


def feature_importance(model, feat_cols: list[str]) -> pd.Series:
    """特征重要性（LightGBM 用 gain，Ridge 用 |系数|）。"""
    import lightgbm as lgb

    if isinstance(model, lgb.Booster):
        imp = model.feature_importance(importance_type="gain")
        return pd.Series(imp, index=model.feature_name()).sort_values(ascending=False)
    return pd.Series(np.abs(model.coef_), index=feat_cols).sort_values(ascending=False)
