#!/usr/bin/env python
"""跑 walk-forward，产出样本外预测与日度 IC 序列。

两个窗口（配置见 configs/data.yaml 的 split.*）：

* `baseline`   —— 2018-01→2026-09，每 6 个月重训。评的是"baseline 本身行不行"
* `pm_compare` —— 2025-01→2026-09，每月重训。**锁死窗口，绝不用于调参。**
                  日度 IC 序列会落盘存档，以后做 Polymarket 配对检验时直接读，
                  不需要重训 baseline。

用法：
    python scripts/03_run_workflow.py                        # 默认 top1500 + lgb + 两个窗口
    python scripts/03_run_workflow.py --universe top300      # 预注册的第二股票池
    python scripts/03_run_workflow.py --model ridge          # 线性下限对照
    python scripts/03_run_workflow.py --window pm_compare    # 只跑锁死窗口
    python scripts/03_run_workflow.py --train-start 2024-01-01   # 匹配样本对照
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pmsp.config import abs_path, load_config, load_model_config
from pmsp.eval.ic import calc_ic_series, ic_summary
from pmsp.model.dataset import build_feature_matrix, daily_returns
from pmsp.model.walkforward import generate_splits, run_walkforward


def run_one_window(
    df: pd.DataFrame,
    feat_cols: list[str],
    dates: pd.DatetimeIndex,
    window: str,
    cfg,
    mcfg,
    model: str,
    universe: str,
    train_start: str | None,
    out_dir: Path,
) -> dict:
    sp_cfg = cfg.split[window]
    oos_start = sp_cfg["oos_start"]
    oos_end = sp_cfg.get("oos_end", cfg.data["end_date"])
    retrain = sp_cfg["retrain_freq_months"]
    wf = mcfg.walkforward

    splits = generate_splits(
        dates,
        oos_start=oos_start,
        oos_end=oos_end,
        retrain_months=retrain,
        embargo_days=cfg.split["embargo_days"],
        valid_days=wf["valid_days"],
        min_train_days=wf["min_train_days"],
        train_start=train_start or wf["train_start"],
    )
    print(f"\n=== 窗口 `{window}`：{oos_start} → {oos_end}，每 {retrain} 个月重训，"
          f"共 {len(splits)} 个滚动窗口 ===")

    lgb_params = dict(mcfg.models["lgb"]) if model == "lgb" else None
    if lgb_params:
        lgb_params.pop("num_boost_round", None)
        lgb_params.pop("early_stopping_rounds", None)

    t0 = time.time()
    res = run_walkforward(df, feat_cols, splits, model=model, lgb_params=lgb_params)
    print(f"  训练完成，用时 {(time.time() - t0) / 60:.1f} 分钟")

    ic, ric = calc_ic_series(res.pred, res.label)
    summ = ic_summary(ic, ric, horizon=cfg.label["horizon"])
    print(f"  样本外 IC={summ['ic_mean']:.4f}  ICIR={summ['icir']:.3f}  "
          f"t_NW={summ['t_nw']:.2f}  RankIC={summ['ric_mean']:.4f}  "
          f"IC胜率={summ['ic_win_rate']:.1%}  ({summ['n_days']} 个交易日)")
    if summ["ic_ric_gap_warn"]:
        print("  [告警] IC 与 RankIC 差距偏大，Pearson IC 可能被少数极端收益个股主导")

    # 存档：预测、日度 IC、训练日志、特征重要性。
    # PM 阶段的配对检验直接读这里的 ic 序列，不需要重训 baseline。
    tag = f"{universe}_{model}_{window}" + (f"_ts{train_start}" if train_start else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    res.pred.to_frame("score").to_parquet(out_dir / f"pred_{tag}.parquet")
    pd.DataFrame({"ic": ic, "rank_ic": ric}).to_parquet(out_dir / f"ic_{tag}.parquet")
    res.log.to_csv(out_dir / f"log_{tag}.csv", index=False)
    if not res.importance.empty:
        res.importance.mean(axis=1).sort_values(ascending=False).to_csv(
            out_dir / f"importance_{tag}.csv", header=["mean_gain"]
        )
    print(f"  存档 -> {out_dir}/*_{tag}.*")
    return {"window": window, "tag": tag, **summ}


def main() -> int:
    cfg, mcfg = load_config(), load_model_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=f"top{cfg.universe['top_n_list'][0]}")
    ap.add_argument("--model", default="lgb", choices=["lgb", "ridge"])
    ap.add_argument("--window", default="all", choices=["all", "baseline", "pm_compare"])
    ap.add_argument("--train-start", default=None,
                    help="限制训练集起点，用于'两模型都只用 PM 期训练'的匹配样本对照")
    ap.add_argument("--force-rebuild", action="store_true", help="忽略特征缓存重新求值")
    args = ap.parse_args()

    qlib_dir = cfg.path("data.qlib_dir")
    if not (qlib_dir / "calendars" / "day.txt").exists():
        raise SystemExit(f"{qlib_dir} 里没有 qlib 数据，请先跑 scripts/02_build_qlib_data.py")

    print(f"[1/3] 构建特征矩阵（{args.universe}，Alpha158 窗口 {mcfg.features['windows']}）...")
    df, feat_cols = build_feature_matrix(
        qlib_dir=qlib_dir,
        universe=args.universe,
        label_expr=cfg.label["expr"],
        start_date=cfg.data["start_date"],
        end_date=cfg.data["end_date"],
        windows=mcfg.features["windows"],
        cache_dir=abs_path(mcfg.output["cache_dir"]),
        force=args.force_rebuild,
    )
    dates = pd.DatetimeIndex(df.index.get_level_values("datetime").unique()).sort_values()
    print(f"      {df.shape[0]:,} 行 × {len(feat_cols)} 个因子，"
          f"{dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d}（{len(dates)} 个交易日）")

    print("\n[2/3] 跑 walk-forward ...")
    windows = ["baseline", "pm_compare"] if args.window == "all" else [args.window]
    out_dir = abs_path(mcfg.output["pred_dir"])

    summaries = []
    for w in windows:
        summaries.append(
            run_one_window(df, feat_cols, dates, w, cfg, mcfg, args.model,
                           args.universe, args.train_start, out_dir)
        )

    print("\n[3/3] 落盘单日收益（回测用）...")
    ret_path = out_dir / f"ret1d_{args.universe}.parquet"
    if not ret_path.exists():
        ret = daily_returns(qlib_dir, args.universe,
                            cfg.data["start_date"], cfg.data["end_date"])
        ret.to_frame().to_parquet(ret_path)
        print(f"      -> {ret_path}")
    else:
        print(f"      已存在，跳过：{ret_path}")

    (out_dir / f"summary_{args.universe}_{args.model}.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("\n完成。下一步：python scripts/04_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
