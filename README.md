# Polymarket-Extended-Stock-Prediction

探究 **Polymarket 预测市场数据能否作为新因子改善 A 股 5 日收益预测**。

当前阶段是 **baseline：只用 OHLCV、215 个量价因子、预测 5 日收益**。
Polymarket 因子的接入点已经建好并验证通过（见下方"Polymarket 怎么接进来"），
但还没有接真实的 Polymarket 数据。

    输入   [date, instruments, features]    215 个 Alpha158 因子
    输出   [date, instruments, 1]           5 日收益预测分
    主指标  IC（Pearson），日度序列 + Newey-West 修正的 t 值

---

## 快速开始

```bash
conda create -n pmsp python=3.12 -y && conda activate pmsp
pip install -r requirements.txt

# 0. 不需要 token、不碰网络，先确认整条链路是通的（约 1 分钟，47 项检查）
python scripts/99_selftest.py

# 1. 数据（**不需要任何 token / 账号**，主数据源是新浪财经的公开接口）
python scripts/01_download.py              # 全市场日线 2010→2026，约 5 小时，可断点续传
python scripts/02_build_qlib_data.py       # 复权 + 股票池 + 转 qlib 二进制

# 2. 训练与评估
python scripts/05_verify.py                # 先验证（前视/标签口径/评估代码）
python scripts/03_run_workflow.py          # walk-forward，产出样本外预测
python scripts/04_report.py                # 出 markdown 报告
```

`scripts/99_selftest.py` 用合成数据（含一次 10 送 10 拆股和一个人为埋入的反转信号）
把 **复权 → 股票池 → dump_bin → qlib 表达式 → walk-forward → IC → 回测 → Polymarket 接入**
全走一遍。**不需要 token**，所以拿数据之前就能确认代码没问题。

### 数据源为什么是新浪，不是 tushare

方案原本走 tushare 免费档，实测走不通：本机 token（120 积分）**`daily` 与
`bak_daily` 均无访问权限**；有权限的 `stock_basic` / `daily_basic` / `adj_factor`
配额是 **1 次/小时**，批量下载无从谈起。

换成新浪财经的公开 K 线接口，反而多拿到一样东西、少拿到一样东西：

| | tushare 免费档（原方案） | 新浪（现在） |
|---|---|---|
| 复权 | 从 `pct_chg` 累乘反推 | **官方口径的后复权累计因子**（`hfq.js`），更硬 |
| 退市股票 | 从各交易日 ts_code 并集推导 | 上交所/深交所官方退市名单，270 只（2010 后） |
| 取数方式 | 按日期，一次全市场 | 按股票，一次全历史（2 个请求/只） |
| 成交额 `amount` | 有 | **没有** → 用 `close × volume` 代理 |
| `$vwap` | 由 `amount/vol` 算出 | **算不出** → Alpha158 少 `VWAP0` 一个因子，216 → 215 |
| token | 需要 | 不需要 |

速率上限实测过：**0.5s 间隔连发会触发 HTTP 456**（新浪的限流码，封 IP 约
273 秒）。所以 `configs/data.yaml` 里设 40 次/分（1.5s 匀速），且 `_get` 对
456 单独用 330s 冷却、不计入常规重试次数。

---

## 关键设计决策

| 决策 | 理由 |
|---|---|
| **用 qlib，不自研因子库** | Alpha158 现成、有 py3.12 wheel、`ChangeInstrument` 算子原生支持"个股 vs 外部时序"——Polymarket 暴露度因子不需要改 qlib 一行代码 |
| **标签 = `Ref($close,-6)/Ref($close,-1)-1`** | T+1 收盘买、T+6 收盘卖。不用 `Ref($close,-5)/$close-1`：那是 T 日收盘算因子、T 日收盘成交，实盘不可执行，会系统性高估 |
| **标签做每日横截面 z-score** | 之后 `objective=mse` 就**严格等价于**最大化 IC（`min MSE = 1-IC²`）。不做这步而直接拟合原始收益，模型会被高波动日主导去拟合大盘涨跌——那是横截面常数项，对 IC 贡献恰为零 |
| **早停指标用验证集 IC，不用 MSE** | MSE 还含"预测幅度校准"一项，模型可能靠把预测整体缩小来降 MSE 而 IC 毫无改善 |
| **股票池 = 每月末流动性前 N，不是沪深300** | 拿不到免费的指数成分股历史。用**当前**成分股回溯是幸存者偏差+前视偏差双重污染，比近似更糟 |
| **下载列表 = 在市 ∪ 官方退市名单** | 只下在市股票就是幸存者偏差。退市名单只剔除"2010 前就退市的"——一只 2012 年退市的股票在 2010–2012 属于当时的股票池 |
| **复权用新浪的官方累计因子** | 除权日是官方给的，不用从收益率序列里猜。代码同时保留 `cumprod(1+pct_chg)` 那条路，自测里**两条路互相验证**收益率逐点一致 |
| **扩张窗口 walk-forward，不是固定切分** | 215 个因子的模型吃数据；且 Polymarket 阶段的训练集和测试集必须在 2024+ 这同一段时间里共存 |
| **embargo 5 天，train↔valid 和 valid↔test 各一处** | 5 日标签让相邻交易日样本共用 4 天收益。少留一处，验证集分数就虚高、早停停得太晚 |
| **Ridge 作为线性下限对照** | 将来判断"Polymarket 的增量到底来自新信息还是只是模型变大了"时必需 |

## 目录

```
configs/data.yaml       数据源、时间范围、股票池、切分、成本  ← 唯一的事实来源
configs/model.yaml      因子窗口、模型超参、walk-forward 参数

src/pmsp/
  datasource/sina_daily.py      主数据源：新浪日线 + 后复权因子（限速/456 冷却/续传）
  datasource/exchange_lists.py  上交所/深交所官方退市名单（防幸存者偏差）
  datasource/tushare_daily.py   备用源（本机 token 无 daily 权限，保留供有权限者用）
  datasource/eastmoney.py       可选的复权交叉校验源
  build/adjust.py        未复权价 × 复权因子 → 后复权（含成交量复权不变量）
  build/universe.py      动态流动性股票池 → qlib instruments 文件
  build/to_qlib.py       parquet → dump_bin
  eval/ic.py             IC/RankIC/ICIR/Newey-West t 值/配对差值检验/功效测算
  eval/backtest.py       分层回测、多空、换手、成本敏感性
  model/dataset.py       特征矩阵构建与缓存
  model/models.py        LightGBM（IC 早停）、Ridge、RobustZScore
  model/walkforward.py   滚动切分与重训   ← 最容易出前视偏差的地方，逻辑集中在这里
  extensions/external_series.py   外部时序 → 伪 instrument（Polymarket 接入点）

scripts/00..05          按序号跑
scripts/99_selftest.py  合成数据端到端自测（不需要 token）
```

---

## Polymarket 怎么接进来

**先说一个容易踩的坑**：Polymarket 的市场都是全球宏观事件（降息、大选、关税）。
一个宏观概率序列在任一天对所有股票都是同一个数，**横截面标准化之后恒等于零**，
直接当因子塞进去对 IC 的贡献严格为 0。自测第 8 节把这一点验证出来了
（宏观序列本身的日均横截面标准差 = 0.00e+00）。

有意义的用法是「**外部时序 × 个股暴露度**」的交叉因子：

```python
from pmsp.extensions.external_series import read_calendar, write_external_instrument, pm_feature_config
from pmsp.build.to_qlib import dump_to_qlib

cal = read_calendar("data/qlib_cn")
write_external_instrument("PM_FED_CUT", fed_cut_prob_series, "data/raw/by_symbol", cal)
dump_to_qlib("data/raw/by_symbol", "data/qlib_cn")     # 注册成伪 instrument

fields, names = pm_feature_config(["PM_FED_CUT"], windows=(20, 60))
# -> Corr($close/Ref($close,1)-1, ChangeInstrument("PM_FED_CUT", $close/Ref($close,1)-1), 60)
```
再把 `fields/names` 传给 `build_feature_matrix(extra_fields=..., extra_names=...)`，
下游代码完全不用改。

外部序列会先对齐到 A 股交易日历，**只向后填充**（Polymarket 7×24 交易、A 股周末休市，
用 bfill 或插值就是把未来信息搬到过去），并且超过 `max_stale_days` 没有新报价就置 NaN，
避免一个冷掉的市场把最后一个价格一路拖几个月。

### 增益检验的三条路径（按顺序做）

1. **Fama-MacBeth 增量检验** —— 每日横截面回归 `ret_{t+5} ~ a + b·baseline预测 + c·PM因子`，
   看 `c` 的时序均值与 Newey-West t 值。**不需要训练、不消耗测试窗、样本效率最高，应第一个做。**
   若 `c` 完全不显著，后面的模型对比大概率白做。
2. **两段式残差建模**（主力）—— baseline 用全历史训练出 `f_base`；第二层小模型只在 PM 期训练，
   输入 `[f_base, PM因子]`。第二层必须很小（Ridge 或强正则浅 LightGBM）：
   12 个月预热 ≈ 72 个独立 5 日区间，容不下大模型。
3. **匹配样本对照**（稳健性）—— 两模型都只用 2024+ 训练，改 `--train-start` 即可。

判决方式是**逐日 IC 差值的配对 t 检验**（`pmsp.eval.ic.paired_ic_test`），
不是分别报两个 IC 均值再肉眼比大小——那会丢掉配对带来的方差削减，白扔一半统计功效。
baseline 在锁死窗口的日度 IC 序列已存档在 `data/predictions/ic_*.parquet`，
到时候直接读，不需要重训 baseline。

### 事前就该认下来的功效上限

锁死的对比窗口是 2025-01→2026-09，约 425 个交易日、有效样本约 85 个独立 5 日区间。
双边 α=0.05、power=80% 下**可检出的最小 ΔIC ≈ 0.008**。

若 Polymarket 的真实增益有 0.008 以上，这个设计能证明；**若只有 0.003，无论怎么做都证不出来**。
这不是方法问题，是 Polymarket 历史长度的物理上限（2024 年美国大选才真正爆发）。

### 预注册

`top_n ∈ {1500, 300}` 两个股票池**实施前就写死**，1500 为主检验、300 为第二窗口
（大盘股对全球宏观更敏感，可能信噪比更高；但 1500 的 IC 抽样噪声更小）。
两个都报并声明多重比较——事后挑一个好看的就是 p-hacking。

---

## 已知局限

免费数据源只给 OHLCV，以下几项因此做不了。前四项买 tushare 2000 积分
（200 元/年）可全部解决，schema 已预留字段：

1. **无市值、无行业分类** → 没做市值/行业中性化，模型可能部分在赚小盘股溢价
2. **无历史股票名称** → 无法精确剔除历史 ST
3. **无指数成分股历史** → 股票池是"流动性前 N"而非真正的沪深300/中证500
4. **无成交额** → 流动性排名用 `close × volume` 代理（真实值是 `vwap × volume`，
   两者只差 `vwap/close`，对**排名**影响可忽略，但要记着这不是真的成交额）
5. **无 `$vwap`** → Alpha158 的 `VWAP0` 拿不到，因子数 215 而非 216。
   不拿 `(H+L+C)/3` 之类的"典型价"冒充：那是 HLC 的线性组合，与
   `KMID/KLEN/KSFT` 高度重复，等于凭空造一个假因子
6. 涨跌停用 `pct_chg` 阈值近似（按主板 10% / 创业板科创板 20% / 北交所 30%
   分段，含 2020-08 与 2019-07 两次规则变更），停牌由"日期序列有缺口"推导
7. **北交所股票的历史含新三板时期**。新浪对这段返回的价格可能全为 0（协议
   转让没有连续竞价价），复权阶段已剔除；未被剔除的部分微观结构与竞价市场
   不同。实际影响很小——北交所成交额远低于主板，进不了流动性前 N 的股票池

复权正确性的验证：`scripts/05_verify.py` 的第 1 项是**自洽检验**，不依赖第二个
数据源（本机东财批量接口连发 6 次即封 IP、tushare 无 `daily` 权限，把这条验证
押在外部源上等于形同虚设）。三条合起来直接检验复权该干的事：因子 ≥1 且非递减、
除权日的未复权跳空被抹平、非除权日逐点不变。外部交叉校验作为可选项保留
（`--only adjust`），打不通就跳过、不算失败。

另：`scripts/05_verify.py --only benchmark` 与 qlib 官方基准的对照是**量级对照而非逐项复现**
（官方用真沪深300 成分股 + 20 个随机种子均值，我们用 top300 近似 + 1 个种子）。
它能发现"差一个数量级"这类严重错误，发现不了小偏差。
