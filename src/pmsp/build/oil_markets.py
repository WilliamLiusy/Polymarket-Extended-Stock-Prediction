"""从 Polymarket 全市场目录里挑出与原油有关的市场。

## 先说结论：Polymarket 上几乎没有"油价水平"市场

抽样普查（2024–2025 的 8 个交易日、19,692 个市场）后，按 `oil|crude|wti|brent|
opec|...` 命中的 46 个市场里，成交额排前几的真实构成是：

    will-iran-close-the-strait-of-hormuz-before-july      $739,214   ← 最大
    will-iran-close-the-strait-of-hormuz-in-2025          $180,163
    israel-strikes-iranian-oil-in-june                     $19,378
    will-iran-strike-gulf-oil-facilities-before-july        $15,723
    will-the-kharg-island-oil-terminal-be-hit-in-june        $7,834
    will-opec-hike-production-by-next-meeting                  $410
    will-opec-cut-production-by-next-meeting                    $45

也就是说：**油相关的流动性全部集中在"地缘供给中断风险"，而不是价格水平**，
而且 OPEC 产量决议这种直接的油价驱动市场成交额只有三位数，等于没有。

这件事必须写在最前面，因为它决定了分析对象：本分析检验的是
**「原油供给中断概率」↔ 航空股**，不是「油价预期」↔ 航空股。两者不是一回事——
供给风险概率上升同时意味着油价上行压力**与**全局风险偏好下降，后者对航空股
也是利空，所以三角验证（`pmsp.eval.assoc.triangle`）不是可选项。

## 关键词匹配会踩的坑

子串匹配在这个数据集上假阳性极多，且全是高成交额的体育/政治市场，不剔掉
会直接污染结果：

    Edmonton **Oilers**（NHL）            "oil" 的子串，$42,231
    Pierre **Poilievre**（加拿大政治）      "oil" 的子串，$34,931
    **Brent**ford FC（英超）               "brent" 的子串，$2,394
    Cracker **Barrel** CEO                "barrel" 的子串，$549
    Secretary of **Energy** 人事           "energy" 命中，但是人事任命
    "will-powell-say-**energy**-during-..."  美联储发布会用词竞猜

所以流程是三步：**宽召回 -> 硬排除 -> 人工归类**，且每一步的结果都落表，
让人能逐条复核（`audit_table`）。不做成"一个正则解决"是因为这里没有能同时
保证召回与精度的正则——`oilers` 与 `oil` 的区别是语义的，不是字符的。
"""

from __future__ import annotations

import re

import pandas as pd

#: 第一步：宽召回。宁滥勿缺，精度交给下一步。
#:
#: `oil` 前面的 `(?<![a-z])` 不是可以省的讲究。裸的 `oil` 会命中
#: **soil / turmoil / broil / spoil / espoiled**，实测在全量目录里多召回
#: 47 个市场、4534 万 USDC 的纯噪声，其中 `us-anti-cartel-operation-on-
#: foreign-soil` 一家就是审计表里成交额第四大的"原油市场"。
#: 同理 `energy` 要挡掉 `tonalenergy`（一个节拍器 App）。
INCLUDE_PATTERNS = [
    r"(?<![a-z])oil", r"crude", r"\bwti\b", r"brent", r"opec", r"petrol",
    r"gasoline", r"barrel", r"refin", r"hormuz", r"pipeline", r"aramco",
    r"(?<![a-z])energy", r"tanker", r"kharg", r"lng", r"natural-?gas",
    r"diesel", r"jet-?fuel",
]

#: 第二步：硬排除。每一条都对应一个实际命中的假阳性，注释写的是它为什么中招。
EXCLUDE_PATTERNS = [
    (r"oilers?\b", "NHL 埃德蒙顿油人队 Edmonton Oilers"),
    (r"poilievre|poilievre", "加拿大政治人物 Pierre Poilievre"),
    (r"brentford", "英超球队 Brentford FC"),
    (r"cracker-?barrel", "餐饮公司 Cracker Barrel"),
    (r"stanley-cup|presidents-trophy|nhl|nba|mlb|premier-league|relegated",
     "体育赛事（球队名里含 oil/brent）"),
    (r"secretary-of-energy|confirmed-as|out-as-|nominee", "人事任免，非商品市场"),
    # 发言用词竞猜的命名花样很多：say-oil / say-oil-or-gas / say-X-N-times /
    # 各种 during-... 场合。这类市场的标的是"某人会不会说某个词"，与油价
    # 毫无关系，但成交额不小，混进来会直接污染主题指数。
    (r"say-[a-z-]*(oil|gas|energy|pipeline)|say-\w+-\d*-?times|"
     r"during-(the-)?\w+[-\w]*(press-conference|press-briefing|debate|summit|"
     r"address|state-of-the-union|rally|podcast|questions)|roundtable",
     "发言用词竞猜（讲话里会不会说某个词）"),
    (r"tonalenergy|metronome", "节拍器 App TonalEnergy（含 energy 字样）"),
    (r"nominate|for-energy-secretary", "人事提名，非商品市场"),
    (r"largest-company-in-the-world|market-cap", "市值排名（Aramco 是标的之一）"),
    (r"energy-drink|monster", "能量饮料"),
    (r"lng-esports|esports|lol-worlds", "电竞战队 LNG Esports"),
]

#: 第三步：人工归类。key 是主题，value 是该主题下的正则。
#: 归类不是装饰——不同主题对航空股的传导方向不同，混在一起会互相抵消：
#:
#:   supply_risk  供给中断风险上升 -> 油价上行 -> 航空股利空（且风险偏好同时恶化）
#:   opec_supply  OPEC **增产** -> 油价下行 -> 航空股利多（方向与上面相反！）
#:   flows        制裁/贸易流向重排 -> 对油价的方向取决于具体条款，不预设符号
#:
#: 所以 `opec_hike` 这类市场若与 `supply_risk` 直接加总，等于把利多和利空
#: 相加。本模块保持分主题输出，合成由调用方按符号显式处理。
THEMES = {
    # `price_ladder` 必须排在最前面。它与 `price_level` 的区别不是措辞而是
    # **聚合方式**：阶梯里的每个市场是一个不同的行权价 K，把 P(max≥100) 与
    # P(max≥200) 做成交额加权平均是没有意义的数（那个平均值不对应任何事件）。
    # 阶梯要横跨 K 积分出一条隐含分布，见 `pmsp.build.oil_ladder`。
    # 放在字典最前面是因为 `classify` 按插入顺序取第一个命中的主题，而
    # 阶梯 slug 里也含 `crude`，会被 `price_level` 抢走。
    "price_ladder": [
        # will-crude-oil-cl-hit-high-100-by-end-of-march  （上破障碍）
        # will-crude-oil-cl-hit-low-85-by-end-of-march    （下破障碍）
        r"crude-oil-cl-hit-(high|low)-\d+",
        # will-crude-oil-cl-settle-at-70-75-in-march      （到期落在区间）
        r"crude-oil-cl-settle-at-\d+",
        r"^crude-oil-all-time-high",
    ],
    "supply_risk": [
        r"hormuz", r"strike.*oil", r"oil.*strike", r"strikes?-iranian-oil",
        r"oil-(facilit|terminal|refiner|infrastructure)", r"kharg",
        r"tanker", r"oil-(field|depot)", r"attack.*(oil|refiner)",
        # 全量目录里漏掉的两类，各自成交额都在百万量级：
        r"energy-infrastructure-ceasefire",          # 乌克兰能源设施停火
        r"(strike|target)-[a-z-]*refinery",          # 袭击某座具体炼厂
        r"target-an?-iranian-oil",
    ],
    "opec_supply": [r"opec"],
    # Nord Stream 刻意**不**放这里：它是天然气管道，归 `gas_lng`。
    # THEMES 是按插入顺序匹配的，同一个 slug 命中多个主题时取先出现的那个，
    # 所以这两组的模式必须互斥，否则归类结果取决于字典顺序这种隐式的东西。
    "flows_sanctions": [
        r"russian-oil", r"oil-(export|import|embargo|sanction)", r"oil-price-cap",
        r"seize[sd]?-an(other)?-[a-z-]*oil-(ship|tanker)",   # 扣押油轮
        r"venezuela-give-the-us-oil", r"tariff-on-oil",
    ],
    # 基本面（库存/产量）。**不**在 OIL_THEMES 默认里：库存下降与供给中断
    # 对油价是同向的，但它是慢变量、公布频率是周度，混进日频的风险概率
    # 指数里只会加噪声。单列出来是为了让"它为什么没进去"可查。
    "fundamentals": [
        r"crude-oil-reserves", r"oil-production-\d", r"crude-oil-production",
        r"oil-production-reach", r"barrels-per-day",
    ],
    "price_level": [
        r"oil-price", r"price-of-oil", r"crude-(price|above|below|hit)",
        r"wti-", r"brent-(crude|price|above|below)",
        r"oil-(above|below|hit|reach|close)", r"barrel-(above|below)",
    ],
    # 天然气/LNG 单列，**不并入原油指数**。航油是原油的馏分，天然气是另一个
    # 商品、另一条价格曲线（2022 年欧洲气价与油价一度完全脱钩）。归类而不是
    # 直接丢掉，是为了让"为什么它没进原油指数"在审计表里看得见。
    "gas_lng": [r"\blng\b", r"natural-?gas", r"gas-transit", r"nord-stream"],
}

#: 默认做成交额加权概率指数的主题。`gas_lng` 不在其中（见 THEMES 里的说明）；
#: `opec_supply` 的符号与 `supply_risk` 相反，需要调用方显式处理，所以也不
#: 默认合并；`fundamentals` 是周度慢变量；**`price_ladder` 尤其不在其中**，
#: 它的聚合方式完全不同（跨行权价积分，见 `pmsp.build.oil_ladder`），
#: 误并进来会算出一个不对应任何事件的"平均概率"。
OIL_THEMES = ["supply_risk", "flows_sanctions", "price_level"]

#: 单独处理、不走成交额加权平均的主题。
LADDER_THEMES = ["price_ladder"]

_INC = re.compile("|".join(INCLUDE_PATTERNS), re.I)
_EXC = [(re.compile(p, re.I), why) for p, why in EXCLUDE_PATTERNS]
_THEME = {k: re.compile("|".join(v), re.I) for k, v in THEMES.items()}


def classify(slug: str) -> tuple[bool, str, str]:
    """单个 slug -> (是否纳入, 主题, 理由)。"""
    if not isinstance(slug, str) or not _INC.search(slug):
        return False, "", "关键词未命中"
    for pat, why in _EXC:
        if pat.search(slug):
            return False, "", f"排除：{why}"
    for theme, pat in _THEME.items():
        if pat.search(slug):
            return True, theme, "纳入"
    return False, "", "关键词命中但不属于任何已定义主题（待人工复核）"


def audit_table(
    catalog: pd.DataFrame, volume: pd.Series | None = None, min_volume: float = 0.0
) -> pd.DataFrame:
    """对整个市场目录做一遍三步筛选，产出**可逐条复核**的审计表。

    Parameters
    ----------
    catalog : pd.DataFrame
        `hf_polymarket.build_hourly_panel` 产出的市场目录，需含
        `condition_id` / `market_slug` / `category` / `category_refined`。
    volume : pd.Series | None
        `index=condition_id` 的累计成交额。用于按流动性排序与过滤——
        一个总成交 45 美元的 OPEC 市场在统计上等于不存在，留着只会
        往主题指数里灌噪声。
    min_volume : float
        纳入的最低累计成交额（USDC）。

    Returns
    -------
    pd.DataFrame
        每个关键词命中的市场一行，含 `included` / `theme` / `reason` / `vol_usdc`。
        `included=False` 的行也保留，这样"为什么某个市场没进来"是可查的。
    """
    cat = catalog.copy()
    res = cat.market_slug.map(classify)
    cat["included"] = [r[0] for r in res]
    cat["theme"] = [r[1] for r in res]
    cat["reason"] = [r[2] for r in res]
    hit = cat[cat.reason != "关键词未命中"].copy()
    if volume is not None:
        hit["vol_usdc"] = hit.condition_id.map(volume).fillna(0.0)
        thin = hit.included & (hit.vol_usdc < min_volume)
        hit.loc[thin, "included"] = False
        hit.loc[thin, "reason"] = f"成交额 < {min_volume:,.0f} USDC，流动性不足"
        hit = hit.sort_values("vol_usdc", ascending=False)
    cols = [
        c
        for c in ["condition_id", "market_slug", "category", "category_refined",
                  "vol_usdc", "included", "theme", "reason", "close_at",
                  "resolution_status", "neg_risk"]
        if c in hit.columns
    ]
    return hit[cols].reset_index(drop=True)


def selected_ids(audit: pd.DataFrame, themes: list[str] | None = None) -> list[str]:
    """从审计表取出最终纳入的 condition_id。"""
    sel = audit[audit.included]
    if themes:
        sel = sel[sel.theme.isin(themes)]
    return sel.condition_id.tolist()
