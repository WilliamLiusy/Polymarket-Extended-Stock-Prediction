"""本项目所有图表共用的一套配色与画布规范（matplotlib 实现）。

配色**不是**挑出来的，是校验过的：调色板取自 `dataviz` skill 的参考实例，
用它的 `validate_palette.js` 跑过六项检查。四槽分类色在浅色底下的结果：

    Lightness band        PASS   4 个色都在 L 0.43–0.77
    Chroma floor          PASS
    CVD separation        PASS   最差相邻对 #eda100↔#1baf7a ΔE 9.1 (protan)
    Normal-vision floor   PASS   最差相邻对 ΔE 22.9
    Contrast vs surface   WARN   aqua 2.74 / yellow 2.11，低于 3:1

最后那条 WARN 不是可以忽略的提示，它带着一个**义务**：对比度不足的颜色必须
有"救济"——可见的直接标注或等价的表格视图。所以本模块里：

* `end_label()` 给每条线在末端打上带色点的文字标注，颜色身份不只靠色相；
* `save()` 每存一张 PNG 就**同时**落一份同名 CSV。

这两件事是强制的，不是锦上添花。一张只有颜色能读出来的图，在色觉障碍、
黑白打印、forced-colors 三种场合下都等于没有图。

## 几条硬规则（违反了就是错图，不是风格问题）

* **绝不画双 Y 轴。**两个量纲不同的序列要对比，走三条路之一：拆成上下分面
  （共享 x 轴）、各自归一化到同一基期（=100）、或者干脆两张图。双轴图的
  两个刻度怎么对齐是任意的，它会**凭空造出**一个数据里没有的相关性——
  这正是本分析要检验的东西，用一张会造假相关的图去展示它，是自相矛盾。
* **颜色跟着实体走，不跟排名走。**`SERIES` 按固定顺序分配，筛掉一条序列
  不会让剩下的换色。
* **分类色最多 8 槽，绝不循环或生成第 9 个色。**超了就并入"其他"或拆分面。
* **顺序型用单色相深浅；双向型用蓝↔红加灰色中点。**相关系数是有符号的，
  所以热图用 `DIVERGING`，且 vmin/vmax 必须对称，否则灰色中点不在 0 上，
  颜色就在说谎。
* **网格与坐标轴是实线细发丝线，不用虚线。**虚线会被读成"预测"或"阈值"。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

#: 分类色（浅色底）。顺序本身就是色觉安全机制的一部分，不要重排。
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

#: 图表底色与各级墨色。
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: 双向色标：蓝 ↔ 中性灰 ↔ 红。中点用灰而不是任何色相——中点必须读成
#: "什么都没有"。蓝↔青这种两头都是冷色的组合不行，读不出"相反"。
DIVERGING = LinearSegmentedColormap.from_list(
    "pmsp_div",
    ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#9e2423"],
)

#: 单向色标：**一个色相**，浅 -> 深。用于"大小"型的量（概率、成交额）。
#: 刻意不用 viridis 之类的多色相标：概率 0.3 与 0.7 之间没有"换了一种东西"
#: 的含义，只有多与少，色相一变读者就会去找那个变化对应什么类别。
#:
#: 起点是一层**很淡但不等于底色**的蓝，既不是纯白也不是 SURFACE。三个都试过：
#: 纯白起点会在米白底上浮出一块比背景更亮的斑；起点取 SURFACE 更糟——
#: 值为 0 的格子与"根本没有这个格子"在画布上完全同色，于是"概率确实接近零"
#: 和"当天没有报价"两件完全不同的事读起来一模一样。色标的最低档必须仍然
#: 看得见，否则"有数据且为零"这个信息就丢了。
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "pmsp_seq", ["#eaf1fa", "#cfe0f6", "#9cc0ea", "#5f96d9", "#2a78d6", "#14498a"],
)

#: 状态色（好/注意/严重/危急）。**保留色**，绝不拿来当第 9 条序列色。
#: 本项目里只用于"显著性"这类状态标注，且必须同时带文字或符号。
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
          "critical": "#d03b3b"}

#: 中文字体。本机只有 Droid Sans Fallback 带 CJK 字形，没有它图上全是豆腐块。
_CJK = "Droid Sans Fallback"


def use() -> None:
    """装上全局画布规范。每个出图脚本开头调一次。"""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["sans-serif"],
        "font.sans-serif": [_CJK, "DejaVu Sans"],
        "axes.unicode_minus": False,          # 负号要用 ASCII，CJK 字体里没有 U+2212
        # 发丝线网格，实线。比底色深一档，只够"看得见"，不抢数据。
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,               # 细笔触
        "lines.markersize": 4.5,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
    })


def title(ax, head: str, sub: str | None = None) -> None:
    """左对齐标题 + 次级说明行。说明行用次级墨色，不用序列色。

    文字永远穿"墨色"，不穿序列色——身份由旁边那个有颜色的图元承担。
    文字也染色的话，一旦两个色相在 CVD 下靠近，连标题都读不出是哪条。

    两行都用 **offset points** 定位，不用 `set_title(pad=...)` + 轴坐标混排：
    轴坐标的 1.015 换算成多少像素取决于分面高度，矮分面上标题与副标题会
    直接叠在一起（第一版就是这么撞的）。points 是绝对的，任何尺寸都不会撞。

    注意：这里的文字**不解析 markdown**。`**粗体**` 在 matplotlib 里就是
    四个星号，会原样画出来。所以图上的文案一律写成纯文本，强调靠语序，
    markdown 只出现在报告的 .md 里。
    """
    kw = dict(xy=(0.0, 1.0), xycoords="axes fraction",
              textcoords="offset points", va="bottom", ha="left",
              annotation_clip=False)
    if sub:
        ax.annotate(head, xytext=(0, 24), fontsize=12, fontweight="semibold",
                    color=INK, **kw)
        ax.annotate(sub, xytext=(0, 8), fontsize=9, color=INK_2, **kw)
    else:
        ax.annotate(head, xytext=(0, 8), fontsize=12, fontweight="semibold",
                    color=INK, **kw)


def end_label(ax, x, y, text: str, color: str, pad_pt: float = 7.0) -> None:
    """在线的末端打直接标注：一个色点 + 一段墨色文字。

    这是对比度 WARN 的"救济"实现。只标末端一处，**不是**每个点都标数字——
    每点都标是噪声，没人读。

    偏移量用 **points**（`textcoords="offset points"`）而不是数据坐标：x 轴
    可能是日期轴，`Timestamp + float` 会直接抛 TypeError；就算是数值轴，
    按坐标范围的百分比偏移也会让不同量纲的分面上留白宽度不一致。
    """
    ax.plot([x], [y], "o", color=color, markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=5,
            clip_on=False)
    ax.annotate(text, xy=(x, y), xytext=(pad_pt, 0),
                textcoords="offset points", color=INK_2, fontsize=9,
                va="center", ha="left", annotation_clip=False)


def end_labels(ax, items, pad_pt: float = 7.0, min_gap_pt: float = 11.0) -> None:
    """一组末端标注，**互相错开**。多序列分面上必须用这个，不要循环调
    `end_label`。

    三条线在末端收敛到很近的位置时（归一化到基期 = 100 的分面上这是常态：
    141 / 137 / 135），逐条调 `end_label` 会把三段文字画在同一个高度上，
    叠成一团谁也读不出来——而末端标注本来就是对比度 WARN 的"救济"，
    叠掉了等于救济没了。

    错开方式：把每条的 y 换算到**显示坐标**（像素），按 y 排序后自下而上
    保证相邻至少隔 `min_gap_pt` 点，然后以差值作为 offset points 画出去。
    在数据坐标里错开是不行的——不同分面的 y 量纲不同，"错开 2 个单位"
    在概率轴上是半个分面，在指数轴上看不出来。

    Parameters
    ----------
    items : Iterable[tuple]
        `(x, y, text, color)`，顺序任意。
    """
    items = list(items)
    if not items:
        return
    ax.figure.canvas.draw()                 # 先落一次版，否则变换还不可用
    trans = ax.transData
    px = [float(trans.transform((_to_num(ax, x), y))[1]) for x, y, _, _ in items]
    dpp = ax.figure.dpi / 72.0              # points -> pixels
    order = sorted(range(len(items)), key=lambda i: px[i])
    adj = list(px)
    for j in range(1, len(order)):
        prev, cur = order[j - 1], order[j]
        if adj[cur] - adj[prev] < min_gap_pt * dpp:
            adj[cur] = adj[prev] + min_gap_pt * dpp
    for i, (x, y, text, color) in enumerate(items):
        ax.plot([x], [y], "o", color=color, markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=5,
                clip_on=False)
        ax.annotate(text, xy=(x, y), xytext=(pad_pt, (adj[i] - px[i]) / dpp),
                    textcoords="offset points", color=INK_2, fontsize=9,
                    va="center", ha="left", annotation_clip=False)


def _to_num(ax, x):
    """把 x 换成轴的数值坐标。日期轴上 `Timestamp` 不能直接喂给 transData。"""
    return ax.xaxis.convert_units(x)


def _wrap_cjk(text: str, width: int) -> str:
    """按字符数硬折行。`textwrap` 在中文上不能用——它只在空格和西文标点处
    断行，一段没有空格的中文会被当成一个超长单词原样吐出来（散点图第一版
    的脚注就是这样，一行 80 个字伸出画布）。

    两处例外，都是踩出来的：

    1. 闭合标点不能顶到行首——断点落在它前面时往后挪一个字。
    2. **不在西文/数字串中间断**。纯按字符数切会把 "|r| > 0.63" 断成
       "0." + "63"，下一行开头一个孤零零的 "63" 读起来是另一个数；图脚里
       的数字是结论的一部分，劈开就等于印错。断点落在 ASCII 串内部时把它
       退到该串的开头，整串挪到下一行。
    """
    close = "，。、）」：；%"
    ascii_run = set("0123456789.,<>=+-|%()")
    ascii_run |= set("abcdefghijklmnopqrstuvwxyz")
    ascii_run |= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    out, i = [], 0
    while i < len(text):
        j = min(i + width, len(text))
        while j < len(text) and text[j] in close:
            j += 1
        # 退到 ASCII 串开头（留一个字符的余地，否则整行可能退成空行）
        if j < len(text) and text[j] in ascii_run and text[j - 1] in ascii_run:
            b = j
            while b > i + 1 and text[b - 1] in ascii_run:
                b -= 1
            if b > i:
                j = b
        out.append(text[i:j].rstrip() if j < len(text) else text[i:j])
        i = j
        while i < len(text) and text[i] == " ":   # 别让退格留下的空格顶行首
            i += 1
    return "\n".join(out)


def foot(fig, text: str, width: int = 58, gap: float = 0.012) -> None:
    """图脚说明：整张图最下方、x 轴标签**下面**的一段灰字。

    挂在 **figure** 上而不是任何一个分面上。这一条是踩出来的，代价是一张
    画废的 1×2 散点图：把长脚注 `annotate` 到分面上（哪怕 xycoords 写的是
    `figure fraction`），`constrained_layout` 会把它算进该分面的 tight bbox，
    然后为了让 bbox 装进网格单元去**压缩分面**；而文字宽度是以 points 固定的，
    压缩分面并不能让它变窄，于是收缩失控——两个分面各被压到宽度 0.128
    （正常 0.474），左右各缩成一条竖带，中间空出一大片白。

    试过的另一条路是给脚注 `set_in_layout(False)`：分面确实恢复了，但
    `savefig.bbox="tight"` 同样按 `get_in_layout` 过滤，于是脚注和标题一起
    被裁掉了——排除在排版之外就等于排除在画布之外。

    figure 级文字不属于任何分面，排版引擎不会为它压缩分面，而 `bbox="tight"`
    仍然会把它算进保存范围。`gap` 取负方向的一点点（y = -gap），让它落在
    x 轴标签下方而不是叠上去。
    """
    fig.text(0.0, -gap, _wrap_cjk(text, width) if width else text,
             fontsize=8.5, color=MUTED, va="top", ha="left")


def fmt_p(p: float, floor: float = 1e-3) -> str:
    """p 值的显示。小于 `floor` 时写 "< floor"，**绝不写成 0.000**。

    置换检验的经验 p 值按构造严格大于 0（分子上有那个 +1），把 5e-5 印成
    "p = 0.000" 是在图上宣称"零假设下绝不可能"，而这个检验只跑了两万次
    置换，支撑不了那句话。
    """
    if not np.isfinite(p):
        return "p = n/a"
    return f"p < {floor:g}" if p < floor else f"p = {p:.3f}"


def zero_line(ax, axis: str = "y") -> None:
    """零参考线。比网格深一档——"0"在有符号的图里是个真实的分界，不是网格。"""
    fn = ax.axhline if axis == "y" else ax.axvline
    fn(0.0, color=AXIS, linewidth=1.0, zorder=1)


def save(fig, path: str | Path, table: pd.DataFrame | None = None) -> Path:
    """存图，并**强制**落一份表格视图（同名 CSV）。

    表格视图不是可选项：上面 aqua/yellow 两个槽的对比度低于 3:1，规范要求
    有等价的非颜色读法；而且"图里那个峰值到底是多少"这种问题，图永远答不了。
    传 `table=None` 会抛错而不是静默跳过——静默跳过就等于规范没落地。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if table is None:
        raise ValueError(f"{path.name} 缺表格视图；每张图必须有等价 CSV")
    table.to_csv(path.with_suffix(".csv"), index=False, float_format="%.6g")
    return path


def index_to_100(s: pd.Series) -> pd.Series:
    """归一化到基期 = 100。多条不同量纲的序列要画在**同一个** y 轴上时用它。

    这是双轴图的正当替代：两条线的单位变成"相对基期的累计涨幅"，
    这个单位是真实可比的，而双轴的刻度对齐是画图的人随手定的。
    """
    s = s.astype(float)
    first = s.loc[s.first_valid_index()] if s.notna().any() else np.nan
    return s / first * 100.0
