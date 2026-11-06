"""めぐコラム『種牡馬系統による適性の違いはあるのか』（sire-line-aptitude）を実データから集計し、
frontend/content/columns/sire-line-aptitude.json を書く。

問い: 父系の系統（smartrc の小系統。ディープ系・キングマンボ系・ロベルト系など）で、産駒の適性（芝ダ・距離・道悪・早熟/晩成・脚質）に違いはあるか。

データ:
  レース: compare_series.load()（2020〜2025 年の JRA 平地。障害・取消・除外を除く）。馬場状態は race_result_flat の track_condition、
          初角の位置は race_context.positions()（同じ流儀で 2019 年から読む）。
  父:     compare_series.SIRE_MAP_PATH（horse_sire_map.parquet）。
  系統:   smartrc の 大系統 → 小系統（data/meta/sire_lines/sire_smartrc_lines.json。2026-11-06 ユーザー指示で netkeiba の系統から切り替え）。
          smartrc で系統が空欄の種牡馬は netkeiba の系統（data/meta/pedigree_lines/sire_line_origin.json の最も下の系統）を使い、
          「smartrc 未分類」の下に置く。父（同じ父の群の検定に使う）は sire_line_origin.json の chain の先頭。

適性の指標（種牡馬ごと、産駒の出走から。能力の差が入らないよう、分布か「オッズの見込みとの差」の差にする）:
  芝↔ダート   （分布）ダートを走った割合。
  距離         （分布）走った距離 − その芝ダの平均距離 の平均（m）。ダート寄りの血統ほど短く出るのを除く。
  脚質         （分布）初角の位置 (順位−1)/(頭数−1) の平均。0 が先頭、1 が最後方。産駒の走り方の記述なので今走の位置を使う。
  道悪         （成績）重・不良での (複勝 − 見込み) − 良での (複勝 − 見込み)。見込みは単勝オッズから（early_maturing_bloodline.expected_top3）。
  早熟↔晩成   （成績）4 歳以上での (複勝 − 見込み) − 2〜3 歳での (複勝 − 見込み)。
  総合         上の 5 つ（縮めた値を種牡馬の間の SD で割ったもの）の第 1 主成分。

系統の木:
  lines（新しい→古い）を逆にした経路。同じ系統名が飛び飛びに出るとき（Raise a Native系 → Mr. Prospector系 → Raise a Native系）は、
  2 回目以降を飛ばす（根に近い最初の位置を採る）。名前の付いた最も下の系統の下は、父系の祖先の馬（その系統の祖から種牡馬の父まで）で枝分かれさせる。
  産駒（データに出走のある馬）が MIN_HORSES 頭未満の種牡馬は、同じ親の下で 1 つの群にまとめる（群の名前は含まれる種牡馬の名前）。子が 1 つだけの節は子と入れ替える。

縮小推定（経験ベイズ）:
  各節の生の値 x と、その標本分散 v（同じ馬の出走をまとめたクラスタの分散）を出し、根から順に
  post = 親の post + τ²/(τ²+v)·(x − 親の post) で親へ縮める。τ² は種牡馬の値と親の値の差の分散から v の平均を引いたもの（指標ごとに 1 つ）。
  95% 区間は事後の SD（親の不確かさも含める）から。

系統の差の検定:
  産駒の出走 MIN_TEST 以上の種牡馬で、生の値を 1/(τ²+v) で重み付けし、系統（名前の付いた最も下の系統、または父が同じ種牡馬の群）で
  説明される分散の割合 η² を出す。系統のラベルを並べ替えた η²（N_PERM 回）と比べて p 値とし、指標 × 2 つの分け方を Holm で補正する。
  「偶然の分を引いた割合」= (η² − 並べ替えの平均) / (1 − 並べ替えの平均)。

使い方:
    python -m src.research.columns.sire_line_treemap            # 集計して記事を書く
    python -m src.research.columns.sire_line_treemap --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import FLAT_DIR, SIRE_MAP_PATH, YEARS, jump_mask, load
from src.research.columns.early_maturing_bloodline import expected_top3
from src.research.columns.race_context import positions
from src.research.race.running_map import AXES
from src.research.race.running_profile import AXES_PATH
from src.scripts.scraping.trace_sire_line_origins import line_path  # noqa: F401（テストからも t.line_path で使う）

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/sire-line-aptitude.json"
ORIGIN_PATH = REPO / "data/meta/pedigree_lines/sire_line_origin.json"
MIN_HORSES = 10       # 木に 1 頭で載せる種牡馬の産駒の頭数（未満は群にまとめ、含まれる種牡馬を名前に出す）
MIN_TEST = 300        # 系統の差の検定に入れる種牡馬の産駒の出走数
MIN_GROUP = 3         # 検定で 1 つの系統として扱う最少の種牡馬数（未満は「その他」にまとめる）
# 勝利距離の分布: 1 着の出走を距離帯に分け、帯ごとの割合を成分（平均）にして縮小推定に乗せ、縮めた割合から四分位などを出す
WIN_EDGES = [1000, 1250, 1450, 1650, 1850, 2050, 2250, 2450, 2800, 3700]       # 帯の境（m）。1200・1300〜1400・1500〜1600… の 9 帯
WIN_BINS = [f"wb{i}" for i in range(len(WIN_EDGES) - 1)]
MIN_RUNS = {"windist": 10, **{b: 10 for b in WIN_BINS}, **{f"ax_{a[0]}": 20 for a in AXES}}      # 色だけの指標: 成分の出走がこれ未満なら値なし（馬が 1〜数頭だと分散が 0 に近く出て、縮められない）
MIN_PART = 20         # 差の指標（道悪・早熟）で、片方の出走がこれ未満なら値なし
N_PERM = 5000
SEED = 20261104
START_SHARE = 0.95    # 最初の表示: 1 つの子が全体のこの割合以上を占める間は、その子から表示する

# 成分（出走ごとの値の平均）。指標はこの平均か、2 つの平均の差
# 道悪・早熟の成分は芝（_t）・ダート（_d）別に持ち、芝ダの割合の違いが差に混ざらないようにする。_o はオッズの見込みとの差の版
SURF = {"_t": "芝", "_d": "ダート"}
AX_COMPS = [f"ax_{a[0]}" for a in AXES]       # 走法の軸（実測ラップのある出走だけ。running_map の 8 軸）
COMPONENTS = ["dirt", "dist", "style"] + [f"{c}{sf}{o}" for o in ("", "_o") for c in ("heavy", "good", "old", "young") for sf in SURF] + ["windist"] + WIN_BINS + AX_COMPS


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    kind: str                 # 分布 / 成績 / 総合
    low: str                  # 値が小さい側の意味
    high: str
    fmt: str                  # pct / m / pos / pt / num
    parts: tuple[tuple[str, ...], ...]   # ((c,),) なら成分 c の平均。((a, b), …) なら a − b を組ごとに出し、精度で重み付けた平均
    desc: str


METRICS = [
    Metric("dirt", "芝↔ダート", "分布", "芝寄り", "ダート寄り", "pct", (("dirt",),), "産駒の出走のうちダートの割合"),
    Metric("dist", "距離", "分布", "短距離寄り", "長距離寄り", "m", (("dist",),), "走った距離 − その芝ダの平均距離（m）"),
    Metric("style", "脚質", "分布", "前に行く", "後ろから", "pos", (("style",),), "初角の位置の平均（0=先頭、1=最後方）"),
    Metric("going", "道悪", "成績", "道悪が苦手", "道悪が得意", "pt", (("heavy_t", "good_t"), ("heavy_d", "good_d")), "重・不良と良の、複勝率 − 頭数からの期待（3/頭数）の差（芝・ダートそれぞれで出して平均）"),
    Metric("mature", "早熟↔晩成", "成績", "早熟（2〜3歳）", "晩成（4歳以上）", "pt", (("old_t", "young_t"), ("old_d", "young_d")), "4歳以上と2〜3歳の、複勝率 − 3/頭数 の差（芝・ダートそれぞれで出して平均）"),
]
# 色で見るだけの指標（総合・検定には入れない。縮小推定は同じ）。勝利距離平均は 1 着の出走の距離（m）の平均、走法の軸は実測ラップから出した 1 走ごとの値の平均
EXTRA_METRICS = [
    Metric("windist", "勝利距離平均", "分布", "短い距離で勝つ", "長い距離で勝つ", "dist", (("windist",),),
           "産駒が 1 着になったレースの距離の平均（m）。芝ダの違いも含む"),
] + [
    Metric(a[1][3:], f"走法: {a[3]}", "走法", a[5], a[4], "val2" if a[0] == "sustain" else "val1", ((f"ax_{a[0]}",),),
           "産駒の出走ごとの値の平均（実測ラップのある 2023 年以降の出走だけ。外れ値は上下 1% で丸めた）")
    for a in AXES
]
# 勝利距離の帯ごとの割合（縮小推定だけに使い、表示はしない）
BIN_METRICS = [Metric(b, b, "分布", "", "", "pct", ((b,),), "") for b in WIN_BINS]
# 縮めた帯の割合から出す、勝利距離の幅（どの距離でも勝てるか）。(key, ラベル, 下の割合, 上の割合, 説明)
SPREAD_DEFS = [
    ("win_iqr", "勝利距離の幅（中央50%）", 0.25, 0.75, "1 着になったレースの距離の、第1四分位〜第3四分位の幅（m）。大きいほど、いろいろな距離で勝つ"),
    ("win_p90", "勝利距離の幅（中央90%）", 0.05, 0.95, "1 着になったレースの距離を短い順に並べ、下から5%〜95%の範囲の幅（m）。両端の5%ずつを除いた、ふだん勝つ距離の広さ"),
]
SPREAD_METRICS = [Metric(k, lab, "分布", "狭い距離で勝つ", "幅広い距離で勝つ", "dist", (), d) for k, lab, _, _, d in SPREAD_DEFS]
# 検定だけに使う、オッズの見込みとの差の版（人気に織り込まれていない分の適性）
ODDS_METRICS = [
    Metric("going_o", "道悪（人気との差）", "成績", "道悪が苦手", "道悪が得意", "pt", (("heavy_t_o", "good_t_o"), ("heavy_d_o", "good_d_o")), "重・不良と良の (複勝率 − オッズの見込み) の差"),
    Metric("mature_o", "早熟↔晩成（人気との差）", "成績", "早熟", "晩成", "pt", (("old_t_o", "young_t_o"), ("old_d_o", "young_d_o")), "4歳以上と2〜3歳の (複勝率 − オッズの見込み) の差"),
]
ROUND = {"pct": 3, "m": 0, "pos": 3, "pt": 3, "num": 2, "dist": 0, "val1": 2, "val2": 3}
SHOWN_EXTRA = EXTRA_METRICS[:1] + SPREAD_METRICS + EXTRA_METRICS[1:]     # 色の選択肢に並べる順（勝利距離平均 → その幅 → 走法の軸）
METRIC_BY_KEY = {m.key: m for m in METRICS + EXTRA_METRICS + ODDS_METRICS}


# ── データ ──────────────────────────────────────────────


def strip_country(name: str) -> str:
    """「Kingmambo (米)」→「Kingmambo」。"""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip()


def load_axes() -> pd.DataFrame:
    """実測ラップの走法の軸（1 走ごと）。異常走は除き、外れ値は上下 1% で丸める。ファイルが無ければ空（全部 NaN になる）。"""
    cols = [a[1][3:] for a in AXES]
    if not Path(AXES_PATH).exists():
        return pd.DataFrame({"race_id": pd.Series(dtype=str), "horse_id": pd.Series(dtype=str), **{c: pd.Series(dtype=float) for c in AX_COMPS}})
    import pyarrow.parquet as pq

    have = set(pq.read_schema(AXES_PATH).names)       # 走法マップに足された軸が、まだ run_axes に無いことがある（その軸は値なし）
    ax = pd.read_parquet(AXES_PATH, columns=["race_id", "horse_id", "abn"] + [c for c in cols if c in have])
    ax["race_id"], ax["horse_id"] = ax["race_id"].astype(str), ax["horse_id"].astype(str)
    ax = ax.drop_duplicates(["race_id", "horse_id"])
    out = ax[["race_id", "horse_id"]].copy()
    for c in cols:
        if c not in have:
            out[f"ax_{c}"] = np.nan
            continue
        v = ax[c].astype(float).where(~ax["abn"].to_numpy(bool))
        lo, hi = v.quantile(0.01), v.quantile(0.99)
        out[f"ax_{c}"] = v.clip(lo, hi)
    return out


def frame() -> pd.DataFrame:
    """出走ごとに、父（sire_id）と成分の値を付ける。"""
    df = load()
    cols = ["race_id", "date", "surface", "horse_id", "passing_order", "odds", "popularity", "race_name", "race_class", "distance", "track_condition"]
    years = [y for y in range(min(YEARS) - 1, max(YEARS) + 1) if (FLAT_DIR / str(y) / "race_result_flat.parquet").exists()]
    r = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=cols) for y in years], ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"]) & ~jump_mask(r) & (r["odds"] > 0) & (r["popularity"] > 0)].copy()
    r["horse_id"] = r["horse_id"].astype(str)
    r = positions(r)[["race_id", "horse_id", "q_now", "track_condition"]].drop_duplicates(["race_id", "horse_id"])
    df = df.merge(r, on=["race_id", "horse_id"], how="left")
    m = pd.read_parquet(SIRE_MAP_PATH, columns=["horse_id", "sire_id"]).dropna(subset=["sire_id"]).drop_duplicates("horse_id")
    m["horse_id"] = m["horse_id"].astype(str)
    m["sire_id"] = m["sire_id"].astype(str)
    df = df.merge(m, on="horse_id", how="inner")
    res = df["top3"] - expected_top3(df["implied"], df["top3"])          # オッズの見込みとの差
    base = df["top3"] - np.minimum(3, df["starters"]) / df["starters"]   # 頭数からの期待との差（人気は考えない）
    df["c_dirt"] = (df["surface"] == "ダート").astype(float)
    df["c_dist"] = df["distance"] - df.groupby("surface")["distance"].transform("mean")
    df["c_style"] = df["q_now"]
    win = df["finish_position"] == 1
    df["c_windist"] = df["distance"].where(win)
    band = pd.cut(df["distance"], WIN_EDGES, labels=False, right=False)
    for i, b in enumerate(WIN_BINS):
        df[f"c_{b}"] = (band == i).astype(float).where(win)
    ax = load_axes()
    df = df.merge(ax, on=["race_id", "horse_id"], how="left")
    for c in AX_COMPS:
        df[f"c_{c}"] = df[c] if c in df else np.nan
    heavy, good = df["track_condition"].isin(["重", "不良"]), df["track_condition"] == "良"
    old, young = df["age"] >= 4, df["age"] <= 3
    for o, y in [("", base), ("_o", res)]:
        for sf, surf in SURF.items():
            on = df["surface"] == surf
            df[f"c_heavy{sf}{o}"] = y.where(heavy & on)
            df[f"c_good{sf}{o}"] = y.where(good & on)
            df[f"c_old{sf}{o}"] = y.where(old & on)
            df[f"c_young{sf}{o}"] = y.where(young & on)
    return df


def sire_stats(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """種牡馬ごとの十分統計量。配列は (成分, 6): N・Σy・Σy²・Σ(馬ごとの和)²・Σ(馬ごとの和 × 馬ごとの数)・Σ(馬ごとの数)²。"""
    out: dict[str, np.ndarray] = {}
    per = []
    for c in COMPONENTS:
        y = df[f"c_{c}"]
        d = pd.DataFrame({"sire_id": df["sire_id"], "horse_id": df["horse_id"], "y": y}).dropna(subset=["y"])
        d["y2"] = d["y"] ** 2
        h = d.groupby(["sire_id", "horse_id"]).agg(n=("y", "size"), s=("y", "sum"), q=("y2", "sum"))
        h["ss"] = h["s"] ** 2
        h["sn"] = h["s"] * h["n"]
        h["nn"] = h["n"] ** 2
        per.append(h.groupby(level="sire_id")[["n", "s", "q", "ss", "sn", "nn"]].sum())
    sires = sorted(set().union(*[p.index for p in per]))
    for s in sires:
        out[s] = np.array([p.loc[s].to_numpy(float) if s in p.index else np.zeros(6) for p in per])
    return out


def comp_mean_var(st: np.ndarray) -> tuple[float, float]:
    """成分の平均と、その分散（同じ馬の出走をまとめたクラスタの分散。独立とみなした分散より小さくはしない）。"""
    n, s, q, ss, sn, nn = st
    if n <= 0:
        return math.nan, math.inf
    m = s / n
    cl = max(0.0, ss - 2 * m * sn + m * m * nn) / (n * n)
    iid = max(0.0, q - n * m * m) / (n * max(n - 1, 1))
    return m, max(cl, iid, 1e-12)


def metric_raw(st: np.ndarray, metric: Metric) -> tuple[float, float, int]:
    """十分統計量（成分の配列）から、指標の生の値・分散・出走数（差の指標は、組ごとの少ない側の数の和）。
    差の指標は、両側に MIN_PART 以上ある組だけで a − b を出し、分散の逆数で重み付けて平均する。"""
    if len(metric.parts) == 1 and len(metric.parts[0]) == 1:
        i = COMPONENTS.index(metric.parts[0][0])
        m, v = comp_mean_var(st[i])
        if st[i][0] < MIN_RUNS.get(metric.parts[0][0], 0):
            v = math.inf
        return m, v, int(st[i][0])
    ds, ws, n = [], [], 0
    for a, b in metric.parts:
        sa, sb = st[COMPONENTS.index(a)], st[COMPONENTS.index(b)]
        if min(sa[0], sb[0]) < MIN_PART:
            continue
        ma, va = comp_mean_var(sa)
        mb, vb = comp_mean_var(sb)
        ds.append(ma - mb); ws.append(1.0 / (va + vb)); n += int(min(sa[0], sb[0]))
    if not ds:
        return math.nan, math.inf, n
    w = np.array(ws)
    return float(np.dot(ds, w) / w.sum()), float(1.0 / w.sum()), n


# ── 系統の木 ──────────────────────────────────────────────


def sire_path(sid: str, rec: dict[str, Any]) -> list[tuple[str, str, str]]:
    """根 → 葉の節の列 [(key, 名前, 種類)]。種類は line（名前の付いた系統）・branch（父系の祖先の馬）・sire（種牡馬の産駒）。
    rec は小さな版の 1 頭分（chain は父から、名前の付いた最も下の系統の祖まで）。"""
    path = line_path(rec["lines"])
    nodes = [("L:" + "/".join(path[: i + 1]), ln, "line") for i, ln in enumerate(path)]
    for hid, hname in reversed(rec["chain"][:-1]):     # 系統の祖の 1 つ下から、種牡馬の父まで
        nodes.append(("H:" + hid, strip_country(hname) + "系", "branch"))
    nodes.append(("S:" + sid, strip_country(rec["name"]), "sire"))
    return nodes


@dataclass
class Node:
    key: str
    name: str
    kind: str
    children: list["Node"] = field(default_factory=list)
    sires: list[str] = field(default_factory=list)     # 葉のとき、まとめる種牡馬の id
    stats: np.ndarray | None = None
    est: dict[str, tuple[float, float, float, int]] = field(default_factory=dict)   # 指標 → (post, sd, raw, n)
    score: float = 0.0
    horses: int = 0                                    # 配下の産駒の頭数（出走のあった馬）
    members: list[tuple[str, int]] = field(default_factory=list)   # pool のとき、まとめた種牡馬の (名前, 産駒の頭数)

    @property
    def leaf(self) -> bool:
        return not self.children

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def build_tree(origin: dict[str, dict[str, Any]], sires: set[str] | list[str]) -> Node:
    """sires（データに出走のある種牡馬）の木。"""
    root = Node("ROOT", "全種牡馬", "root")
    index: dict[str, Node] = {"ROOT": root}
    for sid in sorted(sires):
        rec = origin.get(sid)
        if rec is None:
            continue
        cur = root
        for key, name, kind in sire_path(sid, rec):
            nd = index.get(key)
            if nd is None or kind == "sire":
                nd = Node(key, name, kind, sires=[sid] if kind == "sire" else [])
                index[key] = nd
                cur.children.append(nd)
            cur = nd
    # 種牡馬が、ほかの種牡馬の父系の祖先（branch）でもあるとき、自身の産駒の葉をその branch の下へ移す
    for nd in list(root.walk()):
        branches = {c.key[2:]: c for c in nd.children if c.kind == "branch"}
        keep = []
        for c in nd.children:
            b = branches.get(c.key[2:]) if c.kind == "sire" else None
            if b is not None:
                c.name = c.name + "（自身）"
                b.children.insert(0, c)
            else:
                keep.append(c)
        nd.children = keep
    return root


def aggregate(nd: Node, st: dict[str, np.ndarray], horses: dict[str, int]) -> np.ndarray:
    if nd.leaf:
        nd.stats = sum((st[s] for s in nd.sires), np.zeros((len(COMPONENTS), 6)))
        nd.horses = sum(horses.get(s, 0) for s in nd.sires)
    else:
        nd.stats = sum((aggregate(c, st, horses) for c in nd.children), np.zeros((len(COMPONENTS), 6)))
        nd.horses = sum(c.horses for c in nd.children)
    return nd.stats


def label_pools(root: Node, names: dict[str, str], horses: dict[str, int]) -> None:
    """まとめた群（pool）に、含まれる種牡馬を持たせ、名前にも出す（「ほか N 頭」のような中身の見えない名前にしない）。"""
    for nd in root.walk():
        if nd.kind != "pool":
            continue
        order = sorted(nd.sires, key=lambda s: (-horses.get(s, 0), names.get(s, s)))
        nd.members = [(names.get(s, s), horses.get(s, 0)) for s in order]
        shown = [n for n, _ in nd.members[:3]]       # 名前に出すのは多い順に 3 頭まで（全員は members に持ち、選択時の一覧に出す）
        joined = "・".join(shown) + (f" ほか{len(nd.members) - 3}頭" if len(nd.members) > 3 else "")
        if nd.name.startswith("ほか"):
            nd.name = joined
        elif re.search(r"（\d+ 頭）$", nd.name):
            nd.name = re.sub(r"（\d+ 頭）$", f"（{joined}）", nd.name)


def runs(nd: Node) -> int:
    return int(nd.stats[0][0]) if nd.stats is not None else 0


def prune(nd: Node, names: dict[str, str]) -> Node:
    """産駒の少ない種牡馬を 1 つの群にまとめ、子が 1 つの節を子と入れ替える。"""
    if nd.leaf:
        return nd
    if nd is not None and nd.kind != "root" and nd.horses < MIN_HORSES:
        sids = [s for c in nd.walk() if c.leaf for s in c.sires]
        return Node(nd.key, nd.name if len(sids) > 1 else names.get(sids[0], nd.name), "pool" if len(sids) > 1 else "sire",
                    sires=sids, stats=nd.stats, horses=nd.horses)
    kids = [prune(c, names) for c in nd.children]
    small = [c for c in kids if c.leaf and c.horses < MIN_HORSES]
    if len(small) >= 2:
        sids = [s for c in small for s in c.sires]
        pool = Node(nd.key + "+pool", f"ほか {len(sids)} 頭", "pool", sires=sids,
                    stats=sum((c.stats for c in small), np.zeros((len(COMPONENTS), 6))), horses=sum(c.horses for c in small))
        kids = [c for c in kids if not (c.leaf and c.horses < MIN_HORSES)] + [pool]
    nd.children = kids
    if len(kids) == 1 and nd.kind != "root":
        k = kids[0]
        if k.kind == "pool" and k.name.startswith("ほか"):      # 子が全部まとめられたときは、親の名前を残す
            k.name = f"{nd.name}（{len(k.sires)} 頭）"
        return k
    return nd


# ── 縮小推定 ──────────────────────────────────────────────


def estimate_tau2(pairs: list[tuple[float, float, float, float]]) -> float:
    """(子の生の値, 子の分散, 兄弟（同じ親の残り）の値, その分散) の組から、子が親の系統からずれる真の分散 τ²（モーメント法）。
    E[(子 − 兄弟)²] = τ² + 子の分散 + 兄弟の分散 とみなす。下限は小さな正の値（＝ほぼ全部を親へ縮める）。"""
    ok = [t for t in pairs if all(math.isfinite(u) for u in t)]
    if len(ok) < 3:
        return 1e-12
    d2 = np.array([(x - p) ** 2 for x, _, p, _ in ok])
    vs = np.array([v + pv for _, v, _, pv in ok])
    return float(max(1e-12, np.mean(d2) - np.mean(vs)))


def shrink(x: float, v: float, prior: float, prior_var: float, tau2: float) -> tuple[float, float]:
    """生の値 x（分散 v）を、親の値 prior（分散 prior_var）へ縮める。値が無ければ親のまま（分散は τ² を足す）。"""
    if not math.isfinite(x) or not math.isfinite(v):
        return prior, prior_var + tau2
    b = tau2 / (tau2 + v)
    post = prior + b * (x - prior)
    var = tau2 * v / (tau2 + v) + (1 - b) ** 2 * prior_var
    return post, var


def fit_tau2(sire_st: dict[str, np.ndarray], sire_parent: dict[str, Node], metrics: list[Metric]) -> dict[str, float]:
    """指標ごとの τ² を、種牡馬（MIN_TEST 以上）と、刈り込む前の木で同じ親の下にいる残りの産駒（兄弟）の値から推定する。"""
    tau2 = {}
    for m in metrics:
        pairs = []
        for sid, par in sire_parent.items():
            if sire_st[sid][0][0] < MIN_TEST:
                continue
            x, v, _ = metric_raw(sire_st[sid], m)
            px, pv, _ = metric_raw(par.stats - sire_st[sid], m)
            pairs.append((x, v, px, pv))
        tau2[m.key] = estimate_tau2(pairs)
    return tau2


def win_quantile(shares: np.ndarray, q: float) -> float:
    """帯ごとの割合（合計 1 に直す）から、勝利距離の q 分位（帯の中は一様とみなして補間、m）。"""
    cum = 0.0
    for i, w in enumerate(shares):
        if cum + w >= q and w > 0:
            return WIN_EDGES[i] + (q - cum) / w * (WIN_EDGES[i + 1] - WIN_EDGES[i])
        cum += w
    return float(WIN_EDGES[-1])


def add_win_spread(root: Node) -> None:
    """縮めた帯の割合から、勝利距離の幅（SPREAD_DEFS）を各節に足す。"""
    for nd in root.walk():
        sh = np.maximum(np.array([nd.est[b][0] for b in WIN_BINS], float), 0)
        n = int(nd.stats[COMPONENTS.index("windist")][0])
        for k, _, lo, hi, _ in SPREAD_DEFS:
            w = win_quantile(sh / sh.sum(), hi) - win_quantile(sh / sh.sum(), lo) if sh.sum() > 0 else math.nan
            nd.est[k] = (w, math.nan, w, n)


def fit_shrinkage(root: Node, sire_st: dict[str, np.ndarray], sire_parent: dict[str, Node]) -> dict[str, float]:
    """指標ごとの τ² を推定し（fit_tau2）、木の全部の節を根から順に親へ縮める。"""
    fitted = METRICS + EXTRA_METRICS + BIN_METRICS
    tau2 = fit_tau2(sire_st, sire_parent, fitted)

    def rec(nd: Node, prior: dict[str, tuple[float, float]]):
        for m in fitted:
            x, v, n = metric_raw(nd.stats, m)
            if nd.kind == "root":
                post, var = x, v
            else:
                post, var = shrink(x, v, *prior[m.key], tau2[m.key])
            nd.est[m.key] = (post, math.sqrt(var), x, n)
        mine = {k: (e[0], e[1] ** 2) for k, e in nd.est.items()}
        for c in nd.children:
            rec(c, mine)

    rec(root, {})
    return tau2


# ── 主成分・並べ方 ──────────────────────────────────────────


def pc1(mat: np.ndarray, use: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """列ごとに中心化・標準化した行列の第 1 主成分。(負荷, 平均, SD, 寄与率)。向きは第 1 列（芝↔ダート）が正になるように。
    use が False の列（縮めると全部が同じ値になる指標など）は使わない（負荷 0）。"""
    use = np.ones(mat.shape[1], bool) if use is None else np.asarray(use, bool)
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd[(sd == 0) | ~use] = 1.0
    z = (mat - mu) / sd
    z[:, ~use] = 0.0
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    load = vt[0]
    load[~use] = 0.0
    if load[0] < 0:
        load = -load
    return load, mu, sd, float(s[0] ** 2 / (s ** 2).sum())


def order_children(nd: Node) -> None:
    """各親の中で、子を総合（第 1 主成分）の小さい順に並べる。"""
    nd.children.sort(key=lambda c: (c.score, c.name))
    for c in nd.children:
        order_children(c)


def start_node(root: Node) -> Node:
    cur = root
    while cur.children:
        big = max(cur.children, key=runs)
        if runs(big) < START_SHARE * runs(root) or big.leaf:
            break
        cur = big
    return cur


# ── 検定 ──────────────────────────────────────────────


def eta2(x: np.ndarray, w: np.ndarray, g: np.ndarray) -> float:
    """重み付きの、群で説明される分散の割合。"""
    mu = np.sum(w * x) / np.sum(w)
    sst = np.sum(w * (x - mu) ** 2)
    if sst <= 0:
        return 0.0
    ssb = 0.0
    for k in np.unique(g):
        m = g == k
        wk = w[m].sum()
        ssb += wk * (np.sum(w[m] * x[m]) / wk - mu) ** 2
    return float(ssb / sst)


def perm_test(x: np.ndarray, w: np.ndarray, g: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> tuple[float, float, float]:
    """(η², 並べ替えでの η² の平均, p 値)。群のラベルを種牡馬の間で並べ替える。"""
    rng = np.random.default_rng(seed)
    obs = eta2(x, w, g)
    _, inv = np.unique(g, return_inverse=True)
    k = inv.max() + 1
    mu = np.sum(w * x) / np.sum(w)
    sst = np.sum(w * (x - mu) ** 2)
    wx = w * x
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(inv)
        W = np.bincount(p, weights=w, minlength=k)
        S = np.bincount(p, weights=wx, minlength=k)
        null[i] = np.sum(W * (S / W - mu) ** 2) / sst
    pval = (1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1)
    return obs, float(null.mean()), float(pval)


def holm(ps: list[float]) -> list[float]:
    p = np.asarray(ps, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (n - r) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ── 集計の本体 ──────────────────────────────────────────────


def analyze(df: pd.DataFrame, origin: dict[str, dict[str, Any]]) -> dict[str, Any]:
    st = sire_stats(df)
    horses = {str(k): int(v) for k, v in df.groupby("sire_id")["horse_id"].nunique().items()}
    sires = sorted(s for s in st if s in origin)       # 並べ替え検定・表の並びが実行ごとに変わらないよう、順を固定する
    names = {s: strip_country(origin[s]["name"]) for s in sires}
    full = build_tree(origin, sires)
    aggregate(full, st, horses)
    # 検定・τ² の推定用: 種牡馬ごとの、刈り込む前の木の親
    sire_parent: dict[str, Node] = {}
    for nd in full.walk():
        for c in nd.children:
            if c.kind == "sire":
                sire_parent[c.sires[0]] = nd
    root = prune(full, names)
    aggregate(root, st, horses)
    label_pools(root, names, horses)
    tau2 = fit_shrinkage(root, st, sire_parent)
    add_win_spread(root)
    tau2.update(fit_tau2(st, sire_parent, ODDS_METRICS))

    # 総合: 種牡馬（木の葉で 1 頭のもの、MIN_HORSES 頭以上）の縮めた値の第 1 主成分
    leaves = [nd for nd in root.walk() if nd.kind == "sire" and nd.horses >= MIN_HORSES]
    mat = np.array([[nd.est[m.key][0] for m in METRICS] for nd in leaves])
    flat = {m.key for m in METRICS if tau2[m.key] <= 1e-10}       # 種牡馬の間の違いが偶然の範囲（全部が親の値になる）
    flat_x = {m.key for m in EXTRA_METRICS if tau2[m.key] <= 1e-10}
    load, mu, sd, share = pc1(mat, np.array([m.key not in flat for m in METRICS]))
    for nd in root.walk():
        z = (np.array([nd.est[m.key][0] for m in METRICS]) - mu) / sd
        nd.score = float(z @ load)
    order_children(root)

    # 検定
    test_sires = [s for s in sires if st[s][0][0] >= MIN_TEST]
    line_of = {s: line_path(origin[s]["lines"])[-1] for s in test_sires}
    father_of = {s: strip_country(origin[s]["chain"][0][1]) if origin[s]["chain"] else "?" for s in test_sires}
    tests = []
    raw_cache = {s: {m.key: metric_raw(st[s], m) for m in METRICS + ODDS_METRICS} for s in test_sires}
    for gname, gmap in [("系統", line_of), ("父", father_of)]:
        cnt = pd.Series(gmap).value_counts()
        for m in METRICS + [None] + ODDS_METRICS:
            if m is None:
                xs, vs, ok = [], [], []
                for s in test_sires:
                    vals = [raw_cache[s][mm.key] for mm in METRICS]
                    if all(math.isfinite(v[0]) for v in vals):
                        z = (np.array([v[0] for v in vals]) - mu) / sd
                        xs.append(float(z @ load)); vs.append(1e-9); ok.append(s)
                key, label = "pc1", "総合（第1主成分）"
                tau = 1.0
            else:
                ok = [s for s in test_sires if math.isfinite(raw_cache[s][m.key][0])]
                xs = [raw_cache[s][m.key][0] for s in ok]
                vs = [raw_cache[s][m.key][1] for s in ok]
                key, label, tau = m.key, m.label, tau2[m.key]
            g = np.array([gmap[s] if cnt[gmap[s]] >= MIN_GROUP else "その他" for s in ok])
            x = np.array(xs); w = 1.0 / (tau + np.array(vs))
            e, e0, p = perm_test(x, w, g)
            tests.append({"by": gname, "key": key, "label": label, "eta2": e, "null": e0, "adj": (e - e0) / (1 - e0) if e0 < 1 else 0.0,
                          "p": p, "n_sires": len(ok), "n_groups": len(set(g))})
    q = holm([t["p"] for t in tests])
    for t, qq in zip(tests, q):
        t["q"] = qq

    return {"root": root, "start": start_node(root), "tau2": tau2, "flat": flat, "flat_x": flat_x, "load": load, "mu": mu, "sd": sd, "share": share,
            "tests": tests, "test_sires": test_sires, "line_of": line_of, "father_of": father_of, "st": st, "names": names,
            "raw": raw_cache, "n_runs": len(df), "n_horses": df["horse_id"].nunique(), "n_sires": len(sires),
            "n_leaves": sum(1 for nd in root.walk() if nd.leaf), "origin": origin, "full": full}


# ── 記事 ──────────────────────────────────────────────


def fmt_val(v: float, fmt: str) -> str:
    if not math.isfinite(v):
        return "–"
    if fmt == "pct":
        return f"{v * 100:.1f}%"
    if fmt == "m":
        return "±0m" if round(v) == 0 else f"{v:+.0f}m"
    if fmt == "pos":
        return f"{v:.2f}"
    if fmt == "dist":
        return f"{v:,.0f}m"
    if fmt == "val1":
        return f"{v:.1f}"
    if fmt == "val2":
        return f"{v:.2f}"
    if fmt == "pt":
        return f"{v * 100:+.1f}pt"
    return f"{v:+.2f}"


def node_json(nd: Node, ids: dict[int, str], center: dict[str, float]) -> dict[str, Any]:
    ms = []
    for m in METRICS:
        post, sd, raw, n = nd.est[m.key]
        r = ROUND[m.fmt]
        ms.append([round(post, r), round(post - 1.96 * sd, r), round(post + 1.96 * sd, r), n])
    ms.append([round(nd.score, 2), None, None, runs(nd)])
    for m in SHOWN_EXTRA:
        post, sd, raw, n = nd.est[m.key]
        r = ROUND[m.fmt]
        ci = [None, None] if not math.isfinite(sd) else [round(post - 1.96 * sd, r), round(post + 1.96 * sd, r)]
        ms.append([round(post, r), *ci, n])
    o: dict[str, Any] = {"id": ids[id(nd)], "name": nd.name, "value": nd.horses, "metrics": ms}
    if nd.kind in ("pool", "branch", "line"):
        o["kind"] = nd.kind
    if nd.kind == "pool":
        o["sires"] = len(nd.sires)
        o["members"] = [[n, h] for n, h in nd.members]
    if nd.children:
        o["children"] = [node_json(c, ids, center) for c in nd.children]
    return o


def pc1_label(load: np.ndarray, side: int) -> str:
    """第 1 主成分の片側（side=+1 / −1）の意味を、負荷の大きい指標から作る。"""
    parts = []
    for i in np.argsort(-np.abs(load)):
        if abs(load[i]) < 0.3 or len(parts) >= 3:
            continue
        m = METRICS[i]
        parts.append((m.high if load[i] * side > 0 else m.low).replace("寄り", "").replace("（2〜3歳）", "").replace("（4歳以上）", ""))
    return "・".join(parts) + "寄り"


def treemap_block(a: dict[str, Any]) -> dict[str, Any]:
    root: Node = a["root"]
    ids = {id(nd): f"n{i}" for i, nd in enumerate(root.walk())}
    center = {m.key: root.est[m.key][0] for m in METRICS + SHOWN_EXTRA}
    leaves = [nd for nd in root.walk() if nd.leaf]

    def entry(m: Metric) -> dict[str, Any]:
        dev = np.array([abs(nd.est[m.key][0] - center[m.key]) for nd in leaves])
        scale = float(np.percentile(dev, 95))
        r = ROUND[m.fmt]
        desc = m.desc
        if m.key in a["flat"] or m.key in a["flat_x"]:
            scale = 0.03 if m in METRICS else 10 ** -r
            desc += "。種牡馬の間の違いが偶然の範囲に収まり、縮めるとどの種牡馬も全体の値になる（色が付かない）"
        return {"key": m.key, "label": m.label, "kind": m.kind, "low": m.low, "high": m.high, "format": m.fmt,
                "center": round(center[m.key], r), "scale": round(max(scale, 10 ** -r), r + 1), "desc": desc}

    metrics = [entry(m) for m in METRICS]
    dev = np.array([abs(nd.score) for nd in leaves])
    metrics.append({"key": "pc1", "label": "総合（第1主成分）", "kind": "総合", "low": pc1_label(a["load"], -1), "high": pc1_label(a["load"], +1),
                    "format": "num", "center": 0, "scale": round(float(np.percentile(dev, 95)), 2),
                    "desc": f"{len(METRICS) - len(a['flat'])} つの指標をまとめた 1 軸（ばらつきの {a['share'] * 100:.0f}% を説明）"})
    metrics += [entry(m) for m in SHOWN_EXTRA]
    return {"type": "treemap", "layout": "radial", "valueUnit": "頭",
            "title": "種牡馬の系統樹（円形、色＝適性、似た適性の子が隣り合う）",
            "metrics": metrics, "defaultMetric": 0, "start": ids[id(a["start"])], "maxDepth": 3,
            "root": node_json(root, ids, center),
            "caption": (f"{min(YEARS)}〜{max(YEARS)}年 JRA 平地の産駒の出走。中心が表示中の系統で、外へ向かって父系が枝分かれする。"
                        "枝の先の丸が種牡馬（内側の丸は系統・父系の祖先の群）。丸の中に数字がある枝は、まだ配下に種牡馬がいるのでタップでその系統へ潜り、上のパンくずで戻れる。"
                        "各親の中では、似た適性の子が隣り合うよう総合（第 1 主成分）の順に時計回りに並べている。"
                        f"色は縮小推定（産駒の少ない種牡馬は親の系統の値へ寄せる）の値。産駒が {MIN_HORSES} 頭未満の種牡馬は群にまとめ、含まれる種牡馬の名前を群の名前と選択時の一覧に出した。"
                        "「色」のプルダウンでは、適性の指標のほかに、勝利距離平均とその幅（どの距離でも勝てるか）、走法の軸（実測ラップのある 2023 年以降の出走だけ）も選べる。これらは色で見るだけで、総合や検定には入れていない。")}


def article(a: dict[str, Any]) -> dict[str, Any]:
    tests = a["tests"]
    by_line = [t for t in tests if t["by"] == "系統"]
    by_father = {t["key"]: t for t in tests if t["by"] == "父"}
    st, raw, names = a["st"], a["raw"], a["names"]
    fmts = {m.key: m.fmt for m in METRICS}

    def qtxt(q: float) -> str:
        return "<0.001" if q < 0.001 else f"{q:.3f}"

    main_keys = [m.key for m in METRICS]
    sig = [t for t in by_line if t["key"] in main_keys and t["q"] < 0.05]
    strong = [t for t in sig if t["adj"] >= 0.15]
    weak = [t for t in by_line if t["key"] in main_keys and t["q"] >= 0.05]
    ranked = sorted([t for t in by_line if t["key"] in main_keys], key=lambda t: -t["adj"])
    father_sig = [by_father[t["key"]] for t in weak if by_father[t["key"]]["q"] < 0.05]
    indiv = [t for t in weak if by_father[t["key"]]["q"] >= 0.05]

    # 系統ごとの平均（検定の種牡馬の生の値を、重み 1/(τ²+v) で平均）と、系統の中のばらつき
    line_of = a["line_of"]
    groups: dict[str, list[str]] = {}
    for s in a["test_sires"]:
        groups.setdefault(line_of[s], []).append(s)
    big = sorted([g for g, ss in groups.items() if len(ss) >= MIN_GROUP], key=lambda g: -len(groups[g]))

    def gmean(g: str, key: str) -> tuple[float, float]:
        xs = [(raw[s][key][0], 1 / (a["tau2"][key] + raw[s][key][1])) for s in groups[g] if math.isfinite(raw[s][key][0])]
        if not xs:
            return math.nan, math.nan
        w = sum(v for _, v in xs)
        return sum(x * v for x, v in xs) / w, 1.96 / math.sqrt(w)

    # 系統の差の表
    test_rows = []
    order = main_keys + ["pc1"] + [m.key for m in ODDS_METRICS]
    for t in sorted(by_line, key=lambda t: order.index(t["key"])):
        f = by_father[t["key"]]
        kind = next((m.kind for m in METRICS + ODDS_METRICS if m.key == t["key"]), "総合")
        test_rows.append([t["label"], kind, f"{t['adj'] * 100:.0f}%", qtxt(t["q"]), f"{f['adj'] * 100:.0f}%", qtxt(f["q"])])

    # 系統ごとの値の表（主な系統）
    line_rows = []
    for g in big:
        row: list[Any] = [g, len(groups[g])]
        for m in METRICS:
            v, h = gmean(g, m.key)
            row.append(f"{fmt_val(v, m.fmt)}" if m.fmt != "pt" else f"{fmt_val(v, m.fmt)}±{h * 100:.1f}")
        line_rows.append(row)

    # 系統の中のばらつき: 主な系統ごとに、各指標の最小・最大の種牡馬
    spread_rows = []
    for g in big[:6]:
        for m in METRICS[:3]:
            ss = [s for s in groups[g] if math.isfinite(raw[s][m.key][0])]
            lo = min(ss, key=lambda s: raw[s][m.key][0]); hi = max(ss, key=lambda s: raw[s][m.key][0])
            spread_rows.append([g, m.label, f"{names[lo]} {fmt_val(raw[lo][m.key][0], m.fmt)}", f"{names[hi]} {fmt_val(raw[hi][m.key][0], m.fmt)}"])

    # 系統の差が大きい指標の、両端の系統
    def extremes(key: str) -> tuple[str, str]:
        vals = [(g, gmean(g, key)[0]) for g in big]
        vals = [v for v in vals if math.isfinite(v[1])]
        lo = min(vals, key=lambda v: v[1]); hi = max(vals, key=lambda v: v[1])
        f = fmts[key]
        return f"{lo[0]}（{fmt_val(lo[1], f)}）", f"{hi[0]}（{fmt_val(hi[1], f)}）"

    # 父が同じ群の表（父で有意な指標）
    odds_rows = [t for t in by_line if t["key"] in {m.key for m in ODDS_METRICS}]
    father_keys = [m.key for m in METRICS if by_father[m.key]["q"] < 0.05]
    fgroups: dict[str, list[str]] = {}
    for s_ in a["test_sires"]:
        fgroups.setdefault(a["father_of"][s_], []).append(s_)
    father_rows = []
    for f, ss in sorted(fgroups.items(), key=lambda kv: -len(kv[1])):
        if len(ss) < MIN_GROUP:
            continue
        ss = sorted(ss, key=lambda s_: -st[s_][0][0])
        row: list[Any] = [f, len(ss), "・".join(names[s_] for s_ in ss[:4])]
        for k in father_keys:
            xs = [(raw[s_][k][0], 1 / (a["tau2"][k] + raw[s_][k][1])) for s_ in ss if math.isfinite(raw[s_][k][0])]
            w = sum(v for _, v in xs)
            row.append(fmt_val(sum(x * v for x, v in xs) / w, fmts[k]))
        father_rows.append(row)

    if not sig:
        verdict = "not_supported"
    elif len(strong) == len(METRICS):
        verdict = "supported"
    else:
        verdict = "partial"

    top = ranked[0]
    lo_e, hi_e = extremes(top["key"])
    sig_names = "・".join(t["label"] for t in sorted(sig, key=lambda t: -t["adj"])) or "なし"
    weak_names = "・".join(t["label"] for t in weak) or "なし"
    pc1t = next(t for t in by_line if t["key"] == "pc1")
    tm = treemap_block(a)
    pcm = tm["metrics"][-1]
    load_txt = "、".join(f"{m.label} {a['load'][i]:+.2f}" for i, m in enumerate(METRICS))
    tau_txt = "、".join(f"{m.label} {fmt_val(math.sqrt(a['tau2'][m.key]), 'pt' if m.fmt == 'pt' else m.fmt).lstrip('+')}" for m in METRICS)

    summary = (f"主要 {len(a['test_sires'])} 種牡馬で、系統（名前の付いた父系）の違いが偶然を超えて効くのは {sig_names}"
               f"（多重比較の補正後 q<0.05）。もっとも系統で説明できるのは{top['label']}で、種牡馬の間の違いの約 {top['adj'] * 100:.0f}%。"
               + (f"{'・'.join(t['label'] for t in father_sig)}は父が同じ群まで細かくすると差が出ます。" if father_sig else "")
               + (f"{'・'.join(t['label'] for t in indiv)}は、系統より個々の種牡馬で見るほうがよい結果でした。" if indiv else ""))

    body: list[dict[str, Any]] = [
        {"type": "text", "text": (
            "「キングカメハメハ系はダート」「サンデー系は芝の中距離」のように、血統は系統（父系）でまとめて語られることが多いです。"
            "では、系統でまとめると、産駒の適性の違いはどれくらい説明できるのでしょうか。\n\n"
            f"{min(YEARS)}〜{max(YEARS)}年の JRA 平地 {a['n_runs']:,} 走（{a['n_horses']:,} 頭・{a['n_sires']:,} 種牡馬）を、"
            "smartrc の系統（大系統 → 小系統 → 各種牡馬）の木に並べ、5 つの適性の指標で色を塗りました。")},
        {"type": "stats", "items": [
            {"label": "種牡馬", "value": f"{a['n_sires']:,}", "sub": f"検定は産駒 {MIN_TEST} 走以上の {len(a['test_sires'])} 頭"},
            {"label": "主な系統（検定）", "value": f"{top['n_groups']}", "sub": f"{MIN_GROUP} 頭未満の系統は「その他」"},
            {"label": f"系統で説明（{top['label']}）", "value": f"{top['adj'] * 100:.0f}%", "sub": f"q={qtxt(top['q'])}"},
            {"label": "系統で差があった指標", "value": f"{len(sig)} / {len(METRICS)}", "sub": "Holm 補正後 q<0.05"},
        ]},
        tm,
        {"type": "note", "text": (
            "指標は「どこで走るか（分布）」と「どの条件で成績が上がるか（成績）」の 2 種類です。芝↔ダート・距離・脚質は産駒の出走の分布なので、能力の高い低いは入りません。"
            "道悪・早熟↔晩成は、同じ種牡馬の産駒の中で、条件による複勝率（頭数からの期待 3/頭数 との差）の上がり下がりを比べたもので、"
            "種牡馬の能力の高さは両方の条件に同じだけ入るので打ち消し合います（人気は考えていません。人気との差の版は検定の表に載せました）。")},
        {"type": "heading", "text": "指標の作り方"},
        {"type": "table", "columns": ["指標", "種類", "中身", "全体の値"],
         "rows": [[m.label, m.kind, m.desc, fmt_val(a["root"].est[m.key][0], m.fmt)] for m in METRICS]
                 + [[pcm["label"], "総合", f"{pcm['desc']}。負荷: {load_txt}", "0"]],
         "caption": (f"脚質は今走の初角の位置で、予想に使う指標ではなく産駒の走り方の記述です。総合の両端は「{pcm['low']}」↔「{pcm['high']}」。")},
        {"type": "text", "text": (
            "産駒が少ない種牡馬の値はぶれるので、経験ベイズで親の系統の値へ縮めています（産駒が多いほど自身の値に近く、少ないほど親に近い）。"
            "系統（木の内側の節）の値は、配下の産駒の出走をまとめて出しています。系統樹の色とホバー・タップで出る値・95% 区間は、この縮めた値です。")},
        {"type": "heading", "text": "系統で適性に違いはあるか"},
        {"type": "text", "text": (
            f"産駒の出走 {MIN_TEST} 以上の {len(a['test_sires'])} 種牡馬について、指標ごとに「種牡馬の間の違いのうち、系統の違いで説明できる割合」を出しました。"
            "系統のラベルを種牡馬の間でランダムに入れ替えても、系統が多ければある程度の割合は偶然で出るので、その分を引いた値を載せ、"
            f"入れ替え（{N_PERM:,} 回）で p 値を出して、指標 × 2 つの分け方の {len(tests)} 個を Holm で補正しました。\n\n"
            "分け方は 2 つです。「系統」は smartrc の小系統（ディープ系・Tサンデー系・キングマンボ系・ロベルト系など。smartrc で系統が空欄の種牡馬は netkeiba の系統）、"
            "「父」は種牡馬の父が同じ群（ディープインパクトの産駒の種牡馬など、細かい分け方）です。")},
        {"type": "table", "columns": ["指標", "種類", "系統で説明", "q（系統）", "父で説明", "q（父）"], "rows": test_rows,
         "caption": "説明の割合は、偶然の分（ラベルを入れ替えたときの平均）を引いたもの。q は Holm で補正した p 値。種牡馬の生の値を、標本の精度で重み付けしている。"},
        {"type": "text", "text": (
            f"系統で違いがはっきりしたのは {sig_names} です。中でも {top['label']}は、系統だけで種牡馬の間の違いの約 {top['adj'] * 100:.0f}% を説明し、"
            f"主な系統のうち {lo_e} から {hi_e} まで開きます。"
            + (f"一方、{weak_names}は、名前の付いた系統では違いが偶然の範囲を出ませんでした。" if weak else "")
            + ("ただし父が同じ群という細かい分け方にすると、"
               + "、".join(f"{t['label']}は約 {t['adj'] * 100:.0f}%（q={qtxt(t['q'])}）" for t in father_sig)
               + "を説明します。大きな系統の中で、父ごとに分かれていると読めます。" if father_sig else "")
            + (f"\n\n{'・'.join(METRIC_BY_KEY[k].label for k in a['flat'])}は、系統以前に、種牡馬の間の違いそのものが偶然のぶれの範囲に収まりました"
               "（縮小推定の τ が 0。系統樹では色が付きません）。" if a["flat"] else "")
            + (f"\n\n人気との差の版（{'・'.join(t['label'] for t in odds_rows)}）も、どちらの分け方でも補正後に有意ではありませんでした。"
               "系統で道悪・成長力を狙っても、人気以上の上積みは見込みにくいということです。" if odds_rows and all(t["q"] >= 0.05 and by_father[t["key"]]["q"] >= 0.05 for t in odds_rows) else ""))},
        {"type": "table", "columns": ["系統", "種牡馬"] + [m.label for m in METRICS], "rows": line_rows,
         "caption": f"産駒 {MIN_TEST} 走以上の種牡馬が {MIN_GROUP} 頭以上いる系統。値は種牡馬の生の値の精度での加重平均（道悪・早熟↔晩成は ±95% 区間の半幅）。"},
        *([{"type": "table", "columns": ["父（種牡馬の父）", "種牡馬", "主な種牡馬"] + [METRIC_BY_KEY[k].label for k in father_keys], "rows": father_rows,
            "caption": f"産駒 {MIN_TEST} 走以上の種牡馬が {MIN_GROUP} 頭以上いる父。値は種牡馬の生の値の精度での加重平均。主な種牡馬は産駒の出走の多い順に 4 頭まで。"}]
          if father_rows else []),
        {"type": "heading", "text": "同じ系統の中でも分かれる"},
        {"type": "text", "text": (
            "系統の平均に差があっても、系統の中のばらつきは大きいままです。主な系統ごとに、芝↔ダート・距離・脚質の両端の種牡馬を並べました。"
            "同じ系統の中で、芝の種牡馬とダートの種牡馬にはっきり分かれる例が多くあります。")},
        {"type": "table", "columns": ["系統", "指標", "いちばん小さい", "いちばん大きい"], "rows": spread_rows,
         "caption": f"産駒 {MIN_TEST} 走以上の種牡馬の生の値。"},
        {"type": "heading", "text": "まとめ: 系統で語れる指標・個々の種牡馬で見る指標"},
        {"type": "text", "text": conclusion_text(by_father, sig, weak, father_sig, a["flat"], qtxt)},
        {"type": "accordion", "title": "注記（系統の木の作り方・縮小推定・限界）", "blocks": [
            {"type": "text", "text": (
                "・系統の木: smartrc の系統の分類（大系統 11・小系統 46）で、大系統（例: サンデー系）→ 小系統（例: ディープ系）→ 種牡馬 と並べています"
                "（2026-11-06 に netkeiba の「○○系」から切り替え。smartrc の分類のほうが細かく、サンデー系をディープ系・Tサンデー系などに分ける）。"
                f"smartrc で系統が空欄の種牡馬（比較的新しく輸入された海外の種牡馬など、{sum(1 for v in a['origin'].values() if v['lines'][-1] == UNCLASSIFIED)} 頭）は、"
                "netkeiba の最も下の系統を使い、「smartrc 未分類」の下に置いています。\n\n"
                f"・子が 1 つしかない節は、子と入れ替えて段を詰めています。最初は全体の {START_SHARE * 100:.0f}% 以上を占める系統まで潜った位置から表示します。\n\n"
                f"・縮小推定: 指標ごとに、種牡馬の値が親の値からずれる真の SD（τ）を、ずれの 2 乗の平均から標本のぶれを引いて推定しました（{tau_txt}）。"
                "標本のぶれは、同じ馬の出走をまとめたクラスタの分散で、同じ馬が何度も同じ条件を走る分を考えています。"
                "道悪・早熟↔晩成の差の分散は、2 つの条件を独立とみなして足しているので、やや大きめ（保守的）です。\n\n"
                "・早熟↔晩成は、4 歳以上の産駒がいない若い種牡馬では値が出ず、親の系統の値になります。2〜3 歳で引退した馬は 4 歳以上に入らないので、生き残りの偏りがあります。"
                f"また 4 歳以上はクラスが上がって全体に複勝率が下がる（全体で {fmt_val(a['root'].est['mature'][0], 'pt')}）ので、値は全体との相対で読んでください。"
                "若いころによく走った産駒が多い種牡馬ほど、勝ち上がってクラスが上がる分、早熟寄りに出やすい面もあります（記事『エピファネイアは早熟？』を参照）。\n\n"
                "・道悪・早熟↔晩成は、芝・ダートそれぞれで差を出し、精度で重み付けて平均しています（ダートの多い種牡馬ほど道悪の出走が多い、といった芝ダの偏りを除くため）。"
                "人気との差の版は、頭数からの期待の代わりに単勝オッズから見込まれる複勝率を引いたもので、市場がすでに知っている適性を除いた「人気より走るか」です。\n\n"
                "・検定は種牡馬を 1 単位とした比較です。系統の間で種牡馬の数・時代が偏っている（新しい系統ほど若い種牡馬が多い）ことは補正していません。"
                "系統の記録には、取得元の表記に依存する部分があります。")},
        ]},
    ]
    return {
        "slug": "sire-line-aptitude",
        "title": "種牡馬系統による適性の違いはあるのか",
        "claim": "種牡馬の系統（父系）で、産駒の適性が決まる",
        "verdict": verdict,
        "summary": summary,
        "categories": ["bloodline/lineage"],
        "published": "2026-11-04",
        "updated": "2026-11-06",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年",
                    "conditions": "JRA全競馬場・芝ダート（障害除く）。系統は smartrc の大系統・小系統（空欄の種牡馬は netkeiba の系統）",
                    "sample": f"{a['n_runs']:,} 走（{a['n_horses']:,} 頭・{a['n_sires']:,} 種牡馬）。検定は産駒 {MIN_TEST} 走以上の {len(a['test_sires'])} 種牡馬"},
        "body": body,
    }


def conclusion_text(by_father, sig, weak, father_sig, flat, qtxt) -> str:
    good = sorted(sig, key=lambda t: -t["adj"])
    fkeys = {t["key"] for t in father_sig}
    indiv = [t for t in weak if t["key"] not in fkeys]
    parts = []
    if good:
        parts.append("系統で語るのが有効な指標: " + "、".join(f"{t['label']}（系統で約 {t['adj'] * 100:.0f}%、q={qtxt(t['q'])}）" for t in good) + "。"
                     "ただし系統で説明できるのは種牡馬の間の違いの一部（2 割前後）で、残りは同じ系統の中の違いです。"
                     "系統は「当たりを付ける」目安にとどめ、最後は個々の種牡馬の値を見るのがよいでしょう。")
    if father_sig:
        parts.append("細かい父系（父が同じ群）でなら語れる指標: " + "、".join(f"{t['label']}（父で約 {t['adj'] * 100:.0f}%、q={qtxt(t['q'])}）" for t in father_sig) + "。"
                     "「○○系」という大きなくくりでは平均すると消えますが、どの種牡馬の子かまで下りると似た傾向が出ます。")
    if indiv:
        parts.append("個々の種牡馬で見るべき指標: " + "、".join(f"{t['label']}（系統で約 {max(t['adj'], 0) * 100:.0f}%、q={qtxt(t['q'])}）" for t in indiv) + "。"
                     "系統でも父でも違いは偶然の範囲で、種牡馬ごとの値（系統樹のホバー・タップ）か、その種牡馬の産駒の成績を直接見るほうが確かです。"
                     + ("とくに道悪は、種牡馬の間の違い自体が確かめられませんでした。道悪の得意・苦手は、6 年分のデータでも、種牡馬の単位では偶然のぶれと区別できない大きさです。" if "going" in flat else ""))
    return "\n\n".join(parts)


def _finite(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_finite(x) for x in v]
    return v


def dumps_article(col: dict[str, Any]) -> str:
    """記事の JSON。系統樹の木は 1 行（空白なし）にまとめ、ほかは early_maturing_bloodline.dumps と同じ書き方。"""
    from src.research.columns.early_maturing_bloodline import dumps
    trees = {}
    for i, b in enumerate(col["body"]):
        if b.get("type") == "treemap":
            key = f"__TREE_{i}__"
            trees[key] = json.dumps(b["root"], ensure_ascii=False, separators=(",", ":"))
            b["root"] = key
    text = dumps(col)
    for key, t in trees.items():
        text = text.replace(json.dumps(key), t)
    for i, b in enumerate(col["body"]):
        if b.get("type") == "treemap":
            b["root"] = json.loads(trees[f"__TREE_{i}__"])
    return text


def report(a: dict[str, Any]) -> None:
    print(f"runs={a['n_runs']} horses={a['n_horses']} sires={a['n_sires']} leaves={a['n_leaves']} test_sires={len(a['test_sires'])} start={a['start'].name}")
    print("tau:", {k: round(math.sqrt(v), 4) for k, v in a["tau2"].items()})
    print("pc1 load:", np.round(a["load"], 2), "share", round(a["share"], 3))
    for t in a["tests"]:
        print(f"  {t['by']:3s} {t['label']:12s} eta2={t['eta2']:.3f} null={t['null']:.3f} adj={t['adj']:.3f} p={t['p']:.4f} q={t['q']:.4f} sires={t['n_sires']} groups={t['n_groups']}")


UNCLASSIFIED = "smartrc 未分類"


def smartrc_origin(origin: dict[str, dict[str, Any]], smartrc: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """系統の経路を smartrc の 大系統 → 小系統 に差し替えた origin（lines は新しい→古い）。父系の祖先の枝は使わず、父だけ残す。
    smartrc で系統が空欄の種牡馬は、netkeiba の最も下の系統を「smartrc 未分類」の下に置く。"""
    if smartrc is None:
        from src.scraper import smartrc_sire_lines as ssl

        smartrc = ssl.load().get("sires", {})
    out = {}
    for sid, rec in origin.items():
        sm = smartrc.get(sid)
        if sm and sm.get("small_name"):
            lines = [sm["small_name"], sm.get("large_name")]
        else:
            nk = line_path(rec.get("lines") or [])
            lines = [nk[-1] if nk else "系統不明", UNCLASSIFIED]
        out[sid] = {**rec, "lines": lines, "chain": rec.get("chain", [])[:1], "origin_line": lines[-1]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    origin = smartrc_origin(json.loads(ORIGIN_PATH.read_text(encoding="utf-8")))
    a = analyze(frame(), origin)
    report(a)
    col = _finite(article(a))      # 値の無い軸（NaN）は null に（JSON に NaN を書かない）
    text = dumps_article(col)
    assert json.loads(text) == col
    compact = len(json.dumps(col, ensure_ascii=False, separators=(",", ":")).encode())
    print(f"json: {len(text.encode()) / 1024:.0f}KB (compact {compact / 1024:.0f}KB)")
    if args.dry_run:
        return
    OUT.write_text(text, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
