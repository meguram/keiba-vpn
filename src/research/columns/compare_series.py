"""めぐコラムの比較グラフ（『その条件』と『それ以外』）を実データから集計し、記事の JSON に書き込む。

「ある条件で傾向があるか」を言うには、条件に当てはまる出走と当てはまらない出走を同じ物差しで並べ、
差が偶然の範囲を超えているかを確かめる必要がある。ここでは記事ごとに 2 系列（条件 / それ以外）を集計し、
各点に標本数・95% 信頼区間、横軸の点ごとに 2 系列の差の p 値（Welch の z 検定。0/1 の率なら 2 標本の比率の検定と同じ）を付ける。

対象: 2020〜2025 年の JRA 平地（障害を除く）。元データは data/page_reference/tables/<年>/race_result_flat.parquet。

使い方:
    python -m src.research.columns.compare_series            # 集計して frontend/content/columns/*.json を書き換える
    python -m src.research.columns.compare_series --dry-run  # 書き換えずに結果だけ出す
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FLAT_DIR = REPO / "data/page_reference/tables"
SIRE_MAP_PATH = REPO / "data/page_reference/pedigree_race_index/horse_sire_map.parquet"
RACE_QUALITY_PATH = REPO / "data/local/modeling/shared_cache/race_quality/race_quality.parquet"      # レースごとのペース区分（pace_cls）
COLUMNS_DIR = REPO / "frontend/content/columns"
YEARS = range(2020, 2026)

POP_BANDS = ["1〜3番人気", "4〜6番人気", "7番人気以下"]
DIST_BANDS = ["短距離\n〜1400m", "マイル\n1401〜1800m", "中距離\n1801〜2200m", "長距離\n2201m〜"]
MAJOR_VENUES = {"東京", "中山", "京都", "阪神"}
# 前走と今走の地域（記事『輸送は成績を落とす』の区分: 関東/関西/北海道/中部/九州）
REGION = {"東京": "関東", "中山": "関東", "福島": "関東", "新潟": "関東", "京都": "関西", "阪神": "関西",
          "中京": "中部", "札幌": "北海道", "函館": "北海道", "小倉": "九州"}


# 障害レースの見分け方。race_result_flat の surface は障害レースの多くを「芝」「ダート」で持つ（2020〜2025 年で約 6,600 出走）ので、
# surface だけでは除けない。レース名（「障害」・重賞やOPの「〜ジャンプS」「〜JS」「スプリングJ」「(JGI〜III)」）・クラスと、
# 平地に無い距離（2,600m 超で 3,000/3,200/3,400/3,600m 以外）のどれかに当たれば障害とする。
JUMP_NAME_RE = r"障害|ジャンプ|\(JG|JS\(|スプリングJ\("
FLAT_LONG_DISTANCES = (3000, 3200, 3400, 3600)


def jump_mask(r: pd.DataFrame) -> pd.Series:
    """障害レースの行なら True（列 race_name・race_class・surface・distance を使う）。"""
    name = r["race_name"].fillna("").astype(str).str.contains(JUMP_NAME_RE)
    cls = r["race_class"].fillna("").astype(str).str.contains("障害")
    dist = pd.to_numeric(r["distance"], errors="coerce")
    odd_dist = (dist > 2600) & ~dist.isin(FLAT_LONG_DISTANCES)
    return name | cls | r["surface"].eq("障") | odd_dist


def load() -> pd.DataFrame:
    cols = ["race_id", "date", "venue", "surface", "distance", "field_size", "finish_position", "horse_id",
            "sex_age", "jockey_id", "jockey_name", "odds", "popularity", "grade", "race_name", "race_class"]
    df = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=cols) for y in YEARS], ignore_index=True)
    df = df[df["surface"].isin(["芝", "ダート"]) & ~jump_mask(df)].drop(columns=["race_name", "race_class"]).copy()
    # 出走していない馬（取消・除外。オッズも人気もない）は除く。中止・失格など走った馬は「3着以内でない」として残す
    df = df[(df["odds"] > 0) & (df["popularity"] > 0)].copy()
    df["starters"] = df.groupby("race_id")["race_id"].transform("size")      # 実際に走った頭数（field_size は取消・除外を含む）
    df["horse_id"] = df["horse_id"].astype(str)
    df["win"] = (df["finish_position"] == 1).astype(float)
    df["top3"] = df["finish_position"].between(1, 3).astype(float)
    df["ret"] = np.where(df["finish_position"] == 1, df["odds"].fillna(0), 0.0)      # 単勝 100 円あたりの払戻 / 100
    df["sex"] = df["sex_age"].str[0]
    q = 1.0 / df["odds"]
    df["implied"] = q / q.groupby(df["race_id"]).transform("sum")          # オッズから見込まれる勝率（控除率を除いて合計 1 に直す）
    has_f = df["sex"].eq("牝").groupby(df["race_id"]).transform("any")
    has_m = df["sex"].isin(["牡", "セ"]).groupby(df["race_id"]).transform("any")
    df["mixed_sex"] = has_f & has_m                                          # 牝馬と牡馬（セン馬）が一緒に走ったレース（牝馬限定戦などを除く）
    df["age"] = pd.to_numeric(df["sex_age"].str[1:], errors="coerce")
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["pop_band"] = pd.cut(df["popularity"], [0, 3, 6, 99], labels=POP_BANDS)
    df["dist_band"] = pd.cut(df["distance"], [0, 1400, 1800, 2200, 9999], labels=DIST_BANDS)
    # 同じ馬の前走（平地のみ・日付順）
    df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    prev = df.groupby("horse_id")
    df["prev_surface"] = prev["surface"].shift()
    df["prev_venue"] = prev["venue"].shift()
    df["prev_jockey"] = prev["jockey_id"].shift()
    df["prev_sex"] = prev["sex"].shift()
    return df


# ── 統計 ──────────────────────────────────────────────


def _stat(v: pd.Series) -> tuple[float | None, int, tuple[float, float] | None, float]:
    """平均・標本数・95% 信頼区間・平均の分散。0/1 の列は Wilson の区間。"""
    v = v.dropna().astype(float)
    n = len(v)
    if n == 0:
        return None, 0, None, 0.0
    m = float(v.mean())
    binary = bool(((v == 0) | (v == 1)).all())
    if binary:
        z = 1.96
        d = 1 + z * z / n
        c = (m + z * z / (2 * n)) / d
        h = z * math.sqrt(m * (1 - m) / n + z * z / (4 * n * n)) / d
        ci = (max(0.0, c - h), min(1.0, c + h))
        var = m * (1 - m) / n
    else:
        sd = float(v.std(ddof=1)) if n > 1 else 0.0
        var = sd * sd / n
        ci = (m - 1.96 * math.sqrt(var), m + 1.96 * math.sqrt(var))
    return m, n, ci, var


def _p(va: float, vb: float, ma: float | None, mb: float | None) -> float | None:
    """2 系列の平均の差の p 値（両側、Welch の z）。"""
    if ma is None or mb is None or va + vb <= 0:
        return None
    z = abs(ma - mb) / math.sqrt(va + vb)
    return math.erfc(z / math.sqrt(2))


@dataclass
class Pair:
    """1 つの比較グラフ: 横軸の各点で、条件の出走（a）とそれ以外（b）を比べる。"""
    slug: str
    title: str
    xs: list[str]
    a_name: str
    b_name: str
    metric: str
    rows: Callable[[pd.DataFrame, str], tuple[pd.Series, pd.Series]]   # (データ, 横軸の値) -> (条件の値, それ以外の値)
    x_label: str
    y_label: str
    caption: str = ""
    replace_title: str | None = None     # 既存のグラフを置き換えるときの、そのグラフのタイトル（無ければ記事に足す）
    after_title: str | None = None       # 足すとき、このタイトルのグラフの直後に置く

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        a_vals, b_vals, a_n, b_n, a_ci, b_ci, ps = [], [], [], [], [], [], []
        for x in self.xs:
            sa, sb = self.rows(df, x)
            ma, na, ca, va = _stat(sa)
            mb, nb, cb, vb = _stat(sb)
            a_vals.append(_r(ma)); a_n.append(na); a_ci.append(_rci(ca))
            b_vals.append(_r(mb)); b_n.append(nb); b_ci.append(_rci(cb))
            p = _p(va, vb, ma, mb)
            ps.append(None if p is None else float(f"{p:.3g}"))
        return {
            "type": "chart", "title": self.title, "xLabel": self.x_label, "yLabel": self.y_label, "xs": self.xs,
            "series": [
                {"name": self.a_name, "values": a_vals, "n": a_n, "ci": a_ci},
                {"name": self.b_name, "values": b_vals, "n": b_n, "ci": b_ci},
            ],
            "compare": {"a": self.a_name, "b": self.b_name, "p": ps},
            **({"caption": self.caption} if self.caption else {}),
        }


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def _rci(c: tuple[float, float] | None) -> list[float] | None:
    return None if c is None else [round(c[0], 4), round(c[1], 4)]


# ── 記事ごとの定義 ─────────────────────────────────────


def pairs(df: pd.DataFrame) -> list[Pair]:
    out: list[Pair] = []
    has_prev = df["prev_surface"].notna()

    def by(col: str, metric: str, cond: pd.Series, base: pd.Series | None = None):
        base = has_prev if base is None else base
        def rows(d: pd.DataFrame, x: str):
            m = base & (d[col].astype(str) == x)
            return d.loc[m & cond, metric], d.loc[m & ~cond, metric]
        return rows

    # 初ダート: 前走芝 → 今走ダート vs それ以外のダート出走（前走もダート）
    dirt = (df["surface"] == "ダート") & has_prev
    first_dirt = df["prev_surface"] == "芝"
    out.append(Pair("first-dirt", "初ダート vs 通常 × 人気帯別 複勝率", POP_BANDS, "初ダート", "通常（前走もダート）", "top3",
                    by("pop_band", "top3", first_dirt, dirt), "人気帯", "複勝率",
                    "ダートの出走のうち、前走が芝の馬（初ダート）と前走もダートの馬の比較。",
                    replace_title="初ダート vs 通常 × 人気帯別 複勝率"))

    # 乗り替わり: 前走と騎手が違う vs 同じ
    change = df["jockey_id"].astype(str) != df["prev_jockey"].astype(str)
    out.append(Pair("jockey-change", "乗り替わりの有無 × 人気帯別 複勝率", POP_BANDS, "乗り替わり", "継続騎乗", "top3",
                    by("pop_band", "top3", change), "人気帯", "複勝率",
                    "前走と同じ馬で、騎手が替わった出走と同じ騎手の出走の比較。",
                    replace_title="乗り替わりの有無 × 人気帯別 複勝率"))

    # 夏の牝馬: 月ごとに 牝 vs 牡（セン馬は除く）。牝馬限定戦では牡馬と比べられないので、牝馬と牡馬が一緒に走った混合戦だけで比べる
    mf = df["sex"].isin(["牝", "牡"]) & df["mixed_sex"]
    months = [f"{m}月" for m in range(1, 13)]
    out.append(Pair("summer-mare", "月別 牡・牝馬 勝率の推移（混合戦）", months, "牝", "牡", "win",
                    lambda d, x: (d.loc[mf & (d["sex"] == "牝") & (d["month"] == int(x[:-1])), "win"],
                                  d.loc[mf & (d["sex"] == "牡") & (d["month"] == int(x[:-1])), "win"]),
                    "月", "勝率", "2020〜2025年 JRA 平地。牝馬と牡馬が一緒に走ったレース（混合戦）だけで、各月の勝率を比べる。＊は差が有意（p<0.05）な月。",
                    replace_title="月別 牡・牝馬 勝率の推移"))
    mare = mf & (df["sex"] == "牝")
    out.append(Pair("summer-mare", "月別 牝馬の勝率（実際とオッズの見込み・混合戦）", months, "実際の勝率", "オッズから見込まれる勝率", "win",
                    lambda d, x: (d.loc[mare & (d["month"] == int(x[:-1])), "win"], d.loc[mare & (d["month"] == int(x[:-1])), "implied"]),
                    "月", "勝率", "混合戦の牝馬。オッズから見込まれる勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。実際が見込みを上回る月は、牝馬が過小評価されている。",
                    after_title="月別 牡・牝馬 勝率の推移（混合戦）"))

    # 去勢明け: 年齢別に セン馬 vs 牡馬
    ages = [f"{a}歳" for a in range(2, 9)]
    out.append(Pair("gelding-return", "セン馬の年齢別勝率", ages, "セン馬", "牡馬（セン馬以外）", "win",
                    lambda d, x: (d.loc[(d["sex"] == "セ") & (d["age"] == int(x[:-1])), "win"],
                                  d.loc[(d["sex"] == "牡") & (d["age"] == int(x[:-1])), "win"]),
                    "年齢", "勝率", "2020〜2025年 JRA 平地。同じ年齢の牡馬と比べたセン馬の勝率。",
                    replace_title="セン馬の年齢別勝率"))
    # 去勢明け初戦（前走は牡、今走はセ）vs それ以外の出走
    gelded_now = (df["sex"] == "セ") & (df["prev_sex"] == "牡")
    out.append(Pair("gelding-return", "去勢明け初戦 vs それ以外 × 人気帯別 複勝率", POP_BANDS, "去勢明け初戦", "それ以外の出走", "top3",
                    by("pop_band", "top3", gelded_now), "人気帯", "複勝率",
                    "前走は牡、今走はセン馬として出た出走（去勢明け初戦）と、それ以外の出走の比較。",
                    after_title="セン馬の年齢別勝率"))

    # 騎手の勝率と回収率: 勝率上位 8 名（500 走以上）vs それ以外の騎手
    jk = df.groupby("jockey_id").agg(n=("win", "size"), w=("win", "mean"))
    top8 = set(jk[jk["n"] >= 500].sort_values("w", ascending=False).head(8).index)
    is_top = df["jockey_id"].isin(top8)
    out.append(Pair("jockey-roi", "勝率上位8名 vs それ以外の騎手 × 人気帯別 単勝回収率", POP_BANDS, "勝率上位8名", "それ以外の騎手", "ret",
                    by("pop_band", "ret", is_top, pd.Series(True, index=df.index)), "人気帯", "単勝回収率",
                    "単勝を 100 円ずつ買ったときの回収率。勝率上位 8 名（500 走以上）とそれ以外の騎手の比較。",
                    after_title="勝率上位8名 単勝回収率の比較（500走以上）"))

    # 距離と騎手: 距離帯ごとに 上位 10 名（その距離帯で 50 走以上・勝率順）vs それ以外の騎手
    def top10_rows(d: pd.DataFrame, x: str):
        m = d["dist_band"].astype(str) == x
        sub = d[m]
        g = sub.groupby("jockey_id").agg(n=("win", "size"), w=("win", "mean"))
        top = set(g[g["n"] >= 50].sort_values("w", ascending=False).head(10).index)
        t = sub["jockey_id"].isin(top)
        return sub.loc[t, "win"], sub.loc[~t, "win"]
    out.append(Pair("distance-jockey", "距離帯別 騎手勝率の格差（上位10名 vs それ以外の騎手）", DIST_BANDS, "上位10名", "それ以外の騎手", "win",
                    top10_rows, "距離帯", "勝率",
                    "上位10名は、その距離帯で50走以上の騎手から勝率順に選出。差が距離で広がるかを見る。",
                    replace_title="距離帯別 騎手勝率の格差（上位10名平均 vs 全体平均）"))

    # 荒れやすさ: 距離帯ごとに ローカル vs 主要4場 の 1番人気の複勝率
    fav = df["popularity"] == 1
    local = ~df["venue"].isin(MAJOR_VENUES)
    out.append(Pair("upset-conditions", "距離帯別 1番人気複勝率（ローカル vs 主要4場）", DIST_BANDS, "ローカル", "主要4場（東京・中山・京都・阪神）", "top3",
                    lambda d, x: (d.loc[fav & local & (d["dist_band"].astype(str) == x), "top3"],
                                  d.loc[fav & ~local & (d["dist_band"].astype(str) == x), "top3"]),
                    "距離帯", "1番人気の複勝率",
                    "ローカルは札幌・函館・福島・新潟・中京・小倉の 6 場。1番人気の複勝率が低いほど荒れやすい。＊は差が有意（p<0.05）な距離帯。",
                    after_title="クラス別 1番人気複勝率"))

    # 輸送（transport-vs-stay）は src/research/columns/transport_vs_stay.py が記事全体を書く

    # 種牡馬（sire-roi）は src/research/columns/sire_roi.py が記事全体を書く（回収率のグラフ。勝率のグラフは 2026-11-09 に外した）
    return out


# ── 記事への書き込み ───────────────────────────────────


# 記事の JSON は手で整えた書式なので、全体を書き直さず、対象のグラフの {…} の範囲だけを文字列で差し替える／直後に足す。


def _object_spans(text: str) -> list[tuple[int, int]]:
    """text の中の JSON オブジェクト {…} の範囲（開始, 終了の次）をすべて返す（文字列の中の括弧は数えない）。"""
    spans, stack, i, in_str = [], [], 0, False
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            stack.append(i)
        elif c == "}":
            spans.append((stack.pop(), i + 1))
        i += 1
    return spans


def _find_chart(text: str, title: str) -> tuple[int, int] | None:
    for s, e in _object_spans(text):
        try:
            obj = json.loads(text[s:e])
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "chart" and obj.get("title") == title:
            return s, e
    return None


def _inline(v: Any) -> str:
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {_inline(x)}" for k, x in v.items()) + " }"
    if isinstance(v, list):
        return "[" + ", ".join(_inline(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=False)


def _chart_text(chart: dict[str, Any], indent: str) -> str:
    """記事の既存のグラフと同じ書式（キーごとに 1 行、系列は 1 行に 1 つ）。"""
    inner = indent + "  "
    lines = []
    for k, v in chart.items():
        if k == "series":
            rows = ",\n".join(f"{inner}  {_inline(x)}" for x in v)
            lines.append(f'{inner}"series": [\n{rows}\n{inner}]')
        elif k == "variants":
            rows = ",\n".join(f"{inner}  {json.dumps(vk, ensure_ascii=False)}: {_inline(vv)}" for vk, vv in v.items())
            lines.append(f'{inner}"variants": {{\n{rows}\n{inner}}}')
        else:
            lines.append(f"{inner}{json.dumps(k, ensure_ascii=False)}: {_inline(v)}")
    return "{\n" + ",\n".join(lines) + "\n" + indent + "}"


def _indent_at(text: str, pos: int) -> str:
    line = text[text.rfind("\n", 0, pos) + 1:pos]
    return line[: len(line) - len(line.lstrip())]


@dataclass
class Placement:
    """比較でないグラフの置き場所（Pair と同じ slug / replace_title / after_title を持つ）。"""
    slug: str
    replace_title: str | None = None
    after_title: str | None = None
    remove_titles: tuple[str, ...] = ()     # まとめたことで要らなくなった、記事の中の別のグラフ


PACES = {"high": "ハイ", "mid": "ミドル", "slow": "スロー"}


PACE_POP = {"1番人気": (1, 1), "2〜3番人気": (2, 3), "4〜6番人気": (4, 6), "7番人気以下": (7, 99)}
# プルダウンで切り替える条件（先頭は「全体」）。キーは「距離|芝ダ」
PACE_DIST = {"全体": None, "1000〜1599m": (1000, 1599), "1600〜2099m": (1600, 2099), "2100m〜": (2100, 99999)}
PACE_SURF = {"全体": None, "芝": "芝", "ダート": "ダート"}


def _pace_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    """距離・芝ダの組み合わせごとの出走（ペース区分つき）。"""
    if not RACE_QUALITY_PATH.exists():
        return None
    rq = pd.read_parquet(RACE_QUALITY_PATH, columns=["race_id", "pace_cls"]).dropna().drop_duplicates("race_id")
    d = df[df["popularity"].notna()].merge(rq, on="race_id")
    out = {}
    for dk, dr in PACE_DIST.items():
        for sk, sv in PACE_SURF.items():
            m = pd.Series(True, index=d.index)
            if dr:
                m &= d["distance"].between(*dr)
            if sv:
                m &= d["surface"] == sv
            out[f"{dk}|{sk}"] = d[m]
    return out


def _pace_rate_series(d: pd.DataFrame) -> list[dict[str, Any]]:
    series = []
    for name, (lo, hi) in PACE_POP.items():
        vals, ns, cis = [], [], []
        for k in PACES:
            m, n, ci, _ = _stat(d.loc[(d["pace_cls"] == k) & d["popularity"].between(lo, hi), "top3"])
            vals.append(_r(m)); ns.append(n); cis.append(_rci(ci))
        row = {"name": name, "values": vals, "n": ns, "ci": cis}
        if name == "4〜6番人気":
            row = {"name": name, "values": vals, "baseline": True, "n": ns, "ci": cis}
        series.append(row)
    return series


def _pace_pop_series(d: pd.DataFrame) -> tuple[list[dict[str, Any]], str]:
    t = d[d["finish_position"].between(1, 3)]
    vals, ns, cis, var = [], [], [], {}
    for k in PACES:
        m, n, ci, v = _stat(t.loc[t["pace_cls"] == k, "popularity"])
        vals.append(_r(m)); ns.append(n); cis.append(_rci(ci)); var[k] = (m, v)
    p = _p(var["high"][1], var["slow"][1], var["high"][0], var["slow"][0])
    ptxt = "–" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")
    return [{"name": "3着以内の馬の平均人気", "values": vals, "n": ns, "ci": cis}], f"ハイとスローの差の p 値は {ptxt}。"


def pace_charts(df: pd.DataFrame) -> list[tuple[Placement, dict[str, Any]]]:
    """記事『ハイペースは荒れやすい』の 2 つのグラフ。距離と芝ダートをプルダウンで切り替えられるよう、組み合わせごとに集計する。"""
    frames = _pace_frames(df)
    if frames is None:
        return []
    filters = [{"label": "距離", "options": list(PACE_DIST)}, {"label": "芝・ダート", "options": list(PACE_SURF)}]
    base_key = "全体|全体"
    note = "2020〜2025年 JRA 平地。左軸は人気帯ごとの複勝率（4〜6番人気は点線）、右軸は 3 着以内に来た馬の平均人気（大きいほど人気薄が馬券に絡んでいる）。"
    pop_base, p_base = _pace_pop_series(frames[base_key])
    variants = {}
    for k, d in frames.items():
        if k != base_key:
            ser, ptxt = _pace_pop_series(d)
            variants[k] = {"series": _pace_rate_series(d), "y2Series": ser, "caption": note + "平均人気の" + ptxt}
    chart = {"type": "chart", "title": "ペース別 人気帯ごとの複勝率と馬券内の平均人気", "xLabel": "ペース", "yLabel": "複勝率",
             "xs": list(PACES.values()), "series": _pace_rate_series(frames[base_key]),
             "y2": {"label": "平均人気", "format": "num", "series": pop_base},
             "filters": filters, "variants": variants, "caption": note + "平均人気の" + p_base}
    return [(Placement("pace-advantage", replace_title="ペース別 人気帯ごとの複勝率", remove_titles=("ペース別 馬券内（3着以内）の平均人気",)), chart)]


def apply(pair: Pair | Placement, chart: dict[str, Any]) -> str:
    path = COLUMNS_DIR / f"{pair.slug}.json"
    text = path.read_text(encoding="utf-8")
    if hit := _find_chart(text, chart["title"]):          # 2 回目以降の実行・同じタイトルのグラフ: 置き換える
        s, e = hit
        text, how = text[:s] + _chart_text(chart, _indent_at(text, s)) + text[e:], "updated"
    elif pair.replace_title and (hit := _find_chart(text, pair.replace_title)):
        s, e = hit
        text, how = text[:s] + _chart_text(chart, _indent_at(text, s)) + text[e:], "replaced"
    elif pair.after_title and (hit := _find_chart(text, pair.after_title)):
        s, e = hit
        ind = _indent_at(text, s)
        text, how = text[:e] + ",\n" + ind + _chart_text(chart, ind) + text[e:], "added"
    else:
        raise SystemExit(f"{pair.slug}: 置き場所のグラフ『{pair.replace_title or pair.after_title}』が見つかりません")
    for title in getattr(pair, "remove_titles", ()):
        if hit := _find_chart(text, title):
            s, e = hit
            start = text.rfind(",", 0, s)                    # 直前の区切り（「,\n    」）ごと消す
            text = text[:start] + text[e:]
            how += f"+removed『{title}』"
    json.loads(text)                                     # 壊していないことを確かめてから書く
    path.write_text(text, encoding="utf-8")
    return how

# 記事『荒れるレースの共通点』。「荒れる」は 1 番人気の複勝率（レースごとに 1 頭）と、7 番人気以下が 3 着以内に来たレースの割合で測る。
# 頭数は実際に走った頭数（starters）。頭数が多いほど 1 番人気の複勝率が下がるのは、オッズ（市場の見込み）も同じ向きなので、見込みとの差も出す。
STARTER_BANDS = {"〜9頭": (1, 9), "10〜13頭": (10, 13), "14〜16頭": (14, 16), "17頭〜": (17, 99)}
CLASSES = {"新馬": ["新馬"], "未勝利": ["未勝利"], "1勝": ["1勝"], "2勝": ["2勝"], "3勝": ["3勝"], "OP\n(L含む)": ["OP", "L"], "重賞": ["G1", "G2", "G3"]}


def _one(v: pd.Series) -> tuple[float | None, int, list[float] | None]:
    m, n, ci, _ = _stat(v)
    return _r(m), n, _rci(ci)


def _series(name: str, parts: list[pd.Series]) -> dict[str, Any]:
    st = [_one(v) for v in parts]
    return {"name": name, "values": [x[0] for x in st], "n": [x[1] for x in st], "ci": [x[2] for x in st]}


def upset_charts(df: pd.DataFrame) -> list[tuple[Placement, dict[str, Any]]]:
    d = df.copy()
    q = 1.0 / d["odds"]
    d["implied"] = q / q.groupby(d["race_id"]).transform("sum")          # オッズから見込まれる勝率（控除率を除いて合計 1 に直す）
    fav = d[d["popularity"] == 1]
    xs = list(STARTER_BANDS)
    by_st = [fav[fav["starters"].between(*STARTER_BANDS[x])] for x in xs]
    period = f"{min(YEARS)}〜{max(YEARS)}年 JRA 平地"
    c1 = {"type": "chart", "title": "出走頭数別 1番人気複勝率", "xLabel": "出走頭数（取消・除外を除く）", "yLabel": "1番人気の複勝率", "xs": xs,
          "series": [_series("1番人気の複勝率", [g["top3"] for g in by_st])],
          "caption": f"{period}。頭数が多いほど、1 番人気でも 3 着以内に入る枠は相対的に狭くなる。縦線は 95% 信頼区間。"}
    win = _series("実際の勝率", [g["win"] for g in by_st])
    imp = _series("オッズから見込まれる勝率", [g["implied"] for g in by_st])
    ps = []
    for g in by_st:
        ma, _, _, va = _stat(g["win"])
        mb, _, _, vb = _stat(g["implied"])
        p = _p(va, vb, ma, mb)
        ps.append(None if p is None else float(f"{p:.3g}"))
    c2 = {"type": "chart", "title": "出走頭数別 1番人気の勝率（実際とオッズの見込み）", "xLabel": "出走頭数（取消・除外を除く）", "yLabel": "勝率", "xs": xs,
          "series": [win, imp], "compare": {"a": "実際の勝率", "b": "オッズから見込まれる勝率", "p": ps},
          "caption": f"{period}。オッズから見込まれる勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。頭数が増えると見込みも同じだけ下がり、実際の勝率はどの頭数でも見込みとほぼ同じ（差は有意でない）。"}
    xs3 = list(CLASSES)
    c3 = {"type": "chart", "title": "クラス別 1番人気複勝率", "xLabel": "クラス", "yLabel": "1番人気の複勝率", "xs": xs3,
          "series": [_series("1番人気の複勝率", [fav.loc[fav["grade"].isin(CLASSES[x]), "top3"] for x in xs3])],
          "caption": f"{period}。縦線は 95% 信頼区間。"}
    # レース単位: 7 番人気以下が 1 頭でも 3 着以内に来たか（7 頭立て以下は 7 番人気以下がいない・少ないので、10 頭立て以上に限る）
    races = d.groupby("race_id").agg(surface=("surface", "first"), dist_band=("dist_band", "first"), starters=("starters", "first"))
    hole = d[(d["popularity"] >= 7) & (d["top3"] == 1)]["race_id"].unique()
    races["hole"] = races.index.isin(hole).astype(float)
    races = races[races["starters"] >= 10]
    c4 = {"type": "chart", "title": "距離帯別 7番人気以下が3着以内に来たレースの割合", "xLabel": "距離帯", "yLabel": "レースの割合", "xs": DIST_BANDS,
          "series": [_series(sf, [races.loc[(races["surface"] == sf) & (races["dist_band"].astype(str) == x), "hole"] for x in DIST_BANDS]) for sf in ("芝", "ダート")],
          "caption": f"{period}。10 頭立て以上のレースで、7 番人気以下が 1 頭でも 3 着以内に来た割合。短距離ほど頭数が多いことに注意。頭数・クラス・ローカルをそろえた差は本文の回帰を参照。"}
    return [(Placement("upset-conditions", replace_title="出走頭数別 1番人気複勝率"), c1),
            (Placement("upset-conditions", after_title="出走頭数別 1番人気複勝率"), c2),
            (Placement("upset-conditions", replace_title="クラス別 1番人気複勝率"), c3),
            (Placement("upset-conditions", after_title="距離帯別 1番人気複勝率（ローカル vs 主要4場）"), c4)]


def upset_regression(df: pd.DataFrame) -> None:
    """記事の本文の「条件をそろえると」の数字（1 番人気の複勝率の線形回帰、HC1）。"""
    import statsmodels.formula.api as smf

    f = df[df["popularity"] == 1].copy()
    f["cls"] = f["grade"].replace({"G1": "重賞", "G2": "重賞", "G3": "重賞", "L": "OP", "OP": "OP"})
    f["local"] = (~f["venue"].isin(MAJOR_VENUES)).astype(float)
    f["db"] = f["dist_band"].astype(str).str.split("\n").str[0]
    f["lq"] = np.log(f["odds"])
    for form in ["top3 ~ starters + C(cls, Treatment('1勝')) + C(db, Treatment('マイル')) + local + C(surface)",
                 "top3 ~ starters + C(cls, Treatment('1勝')) + C(db, Treatment('マイル')) + local + C(surface) + lq"]:
        m = smf.ols(form, f).fit(cov_type="HC1")
        print("  " + form)
        for k in m.params.index.drop("Intercept"):
            print(f"    {k:<40} {m.params[k] * 100:+.2f}pt  p={m.pvalues[k]:.3g}")
    # 7 番人気以下が 3 着以内に来たか（10 頭立て以上のレース単位）。芝・ダートそれぞれで、マイルと比べた距離帯の差
    r = df.groupby("race_id").agg(surface=("surface", "first"), db=("dist_band", "first"), starters=("starters", "first"),
                                  grade=("grade", "first"), venue=("venue", "first"))
    r["hole"] = r.index.isin(df[(df["popularity"] >= 7) & (df["top3"] == 1)]["race_id"].unique()).astype(float)
    r = r[r["starters"] >= 10]
    r["db"] = r["db"].astype(str).str.split("\n").str[0]
    r["cls"] = r["grade"].replace({"G1": "重賞", "G2": "重賞", "G3": "重賞", "L": "OP"})
    r["local"] = (~r["venue"].isin(MAJOR_VENUES)).astype(float)
    for sf in ("芝", "ダート"):
        x = r[r["surface"] == sf]
        m = smf.ols("hole ~ starters + C(db, Treatment('マイル')) + C(cls) + local", x).fit(cov_type="HC1")
        print(f"  7番人気以下が3着以内（{sf}・{len(x)}レース）")
        for k in [k for k in m.params.index if "db" in k]:
            print(f"    {k:<40} {m.params[k] * 100:+.2f}pt  p={m.pvalues[k]:.3g}")


def mare_regression(df: pd.DataFrame) -> None:
    """記事『夏は牝馬』の本文の数字。混合戦の牡・牝で、勝ちを オッズ（log とその 2 乗）・年齢区分 でそろえ、牝 × 夏（6〜8月）の上積みを見る。"""
    import statsmodels.formula.api as smf

    x = df[df["sex"].isin(["牝", "牡"]) & df["mixed_sex"]].copy()
    x["mare"] = (x["sex"] == "牝").astype(float)
    x["summer"] = x["month"].isin([6, 7, 8]).astype(float)
    x["lq"] = np.log(x["odds"])
    x["ag"] = np.where(x["age"] == 2, "2歳", np.where(x["age"] == 3, "3歳", "4歳以上"))
    x["year"] = pd.to_datetime(x["date"]).dt.year
    print("  夏は牝馬（混合戦）: 季節 × 性別")
    x["season"] = x["month"].map(lambda m: "春3-5月" if m in (3, 4, 5) else "夏6-8月" if m in (6, 7, 8) else "秋9-11月" if m in (9, 10, 11) else "冬12-2月")
    g = x.groupby(["season", "sex"]).agg(n=("win", "size"), win=("win", "mean"), implied=("implied", "mean"), roi=("ret", "mean"))
    print(g.round(4).to_string())
    for name, m in [("全体", x["year"] > 0), ("2020〜2022年", x["year"] <= 2022), ("2023〜2025年", x["year"] >= 2023), ("3歳以上", x["age"] >= 3), ("4歳以上", x["age"] >= 4)]:
        z = x[m]
        r = smf.ols("win ~ mare*summer + lq + I(lq**2) + C(ag)", z).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(z["race_id"])[0]})
        print(f"    {name:<12} 牝×夏 {r.params['mare:summer'] * 100:+.2f}pt (p={r.pvalues['mare:summer']:.3g})  夏以外の牝 {r.params['mare'] * 100:+.2f}pt (p={r.pvalues['mare']:.3g})")
    lim = df[df["sex"].eq("牝") & ~df["mixed_sex"]]
    print("  牝馬限定戦（対照）: 季節ごとの 実際 − 見込み")
    print(lim.assign(s=lim["month"].isin([6, 7, 8])).groupby("s").agg(n=("win", "size"), win=("win", "mean"), implied=("implied", "mean"), roi=("ret", "mean")).round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    df = load()
    for place, chart in pace_charts(df):
        how = "dry-run" if args.dry_run else apply(place, chart)
        print(f"[{how}] {place.slug}: {chart['title']}")
        for key, v in [("全体|全体", {"series": chart["series"], "y2Series": chart["y2"]["series"]})] + list(chart["variants"].items()):
            print(f"    {key:<18} " + " / ".join(f"{s['name']}=" + ",".join(str(x) for x in s["values"]) for s in v["series"] + v["y2Series"]))
    for place, chart in upset_charts(df):
        how = "dry-run" if args.dry_run else apply(place, chart)
        print(f"[{how}] {place.slug}: {chart['title']}")
        for s_ in chart["series"]:
            print(f"    {s_['name']}: " + ", ".join(f"{v}(n={n})" for v, n in zip(s_["values"], s_["n"])))
    upset_regression(df)
    mare_regression(df)
    for pair in pairs(df):
        chart = pair.compute(df)
        how = "dry-run" if args.dry_run else apply(pair, chart)
        print(f"[{how}] {pair.slug}: {pair.title}")
        a, b = chart["series"]
        for k, x in enumerate(pair.xs):
            p = chart["compare"]["p"][k]
            print(f"    {x.replace(chr(10), ' '):<22} {a['values'][k]!s:>8} (n={a['n'][k]:>6})  vs {b['values'][k]!s:>8} (n={b['n'][k]:>6})  p={p}")


if __name__ == "__main__":
    main()
