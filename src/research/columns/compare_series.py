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


def load() -> pd.DataFrame:
    cols = ["race_id", "date", "venue", "surface", "distance", "field_size", "finish_position", "horse_id",
            "sex_age", "jockey_id", "jockey_name", "odds", "popularity"]
    df = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=cols) for y in YEARS], ignore_index=True)
    df = df[df["surface"].isin(["芝", "ダート"])].copy()
    df["horse_id"] = df["horse_id"].astype(str)
    df["win"] = (df["finish_position"] == 1).astype(float)
    df["top3"] = df["finish_position"].between(1, 3).astype(float)
    df["ret"] = np.where(df["finish_position"] == 1, df["odds"].fillna(0), 0.0)      # 単勝 100 円あたりの払戻 / 100
    df["sex"] = df["sex_age"].str[0]
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

    # 夏の牝馬: 月ごとに 牝 vs 牡（セン馬は除く）
    mf = df["sex"].isin(["牝", "牡"])
    out.append(Pair("summer-mare", "月別 牡・牝馬 勝率の推移", [f"{m}月" for m in range(1, 13)], "牝", "牡", "win",
                    lambda d, x: (d.loc[mf & (d["sex"] == "牝") & (d["month"] == int(x[:-1])), "win"],
                                  d.loc[mf & (d["sex"] == "牡") & (d["month"] == int(x[:-1])), "win"]),
                    "月", "勝率", "2020〜2025年 JRA 平地。各月の牝馬と牡馬の勝率。",
                    replace_title="月別 牡・牝馬 勝率の推移"))

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
                    "1番人気の複勝率が低いほど荒れやすい。短距離・ローカルで下がるかを見る。",
                    after_title="クラス別 1番人気複勝率"))

    # 輸送: 前走と地域が違う（遠征）vs 同じ地域
    away = df["venue"].map(REGION) != df["prev_venue"].map(REGION)
    out.append(Pair("transport-vs-stay", "地域またぎ（遠征）の有無 × 人気帯別 複勝率", POP_BANDS, "地域またぎ（遠征）", "同地域継続", "top3",
                    by("pop_band", "top3", away), "人気帯", "複勝率",
                    "前走と今走の競馬場の地域（関東/関西/北海道/中部/九州）が違う出走と、同じ出走の比較。",
                    after_title="地域またぎ（遠征）の有無 × 複勝率・勝率"))

    # 種牡馬: 主要種牡馬の産駒 vs それ以外の種牡馬の産駒（勝率）
    sire_rows = _sire_pair(df)
    if sire_rows is not None:
        out.append(sire_rows)
    return out


def _sire_pair(df: pd.DataFrame) -> Pair | None:
    if not SIRE_MAP_PATH.exists():
        return None
    m = pd.read_parquet(SIRE_MAP_PATH, columns=["horse_id", "sire_id", "sire_name"]).dropna(subset=["sire_id"]).drop_duplicates("horse_id")
    m["horse_id"] = m["horse_id"].astype(str)
    d = df.merge(m, on="horse_id", how="inner")
    # 記事に載せている種牡馬（横軸の表記 -> 種牡馬名の先頭）。記事の本文と同じ顔ぶれで比べる
    ids: dict[str, str] = {}
    for label, prefix in SIRES.items():
        hit = d.loc[d["sire_name"].astype(str).str.startswith(prefix), "sire_id"]
        if hit.empty:
            raise SystemExit(f"sire-roi: 種牡馬『{prefix}』が血統表にありません")
        ids[label] = hit.mode().iloc[0]

    def rows(_: pd.DataFrame, x: str):
        k = d["sire_id"] == ids[x]
        return d.loc[k, "win"], d.loc[~k, "win"]
    return Pair("sire-roi", "主要種牡馬の産駒 vs それ以外の産駒 勝率", list(SIRES), "その種牡馬の産駒", "それ以外の種牡馬の産駒", "win",
                rows, "種牡馬", "勝率", "2020〜2025年 JRA 平地。各種牡馬の産駒と、それ以外の全産駒の勝率の比較。",
                replace_title="主要種牡馬 産駒勝率（産駒200走以上、2020〜2025年）")


SIRES = {"Into\nMischief": "Into Mischief", "ナダル": "ナダル", "Am.\nPharoah": "American Pharoah", "Kgm\n(英)": "Kingman",
         "アドマイヤ\nマーズ": "アドマイヤマーズ", "サートゥル": "サートゥルナーリア", "キタサン": "キタサンブラック", "キズナ": "キズナ",
         "ディープ": "ディープインパクト", "リアルS": "リアルスティール"}


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
