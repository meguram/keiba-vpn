"""騎手コラム（jockey-long-distance・jockey-class-venue-season・jockey-change）で共通に使う集計。

データ: compare_series.load()（2020〜2025 年の JRA 平地、取消・除外を除く）。
  障害レースは surface が芝・ダートで入っているものも load() が除く（compare_series.jump_mask）。

人気から見込まれる率（exp_win・exp_top3）:
  単勝オッズから求めた勝率の見込み implied（1/オッズ をレース内で合計 1 に直したもの）を 50 分位に分け、
  頭数の帯（〜8・9〜12・13〜16・17 頭〜）ごとに、実際の勝率・複勝率の平均をその出走の見込みとする（較正）。
  騎乗馬の強さ（市場の評価）の差を除くための物差し。オッズには騎手の評価も入っているので、
  「実際 − 見込み」は『市場が織り込んだ以上の上積み』で、騎手の腕そのものより小さく出る。

「得意・苦手」の 2 軸（two_axis）:
  (a) 騎手間: その条件で、騎手の率が全騎手の平均（同じ条件）より高いか低いか。生の率で比べる（騎乗馬の強さを含む）。
  (b) 騎手内: その条件の「実際 − 見込み」が、同じ騎手のほかの条件の「実際 − 見込み」より高いか低いか。
      見込みとの差で比べるので、条件ごとの騎乗馬の強さの違いは除かれる。
  検定は条件・騎手の組の数だけ繰り返すので、Benjamini–Hochberg で q 値（偽発見率）に直し、q<0.10 を「差あり」とする。
  両方とも下なら「苦手」、両方とも上なら「得意」。片方だけのときは表現を分ける（LABELS）。

騎手のランク（ranks）:
  騎手ごとの複勝率を、経験ベイズ（正規-正規、モーメント法）で全騎手の平均に向けて縮め、分位で S〜D に分ける。
  騎乗が少ない騎手ほど強く縮むので、少ない騎乗での偶然の好成績で上位に入らない。
  「実際 − 人気の見込み」で並べる案も試したが、市場が騎手をよく織り込んでいて騎手間の差が小さく（標準偏差 0.7pt 弱）、
  人気を集める上位騎手がむしろ下位に来るため、ランクは成績（複勝率）で決め、人気との差は乗り替わりの結果の側で見る。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import FLAT_DIR, YEARS, _r, _rci, _stat, jump_mask, load
from src.utils.person_reading import kana_sort_key, lookup

CLASSES = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP・L", "重賞"]
CLASS_OF = {"新馬": "新馬", "未勝利": "未勝利", "1勝": "1勝", "2勝": "2勝", "3勝": "3勝", "OP": "OP・L", "L": "OP・L",
            "G3": "重賞", "G2": "重賞", "G1": "重賞"}
SEASONS = ["春\n3〜5月", "夏\n6〜8月", "秋\n9〜11月", "冬\n12〜2月"]
SEASON_OF = {3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 3, 1: 3, 2: 3}
VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
# 距離（m）を細かい帯に（グラフ用）。1150・1300 などの半端な距離は近い帯に寄せる
DIST9 = ["1000m", "1200m", "1400m", "1600m", "1800m", "2000m", "2200m", "2400m", "2500m〜"]
DIST9_EDGES = [0, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2450, 9999]
DIST4 = ["短距離\n〜1400m", "マイル\n1401〜1800m", "中距離\n1801〜2200m", "長距離\n2201m〜"]
DIST4_EDGES = [0, 1400, 1800, 2200, 9999]

Q_SIG = 0.10          # 2 軸の判定で「差あり」とする q 値（BH）
RANKS = ["S", "A", "B", "C", "D"]
RANK_CUTS = [0.05, 0.25, 0.75, 0.95]     # 上から 5%・25%・75%・95% の位置で切る（B が真ん中の半分 = 平均くらい）


def jump_race_ids() -> set[str]:
    """障害レースの race_id（load() は既に除いている。前走をたどるときなどの確認用）。"""
    cols = ["race_id", "race_name", "race_class", "surface", "distance"]
    r = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=cols) for y in YEARS], ignore_index=True)
    return set(r.loc[jump_mask(r), "race_id"].astype(str))


def calibrate(df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
    """人気（単勝オッズ）と頭数から見込まれる勝率・複勝率（exp_win・exp_top3）と、実際との差（res_win・res_top3）を付ける。"""
    d = df.copy()
    d["_q"] = pd.qcut(d["implied"].rank(method="first"), bins, labels=False)
    d["_fs"] = pd.cut(d["starters"], [0, 8, 12, 16, 99], labels=False)
    g = d.groupby(["_q", "_fs"])
    d["exp_win"] = g["win"].transform("mean")
    d["exp_top3"] = g["top3"].transform("mean")
    d["res_win"] = d["win"] - d["exp_win"]
    d["res_top3"] = d["top3"] - d["exp_top3"]
    return d.drop(columns=["_q", "_fs"])


def frame() -> pd.DataFrame:
    """騎手コラム用の出走データ（障害を除き、クラス・季節・距離帯・人気からの見込みを付ける）。"""
    df = load()
    df = df[~df["race_id"].astype(str).isin(jump_race_ids())].copy()
    df["jockey_id"] = df["jockey_id"].astype(str)
    df["cls"] = df["grade"].map(CLASS_OF)
    df["season"] = df["month"].map(lambda m: SEASONS[SEASON_OF[int(m)]])
    df["dist9"] = pd.cut(df["distance"], DIST9_EDGES, labels=DIST9).astype(str)
    df["dist4"] = pd.cut(df["distance"], DIST4_EDGES, labels=DIST4).astype(str)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    # 前走の騎手は、ここで絞ったあとの同じ馬の前走で付け直す
    df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    df["prev_jockey"] = df.groupby("horse_id")["jockey_id"].shift()
    return calibrate(df)


# ── 騎手の名前 ─────────────────────────────────────


def jockey_names(df: pd.DataFrame) -> dict[str, str]:
    """jockey_id → 表示名（マスタの正式名。無ければ出走表の略称）。"""
    short = df.groupby("jockey_id")["jockey_name"].agg(lambda s: s.mode().iat[0])
    out = {}
    for jid, nm in short.items():
        p = lookup(jid, "jockey") or lookup(nm, "jockey")
        out[jid] = p["name"] if p else str(nm)
    return out


def jockey_reading_of(jid: str, name: str) -> str:
    p = lookup(jid, "jockey") or lookup(name, "jockey")
    return p["reading"] if p else name


def kana_order(ids: list[str], names: dict[str, str]) -> list[str]:
    """jockey_id を読みの 50 音順に。"""
    return sorted(ids, key=lambda j: (kana_sort_key(jockey_reading_of(j, names[j])), names[j]))


# ── 統計 ──────────────────────────────────────────


def norm_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def bh(ps: list[float]) -> list[float]:
    """Benjamini–Hochberg の q 値。nan はそのまま。"""
    arr = np.asarray(ps, dtype=float)
    ok = ~np.isnan(arr)
    q = np.full(len(arr), np.nan)
    p = arr[ok]
    m = len(p)
    if not m:
        return q.tolist()
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q.tolist()


def cell(rows: pd.DataFrame, metric: str) -> dict[str, Any]:
    """出走の集まりの n・率（95% CI）・見込み・実際 − 見込み（95% CI）。"""
    m, n, ci, _ = _stat(rows[metric])
    e = float(rows[f"exp_{metric}"].mean()) if n else None
    # 実際 − 見込み の分散は、見込みどおりに当たる場合の二項分布の分散 Σe(1−e)/n² を使う
    # （出走ごとの残差の標本分散だと、0 勝の騎手で分散がほぼ 0 になり、少ない騎乗で偶然の 0 勝を「有意」にしてしまう）
    ev = rows[f"exp_{metric}"].clip(0, 1)
    dm = float(rows[f"res_{metric}"].mean()) if n else None
    dvar = float((ev * (1 - ev)).sum()) / (n * n) if n else 0.0
    dci = (dm - 1.96 * math.sqrt(dvar), dm + 1.96 * math.sqrt(dvar)) if n else None
    return {"n": n, "rate": _r(m), "ci": _rci(ci), "exp": _r(e), "diff": _r(dm), "diff_ci": _rci(dci), "diff_var": dvar,
            "rate_var": (m * (1 - m) / n) if n else 0.0}


LABELS = {
    ("down", "down"): "苦手",
    ("up", "up"): "得意",
    ("up", "down"): "平均より上だが、ほかの条件ほどではない",
    ("flat", "down"): "ほかの条件より落ちる（全騎手の平均並み）",
    ("down", "up"): "平均より下だが、この騎手の中では良い",
    ("flat", "up"): "この騎手の中では良い（全騎手の平均並み）",
    ("up", "flat"): "全体に平均より上（この条件が特に良いわけではない）",
    ("down", "flat"): "全体に平均より下（この条件が特に悪いわけではない）",
    ("flat", "flat"): "はっきりした差なし",
}
SHORT = {"苦手": "苦手", "得意": "得意"}


def short_label(a: str, b: str) -> str:
    """表に出す短い判定。"""
    if (a, b) == ("down", "down"):
        return "苦手"
    if (a, b) == ("up", "up"):
        return "得意"
    if b == "down":
        return "相対的に下"
    if b == "up":
        return "相対的に上"
    return "–"


def level_adjusted(d: pd.DataFrame, col: str, metric: str) -> pd.DataFrame:
    """人気からの見込み（較正）は条件を区別していないので、条件ごとの全騎手の 実際 − 見込み の平均を見込みに足し、
    全騎手で平均すると各条件で 実際 − 見込み = 0 になるようにする（条件そのものの有利・不利を騎手の差に混ぜない）。"""
    shift = d.groupby(col)[f"res_{metric}"].transform("mean")
    return d.assign(**{f"exp_{metric}": d[f"exp_{metric}"] + shift, f"res_{metric}": d[f"res_{metric}"] - shift})


def two_axis(d: pd.DataFrame, jids: list[str], col: str, levels: list[str], metric: str, min_n: int = 50) -> pd.DataFrame:
    """騎手 × 条件ごとの 2 軸の判定。d は全騎手の出走（条件の平均は全騎手で求める）。

    返り値の列: jockey_id, level, n, rate, ci, exp, diff, diff_ci, avg（全騎手の率）, pa（(a) の p）, pb（(b) の p）,
              qa・qb（BH）, a・b（up/down/flat）, label, short。n<min_n の組は判定しない（label は「標本不足」）。
    """
    d = level_adjusted(d, col, metric)
    avg = d.groupby(col)[metric].mean()
    rows = []
    for j in jids:
        dj = d[d["jockey_id"] == j]
        for lv in levels:
            inn = dj[dj[col] == lv]
            c = cell(inn, metric)
            o = cell(dj[dj[col] != lv], metric)
            pa = pb = float("nan")
            if c["n"] >= min_n and c["rate_var"] > 0:
                pa = norm_p((c["rate"] - avg.get(lv, np.nan)) / math.sqrt(c["rate_var"]))
            if c["n"] >= min_n and o["n"] >= min_n and c["diff_var"] + o["diff_var"] > 0:
                pb = norm_p((c["diff"] - o["diff"]) / math.sqrt(c["diff_var"] + o["diff_var"]))
            rows.append({"jockey_id": j, "level": lv, **c, "avg": float(avg.get(lv, np.nan)), "other_diff": o["diff"],
                         "pa": pa, "pb": pb})
    t = pd.DataFrame(rows)
    t["qa"], t["qb"] = bh(t["pa"].tolist()), bh(t["pb"].tolist())

    def side(q: float, delta: float) -> str:
        if not (q < Q_SIG):
            return "flat"
        return "up" if delta > 0 else "down"

    t["a"] = [side(q, r - a) for q, r, a in zip(t["qa"], t["rate"].fillna(0), t["avg"])]
    t["b"] = [side(q, dd - od) for q, dd, od in zip(t["qb"], t["diff"].fillna(0), t["other_diff"].fillna(0))]
    small = t["n"] < min_n
    t["label"] = [("標本不足" if s else LABELS[(a, b)]) for s, a, b in zip(small, t["a"], t["b"])]
    t["short"] = [("標本不足" if s else short_label(a, b)) for s, a, b in zip(small, t["a"], t["b"])]
    return t


def eb_effect(d: pd.DataFrame, col: str = "top3", min_n: int = 1) -> pd.DataFrame:
    """騎手ごとの col の平均（raw）と、経験ベイズで全騎手の平均に向けて縮めた値（eb）。列: n, raw, se, eb。
    col は top3（複勝率そのもの）か res_top3（実際 − 人気の見込み）など。"""
    g = d.groupby("jockey_id")[col]
    t = pd.DataFrame({"n": g.size(), "raw": g.mean(), "sd": g.std(ddof=1)})
    t = t[t["n"] >= min_n].copy()
    t["se2"] = (t["sd"].fillna(t["sd"].median()) ** 2) / t["n"]
    w = t["n"] / t["n"].sum()
    mu = float((w * t["raw"]).sum())
    tau2 = max(0.0, float((w * (t["raw"] - mu) ** 2).sum()) - float((w * t["se2"]).sum()))
    t["eb"] = mu + (t["raw"] - mu) * (tau2 / (tau2 + t["se2"])) if tau2 > 0 else mu
    t["se"] = np.sqrt(t["se2"])
    t.attrs["tau"] = math.sqrt(tau2)
    t.attrs["mu"] = mu
    return t.drop(columns=["sd", "se2"])


def rank_of(eb: pd.Series) -> pd.Series:
    """縮めた効果の大きい順に、上位 5% = S、〜25% = A、〜75% = B、〜95% = C、残り = D。"""
    pos = eb.rank(ascending=False, method="first") / len(eb)
    return pd.cut(pos, [0, *RANK_CUTS, 1.0], labels=RANKS).astype(str)


def ranks(d: pd.DataFrame, min_n: int, col: str = "top3") -> pd.DataFrame:
    """騎乗 min_n 以上の騎手のランク（既定は複勝率を縮めた値で）。列: n, raw, se, eb, rank。"""
    t = eb_effect(d, col)                    # 縮める強さ（tau）は全騎手から求める
    t = t[t["n"] >= min_n].copy()
    t["rank"] = rank_of(t["eb"])
    return t


def chart_values(cells: list[dict[str, Any]], key: str = "rate") -> dict[str, Any]:
    """cell() の並びを ChartSeries の values・n・ci に。key は rate か diff。"""
    ci_key = "ci" if key == "rate" else "diff_ci"
    return {"values": [c[key] for c in cells], "n": [c["n"] for c in cells], "ci": [c[ci_key] for c in cells]}


def pct(v: float | None, d: int = 1) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.{d}f}%"


def pt(v: float | None, d: int = 1) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:+.{d}f}pt"


def pt_ci(ci: list[float] | None, d: int = 1) -> str:
    return "–" if not ci else f"{ci[0] * 100:+.{d}f}〜{ci[1] * 100:+.{d}f}"


# ── 集計に使っていない年で確かめる ─────────────────────


HOLDOUT_YEAR = 2026


def holdout_frame(year: int = HOLDOUT_YEAR) -> pd.DataFrame:
    """集計期間（YEARS）の後の 1 年の出走（frame と同じ列）。人気からの見込みは、その年だけで較正し直す。"""
    import src.research.columns.compare_series as cs

    if not (FLAT_DIR / str(year) / "race_result_flat.parquet").exists():
        return pd.DataFrame()
    saved = cs.YEARS
    try:
        cs.YEARS = range(year, year + 1)
        globals()["YEARS"] = cs.YEARS
        return frame()
    finally:
        cs.YEARS = saved
        globals()["YEARS"] = saved


def holdout_check(h: pd.DataFrame, jid: str, col: str, level: str, metric: str) -> dict[str, Any]:
    """集計に使っていない年で、(b) 騎手内の差（その条件の 実際 − 見込み − ほかの条件の 実際 − 見込み）。条件ごとの全騎手の平均はずらして除く。"""
    if h.empty:
        return {"n": 0}
    r = "res_" + metric
    hh = h.copy()
    hh[r] = hh[r] - hh.groupby(col)[r].transform("mean")
    x = hh[hh["jockey_id"] == jid]
    a, o = x.loc[x[col] == level, r], x.loc[x[col] != level, r]
    if len(a) < 2 or len(o) < 2:
        return {"n": int(len(a))}
    diff = float(a.mean() - o.mean())
    se = math.sqrt(a.var(ddof=1) / len(a) + o.var(ddof=1) / len(o))
    return {"n": int(len(a)), "diff": diff, "ci": [diff - 1.96 * se, diff + 1.96 * se], "p": norm_p(diff / se) if se > 0 else float("nan")}


def holdout_text(rows: list[tuple[str, str, float, dict[str, Any]]], year: int = HOLDOUT_YEAR) -> str:
    """rows: (騎手名, 条件, 集計期間の差, holdout_check の結果)。本文用の 1 段落。"""
    usable = [r for r in rows if r[3].get("n", 0) >= 10 and "diff" in r[3]]
    if not usable:
        return f"{year} 年のデータでは、騎乗数が足りず確かめられませんでした。"
    same = sum(1 for _, _, d0, h in usable if (d0 > 0) == (h["diff"] > 0))
    sig = [r for r in usable if r[3]["p"] < 0.05 and (r[2] > 0) == (r[3]["diff"] > 0)]
    parts = "、".join(f"{nm} {lv}（{pt(d0)} → {pt(h['diff'])}・{h['n']}騎乗）" for nm, lv, d0, h in usable)
    return (f"集計に使っていない {year} 年（1〜10 月）で同じ差を見ると、{len(usable)} 組のうち同じ向きは {same} 組でした：{parts}。"
            + (f"同じ向きで p<0.05 なのは {'・'.join(f'{r[0]} {r[1]}' for r in sig)} です。" if sig else "同じ向きで p<0.05 になった組はありません。")
            + "騎乗数が少なく区間が広いので、向きがそろうかどうかは偶然でも半々で起こります。今のところ、これらの「得意・苦手」が続くとは確かめられていません。")
