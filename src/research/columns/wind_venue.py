"""めぐコラム『競馬場ごとの風の影響』を実データから集計し、frontend/content/columns/wind-venue.json を書く。

問い: 風（特に最後の直線の向かい風・追い風）は、時計・有利な脚質・荒れやすさ・次走の評価に影響するか。場ごとに違うか。

データ:
  レース: data/page_reference/tables/<年>/race_result_flat.parquet（JRA 平地。脚質の履歴のため 2019 年から読む）
  風:     data/page_reference/tables/jma_weather_hourly_flat.parquet（python -m src.scraper.jma_weather export。気象庁の時別値、
          各場の最寄り観測所。その時刻の直前 10 分の平均風）
  直線の向き: data/meta/course_master/geo/routes.json（OpenStreetMap のコース形状）。ゴール手前 250m の進行方位の、場ごとの全コースの中央値
定義:
  発走の直前の正時の観測を使う（発走後の観測は混ぜない）。向かい風成分 = 風速 × cos(風向 − 直線の進行方位)（正 = 直線で向かい風）。
  当日の脚質 = 初角の位置（(順位−1)/(頭数−1)）。事前の脚質 = 前走まで直近 3 走の初角の位置の平均（予想の時点で分かるもの）。
  出走していない馬（取消・除外。オッズも人気もない）は除く。中止・失格は走ったので「3着以内でない」として数える。

使い方:
    python -m src.scraper.jma_weather export      # 風の表を作る（初回・更新時）
    python -m src.research.columns.wind_venue      # 集計して記事を書く
    python -m src.research.columns.wind_venue --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TABLES = REPO / "data/page_reference/tables"
WIND = TABLES / "jma_weather_hourly_flat.parquet"
ROUTES = REPO / "data/meta/course_master/geo/routes.json"
OUT = REPO / "frontend/content/columns/wind-venue.json"
FIRST_YEAR, LAST_YEAR = 2020, 2026
HOME_M = 250.0       # 直線の向きを測る区間（ゴール手前）。最も短い函館の直線（約 262m）にも収まる
VENUE_ORDER = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
RIGHT_HANDED = {"札幌", "函館", "福島", "中山", "京都", "阪神", "小倉"}
COMPASS16 = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東", "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
STYLES = ["逃げ", "先行", "差し", "追込"]
PRE_STYLES = ["前寄り", "好位", "中団", "後方"]
SPEED_BINS = ([-1, 2, 4, 6, 99], ["〜2m/s", "2〜4m/s", "4〜6m/s", "6m/s〜"])
HEAD_BINS = ([-99, -4, -2, 2, 4, 99], ["追い風\n4m/s〜", "追い風\n2〜4m/s", "弱い\n±2m/s", "向かい風\n2〜4m/s", "向かい風\n4m/s〜"])
PREV_BINS = ([-99, -2, 2, 4, 99], ["追い風\n2m/s〜", "弱い\n±2m/s", "向かい風\n2〜4m/s", "向かい風\n4m/s〜"])


# ── データ ──────────────────────────────────────────


def _circ_mean(a) -> float:
    r = np.radians(np.asarray(a, dtype=float))
    return float((np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) + 360) % 360)


def _circ_median(a) -> float:
    c = _circ_mean(a)
    d = ((np.asarray(a, dtype=float) - c + 180) % 360) - 180
    return float((c + np.median(d)) % 360)


def home_headings() -> pd.DataFrame:
    """場ごとの最後の直線の進行方位（北=0、時計回り）。芝とダートの直線は並行なので、全コースの中央値で外れ値（形状推定のぶれ）を抑える。"""
    from src.scripts.data.ptrain_wind import PTS_STEP, course_headings

    rows = []
    for (vc, _surf, dist), hs in course_headings(json.loads(ROUTES.read_text(encoding="utf-8"))).items():
        for h in hs:
            s = np.arange(len(h)) * PTS_STEP
            rows.append((vc, _circ_mean(h[s >= dist - HOME_M])))
    return pd.DataFrame(rows, columns=["venue_code", "h"]).groupby("venue_code")["h"].agg(_circ_median).rename("home_heading").reset_index()


def load() -> pd.DataFrame:
    cols = ["race_id", "date", "venue", "venue_code", "surface", "distance", "field_size", "finish_position", "horse_id", "horse_number",
            "odds", "popularity", "grade", "track_condition", "start_time", "passing_order", "last_3f"]
    years = [y for y in range(FIRST_YEAR - 1, LAST_YEAR + 1) if (TABLES / str(y) / "race_result_flat.parquet").exists()]
    r = pd.concat([pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in years], ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"])].copy()
    r["venue_code"] = r["venue_code"].astype(str).str.zfill(2)
    r["date"] = pd.to_datetime(r["date"].astype(str).str[:10])
    r["fin"] = pd.to_numeric(r["finish_position"], errors="coerce")
    r = r[(r["odds"] > 0) & (r["popularity"] > 0)].copy()                  # 出走していない馬（取消・除外）を除く。中止・失格は「3着以内でない」として残す
    r["starters"] = r.groupby("race_id")["fin"].transform("size")          # 実際に走った頭数
    po = r["passing_order"].astype(str).str.split("-")
    r["q_first"] = (pd.to_numeric(po.str[0], errors="coerce") - 1) / (r["starters"] - 1)
    r["style"] = pd.cut(r["q_first"], [-0.01, 1e-4, 0.25, 0.65, 1.01], labels=STYLES)
    r = r.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    g = r.groupby("horse_id")
    r["q_prev"] = g["q_first"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    r["style_pre"] = pd.cut(r["q_prev"], [-0.01, 0.15, 0.4, 0.7, 1.01], labels=PRE_STYLES)
    r = r.merge(home_headings(), on="venue_code", how="left")

    w = pd.read_parquet(WIND, columns=["venue_code", "observed_at", "wind_dir", "wind_dir_deg", "wind_speed_ms", "station_name", "station_distance_km"])
    w["venue_code"] = w["venue_code"].astype(str).str.zfill(2)
    w["obs_time"] = pd.to_datetime(w["observed_at"].astype(str).str[:19])
    w = w.drop(columns="observed_at").drop_duplicates(["venue_code", "obs_time"])
    ts = pd.to_datetime(r["date"].dt.strftime("%Y-%m-%d") + " " + r["start_time"].astype(str), errors="coerce")
    r["obs_time"] = ts.dt.floor("h")                                       # 発走の直前の正時（発走後の観測は使わない）
    r = r.merge(w, on=["venue_code", "obs_time"], how="left")
    rel = np.radians(r["wind_dir_deg"] - r["home_heading"])
    r["head"] = r["wind_speed_ms"] * np.cos(rel)
    r["cross"] = r["wind_speed_ms"] * np.sin(rel)
    calm = r["wind_dir"] == "静穏"
    r.loc[calm, ["head", "cross"]] = 0.0
    # 前走の風（前走も平地・風の観測ありのときだけ）。前走から 120 日以内
    r = r.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    g = r.groupby("horse_id")
    r["prev_head"] = g["head"].shift()
    r["prev_fin"] = g["fin"].shift()
    r["prev_venue"] = g["venue"].shift()
    r["gap"] = (r["date"] - g["date"].shift()).dt.days
    r = r[(r["date"].dt.year >= FIRST_YEAR) & r["head"].notna()].copy()
    r["top3"] = r["fin"].between(1, 3).astype(float)
    r["win"] = (r["fin"] == 1).astype(float)
    r["ret"] = np.where(r["fin"] == 1, r["odds"], 0.0)
    r["month"] = r["date"].dt.month
    r["key"] = r["venue"] + r["surface"] + r["distance"].astype(str)
    return r


# ── 統計 ──────────────────────────────────────────


def wilson(k: float, n: int) -> list[float] | None:
    if n == 0:
        return None
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mean_ci(v: pd.Series) -> list[float] | None:
    n = len(v)
    if n < 2:
        return None
    m, se = float(v.mean()), float(v.std(ddof=1)) / math.sqrt(n)
    return [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]


def ols(formula: str, d: pd.DataFrame, term: str, cluster: bool = True) -> tuple[float, float, float]:
    """係数・標準誤差・p 値（同じレースの出走は独立でないので、レース単位のクラスタ頑健 SE）。"""
    import statsmodels.formula.api as smf

    kw = {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(d["race_id"])[0]}} if cluster else {"cov_type": "HC1"}
    m = smf.ols(formula, d).fit(**kw)
    return float(m.params[term]), float(m.bse[term]), float(m.pvalues[term])


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def rate_series(d: pd.DataFrame, group: str, groups: list[str], xcol: str, xs: list[str], metric: str = "top3") -> list[dict[str, Any]]:
    out = []
    for gname in groups:
        vals, ns, cis = [], [], []
        for x in xs:
            v = d.loc[(d[group] == gname) & (d[xcol] == x), metric]
            n = len(v)
            vals.append(round(float(v.mean()), 4) if n else None)
            ns.append(n)
            cis.append(wilson(v.sum(), n) if metric in ("top3", "win") else mean_ci(v))
        out.append({"name": gname, "values": vals, "n": ns, "ci": cis})
    return out


def pct(v: float, d: int = 1) -> str:
    return f"{v * 100:.{d}f}%"


def pfmt(p: float) -> str:
    """「p<0.001」「p=0.052」の形（p= を含む）。"""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


# ── 集計 ──────────────────────────────────────────


def analyze(r: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    races = r.drop_duplicates("race_id")
    out["n_races"], out["n_runs"] = len(races), len(r)
    out["period"] = (races["date"].min().strftime("%Y-%m-%d"), races["date"].max().strftime("%Y-%m-%d"))

    # 1. 上がり3F（レース平均）への向かい風の効果。コース（場・芝ダ・距離）・馬場・クラス・月・年をそろえる
    r["l3"] = pd.to_numeric(r["last_3f"], errors="coerce")
    rr = r[r["l3"].between(30, 45)].groupby("race_id").agg(
        l3=("l3", "mean"), head=("head", "first"), venue=("venue", "first"), key=("key", "first"), tc=("track_condition", "first"),
        grade=("grade", "first"), month=("month", "first"), year=("date", lambda s: s.iloc[0].year)).reset_index()
    f_l3 = "l3 ~ head + C(key) + C(tc) + C(grade) + C(month) + C(year)"
    out["l3_all"] = ols(f_l3, rr, "head", cluster=False)
    venues = {}
    for v in VENUE_ORDER:
        g = r[r["venue"] == v]
        gr = g.drop_duplicates("race_id")
        e = ols(f_l3, rr[rr["venue"] == v], "head", cluster=False)
        g2 = g[g["style"].notna()].assign(front=lambda x: x["style"].isin(["逃げ", "先行"]).astype(float))
        fe = ols("top3 ~ front*head + C(key) + C(track_condition)", g2, "front:head")
        venues[v] = {
            "races": len(gr), "heading": float(gr["home_heading"].iloc[0]), "station": gr["station_name"].iloc[0],
            "station_km": float(gr["station_distance_km"].iloc[0]), "speed": float(gr["wind_speed_ms"].mean()),
            "head_share": float((gr["head"] > 2).mean()), "tail_share": float((gr["head"] < -2).mean()),
            "top_dirs": gr["wind_dir"].value_counts(normalize=True).head(2).index.tolist(),
            "l3": e, "front_head": fe,
        }
    adj = holm([venues[v]["front_head"][2] for v in VENUE_ORDER])
    for v, a in zip(VENUE_ORDER, adj):
        venues[v]["front_head_holm"] = a
    out["venues"] = venues

    # 2. 当日の脚質 × 風速 / 直線の風
    s = r[r["style"].notna()].copy()
    s["sb"] = pd.cut(s["wind_speed_ms"], SPEED_BINS[0], labels=SPEED_BINS[1]).astype(str)
    s["hb"] = pd.cut(s["head"], HEAD_BINS[0], labels=HEAD_BINS[1]).astype(str)
    s["back"] = s["style"].isin(["差し", "追込"]).astype(float)
    out["style_speed"] = rate_series(s, "style", STYLES, "sb", SPEED_BINS[1])
    out["style_head"] = rate_series(s, "style", STYLES, "hb", HEAD_BINS[1])
    f = "top3 ~ back*wind_speed_ms + back*head + C(key) + C(track_condition) + C(month) + back:C(month)"
    out["back_speed"] = ols(f, s, "back:wind_speed_ms")
    out["back_head"] = ols(f, s, "back:head")

    # 3. 事前の脚質（予想の時点で分かる）× 風速。オッズをそろえる前・後
    e = s[s["style_pre"].notna() & (s["odds"] > 0)].copy()
    e["back_pre"] = e["style_pre"].isin(["中団", "後方"]).astype(float)
    e["lq"] = np.log(e["odds"])
    base = "C(key) + C(track_condition) + C(month) + back_pre:C(month)"
    out["pre_speed"] = ols(f"top3 ~ back_pre*wind_speed_ms + {base}", e, "back_pre:wind_speed_ms")
    out["pre_speed_odds"] = ols(f"top3 ~ back_pre*wind_speed_ms + lq + I(lq**2) + {base}", e, "back_pre:wind_speed_ms")
    e["sb"] = pd.cut(e["wind_speed_ms"], SPEED_BINS[0], labels=SPEED_BINS[1]).astype(str)
    out["pre_roi"] = {lab: {sb: (float(x["ret"].mean()), len(x), mean_ci(x["ret"])) for sb, x in e[e["back_pre"] == bp].groupby("sb")}
                      for bp, lab in [(1.0, "中団・後方"), (0.0, "前寄り・好位")]}

    # 4. 1 番人気（荒れやすさ）× 風速
    fav = r[(r["popularity"] == 1) & (r["odds"] > 0)].copy()
    fav["lq"] = np.log(fav["odds"])
    fav["sb"] = pd.cut(fav["wind_speed_ms"], SPEED_BINS[0], labels=SPEED_BINS[1]).astype(str)
    out["fav"] = {sb: (float(x["top3"].mean()), len(x)) for sb, x in fav.groupby("sb")}
    out["fav_speed"] = ols("top3 ~ wind_speed_ms + lq + C(key) + C(track_condition) + C(month)", fav, "wind_speed_ms", cluster=False)

    # 5. 横風 × 枠（内から吹く横風 = 正）
    c = r.copy()
    c["cross_in"] = np.where(c["venue"].isin(RIGHT_HANDED), c["cross"], -c["cross"])
    c["outer"] = (c["horse_number"] / c["starters"] > 2 / 3).astype(float)
    out["cross_gate"] = ols("top3 ~ outer*cross_in + C(key) + C(track_condition)", c, "outer:cross_in")

    # 6. 前走の直線の風 → 今走（オッズをそろえる）
    n = r[r["prev_head"].notna() & (r["gap"] <= 120) & (r["odds"] > 0)].copy()
    n["lq"] = np.log(n["odds"])
    n["pb"] = pd.cut(n["prev_head"], PREV_BINS[0], labels=PREV_BINS[1]).astype(str)
    n["grp"] = np.where(n["popularity"] >= 7, "7番人気以下", "6番人気以内")
    fp = "top3 ~ prev_head + lq + I(lq**2) + C(key) + C(month) + C(prev_venue) + C(track_condition)"
    out["prev"] = ols(fp, n, "prev_head")
    out["prev_half"] = {nm: ols(fp, n[m], "prev_head") for nm, m in [("2020〜2022年", n["date"].dt.year <= 2022), ("2023年〜", n["date"].dt.year >= 2023)]}
    alls = n.assign(grp="全体")
    out["prev_roi"] = rate_series(pd.concat([alls, n]), "grp", ["全体", "6番人気以内", "7番人気以下"], "pb", PREV_BINS[1], metric="ret")
    out["prev_roi_half"] = {nm: rate_series(n[m].assign(grp="全体"), "grp", ["全体"], "pb", PREV_BINS[1], metric="ret")[0]["values"]
                            for nm, m in [("2020〜2022年", n["date"].dt.year <= 2022), ("2023年〜", n["date"].dt.year >= 2023)]}
    return out


# ── 記事 ──────────────────────────────────────────


def compass(deg: float) -> str:
    return COMPASS16[int(((deg % 360) + 11.25) // 22.5) % 16]


def article(a: dict[str, Any]) -> dict[str, Any]:
    V = a["venues"]
    l3, l3se, _ = a["l3_all"]
    bs, bsse, bsp = a["back_speed"]
    bh, _, bhp = a["back_head"]
    ps, _, psp = a["pre_speed"]
    pso, _, psop = a["pre_speed_odds"]
    pv, _, pvp = a["prev"]
    fs, _, fsp = a["fav_speed"]
    cg, _, cgp = a["cross_gate"]
    st_speed = {s["name"]: s for s in a["style_speed"]}
    nige_lo, nige_hi = st_speed["逃げ"]["values"][0], st_speed["逃げ"]["values"][-1]
    oi_lo, oi_hi = st_speed["追込"]["values"][0], st_speed["追込"]["values"][-1]
    roi = {s["name"]: s for s in a["prev_roi"]}
    r_all, r_lo = roi["全体"]["values"], roi["7番人気以下"]["values"]
    ci_all, ci_lo = roi["全体"]["ci"][-1], roi["7番人気以下"]["ci"][-1]
    pr = a["pre_roi"]
    calm, windy = SPEED_BINS[1][0], SPEED_BINS[1][-1]
    sig = [v for v in VENUE_ORDER if V[v]["front_head"][2] < 0.05]
    sig_txt = "、".join(f"{v}（{V[v]['front_head'][0] * 100:+.2f}pt、{pfmt(V[v]['front_head'][2])}、補正後 {pfmt(V[v]['front_head_holm'])}）" for v in sig) or "なし"
    halves = a["prev_half"]
    p0, p1 = a["period"]

    venue_rows = []
    for v in VENUE_ORDER:
        x = V[v]
        e, se, p = x["l3"]
        venue_rows.append([v, f"{compass(x['heading'])}（{x['heading']:.0f}°）", f"{x['station']}（{x['station_km']:.1f}km）", f"{x['speed']:.1f}m/s",
                           "・".join(x["top_dirs"]), pct(x["head_share"], 0), pct(x["tail_share"], 0), f"{e:+.3f}秒（±{1.96 * se:.3f}）",
                           f"{x['front_head'][0] * 100:+.2f}pt（{pfmt(x['front_head'][2])}）", x["races"]])
    windiest = sorted(VENUE_ORDER, key=lambda v: -V[v]["speed"])[:3]
    tokyo = V["東京"]

    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "直線の向かい風 1m/s で上がり3F", "value": f"{l3:+.3f}秒", "sub": f"95%区間 ±{1.96 * l3se:.3f}秒。10場すべてで遅くなる"},
            {"label": f"逃げ馬の複勝率（風速{calm}→{windy}）", "value": f"{pct(nige_lo)}→{pct(nige_hi)}", "sub": "効くのは向きより風の強さ"},
            {"label": "前走が向かい風4m/s〜の馬の次走単勝回収率", "value": pct(r_all[-1]), "sub": f"95%区間 {pct(ci_all[0])}〜{pct(ci_all[1])}。前走の風が弱い馬は {pct(r_all[1])}"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"1. 直線の向かい風は、上がり3F をはっきり遅くします。向かい風が 1m/s 強まるごとにレースの上がり3F（平均）は約 {l3:.2f} 秒遅くなり、10 場すべてで同じ向きです。"
            f"5m/s の向かい風なら約 {l3 * 5:.2f} 秒で、上がりを比べるときは風の分を割り引いて見る必要があります（向こう正面は逆に追い風になるので、走破タイム全体が同じだけ遅くなるとは限りません）。\n\n"
            f"2. 「向かい風なら差し有利（または前有利）」という向きの効果は、全体ではほぼ見られません（差し・追込の有利さは向かい風 1m/s あたり {bh * 100:+.2f}pt、{pfmt(bhp)}）。"
            f"効いているのは風の強さで、風が強い日ほど逃げ馬の複勝率が下がり、差し・追込が上がります（風速 1m/s あたり {bs * 100:+.2f}pt、{pfmt(bsp)}）。\n\n"
            f"3. ただしこの「強風で差し」は、単勝オッズにほぼ織り込まれています。予想の時点で分かる脚質（前走までの位置取り）で見ると効果は {ps * 100:+.2f}pt/(m/s)（{pfmt(psp)}）ありますが、"
            f"オッズをそろえると {pso * 100:+.2f}pt（{pfmt(psop)}）に縮み、強風の日に中団・後方型を買っても回収率はほとんど変わりません。\n\n"
            f"4. 予想に使えそうなのは「前走の向かい風」です。前走の直線で強い向かい風を受けた馬は、次走でオッズから見込まれるより少し好走し（前走の向かい風 1m/s あたり {pv * 100:+.2f}pt、{pfmt(pvp)}）、"
            f"7 番人気以下では回収率の差が大きく見えます（前走が向かい風 4m/s〜: {pct(r_lo[-1])}、弱い: {pct(r_lo[1])}。ただし 95% 区間は広い）。向かい風で時計がかかった分を、市場が割り引けていない可能性があります。")},
        {"type": "heading", "text": "競馬場ごとの直線の風"},
        {"type": "text", "text": (
            f"同じ「北風」でも、直線の向きによって追い風にも向かい風にもなります。そこで各場の最後の直線の向き（コース形状から計算）と、発走直前の風から、直線での向かい風の成分を出しました。\n\n"
            f"風が強いのは{'・'.join(windiest)}で、直線で 2m/s を超える向かい風・追い風を受けるレースが多くなります。"
            f"一方、東京は直線が西を向いていて、風は南北に吹くことが多いため、直線の向かい風・追い風は小さめです（2m/s 超の向かい風は {pct(tokyo['head_share'], 0)}、追い風は {pct(tokyo['tail_share'], 0)}）。"
            f"東京の風は主に横風として吹いています。")},
        {"type": "table", "columns": ["競馬場", "直線の向き", "観測所（場からの距離）", "平均風速", "多い風向", "向かい風 2m/s超", "追い風 2m/s超", "上がり3F（向かい風 1m/s あたり）", "逃げ・先行の有利さ（向かい風 1m/s あたり）", "レース数"],
         "rows": venue_rows,
         "caption": (f"上がり3F は、コース（場・芝ダ・距離）・馬場・クラス・月・年をそろえた回帰の係数（括弧は 95% 区間の幅）。"
                     f"「逃げ・先行の有利さ」は、当日の初角で前にいた馬の複勝率が、ほかの馬と比べて向かい風 1m/s あたり何 pt 変わるか。"
                     f"10 場を同時に調べているので、p 値は多重比較の補正（Holm 法）が必要です。p<0.05 の場: {sig_txt}。補正後に有意な場はありません。"
                     "東京は直線で強い向かい風・追い風になることが少ないため、上がり3F の係数は弱い風の範囲だけからの推定です。")},
        {"type": "heading", "text": "どの脚質が有利になるか"},
        {"type": "chart", "title": "風速別 脚質ごとの複勝率", "xLabel": "発走直前の風速", "yLabel": "複勝率", "xs": SPEED_BINS[1],
         "series": a["style_speed"],
         "caption": f"{p0[:4]}〜{p1[:4]}年 JRA 平地。脚質はそのレースの初角の位置（逃げ=先頭、先行=前から25%以内、差し=65%以内、追込=それより後ろ）。縦線は 95% 信頼区間。"},
        {"type": "chart", "title": "直線の風（向かい風・追い風）別 脚質ごとの複勝率", "xLabel": "最後の直線での風", "yLabel": "複勝率", "xs": HEAD_BINS[1],
         "series": a["style_head"],
         "caption": "4m/s を超えると、向かい風・追い風のどちらでも逃げが下がり追込が上がる。2〜4m/s では向きによる差ははっきりしない。"},
        {"type": "text", "text": (
            f"風速が {calm} のとき逃げ馬の複勝率は {pct(nige_lo)} ですが、{windy} では {pct(nige_hi)} に下がります。追込は {pct(oi_lo)} から {pct(oi_hi)} に上がります。"
            f"月（季節風の時期）・コース・馬場をそろえても、差し・追込の有利さは風速 1m/s あたり {bs * 100:+.2f}pt（95% 区間 ±{1.96 * bsse * 100:.2f}pt）です。\n\n"
            "4m/s を超える強い風のとき、向かい風と追い風の両方で同じ傾向になるのは、直線で向かい風なら向こう正面は追い風（逆も同じ）で、どちら向きでも一周のどこかで風を受けるためと考えられます。"
            "下がるのはほぼ逃げ馬だけで、先行は横ばいです。先頭で風を受け続ける逃げ馬が消耗し、後ろで風よけができる差し・追込が相対的に有利になる、という説明と整合します。")},
        {"type": "heading", "text": "予想に使えるか"},
        {"type": "text", "text": (
            "上のグラフの脚質はレースの結果（実際の位置取り）なので、レース前には分かりません。予想で使うには、前走までの位置取りから決めた脚質で見る必要があります。\n\n"
            f"前走まで直近 3 走の初角の位置で「中団・後方型」とした馬でも、風が強いほど有利になる傾向は残ります（{ps * 100:+.2f}pt/(m/s)、{pfmt(psp)}）。"
            f"しかし単勝オッズを入れてそろえると {pso * 100:+.2f}pt（{pfmt(psop)}）とほぼ消え、市場はすでに風を織り込んでいると考えられます。"
            f"中団・後方型の単勝回収率も、風速 {calm} で {pct(pr['中団・後方'][calm][0])}、{windy} で {pct(pr['中団・後方'][windy][0])} と、誤差の範囲（95% 区間 {pct(pr['中団・後方'][windy][2][0])}〜{pct(pr['中団・後方'][windy][2][1])}）の違いです。")},
        {"type": "chart", "title": "前走の直線の風 × 今走の単勝回収率", "xLabel": "前走の最後の直線での風", "yLabel": "単勝回収率", "xs": PREV_BINS[1],
         "series": a["prev_roi"],
         "caption": "前走から 120 日以内の出走。単勝を 100 円ずつ買ったときの回収率（縦線は 95% 信頼区間。回収率は高配当 1 回で大きく動くので区間が広い）。"},
        {"type": "text", "text": (
            "一方、前走の風は市場が割り引けていない可能性があります。前走の直線で向かい風が強かった馬ほど、今走でオッズから見込まれるより好走しています"
            f"（オッズ・コース・月・前走の場・馬場をそろえて、前走の向かい風 1m/s あたり複勝率 {pv * 100:+.2f}pt、{pfmt(pvp)}）。"
            f"期間を分けても向きは同じです（{'、'.join(f'{k}: {v[0] * 100:+.2f}pt・{pfmt(v[2])}' for k, v in halves.items())}）。\n\n"
            f"単勝回収率は、前走が向かい風 4m/s〜の馬で {pct(r_all[-1])}、前走の風が弱い馬で {pct(r_all[1])} です。7 番人気以下に限ると {pct(r_lo[-1])}（95% 区間 {pct(ci_lo[0])}〜{pct(ci_lo[1])}）対 {pct(r_lo[1])} と差が広がりますが、区間の下限は比較相手とほぼ同じで、人気薄に絞った差は確かとは言えません。"
            "向かい風で上がりがかかった分が、次走の人気で割り引かれていない可能性があります。\n\n"
            "ただし回収率は 100% を下回っており、これだけで儲かる条件ではありません。「前走の着順や上がりを、風の分だけ割り引いて見直す材料」として使うのが妥当です。")},
        {"type": "heading", "text": "影響が見られなかったこと"},
        {"type": "text", "text": (
            f"・荒れやすさ: 1 番人気の複勝率は、風速 {calm} で {pct(a['fav'][calm][0])}、{windy} で {pct(a['fav'][windy][0])} ですが、"
            f"オッズとコースをそろえると風速 1m/s あたり {fs * 100:+.2f}pt（{pfmt(fsp)}）で、強風で荒れるとは言えません。\n\n"
            f"・横風と枠順: 「内から吹く横風なら外枠が不利」などの関係は、外枠×横風 1m/s あたり {cg * 100:+.2f}pt（{pfmt(cgp)}）と小さく、"
            "多くの側面を調べた中の 1 つなので、偶然の範囲と見ています。\n\n"
            f"・場ごとの向きの効果: 向かい風で逃げ・先行が不利になるかを場ごとに見ると、p<0.05 は {sig_txt} でしたが、10 場の多重比較を補正すると有意ではありません。")},
        {"type": "accordion", "title": "データと方法・注意点", "blocks": [
            {"type": "text", "text": (
                f"・対象: {p0}〜{p1} の JRA 平地 {a['n_races']:,} レース・{a['n_runs']:,} 出走（取消・除外を除く。中止・失格は 3 着以内でないとして数える）。2026 年は風のデータがそろっている {p1} までです。\n\n"
                "・風: 気象庁「過去の気象データ検索」の時別値。各場に最も近い、風を観測している観測所の、発走の直前の正時の値（その時刻の前 10 分間の平均）。"
                "発走時刻をまたぐ観測は使っていません。\n\n"
                "・直線の向き: OpenStreetMap のコース形状から、ゴール手前 250m の進行方位を場ごとに求めたもの（芝とダートの直線は並行なので場で 1 つ）。"
                "向かい風の成分 = 風速 × cos(風向 − 直線の向き)。\n\n"
                "・注意: 観測所は場から 1.5〜12km 離れていて（小倉は約 12km の八幡）、スタンドや地形で、コース上の風は観測値と違うことがあります。"
                "風向は 16 方位の観測で、風の強さは 10 分平均のため、レース中の突風は分かりません。"
                "また、予想の時点で使えるのは予報で、ここで使った観測値ではありません。\n\n"
                "・季節: 冬は季節風で風が強く、馬場やクラス構成も季節で変わるため、回帰では月をそろえています。"
                "単勝回収率はばらつきが大きく、95% 区間が広いことに注意してください。")},
        ]},
    ]
    return {
        "slug": "wind-venue",
        "title": "競馬場ごとの風の影響——効くのは向きより強さ",
        "claim": "風（直線の向かい風・追い風、風の強さ）で有利な脚質が変わる",
        "verdict": "partial",
        "summary": (f"直線の向かい風は 1m/s で上がり3F を約 {l3:.2f} 秒遅くします。有利な脚質を変えるのは向きより風の強さで、強風ほど差し・追込が有利ですが、オッズに織り込み済み。"
                    "狙い目になりうるのは「前走で強い向かい風を受けた馬」です。"),
        "categories": ["weather/wind", "style/front-runner"],
        "published": "2026-11-01",
        "dataset": {
            "period": f"{p0[:4]}年1月〜{p1[:4]}年{int(p1[5:7])}月",
            "conditions": "JRA 全 10 場・芝ダート（障害除く）・気象庁の時別値（各場の最寄り観測所）",
            "sample": f"{a['n_races']:,} レース・{a['n_runs']:,} 出走",
        },
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not WIND.exists():
        raise SystemExit(f"{WIND} がありません。先に python -m src.scraper.jma_weather export を実行してください")
    col = article(analyze(load()))
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))      # NaN を書かない
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
