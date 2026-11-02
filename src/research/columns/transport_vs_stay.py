"""めぐコラム『輸送は成績を落とす？』を実データから集計し、frontend/content/columns/transport-vs-stay.json を書く。

問い: 厩舎（美浦・栗東のトレセン）から競馬場までが遠いほど、成績は落ちるか。現地に滞在して調整した馬と、
      当週に輸送した馬で違うか。2〜3 歳と 4 歳以上で効き方が違うか。

データ:
  レース:   data/page_reference/tables/<年>/race_result_flat.parquet（compare_series.load と同じ。2020〜2025 年の JRA 平地、取消・除外を除く）
  追い切り: data/local/modeling/training_pipeline/features/training_laps.parquet（horse_training 由来。馬・レース日ごとの追い切りのコース）
定義:
  所属（美浦/栗東）: 調教師ごとに、その調教師の馬の追い切りが美浦・栗東のどちらの施設で多いか（tp33_transport.trainer_home）。
                     出走データには所属の列がないため。
  距離: 所属トレセンから競馬場までの大圏距離（緯度経度から求めた直線距離。道のりではない）。
  滞在 / 輸送: レース前 1〜28 日の追い切りのうち、場所の分かる最後の 1 本が
               所属トレセン以外（函館・札幌・小倉などの競馬場、もう一方のトレセン）なら「滞在」、所属トレセンなら「輸送」。
               28 日以内に場所の分かる追い切りが無い出走は「判定不能」として除く。
  参考の判定: 前走が同じ地域の遠方の場（距離 300km 以上）で、間隔 35 日以内の連闘・続戦を「滞在の可能性が高い」とする
              （追い切りが無い場合の代わりの目安。追い切りによる判定とどれくらい一致するかを示す）。

使い方:
    python -m src.research.columns.transport_vs_stay            # 集計して記事を書く
    python -m src.research.columns.transport_vs_stay --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.features.tp33_transport import COORD, VENUES, prep_laps, trainer_home
from src.research.columns.compare_series import FLAT_DIR, REGION, YEARS, _p, _stat, load
from src.research.columns.race_context import joint_p, ols_fit, pair_chart, series
from src.research.columns.wind_venue import holm, pct, pfmt

REPO = Path(__file__).resolve().parents[3]
LAPS = REPO / "data/local/modeling/training_pipeline/features/training_laps.parquet"
OUT = REPO / "frontend/content/columns/transport-vs-stay.json"

HOMES = {"MI": "関東馬（美浦）", "KU": "関西馬（栗東）"}
KM_EDGES = [0, 100, 300, 500, 800, 99999]
KM_BANDS = ["〜100km", "100〜\n300km", "300〜\n500km", "500〜\n800km", "800km〜"]
AGES = {"2〜3歳": (2, 3), "4歳以上": (4, 99)}
CLASSES = ["輸送", "滞在"]
STAY_ROT_KM = 300        # 参考の判定（前走と同じ地域の遠方の場）で「遠方」とみなす距離
STAY_ROT_DAYS = 35


def gc_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """2 点の大圏距離（km）。道のりではない。"""
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(x))


VENUE_CODE = {v: k for k, v in VENUES.items()}
KM = {(h, VENUES[c]): gc_km(COORD[h], COORD[c]) for h in HOMES for c in VENUES}       # (所属, 競馬場名) → km


def km_band(km: pd.Series) -> pd.Series:
    return pd.cut(km, KM_EDGES, labels=KM_BANDS, right=False).astype(str)


def stay_class(last_loc: pd.Series, home: pd.Series) -> pd.Series:
    """最後の追い切りの場所（KU/MI/01〜10、無ければ NaN）と所属から、輸送 / 滞在 / 判定不能。"""
    known = last_loc.notna()
    return pd.Series(np.where(~known, "判定不能", np.where(last_loc == home, "輸送", "滞在")), index=last_loc.index)


def rotation_stay(d: pd.DataFrame) -> pd.Series:
    """参考の判定: 前走が同じ地域（北海道は札幌・函館をまとめる）の遠方の場で、間隔が短い。d は馬・日付順。"""
    g = d.groupby("horse_id")
    pv, pdate = g["venue"].shift(), pd.to_datetime(g["date"].shift())
    days = (pd.to_datetime(d["date"]) - pdate).dt.days
    return (pv.map(REGION) == d["venue"].map(REGION)) & (d["km"] >= STAY_ROT_KM) & (days <= STAY_ROT_DAYS)


def frame() -> pd.DataFrame:
    df = load()
    tr = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_id", "trainer_id"]) for y in YEARS])
    tr["horse_id"], tr["trainer_id"] = tr["horse_id"].astype(str), tr["trainer_id"].astype(str)
    df = df.merge(tr.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    L = prep_laps(pd.read_parquet(LAPS, columns=["horse_id", "race_date", "train_date", "course", "days_before"]))
    df["home"] = df["trainer_id"].map(trainer_home(L, df))
    df = df[df["home"].isin(list(HOMES))].copy()
    df["km"] = [KM[(h, v)] for h, v in zip(df["home"], df["venue"])]
    df["kb"] = km_band(df["km"])
    # 滞在の判定: レース前 1〜28 日で、場所の分かる最後の追い切り
    W = L[L["days_before"].between(1, 28) & L["loc"].isin(list(HOMES) + list(VENUES))]
    last = W.sort_values("days_before").drop_duplicates(["horse_id", "race_date"])
    last = last.rename(columns={"race_date": "date", "loc": "last_loc"})[["horse_id", "date", "last_loc"]]
    df = df.merge(last, on=["horse_id", "date"], how="left")
    df["cls"] = stay_class(df["last_loc"], df["home"])
    df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    df["rot_stay"] = rotation_stay(df)
    g = df.groupby("horse_id")
    df["prev_pop"] = g["popularity"].shift()
    df["prev_fin"] = g["finish_position"].shift()
    df["age_g"] = np.where(df["age"] <= 3, "2〜3歳", "4歳以上")
    df["young"] = (df["age"] <= 3).astype(float)
    df["lq"] = np.log(df["odds"])
    df["key"] = df["venue"] + df["surface"] + df["distance"].astype(str)
    df["km100"] = df["km"] / 100
    df["stay"] = (df["cls"] == "滞在").astype(float)
    df["ex"] = df["win"] - df["implied"]          # 実際の勝ち − オッズから見込まれる勝率
    return df


# ── 集計 ──────────────────────────────────────────────

FILTERS = {
    "所属": {"全体": None, **{v: (lambda h: (lambda d: d["home"] == h))(k) for k, v in HOMES.items()}},
    "年齢": {"全体": None, **{k: (lambda lo, hi: (lambda d: d["age"].between(lo, hi)))(*v) for k, v in AGES.items()}},
}


def market_chart(x: pd.DataFrame) -> dict[str, Any]:
    """輸送・滞在それぞれの、実際の勝率とオッズから見込まれる勝率（距離帯ごと）。所属・年齢・輸送/滞在で切り替える。"""
    names = ("実際の勝率", "オッズから見込まれる勝率")

    def build(w: pd.DataFrame):
        parts = {nm: [w.loc[w["kb"] == b, m] for b in KM_BANDS] for nm, m in zip(names, ("win", "implied"))}
        ser = [series(nm, parts[nm]) for nm in names]
        ps = []
        for a, b in zip(parts[names[0]], parts[names[1]]):
            ma, _, _, va = _stat(a)
            mb, _, _, vb = _stat(b)
            p = _p(va, vb, ma, mb)
            ps.append(None if p is None else float(f"{p:.3g}"))
        return ser, {"a": names[0], "b": names[1], "p": ps}

    cls_opts = {"輸送": "輸送", "滞在": "滞在"}
    base = x[x["cls"] == "輸送"]
    ser, cmp = build(base)
    variants = {}
    for c in cls_opts:
        for hk, hf in FILTERS["所属"].items():
            for ak, af in FILTERS["年齢"].items():
                if c == "輸送" and hf is None and af is None:
                    continue
                w = x[x["cls"] == c]
                if hf is not None:
                    w = w[hf(w)]
                if af is not None:
                    w = w[af(w)]
                s2, c2 = build(w)
                variants[f"{c}|{hk}|{ak}"] = {"series": s2, "compare": c2}
    return {"type": "chart", "title": "厩舎からの距離別 勝率（実際とオッズの見込み）", "xLabel": "所属トレセンから競馬場までの直線距離", "yLabel": "勝率",
            "xs": KM_BANDS, "series": ser, "compare": cmp,
            "filters": [{"label": "輸送・滞在", "options": list(cls_opts)}] + [{"label": k, "options": list(v)} for k, v in FILTERS.items()],
            "variants": variants,
            "caption": "オッズから見込まれる勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。2 本の線が重なっていれば、遠征の影響は市場がオッズに織り込んでいる。＊は実際と見込みの差が有意（p<0.05）。"}


def age_chart(x: pd.DataFrame) -> dict[str, Any]:
    t = x[x["cls"] == "輸送"]
    return pair_chart(t, t["age"] <= 3, "kb", KM_BANDS, "top3", "2〜3歳", "4歳以上",
                      title="厩舎からの距離別 複勝率（輸送した馬・2〜3歳 vs 4歳以上）", x_label="所属トレセンから競馬場までの直線距離", y_label="複勝率",
                      caption="当週に輸送した馬（直前の追い切りが所属トレセン）だけ。年齢による元の差（2〜3歳は未勝利・新馬が多い）があるので、"
                              "見るのは線の高さではなく、距離が延びたときの下がり方の違い。＊は 2 つの年齢の差が有意（p<0.05）。",
                      filters={"所属": FILTERS["所属"]})


def band_rows(t: pd.DataFrame) -> list[dict[str, Any]]:
    """距離帯ごとの 実際 − 見込み（勝率）と、〜100km を基準にしたオッズ・コースをそろえた複勝率の差。"""
    rows = []
    for b in KM_BANDS:
        w = t[t["kb"] == b]
        m, n, ci, _ = _stat(w["ex"])
        rows.append({"kb": b, "n": n, "win": float(w["win"].mean()) if n else float("nan"), "top3": float(w["top3"].mean()) if n else float("nan"),
                     "implied": float(w["implied"].mean()) if n else float("nan"), "ex": m, "ex_ci": ci, "pop": float(w["popularity"].mean()) if n else float("nan")})
    return rows


def tests(x: pd.DataFrame) -> dict[str, Any]:
    """輸送した馬で、距離が効くか（オッズ・コース・所属をそろえる）、年齢で効き方が変わるか。滞在と輸送の差。"""
    t = x[x["cls"] == "輸送"].copy()
    a: dict[str, Any] = {}
    ctrl = "C(home) + lq + I(lq**2) + C(key)"
    for m in ("top3", "win"):
        f = ols_fit(f"{m} ~ km100 + {ctrl}", t)
        a[f"km_{m}"] = (float(f.params["km100"]), float(f.bse["km100"]), float(f.pvalues["km100"]))
        f = ols_fit(f"{m} ~ C(kb, Treatment('{KM_BANDS[0]}')) + {ctrl}", t)
        a[f"kb_{m}"] = joint_p(f, "C(kb")
        a[f"kb_{m}_coef"] = {k.split("[T.")[1].rstrip("]"): (float(f.params[k]), float(f.bse[k]), float(f.pvalues[k])) for k in f.params.index if k.startswith("C(kb")}
    # 能力の目安を前走（人気・着順）にした場合（今回のオッズには遠征の見込みも入るため、その影響を含めた差を見る）
    tp = t[t["prev_pop"].notna() & (t["prev_fin"] >= 1)].copy()
    tp["lpp"] = np.log(tp["prev_pop"])
    tp["pfb"] = pd.cut(tp["prev_fin"], [0, 1, 3, 5, 9, 99]).astype(str)
    f = ols_fit("top3 ~ km100 + C(home) + lpp + C(pfb) + C(key)", tp)
    a["km_prev"] = (float(f.params["km100"]), float(f.bse["km100"]), float(f.pvalues["km100"]), len(tp))
    # 年齢で効き方が変わるか（統制の項もすべて年齢ごとに分けるので、年齢別に当てはめた傾きの差と一致する）
    f = ols_fit("top3 ~ km100*young + (C(home) + lq + I(lq**2) + C(key))*young", t)
    a["age_int"] = (float(f.params["km100:young"]), float(f.bse["km100:young"]), float(f.pvalues["km100:young"]))
    for ag, (lo, hi) in AGES.items():
        w = t[t["age"].between(lo, hi)]
        f = ols_fit(f"top3 ~ km100 + {ctrl}", w)
        a[f"km_age_{ag}"] = (float(f.params["km100"]), float(f.bse["km100"]), float(f.pvalues["km100"]), len(w))
    f = ols_fit("win ~ km100*young + (C(home) + lq + I(lq**2) + C(key))*young", t)
    a["age_int_win"] = (float(f.params["km100:young"]), float(f.bse["km100:young"]), float(f.pvalues["km100:young"]))
    # 同じ場・同じオッズで、遠征してきた馬と地元の馬を比べる（関西の 3 場では関東馬、関東の 2 場では関西馬が遠征組）
    for name, venues, away_home in (("kansai", ("京都", "阪神", "中京"), "MI"), ("kanto", ("東京", "中山"), "KU")):
        w = t[t["venue"].isin(venues)].assign(away=lambda d: (d["home"] == away_home).astype(float))
        for m in ("top3", "win"):
            f = ols_fit(f"{m} ~ away + lq + I(lq**2) + C(key)", w)
            a[f"venue_{name}_{m}"] = (float(f.params["away"]), float(f.bse["away"]), float(f.pvalues["away"]), int(w["away"].sum()))
    # 滞在と輸送: 同じ距離帯（300km 以上）で、オッズ・コースをそろえて比べる
    s = x[x["cls"].isin(CLASSES) & (x["km"] >= 300)].copy()
    for m in ("top3", "win"):
        f = ols_fit(f"{m} ~ stay + C(kb) + {ctrl}", s)
        a[f"stay_{m}"] = (float(f.params["stay"]), float(f.bse["stay"]), float(f.pvalues["stay"]), int(s["stay"].sum()), int((1 - s["stay"]).sum()))
    f = ols_fit("top3 ~ stay*young + (C(kb) + C(home) + lq + I(lq**2) + C(key))*young", s)
    a["stay_age_int"] = (float(f.params["stay:young"]), float(f.bse["stay:young"]), float(f.pvalues["stay:young"]))
    fams = ["km_top3", "km_win", "age_int", "stay_top3", "venue_kansai_top3", "venue_kanto_top3"]
    for k, ph in zip(fams, holm([a[k][2] for k in fams])):
        a[k + "_holm"] = ph
    return a


def coverage(x: pd.DataFrame) -> dict[str, Any]:
    """判定の内訳と、参考の判定（ローテーション）と追い切りによる判定の一致。"""
    c: dict[str, Any] = {"n_all": len(x), "cls": x["cls"].value_counts().to_dict()}
    far = x[x["km"] >= STAY_ROT_KM]
    c["far_cls"] = far["cls"].value_counts().to_dict()
    k = far[far["cls"].isin(CLASSES)]
    rs, st = k["rot_stay"], k["cls"] == "滞在"
    c["rot_precision"] = float(st[rs].mean()) if rs.any() else float("nan")       # ローテーションで滞在とした中で、追い切りでも滞在
    c["rot_recall"] = float(rs[st].mean()) if st.any() else float("nan")          # 追い切りで滞在の中で、ローテーションでも滞在
    c["rot_n"] = int(rs.sum())
    venue = far[far["cls"].isin(CLASSES)].groupby("venue")["stay"].agg(["mean", "size"])
    c["stay_by_venue"] = {v: (float(r["mean"]), int(r["size"])) for v, r in venue.iterrows()}
    c["years"] = x.groupby(x["date"].str.slice(0, 4))["cls"].apply(lambda s: float((s != "判定不能").mean())).to_dict()
    return c


def analyze(x: pd.DataFrame) -> dict[str, Any]:
    k = x[x["cls"].isin(CLASSES)]
    a = {"cov": coverage(x), "tests": tests(k)}
    t = k[k["cls"] == "輸送"]
    a["bands"] = {h: band_rows(t[t["home"] == h]) for h in HOMES}
    a["bands_all"] = band_rows(t)
    a["bands_age"] = {ag: band_rows(t[t["age"].between(*AGES[ag])]) for ag in AGES}
    a["n_cls"] = {c: int((k["cls"] == c).sum()) for c in CLASSES}
    f5 = k[k["km"] >= 500]
    a["far_pop"] = {c: (float(f5.loc[f5["cls"] == c, "popularity"].mean()), float(f5.loc[f5["cls"] == c, "top3"].mean()), int((f5["cls"] == c).sum())) for c in CLASSES}
    a["c_top3"] = pair_chart(k, k["cls"] == "輸送", "kb", KM_BANDS, "top3", "輸送", "滞在",
                             title="厩舎からの距離別 複勝率（輸送 vs 滞在）", x_label="所属トレセンから競馬場までの直線距離", y_label="複勝率",
                             caption="直前の追い切りが所属トレセンなら輸送、競馬場やもう一方のトレセンなら滞在。距離は道のりでなく直線距離。"
                                     "人気・能力の違いはそろえていない（下のオッズとの比較を参照）。＊は輸送と滞在の差が有意（p<0.05）。",
                             filters=FILTERS)
    a["c_win"] = pair_chart(k, k["cls"] == "輸送", "kb", KM_BANDS, "win", "輸送", "滞在",
                            title="厩舎からの距離別 勝率（輸送 vs 滞在）", x_label="所属トレセンから競馬場までの直線距離", y_label="勝率",
                            caption="複勝率のグラフと同じ区分で勝率を見たもの。＊は輸送と滞在の差が有意（p<0.05）。", filters=FILTERS)
    a["c_market"] = market_chart(k)
    a["c_age"] = age_chart(k)
    return a


# ── 記事 ──────────────────────────────────────────────


def pt(v: float, d: int = 1) -> str:
    return f"{v * 100:+.{d}f}pt"


def pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def venue_table() -> list[list[str]]:
    rows = []
    for c in VENUES:
        v = VENUES[c]
        rows.append([v, f"{KM[('MI', v)]:.0f}km", km_band(pd.Series([KM[('MI', v)]])).iloc[0].replace("\n", ""),
                     f"{KM[('KU', v)]:.0f}km", km_band(pd.Series([KM[('KU', v)]])).iloc[0].replace("\n", "")])
    return rows


def article(a: dict[str, Any]) -> dict[str, Any]:
    T, C = a["tests"], a["cov"]
    BM = {r["kb"]: r for r in a["bands"]["MI"]}
    BK = {r["kb"]: r for r in a["bands"]["KU"]}
    near, mid = KM_BANDS[0], KM_BANDS[2]
    e3, s3, p3 = T["km_top3"]
    ew, sw, pw = T["km_win"]
    ei, si, pi = T["age_int"]
    ey, eo = T["km_age_2〜3歳"], T["km_age_4歳以上"]
    es3, esw, ep = T["stay_top3"], T["stay_win"], T["km_prev"]
    vk3, vkw = T["venue_kansai_top3"], T["venue_kansai_win"]
    vt3, vtw = T["venue_kanto_top3"], T["venue_kanto_win"]
    kb3 = T["kb_top3_coef"][mid]
    n_t, n_s = a["n_cls"]["輸送"], a["n_cls"]["滞在"]
    n_unk = C["cls"].get("判定不能", 0)
    FP = a["far_pop"]
    km_sig = T["km_top3_holm"] < 0.05 and e3 < 0
    age_sig = T["age_int_holm"] < 0.05
    stay_sig = T["stay_top3_holm"] < 0.05
    mi_bad = T["venue_kansai_top3_holm"] < 0.05 and vk3[0] < 0
    ku_bad = T["venue_kanto_top3_holm"] < 0.05 and vt3[0] < 0
    band_tbl = []
    for h, rows in a["bands"].items():
        for r in rows:
            if r["n"] == 0:
                continue
            ci = r["ex_ci"]
            band_tbl.append([HOMES[h], r["kb"].replace("\n", ""), f"{r['n']:,}", pct(r["top3"]), pct(r["win"]), pct(r["implied"]),
                             f"{pt(r['ex'], 2)}（{pt(ci[0], 2)}〜{pt(ci[1], 2)}）", f"{r['pop']:.1f}"])
    kb_rows = [[k.replace("\n", ""), pt(v[0], 2), f"±{1.96 * v[1] * 100:.2f}pt", pval(v[2]),
                pt(T["kb_win_coef"][k][0], 2), pval(T["kb_win_coef"][k][2])] for k, v in T["kb_top3_coef"].items()]

    def row(label: str, v: tuple, holm_key: str | None = None) -> list[str]:
        return [label, pt(v[0], 2), f"±{1.96 * v[1] * 100:.2f}pt", pval(v[2]), pval(T[holm_key]) if holm_key else "–"]

    test_rows = [
        row("距離 100km ごとの複勝率の変化（輸送した馬・オッズ、コース、所属をそろえる）", T["km_top3"], "km_top3_holm"),
        row("同 勝率", T["km_win"], "km_win_holm"),
        row("同 複勝率（オッズの代わりに前走の人気・着順でそろえる）", ep),
        row("関西の 3 場（京都・阪神・中京）: 関東馬 − 関西馬 の複勝率（同じオッズ・コース）", vk3, "venue_kansai_top3_holm"),
        row("  同 勝率", vkw),
        row("関東の 2 場（東京・中山）: 関西馬 − 関東馬 の複勝率（同じオッズ・コース）", vt3, "venue_kanto_top3_holm"),
        row("  同 勝率", vtw),
        row("年齢で変わるか: 100km ごとの変化の 2〜3歳 − 4歳以上", T["age_int"], "age_int_holm"),
        row("  2〜3歳だけ: 100km ごとの複勝率の変化", ey),
        row("  4歳以上だけ: 100km ごとの複勝率の変化", eo),
        row("  同 勝率の年齢の違い", T["age_int_win"]),
        row("滞在 − 輸送 の複勝率（300km 以上・オッズ、コース、距離帯をそろえる）", es3, "stay_top3_holm"),
        row("  同 勝率", esw),
        row("  滞在の差は年齢で変わるか（2〜3歳 − 4歳以上）", T["stay_age_int"]),
    ]
    stay_v = sorted(C["stay_by_venue"].items(), key=lambda kv: -kv[1][0])
    stay_txt = "、".join(f"{v} {pct(r)}" for v, (r, n) in stay_v if n >= 500)

    raw_txt = (f"まず生の成績です。関東馬（美浦所属）は、地元（〜100km、東京・中山）の複勝率 {pct(BM[near]['top3'])} に対して、"
               f"関西の 3 場（300〜500km）へ輸送すると {pct(BM[mid]['top3'])} まで下がります。"
               f"一方、関西馬（栗東所属）は地元 {pct(BK[near]['top3'])}、関東の 2 場などへの遠征（300〜500km）でも {pct(BK[mid]['top3'])} と、ほとんど変わりません。"
               "ただし遠征する馬の顔ぶれ（人気・クラス）は地元と違うので、生の率だけでは輸送の影響と馬の強さを区別できません。")
    odds_txt = (f"そこで、単勝オッズ（人気）とコースをそろえて比べました。同じ場・同じオッズの馬どうしで、関西の 3 場では関東馬が関西馬より"
                f"複勝率 {pt(vk3[0])}（{pfmt(vk3[2])}）・勝率 {pt(vkw[0])}（{pfmt(vkw[2])}）、"
                f"関東の 2 場では関西馬が関東馬より複勝率 {pt(vt3[0])}（{pfmt(vt3[2])}）・勝率 {pt(vtw[0])}（{pfmt(vtw[2])}）でした。"
                + ("遠征した側が人気ほど走れていないのは関東馬の関西遠征のほうで、関西馬の関東遠征では見えません。" if mi_bad and not ku_bad else
                   "どちらの向きの遠征でも、遠征した側が人気ほど走れていません。" if mi_bad and ku_bad else
                   "どちらの向きでも、はっきりした差は見られません。" if not mi_bad and not ku_bad else
                   "遠征した側が人気ほど走れていないのは関西馬の関東遠征のほうです。")
                + f"\n\n距離を連続的に見ると、輸送した馬の複勝率は厩舎から 100km 遠くなるごとに {pt(e3, 2)}（{pfmt(p3)}）で、"
                f"300〜500km の区分では地元（〜100km）より {pt(kb3[0])}（{pfmt(kb3[2])}）です。勝率では 100km あたり {pt(ew, 2)}（{pfmt(pw)}）で有意ではありません。"
                f"オッズの代わりに前走の人気・着順でそろえると複勝率は 100km あたり {pt(ep[0], 2)}（{pfmt(ep[2])}）と大きくなるので、"
                "遠征による割引の多くは、市場がすでにオッズに織り込んでいると考えられます。")
    age_txt = (f"2〜3歳と4歳以上に分けると、100km あたりの複勝率の変化は 2〜3歳 {pt(ey[0], 2)}（{pfmt(ey[2])}）、4歳以上 {pt(eo[0], 2)}（{pfmt(eo[2])}）で、"
               f"違いは {pt(ei, 2)}（{pfmt(pi)}、多重比較の補正後 {pfmt(T['age_int_holm'])}）です。"
               + ("若い馬ほど遠征の影響が大きい、という見方を支持します。" if age_sig and ei < 0 else
                  "どちらの年齢でも遠いほどわずかに下がりますが、「若い馬ほど遠征に弱い」と言えるほどの違いはありません。" if not age_sig else
                  "むしろ 4 歳以上のほうが遠征の影響が大きく出ています。"))
    stay_txt2 = (f"500km 以上の遠征では、滞在した馬（{FP['滞在'][2]:,} 走）の複勝率は {pct(FP['滞在'][1])} で、輸送した馬（{FP['輸送'][2]:,} 走）の {pct(FP['輸送'][1])} を大きく上回ります。"
                 f"ただし平均人気も滞在 {FP['滞在'][0]:.1f} 番・輸送 {FP['輸送'][0]:.1f} 番と違い、北海道へ当週輸送する馬は少数で人気薄が中心です。"
                 f"300km 以上の遠征でオッズ・コース・距離帯をそろえると、滞在 − 輸送 は複勝率 {pt(es3[0], 2)}（{pfmt(es3[2])}）・勝率 {pt(esw[0], 2)}（{pfmt(esw[2])}）で、"
                 + ("滞在したほうが人気以上に走っています。" if stay_sig and es3[0] > 0 else
                    "滞在したほうが人気ほど走れていません。" if stay_sig else "人気から見込まれる以上の差はありません。")
                 + f"年齢で滞在の差が変わるかも {pfmt(T['stay_age_int'][2])} で、はっきりした違いは見られません。")
    verdict = "partial" if km_sig or mi_bad or ku_bad else "not_supported"

    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "関西の場での関東馬（同じオッズの関西馬との差）", "value": pt(vk3[0]), "sub": f"複勝率（{pfmt(vk3[2])}）"},
            {"label": "100km 遠くなるごとの複勝率", "value": pt(e3, 2), "sub": f"輸送した馬・オッズとコースをそろえる（{pfmt(p3)}）"},
            {"label": "年齢による違い", "value": pt(ei, 2), "sub": f"100km あたり、2〜3歳 − 4歳以上（{pfmt(pi)}）"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": f"{raw_txt}\n\n{odds_txt}\n\n{age_txt}\n\n{stay_txt2}"},
        a["c_top3"],
        a["c_win"],
        {"type": "heading", "text": "人気（オッズ）から見込まれる勝率と比べる"},
        {"type": "text", "text": (
            "遠征する馬は人気の傾向が違うため、勝率の高い・低いだけでは輸送の影響と馬の強さを区別できません。"
            "下のグラフは、輸送した馬・滞在した馬それぞれの実際の勝率と、単勝オッズから見込まれる勝率を距離帯ごとに並べたものです。"
            "2 本の線の差が、人気以上（以下）に走ったぶんです。所属・年齢でも切り替えられます。\n\n"
            f"関東馬は地元（〜100km）では実際と見込みがほぼ同じ（{pt(BM[near]['ex'], 2)}）ですが、関西の 3 場などの 300〜500km では {pt(BM[mid]['ex'], 2)} と、見込みを下回ります。"
            f"関西馬は地元 {pt(BK[near]['ex'], 2)}、300〜500km {pt(BK[mid]['ex'], 2)} で、遠征でも見込みどおりかやや上です。")},
        a["c_market"],
        {"type": "table", "columns": ["所属", "距離帯", "出走数", "複勝率", "勝率", "見込みの勝率", "実際 − 見込み（95%区間）", "平均人気"], "rows": band_tbl,
         "caption": "輸送した馬だけ。見込みの勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。関西馬は 100〜300km・500〜800km に当たる競馬場がない。"},
        {"type": "heading", "text": "2〜3歳と4歳以上で違うか"},
        {"type": "text", "text": age_txt + "\n\nグラフは輸送した馬の複勝率です。2〜3歳と4歳以上では元の率が違うため（クラス・頭数の違い）、線の高さではなく、距離が延びたときの下がり方を比べてください。"},
        a["c_age"],
        {"type": "heading", "text": "検定のまとめ"},
        {"type": "table", "columns": ["調べたこと", "差・変化", "95%区間の幅", "p 値", "補正後の p 値"], "rows": test_rows,
         "caption": "線形確率モデル（同じレースの出走は独立でないので、レース単位のクラスタ頑健 SE）。そろえた条件は、単勝オッズの対数とその 2 乗・コース（場×芝ダ×距離）・所属。"
                    "主な 6 つの検定（補正後の p 値のある行）は Holm 法で補正した。"},
        {"type": "table", "columns": ["距離帯（〜100km が基準）", "複勝率の差", "95%区間の幅", "p 値", "勝率の差", "勝率の p 値"], "rows": kb_rows,
         "caption": f"輸送した馬で、距離帯ごとの差を 〜100km と比べたもの（オッズ・コース・所属をそろえる）。距離帯をまとめた検定は 複勝率 {pfmt(T['kb_top3'])}、勝率 {pfmt(T['kb_win'])}。"
                    "500km 以上は当週に輸送した馬が少なく、区間が広い。"},
        {"type": "accordion", "title": "データと方法・注意点", "blocks": [
            {"type": "text", "text": (
                f"・対象: {min(YEARS)}〜{max(YEARS)}年の JRA 平地。取消・除外は除きます。所属の分かる {C['n_all']:,} 走のうち、輸送 {n_t:,} 走・滞在 {n_s:,} 走・判定不能 {n_unk:,} 走（レース前 28 日に場所の分かる追い切りの記録がない）。"
                f"判定できた割合は年によって違います（{'、'.join(f'{y}年 {pct(v, 0)}' for y, v in C['years'].items())}）。\n\n"
                "・所属（美浦/栗東）: 出走データに所属の列がないため、調教師ごとに、その調教師の馬の追い切りが美浦・栗東のどちらで多いかで決めました（どの調教師も、ほぼ一方の施設だけで追い切っています）。転厩は考えていません。\n\n"
                "・距離: 所属トレセン（美浦: 茨城県美浦村、栗東: 滋賀県栗東市）から各競馬場までの、緯度経度から求めた直線距離（大圏距離）です。道のり・輸送時間ではありません。北海道へは海を渡るため、実際の移動の負担は直線距離以上に大きくなります。\n\n"
                "・滞在と輸送: レース前 1〜28 日の追い切りのうち、場所の分かる最後の 1 本で決めました。所属トレセンなら輸送、競馬場（函館・札幌・小倉など）やもう一方のトレセンなら滞在です。"
                f"300km 以上の遠征で滞在と判定された割合は、{stay_txt}（500 走以上の場）。新潟・福島などは現地で追い切ることがほとんどなく、ほぼすべて輸送と判定されます。"
                "外厩（トレセン外の育成牧場）での調整は追い切りの記録に出ないため、区別できません。\n\n"
                f"・追い切りが無い場合の代わりの目安として「前走と同じ地域の遠方（{STAY_ROT_KM}km 以上）の場で、間隔 {STAY_ROT_DAYS} 日以内」を滞在とみなす方法も試しました。"
                f"追い切りで判定できる出走で比べると、この目安で滞在とした {C['rot_n']:,} 走のうち追い切りでも滞在だったのは {pct(C['rot_precision'])}、"
                f"追い切りで滞在の出走のうちこの目安でも滞在になるのは {pct(C['rot_recall'])} でした。精度が十分でないため、集計には使っていません。\n\n"
                "・距離帯と競馬場は重なっています（100〜300km は美浦から福島・新潟だけ、500〜800km は美浦から函館・札幌だけ、800km〜は美浦から小倉と栗東から北海道）。"
                "距離の効果と、その場の特徴（夏の開催・滞在の多さなど）は完全には分けられません。下の表に各場の距離と区分を示します。\n\n"
                "・オッズには「遠征だから割り引く」という市場の見込みも入っています。オッズをそろえた差は「市場の見込みよりさらに走れないか」を測ったものです。"
                "そのため、オッズの代わりに前走の人気・着順でそろえた結果も示しました（ただし前走から今回までの成長・状態の変化は入りません）。\n\n"
                "・関西の 3 場での「関東馬 − 関西馬」には、遠征の影響と、関東馬・関西馬の地力の差（オッズで測りきれない分）が混ざります。"
                "地元の場では関東馬・関西馬とも実際と見込みがほぼ同じなので、地力の差の影響は小さいと考えています。")},
            {"type": "table", "columns": ["競馬場", "美浦から", "区分", "栗東から", "区分"], "rows": venue_table(),
             "caption": "所属トレセンから各競馬場までの直線距離（大圏距離）。"},
        ]},
    ]
    summary = (f"厩舎（美浦・栗東）から競馬場までの直線距離で分けると、関東馬は関西の場で同じオッズの関西馬より複勝率が {pt(vk3[0])}（{pfmt(vk3[2])}）低く、"
               f"関西馬の関東遠征では差がはっきりしません（{pt(vt3[0])}）。輸送した馬全体では 100km あたり {pt(e3, 2)} と小さく、"
               f"2〜3歳と4歳以上の違い（{pfmt(pi)}）、滞在と輸送の違い（{pfmt(es3[2])}）も、オッズをそろえると有意ではありません。")
    if mi_bad and not ku_bad:
        title = "輸送は成績を落とす？——関東馬の関西遠征だけ人気ほど走れない。年齢・滞在では差なし"
    elif verdict == "partial":
        title = "輸送は成績を落とす？——遠いほどわずかに人気以下。年齢・滞在では差なし"
    else:
        title = "輸送は成績を落とす？——厩舎からの距離でも、滞在か輸送かでも、人気以上の差はない"
    return {
        "slug": "transport-vs-stay",
        "title": title,
        "claim": "輸送は成績を落とす",
        "verdict": verdict,
        "summary": summary,
        "categories": ["travel/transport"],
        "published": "2026-10-28",
        "updated": "2026-11-02",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年",
                    "conditions": "JRA全競馬場・芝ダート（障害除く）。所属は調教師の追い切り場所、滞在/輸送は直前の追い切り場所で判定",
                    "sample": f"輸送 {n_t:,} 走・滞在 {n_s:,} 走"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not LAPS.exists():
        raise SystemExit(f"{LAPS} がありません")
    a = analyze(frame())
    for k, v in a["tests"].items():
        print(k, v)
    print(a["cov"])
    col = article(a)
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
