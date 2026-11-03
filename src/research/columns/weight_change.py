"""めぐコラム『馬体重の大幅変化』を実データから集計し、frontend/content/columns/weight-change.json を書く。

問い: 前走比の馬体重の増減で成績は変わるか。特に「大きく増えた馬」は、増えた理由（太め・ベスト体重へ戻り・成長）で
      成績が違うか。年齢・休養明けかどうかで違うか。追い切りの評価で「成長」の裏付けが取れるか。

データ:
  レース:   data/page_reference/tables/<年>/race_result_flat.parquet（compare_series.load と同じ。2020〜2025 年の JRA 平地、取消・除外を除く）
  馬の全戦績: data/local/modeling/training_pipeline_v4/tp32/career_eq.parquet（smartrc の馬ごとの全戦績。デビューからの馬体重・着順・日付。
            2020 年より前・地方の出走も含むので、ベスト体重や過去最高体重をデビューからさかのぼって出せる）
  追い切り: data/page_reference/tables/<年>/race_oikiri_flat.parquet（各レースの最終追い切りの評価 A〜D。レース前に公表されるもの）
            data/local/modeling/training_pipeline/features/training_laps.parquet（馬ごとの追い切りの時計。horse_training を表にしたもの。
            一部の馬だけ。レース前 14 日以内の坂路・ウッドの 4F 時計を、同じコース・同じ年の全追い切りで標準化し、最も速いものを取る。
            「自己比」= その値 − 同じ馬の過去のレース前の値の平均。負なら自分の過去より速い）

定義（すべて今回より前の出走だけで決める。先読みなし）:
  増減 = 今回の馬体重 − 直前の出走（地方を含む）の馬体重。変わらず = ±2kg 以内。大幅増 = +10kg 以上（既存の区分「増 +6〜+10」「激増 +11 超」の境目）。
  ベスト体重 = 過去の JRA の出走で 3 着以内に入ったときの馬体重の中央値（3 着以内が 1 回もない馬は「なし」）。
  休養明け = 前走から 70 日（10 週）以上。長期休養（けが明けの近似）= 182 日（約半年）以上。故障・休養の理由はデータに無い。
  大幅増の 3 分類（上から順に判定）:
    (2) ベスト体重へ戻り: ベスト体重があり、今回がその ±4kg 以内で、前走はベストより 4kg を超えて軽かった。
    (3) 成長: 休養明けか 2〜3 歳で、今回が過去最高の体重を上回った。
    (1) 太め: 上のどちらでもなく、過去最高の体重を上回ったか、ベスト体重を 8kg 以上上回った。
    どれにも当てはまらない大幅増（ベスト体重が無く過去最高も超えない、など）は「その他の増加」。
  オッズから見込まれる勝率 = 1/単勝オッズ をレース内で合計 1 に直したもの。見込まれる複勝率 = 同じ見込み勝率の帯（全出走で 40 分位）の実際の複勝率。

使い方:
    python -m src.research.columns.weight_change            # 集計して記事を書く
    python -m src.research.columns.weight_change --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import FLAT_DIR, YEARS, _p, _r, _rci, _stat, load
from src.research.columns.wind_venue import holm, mean_ci, ols, pct, pfmt, wilson

REPO = Path(__file__).resolve().parents[3]
CAREER = REPO / "data/local/modeling/training_pipeline_v4/tp32/career_eq.parquet"
LAPS = REPO / "data/local/modeling/training_pipeline/features/training_laps.parquet"
OUT = REPO / "frontend/content/columns/weight-change.json"

GAIN_MIN, TOL, REST_DAYS, FAT_MARGIN = 10, 4, 70, 8
LONG_REST = 182
SAME = "変わらず（±2kg）"
BACK, GROW, FAT, OTHER = "ベスト体重へ戻り", "成長", "太め", "その他の増加"
CLASSES = [BACK, GROW, FAT, OTHER]
CHANGE_BANDS = ([-999, -10, -5, -3, 2, 5, 9, 999], ["大幅減\n-10以下", "減\n-9〜-5", "微減\n-4〜-3", "変わらず\n±2", "微増\n+3〜+5", "増\n+6〜+9", "大幅増\n+10以上"])
AGES = ["2歳", "3歳", "4歳", "5歳", "6歳以上"]
RESTS = ["10週未満", "10週〜半年", "半年以上"]


# ── データ ──────────────────────────────────────────


def history(c: pd.DataFrame) -> pd.DataFrame:
    """全戦績（horse_id, race_id, date, weight, jra_fin）から、各出走より前の馬体重の履歴を出す。

    prev_w: 直前の出走の馬体重（無ければその前）、prev_date: 直前の出走日、max_prev: 過去最高の体重、
    best: 過去の JRA の 3 着以内の馬体重の中央値、n_best: その回数。"""
    c = c.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    g = c.groupby("horse_id")
    c["prev_w"] = g["weight"].transform(lambda s: s.shift().ffill())
    c["prev_date"] = g["date"].shift()
    c["max_prev"] = g["weight"].transform(lambda s: s.shift().cummax())
    best = np.full(len(c), np.nan)
    nb = np.zeros(len(c), dtype=int)
    hid, w, fin = c["horse_id"].to_numpy(), c["weight"].to_numpy(float), c["jra_fin"].to_numpy(float)
    cur, good = None, []
    for i in range(len(c)):
        if hid[i] != cur:
            cur, good = hid[i], []
        if good:
            best[i], nb[i] = float(np.median(good)), len(good)
        if 1 <= fin[i] <= 3 and np.isfinite(w[i]):
            good.append(w[i])
    c["best"], c["n_best"] = best, nb
    return c


def load_career() -> pd.DataFrame:
    c = pd.read_parquet(CAREER, columns=["horse_id", "race_id", "year", "monthday", "weight", "arrival", "jyocd"])
    c["horse_id"] = c["horse_id"].astype(str)
    c = c.drop_duplicates(["race_id", "horse_id"])
    c["date"] = pd.to_datetime(c["year"].astype(str) + c["monthday"].astype(str), format="%Y%m%d", errors="coerce")
    w = pd.to_numeric(c["weight"], errors="coerce")
    c["weight"] = w.where(w.between(300, 700))
    jra = pd.to_numeric(c["jyocd"], errors="coerce") <= 10
    c["jra_fin"] = pd.to_numeric(c["arrival"], errors="coerce").where(jra)
    return history(c[c["date"].notna()][["horse_id", "race_id", "date", "weight", "jra_fin"]])


def classify(d: pd.DataFrame, gain_min: float = GAIN_MIN, tol: float = TOL, rest_days: int = REST_DAYS,
             fat_margin: float = FAT_MARGIN) -> pd.Series:
    """出走ごとの区分（変わらず / 大幅増の 4 区分 / それ以外は NaN）。列: w, gain, best, max_prev, prev_w, rest, age。"""
    big = d["gain"] >= gain_min
    back = big & d["best"].notna() & ((d["w"] - d["best"]).abs() <= tol) & (d["prev_w"] < d["best"] - tol)
    grow = big & ~back & ((d["rest"] >= rest_days) | (d["age"] <= 3)) & (d["w"] > d["max_prev"])
    fat = big & ~back & ~grow & ((d["w"] > d["max_prev"]) | (d["w"] >= d["best"] + fat_margin))
    out = pd.Series(np.nan, index=d.index, dtype=object)
    out[d["gain"].abs() <= 2] = SAME
    out[big] = OTHER
    out[fat], out[grow], out[back] = FAT, GROW, BACK
    return out


def oikiri() -> pd.DataFrame:
    fs = [FLAT_DIR / str(y) / "race_oikiri_flat.parquet" for y in YEARS]
    o = pd.concat([pd.read_parquet(f, columns=["race_id", "horse_id", "evaluation"]) for f in fs if f.exists()], ignore_index=True)
    o["horse_id"] = o["horse_id"].astype(str)
    o = o[o["evaluation"].isin(["A", "B", "C", "D"])].drop_duplicates(["race_id", "horse_id"])
    o["work"] = np.where(o["evaluation"].isin(["A", "B"]), "追い切り A・B", "追い切り C・D")
    return o[["race_id", "horse_id", "evaluation", "work"]]


def work_time(t: pd.DataFrame) -> pd.DataFrame:
    """追い切りの時計（列: horse_id, race_date, course, kind, t4, days_before）から、出走ごとの時計の指標を作る。

    wz: レース前 14 日以内の坂路・ウッドの 4F 時計を、同じコース・同じ年で標準化した値の最小（速いほど小さい）。
    dz: wz − 同じ馬の過去のレース前の wz の平均（今回より前だけ）。"""
    t = t[t["kind"].isin(["slope", "wood"]) & t["t4"].notna() & t["days_before"].between(0, 14) & (t["race_date"] > "2000")].copy()
    t["yr"] = t["race_date"].str[:4]
    g = t.groupby(["course", "yr"])["t4"]
    t = t[g.transform("size") >= 200].copy()
    g = t.groupby(["course", "yr"])["t4"]
    t["z"] = (t["t4"] - g.transform("mean")) / g.transform("std")
    b = t.groupby(["horse_id", "race_date"])["z"].min().rename("wz").reset_index()
    b["horse_id"] = b["horse_id"].astype(str)
    b = b.sort_values(["horse_id", "race_date"])
    b["self"] = b.groupby("horse_id")["wz"].transform(lambda s: s.shift().expanding().mean())
    b["dz"] = b["wz"] - b["self"]
    return b[["horse_id", "race_date", "wz", "dz"]]


def frame() -> pd.DataFrame:
    d = load()
    h = load_career()
    d = d.merge(h[["race_id", "horse_id", "weight", "prev_w", "prev_date", "max_prev", "best", "n_best"]], on=["race_id", "horse_id"], how="left")
    d = d.rename(columns={"weight": "w"})
    d["gain"] = d["w"] - d["prev_w"]
    d["rest"] = (pd.to_datetime(d["date"]) - d["prev_date"]).dt.days
    d = d[d["gain"].notna() & d["rest"].notna()].copy()
    d["cls"] = classify(d)
    d["band"] = pd.cut(d["gain"], CHANGE_BANDS[0], labels=CHANGE_BANDS[1]).astype(str)
    d["age_band"] = np.where(d["age"] >= 6, "6歳以上", d["age"].astype("Int64").astype(str) + "歳")
    d["rest_band"] = np.select([d["rest"] < REST_DAYS, d["rest"] < LONG_REST], RESTS[:2], RESTS[2])
    # オッズから見込まれる複勝率: 見込み勝率の 40 分位ごとの実際の複勝率
    q = pd.qcut(d["implied"].rank(method="first"), 40, labels=False)
    d["implied_top3"] = d.groupby(q)["top3"].transform("mean")
    d["ex_win"] = d["win"] - d["implied"]
    d["ex_top3"] = d["top3"] - d["implied_top3"]
    d["lq"] = np.log(d["odds"])
    d["key"] = d["venue"] + d["surface"] + d["distance"].astype(str)
    d = d.merge(oikiri(), on=["race_id", "horse_id"], how="left")
    d["race_date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    if LAPS.exists():
        laps = pd.read_parquet(LAPS, columns=["horse_id", "race_date", "course", "kind", "t4", "days_before"])
        d = d.merge(work_time(laps), on=["horse_id", "race_date"], how="left")
    else:
        d["wz"], d["dz"] = np.nan, np.nan
    d["selfw"] = np.select([d["dz"] < 0, d["dz"] >= 0], ["自己比 速い", "自己比 遅い"], None)
    return d


# ── 集計 ──────────────────────────────────────────


def summary(x: pd.DataFrame) -> dict[str, Any]:
    return {"n": len(x), "win": x["win"].mean(), "top3": x["top3"].mean(), "implied": x["implied"].mean(),
            "implied_top3": x["implied_top3"].mean(), "ex_win": x["ex_win"].mean(), "ex_win_ci": mean_ci(x["ex_win"]),
            "ex_top3": x["ex_top3"].mean(), "ex_top3_ci": mean_ci(x["ex_top3"]),
            "win_ci": wilson(x["win"].sum(), len(x)), "top3_ci": wilson(x["top3"].sum(), len(x))}


def series(d: pd.DataFrame, gcol: str, groups: list[str], xcol: str, xs: list[str], metric: str) -> list[dict[str, Any]]:
    out = []
    for g in groups:
        vals, ns, cis = [], [], []
        for x in xs:
            m, n, ci, _ = _stat(d.loc[(d[gcol] == g) & (d[xcol] == x), metric])
            vals.append(_r(m) if n >= 30 else None); ns.append(n); cis.append(_rci(ci) if n >= 30 else None)
        out.append({"name": g, "values": vals, "n": ns, "ci": cis})
    return out


def adj(d: pd.DataFrame, g: str, metric: str) -> tuple[float, float, float]:
    """区分 g と「変わらず」の差を、オッズ（とその 2 乗）・コースをそろえて測る。"""
    z = d[d["cls"].isin([g, SAME])].assign(t=lambda w: (w["cls"] == g).astype(float))
    return ols(f"{metric} ~ t + lq + I(lq**2) + C(key)", z, "t")


def robustness(d: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for gain_min, tol, rest_days, label in [(10, 4, 70, "基本（+10kg・±4kg・10週）"), (8, 4, 70, "大幅増を +8kg 以上"), (12, 4, 70, "大幅増を +12kg 以上"),
                                            (10, 2, 70, "ベストの幅 ±2kg"), (10, 6, 70, "ベストの幅 ±6kg"),
                                            (10, 4, 56, "休養明けを 8 週以上"), (10, 4, 84, "休養明けを 12 週以上")]:
        cl = classify(d, gain_min, tol, rest_days)
        r = {"label": label}
        for g in (BACK, GROW, FAT):
            x = d[cl == g]
            r[g] = {"n": len(x), "top3": x["top3"].mean(), "ex_top3": x["ex_top3"].mean(), "ex_win": x["ex_win"].mean(), "ci": mean_ci(x["ex_top3"])}
        rows.append(r)
    return rows


def analyze(d: pd.DataFrame) -> dict[str, Any]:
    a: dict[str, Any] = {}
    a["n_all"] = len(d)
    a["bands"] = {b: summary(d[d["band"] == b]) for b in CHANGE_BANDS[1]}
    groups = [SAME] + CLASSES
    a["cls"] = {g: summary(d[d["cls"] == g]) for g in groups}
    big = d[d["gain"] >= GAIN_MIN]
    a["n_big"] = len(big)
    a["share"] = {g: (big["cls"] == g).mean() for g in CLASSES}
    a["adj"] = {g: {"top3": adj(d, g, "top3"), "win": adj(d, g, "win")} for g in CLASSES}
    ps = [a["adj"][g]["top3"][2] for g in CLASSES]
    for g, ph in zip(CLASSES, holm(ps)):
        a["adj"][g]["top3_holm"] = ph
    for g, ph in zip(CLASSES, holm([a["adj"][g]["win"][2] for g in CLASSES])):
        a["adj"][g]["win_holm"] = ph
    # 区分の中身（年齢・休養）
    a["profile"] = {g: {"age": d.loc[d["cls"] == g, "age"].mean(), "rest": d.loc[d["cls"] == g, "rest"].median(),
                        "gain": d.loc[d["cls"] == g, "gain"].mean(), "young": (d.loc[d["cls"] == g, "age"] <= 3).mean(),
                        "long": (d.loc[d["cls"] == g, "rest"] >= LONG_REST).mean()} for g in groups}
    # 年齢別・休養別（区分ごとの、オッズの見込みとの差）
    a["by_age"] = series(d, "cls", groups, "age_band", AGES, "ex_top3")
    a["by_rest"] = series(d, "cls", groups, "rest_band", RESTS, "ex_top3")
    a["by_age_top3"] = series(d, "cls", groups, "age_band", AGES, "top3")
    a["by_rest_top3"] = series(d, "cls", groups, "rest_band", RESTS, "top3")
    a["long_rest"] = {g: summary(d[(d["cls"] == g) & (d["rest"] >= LONG_REST)]) for g in groups}
    a["robust"] = robustness(d)
    # 追い切り（評価 A・B / C・D と、時計の自己比 速い / 遅い）
    w = d[d["cls"].isin(groups)]
    a["work_rate"] = w["work"].notna().mean()
    a["time_rate"] = w["selfw"].notna().mean()
    a["work"] = work_compare(w, groups, "work", "追い切り A・B", "追い切り C・D")
    a["time"] = work_compare(w, groups, "selfw", "自己比 速い", "自己比 遅い")
    a["c_work"] = work_chart(groups, a["work"], "追い切り A・B", "追い切り C・D", "最終追い切りの評価別 複勝率（実際と見込み）",
                             "評価は A〜D（レース前に公表）。実線が実際、点線がオッズから見込まれる複勝率。実線と点線がほぼ重なる＝評価は人気に織り込まれている。")
    a["c_time"] = work_chart(groups, a["time"], "自己比 速い", "自己比 遅い", "追い切りの時計（自分の過去と比べて）別 複勝率（実際と見込み）",
                             "レース前 14 日以内の坂路・ウッドの 4F 時計が、その馬の過去のレース前より速いか遅いか。時計の記録がある馬だけ。点線はオッズの見込み。")
    return a


def work_compare(w: pd.DataFrame, groups: list[str], col: str, ka: str, kb: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in groups:
        xa, xb = w[(w["cls"] == g) & (w[col] == ka)], w[(w["cls"] == g) & (w[col] == kb)]
        r = {ka: summary(xa), kb: summary(xb), "share_a": len(xa) / max(1, len(xa) + len(xb))}
        p = _p(xa["ex_top3"].var() / len(xa), xb["ex_top3"].var() / len(xb), xa["ex_top3"].mean(), xb["ex_top3"].mean()) if min(len(xa), len(xb)) > 1 else None
        r["p_ex_top3"] = p
        p2 = _p(xa["top3"].var() / len(xa), xb["top3"].var() / len(xb), xa["top3"].mean(), xb["top3"].mean()) if min(len(xa), len(xb)) > 1 else None
        r["p_top3"] = p2
        if g != SAME:
            # 「変わらず」と比べた、よい追い切りの上乗せ（区分 × 追い切りの交互作用。オッズ・コース調整）
            z = w[w["cls"].isin([g, SAME]) & w[col].isin([ka, kb])].assign(t=lambda x: (x["cls"] == g).astype(float),
                                                                          ab=lambda x: (x[col] == ka).astype(float))
            r["inter"] = ols("top3 ~ t * ab + lq + I(lq**2) + C(key)", z, "t:ab")
        out[g] = r
    return out


def work_chart(groups: list[str], res: dict[str, Any], ka: str, kb: str, title: str, caption: str) -> dict[str, Any]:
    """区分ごとに、よい / よくない追い切りの実際の複勝率と、オッズから見込まれる複勝率（点線）を並べる。"""
    ser = []
    for k in (ka, kb):
        ser.append({"name": f"{k}（実際）", "values": [_r(res[g][k]["top3"]) for g in groups], "n": [res[g][k]["n"] for g in groups],
                    "ci": [res[g][k]["top3_ci"] for g in groups]})
        ser.append({"name": f"{k}（見込み）", "values": [_r(res[g][k]["implied_top3"]) for g in groups], "baseline": True})
    return {"type": "chart", "title": title, "xLabel": "区分", "yLabel": "複勝率", "xs": groups, "series": ser, "caption": caption}


# ── 記事 ──────────────────────────────────────────


def pt(v: float) -> str:
    return f"{v * 100:+.1f}pt"


def ci_pt(c: list[float] | None) -> str:
    return "–" if c is None else f"{c[0] * 100:+.1f}〜{c[1] * 100:+.1f}pt"


def article(a: dict[str, Any]) -> dict[str, Any]:
    B, C = a["bands"], a["cls"]
    s, bk, gr, ft, ot = C[SAME], C[BACK], C[GROW], C[FAT], C[OTHER]
    hi, same_b, lo = B[CHANGE_BANDS[1][-1]], B[CHANGE_BANDS[1][3]], B[CHANGE_BANDS[1][0]]
    A = a["adj"]
    P = a["profile"]
    W = a["work"]

    band_chart = {"type": "chart", "title": "前走比の馬体重の増減別 複勝率・勝率", "xLabel": "前走からの増減", "yLabel": "率",
                  "xs": CHANGE_BANDS[1],
                  "series": [{"name": "複勝率", "values": [_r(B[b]["top3"]) for b in CHANGE_BANDS[1]], "n": [B[b]["n"] for b in CHANGE_BANDS[1]],
                              "ci": [B[b]["top3_ci"] for b in CHANGE_BANDS[1]]},
                             {"name": "勝率", "values": [_r(B[b]["win"]) for b in CHANGE_BANDS[1]], "n": [B[b]["n"] for b in CHANGE_BANDS[1]],
                              "ci": [B[b]["win_ci"] for b in CHANGE_BANDS[1]]},
                             {"name": "オッズから見込まれる複勝率", "values": [_r(B[b]["implied_top3"]) for b in CHANGE_BANDS[1]], "baseline": True}],
                  "caption": "前走（地方を含む）からの増減。点線はオッズから見込まれる複勝率。縦線は 95% 信頼区間。"}
    groups = [SAME] + CLASSES
    cls_chart = {"type": "chart", "title": "大幅増（+10kg 以上）の区分別 複勝率・勝率", "xLabel": "区分", "yLabel": "率", "xs": groups,
                 "series": [{"name": "複勝率", "values": [_r(C[g]["top3"]) for g in groups], "n": [C[g]["n"] for g in groups], "ci": [C[g]["top3_ci"] for g in groups]},
                            {"name": "オッズから見込まれる複勝率", "values": [_r(C[g]["implied_top3"]) for g in groups], "baseline": True},
                            {"name": "勝率", "values": [_r(C[g]["win"]) for g in groups], "n": [C[g]["n"] for g in groups], "ci": [C[g]["win_ci"] for g in groups]},
                            {"name": "オッズから見込まれる勝率", "values": [_r(C[g]["implied"]) for g in groups], "baseline": True}],
                 "caption": "実線が実際、点線がオッズから見込まれる率。変わらず（±2kg）は比べるための基準。"}
    age_chart = {"type": "chart", "title": "年齢別 区分ごとの複勝率", "xLabel": "年齢", "yLabel": "複勝率", "xs": AGES, "series": a["by_age_top3"],
                 "caption": "30 走未満の点は出さない。縦線は 95% 信頼区間。人気の差を除いた「見込みとの差」は下の表。"}
    rest_chart = {"type": "chart", "title": "前走からの間隔別 区分ごとの複勝率", "xLabel": "前走からの間隔", "yLabel": "複勝率", "xs": RESTS, "series": a["by_rest_top3"],
                  "caption": "半年以上の休養は「けが明け」の近似（故障の記録は無い）。30 走未満の点は出さない。"}

    def ex_table(sers: list[dict[str, Any]], xs: list[str], head: str, cap: str) -> dict[str, Any]:
        rows = [[sr["name"]] + [("–" if v is None else f"{pt(v)}（{n:,}）") for v, n in zip(sr["values"], sr["n"])] for sr in sers]
        return {"type": "table", "columns": [head] + xs, "rows": rows, "caption": cap}

    age_table = ex_table(a["by_age"], AGES, "区分 \\ 年齢", "複勝率 − オッズから見込まれる複勝率（かっこ内は出走数）。30 走未満は –。")
    rest_table = ex_table(a["by_rest"], RESTS, "区分 \\ 間隔", "複勝率 − オッズから見込まれる複勝率（かっこ内は出走数）。30 走未満は –。")
    cls_rows = [[g, f"{C[g]['n']:,}", pct(C[g]["top3"]), pct(C[g]["implied_top3"]), f"{pt(C[g]['ex_top3'])}（{ci_pt(C[g]['ex_top3_ci'])}）",
                 pct(C[g]["win"]), pct(C[g]["implied"]), f"{P[g]['age']:.1f}歳", f"{P[g]['rest']:.0f}日"] for g in groups]
    adj_rows = [[g, f"{C[g]['n']:,}", f"{A[g]['top3'][0] * 100:+.2f}pt", pfmt(A[g]["top3"][2]), pfmt(A[g]["top3_holm"]),
                 f"{A[g]['win'][0] * 100:+.2f}pt", pfmt(A[g]["win"][2])] for g in CLASSES]
    rob_rows = [[r["label"]] + [f"{r[g]['n']:,} / {pt(r[g]['ex_top3'])}" for g in (BACK, GROW, FAT)] for r in a["robust"]]
    W, T = a["work"], a["time"]
    KA, KB, TA, TB = "追い切り A・B", "追い切り C・D", "自己比 速い", "自己比 遅い"

    def work_rows(res: dict[str, Any], ka: str, kb: str) -> list[list[str]]:
        return [[g, f"{res[g][ka]['n']:,}", pct(res[g][ka]["top3"]), pt(res[g][ka]["ex_top3"]), f"{res[g][kb]['n']:,}", pct(res[g][kb]["top3"]),
                 pt(res[g][kb]["ex_top3"]), "–" if res[g]["p_ex_top3"] is None else pfmt(res[g]["p_ex_top3"]),
                 "–" if "inter" not in res[g] else f"{res[g]['inter'][0] * 100:+.1f}pt（{pfmt(res[g]['inter'][2])}）"] for g in groups]
    LR = a["long_rest"]
    age = {sr["name"]: dict(zip(AGES, sr["values"])) for sr in a["by_age"]}
    rest = {sr["name"]: dict(zip(RESTS, sr["values"])) for sr in a["by_rest"]}
    inter_sig = [g for g in (BACK, GROW) for res in (W, T) if res[g]["inter"][2] < 0.05]

    # 文中の判断（数値から）
    def sig_dir(g: str) -> str:
        e, _, p = A[g]["top3"]
        if A[g]["top3_holm"] >= 0.05:
            return "はっきりした差がなく"
        return ("わずかに" if abs(e) < 0.02 else "") + ("高く" if e > 0 else "低く")

    # 表示で ±0.0pt になる差（|差| < 0.05pt）は向きを持たないとみなす
    robust_sign = {g: sorted({np.sign(round(r[g]["ex_top3"], 3)) for r in a["robust"]} - {0.0}) for g in (BACK, GROW, FAT)}
    stable = all(len(v) == 1 for v in robust_sign.values())

    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "大幅増（+10kg以上）の複勝率", "value": pct(hi["top3"]), "sub": f"見込み {pct(hi['implied_top3'])}。変わらず {pct(same_b['top3'])}"},
            {"label": "うち『太め』の見込みとの差", "value": pt(ft["ex_top3"]), "sub": f"{ft['n']:,} 走。複勝率 {pct(ft['top3'])}"},
            {"label": "うち『成長』の見込みとの差", "value": pt(gr["ex_top3"]), "sub": f"{gr['n']:,} 走。複勝率 {pct(gr['top3'])}"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"前走から大きく増えた（+10kg 以上）馬の複勝率は {pct(hi['top3'])} で、変わらず（±2kg）の {pct(same_b['top3'])} より低く、"
            f"大きく減った（−10kg 以下）馬は {pct(lo['top3'])} とさらに低めです。ただ、どちらもオッズで人気を落としていて、"
            f"見込み（大幅増 {pct(hi['implied_top3'])}・大幅減 {pct(lo['implied_top3'])}）と比べると差は小さくなります。\n\n"
            f"大幅増を、増えた理由で 3 つに分けました（各馬の過去の出走だけで判定）。"
            f"『太め』（休み明けでも若駒でもないのに過去最高を超えた、またはベスト体重を 8kg 以上超えた）は {ft['n']:,} 走で、複勝率 {pct(ft['top3'])}・見込みとの差 {pt(ft['ex_top3'])}。"
            f"『ベスト体重へ戻り』（好走時の体重の ±4kg に戻った）は {bk['n']:,} 走で {pct(bk['top3'])}・{pt(bk['ex_top3'])}。"
            f"『成長』（休み明けか 2〜3 歳で過去最高を更新）は {gr['n']:,} 走で {pct(gr['top3'])}・{pt(gr['ex_top3'])} です。\n\n"
            f"オッズとコースをそろえて「変わらず」と比べると、複勝率は太めは{sig_dir(FAT)}（{A[FAT]['top3'][0] * 100:+.1f}pt、補正後 {pfmt(A[FAT]['top3_holm'])}）、"
            f"成長は{sig_dir(GROW)}（{A[GROW]['top3'][0] * 100:+.1f}pt、補正後 {pfmt(A[GROW]['top3_holm'])}）、"
            f"ベスト体重へ戻りは{sig_dir(BACK)}（{A[BACK]['top3'][0] * 100:+.1f}pt、補正後 {pfmt(A[BACK]['top3_holm'])}）、という結果です。"
            f"勝率で見ると、成長（{A[GROW]['win'][0] * 100:+.1f}pt、{pfmt(A[GROW]['win'][2])}）とベスト体重へ戻り（{A[BACK]['win'][0] * 100:+.1f}pt、{pfmt(A[BACK]['win'][2])}）はむしろ見込みをわずかに上回る傾向ですが、"
            + ("有意ではありません。" if min(A[GROW]["win"][2], A[BACK]["win"][2]) >= 0.05
               else f"4 区分を同時に見ていることを補正すると有意ではありません（補正後 {pfmt(A[GROW]['win_holm'])}・{pfmt(A[BACK]['win_holm'])}）。"
               if max(A[GROW]["win_holm"], A[BACK]["win_holm"]) >= 0.05 else "補正後も有意です。")
            + ("閾値を変えても、3 区分の見込みとの差の向きは変わりませんでした。" if stable else "ただし閾値を変えると向きが変わる区分があり、結論は閾値に左右されます（下の表）。")
            + "\n\nつまり「増えた理由」で成績の傾向は分かれますが、差は 1〜2pt 程度で、多くは人気（オッズ）にすでに表れています。"
            "『成長』でも見込みを上回るわけではなく、ベスト体重へ戻った馬だけが変わらずの馬と同じくらい走る、というのが実際のところです。")},
        band_chart,
        {"type": "heading", "text": "大幅増を 3 つに分ける"},
        {"type": "text", "text": (
            f"大幅増 {a['n_big']:,} 走の内訳は、成長 {pct(a['share'][GROW], 0)}・太め {pct(a['share'][FAT], 0)}・ベスト体重へ戻り {pct(a['share'][BACK], 0)}・"
            f"その他 {pct(a['share'][OTHER], 0)} です。成長は平均 {P[GROW]['age']:.1f} 歳（2〜3 歳が {pct(P[GROW]['young'], 0)}）・前走から中 {P[GROW]['rest']:.0f} 日、"
            f"太めは平均 {P[FAT]['age']:.1f} 歳・中 {P[FAT]['rest']:.0f} 日と、中身がはっきり違います。")},
        cls_chart,
        {"type": "table", "columns": ["区分", "出走数", "複勝率", "見込み複勝率", "差（95%区間）", "勝率", "見込み勝率", "平均年齢", "間隔の中央値"],
         "rows": cls_rows, "caption": "見込み複勝率は、オッズから見込まれる勝率が同じくらいの馬（全出走を 40 帯に分けたもの）の実際の複勝率。"},
        {"type": "table", "columns": ["区分", "出走数", "複勝率の差", "p 値", "補正後の p 値", "勝率の差", "p 値（勝率）"], "rows": adj_rows,
         "caption": "「変わらず（±2kg）」との差。オッズ（とその 2 乗）・コース（場・芝ダ・距離）をそろえた回帰。4 区分を同時に調べているので Holm 法の補正後の p 値も示す。"},
        {"type": "heading", "text": "年齢・休養明けで違うか"},
        age_chart,
        age_table,
        {"type": "text", "text": (
            f"太めは 3〜5 歳で見込みを下回り（3 歳 {pt(age[FAT]['3歳'])}・4 歳 {pt(age[FAT]['4歳'])}・5 歳 {pt(age[FAT]['5歳'])}）、6 歳以上では {pt(age[FAT]['6歳以上'])} と見込みを下回らなくなります。"
            f"成長は 2 歳で {pt(age[GROW]['2歳'])} と見込みを上回る一方、3 歳では {pt(age[GROW]['3歳'])} です。"
            f"ただ 2 歳は体重が変わらない馬も {pt(age[SAME]['2歳'])} と見込みを上回っており（2 歳戦は人気の見込みが粗い）、2 歳の大幅増に特別な上積みがあるとは言えません。"
            "3 歳以降の大幅増は、休み明けや若さで説明がつきそうに見えても、見込みを少し下回りがちです。"
            "ただし年齢 × 区分の 25 通りを見ているので、ひとつひとつの点の差（多くは ±2pt 以内）は偶然でも出る大きさです。")},
        rest_chart,
        rest_table,
        {"type": "text", "text": (
            f"半年以上の休養明け（けが明けの近似）だけで見ると、成長 {LR[GROW]['n']:,} 走は見込みとの差 {pt(LR[GROW]['ex_top3'])}（{ci_pt(LR[GROW]['ex_top3_ci'])}）、"
            f"太め {LR[FAT]['n']:,} 走は {pt(LR[FAT]['ex_top3'])}、ベスト体重へ戻り {LR[BACK]['n']:,} 走は {pt(LR[BACK]['ex_top3'])}、"
            f"変わらず {LR[SAME]['n']:,} 走は {pt(LR[SAME]['ex_top3'])} です。"
            "長い休み明けは体重にかかわらず市場の評価どおりかそれ以下になりやすく、体重が増えたこと自体より、休み明けであることの影響のほうが目立ちます。"
            if LR[SAME]["n"] >= 30 else "")},
        {"type": "heading", "text": "閾値を変えても同じか"},
        {"type": "table", "columns": ["定義", "ベスト体重へ戻り（出走数 / 差）", "成長", "太め"], "rows": rob_rows,
         "caption": "差は、複勝率 − オッズから見込まれる複勝率。基本の定義から 1 か所ずつ変えた。"},
        {"type": "heading", "text": "追い切りで裏付けが取れるか"},
        {"type": "text", "text": (
            f"2 つの物差しで調べました。① 最終追い切りの評価（A〜D。レース前に公表。上の区分の {pct(a['work_rate'], 0)} に付いている）。"
            f"② 追い切りの時計を、その馬自身の過去と比べたもの（レース前 14 日以内の坂路・ウッドの 4F。時計の記録がある馬だけで、上の区分の {pct(a['time_rate'], 0)}）。\n\n"
            f"① 評価 A・B の馬はよく走ります。成長で複勝率 {pct(W[GROW][KA]['top3'])} 対 C・D {pct(W[GROW][KB]['top3'])}、ベスト体重へ戻りで {pct(W[BACK][KA]['top3'])} 対 {pct(W[BACK][KB]['top3'])} です。"
            f"しかしオッズの見込みとの差にすると、成長で {pt(W[GROW][KA]['ex_top3'])} 対 {pt(W[GROW][KB]['ex_top3'])}（{pfmt(W[GROW]['p_ex_top3'])}）、"
            f"ベスト体重へ戻りで {pt(W[BACK][KA]['ex_top3'])} 対 {pt(W[BACK][KB]['ex_top3'])}（{pfmt(W[BACK]['p_ex_top3'])}）と、ほとんど同じです。"
            "追い切りの評価は人気にそのまま織り込まれている、ということです。\n\n"
            f"② 自分の過去より速い時計を出した馬は、体重が変わらない馬でも見込みを少し上回ります（{pt(T[SAME][TA]['ex_top3'])} 対 {pt(T[SAME][TB]['ex_top3'])}、{pfmt(T[SAME]['p_ex_top3'])}）。"
            f"ベスト体重へ戻りでは {pt(T[BACK][TA]['ex_top3'])} 対 {pt(T[BACK][TB]['ex_top3'])}（{T[BACK][TA]['n']:,} 走と {T[BACK][TB]['n']:,} 走、{pfmt(T[BACK]['p_ex_top3'])}）、"
            f"成長では {pt(T[GROW][TA]['ex_top3'])} 対 {pt(T[GROW][TB]['ex_top3'])}（{pfmt(T[GROW]['p_ex_top3'])}）で、向きは同じですが有意ではありません。\n\n"
            + ("「変わらず」の馬と比べた上乗せ（区分 × 追い切りの交互作用。表の右端）はどれも有意ではなく、"
               "追い切りが「増えた分は成長・戻り」の裏付けとして特に効く、とはデータからは言えません。"
               "追い切りがよいことは、体重の増減にかかわらず同じように効く材料、と考えるのが安全です。" if not inter_sig else
               "「変わらず」の馬と比べた上乗せ（区分 × 追い切りの交互作用）が有意なのは " + "・".join(dict.fromkeys(inter_sig)) + " で、追い切りが裏付けとして効く場面があります。"))},
        a["c_work"],
        {"type": "table", "columns": ["区分", "A・B の出走数", "A・B の複勝率", "A・B の見込みとの差", "C・D の出走数", "C・D の複勝率", "C・D の見込みとの差", "p 値", "変わらずと比べた上乗せ"],
         "rows": work_rows(W, KA, KB), "caption": "p 値は A・B と C・D の「見込みとの差」の比較（Welch）。上乗せは区分 × 評価の交互作用（オッズ・コース調整、レース単位のクラスタ頑健 SE）。"},
        a["c_time"],
        {"type": "table", "columns": ["区分", "速い の出走数", "速い の複勝率", "速い の見込みとの差", "遅い の出走数", "遅い の複勝率", "遅い の見込みとの差", "p 値", "変わらずと比べた上乗せ"],
         "rows": work_rows(T, TA, TB), "caption": "時計は、同じコース・同じ年の全追い切りで標準化した 4F 時計の、その馬の過去のレース前の値の平均との差。"},
        {"type": "accordion", "title": "データと方法・注意点", "blocks": [
            {"type": "text", "text": (
                f"・対象: {min(YEARS)}〜{max(YEARS)}年の JRA 平地で、前走の馬体重が分かる出走 {a['n_all']:,} 走。取消・除外は除き、中止・失格は 3 着以内でないとして数えています。\n\n"
                "・馬体重の履歴: 各馬のデビューからの全戦績（2020 年より前・地方を含む）。ベスト体重は過去の JRA の出走で 3 着以内だったときの体重の中央値で、今回より前の出走だけから計算しています。\n\n"
                f"・休養明け = 前走から {REST_DAYS} 日以上、長期休養 = {LONG_REST} 日以上。けが・故障の理由はデータに無いため、長期休養で近似しています（放牧のための休養も含みます）。\n\n"
                "・3 分類は体重の数字からの推定で、実際に太かったか・成長したかを見たものではありません。2〜3 歳は過去最高を更新しやすいので、多くが『成長』に入ります。\n\n"
                "・追い切りの評価はレース別の最終追い切りの表（A〜D）。時計は馬ごとの追い切りの記録で、取得済みの一部の馬（2021 年は約 6 割、2025 年は 9 割超）に限られます。"
                "追い切りの内容（併せ馬・騎乗者・脚色など）や、厩舎のコメントの「馬体が成長」といった文言は使っていません。")},
        ]},
    ]
    any_sig = any(A[g]["top3_holm"] < 0.05 for g in (FAT, BACK, GROW))
    return {
        "slug": "weight-change",
        "title": "馬体重の大幅増は危険信号？——増えた理由で分けると",
        "claim": "馬体重の変化が大きいほど成績が落ちる",
        "verdict": "partial" if any_sig else "not_supported",
        "summary": (f"+10kg 以上の大幅増の複勝率は {pct(hi['top3'])}（変わらず {pct(same_b['top3'])}）ですが、多くはオッズに織り込み済みです。"
                    f"増えた理由で分けると、オッズをそろえた差は太め {A[FAT]['top3'][0] * 100:+.1f}pt・ベスト体重へ戻り {A[BACK]['top3'][0] * 100:+.1f}pt・成長 {A[GROW]['top3'][0] * 100:+.1f}pt でした。"),
        "categories": ["horse/sex"],
        "published": "2026-10-28",
        "updated": "2026-11-02",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年",
                    "conditions": "JRA全競馬場・芝ダート（障害除く）。前走の馬体重が分かる出走。馬体重の履歴はデビューからの全戦績",
                    "sample": f"{a['n_all']:,} 走（うち +10kg 以上 {a['n_big']:,} 走）"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not CAREER.exists():
        raise SystemExit(f"{CAREER} がありません。先に python -m src.scripts.data.tp32_extract_equipment を実行してください")
    col = article(analyze(frame()))
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
