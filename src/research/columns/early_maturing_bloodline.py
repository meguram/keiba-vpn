"""めぐコラム『エピファネイアは早熟？』を実データから集計し、frontend/content/columns/early-maturing-bloodline.json を書く。

問い:
  1. 種牡馬によって、年齢が上がると成績が落ちる（早熟）傾向に統計的な根拠があるか。
     全産駒の年齢別の成績と、各種牡馬の産駒の成績に、偶然を超える差があるか。
  2. 各種牡馬の産駒の年齢別の平均馬体重（牡・牝別）。体重が伸びない種牡馬ほど、年齢で成績が落ちるか。

データ:
  レース: compare_series.load()（2020〜2025 年の JRA 平地、取消・除外を除く）。馬体重は同じ race_result_flat の weight。
  父:     compare_series.SIRE_MAP_PATH（horse_sire_map.parquet）。対象の出走の 100% に父が付く。
定義:
  年齢はレース時の馬齢（1 月 1 日に加齢）。若い時期 = 2〜3 歳、古馬 = 4 歳以上。
  主要種牡馬 = 産駒の出走が MIN_RUNS 以上、かつ下の「同じ馬の比較」に使える産駒が MIN_HORSES 頭以上。
落とし穴と対処:
  ・年齢別の複勝率をそのまま並べると、走らない馬から先に引退する（生き残りの偏り）・勝つとクラスが上がって相手が強くなる、の影響が混ざる。
    → 同じ馬の 2〜3 歳と 4 歳以上の成績を比べる（2〜3 歳・4 歳以上の両方に出走した、2018 年以降生まれの馬）。
      馬ごとの変化 Δ = 4 歳以上の複勝率 − 2〜3 歳の複勝率。全産駒の平均の Δ との差を、種牡馬ごとに Welch の t で検定し、
      多数の種牡馬を比べるので Benjamini–Hochberg（FDR）で補正する。種牡馬で Δ が違うか全体の検定（一元配置分散分析）も行う。
    → オッズの見込みとも比べる。見込み複勝率 = 見込み勝率（1/オッズをレース内で合計 1 に直したもの）が同じくらいの馬
      （全出走を 40 帯に分けたもの）の実際の複勝率。Δ（実際 − 見込み）が 0 なら、市場は年齢による変化を織り込んでいる。
  ・体重の伸び: 同じ馬の 4 歳以上の平均体重 − 2〜3 歳の平均体重。牡（セン含む）と牝で平均が違うので、性別ごとの平均を引いてから種牡馬で平均する。
    種牡馬ごとの「体重の伸び」と「Δ − 全体の Δ」の相関（Pearson・Spearman）を見る。馬ごとの相関も見る。

使い方:
    python -m src.research.columns.early_maturing_bloodline            # 集計して記事を書く
    python -m src.research.columns.early_maturing_bloodline --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.research.columns.compare_series import FLAT_DIR, SIRE_MAP_PATH, YEARS, _p, _r, _rci, _stat, load
from src.research.columns.wind_venue import pct, pfmt

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/early-maturing-bloodline.json"
AGES = ["2歳", "3歳", "4歳", "5歳", "6歳以上"]
SEXES = {"牡・牝とも": None, "牡・セン": ("牡", "セ"), "牝": ("牝",)}
MIN_RUNS = 1000          # 主要種牡馬: 産駒の出走数
MIN_HORSES = 40          # 主要種牡馬: 同じ馬の比較に使える産駒の頭数
MIN_BIRTH_YEAR = 2018    # 2020 年以降のデータでデビューから追える世代（2018 年生まれは 2020 年に 2 歳）
HEADLINE = "エピファネイア"


def short_name(name: str) -> str:
    """「ハービンジャーHarbinger(英)」→「ハービンジャー」。英字だけの名前（Into Mischief）はそのまま。"""
    m = re.match(r"^(.*?[^\x00-\x7F])[A-Za-z].*$", str(name))
    return m.group(1) if m else str(name)


def bh(ps: list[float]) -> list[float]:
    """Benjamini–Hochberg の補正（q 値）。"""
    p = np.asarray(ps, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


def expected_top3(implied: pd.Series, top3: pd.Series, bins: int = 40) -> pd.Series:
    """見込み勝率が同じくらいの馬（全出走を bins 帯に分けたもの）の実際の複勝率。"""
    q = pd.qcut(implied.rank(method="first"), bins, labels=False)
    return top3.groupby(q).transform("mean")


def frame() -> pd.DataFrame:
    df = load()
    w = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_id", "weight"]) for y in YEARS])
    w["horse_id"] = w["horse_id"].astype(str)
    df = df.merge(w.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").where(lambda s: s.between(300, 700))     # 0 は計量なし
    m = pd.read_parquet(SIRE_MAP_PATH, columns=["horse_id", "sire_id", "sire_name"]).dropna(subset=["sire_id"]).drop_duplicates("horse_id")
    m["horse_id"] = m["horse_id"].astype(str)
    df = df.merge(m, on="horse_id", how="inner")
    df["sire"] = df["sire_name"].map(short_name)
    df["age_band"] = np.where(df["age"] >= 6, "6歳以上", df["age"].astype("Int64").astype(str) + "歳")
    df["old"] = (df["age"] >= 4).astype(int)
    df["sex2"] = np.where(df["sex"] == "牝", "牝", "牡・セン")
    df["exp3"] = expected_top3(df["implied"], df["top3"])
    df["res3"] = df["top3"] - df["exp3"]
    df["birth"] = pd.to_numeric(df["horse_id"].str[:4], errors="coerce")
    return df


def horses(df: pd.DataFrame) -> pd.DataFrame:
    """同じ馬の 2〜3 歳と 4 歳以上の比較（両方に出走した、MIN_BIRTH_YEAR 年以降生まれの馬）。"""
    d = df[df["birth"] >= MIN_BIRTH_YEAR]
    g = d.groupby(["horse_id", "old"]).agg(top3=("top3", "mean"), res3=("res3", "mean"), weight=("weight", "mean"), n=("top3", "size")).unstack("old")
    g = g.dropna(subset=[("n", 0), ("n", 1)])
    h = pd.DataFrame({"d_top3": g[("top3", 1)] - g[("top3", 0)], "d_res3": g[("res3", 1)] - g[("res3", 0)],
                      "grow": g[("weight", 1)] - g[("weight", 0)], "n_young": g[("n", 0)], "n_old": g[("n", 1)], "young3": g[("top3", 0)]})
    meta = d.drop_duplicates("horse_id").set_index("horse_id")[["sire", "sex2"]]
    h = h.join(meta)
    # 若い時期によく走った馬ほど、勝ってクラスが上がり・平均へ戻るので、4 歳以降に大きく落ちる。2〜3 歳の複勝率をそろえた変化（回帰の残差）も出す
    b = np.polyfit(h["young3"], h["d_top3"], 1)
    h["d_adj"] = h["d_top3"] - np.polyval(b, h["young3"])
    # 牡馬は牝馬より体重が伸びる。性別の平均を引いて、性別の割合が違う種牡馬どうしを比べられるようにする
    h["grow_c"] = h["grow"] - h.groupby("sex2")["grow"].transform("mean")
    return h


def welch(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def sire_table(df: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    runs = df.groupby("sire").size()
    nh = h.groupby("sire").size()
    sires = [s for s in runs.index if runs[s] >= MIN_RUNS and nh.get(s, 0) >= MIN_HORSES]
    rows = []
    for s in sires:
        x, hs, ho = df[df["sire"] == s], h[h["sire"] == s], h[h["sire"] != s]
        ci = stats.t.interval(0.95, len(hs) - 1, loc=hs["d_top3"].mean(), scale=stats.sem(hs["d_top3"]))
        rows.append({
            "sire": s, "runs": len(x), "horses": len(hs),
            "young": x.loc[x["old"] == 0, "top3"].mean(), "older": x.loc[x["old"] == 1, "top3"].mean(),
            "d": hs["d_top3"].mean(), "d_ci": ci, "d_vs": hs["d_top3"].mean() - h["d_top3"].mean(),
            "p": welch(hs["d_top3"], ho["d_top3"]),
            "da": hs["d_adj"].mean() - h["d_adj"].mean(), "pa": welch(hs["d_adj"], ho["d_adj"]),
            "dr": hs["d_res3"].mean(), "dr_vs": hs["d_res3"].mean() - h["d_res3"].mean(), "pr": welch(hs["d_res3"], ho["d_res3"]),
            "grow": hs["grow"].mean(), "grow_c": hs["grow_c"].mean(),
            "grow_m": hs.loc[hs["sex2"] == "牡・セン", "grow"].mean(), "grow_f": hs.loc[hs["sex2"] == "牝", "grow"].mean(),
            "share_old": (x["old"] == 1).mean(),
        })
    t = pd.DataFrame(rows)
    t["q"] = bh(t["p"].tolist())
    t["qr"] = bh(t["pr"].tolist())
    t["qa"] = bh(t["pa"].tolist())
    return t.sort_values("d_vs").reset_index(drop=True)


def _r3(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


def _ci3(c: tuple[float, float] | None) -> list[float] | None:
    return None if c is None else [round(c[0], 3), round(c[1], 3)]


def pair(df: pd.DataFrame, sire: str, metric: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """年齢ごとに、その種牡馬の産駒と全産駒の率。p はその種牡馬の産駒と他の産駒の差（Welch の z）。"""
    a_ser = {"name": "その種牡馬の産駒", "values": [], "n": [], "ci": []}
    b_ser = {"name": "全産駒", "values": [], "n": [], "ci": []}
    ps = []
    for a in AGES:
        xa = df[df["age_band"] == a]
        sa, sb = xa.loc[xa["sire"] == sire, metric], xa.loc[xa["sire"] != sire, metric]
        ma, na, ca, va = _stat(sa)
        mb, nb, cb, _ = _stat(xa[metric])
        mo, _, _, vo = _stat(sb)
        ok = na >= 20
        a_ser["values"].append(_r3(ma) if ok else None); a_ser["n"].append(na); a_ser["ci"].append(_ci3(ca) if ok else None)
        b_ser["values"].append(_r3(mb)); b_ser["n"].append(nb); b_ser["ci"].append(_ci3(cb))
        p = _p(va, vo, ma, mo) if ok else None
        ps.append(None if p is None else float(f"{p:.2g}"))
    return [a_ser, b_ser], {"a": a_ser["name"], "b": "全産駒", "p": ps}


def age_chart(df: pd.DataFrame, sires: list[str]) -> dict[str, Any]:
    """種牡馬をプルダウンで切り替える、年齢別の複勝率（その種牡馬の産駒 vs 全産駒）。"""
    opts = [HEADLINE] + [s for s in sires if s != HEADLINE]
    variants = {}
    for s in opts[1:]:
        ser, cmp = pair(df, s, "top3")
        variants[s] = {"series": ser, "compare": cmp}
    ser, cmp = pair(df, opts[0], "top3")
    return {"type": "chart", "title": "種牡馬別 年齢ごとの複勝率（その種牡馬の産駒 vs 全産駒）", "xLabel": "年齢（レース時）", "yLabel": "複勝率",
            "xs": AGES, "series": ser, "compare": cmp, "filters": [{"label": "種牡馬", "options": opts}], "variants": variants,
            "caption": (f"{min(YEARS)}〜{max(YEARS)}年 JRA 平地。プルダウンで種牡馬を切り替えられる（主要 {len(opts)} 頭、産駒の出走が多い順）。"
                        "＊は他の産駒との差が有意（p<0.05）な年齢。年齢別の率には生き残りの偏り・クラスの上昇が混ざるので、結論は本文の「同じ馬での比較」を見てほしい。")}


GROW_SEXES = {"牡・セン": ("牡", "セ"), "牝": ("牝",)}


def growth_series(x: pd.DataFrame, s: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    a_ser = {"name": "その種牡馬の産駒", "values": [], "n": []}
    b_ser = {"name": "全産駒", "values": [], "n": []}
    ps = []
    for a in AGES:
        xa = x[x["age_band"] == a]
        sa, so = xa.loc[xa["sire"] == s, "grow_debut"], xa.loc[xa["sire"] != s, "grow_debut"]
        ok = len(sa) >= 20
        a_ser["values"].append(round(float(sa.mean()), 1) if ok else None); a_ser["n"].append(len(sa))
        b_ser["values"].append(round(float(xa["grow_debut"].mean()), 1) if len(xa) else None); b_ser["n"].append(len(xa))
        p = welch(sa, so) if ok else float("nan")
        ps.append(None if not np.isfinite(p) else float(f"{p:.2g}"))
    return [a_ser, b_ser], {"a": a_ser["name"], "b": "全産駒", "p": ps}


def growth_chart(df: pd.DataFrame, sires: list[str]) -> dict[str, Any]:
    """デビュー（JRA 初出走）の体重からの増加の年齢別平均。種牡馬・性別で切り替える。"""
    opts = [HEADLINE] + [s for s in sires if s != HEADLINE]
    x0 = df[(df["birth"] >= MIN_BIRTH_YEAR) & df["grow_debut"].notna()]
    variants = {}
    base = None
    for sk, sv in GROW_SEXES.items():
        x = x0[x0["sex"].isin(sv)]
        for s in opts:
            ser, cmp = growth_series(x, s)
            if base is None:
                base = (ser, cmp)
            else:
                variants[f"{s}|{sk}"] = {"series": ser, "compare": cmp}
    return {"type": "chart", "title": "種牡馬別 年齢ごとの体重の増え方（デビューからの増加）", "xLabel": "年齢（レース時）", "yLabel": "デビューからの増加（kg）",
            "xs": AGES, "series": base[0], "compare": base[1], "format": "num",
            "filters": [{"label": "種牡馬", "options": opts}, {"label": "性別", "options": list(GROW_SEXES)}], "variants": variants,
            "caption": (f"{MIN_BIRTH_YEAR} 年以降生まれ（デビューから追える世代）の、各出走の体重 − その馬の JRA 初出走の体重 の平均（kg）。"
                        "性別で切り替えられる。＊は他の産駒との差が有意（p<0.05）。年齢別の平均馬体重そのものは下の表。")}


def analyze(df: pd.DataFrame) -> dict[str, Any]:
    a: dict[str, Any] = {}
    d = df.sort_values(["horse_id", "date", "race_id"])
    d["debut_w"] = d.groupby("horse_id")["weight"].transform("first")
    d["grow_debut"] = d["weight"] - d["debut_w"]
    # デビュー（JRA 初出走）が 2020 年以降のデータにある馬だけ（2018 年以降生まれ）。debut_w は他の世代では意味がない
    d.loc[d["birth"] < MIN_BIRTH_YEAR, ["debut_w", "grow_debut"]] = np.nan
    df = d
    h = horses(df)
    t = sire_table(df, h)
    a["n_runs"], a["n_horses_all"], a["n_sires_all"] = len(df), df["horse_id"].nunique(), df["sire"].nunique()
    a["h"], a["t"] = h, t
    a["all_young"] = df.loc[df["old"] == 0, "top3"].mean()
    a["all_older"] = df.loc[df["old"] == 1, "top3"].mean()
    a["all_d"], a["all_dr"] = h["d_top3"].mean(), h["d_res3"].mean()
    a["curve_all"] = {ab: df.loc[df["age_band"] == ab, "top3"].mean() for ab in AGES}
    hm = h[h["sire"].isin(t["sire"])]
    a["anova_p"] = float(stats.f_oneway(*[g["d_top3"].values for _, g in hm.groupby("sire")]).pvalue)
    a["anova_pa"] = float(stats.f_oneway(*[g["d_adj"].values for _, g in hm.groupby("sire")]).pvalue)
    a["anova_pr"] = float(stats.f_oneway(*[g["d_res3"].values for _, g in hm.groupby("sire")]).pvalue)
    a["n_h_major"] = len(hm)
    # 体重の伸び（性別の平均を引いたもの）と、年齢での落ち方（全体の Δ との差）の相関。種牡馬単位と馬単位
    a["corr_sire"] = (stats.pearsonr(t["grow_c"], t["d_vs"]), stats.spearmanr(t["grow_c"], t["d_vs"]))
    a["corr_sire_a"] = (stats.pearsonr(t["grow_c"], t["da"]), stats.spearmanr(t["grow_c"], t["da"]))
    a["corr_sire_r"] = (stats.pearsonr(t["grow_c"], t["dr_vs"]), stats.spearmanr(t["grow_c"], t["dr_vs"]))
    hh = h.dropna(subset=["grow_c"])
    a["corr_horse"] = stats.pearsonr(hh["grow_c"], hh["d_top3"])
    a["corr_horse_a"] = stats.pearsonr(hh["grow_c"], hh["d_adj"])
    a["corr_horse_r"] = stats.pearsonr(hh["grow_c"], hh["d_res3"])
    # 体重の伸びで馬を 3 分し、Δ を比べる（性別の平均を引いた伸び）
    hh = hh.assign(gb=pd.qcut(hh["grow_c"], 3, labels=["伸び小", "中", "伸び大"]))
    a["grow_bands"] = hh.groupby("gb", observed=True).agg(n=("d_top3", "size"), grow=("grow", "mean"), d=("d_top3", "mean"), da=("d_adj", "mean"), dr=("d_res3", "mean")).reset_index()
    a["grow_sex"] = h.groupby("sex2")["grow"].agg(["mean", "size"])
    sires = t.sort_values("runs", ascending=False)["sire"].tolist()
    a["sires_by_runs"] = sires
    a["c_age"] = age_chart(df, sires)
    a["c_grow"] = growth_chart(df, sires)
    # 種牡馬 × 性別 × 年齢 の平均馬体重の表
    a["wt_rows"] = []
    for s in sires:
        row = [s]
        for sv in (("牡", "セ"), ("牝",)):
            x = df[(df["sire"] == s) & df["sex"].isin(sv)]
            for lo, hi in ((2, 2), (3, 3), (4, 4), (5, 99)):
                v = x.loc[x["age"].between(lo, hi), "weight"]
                row.append("–" if v.count() < 20 else f"{v.mean():.0f}")
        a["wt_rows"].append(row)
    return a


def pt(v: float, d: int = 1) -> str:
    return f"{v * 100:+.{d}f}pt"


def qfmt(q: float) -> str:
    return "<0.001" if q < 0.001 else f"{q:.3f}"


def qeq(q: float) -> str:
    """「q<0.001」「q=0.046」の形。"""
    return "q<0.001" if q < 0.001 else f"q={q:.3f}"


MEASURES = [("d_vs", "q", "そのままの変化"), ("da", "qa", "2〜3歳の成績をそろえた変化"), ("dr_vs", "qr", "見込みとの差の変化")]


def sig_list(t: pd.DataFrame, col: str, qcol: str) -> pd.DataFrame:
    return t[t[qcol] < 0.05].sort_values(col)


def names(x: pd.DataFrame, col: str, qcol: str) -> str:
    return "、".join(f"{r['sire']}（{pt(r[col])}、{qeq(r[qcol])}）" for _, r in x.iterrows())


def polite(s: str) -> str:
    """「〜有意でない」「〜有意」で終わる短い文を、です・ます調にする（要約用）。"""
    if s.endswith("有意でない"):
        return s[: -len("有意でない")] + "有意ではありません"
    return s + "です" if s.endswith("有意") else s


def article(a: dict[str, Any]) -> dict[str, Any]:
    t: pd.DataFrame = a["t"]
    T = t.set_index("sire")
    E = T.loc[HEADLINE]
    nsig = {c: sig_list(t, c, q) for c, q, _ in MEASURES}
    # 3 つの物差しのうち、2 つ以上で同じ向きに有意な種牡馬（頑健な結果）
    hits: dict[str, list[float]] = {}
    for c, q, _ in MEASURES:
        for _, r in nsig[c].iterrows():
            hits.setdefault(r["sire"], []).append(r[c])
    robust = {s: v for s, v in hits.items() if len(v) >= 2 and (all(x < 0 for x in v) or all(x > 0 for x in v))}
    early = [s for s, v in robust.items() if v[0] < 0]
    late = [s for s, v in robust.items() if v[0] > 0]
    (pr_s, pp_s), (sr_s, _) = a["corr_sire"]
    (pr_a, pp_a), _ = a["corr_sire_a"]
    (pr_r, pp_r), _ = a["corr_sire_r"]
    ch, cha, chr_ = a["corr_horse"], a["corr_horse_a"], a["corr_horse_r"]
    gb = a["grow_bands"].set_index("gb")
    gs = a["grow_sex"]
    nh = len(a["h"])

    def cell(r, c, q):
        return f"{pt(r[c])}（{qeq(r[q])}）" + ("＊" if r[q] < 0.05 else "")
    rows = [[r["sire"], f"{r['runs']:,}", f"{r['horses']:,}", f"{pct(r['young'])} → {pct(r['older'])}",
             f"{pt(r['d'])}（{r['d_ci'][0] * 100:+.1f}〜{r['d_ci'][1] * 100:+.1f}）",
             cell(r, "d_vs", "q"), cell(r, "da", "qa"), cell(r, "dr_vs", "qr"), f"{r['grow_m']:+.1f} / {r['grow_f']:+.1f}"] for _, r in t.iterrows()]

    sig_lines = []
    for c, q, label in MEASURES:
        x = nsig[c]
        sig_lines.append(f"・{label}: " + (names(x, c, q) if len(x) else "なし"))
    epi_sig = [label for c, q, label in MEASURES if E[q] < 0.05]
    epi_txt = (f"エピファネイア産駒は、年齢別の複勝率をそのまま見ると 2〜3 歳 {pct(E['young'])}、4 歳以上 {pct(E['older'])} です。"
               f"同じ馬（{int(E['horses'])} 頭）で比べた変化は {pt(E['d'])} で、全産駒の平均（{pt(a['all_d'])}）より {abs(E['d_vs']) * 100:.1f}pt "
               + ("大きく落ちています" if E["d_vs"] < 0 else "小さい落ち方です")
               + f"（補正前 p={qfmt(E['p'])}、多重比較の補正後 {qeq(E['q'])}）。"
               f"2〜3 歳の成績をそろえると {pt(E['da'])}（{qeq(E['qa'])}）、オッズの見込みとの差では {pt(E['dr_vs'])}（{qeq(E['qr'])}）です。"
               + ("どの物差しでも補正後は有意でなく、「やや早く落ちる傾向はあるが、偶然と区別できるほどではない」というのがデータの答えです。" if not epi_sig
                  else "そのままの変化では補正後も有意ですが、2〜3 歳の成績をそろえる・オッズの見込みと比べると差は縮んで有意でなくなります。"
                  "若いころによく走った分の反動（平均への回帰・クラスの上昇）が大きく、市場もその落ち方をある程度織り込んでいる、というのがデータの答えです。"
                  if epi_sig == [MEASURES[0][2]]
                  else f"{'・'.join(epi_sig)}で有意です。"))
    # エピファネイアの短い言い方（要約・まとめ・タイトルで使う）
    if not epi_sig:
        epi_short = f"エピファネイアは全産駒より {abs(E['d_vs']) * 100:.1f}pt {'大きく' if E['d_vs'] < 0 else '小さく'}落ちるが、補正後は有意でない（{qeq(E['q'])}）"
        epi_title = "同じ馬で比べると、はっきりした差はない"
    elif epi_sig == [MEASURES[0][2]]:
        epi_short = (f"エピファネイアは全産駒より {abs(E['d_vs']) * 100:.1f}pt {'大きく' if E['d_vs'] < 0 else '小さく'}落ちる（補正後 {qeq(E['q'])}）が、"
                     f"2〜3 歳の成績をそろえると {pt(E['da'])}（{qeq(E['qa'])}）、見込みとの差では {pt(E['dr_vs'])}（{qeq(E['qr'])}）で有意でない")
        epi_title = "落ち方は大きいが、若いころの好成績の反動を除くと有意差なし"
    else:
        epi_short = f"エピファネイアは全産駒より {abs(E['d_vs']) * 100:.1f}pt {'大きく' if E['d_vs'] < 0 else '小さく'}落ち、{'・'.join(epi_sig)}で補正後も有意"
        epi_title = "同じ馬で比べても、全産駒より早く落ちる"
    robust_txt = ("3 つの物差しのうち 2 つ以上で同じ向きに有意なのは、"
                  + ("早熟寄り " + "、".join(early) if early else "")
                  + ("、" if early and late else "")
                  + ("晩成寄り " + "、".join(late) if late else "")
                  + " だけです。" if robust else "3 つの物差しのうち 2 つ以上で同じ向きに有意な種牡馬はいません。")
    verdict = "partial" if robust or any(len(v) for v in nsig.values()) else "not_supported"
    head = early[0] if early else None
    title = "エピファネイアは早熟？——" + (f"同じ馬で比べると、はっきり早く落ちるのは{head}" if head and head != HEADLINE else epi_title)

    corr_txt = (f"種牡馬ごとの体重の伸び（同じ馬の 4 歳以上 − 2〜3 歳の平均体重。性別の平均を引いたもの）と、年齢での成績の落ち方の相関は、"
                f"そのままの変化で r={pr_s:+.2f}（{pfmt(pp_s)}）、2〜3 歳の成績をそろえた変化で r={pr_a:+.2f}（{pfmt(pp_a)}）、見込みとの差で r={pr_r:+.2f}（{pfmt(pp_r)}）です"
                f"（主要 {len(t)} 種牡馬）。"
                + ("どれも有意でなく、「体重が伸びない種牡馬ほど早く落ちる」という関係は見られません。" if max(pp_s, pp_a, pp_r) >= 0.05 and min(pp_s, pp_a, pp_r) >= 0.05 else "")
                + f"\n\n馬 1 頭ずつ（{len(a['h'].dropna(subset=['grow_c'])):,} 頭）で見ても、体重の伸びと複勝率の変化の相関は r={ch[0]:+.3f}（{pfmt(ch[1])}）、"
                f"2〜3 歳の成績をそろえると r={cha[0]:+.3f}（{pfmt(cha[1])}）、見込みとの差では r={chr_[0]:+.3f}（{pfmt(chr_[1])}）です。"
                "頭数が多いので p 値が小さく出るものもありますが、相関はどれも ±0.04 以内で、向きもそろいません。"
                f"体重の伸びで馬を 3 等分すると、伸び小（平均 {gb.loc['伸び小', 'grow']:+.1f}kg）・中（{gb.loc['中', 'grow']:+.1f}kg）・伸び大（{gb.loc['伸び大', 'grow']:+.1f}kg）の複勝率の変化は "
                f"そのままで {pt(gb.loc['伸び小', 'd'])}・{pt(gb.loc['中', 'd'])}・{pt(gb.loc['伸び大', 'd'])}、"
                f"2〜3 歳の成績をそろえると {pt(gb.loc['伸び小', 'da'])}・{pt(gb.loc['中', 'da'])}・{pt(gb.loc['伸び大', 'da'])} です。"
                "伸び小の馬が少し大きく落ちて見えるのは成績をそろえたときだけで、その差も 1〜2pt 程度です。"
                "「馬体重が変わらない＝成長しない＝相対的に早熟」という見方は、少なくとも JRA の出走時の体重では裏付けられません。"
                "体重の伸びは骨格・筋肉の成長だけでなく、仕上げ方（絞る・余裕を持たせる）でも変わるので、成長の物差しとしては粗いことに注意が必要です。")

    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "全産駒の複勝率の変化（同じ馬）", "value": pt(a["all_d"]), "sub": f"2〜3歳 → 4歳以上。{nh:,} 頭"},
            {"label": "全産駒と落ち方が違う種牡馬", "value": f"{len(robust)} / {len(t)} 頭", "sub": "3 つの物差しのうち 2 つ以上で、補正後も有意"},
            {"label": "エピファネイア（全産駒との差）", "value": pt(E["d_vs"]), "sub": f"補正後 {qeq(E['q'])}。2〜3歳をそろえると {pt(E['da'])}"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"年齢別の複勝率をそのまま並べると、全産駒で 2〜3 歳 {pct(a['all_young'])}、4 歳以上 {pct(a['all_older'])} です。"
            "ただし、走らない馬から先に引退し、勝った馬はクラスが上がって相手が強くなるので、この数字だけでは血統の早熟・晩成は分かりません。"
            f"そこで 2〜3 歳と 4 歳以上の両方を走った同じ馬（{MIN_BIRTH_YEAR} 年以降生まれ、{nh:,} 頭）で、複勝率がどれだけ変わったかを比べました。"
            f"全産駒の平均は {pt(a['all_d'])} で、どの血統でも 4 歳以降は大きく下がります（クラスが上がる分を含む）。\n\n"
            f"産駒の出走が {MIN_RUNS:,} 以上の主要 {len(t)} 種牡馬で、この変化が全産駒の平均と違うかを、3 つの物差しで検定しました"
            "（多数の種牡馬を比べるので、Benjamini–Hochberg 法で補正した q 値が 0.05 未満を「差がある」とする）。\n"
            "・そのままの変化\n・2〜3 歳の成績をそろえた変化（若いころによく走った馬ほど、平均への回帰とクラスの上昇で大きく落ちる分を除く）\n"
            "・オッズの見込みとの差の変化（市場が年齢の変化を織り込んでいるか）\n\n"
            + "\n".join(sig_lines) + "\n\n" + robust_txt
            + f"種牡馬によって変化が違うか全体の検定（分散分析）は、そのままで {pfmt(a['anova_p'])}、2〜3 歳をそろえて {pfmt(a['anova_pa'])}、見込みとの差で {pfmt(a['anova_pr'])} で、"
            "「血統によって年齢での成績の変わり方が違う」こと自体は統計的に支持されます。ただ、1 頭ずつの差は小さく、物差しを変えると顔ぶれが入れ替わります。\n\n" + epi_txt)},
        a["c_age"],
        {"type": "heading", "text": "同じ馬で比べた、年齢による成績の変化（種牡馬別）"},
        {"type": "text", "text": (
            "表は、全産駒と比べて落ち方が大きい順です。＊は多重比較の補正後も有意（q<0.05）。"
            "「そのまま」で落ち方が大きく見えても、「2〜3 歳をそろえた」差が小さければ、若いころの好成績の反動（平均への回帰・クラスの上昇）でほぼ説明できます。"
            "「見込みとの差」が 0 に近ければ、市場はすでにその落ち方を人気に織り込んでいて、馬券の狙い目にはなりません。")},
        {"type": "table",
         "columns": ["父", "産駒の出走", "比較できた馬", "複勝率 2〜3歳 → 4歳以上", "同じ馬の変化（95%区間）", "全産駒との差（そのまま）", "全産駒との差（2〜3歳をそろえる）", "全産駒との差（見込みとの差）", "体重の伸び kg 牡・セン / 牝"],
         "rows": rows,
         "caption": (f"{min(YEARS)}〜{max(YEARS)}年 JRA 平地。主要種牡馬は産駒の出走 {MIN_RUNS:,} 以上かつ比較できた馬 {MIN_HORSES} 頭以上。"
                     "「複勝率 2〜3歳 → 4歳以上」は全出走をそのまま集計したもの。「同じ馬の変化」は 2〜3 歳と 4 歳以上の両方を走った馬の、4 歳以上の複勝率 − 2〜3 歳の複勝率の平均。"
                     "全産駒との差の q 値は、その種牡馬の馬と他の種牡馬の馬の Welch の t 検定の p 値を Benjamini–Hochberg 法で補正したもの。"
                     "見込みとの差は 実際の複勝率 − オッズから見込まれる複勝率。体重の伸びは同じ馬の 4 歳以上 − 2〜3 歳の平均体重。")},
        {"type": "heading", "text": "年齢別の馬体重と「成長しない馬は早熟」説"},
        {"type": "text", "text": (
            f"同じ馬の 2〜3 歳から 4 歳以上への体重の伸びは、平均で 牡・セン {gs.loc['牡・セン', 'mean']:+.1f}kg、牝 {gs.loc['牝', 'mean']:+.1f}kg です。"
            "種牡馬・性別ごとの体重の増え方は下のグラフ、年齢別の平均馬体重は表で見られます。\n\n" + corr_txt)},
        a["c_grow"],
        {"type": "accordion", "title": "種牡馬別 年齢ごとの平均馬体重（牡・セン / 牝）", "blocks": [
            {"type": "table", "columns": ["父", "牡・セン 2歳", "牡・セン 3歳", "牡・セン 4歳", "牡・セン 5歳以上", "牝 2歳", "牝 3歳", "牝 4歳", "牝 5歳以上"], "rows": a["wt_rows"],
             "caption": "各年齢の出走時の馬体重の平均（kg）。出走が 20 未満は「–」。全世代の出走をそのまま集計しているので、年齢が上がるほど体重の重い（丈夫な）馬が残る偏りも含む。"},
        ]},
        {"type": "heading", "text": "まとめ"},
        {"type": "text", "text": (
            "・年齢別の成績をそのまま比べると、生き残りの偏りとクラスの上昇が混ざる。同じ馬の 2〜3 歳と 4 歳以上で比べるのが妥当。\n"
            f"・同じ馬で比べると、全産駒でも 4 歳以上で複勝率は {pt(a['all_d'])} 変わる。血統によって変わり方が違うこと自体は有意だが、1 頭ずつの差ははっきりしない。\n"
            + (f"・物差しを変えても早く落ちるのは {'、'.join(early)}。" if early else "") + (f"4 歳以降も保つのは {'、'.join(late)}。" if late else "") + ("\n" if robust else "")
            + f"・{epi_short}。\n"
            f"・体重が伸びない種牡馬・馬ほど早く落ちる、という関係は見られない（種牡馬 r={pr_s:+.2f}、馬 r={ch[0]:+.3f}）。")},
        {"type": "link", "href": "/analysis", "label": "めぐ分析で出走馬の血統を見る", "desc": "種牡馬×コースなど、今週のレースの血統別成績を確認できます。"},
        {"type": "accordion", "title": "データと方法・注意点", "blocks": [
            {"type": "text", "text": (
                f"・対象: {min(YEARS)}〜{max(YEARS)}年の JRA 平地（障害を除く）。取消・除外は除き、中止・失格は 3 着以内でないとして数えています。父は血統表から付け、対象の出走のすべてに付いています。\n\n"
                f"・同じ馬での比較は、{MIN_BIRTH_YEAR} 年以降生まれ（2020 年からデビューまで追える世代）で、2〜3 歳と 4 歳以上の両方に JRA 平地で出走した馬です。"
                "2〜3 歳で引退した馬は入らないので、「4 歳まで走り続けた馬の中での変化」を比べています。牝馬は 4〜5 歳で繁殖に上がることが多く、よく走った牝馬ほど早く抜ける偏りもありえます。"
                f"また {MIN_BIRTH_YEAR}〜2021 年生まれの馬の 4 歳以上は、多くて 2025 年までの数年分です（7 歳以降はまだ少ない）。\n\n"
                "・2〜3 歳の成績をそろえた変化: 全馬で「変化 = a + b × 2〜3 歳の複勝率」の直線を当てはめ、その残差を種牡馬ごとに平均したものです。\n\n"
                "・見込み複勝率: 1/単勝オッズ をレース内で合計 1 に直した見込み勝率が同じくらいの馬（全出走を 40 帯に分けたもの）の、実際の複勝率です。\n\n"
                "・グラフの点ごとの p 値は、出走を独立とみなした比較で、同じ馬の複数の出走を考えていないので、やや有意に出やすくなります。結論には同じ馬での比較（表）を使っています。\n\n"
                "・当初の予定のうち、年齢による好走距離・舞台の変化は、今回は扱っていません。")},
        ]},
    ]
    return {
        "slug": "early-maturing-bloodline",
        "title": title,
        "claim": "早熟血統は年齢が上がると走らなくなる",
        "verdict": verdict,
        "summary": (f"同じ馬の 2〜3 歳と 4 歳以上を比べると、全産駒でも複勝率は {pt(a['all_d'])} 変わります。"
                    f"主要 {len(t)} 種牡馬のうち、物差しを変えても全産駒と落ち方が違うのは {len(robust)} 頭"
                    + (f"（{'、'.join(robust)}）" if robust else "")
                    + f"。{polite(epi_short)}。体重が伸びない血統ほど早熟、という関係も見られませんでした。"),
        "categories": ["bloodline/maturity"],
        "published": "2026-10-28",
        "updated": "2026-11-02",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年",
                    "conditions": f"JRA全競馬場・芝ダート（障害除く）。主要種牡馬は産駒の出走 {MIN_RUNS:,} 以上",
                    "sample": f"{a['n_runs']:,} 走（{a['n_horses_all']:,} 頭・{a['n_sires_all']:,} 種牡馬）。同じ馬の比較 {nh:,} 頭"},
        "body": body,
    }


_FLAT_LIST = re.compile(r"\[[^\[\]{}]*\]")


def dumps(col: dict[str, Any]) -> str:
    """indent=2 の JSON。ただし、中に配列・オブジェクトを含まない配列（数値の列・表の行）は 1 行にまとめる（ファイルを小さく、読みやすく）。"""
    text = json.dumps(col, ensure_ascii=False, indent=2)
    text = _FLAT_LIST.sub(lambda m: json.dumps(json.loads(m.group(0)), ensure_ascii=False, separators=(", ", ": ")), text)
    return text + "\n"


def report(a: dict[str, Any]) -> None:
    t = a["t"]
    print(f"runs={a['n_runs']} horses(h)={len(a['h'])} sires={len(t)} all_d={a['all_d']:.4f} all_dr={a['all_dr']:.4f} anova p={a['anova_p']:.3g} pa={a['anova_pa']:.3g} pr={a['anova_pr']:.3g}")
    print(t[["sire", "runs", "horses", "young", "older", "d", "d_vs", "p", "q", "da", "pa", "qa", "dr_vs", "pr", "qr", "grow_m", "grow_f", "grow_c"]].round(4).to_string())
    print("corr sire", a["corr_sire"], a["corr_sire_a"], a["corr_sire_r"])
    print("corr horse", a["corr_horse"], a["corr_horse_a"], a["corr_horse_r"])
    print(a["grow_bands"].round(4).to_string())
    print(a["grow_sex"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    a = analyze(frame())
    report(a)
    col = article(a)
    text = dumps(col)
    assert json.loads(text) == col
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(f"(dry-run) {len(text):,} bytes")
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT} ({len(text):,} bytes)")


if __name__ == "__main__":
    main()
