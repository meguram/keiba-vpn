"""めぐコラム『ペースは読めるか——2026年で確かめる』を実データから集計し、frontend/content/columns/pace-scenario.json を書く。

問い: 「ペースを予測できれば有利な馬が分かる」は成り立つか。
  1. ペース（結果）と人気馬の成績の関係は、年が変わっても同じか（2020〜2026 年の年別）。
  2. ペースはレース前に予測できるか（前寄りの頭数・頭数・芝ダで、2020〜2025 年で学習 → 2026 年で当てる。AUC）。
  3. 記事『ハイペースになるほど…』の「前寄りが多いと 1 番人気が崩れる（オッズをそろえても）」は、集計に使っていない 2026 年でも再現するか。

ペースの区分・前寄りの定義は src/research/columns/pace_advantage.py と同じ。2026 年は race_quality にペースの区分があるレースだけ。

使い方:
    python -m src.research.columns.pace_scenario            # 集計して記事を書く
    python -m src.research.columns.pace_scenario --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.research.columns.compare_series as cs
import src.research.columns.race_context as rc
from src.research.columns.race_context import ols_fit, series
from src.research.columns.wind_venue import pct, pfmt

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/pace-scenario.json"
TRAIN_LAST, TEST_YEAR = 2025, 2026
PACES = {"high": "ハイ", "mid": "ミドル", "slow": "スロー"}
NF_BINS = ([-1, 0, 1, 2, 3, 99], ["0頭", "1頭", "2頭", "3頭", "4頭以上"])


def frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    years = range(2020, TEST_YEAR + 1)
    cs.YEARS, rc.YEARS = years, years                     # 2026 年まで読む（ほかの記事は 2020〜2025 年のまま）
    d = rc.with_context(cs.load())
    rq = pd.read_parquet(cs.RACE_QUALITY_PATH, columns=["race_id", "pace_cls"]).dropna().drop_duplicates("race_id")
    d = d.merge(rq, on="race_id", how="left")
    d["year"] = pd.to_datetime(d["date"]).dt.year
    d["lq"] = np.log(d["odds"])
    d["key"] = d["venue"] + d["surface"] + d["distance"].astype(str)
    d["front"] = (d["style_pre"] == "前寄り").astype(int)
    r = d.groupby("race_id").agg(nf=("front", "sum"), starters=("starters", "first"), pace=("pace_cls", "first"), year=("year", "first"),
                                 surface=("surface", "first"))
    r["nfb"] = pd.cut(r["nf"], NF_BINS[0], labels=NF_BINS[1]).astype(str)
    d = d.merge(r[["nf", "nfb"]], left_on="race_id", right_index=True)
    return d, r[r["pace"].notna()].copy()


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def analyze(d: pd.DataFrame, r: pd.DataFrame) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    a: dict[str, Any] = {}
    years = sorted(r["year"].unique())
    a["years"] = [str(y) for y in years]
    p = d[d["pace_cls"].notna()]
    fav = p[p["popularity"] == 1]
    win = p[p["finish_position"] == 1]
    a["c_fav_year"] = {"type": "chart", "title": "年別 ペースごとの 1番人気の複勝率", "xLabel": "年", "yLabel": "1番人気の複勝率", "xs": a["years"],
                       "series": [series(v, [fav.loc[(fav["year"] == y) & (fav["pace_cls"] == k), "top3"] for y in years]) for k, v in PACES.items()],
                       "caption": f"{TEST_YEAR} 年はペースの区分があるレースだけ。縦線は 95% 信頼区間。"}
    a["c_win_year"] = {"type": "chart", "title": "年別 ペースごとの勝ち馬の平均人気", "xLabel": "年", "yLabel": "勝ち馬の平均人気", "xs": a["years"], "format": "num",
                       "series": [series(v, [win.loc[(win["year"] == y) & (win["pace_cls"] == k), "popularity"] for y in years]) for k, v in PACES.items()],
                       "caption": "大きいほど人気薄が勝っている。縦線は 95% 信頼区間。"}
    a["fav_year"] = {(y, k): fav.loc[(fav["year"] == y) & (fav["pace_cls"] == k), "top3"].mean() for y in years for k in PACES}
    a["n_test"] = int((r["year"] == TEST_YEAR).sum())
    # 予測できるか
    tr, te = r[r["year"] <= TRAIN_LAST], r[r["year"] == TEST_YEAR]

    def X(x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"nf": x["nf"], "starters": x["starters"], "nfr": x["nf"] / x["starters"], "turf": (x["surface"] == "芝").astype(float)})

    a["auc"] = {}
    for k in ("high", "slow"):
        ytr, yte = (tr["pace"] == k).astype(int), (te["pace"] == k).astype(int)
        m = LogisticRegression(max_iter=1000).fit(X(tr), ytr)
        a["auc"][k] = {"nf": auc(yte, te["nf"] * (1 if k == "high" else -1)), "model": auc(yte, m.predict_proba(X(te))[:, 1]), "base": float(yte.mean())}
    a["c_nf_pace"] = {"type": "chart", "title": "前寄りの頭数別 ハイペースになった割合（集計期間と2026年）", "xLabel": "出走馬のうち前寄りの頭数",
                      "yLabel": "ハイペースのレースの割合", "xs": NF_BINS[1],
                      "series": [series(nm, [(x.loc[x["nfb"] == b, "pace"] == "high").astype(float) for b in NF_BINS[1]])
                                 for nm, x in ((f"2020〜{TRAIN_LAST}年", tr), (f"{TEST_YEAR}年", te))],
                      "caption": "前寄り=前走まで直近 3 走の初角の位置の平均が前から 15% 以内の馬。縦線は 95% 信頼区間。"}
    a["nf_high"] = {nm: {b: (x.loc[x["nfb"] == b, "pace"] == "high").mean() for b in NF_BINS[1]} for nm, x in (("train", tr), ("test", te))}
    # 1 番人気 × 前寄りの頭数の再現
    f = d[d["popularity"] == 1]
    a["rep"] = {}
    for nm, x in (("train", f[f["year"] <= TRAIN_LAST]), ("test", f[f["year"] == TEST_YEAR])):
        m = ols_fit("top3 ~ nf + lq + I(lq**2) + starters + C(key)", x)
        a["rep"][nm] = (float(m.params["nf"]), float(m.bse["nf"]), float(m.pvalues["nf"]), len(x))
    ftr, fte = f[f["year"] <= TRAIN_LAST], f[f["year"] == TEST_YEAR]
    a["c_nf_fav"] = {"type": "chart", "title": "前寄りの頭数別 1番人気の複勝率（集計期間と2026年）", "xLabel": "出走馬のうち前寄りの頭数", "yLabel": "1番人気の複勝率",
                     "xs": NF_BINS[1], "series": [series(nm, [x.loc[x["nfb"] == b, "top3"] for b in NF_BINS[1]])
                                                  for nm, x in ((f"2020〜{TRAIN_LAST}年", ftr), (f"{TEST_YEAR}年", fte))],
                     "caption": f"{TEST_YEAR} 年は 1 区分あたり 240〜740 レースと少なく、区間が広い。縦線は 95% 信頼区間。"}
    return a


def article(a: dict[str, Any]) -> dict[str, Any]:
    F, AU, R = a["fav_year"], a["auc"], a["rep"]
    ty = TEST_YEAR
    rows = [[f"{PACES[k]}ペースになるか", pct(AU[k]["base"], 0), f"{AU[k]['nf']:.3f}", f"{AU[k]['model']:.3f}"] for k in ("high", "slow")]
    tr, te = R["train"], R["test"]
    hi_tr, hi_te = a["nf_high"]["train"], a["nf_high"]["test"]
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": f"{ty}年 1番人気の複勝率（スロー → ハイ）", "value": f"{pct(F[(ty, 'slow')])} → {pct(F[(ty, 'high')])}", "sub": "2020〜2025 年と同じ向き"},
            {"label": f"ハイペースの予測（{ty}年）", "value": f"AUC {AU['high']['model']:.2f}", "sub": "0.5 = 当てずっぽう、1.0 = 完全。ほぼ読めない"},
            {"label": "前寄り 1 頭あたり 1番人気の複勝率（オッズ調整）", "value": f"{te[0] * 100:+.1f}pt", "sub": f"{ty}年（{pfmt(te[2])}）。2020〜2025 年は {tr[0] * 100:+.1f}pt"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"「ペースが分かれば有利な馬が分かる」のうち、前半は正しく、後半（予想に使う）は難しい、というのが {ty} 年のデータで確かめた結果です。\n\n"
            f"・ペースと人気馬の成績の関係は、どの年も同じです。{ty} 年も 1 番人気の複勝率はスロー {pct(F[(ty, 'slow')])}・ミドル {pct(F[(ty, 'mid')])}・ハイ {pct(F[(ty, 'high')])} で、"
            "2020〜2025 年の各年と同じ並びでした。\n"
            f"・しかしペースはレース前にはほとんど読めません。前寄りの頭数・頭数・芝ダートから 2020〜{TRAIN_LAST} 年で予測の式を作り、{ty} 年に当てると、"
            f"ハイペースの AUC は {AU['high']['model']:.2f}、スローは {AU['slow']['model']:.2f} でした（0.5 が当てずっぽう）。\n"
            f"・記事『ハイペースになるほど、1番人気は信頼できない』の「前寄りが多いと、オッズをそろえても 1 番人気が 3 着を外しやすい」"
            f"（2020〜{TRAIN_LAST} 年 1 頭あたり {tr[0] * 100:+.2f}pt、{pfmt(tr[2])}）は、{ty} 年では {te[0] * 100:+.2f}pt（{pfmt(te[2])}、{te[3]:,} レース）で、再現しませんでした。")},
        {"type": "heading", "text": "ペースと人気馬の関係は、年が変わっても同じ"},
        a["c_fav_year"],
        a["c_win_year"],
        {"type": "heading", "text": "ペースはレース前に読めるか"},
        {"type": "table", "columns": ["当てたいこと", f"{ty}年の割合", "AUC（前寄りの頭数だけ）", "AUC（頭数・芝ダも使う）"], "rows": rows,
         "caption": f"2020〜{TRAIN_LAST} 年で式を作り、{ty} 年のレース（{a['n_test']:,}）で評価。AUC は 0.5 が当てずっぽう、1.0 が完全。"},
        a["c_nf_pace"],
        {"type": "text", "text": (
            f"前寄りの馬が多いほどハイペースになりやすい傾向は {ty} 年にもありますが（0 頭 {pct(hi_te['0頭'], 0)} → 3 頭 {pct(hi_te['3頭'], 0)}）、"
            f"4 頭以上では {pct(hi_te['4頭以上'], 0)} と 2020〜{TRAIN_LAST} 年（{pct(hi_tr['4頭以上'], 0)}）より低く、どの区分でもハイペースは 2〜3 割にとどまります。"
            "前に行く馬の頭数だけでは、ペースはほとんど決まりません。")},
        {"type": "heading", "text": "「前寄りが多いと 1 番人気が崩れる」は再現したか"},
        a["c_nf_fav"],
        {"type": "text", "text": (
            f"集計期間（2020〜{TRAIN_LAST} 年）では、前寄りが 1 頭増えるごとに 1 番人気の複勝率は、オッズ・頭数・コースをそろえても {tr[0] * 100:+.2f}pt（{pfmt(tr[2])}）下がっていました。"
            f"{ty} 年は {te[0] * 100:+.2f}pt（95% 区間 {(te[0] - 1.96 * te[1]) * 100:+.1f}〜{(te[0] + 1.96 * te[1]) * 100:+.1f}pt、{pfmt(te[2])}）で、ほぼ 0 でした。"
            f"{ty} 年は 10 か月分でレースが少なく区間は広いので「効果がない」とまでは言えませんが、市場の見落としとして馬券に使うには根拠が足りません。")},
        {"type": "heading", "text": "予想への使い方"},
        {"type": "text", "text": (
            "ペースが決まったあとの成績差は大きく、年が変わっても安定しているので、レース後の振り返り（ハイペースで崩れた先行馬・スローで届かなかった差し馬を次走で見直す）に使える可能性があります（次走の成績で確かめてはいません）。"
            "一方、レース前にペースを決め打ちして 1 番人気を外す・人気薄に広げる、といった戦略は、ペースの予測が当たらないため効果が見込めません。")},
        {"type": "link", "href": "/column/pace-advantage", "label": "ハイペースになるほど、1番人気は信頼できない"},
        {"type": "accordion", "title": "データと方法", "blocks": [
            {"type": "text", "text": (
                "・ペースの区分（ハイ・ミドル・スロー）は、前半 3F と上がり 3F の、コース標準からのずれの差で決めたもの（記事『ハイペースになるほど…』と同じ）。\n\n"
                f"・{ty} 年は 1〜10 月のうち、ペースの区分が作られているレースだけです。\n\n"
                "・予測はロジスティック回帰（前寄りの頭数・その割合・頭数・芝ダ）。2020〜2025 年で式を作り、2026 年には一切使っていません。\n\n"
                "・以前の版にあった「逃げ候補が複数ならハイペース想定」などの戦略は、上の検証でペースの予測がほとんど当たらないため外しました。")},
        ]},
    ]
    return {
        "slug": "pace-scenario",
        "title": "ペースは読めるか——2026年のデータで確かめる",
        "claim": "ペースを予測できれば有利な馬がわかる",
        "verdict": "partial",
        "summary": (f"ペースと人気馬の成績の関係は {ty} 年も同じ（1 番人気の複勝率 スロー {pct(F[(ty, 'slow')])}・ハイ {pct(F[(ty, 'high')])}）。"
                    f"ただしペースはレース前にほとんど読めず（ハイペースの AUC {AU['high']['model']:.2f}）、「前寄りが多いと 1 番人気が崩れる」も {ty} 年では再現しませんでした。"),
        "categories": ["style/front-runner"],
        "published": "2026-10-28",
        "updated": "2026-11-03",
        "dataset": {"period": f"2020〜{ty}年（{ty}年は検証用。1〜10月）", "conditions": "JRA全競馬場・芝ダート（障害除く）。ペースの区分があるレース",
                    "sample": f"{ty}年 {a['n_test']:,} レース（検証）"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    col = article(analyze(*frame()))
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
