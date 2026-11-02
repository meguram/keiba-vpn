"""めぐコラム『初ブリンカーは効く？』を実データから集計し、frontend/content/columns/first-blinkers.json を書く。

問い: ブリンカーを初めて着けたレースで、馬は走るようになるか。予想（馬券）に使えるか。条件で違うか。

データ:
  レース: data/page_reference/tables/<年>/race_result_flat.parquet（compare_series.load と同じ。2020〜2025 年の JRA 平地、取消・除外を除く）
  馬具:   data/local/modeling/training_pipeline_v4/tp32/career_eq.parquet（python -m src.scripts.data.tp32_extract_equipment。
          smartrc の馬ごとの全戦績。デビューからの各出走の blinker（0/1）。地方・障害の出走も含む）
定義:
  初装着 = 今回 blinker=1 で、それまでの全戦績で一度も 1 がない（前走のある馬）。
  装着なし = 今回も過去も 0（前走のある馬）。装着継続 = 今回 1 で過去にも 1 がある。外した初戦 = 前走 1・今回 0。
  前走 = 同じ馬の直前の JRA 平地の出走（着順・人気の比較用）。
落とし穴: ブリンカーは不振のあとに着けることが多いので、「前走より着順が上がった」だけでは平均への回帰と区別できない。
          前走の着順が同じ装着なしの馬と比べ、さらにオッズ（市場の見込み）と比べる。

使い方:
    python -m src.research.columns.first_blinkers            # 集計して記事を書く
    python -m src.research.columns.first_blinkers --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import YEARS, _p, _r, _rci, _stat, load
from src.research.columns.wind_venue import holm, mean_ci, ols, pct, pfmt

REPO = Path(__file__).resolve().parents[3]
CAREER = REPO / "data/local/modeling/training_pipeline_v4/tp32/career_eq.parquet"
OUT = REPO / "frontend/content/columns/first-blinkers.json"
PREV_BANDS = ([0, 3, 5, 9, 99], ["前走\n1〜3着", "前走\n4〜5着", "前走\n6〜9着", "前走\n10着〜"])
GROUPS = ["初装着", "装着継続", "外した初戦", "装着なし"]


def frame() -> pd.DataFrame:
    c = pd.read_parquet(CAREER, columns=["race_id", "hcode", "blinker", "year", "monthday"])
    c = c.rename(columns={"hcode": "horse_id"}).drop_duplicates(["race_id", "horse_id"])
    c["horse_id"] = c["horse_id"].astype(str)
    c["bl"] = c["blinker"].astype(int)
    c = c.sort_values(["horse_id", "year", "monthday", "race_id"])
    g = c.groupby("horse_id")["bl"]
    c["bl_before"] = g.transform(lambda s: s.shift().fillna(0).cummax())     # それまでに一度でも着けたか
    c["bl_prev"] = g.shift()
    c["nprev"] = c.groupby("horse_id").cumcount()
    d = load().merge(c[["race_id", "horse_id", "bl", "bl_before", "bl_prev", "nprev"]], on=["race_id", "horse_id"], how="left")
    d = d.sort_values(["horse_id", "date", "race_id"])
    g = d.groupby("horse_id")
    d["prev_fin"] = g["finish_position"].shift()
    d["prev_pop"] = g["popularity"].shift()
    d["grp"] = np.select(
        [(d["bl"] == 1) & (d["bl_before"] == 0) & (d["nprev"] > 0), (d["bl"] == 1) & (d["bl_before"] == 1),
         (d["bl"] == 0) & (d["bl_prev"] == 1), (d["bl"] == 0) & (d["bl_before"] == 0) & (d["nprev"] > 0)],
        GROUPS, "他")
    x = d[d["prev_fin"].notna() & (d["prev_fin"] >= 1) & (d["grp"] != "他")].copy()
    fin = x["finish_position"].where(x["finish_position"] >= 1, x["starters"])       # 中止・失格は最下位として扱う
    x["improved"] = (fin < x["prev_fin"]).astype(float)
    x["pfb"] = pd.cut(x["prev_fin"], PREV_BANDS[0], labels=PREV_BANDS[1]).astype(str)
    x["lq"] = np.log(x["odds"])
    x["key"] = x["venue"] + x["surface"] + x["distance"].astype(str)
    x["db"] = x["dist_band"].astype(str).str.split("\n").str[0]
    return x


def compare_chart(x: pd.DataFrame, metric: str, title: str, ylabel: str, caption: str) -> dict[str, Any]:
    ser = {g: {"name": g, "values": [], "n": [], "ci": []} for g in ("初装着", "装着なし")}
    ps = []
    for b in PREV_BANDS[1]:
        st = {}
        for g in ser:
            m, n, ci, v = _stat(x.loc[(x["grp"] == g) & (x["pfb"] == b), metric])
            ser[g]["values"].append(_r(m)); ser[g]["n"].append(n); ser[g]["ci"].append(_rci(ci)); st[g] = (m, v)
        p = _p(st["初装着"][1], st["装着なし"][1], st["初装着"][0], st["装着なし"][0])
        ps.append(None if p is None else float(f"{p:.3g}"))
    return {"type": "chart", "title": title, "xLabel": "前走の着順", "yLabel": ylabel, "xs": PREV_BANDS[1],
            "series": list(ser.values()), "compare": {"a": "初装着", "b": "装着なし", "p": ps}, "caption": caption}


def analyze(x: pd.DataFrame) -> dict[str, Any]:
    a: dict[str, Any] = {}
    a["groups"] = {g: {"n": int((x["grp"] == g).sum()), "top3": x.loc[x["grp"] == g, "top3"].mean(), "win": x.loc[x["grp"] == g, "win"].mean(),
                       "implied": x.loc[x["grp"] == g, "implied"].mean(), "roi": x.loc[x["grp"] == g, "ret"].mean(),
                       "roi_ci": mean_ci(x.loc[x["grp"] == g, "ret"]), "improved": x.loc[x["grp"] == g, "improved"].mean(),
                       "prev_fin": x.loc[x["grp"] == g, "prev_fin"].mean()} for g in GROUPS}
    z = x[x["grp"].isin(["初装着", "装着なし"])].copy()
    z["fb"] = (z["grp"] == "初装着").astype(float)
    a["adj_prev"] = ols("top3 ~ fb + C(pfb) + np.log(prev_pop) + C(key)", z, "fb")
    a["adj_odds_top3"] = ols("top3 ~ fb + lq + I(lq**2) + C(key)", z, "fb")
    a["adj_odds_win"] = ols("win ~ fb + lq + I(lq**2) + C(key)", z, "fb")
    rows = []
    for sf in ("芝", "ダート"):
        for db in ("短距離", "マイル", "中距離", "長距離"):
            w = z[(z["surface"] == sf) & (z["db"] == db)]
            e, se, p = ols("top3 ~ fb + lq + I(lq**2)", w, "fb")
            rows.append({"surface": sf, "db": db, "n": int(w["fb"].sum()), "e": e, "se": se, "p": p})
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
    a["cond"] = rows
    k = x[x["grp"].isin(["装着継続", "装着なし"])].assign(k=lambda w: (w["grp"] == "装着継続").astype(float))
    a["keep_win"] = ols("win ~ k + lq + I(lq**2) + C(key)", k, "k")
    a["c_improved"] = compare_chart(x, "improved", "前走着順別 前走より着順が上がった割合（初装着 vs 装着なし）", "着順が上がった割合",
                                    "前走の着順が同じ馬どうしで比べる。前走で負けた馬ほど、ブリンカーの有無にかかわらず着順は上がりやすい（平均への回帰）。＊は差が有意（p<0.05）。")
    a["c_top3"] = compare_chart(x, "top3", "前走着順別 複勝率（初装着 vs 装着なし）", "複勝率",
                                "前走の着順が同じ馬どうしで、今回の複勝率を比べる。縦線は 95% 信頼区間、＊は差が有意（p<0.05）。")
    return a


def pval(p: float) -> str:
    """表の p 値の列（「p=」を付けない）。"""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def article(a: dict[str, Any]) -> dict[str, Any]:
    G = a["groups"]
    f, n = G["初装着"], G["装着なし"]
    ep, _, pp = a["adj_prev"]
    eo, _, po = a["adj_odds_top3"]
    ew, _, pw = a["adj_odds_win"]
    ek, _, pk = a["keep_win"]
    neg = [r for r in a["cond"] if r["e"] < 0 and r["p_holm"] < 0.05]
    pos = [r for r in a["cond"] if r["e"] > 0 and r["p"] < 0.05]
    cond_rows = [[f"{r['surface']} {r['db']}", f"{r['n']:,}", f"{r['e'] * 100:+.2f}pt", f"±{1.96 * r['se'] * 100:.2f}pt", pval(r["p"]), pval(r["p_holm"])] for r in a["cond"]]
    grp_rows = [[g, f"{G[g]['n']:,}", pct(G[g]["top3"]), pct(G[g]["win"]), pct(G[g]["implied"]),
                 f"{pct(G[g]['roi'])}（{pct(G[g]['roi_ci'][0])}〜{pct(G[g]['roi_ci'][1])}）", f"{G[g]['prev_fin']:.1f}着"] for g in GROUPS]
    imp = {g: dict(zip(PREV_BANDS[1], s["values"])) for g, s in zip(("初装着", "装着なし"), a["c_improved"]["series"])}
    best, worst = PREV_BANDS[1][0], PREV_BANDS[1][-1]
    imp_p = dict(zip(PREV_BANDS[1], a["c_improved"]["compare"]["p"]))
    n_best = a["c_improved"]["series"][0]["n"][0]
    neg_txt = "、".join(f"{r['surface']}の{r['db']}（{r['e'] * 100:+.1f}pt、補正後 {pfmt(r['p_holm'])}）" for r in neg)
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "前走より着順が上がった割合", "value": f"{pct(f['improved'])} 対 {pct(n['improved'])}", "sub": "初装着 対 装着なし。ただし平均への回帰"},
            {"label": "前走の着順・人気をそろえた複勝率の差", "value": f"{ep * 100:+.2f}pt", "sub": f"初装着 − 装着なし（{pfmt(pp)}）"},
            {"label": "初装着の単勝回収率", "value": pct(f["roi"]), "sub": f"95%区間 {pct(f['roi_ci'][0])}〜{pct(f['roi_ci'][1])}。装着なし {pct(n['roi'])}"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"初めてブリンカーを着けた馬は、前走より着順を上げることが多く（{pct(f['improved'])}。装着なしの馬は {pct(n['improved'])}）、一見よく効いているように見えます。"
            f"しかしブリンカーは負けが続いたあとに着けることが多く、初装着の馬の前走は平均 {f['prev_fin']:.1f} 着と悪い（装着なしは {n['prev_fin']:.1f} 着）ため、何もしなくても着順は上がりやすい状態です（平均への回帰）。\n\n"
            f"前走の着順が同じ馬どうしで比べると、着順が上がる割合の差はずっと小さくなり、前走10着〜ではむしろ初装着のほうが低くなります（{pct(imp['初装着'][worst])} 対 {pct(imp['装着なし'][worst])}、{pfmt(imp_p[worst])}）。"
            f"前走1〜3着の馬だけは初装着のほうが着順を上げていますが（{pct(imp['初装着'][best])} 対 {pct(imp['装着なし'][best])}、{n_best} 走）、同じ帯の複勝率の差は有意ではありません。"
            f"前走の着順・人気・コースをそろえた複勝率の差は {ep * 100:+.2f}pt（{pfmt(pp)}）で、データからは、初ブリンカーで走るようになるとは言えません。\n\n"
            f"オッズの見込みと比べても、初装着の勝率は {pct(f['win'])} で見込み（{pct(f['implied'])}）とほぼ同じです（オッズ・コースをそろえて勝率 {ew * 100:+.2f}pt、{pfmt(pw)}。複勝率は {eo * 100:+.2f}pt、{pfmt(po)} とわずかに下回る傾向）。"
            "市場は「初ブリンカー」をほぼ正しく評価していて、狙い目にも消しの材料にもなりにくいと言えます。")},
        a["c_improved"],
        a["c_top3"],
        {"type": "heading", "text": "装着の状態ごとの成績"},
        {"type": "table", "columns": ["今回の装着", "出走数", "複勝率", "勝率", "オッズから見込まれる勝率", "単勝回収率（95%区間）", "前走の平均着順"], "rows": grp_rows,
         "caption": "前走（JRA 平地）のある出走。オッズから見込まれる勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。"},
        {"type": "text", "text": (
            f"装着を続けている馬は単勝回収率が {pct(G['装着継続']['roi'])} と高めですが、オッズ・コースをそろえた勝率の差は {ek * 100:+.2f}pt（{pfmt(pk)}）で有意ではありません。"
            "回収率は高配当の数回で大きく動くので、これを狙い目と読むのは早いと考えています。")},
        {"type": "heading", "text": "条件別（芝・ダート × 距離）"},
        {"type": "table", "columns": ["条件", "初装着の出走数", "複勝率の差（オッズ調整）", "95%区間の幅", "p 値", "補正後の p 値"], "rows": cond_rows,
         "caption": "初装着と装着なしの複勝率の差を、オッズ（とその 2 乗）をそろえて条件ごとに見たもの。8 条件を同時に調べているので、Holm 法で補正した p 値も示す。"},
        {"type": "text", "text": (
            ("オッズをそろえて初装着がはっきりプラスになる条件はありません。" if not pos else
             "p<0.05 でプラスになる条件は " + "、".join(f"{r['surface']}の{r['db']}" for r in pos) + " ですが、多重比較を補正すると有意ではありません。")
            + (f"逆に {neg_txt} では、初装着の馬はオッズの見込みより走っていません。" if neg else ""))},
        {"type": "accordion", "title": "データと方法・注意点", "blocks": [
            {"type": "text", "text": (
                f"・対象: {min(YEARS)}〜{max(YEARS)}年の JRA 平地で、前走（JRA 平地）のある出走。取消・除外は除き、中止・失格は 3 着以内でない（着順は最下位）として数えています。\n\n"
                "・ブリンカー: 各馬のデビューからの全戦績（地方・障害を含む）の装着記録。「初装着」はそれまで一度も着けたことがない馬の、初めて着けたレースです。\n\n"
                "・「ほかの馬具（チークピーシーズ・シャドーロールなど）」は、手元のデータに記録がないため調べていません。\n\n"
                "・ブリンカーを着ける判断には、調教での様子など、データにない理由があります。ここで分かるのは「初ブリンカーの馬が、同じ前走・同じオッズの馬と比べてどうだったか」で、装着そのものの効果を厳密に測ったものではありません。")},
        ]},
    ]
    return {
        "slug": "first-blinkers",
        "title": "初ブリンカーは効く？——前走より走るのは「平均への回帰」",
        "claim": "初ブリンカーは効果がある",
        "verdict": "not_supported",
        "summary": (f"初ブリンカーの馬は前走より着順を上げやすいものの、前走が悪い馬ほど着順が上がるのは装着なしでも同じです。"
                    f"前走の着順・人気をそろえると差は {ep * 100:+.1f}pt で、オッズの見込みともほぼ一致します。"),
        "categories": ["horse/gear"],
        "published": "2026-10-28",
        "updated": "2026-11-02",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年",
                    "conditions": "JRA全競馬場・芝ダート（障害除く）。前走のある出走。ブリンカーは各馬の全戦績の装着記録",
                    "sample": f"初装着 {f['n']:,} 走・装着なし {n['n']:,} 走"},
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
