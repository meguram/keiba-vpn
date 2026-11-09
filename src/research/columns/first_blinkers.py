"""めぐコラム『初ブリンカーは効く？』を実データから集計し、frontend/content/columns/first-blinkers.json を書く。

問い: ブリンカーを初めて着けたレースで、馬は走るようになるか。予想（馬券）に使えるか。条件で違うか。

データ:
  レース: data/page_reference/tables/<年>/race_result_flat.parquet（compare_series.load と同じ。2016〜2025 年の JRA 平地、取消・除外を除く）
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

from src.research.columns.compare_series import SMARTRC_YEARS as YEARS, _p, _r, _rci, _stat, load
from src.research.columns.race_context import GATE_POS, PRE_STYLES, joint_p, ols_fit, with_context
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
    d = with_context(load(YEARS)).merge(c[["race_id", "horse_id", "bl", "bl_before", "bl_prev", "nprev"]], on=["race_id", "horse_id"], how="left")
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


SURF = {"全体": None, "芝": "芝", "ダート": "ダート"}


def market_chart(z: pd.DataFrame, col: str, xs: list[str], title: str, x_label: str, caption: str) -> dict[str, Any]:
    """初装着の馬の、実際の勝率とオッズから見込まれる勝率を、xs の区分ごとに並べる（芝・ダートで切り替え）。"""
    def build(w: pd.DataFrame):
        ser = [{"name": n, "values": [], "n": [], "ci": []} for n in ("初装着の実際の勝率", "オッズから見込まれる勝率")]
        ps = []
        for xv in xs:
            v = w[(w["grp"] == "初装着") & (w[col].astype(str) == xv)]
            st = []
            for sr, m in zip(ser, ("win", "implied")):
                mm, n, ci, var = _stat(v[m])
                sr["values"].append(_r(mm)); sr["n"].append(n); sr["ci"].append(_rci(ci)); st.append((mm, var))
            p = _p(st[0][1], st[1][1], st[0][0], st[1][0])
            ps.append(None if p is None else float(f"{p:.3g}"))
        return ser, {"a": ser[0]["name"], "b": ser[1]["name"], "p": ps}
    ser, cmp = build(z)
    variants = {}
    for k, sf in SURF.items():
        if sf:
            s2, c2 = build(z[z["surface"] == sf])
            variants[k] = {"series": s2, "compare": c2}
    return {"type": "chart", "title": title, "xLabel": x_label, "yLabel": "勝率", "xs": xs, "series": ser, "compare": cmp,
            "filters": [{"label": "芝・ダート", "options": list(SURF)}], "variants": variants, "caption": caption}


def by_level(z: pd.DataFrame, col: str, levels: list[str]) -> list[dict[str, Any]]:
    """区分ごとに、オッズ・コースをそろえた 初装着 − 装着なし の複勝率・勝率の差（複勝率は Holm 補正つき）。"""
    rows = []
    for lv in levels:
        w = z[z[col].astype(str) == lv]
        f3 = ols_fit("top3 ~ fb + lq + I(lq**2) + C(key)", w)
        fw = ols_fit("win ~ fb + lq + I(lq**2) + C(key)", w)
        rows.append({"lv": lv, "n": int(w["fb"].sum()), "e": float(f3.params["fb"]), "se": float(f3.bse["fb"]), "p": float(f3.pvalues["fb"]),
                     "ew": float(fw.params["fb"]), "pw": float(fw.pvalues["fb"])})
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
    return rows


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
    # 芝・ダート、脚質（前走まで直近 3 走の初角の位置）、枠の位置で違うか
    a["surf"] = by_level(z, "surface", ["芝", "ダート"])
    t, dd = a["surf"]
    a["surf_diff"] = (t["e"] - dd["e"], _p(t["se"] ** 2, dd["se"] ** 2, t["e"], dd["e"]))      # 芝とダートは別のレースなので、2 つの推定は独立
    zs = z[z["style_pre"].notna()]
    a["style"] = by_level(zs, "style_pre", PRE_STYLES)
    # 同じレースに別の区分の馬がいるので、区分ごとの差の違いは 1 つのモデル（オッズの効き方も区分ごと）で検定する
    a["style_int"] = joint_p(ols_fit("top3 ~ fb*C(style_pre) + (lq + I(lq**2))*C(style_pre) + C(key)", zs), "fb:")
    a["gate"] = by_level(z, "gate_pos", GATE_POS)
    a["gate_int"] = joint_p(ols_fit("top3 ~ fb*C(gate_pos) + (lq + I(lq**2))*C(gate_pos) + C(key)", z), "fb:")
    a["c_style"] = market_chart(x, "style_pre", PRE_STYLES, "脚質（前走までの位置取り）別 初装着の勝率（実際とオッズの見込み）", "前走まで直近3走の初角の位置",
                                "初装着の馬だけ。脚質は前走まで直近 3 走の最初のコーナーの位置の平均（今回の位置取りは使わない）。芝・ダートで切り替えられる。＊は実際と見込みの差が有意（p<0.05）。")
    a["c_gate"] = market_chart(x, "gate_pos", GATE_POS, "枠の位置別 初装着の勝率（実際とオッズの見込み）", "馬番の位置（馬番÷頭数）",
                               "初装着の馬だけ。馬番の位置 =（馬番−1）÷（登録頭数−1）で、内は前 1/3、外は後 1/3。芝・ダートで切り替えられる。＊は実際と見込みの差が有意（p<0.05）。")
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
    ST = {r["lv"]: r for r in a["surf"]}
    SY = {r["lv"]: r for r in a["style"]}
    lvl_rows = [[r["lv"].replace("\n", ""), f"{r['n']:,}", f"{r['e'] * 100:+.2f}pt", f"±{1.96 * r['se'] * 100:.2f}pt", pval(r["p"]), pval(r["p_holm"]),
                 f"{r['ew'] * 100:+.2f}pt（{pfmt(r['pw'])}）"] for r in a["surf"] + a["style"] + a["gate"]]
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
        {"type": "heading", "text": "芝・ダート、脚質、枠で違うか"},
        {"type": "text", "text": (
            f"芝とダートで分けると、オッズ・コースをそろえた複勝率の差は 芝 {ST['芝']['e'] * 100:+.1f}pt（{pfmt(ST['芝']['p'])}）、"
            f"ダート {ST['ダート']['e'] * 100:+.1f}pt（{pfmt(ST['ダート']['p'])}）です（両者の違いは {pfmt(a['surf_diff'][1])}）。"
            + ("芝の初ブリンカーはオッズの見込みより 3 着以内に来ておらず、ダートではほぼ見込みどおりです。"
               if ST["芝"]["e"] < 0 and ST["芝"]["p_holm"] < 0.05 and ST["ダート"]["p"] >= 0.05 and a["surf_diff"][1] < 0.05 else "芝とダートの違いははっきりしません。")
            + f"勝率では 芝 {ST['芝']['ew'] * 100:+.1f}pt（{pfmt(ST['芝']['pw'])}）、ダート {ST['ダート']['ew'] * 100:+.1f}pt（{pfmt(ST['ダート']['pw'])}）で、どちらも見込みと有意な差はありません。\n\n"
            "脚質（前走まで直近 3 走の最初のコーナーの位置）では、複勝率の差は "
            + "、".join(f"{r['lv']} {r['e'] * 100:+.1f}pt" for r in a["style"])
            + f"、脚質で差が変わるかの検定は {pfmt(a['style_int'])} です。"
            + "".join(f"勝率では{r['lv']}の初装着が見込みを上回っていますが（{r['ew'] * 100:+.1f}pt、{pfmt(r['pw'])}）、9 区分を見た中の 1 つで、複勝率では差がありません。"
                      for r in a["style"] if r["pw"] < 0.05 and r["ew"] > 0 and r["p"] >= 0.05)
            + ("前に行く馬ほどマイナスが大きく見えますが、有意とまでは言えません。" if a["style_int"] >= 0.05 and min(SY['前寄り']['e'], SY['好位']['e']) < min(SY['中団']['e'], SY['後方']['e'])
               else "脚質で差が変わる可能性があります。" if a["style_int"] < 0.05 else "脚質による違いは見られません。")
            + "\n\n枠（馬番を頭数で割った内・中・外）では "
            + "、".join(f"{r['lv'].replace(chr(10), '')} {r['e'] * 100:+.1f}pt" for r in a["gate"])
            + f"で、枠で差が変わるかの検定は {pfmt(a['gate_int'])} です。"
            + ("枠による違いは見られません。" if a["gate_int"] >= 0.05 else "枠で差が変わる可能性があります。")
            + "\n\nまとめると、初ブリンカーが見込み以上に走る条件は、芝ダート・脚質・枠のどれで分けても見つかりませんでした。"
            + ("芝では見込みを下回るので、芝の初ブリンカーを人気ほどには信用しない、というのがデータから言える範囲です。" if ST["芝"]["e"] < 0 and ST["芝"]["p_holm"] < 0.05 else ""))},
        a["c_style"],
        a["c_gate"],
        {"type": "table", "columns": ["区分", "初装着の出走数", "複勝率の差（オッズ調整）", "95%区間の幅", "p 値", "補正後の p 値", "勝率の差（オッズ調整）"],
         "rows": lvl_rows,
         "caption": "初装着と装着なしの差を、オッズ（とその 2 乗）・コースをそろえて区分ごとに見たもの。補正後の p 値は、芝ダート（2）・脚質（4）・枠（3）のそれぞれの中で Holm 法で補正した。"},
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
                    f"前走の着順・人気をそろえると差は {ep * 100:+.1f}pt で、オッズの見込みともほぼ一致します。"
                    + (f"芝では複勝率が見込みを下回り（{ST['芝']['e'] * 100:+.1f}pt）、脚質・枠で分けても、はっきりした上積みのある条件はありません。" if ST["芝"]["p_holm"] < 0.05 and ST["芝"]["e"] < 0 else "")),
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
