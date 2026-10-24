"""tp-72 の結果 JSON（stage1 / cc / stage2 / extra）から章 HTML を作る。

python -m src.scripts.data.tp72_report --in DIR --out docs/html/modeling/chapters/edge_factor_mining.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

GROUPS = {
    "体重・追い切り・間隔": ["weight_chg", "weight_dev", "train_eval", "train_delta", "comment", "lay", "run_no"],
    "馬の属性・替わり": ["sex", "age", "seas_aff", "first_surface", "dist_chg", "jw_chg", "blinker"],
    "レース条件": ["season", "venue", "local", "surface", "dist", "field", "class", "meeting_day", "g1_week"],
    "位置取り・展開": ["draw", "style4", "pace_front", "prev_late", "prev_agari", "pincer_d1"],
    "馬場": ["cushion", "dirt_moist", "going", "cush_pref"],
    "関係者": ["jockey_rate", "trainer_rate", "chg_jockey", "local_aff", "jockey_top", "jockey_cls_chg"],
    "血統": ["sire_group", "damsire_group"],
    "前走": ["prev_fin", "prev_pop"],
}
G_OF = {d: g for g, ds in GROUPS.items() for d in ds}


def esc(s) -> str:
    return html.escape(str(s))


def f(x, nd=2, pct=False, sign=False) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    v = x * 100 if pct else x
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def ci(v) -> str:
    if not v or v[0] is None or (isinstance(v[0], float) and math.isnan(v[0])):
        return "—"
    return f"{v[0] * 100:.0f}%（{v[1] * 100:.0f}〜{v[2] * 100:.0f}）"


def table(head: list[str], rows: list[list[str]], num_from: int = 1) -> str:
    h = "".join("<th" + (' class="n"' if i >= num_from else '') + f">{esc(c)}</th>" for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f"<td{' class=\"n\"' if i >= num_from else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def build(d: Path) -> str:
    s1 = json.loads((d / "stage1.json").read_text(encoding="utf-8"))
    cc = json.loads((d / "cc.json").read_text(encoding="utf-8"))
    s2 = json.loads((d / "stage2.json").read_text(encoding="utf-8"))
    ex = json.loads((d / "extra.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "mining_meta.json").read_text(encoding="utf-8"))
    cc1 = json.loads((d / "cc_run1.json").read_text(encoding="utf-8")) if (d / "cc_run1.json").exists() else None
    rows = s1["rows"]
    by = {r["name"]: r for r in rows}
    sm = s1["summary"]

    # ---- 家族別
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    FN = {"F1_sex_season": "性別×季節", "F2_sex_month": "性別×月", "F3_female_summer": "夏の牝馬の細分", "F4_age_season": "年齢×季節（上がり馬・休み明け）", "F5_sire_cond": "種牡馬×季節・クッション値",
          "F6_horse_season": "馬ごとの同季節の実績", "F7_weight": "馬体重（寒い時期）", "F9_local": "騎手・調教師のローカル場実績", "F10_meeting": "開催の節目（初週・最終週）", "F11_g1": "G1 開催日の前後",
          "F12_cushion": "クッション値・含水率 × 脚質 × 枠", "F13_pace": "展開（隊列）", "F14_prev": "前走の不利・人気と着順の乖離", "F15_misc": "初経験・距離・斤量・頭数・間隔",
          "F16_pincer": "逃げはさみ", "F17_season_style": "季節×馬場×脚質", "F18_combo": "体重・追い切り・間隔の同時発生", "F19_blinker_style": "ブリンカー×脚質", "F20_draw_style": "枠×脚質（コース別含む）",
          "F21_sire_jockey": "種牡馬×騎手", "F22_damsire_jockey": "母父×騎手", "F23_sire_jockey_chg": "種牡馬×騎手の格の変化"}
    frows = []
    for k in sorted(fam, key=lambda x: int(x.split("_")[0][1:])):
        xs = fam[k]
        t = [x for x in xs if x["z_min_win"] is not None]
        zw = sum(x["z_min_win"] > 1.645 for x in t)
        zp = sum(x["z_min_place"] > 1.645 for x in t)
        best = max(t, key=lambda x: max(x["z_min_win"], x["z_min_place"])) if t else None
        frows.append([esc(FN.get(k, k)), str(len(xs)), str(len(t)), f"{zw} / {zp}", f"{0.05 * len(t):.1f}", esc(best["name"]) if best else "—", f(max(best["z_min_win"], best["z_min_place"])) if best else "—",
                      str(sum(x["race_level"] for x in xs))])
    fam_table = table(["系統", "型の数", "検定できた数", "z>1.645（勝ち / 複勝）", "偶然の期待数（各）", "最大の型", "最大 z", "レース単位の型"], frows, 1)

    # ---- 通過した型
    surv = []
    for r in rows:
        for o, s in (("win", "1"), ("place", "3")):
            if r["q_" + o] < 0.10:
                st = r["stats"]
                row = [esc(r["name"]), "勝ち" if o == "win" else "複勝", f(r["q_" + o], 3), f"{st['disc']['n']:,}", f(st["disc"]["mean_m" + s], 2, pct=True, sign=True), f"{f(st['disc']['z_m' + s],1)} / {f(st['disc']['z_k' + s],1)}",
                       " / ".join(f(st[y].get("z_m" + s, float('nan')), 1) for y in ("y2022", "y2023", "y2024")),
                       f"{f(st['y2025'].get('z_m' + s, float('nan')),1)} / {f(st['y2025'].get('z_k' + s, float('nan')),1)}", f(st["y2026"].get("z_m" + s, float('nan')), 1),
                       f"{st['disc']['roi_' + ('win' if o == 'win' else 'place')] * 100:.0f}% → {st['y2025']['roi_' + ('win' if o == 'win' else 'place')] * 100:.0f}%"]
                surv.append(row)
    surv_t = table(["型", "対象", "q（BH）", "発見 n", "モデル比の差 (pt)", "発見 z（モデル/市場）", "年別 z（22/23/24）", "2025 z（モデル/市場）", "2026 z", "回収率 発見→2025"], surv, 1)

    # ---- 性別×季節
    sx = []
    for r in rows:
        if r["family"] == "F1_sex_season":
            st = r["stats"]
            sx.append([esc(r["name"]), f"{st['disc']['n']:,}", f(st["disc"]["mean_m1"], 2, True, True), f(st["disc"]["z_m1"], 1), f(st["disc"]["mean_m3"], 2, True, True), f(st["disc"]["z_m3"], 1),
                       f(st["y2025"]["z_m1"], 1), f(st["y2025"]["z_m3"], 1), f(st["y2026"]["z_m1"], 1), f"{st['disc']['roi_win'] * 100:.0f}% / {st['disc']['roi_place'] * 100:.0f}%", f"{st['y2025']['roi_win'] * 100:.0f}% / {st['y2025']['roi_place'] * 100:.0f}%"])
    sx_t = table(["性別_季節", "発見 n", "勝ち 実績−モデル (pt)", "z", "複勝 実績−モデル (pt)", "z", "2025 勝ち z", "2025 複勝 z", "2026 勝ち z", "回収率 単勝/複勝（発見）", "同（2025）"], sx, 1)
    fsr = []
    for nm in ["female_summer", "female_summer_a3", "female_summer_a4p", "female_summer_turf", "female_summer_dirt", "female_summer_local", "female_summer_central",
               "female_summer_a3_turf_local", "female_summer_a3_dirt_local", "female_summer_a4p_turf_local", "female_summer_a4p_dirt_local", "female_summer_a3_turf_central", "female_summer_a3_dirt_central",
               "female_summer_a4p_turf_central", "female_summer_a4p_dirt_central"]:
        r = by.get(nm)
        if not r:
            continue
        st = r["stats"]
        fsr.append([esc(nm), f"{st['disc']['n']:,}", f(st["disc"].get("mean_m1"), 2, True, True), f(st["disc"].get("z_m1"), 1), f(st["disc"].get("mean_m3"), 2, True, True), f(st["disc"].get("z_m3"), 1),
                    f(st["y2025"].get("z_m1"), 1), f(st["y2025"].get("z_m3"), 1), f(st["y2026"].get("z_m1"), 1), (f(r["q_win"], 2) + " / " + f(r["q_place"], 2)) if r["z_min_win"] is not None else "小さい"])
    fsr_t = table(["型", "発見 n", "勝ち pt", "z", "複勝 pt", "z", "2025 勝ち z", "2025 複勝 z", "2026 勝ち z", "q（勝ち/複勝）"], fsr, 1)

    # ---- クッション
    cu = ex["A_cushion"]
    cu_rows = []
    lab = {0: "低（柔らかい/湿）", 1: "中", 2: "高（硬い/乾）"}
    for r in cu["turf_cushion"]:
        cu_rows.append([f"芝クッション値 {lab[r['bin']]}", {"nige": "逃げ", "senko": "先行", "sashi": "差し", "oikomi": "追込"}[r["style"]], f"{r['n']:,}", f(r["win_rate"], 1, True), f(r["p_win"], 1, True), f(r["resid_pt"], 2, False, True),
                        " / ".join(f(v, 1, False, True) for v in r["resid_by_year_pt"].values()), f"{r['roi_win'] * 100:.0f}%"])
    cu_t = table(["区分", "脚質", "n（2022-25）", "勝率", "モデル勝率", "実績−モデル (pt)", "年別 (22/23/24/25)", "単勝回収率"], cu_rows, 2)
    dm_rows = []
    for r in cu["dirt_moisture"]:
        dm_rows.append([f"ダート含水率 {lab[r['bin']].replace('柔らかい/湿','乾').replace('硬い/乾','湿')}", {"nige": "逃げ", "senko": "先行", "sashi": "差し", "oikomi": "追込"}[r["style"]], f"{r['n']:,}", f(r["win_rate"], 1, True), f(r["p_win"], 1, True), f(r["resid_pt"], 2, False, True), f"{r['roi_win'] * 100:.0f}%"])
    dm_t = table(["区分", "脚質", "n", "勝率", "モデル勝率", "実績−モデル (pt)", "単勝回収率"], dm_rows, 2)
    f12 = sorted([r for r in fam["F12_cushion"] if r["z_min_win"] is not None], key=lambda x: -max(x["z_min_win"], x["z_min_place"]))[:8]
    f12_t = table(["型", "発見 n", "z勝ち", "z複勝", "2025 勝ち z（モデル/市場）", "2025 n", "回収率（発見 単/複）"],
                  [[esc(r["name"]), f"{r['stats']['disc']['n']:,}", f(r["z_min_win"], 1), f(r["z_min_place"], 1), f(r["stats"]["y2025"].get("z_m1"), 1) + " / " + f(r["stats"]["y2025"].get("z_k1"), 1), f"{r['stats']['y2025']['n']:,}",
                    f"{r['stats']['disc']['roi_win'] * 100:.0f}% / {r['stats']['disc']['roi_place'] * 100:.0f}%"] for r in f12], 1)

    # ---- 機構
    mech = ex["B_mechanism"]
    mrows = []
    for k, v in mech.items():
        for x in v:
            mrows.append([esc(k), {"nige": "逃げ", "senko": "先行", "sashi": "差し", "oikomi": "追込"}[x["style"]], f"{x['n']:,}", f(x["pos1_chg_sel"], 3, False, True), f(x["pos1_chg_ctl"], 3, False, True), f(x["diff"], 3, False, True), f(x["z"], 1)])
    mech_t = table(["型", "脚質", "n", "初角位置の変化（該当）", "同（対照）", "差", "z"], mrows, 2)
    def best_of(fam_key, n=6):
        xs = sorted([r for r in fam[fam_key] if r["z_min_win"] is not None], key=lambda x: -max(x["z_min_win"], x["z_min_place"]))[:n]
        return table(["型", "発見 n", "z勝ち", "z複勝", "勝ち 実績−モデル pt", "2025 z（モデル/市場, 勝ち）", "回収率 単/複（発見）"],
                     [[esc(r["name"]), f"{r['stats']['disc']['n']:,}", f(r["z_min_win"], 1), f(r["z_min_place"], 1), f(r["stats"]["disc"].get("mean_m1"), 2, True, True),
                       f(r["stats"]["y2025"].get("z_m1"), 1) + " / " + f(r["stats"]["y2025"].get("z_k1"), 1), f"{r['stats']['disc']['roi_win'] * 100:.0f}% / {r['stats']['disc']['roi_place'] * 100:.0f}%"] for r in xs], 1)

    # ---- 穴馬
    T = cc["tests"]
    ccs = cc["summary"]
    topc = sorted([t for t in T if t["p"] < 1.0], key=lambda t: -t["disc"]["z"])[:12]
    def roi_s(t, tag, k):
        v = (t.get("roi") or {}).get(tag, {}).get(k)
        return ci(v) if v else "—"
    cc_t = table(["仕様（穴馬|好走）", "記述子=水準", "q", "発見 n", "z", "好走率 − 同条件平均 (pt)", "2025 z", "2026 z", "2025 回収率 単勝", "複勝", "ワイド（1番人気との1点）"],
                 [[esc(t["spec"]), esc(f"{t['desc']}={t['level']}"), f(t["q"], 3), f"{t['disc']['n']:,}", f(t["disc"]["z"], 1), f(t["disc"]["lift_pt"], 2, False, True), f(t["y2025"]["z"], 1), f(t["y2026"]["z"], 1),
                   roi_s(t, "y2025", "win"), roi_s(t, "y2025", "place"), roi_s(t, "y2025", "wide_fav")] for t in topc], 3)
    lg = cc.get("logit", {})
    lg_pos = ", ".join(f"{esc(n)}（×{v:.2f}）" for n, v in lg.get("top_pos", [])[:10])
    lg_neg = ", ".join(f"{esc(n)}（×{v:.2f}）" for n, v in lg.get("top_neg", [])[:6])
    c1 = ""
    if cc1:
        t1 = [t for t in cc1["tests"] if t["q"] < 0.10]
        c1 = f"<p>対照を『同年・同オッズ帯』だけで取った初回の実行では、少頭数（10 頭以下）の穴馬の好走が z=7 で出た（q&lt;0.001、{len(t1)} 本が通過）が、頭数を揃えていなかったための取り違え（少頭数は同じオッズでも 3 着以内に入りやすい）。仕様の『同じレースの同じ人気帯』に合わせて頭数を揃えると、通過は 0 本になった。</p>"

    # ---- ペア走査の地図
    P = s2["pairs"]
    import pandas as pd

    dfp = pd.read_parquet(Path(d) / "stage2_pairs.parquet")
    dfp["gi"] = dfp["i"].map(G_OF)
    dfp["gj"] = dfp["j"].map(G_OF)
    gm = []
    gnames = list(GROUPS)
    for a in gnames:
        row = [esc(a)]
        for b in gnames:
            sub = dfp[((dfp["gi"] == a) & (dfp["gj"] == b)) | ((dfp["gi"] == b) & (dfp["gj"] == a))]
            sub = sub[sub["target"] == "top3"]
            row.append(f"{sub['z_adj'].max():.1f}" if len(sub) else "—")
        gm.append(row)
    gm_t = table(["複勝残差：群×群の最大 z（補正後）"] + [esc(g) for g in gnames], gm, 1)
    pair_rows = []
    for p in P[:10]:
        pair_rows.append([esc(p["target"]), esc(p["i"]), esc(p["j"]), f"{p['n']:,}", f(p["mean_gain"] * 1e5, 1), f(p["z"], 2), f(p["z_adj"], 2), f(p["q"], 3)])
    pair_t = table(["目的変数", "記述子 i", "記述子 j", "n", "ΔLL（外側の年, ×1e-5）", "z", "z（経験的帰無で補正）", "q"], pair_rows, 3)
    wl = dfp[(dfp["i"].isin(["weight_chg", "weight_dev", "train_eval", "train_delta", "lay"])) & (dfp["j"].isin(["weight_chg", "weight_dev", "train_eval", "train_delta", "lay"]))]
    wl_s = ", ".join(f"{esc(r['i'])}×{esc(r['j'])}({r['target']}) z_adj={r['z_adj']:.2f}" for _, r in wl.sort_values("z_adj", ascending=False).head(4).iterrows())
    sj = dfp[((dfp["i"].isin(["sire_group", "damsire_group"])) & (dfp["j"].isin(["jockey_top", "jockey_cls_chg", "jockey_rate"]))) | ((dfp["j"].isin(["sire_group", "damsire_group"])) & (dfp["i"].isin(["jockey_top", "jockey_cls_chg", "jockey_rate"])))]
    sj_s = ", ".join(f"{esc(r['i'])}×{esc(r['j'])}({r['target']}) z_adj={r['z_adj']:.2f}" for _, r in sj.sort_values("z_adj", ascending=False).head(4).iterrows())
    cells = s2["cells"][:10]
    cell_t = table(["目的変数", "ペア", "セル", "発見 δ（logit）", "発見 n", "2025 n", "2025 z（モデル/市場）", "2025 単勝回収率"],
                   [[esc(c["target"]), esc(f"{c['i']}×{c['j']}"), esc(c["cell"]), f(c["delta"], 3), f"{c['n_disc']:,}", f"{c['y2025']['n']:,}", f(c["y2025"].get("z_m1"), 1) + " / " + f(c["y2025"].get("z_k1"), 1), ci(c["y2025"].get("roi_win"))] for c in cells], 2)

    # ---- 特徴量化の見込み
    fe = ex["C_feasibility"]
    fe_t = table(["シフトの粒度×目的", "2025 ΔLL（負が改善）", "2026 ΔLL（記述）"], [[esc(k), f(v["2025"]["delta"], 5, False, True), f(v["2026"]["delta"], 5, False, True)] for k, v in fe.items()], 1)
    rb = ex["D_robust"]
    rb_t = table(["型", "発見 z（V5 OOF・勝ち）", "発見 z（V5 OOF・複勝）", "年別 勝ち z（22/23/24）"],
                 [[esc(k), f(v["disc"].get("z_v5_win"), 1), f(v["disc"].get("z_v5_top3"), 1), " / ".join(f(v[y].get("z_v5_win"), 1) for y in ("y2022", "y2023", "y2024"))] for k, v in rb.items() if k in ("male_spring", "male_winter", "female_summer", "female_summer_a3", "male_m01")], 1)

    body = f"""
<div class="note"><b>要点（先に答え）.</b>
<ul>
<li><b>今回の探索の規模</b>: 買う要素の候補 <b>{sm['n_candidates']} 型（検定 {sm['n_tests']} 本、うち検定できたのは {sm['n_testable']} 型×勝ち・複勝）</b>、記述子 43 個のペア走査 <b>903 ペア × 3 目的変数</b>、穴馬の共通要素 <b>{ccs['n_tests']} 本</b>を、結果を見る前に登録して（コミット 3eea55b・d1bb162・98eb011）検定した。発見は 2020〜2024（モデルの OOF は 2022〜）、確認は 2025 を 1 回、2026 は記述のみ。</li>
<li><b>『期待値が取れる要素』は見つからなかった</b>。回収率の区間が 100% を超える型は 0。第 1 段で多重比較（BH）を通ったのは {sm['n_q_lt_0.10']} 本（実質 4 型）だけで、すべて<b>性別×季節</b>（牡: 春・冬が過小評価・夏が過大評価、牝: 夏が過小評価・春冬が過大評価）。経験的帰無で補正すると 0 本。</li>
<li><b>唯一、2025 でも再現したのは『夏の牝馬』</b>。モデルにも市場にも過小評価されている（勝ち +0.55pt・複勝 +0.85pt、勝率の相対で約 +7%）。ただし単勝回収率は 77%（発見）→ 74%（2025）で、基準の約 70% から 4〜7pt 上がる程度。100% には遠く、<b>買い目の根拠にはならない</b>（特徴量化しても 2025 の対数損失は勝ち −0.00003・複勝 −0.00018 しか下がらない）。</li>
<li><b>クッション値の深掘り</b>: モデルはクッション値（芝）・含水率（ダート）× 脚質 × 枠で ±0.5pt 以内に較正済みで、年別の符号も揃わない。クッション値の水準が脚質・枠の有利をずらす、という材料は買い目の優位にならなかった（F12 の {len(fam['F12_cushion'])} 型で偶然を超える数は出ない）。</li>
<li><b>逃げはさみ</b>: 理論とは逆に、両隣が前走で先行の馬は初角が<b>後ろに下がる</b>（対照より +0.04〜0.045 後方、z 5〜9）。成績の優位もなし。<b>ブリンカー初着用</b>は理論どおり初角が前へ出る（−0.055 前後、z −6〜−8）が、勝率・複勝率は市場とモデルが織り込み済み。</li>
<li><b>穴馬の共通要素</b>: 頭数を揃えた対照で、好走の確率を上げる要素は {ccs['n_q_lt_0.10']} 本。z&gt;1.645 の数（{ccs['n_z_gt_1.645']}）は偶然の期待（{ccs['expected_false_pos']:.0f}）と同じ。</li>
<li><b>第 2 段（ペア走査）</b>: 補正後に通るペアは 0。分野ごとの寄与マップで、体重×追い切り×間隔も、血統×騎手も、上位に出ない。</li>
<li><b>凍結</b>: 買い目の規則として凍結する候補はなし。夏の牝馬だけ『確率の補正項』として記録（<code>config/tp72_frozen_rule.json</code>）し、新しいレースで同じ向きに出続けるかを見る。</li>
</ul></div>

<h2>1. 既に試した要素の台帳と、今回の位置づけ</h2>
<p class="muted">決定ログ・特徴量調査の総覧（<a href="feature_investigation_catalog.html">feature_investigation_catalog</a>）・やり残しの棚卸し（<a href="feature_gap_audit.html">feature_gap_audit</a>）・各章から作成。『特徴量として足したときの増分』は多くの章にあるが、『状況別の、モデルと市場に対する残差と回収率』は初。</p>
<table><tr><th>領域</th><th>既存の扱い（採否・根拠）</th><th>今回</th></tr>
<tr><td>クッション値・含水率</td><td>tp-16 / 16b で特徴量として評価（訂正後のデータでも）、基準未満で不採用（§13・§27）。脚質・枠との交互作用や回収率は未検証</td><td>F12（73 型）・補助 A</td></tr>
<tr><td>季節性</td><td>tp-16 CAL（年周期 sin/cos）+0.00004 で不採用。性別×季節・夏の牝馬は未検証</td><td>F1〜F4, F6, F17</td></tr>
<tr><td>ブリンカー</td><td>tp-32 EQ 群 +0.00007 で不採用。馬具の着用と SmartRC の戦績は 100% 一致。脚質との関係・位置取りの変化は未検証</td><td>F19・補助 B</td></tr>
<tr><td>体重・追い切り・間隔</td><td>体重 tp-32 / 成長曲線 tp-39（境界）、追い切り tp-19 TL_A（採用）・tp-28 / 37（不採用）・tp-40 CY（V5 採用）、間隔 tp-34（採用ゼロ）。いずれも単独。同時発生は未検証</td><td>F18（54 型）</td></tr>
<tr><td>枠・脚質・展開</td><td>tp-25（位置取り前段）不採用、tp-31 コース特注（東京芝 1700〜2000 で逃げ先行が期待以上ほか）、tp-54 追走難度 v2、tp-61 展開タイプ。コース別の枠×脚質の残差・回収率は未検証</td><td>F13・F20（132 型）</td></tr>
<tr><td>逃げはさみ（隣の馬の前走位置取り）</td><td>該当なし</td><td>F16（63 型）・補助 B</td></tr>
<tr><td>騎手・調教師、乗り替わり</td><td>tp-59（回収率が再現せず・100% 超の関係者なし）、jockey_switch_ev（有力↔非有力の乗り替わり）、tp-32 JOCKEY 群</td><td>F9（ローカル場実績）のみ。乗り替わりは重複を避けた</td></tr>
<tr><td>血統×条件、血統×騎手</td><td>tp-38 / 60 / pedigree_roi（規則 198 本、Reality Check p=1.00）。血統グループ×騎手の相性そのものは未検証</td><td>F5・F21〜F23・ペア走査</td></tr>
<tr><td>去勢明け</td><td>去勢明けの章（§60）が担当</td><td>除外</td></tr>
<tr><td>掛け合わせの網羅</td><td>tp-58（V5 の列の積・差・比 44,631 候補、NN、残差ブースティング）で上乗せなし</td><td>記述子 43 のペアを残差で走査（第 2 段）</td></tr>
<tr><td>穴馬の好走の共通要素</td><td>tp-56 の穴軸 3 連複、mark_rules の ★（10〜30 倍）、betting_edge_search の『大穴依存』の節はあるが、共通要素の分析は<b>未実施</b></td><td>第 C 部</td></tr>
</table>

<h2>2. 方法（結果を見る前に固定したもの）</h2>
<ul>
<li><b>土台の表</b>: 1 出走 1 行（2020〜2026 年、{meta['rows']:,} 行）。モデルは tp-59 の master にある V4 の予測（2022〜2024 は OOF、2025 は保留年、2026 は予測）。V5 の OOF（LightGBM 単体、2022〜2024）で通過した型を再検定して、結論が V4 に依存しないことを確認した（補助 D）。V6e の OOF はこの表に無く、配備前のため使っていない。市場は確定単勝オッズの正規化確率と Harville の複勝確率。市場オッズは分析にだけ使い、特徴量にしない。</li>
<li><b>履歴由来の列</b>はすべて as-of（その日より前の出走・累積のみ。騎手・調教師・種牡馬は merge_asof の厳密な前日まで）。クッション値・含水率の区分は 2020〜2021 年の三分位（芝のクッション値 {meta['cuts']['cushion_turf']}、ダート含水率 {meta['cuts']['dirt_moist']}）。種牡馬上位15・母父上位10・有力騎手12 は 2020〜2021 年の出走・勝利数で固定。</li>
<li><b>型の検定</b>: 型ごとに勝ち・複勝の 2 アウトカムを片側で検定。統計量はレース単位のクラスタ頑健な z（該当馬の残差のレース内合計 ÷ その二乗和の平方根）。<code>min(z_モデル, z_市場)</code> が正で大きいものだけが通る（交差検定の考え方）。BH はブロックごと（F1〜F20 / 血統×騎手）に補正し、全体で補正した q も併記（結論は同じ）。最小件数は発見期間で 300 頭・200 レース。回収は『同年・同単勝オッズ帯・同頭数の平均払戻』との差を z_r として併記。</li>
<li><b>出走馬全員が当たるレース単位の型</b>（開催の節目・G1 週・頭数など 20 型）は、勝ち（複勝）の確率の合計が 1（3）に決まっているため残差の検定が成り立たない。初回の実行でこの型が z=√2 などの偽の値で上位に出たので、これらは回収の超過 z_r だけで検定するよう直した（初回の出力は保存してある）。</li>
<li><b>確認</b>: 通過した型と上位を 2025 で 1 回。2026 は記述のみ。年別（22/23/24）の符号で LOYO 的な再現を見る。</li>
</ul>

<h2>3. 第 1 段の結果（個別の型）</h2>
<p>系統ごとの要約。『偶然の期待数』は検定できた型の 5% で、系統ごとに z&gt;1.645 の数と比べる。</p>
{fam_table}
<p class="muted">全体では、z&gt;1.645 が勝ち {sm['n_z_gt_1.645_win']}・複勝 {sm['n_z_gt_1.645_place']}（偶然の期待は各 {sm['expected_false_pos_at_z1.645']/2:.0f}）。z の MAD 由来の標準偏差は勝ち {sm['z_min_win_mad_sd']:.2f}・複勝 {sm['z_min_place_mad_sd']:.2f} と 1 より広く（型の重なりと較正のずれ）、この経験的帰無で補正すると通過は 0 本。</p>
<h3>多重比較を通った型と、2025 での確認</h3>
{surv_t}
<p class="muted">『モデル比の差』は実績−モデル確率（pt）。z は（モデル / 市場）。2025 の確認で再現した（z&gt;1.28 が両方）のは female_summer（勝ち・複勝とも）と male_winter（勝ちのみ）。male_spring は勝ちで再現せず、male_m01 は 2025 で逆向き。</p>

<h3>性別 × 季節（12 区分）</h3>
{sx_t}
<p>符号は規則的で、牡は春・冬が過小評価・夏が過大評価、牝は夏が過小評価・春冬が過大評価（鏡像）。モデルにも市場にも出る。ただし大きさは 0.4〜0.9pt（勝率の相対で 5〜10%）で、回収率は単勝 63〜77%。鏡像（過大評価側）を避ける効果を含めても、買う根拠になる大きさではない。</p>
<h3>夏の牝馬の細分（古馬／3歳、芝／ダート、ローカル／中央）</h3>
{fsr_t}
<p>効果の中心は<b>3 歳の夏の牝馬</b>（勝ち +0.67pt、z 3.4。2025 も z 2.1 で再現）で、古馬の夏の牝馬は勝ちがほぼ 0（+0.08pt、z 0.2）。ローカル場に多く（9,026 頭）、芝では複勝、ダートでは勝ちに出やすいが、細分の多くは件数が小さく BH を通らない。2026 は 3 歳が逆向き（z −0.7）。</p>

<h2>4. クッション値・含水率の深掘り</h2>
<p>tp-16b で特徴量としては不採用（訂正後データ）。ここでは『状況別の残差と回収率』を、クッション値（芝、2020〜2021 年の三分位）・ダート含水率 × 脚質（as-of の初角位置: 逃げ≤0.10 / 先行≤0.30 / 差し≤0.60 / 追込）× 枠で見た。</p>
{cu_t}
<p class="muted">回収率が 100% を超えるセル（ダート含水率 中×逃げ 110%、F12 の dirt_moist1_front_outer = 含水率 中×逃げ先行×外枠 123%）は、いずれも区間が 100% を跨ぎ（発見 82〜174%、2025 は 48〜224%）、残差も小さい。後者の 2026 は 47%（63 頭）。年別の符号が揃うのは『芝クッション値 高×逃げ』（22/23/24/25 とも正、n=836、単勝回収率 74%）と『芝クッション値 中×追込』（4 年とも正、+0.45pt、回収率 67%）程度で、いずれも検定を通らない大きさ。</p>
{dm_t}
<p>F12 の上位の型（全 {len(fam['F12_cushion'])} 型）:</p>
{f12_t}
<p>結論: クッション値・含水率が脚質や枠の有利をずらすことはあるが、<b>その分はモデルの勝率にも市場にも織り込まれており</b>、残差は ±0.5pt 以内で年ごとに符号が入れ替わる。horse ごとの『硬い馬場が得意』（as-of の着順パーセンタイルの差）や、種牡馬×クッション値の as-of 超過複勝率も、通過しなかった（F12・F5）。</p>

<h2>5. 逃げはさみとブリンカーの機構</h2>
<p>『初角位置の変化』= 今回の初角の相対位置 − その馬の as-of の普段の位置（負なら普段より前）。対照は同じ脚質で該当しない馬。結果由来なので機構の確認にだけ使う。</p>
{mech_t}
<ul>
<li><b>逃げはさみ（d1: 前走の初角 3 番手以内、d2: 前走の 4 角 3 番手以内、d3: 前走の初角 1 番手）</b>: 両隣が先行でも片隣だけでも、該当馬は<b>普段より後ろ</b>から入る（理論と逆向き）。成績の型（F16 の {len(fam['F16_pincer'])} 型、スタイル・芝ダ・距離・頭数・クッション値との交互作用を含む）は 1 本も通過せず、z&gt;1.645 の数は偶然の期待以下。</li>
<li><b>ブリンカー</b>: 初着用は先行・差し・追込のいずれも初角が前に出る（約 −0.055）、外した初戦は後ろに下がる（+0.02〜0.09）。理論どおりの機構だが、勝率・複勝率の残差は出ない（F19 の最大 z は 2.7、q&gt;0.2）ので、市場とモデルは着用を織り込んでいる。</li>
</ul>
<h3>F16・F19・F20 の上位（参考）</h3>
{best_of('F16_pincer', 5)}
{best_of('F19_blinker_style', 5)}
{best_of('F20_draw_style', 6)}

<h2>6. 体重・追い切り・間隔の同時発生</h2>
<p>休み明け（90 日・180 日）×体重（前走比 ±10kg、自己平均からのずれ ±8kg）×追い切り評価（A/B 以上、前走より上）、および 3 歳春〜夏の増加×追い切りの上昇、厩舎コメントの前向き語を、54 型で検定。</p>
{best_of('F18_combo', 8)}
<p>結論: 通過なし（z&gt;1.645 は複勝で 1 型、偶然の期待 2.5）。</p>

<h2>7. 血統 × 騎手</h2>
<ul>
<li>種牡馬上位15 × 有力騎手12（180 セル）、母父上位10 × 有力騎手12（120 セル）は、発見期間（3 年）で 1 セルあたり 300 頭に届かず、<b>個別セルは検定できなかった</b>（F21・F22 は 0 型）。種牡馬 × 騎手の格の変化（F23）は 8 型が検定でき、通過なし。</li>
<li>階層縮約（主効果 → セル、リッジ λ=20）の枠組みでペア走査に入れた結果: {sj_s}（いずれも補正後に通らない）。</li>
</ul>

<h2>8. 穴馬が好走するときの共通要素（第 C 部）</h2>
<p>穴馬 = 確定人気 9 番以下 / 単勝 12 倍以上 / 20 倍以上。好走 = 3 着以内、または 1 着のみ。対照 = 同じ年・同じ単勝オッズ帯・同じ頭数の穴馬の平均好走率。記述子は 43 個（市場オッズ・人気は含めない）の各水準について、『好走 − 平均』のレース単位クラスタ z を片側で検定し BH。</p>
<p>結果: 検定できた {ccs['n_testable']} 本のうち z&gt;1.645 は {ccs['n_z_gt_1.645']} 本（偶然の期待 {ccs['expected_false_pos']:.0f}）、補正後に通るのは {ccs['n_q_lt_0.10']} 本。上位を示す（2025 は確認 1 回、2026 は記述）:</p>
{cc_t}
{c1}
<p class="muted">多変量ロジスティック（L2、発見期間・単勝 12 倍以上・3 着以内）の係数の大きい順: {lg_pos}。小さい順: {lg_neg}。（参考であり、検定は上の表。）</p>
<p>結論: 穴馬が好走するときに共通する要素は、単独でも、事前に定義した 43 個の範囲では見つからない。上位の水準でも、複勝・ワイドの回収率は 100% を大きく下回る（95% 区間の上限も 100% 未満が大半）。</p>

<h2>9. 第 2 段: 記述子ペアの系統走査（学習を伴わない集計版）</h2>
<ul>
<li>記述子 43 個（水準 2〜16）→ <b>903 ペア</b>。目的変数 3（勝ち残差・複勝残差・穴馬の複勝残差）で 2,709 本。ユーザー指示により、GBM・木の学習を伴う走査は行わず、<b>セル集計</b>（主効果 → セルの logit シフト、リッジ縮約）で、発見期間 2022〜2024 を 1 年ずつ外した予測で ΔLL を測る（計算量は bincount のみ、約 4 分）。</li>
<li>帰無の分布: 外側年の ΔLL は平均が負（過学習）で、z の中央値 −1.4、MAD 由来の SD 1.3〜1.4。経験的に補正し BH しても、<b>q&lt;0.10 のペアは 0</b>（補正なしでも 0）。</li>
</ul>
{pair_t}
<h3>分野ごとの寄与マップ（複勝残差、群×群の最大 z）</h3>
{gm_t}
<p class="muted">体重・追い切り・間隔どうしのペアの最大: {wl_s}。血統×騎手の最大: {sj_s}。いずれも 2 を超える程度で、903 ペアの最大値として偶然の範囲。</p>
<h3>参考: 上位ペアのセルの 2025 での確認</h3>
{cell_t}

<h2>10. 特徴量化の見込みと頑健性</h2>
<p>性別×季節（12 区分）または性別×月の logit シフトを発見期間で当てはめ、2025 に当てたときの対数損失の変化（負が改善）。採用基準の 0.0002 に届かない。</p>
{fe_t}
<p>V5 の OOF（LightGBM 単体）に取り替えても、通過した型の z は同水準で、V4 に依存した結果ではない。</p>
{rb_t}

<h2>11. 凍結と結論</h2>
<ul>
<li><b>期待値が取れる要素は見つからなかった。</b>回収率の区間が 100% を超える型・セル・ペアはなし。</li>
<li><b>凍結（記録のみ・買い目にしない）</b>: 『夏（7〜8 月）の牝馬』は、モデルにも市場にも +0.55pt（勝ち）・+0.85pt（複勝）過小評価されることが 2025 でも再現した。確率の補正項として <code>config/tp72_frozen_rule.json</code> に定義を残し、新しいレース（2026-10-22 以降）で同じ向きに出るかを、同じ式で見る。2026 は向きは同じだが弱い（z 約 1）ので、効果が続く保証はない。</li>
<li><b>限界</b>: モデルは V4（OOF）。V6e / V5 アンサンブルでの再計算は、V6e の OOF が表にまだ無い。SmartRC のブリンカーは 2026 年の表が無く、初着用は 2020 年以降の履歴で判定（run_no≥3）。厩舎コメントは約 4 割の馬にしか無い。血統×騎手は件数不足で個別セルを検定できない（階層縮約はペア走査）。1 走の確定オッズでの回収率で、購入時点（T-60/T-15）のオッズとの差は見ていない。</li>
<li><b>次の案</b>: (1) V6e の OOF が出たらこの表の p を差し替えて再走査、(2) 夏の牝馬を含む性別×季節を EV の補正項にして、T-60/T-15 の実オッズが貯まってから回収率で確認、(3) 関係者の件数を増やせる 2020 年以前のデータ（取得できれば）で血統×騎手の個別セルを検定。</li>
</ul>
<h3>再現コマンド</h3>
<pre>python -m src.scripts.data.tp72_cands            # 事前登録（tp72_candidates.json）
python -m src.scripts.data.tp72_data build       # ホストで。1 出走 1 行の表 mining.parquet
python -m src.scripts.data.tp72_eval stage1      # 第1段
python -m src.scripts.data.tp72_cc run           # 第C部（穴馬の共通要素）
python -m src.scripts.data.tp72_stage2 run       # 第2段（学習を伴わない集計版）
python -m src.scripts.data.tp72_extra run        # 補助 A〜D
python -m src.scripts.data.tp72_report --in DIR --out docs/html/modeling/chapters/edge_factor_mining.html</pre>
"""
    return body


def page(body: str) -> str:
    head = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>買う要素の網羅探索（tp-72）— keiba-vpn</title>
<style>:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; --good:#34d399; --bad:#f87171; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1250px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 12.5px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 4px 9px; text-align: left; vertical-align: top; } th { background: #1b2a4d; color: var(--muted); }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
code, pre { background: #0b1428; padding: 1px 5px; border-radius: 4px; } pre { padding: 10px 14px; overflow-x: auto; }</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>買う要素の網羅探索（tp-72）</h1>
<p class="muted">2026-10-22、ユーザー依頼『去勢明け2戦目など、買う要素となりうる材料をもっと漁ってほしい。クッション値を用いた深い分析なども』と、その後の追加（夏の牝馬・季節、逃げはさみ、ブリンカー×脚質、枠×脚質、体重・追い切り・間隔の同時発生、血統×騎手、穴馬の共通要素、記述子ペアの系統走査）。
資料: <a href="training_pipeline_decisions.html">決定ログ</a>（§62）・<a href="feature_investigation_catalog.html">特徴量調査の総覧</a>・<a href="jockey_trainer_roi.html">騎手・調教師の回収率（tp-59）</a>・<a href="equipment_and_change.html">馬具・条件替わり（tp-32）</a>。
回収率は『1 出走（1 点）に 100 円』、区間はレース単位ブートストラップ。選択は 2022〜2024、2025 は確認 1 回、2026 年はテストセット②（選択に使わない。記述のみ）。</p>"""
    return head + body + """
<p class="muted">関連: <a href="training_pipeline_decisions.html">決定ログ</a>・<a href="jockey_trainer_roi.html">tp-59</a>・<a href="feature_cross_search.html">tp-58</a>・<a href="betting_edge_search.html">tp-36</a>。</p>
</body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_text(page(build(Path(a.inp))), encoding="utf-8")
    print("wrote", a.out)
