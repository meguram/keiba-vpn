"""tp-75 第 2 ラウンド（原子条件の連言）と 2c（完全な母方 10 世代表での血統の系統の再測定）の節・台帳 CSV・凍結ファイル（v3）を作る。

tp75_report.chapter() が section_r2 / section_rc を章に差し込み、write_all() が CSV と凍結ファイルを書く。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from src.scripts.data.tp75_report import FAM, FAM_JA, FROZEN, PERSONA_JA, ROOT, ci, esc, pc, rs

LEDGER_R2_CSV = ROOT / "docs/operations/tp75_rule_ledger_r2.csv"
LEDGER_RC_CSV = ROOT / "docs/operations/tp75_rule_ledger_rc.csv"
MATERNAL_README = ROOT / "docs/operations/tp75_maternal_readme.json"
FROZEN_ON_V2 = "2026-10-23"


def load(src: Path) -> tuple[dict, list[dict]]:
    return json.loads((src / "results_r2.json").read_text(encoding="utf-8")), json.loads((src / "ledger_r2.json").read_text(encoding="utf-8"))


def load_extra(src: Path) -> dict:
    out = {}
    for k, f in (("r2b", "results_r2b.json"), ("rc", "results_rc.json"), ("lrc", "ledger_rc.json"), ("rcp", "results_rcp.json")):
        p = src / f
        out[k] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return out


# ------------------------------------------------------------------------------------------------ 集計
def _z(r: dict, outcome: str, yr: str) -> float | None:
    return r[f"z_{yr}_win"] if outcome == "win" else r[f"z_{yr}_top3"]


def rep_by_subset(L: list[dict], key: str) -> dict:
    """族ごとに、最終採用 / BH 頑健の最終採用 / それ以外 の 2025・2026 での再現。"""
    out = {}
    for fam, (outcome, sign, _p, _d) in FAM.items():
        fin = [r for r in L if r[fam].get(key)]
        rob = [r for r in fin if r[fam]["robust_bh"]]
        non = [r for r in fin if not r[fam]["robust_bh"]]
        o = {}
        for tag, rs_ in (("final", fin), ("robust", rob), ("non_robust", non)):
            z25 = [sign * _z(r, outcome, "2025") for r in rs_ if _z(r, outcome, "2025") is not None]
            z26 = [sign * _z(r, outcome, "2026") for r in rs_ if _z(r, outcome, "2026") is not None]
            o[tag] = {"n": len(rs_), "n25": len(z25), "same25": int(sum(z > 0 for z in z25)), "gt25": int(sum(z > 1.645 for z in z25)), "exp25": round(0.05 * len(z25), 1),
                      "n26": len(z26), "same26": int(sum(z > 0 for z in z26))}
        out[fam] = o
    return out


LINE_TYPES = [("mmmf_", "母母母父の系統"), ("mmf_", "母母父の系統"), ("mff_", "母父父の系統"), ("mat_near_", "母方・近い世代の系統"), ("mat_deep_heavy_", "母方・深い世代の厚み"),
              ("mat_hf", "母方の高頻度種牡馬"), ("mat_cl", "母方クラスタ"), ("dsire", "母父の系統"), ("ds_", "母父の系統"), ("sire_", "父方の系統（対照）")]


def line_types(rule_id: str) -> set[str]:
    out = set()
    for atom in rule_id.split("/", 1)[1].split("&"):
        for pre, lab in LINE_TYPES:
            if atom.startswith(pre):
                out.add(lab)
                break
    return out


def type_table(LC: list[dict]) -> str:
    rows = ""
    labs: list[str] = []
    for _p, lab in LINE_TYPES:
        if lab not in labs:
            labs.append(lab)
    for lab in labs:
        for fam, (outcome, sign, _pp, _d) in FAM.items():
            fin = [r for r in LC if r[fam].get("final") and lab in line_types(r["id"])]
            us = [r for r in fin if r[fam]["robust_bh"]]
            if not fin:
                continue
            z25 = [sign * _z(r, outcome, "2025") for r in us if _z(r, outcome, "2025") is not None]
            rows += (f"<tr><td>{lab}</td><td>{FAM_JA[fam]}</td><td class='num'>{len(fin)}</td><td class='num'>{len(us)}</td><td class='num'>{sum(z > 0 for z in z25)} / {len(z25)}</td>"
                     f"<td class='num'>{sum(z > 1.645 for z in z25)}（期待 {0.05 * len(z25):.1f}）</td></tr>")
    return "<table><tr><th>血統の系統の種類</th><th>族</th><th>最終採用（その種類を含む連言）</th><th>うち BH 頑健（使用）</th><th>使用のうち 2025 同符号</th><th>2025 で z&gt;1.645（偶然の期待）</th></tr>" + rows + "</table>"


# ------------------------------------------------------------------------------------------------ CSV・凍結ファイル
def _csv(path: Path, L: list[dict]) -> None:
    cols = ["id", "order", "expr", "n_disc", "races_disc", "win_resid_pt", "win_resid_z", "top3_resid_pt", "top3_resid_z", "z_2025_win", "z_2025_top3", "z_2026_win", "z_2026_top3"]
    ext = [f"{f}_{k}" for f in FAM for k in ("tier", "adopted", "final", "usable", "robust_bh", "q_combined", "cluster", "inc_parents", "inc_round1")]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ext)
        for r in L:
            row = [r.get(c) for c in cols]
            for f in FAM:
                a = r[f]
                row += [a["tier"], a["adopted"], a["final"], bool(a["final"] and a["robust_bh"]), a["robust_bh"], None if a["q_combined"] is None else round(a["q_combined"], 6), a["cluster"],
                        json.dumps({k: round(v, 2) for k, v in (a["inc_parents"] or {}).items()}, ensure_ascii=False) if a["inc_parents"] else "",
                        json.dumps({k: round(v, 2) for k, v in (a["inc_round1"] or [])}, ensure_ascii=False) if a["inc_round1"] else ""]
            w.writerow(row)


def write_frozen_v3(RCP: dict, L2: list[dict], LC: list[dict]) -> None:
    by2 = {r["id"]: r for r in L2}
    byc = {r["id"]: r for r in LC}
    for persona, path in FROZEN.items():
        spec = json.loads(path.read_text(encoding="utf-8"))
        base = [r for r in spec["rules"] if r.get("round", 1) == 1]
        for r in base:
            r["round"] = 1
        rules = list(base)
        have = {r["id"] for r in rules}
        added2 = addedc = 0
        clusters_out = []
        for fam, (outcome, sign, p, direction) in FAM.items():
            if p != persona:
                continue
            for c in RCP["clusters"][fam]:
                clusters_out.append({"family": fam, "representative": c["rep"], "round": c["round"], "members": c["members"], "weight_pt": round(100 * c["weight"], 3), "cover_disc": round(c["cover"], 4), "tier": c["tier"], "robust_bh": c["robust_bh"]})
                for m in c["members"]:
                    if m in have:
                        continue
                    src_ = byc.get(m) or by2.get(m)
                    if src_ is None:
                        continue
                    rnd = "2c" if m in byc else 2
                    a = src_[fam]
                    z25, z26 = _z(src_, outcome, "2025"), _z(src_, outcome, "2026")
                    rules.append({"id": src_["id"], "expr": src_["expr"], "source": "第 2 ラウンド（原子条件の連言）" if rnd == 2 else "2c（完全な母方 10 世代表）", "direction": direction, "outcome": outcome, "family": fam,
                                  "tier": a["tier"], "robust_bh": a["robust_bh"], "market_free": True, "cluster_representative": c["rep"], "n_disc": src_["n_disc"],
                                  "effect_pt": round(src_["win_resid_pt"] if outcome == "win" else src_["top3_resid_pt"], 3), "z_disc": round(sign * (src_["win_resid_z"] if outcome == "win" else src_["top3_resid_z"]), 3),
                                  "q_bh_combined": a["q_combined"], "z_2025": None if z25 is None else round(sign * z25, 3), "z_2026_v4": None if z26 is None else round(sign * z26, 3),
                                  "round": rnd, "order": src_["order"], "inc_parents": a["inc_parents"], "inc_round1": a["inc_round1"],
                                  "lineage_table": "不完全な表（旧）" if "dds_" in src_["id"] else ("完全な母方 10 世代表" if rnd == "2c" else "なし")})
                    have.add(m)
                    if rnd == 2:
                        added2 += 1
                    else:
                        addedc += 1
        spec["version"] = 3
        spec["market_free"] = all(r["market_free"] for r in rules)
        spec["rules"] = rules
        spec["n_rules"] = len(rules)
        spec["clusters"] = clusters_out
        spec["n_clusters"] = len(clusters_out)
        spec["lineage_note"] = ("第 2 ラウンドの母母父（dds_*）は不完全な 10 世代表（5 世代 + 牡系アンカー）に基づく旧版。2c は完全な母方 10 世代表（1,023 位置）に基づく。"
                                "2024 年生まれの馬は表が未整備（母方が完全な割合 27%）で、該当しない扱いになる。表が揃ったら再実行する（rerun_when）。")
        spec["rerun_when"] = ["2023〜2024 年生まれの馬の 10 世代表が揃ったとき", "v6e の 2026 年全体の予測が揃ったとき"]
        spec["operations"]["reeval_command"] = f"python -m src.scripts.data.tp75c_rules reeval {path.relative_to(ROOT)} --years <新しい年>   # 第 2 ラウンド・2c の列（血統の系統・前走・母方の表）を足した表で評価する"
        pr = RCP["v3"]["personas"][persona]
        spec["evaluation_summary_v3"] = {"top_group_def_b": {pn: {k: pr[f"def_b/{pn}/top_group"].get(k) for k in ("n_picks", "n_races", "win_rate", "top3_rate", "roi_win", "roi_place")} for pn in ("disc", "y2025", "y2026")}}
        spec["history"].append({"date": FROZEN_ON_V2, "action": "v2: 第 2 ラウンド（原子条件の連言）のうち BH 頑健なものを追加", "added": added2, "expired": 0,
                                "note": "候補 243,152 型。A/B・再現・増分だけの採用は 2.6〜3.5 千個になり 2025 年の標本外で偶然の期待とほぼ同じだったため、BH 頑健（合算の母数）なものだけを使う（事後の判断）"})
        spec["history"].append({"date": FROZEN_ON_V2, "action": "v3: 2c（完全な母方 10 世代表での血統の系統の再測定）のうち BH 頑健なものを追加", "added": addedc, "expired": 0,
                                "note": "候補 44,470 型。不完全な表に基づく旧ルール（dds_*）は台帳に『不完全な表（旧）』として残す"})
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")


def write_all(src: Path) -> None:
    R2, L2 = load(src)
    X = load_extra(src)
    _csv(LEDGER_R2_CSV, L2)
    if X["lrc"] is not None:
        _csv(LEDGER_RC_CSV, X["lrc"])
    if X["rcp"] is not None and X["lrc"] is not None:
        write_frozen_v3(X["rcp"], L2, X["lrc"])
    mr = src / "maternal_lineage" / "README.json"
    if mr.exists():
        MATERNAL_README.write_text(mr.read_text(encoding="utf-8"), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ 章の節
def _cmp_rows(P: dict[str, dict]) -> str:
    rows = ""
    for p in ("ev_seeker", "place_seeker"):
        for pn, pl in (("y2025", "2025（評価 1 回）"), ("y2026", "2026（記述）")):
            for lab, blk in P.items():
                t = blk["personas"][p][f"def_b/{pn}/top_group"]
                if not t.get("n_picks"):
                    continue
                rows += (f"<tr><td>{PERSONA_JA[p]}</td><td>{pl}</td><td>{lab}</td><td class='num'>{t['n_races']:,}</td><td class='num'>{ci(t['win_rate'])}</td><td class='num'>{ci(t['top3_rate'])}</td>"
                         f"<td class='num'>{ci(t['roi_win'], 0)}</td><td class='num'>{ci(t['roi_place'], 0)}</td><td class='num'>{rs(t['resid_top3'])}</td></tr>")
    return rows


def _bins_rows(P: dict[str, dict]) -> str:
    lab = {"neg": "消し優勢", "pos_low": "買い・弱", "pos_high": "買い・強"}
    rows = ""
    for p in ("ev_seeker", "place_seeker"):
        for pn, pl in (("y2025", "2025"), ("y2026", "2026")):
            for tag, blk in P.items():
                for k in ("neg", "pos_high"):
                    v = blk["personas"][p][f"bins/{pn}"].get(k)
                    if v and v["n"] >= 100:
                        rows += f"<tr><td>{PERSONA_JA[p]}</td><td>{pl}</td><td>{tag}</td><td>{lab[k]}</td><td class='num'>{v['n']:,}</td><td class='num'>{rs(v['resid_win'])}</td><td class='num'>{rs(v['resid_top3'])}</td><td class='num'>{ci(v['roi_place'], 0)}</td></tr>"
    return rows


def _ov_rows(P: dict[str, dict]) -> str:
    rows = ""
    for p in ("ev_seeker", "place_seeker"):
        for tag, blk in P.items():
            o = blk["personas"][p]["overlay"]
            rows += f"<tr><td>{PERSONA_JA[p]}</td><td>{tag}</td>" + "".join(f"<td class='num'>{o[pn]['delta'] * 1e4:+.1f}<span class='muted'> [{o[pn]['lo'] * 1e4:+.1f}, {o[pn]['hi'] * 1e4:+.1f}]</span></td>" for pn in ("disc", "y2025", "y2026")) + "</tr>"
    return rows


def _rep_rows(rep: dict) -> str:
    rows = ""
    for fam in FAM:
        for tag, lab in (("final", "最終採用（事前登録の基準）"), ("robust", "うち BH 頑健"), ("non_robust", "うち BH 頑健でない")):
            v = rep[fam][tag]
            if v["n"]:
                rows += (f"<tr><td>{FAM_JA[fam]}</td><td>{lab}</td><td class='num'>{v['n']:,}</td><td class='num'>{v['same25']} / {v['n25']}（{100 * v['same25'] / max(v['n25'], 1):.0f}%）</td>"
                         f"<td class='num'>{v['gt25']}（期待 {v['exp25']}）</td><td class='num'>{v['same26']} / {v['n26']}（{100 * v['same26'] / max(v['n26'], 1):.0f}%）</td></tr>")
    return rows


def _top_rules(L: list[dict], k: int = 6) -> str:
    rows = ""
    for fam, (outcome, sign, _p, _d) in FAM.items():
        fin = [r for r in L if r[fam].get("final") and r[fam]["robust_bh"]]
        fin.sort(key=lambda r: -abs((r["win_resid_z"] if outcome == "win" else r["top3_resid_z"]) or 0))
        for r in fin[:k]:
            inc = r[fam]["inc_parents"] or {}
            incs = "、".join(f"{v:+.1f}" for v in list(inc.values())[:3]) if inc else "–"
            z25 = _z(r, outcome, "2025")
            rows += (f"<tr><td>{FAM_JA[fam]}</td><td class='expr'>{esc(r['id'].split('/', 1)[1])}</td><td class='num'>{r['order']}</td><td class='num'>{r['n_disc']:,}</td>"
                     f"<td class='num'>{(r['win_resid_pt'] if outcome == 'win' else r['top3_resid_pt']):+.2f}</td><td>{r[fam]['tier']}</td><td class='num'>{'–' if z25 is None else f'{sign * z25:+.1f}'}</td><td class='num'>{incs}</td></tr>")
    return rows


def _dose_rows(D: dict, with_win: bool = True) -> str:
    rows = ""
    for fam in FAM:
        for pn, pl in (("y2025", "2025"), ("y2026", "2026")):
            for lab, v in D[fam][pn].items():
                if v["n"] >= 100:
                    rows += (f"<tr><td>{FAM_JA[fam]}</td><td>{pl}</td><td class='num'>{lab}</td><td class='num'>{v['n']:,}</td><td class='num'>{rs(v['resid'])}</td><td class='num'>{ci(v['top3'])}</td>"
                             + (f"<td class='num'>{ci(v['roi_win'], 0)}</td>" if with_win else "") + f"<td class='num'>{ci(v['roi_place'], 0)}</td></tr>")
    return rows


def section_r2(R1: dict, R2: dict, L2: list[dict], X: dict) -> str:
    S, en, pc2 = R2["summary"], R2["enumeration"], R2["prereg_counts"]
    R2B = X["r2b"]
    rep = rep_by_subset(L2, "final")
    srows = ""
    for fam in FAM:
        s = S[fam]
        o = s["by_order_final"]
        srows += (f"<tr><td>{FAM_JA[fam]}</td><td class='num'>{s['candidates']:,}</td><td class='num'>{s['promising']:,}</td><td class='num'>{s['adopted_before_increment']:,}</td><td class='num'>{s['pass_parent_increment']:,}</td>"
                  f"<td class='num'>{s['pass_round1_increment']:,}</td><td class='num'>{s['final']:,}（A {s['final_A']:,}）</td><td class='num'><b>{s['final_robust_bh']}</b></td><td class='num'>{o['1']} / {o['2']} / {o['3']}</td>"
                  f"<td class='num'>{s['expected_false_pos_at_1.645']:,.0f}</td></tr>")
    cmp_rows = _cmp_rows({"第 1 ラウンド": R1["primary"], "第 2 ラウンド（BH 頑健を足した）": R2B["v2"]})
    brow = _bins_rows({"第 1": R1["primary"], "第 2（頑健）": R2B["v2"]})
    ov = _ov_rows({"第 1": R1["primary"], "第 2（頑健）": R2B["v2"]})
    used = R2B["round2_rules_used"]

    def same(f: str, t: str) -> str:
        v = rep[f][t]
        return f"{100 * v['same25'] / max(v['n25'], 1):.0f}%"
    return f"""
<h2 id="round2">12. 第 2 ラウンド: 原子条件の連言（2・3 条件）で絞り込む</h2>
<p class="muted">ユーザーの追加依頼（2026-10-22）: 『期待値追求家・複勝率追求家の両方について、ルールの要素と組み合わせを広げる。無駄な重複を避けて様々な組み合わせを試す』。コード: tp75b_rules.py・tp75b_report.py。事前登録（原子条件・列挙規則・候補総数）は評価の実行前にコミット（f78dea2、<code>tp75b_prereg.json</code>）。台帳 CSV: <code>docs/operations/tp75_rule_ledger_r2.csv</code>。モデルの学習はしていない。</p>
<div class="note"><b>要点（第 2 ラウンド）.</b>
<ul>
<li><b>候補 {pc2['candidates']:,} 型.</b> 原子条件 {pc2['atoms']} 個（血統の系統・前走・今走・馬の状態。市場情報なし）の連言を、頻出集合の列挙（発見期間で 300 走・100 レース以上。サポートの計算に結果は使わない）で数え上げた: 1 条件 {pc2['by_order']['1']}・2 条件 {pc2['by_order']['2']:,}・3 条件 {pc2['by_order']['3']:,}（第 1 ラウンドや先の候補と該当行が同じ {en['duplicates_dropped']:,} 型を除いた）。4 条件以上は標本不足のため打ち切り。BH の母数は第 1 ラウンドの検定可能 {pc2['round1_testable']} 型と合算の {pc2['combined_tests_per_family']:,} 本（族ごと）。</li>
<li><b>事前登録の基準（A/B・再現・親と第 1 ラウンドに対する増分）だけで採用すると、族あたり 2.6〜3.5 千個になるが、標本外では偶然とほぼ同じ.</b> 2025 年に同じ向きだった割合は 期待値の買い {same('ev_buy', 'final')}・複勝率の買い {same('place_buy', 'final')}・期待値の消し {same('ev_kill', 'final')}・複勝率の消し {same('place_kill', 'final')}。<b>理由:</b> 発見期間の中で増分を検定しても、z が高い候補を選んだあとの増分は選択バイアスで通りやすい（増分は独立なデータの確認になっていない）。</li>
<li><b>BH 頑健（合算の母数 {pc2['combined_tests_per_family']:,}）な連言だけは標本外でも再現する.</b> 期待値の買い {S['ev_buy']['final_robust_bh']}・消し {S['ev_kill']['final_robust_bh']}、複勝率の買い {S['place_buy']['final_robust_bh']}・消し {S['place_kill']['final_robust_bh']}。2025 年に同じ向き: 買いは {same('ev_buy', 'robust')}・{same('place_buy', 'robust')}、消しは {same('ev_kill', 'robust')}・{same('place_kill', 'robust')}（12.3）。<b>事後の判断:</b> この結果を見たうえで、順位付けと凍結に使うのは BH 頑健なものだけにした（事前登録の基準は『頑健性を併記』で、母数の大きさを見込んでいなかった）。</li>
<li><b>ルールを足しても順位付けはほぼ変わらない（12.6）.</b> スコア最上位グループの成績は第 1 ラウンドと同水準で、回収率は 100% を超えない。</li>
<li><b>血統の系統の結果は不完全な 10 世代表（5 世代 + 牡系アンカー）に基づく.</b> 完全な母方 10 世代表での再測定は 2c（第 13 節）。第 2 ラウンドの母母父（dds_*）は『不完全な表（旧）』として台帳に残す。</li>
</ul></div>

<h3>12.1 設計（事前に固定）</h3>
<ul>
<li><b>原子条件.</b> 血統の系統（父・母父・母母父。main 5 + coarse 12 + fine 36 の 3 段）、前走条件（距離の延長・短縮の幅、芝ダ替わり、クラスの昇降、着順・着差、上がり順位、初角・4角の位置、馬場、開催場、間隔、騎手の継続・乗り替わり）、今走の条件（競馬場・面・距離・馬場・季節・枠・頭数・クラス・性別・年齢・斤量差・馬体重の増減）、馬の状態（去勢明け N 戦目、ブリンカー、母の年齢・初仔、種牡馬の供用年、脚質、初めての面）。同じ次元（排他・包含）の原子は同じ連言に入れない。前走の不利は列が無く入れていない。</li>
<li><b>列挙.</b> Apriori 型: 原子 → 2 条件 → 3 条件。サブセットがすべて頻出のものだけ延ばす。該当行が第 1 ラウンドのルールまたは先の候補と完全に同じ連言は重複として除く。</li>
<li><b>採用.</b> 第 1 ラウンドと同じ基準（現行モデルの残差、A=95%・B=90%、年をまたぐ再現）。BH は合算の母数で、頑健性を併記。<b>増分</b>: 連言は、すべての直近の親（1 条件減らしたもの）に対して『連言 − 親の残り』の差が採用の向きに片側 95% の水準で有意な場合だけ採用。第 1 ラウンドの採用ルールと該当行が重なるもの（包含≥0.9 または Jaccard≥0.5）は、重なりを除いた部分に効果がある場合だけ採用。</li>
</ul>

<h3>12.2 候補と採用</h3>
<table><tr><th>族</th><th>候補</th><th>有望（片側 B の水準）</th><th>採用（再現あり）</th><th>親の増分を通過</th><th>第 1 ラウンドに対する増分を通過</th><th>最終採用（事前登録の基準）</th><th>うち BH 頑健</th><th>条件数 1 / 2 / 3</th><th>偶然の期待数（z&gt;1.645）</th></tr>{srows}</table>
<h3>12.3 標本外での再現: BH 頑健かどうかで分ける</h3>
<table><tr><th>族</th><th>集合</th><th>ルール数</th><th>2025 年に同じ向き</th><th>2025 で z&gt;1.645 の向き（偶然の期待）</th><th>2026 年に同じ向き（v4・記述）</th></tr>{_rep_rows(rep)}</table>
<h3>12.4 BH 頑健な連言（上位）</h3>
<p class="muted">『親の増分 z』は、連言と、1 条件減らした親との差の z（採用の向きを正）。</p>
<table><tr><th>族</th><th>ルール</th><th>条件数</th><th>走数（発見）</th><th>残差（pt）</th><th>区分</th><th>z（2025）</th><th>親の増分 z</th></tr>{_top_rules(L2)}</table>
<h3>12.5 用量反応: 合致した（頑健な）クラスタの数 k と残差</h3>
<p class="muted">各族のクラスタ（第 1 ラウンドの採用 + 第 2 ラウンドの BH 頑健な最終採用 {used['ev_buy']}・{used['ev_kill']}・{used['place_buy']}・{used['place_kill']} 個）に何個当たるか。残差は『採用の列』（期待値は勝ち、複勝率は3着内）。買いは正、消しは負が採用の向き。</p>
<table><tr><th>族</th><th>期間</th><th>k</th><th>走数</th><th>残差（pt）</th><th>3着内率 %</th><th>単勝の回収率 %</th><th>複勝の回収率 %</th></tr>{_dose_rows(R2B['dose_response'])}</table>
<h3>12.6 順位付け（合致率）の再評価: 第 1 ラウンドとの差</h3>
<table><tr><th>ペルソナ</th><th>期間</th><th>集合</th><th>選んだレース</th><th>勝率 %</th><th>3着内率 %</th><th>単勝の回収率 %</th><th>複勝の回収率 %</th><th>3着内の残差（pt）</th></tr>{cmp_rows}</table>
<table><tr><th>ペルソナ</th><th>期間</th><th>集合</th><th>スコア帯</th><th>走数</th><th>勝ちの残差（pt）</th><th>3着内の残差（pt）</th><th>複勝の回収率 %</th></tr>{brow}</table>
<p>確率への補正（ロジットのずれを足した LogLoss の差、×10⁻⁴、負が改善。足し合わせるずれは ±0.3 に上限。第 1 ラウンドも同じ上限で計算し直した）:</p>
<table><tr><th>ペルソナ</th><th>集合</th><th>発見</th><th>2025</th><th>2026（v4・参考）</th></tr>{ov}</table>
"""


def section_rc(R1: dict, R2: dict, X: dict) -> str:
    RC, LC, RCP, R2B = X["rc"], X["lrc"], X["rcp"], X["r2b"]
    S, pc2 = RC["summary"], RC["prereg_counts"]
    rep = rep_by_subset(LC, "final")
    srows = ""
    for fam in FAM:
        s = S[fam]
        o = s["by_form_final"]
        srows += (f"<tr><td>{FAM_JA[fam]}</td><td class='num'>{s['candidates']:,}</td><td class='num'>{s['promising']:,}</td><td class='num'>{s['adopted_before_increment']:,}</td><td class='num'>{s['pass_parent_increment']:,}</td>"
                  f"<td class='num'>{s['pass_prior_increment']:,}</td><td class='num'>{s['final']:,}</td><td class='num'><b>{s['final_robust_bh']}</b>（全ラウンド合算の母数でも頑健 {s['final_robust_bh_all']}）</td><td class='num'>{o['1']} / {o['2']} / {o['3']}</td></tr>")
    old = RC["old_rules_incomplete_table"]
    old_fin = {fam: sum(1 for x in old if x.get(f"{fam}_final")) for fam in FAM}
    new_mmf = [r for r in LC if re.search(r"(^|/|&)(mmf_|mmmf_)", r["id"])]
    new_fin = {fam: sum(1 for r in new_mmf if r[fam].get("final")) for fam in FAM}
    new_us = {fam: sum(1 for r in new_mmf if r[fam].get("final") and r[fam]["robust_bh"]) for fam in FAM}
    cmp_old = "".join(f"<tr><td>{FAM_JA[fam]}</td><td class='num'>{len(old)}</td><td class='num'>{old_fin[fam]}</td><td class='num'>{len(new_mmf)}</td><td class='num'>{new_fin[fam]}</td><td class='num'>{new_us[fam]}</td></tr>" for fam in FAM)

    def z25(x: dict, key: str) -> str:
        return "–" if x[key] is None else f"{x[key]:+.1f}"
    old_rows = "".join(f"<tr><td class='expr'>{esc(x['id'].split('/', 1)[1])}</td><td class='num'>{x['n_disc']:,}</td><td class='num'>{x['win_resid_pt']:+.2f}</td><td class='num'>{x['top3_resid_pt']:+.2f}</td><td class='num'>{z25(x, 'z_2025_top3')}</td></tr>"
                       for x in sorted(old, key=lambda r: -abs(r["top3_z"] or 0))[:8])
    new_rows = "".join(f"<tr><td class='expr'>{esc(x['id'].split('/', 1)[1])}</td><td class='num'>{x['n_disc']:,}</td><td class='num'>{x['win_resid_pt']:+.2f}</td><td class='num'>{x['top3_resid_pt']:+.2f}</td><td class='num'>{z25(x, 'z_2025_top3')}</td></tr>"
                       for x in sorted(new_mmf, key=lambda r: -abs(r["top3_resid_z"] or 0))[:8])
    cmp_rows = _cmp_rows({"第 1 ラウンド": R1["primary"], "第 2（頑健）": R2B["v2"], "2c を足した": RCP["v3"]})
    brow = _bins_rows({"第 1": R1["primary"], "第 2（頑健）": R2B["v2"], "2c を足した": RCP["v3"]})
    ov = _ov_rows({"第 1": R1["primary"], "第 2（頑健）": R2B["v2"], "2c を足した": RCP["v3"]})
    mat = json.loads(MATERNAL_README.read_text(encoding="utf-8")) if MATERNAL_README.exists() else {}
    hf = "、".join(f"{h['name']}（{h['prevalence'] * 100:.0f}%）" for h in mat.get("hf_sires", [])[:20])
    return f"""
<h2 id="round2c">13. 2c: 血統の系統ルールを、完全な母方 10 世代表で再測定する</h2>
<p class="muted">コーディネータの更新（2026-10-22）: 2020 年以降の出走馬の母方 10 世代血統が完全になった（母方 1,023 位置、世代別の埋まり率は 1〜10 世代とも 100%。<code>docs/operations/backfill_2016.md</code> §9）。第 2 ラウンドの血統の系統ルールを、この表で再測定した。父方の系統は対照（sire_main・sire_coarse だけ）。事前登録（原子条件 {pc2['atoms']} 個・候補 {pc2['candidates']:,} 型）は評価の実行前にコミット（0d7d93f、<code>tp75c_prereg.json</code>）。コード: tp75c_maternal.py（共有表の作成）・tp75c_rules.py。台帳 CSV: <code>docs/operations/tp75_rule_ledger_rc.csv</code>。共有表: <code>maternal_lineage.parquet</code>（作成物の README は <code>docs/operations/tp75_maternal_readme.json</code>）。</p>
<div class="note"><b>要点（2c）.</b>
<ul>
<li><b>共有表.</b> 完全な母方 10 世代表（<code>data/local/horse_pedigree_10gen/</code>。path_fm が M で始まる 1,023 位置）から、出走馬ごとの列を作った（構造だけ。成績・オッズは使わない）: 母母父・母母母父・母父父の系統、母方の牡馬の系統の本数（近い世代 2〜5・深い世代 6〜10）、母方の高頻度種牡馬（母方に現れる割合が 5〜60% の上位 20 頭 = {hf}）の有無・2 回以上・深い世代だけ、母方クラスタ（系統の割合の KMeans K=8, 24）。</li>
<li><b>候補 {pc2['candidates']:,} 型.</b> 形は L・L,L'・L,C・L,C,C'（L = 血統の系統、C = 文脈。L,C,C' の C は競馬場・面・距離・馬場・季節・クラスだけ）。最小サポートは 400 走・120 レース。BH の母数は第 1 ラウンドの検定可能 {pc2['round1_testable']} 型との合算 {pc2['combined_tests_per_family']:,} 本（族ごと）。全ラウンド合算 {pc2['all_rounds_tests_per_family']:,} 本の q も併記。</li>
<li><b>使うのは最終採用のうち BH 頑健なものだけ（結果を見る前に固定）.</b> 最終採用 {S['ev_buy']['final']}・{S['ev_kill']['final']}・{S['place_buy']['final']}・{S['place_kill']['final']}（期待値の買い・消し、複勝率の買い・消し）のうち、BH 頑健（使用）は {S['ev_buy']['final_robust_bh']}・{S['ev_kill']['final_robust_bh']}・{S['place_buy']['final_robust_bh']}・{S['place_kill']['final_robust_bh']}。</li>
<li><b>旧ルール（不完全な表の母母父）は台帳に残し、完全な表の結果と並べた（13.4）.</b></li>
<li><b>注意.</b> 母方の表が未整備の馬（2024 年生まれは母方が完全な割合 27%、2023 年生まれ 89%）は、血統の系統の条件に該当しない扱いになる。2025・2026 年の若い馬の評価は過小になる。最初の試算で L,L',C の形を含めると候補が数百万に膨らんだので、その形は事前登録から外した（試算は結果を使っていない）。</li>
</ul></div>
<h3>13.1 候補と採用</h3>
<table><tr><th>族</th><th>候補</th><th>有望</th><th>採用（再現あり）</th><th>親の増分</th><th>第 1・第 2 ラウンドに対する増分</th><th>最終採用</th><th>うち BH 頑健（使用）</th><th>条件数 1 / 2 / 3</th></tr>{srows}</table>
<h3>13.2 標本外での再現</h3>
<table><tr><th>族</th><th>集合</th><th>ルール数</th><th>2025 年に同じ向き</th><th>2025 で z&gt;1.645 の向き（偶然の期待）</th><th>2026 年に同じ向き（v4・記述）</th></tr>{_rep_rows(rep)}</table>
<h3>13.3 血統の系統の種類ごと（母方と父方の対照）</h3>
<p class="muted">連言に含まれる血統の系統の原子の種類ごとに、最終採用の数と、BH 頑健（使用）のうち 2025 年に同じ向きだった数。</p>
{type_table(LC)}
<h3>13.4 旧（不完全な表の母母父）と新（完全な母方の表）</h3>
<table><tr><th>族</th><th>旧: 第 2 ラウンドの母母父を含む有望な候補</th><th>旧: 最終採用</th><th>新: 母母父・母母母父を含む有望な候補</th><th>新: 最終採用</th><th>新: BH 頑健（使用）</th></tr>{cmp_old}</table>
<p class="muted">旧（不完全な表）の上位（3着内の残差の z が大きい順）:</p>
<table><tr><th>ルール（第 2 ラウンド）</th><th>走数（発見）</th><th>勝ちの残差（pt）</th><th>3着内の残差（pt）</th><th>z（2025・3着内）</th></tr>{old_rows}</table>
<p class="muted">新（完全な母方の表）の上位:</p>
<table><tr><th>ルール（2c）</th><th>走数（発見）</th><th>勝ちの残差（pt）</th><th>3着内の残差（pt）</th><th>z（2025・3着内）</th></tr>{new_rows}</table>
<h3>13.5 BH 頑健な連言（上位）</h3>
<table><tr><th>族</th><th>ルール</th><th>条件数</th><th>走数（発見）</th><th>残差（pt）</th><th>区分</th><th>z（2025）</th><th>親の増分 z</th></tr>{_top_rules(LC)}</table>
<h3>13.6 順位付けの再評価（第 1・第 2 ラウンド・2c の比較）</h3>
<table><tr><th>ペルソナ</th><th>期間</th><th>集合</th><th>選んだレース</th><th>勝率 %</th><th>3着内率 %</th><th>単勝の回収率 %</th><th>複勝の回収率 %</th><th>3着内の残差（pt）</th></tr>{cmp_rows}</table>
<table><tr><th>ペルソナ</th><th>期間</th><th>集合</th><th>スコア帯</th><th>走数</th><th>勝ちの残差（pt）</th><th>3着内の残差（pt）</th><th>複勝の回収率 %</th></tr>{brow}</table>
<p>確率への補正（LogLoss の差、×10⁻⁴、負が改善）:</p>
<table><tr><th>ペルソナ</th><th>集合</th><th>発見</th><th>2025</th><th>2026（v4・参考）</th></tr>{ov}</table>
<h3>13.7 用量反応（合致した頑健なクラスタの数 k）</h3>
<table><tr><th>族</th><th>期間</th><th>k</th><th>走数</th><th>残差（pt）</th><th>3着内率 %</th><th>単勝の回収率 %</th><th>複勝の回収率 %</th></tr>{_dose_rows(RCP['dose_response'])}</table>
"""
