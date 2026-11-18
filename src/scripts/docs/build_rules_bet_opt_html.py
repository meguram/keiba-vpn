"""ルールの一覧の ◎ の買う／見送るの最適化（rules-bet-opt）の HTML。

使い方: python -m src.scripts.docs.build_rules_bet_opt_html [出力 HTML]
入力（計算はしない。結果のファイルを読むだけ）: <COURSE_OUT>/bet_opt/ の wf_summary.json・search_result.json・final_result.json（あれば）・diag_gamma.json、
config/rules_bet_opt/prereg_final.json。出力: docs/html/operations/rules_bet_opt.html（外部の JS なし。ライト・ダーク、スマホ幅）。
記録（文章）は docs/operations/rules_bet_opt_20261118.md。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

from src.scripts.docs.build_graded_handicapper_html import CSS

REPO = Path(__file__).resolve().parents[3]
BO = Path(os.environ.get("COURSE_OUT") or "/home/jovyan/work/course_out") / "bet_opt"
OUT = REPO / "docs/html/operations/rules_bet_opt.html"
NAMES = {"B0_fuku": "B0: ◎ が T-60 で 3 番人気以内の複勝（第 3 版の凍結の規則）", "B_all_fuku": "◎ を全部、複勝",
         "B_fav_fuku": "T-60 の 1 番人気の複勝（同じレース）", "B0_tan": "B0 の単勝", "B_fav_tan": "T-60 の 1 番人気の単勝"}
GROUPS = {"course_main": "通常 2026 年 1/1〜10/12（主・再現 20 通り）", "course_separate": "通常 2026 年 10/17〜（実物の T-60）",
          "course_separate_repro": "同じレースを再現で（参考）", "course_2025": "通常 2025 年（第 3 版を選んだ年）",
          "graded_2025": "重賞 2025 年", "graded_2026": "重賞 2026 年", "graded_2025_2026": "重賞 2025・2026 年（合算）"}


def pct(x, d=1):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def pt(x):
    return "—" if x is None else f"{x * 100:+.1f}pt"


def ci(m):
    return f"{pct(m.get('roi'))}（{pct(m.get('roi_lo'))}〜{pct(m.get('roi_hi'))}）" if m.get("roi_lo") is not None else pct(m.get("roi"))


def diff(d):
    return "—" if not d else f"{pt(d[0])}（{pt(d[1])}〜{pt(d[2])}）"


def table(head, rows, cls=""):
    h = "".join(f"<th>{escape(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="wrap"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def met_row(name, m, b0=True):
    dd = (m.get("diff") or {}).get("B0")
    return [escape(name), f"{m.get('bet_races', 0):,}", pct(m.get("hit_rate")), ci(m), pct(m.get("roi_ex_top5")),
            f"{(m.get('avg_return') or 0) * 100:.0f} 円", f"{(m.get('median_return') or 0) * 100:.0f} 円", diff(dd) if b0 else "—"]


HEAD = ["買い方", "参戦", "的中率", "回収率（95% 区間）", "上位 5 件除き", "平均回収額", "中央値回収額", "B0 との差（95% 区間）"]


def build(out: Path = OUT) -> Path:
    S = json.loads((BO / "search_result.json").read_text(encoding="utf-8"))
    WF = json.loads((BO / "wf_summary.json").read_text(encoding="utf-8"))
    F = json.loads((BO / "final_result.json").read_text(encoding="utf-8")) if (BO / "final_result.json").exists() else None
    P = json.loads((REPO / "config/rules_bet_opt/prereg_final.json").read_text(encoding="utf-8"))
    best = S["best"]["id"]
    sec = []
    # 期間と作り
    sec.append("<h2>期間と作り</h2>" + table(["区分", "通常（第 3 版）", "重賞（B）"], [
        ["◎ の作り", "前年までの訓練（2016〜Y-1）で条件を選び直す walk-forward", "凍結した B（2018〜24 年で選んだ規則）"],
        ["規則を探す", "2020〜2022 年", "探さない（材料が 2022 年以降を見て決めたもの・件数も少ない）"],
        ["評価（採否）", "2023〜2024 年", "—"],
        ["凍結の後に 1 回だけ", "2026 年（主）・2025 年（第 3 版を選んだ年）・実物の T-60", "2025・2026 年"],
        ["判定のオッズ", "T-60（2026-10-17 より前は確定オッズから 20 通り再現）", "同じ"],
        ["精算", "確定の払戻", "同じ"]]))
    # walk-forward
    rows = [[y, f"{v['train'][0]}〜{v['train'][1]}", f"{v['races']:,}", f"{v['honmei']:,}", pct(v["coverage"]), f"{v['buy_cells']:,}", pct(v["sub_layer_share"], 0)]
            for y, v in WF.items() if y.isdigit()]
    fz = WF.get("frozen_vs_wf_2024", {})
    sec.append("<h2>walk-forward の ◎</h2>" + table(["年", "訓練", "レース", "◎", "割合", "買いの条件", "副の層"], rows)
               + f'<p class="muted">2024 年の凍結の ◎ は {fz.get("frozen")} 件、walk-forward は {fz.get("wf")} 件。同じ馬が ◎ は {fz.get("same_pick")} 件'
               f'（同じ形ではない）。選んだ規則はほぼオッズだけで決まるので、◎ の中身の違いには強いと見込む。</p>')
    # 探す・評価
    c = S["search"]["cands"]
    e = S["evaluate"]["all_cands"]
    m = S["search"]["multiple"]
    rows = []
    for k in sorted(c, key=lambda k: -(c[k].get("roi") or 0)):
        rows.append([("<b>" + escape(k) + "</b>") if k == best else escape(k), f"{c[k].get('bet_races', 0):,}", pct(c[k].get("roi")),
                     pct(c[k].get("roi_ex_top5")), f"{m.get(k, {}).get('p_rw', float('nan')):.2f}", f"{e[k].get('bet_races', 0):,}", pct(e[k].get("roi")),
                     pct(e[k].get("roi_ex_top5")), pt(e[k]["diff"]["B0"][0]) if e[k].get("bet_races") else "—"])
    sec.append("<h2>探す（2020〜22）・評価（2023〜24）— 37 候補</h2>"
               "<p>名前: g{帯}_{確からしさ}_{券種}。帯 0: 〜3 倍・1: 〜5・2: 〜10・3: 3〜10・4: 5〜20・5: 10 倍〜（T-60 の単勝オッズ）。"
               "確からしさ: all = 全部・top_half / top_third = ◎ の点の上位半分・3 分の 1。<b>太字</b> = 探す期間で選んだ候補。"
               "p（RW）= Romano–Wolf の補正後の p（H0: 回収率 ≤ 100%）。</p>"
               + table(["候補", "探す 参戦", "探す 回収率", "上位 5 件除き", "p（RW）", "評価 参戦", "評価 回収率", "上位 5 件除き", "B0 との差"], rows))
    ev = S["evaluate"]
    rows = [met_row(f"選んだ候補 {best}", ev["best"])] + [met_row(NAMES.get(k, k), v, b0=False) for k, v in ev["baselines"].items() if k in NAMES]
    sec.append("<h3>評価 2023〜24 の必須の指標</h3>" + table(HEAD, rows))
    g = ev["gamma"]
    sec.append("<h3>根拠の検査</h3>" + table(["検査", "値"], [
        ["γ の感度（締切前の売れ方の情報）", "・".join(f"γ={k}: {pct(v['cand']['roi'])}（B0 との差 {pt(v['cand']['diff']['B0'][0])}）" for k, v in g.items())],
        ["閾値の 1 段ずらし（評価）", "・".join(f"{k}: {pct(v['roi'])}" for k, v in ev["neighbors"].items())],
        ["年ごと（候補／B0）", "・".join(f"{y}: {pct(v['cand']['roi'])}／{pct(v['B0']['roi'])}" for y, v in S["per_year"].items())],
        ["似た単位（評価、芝ダ × 距離帯 × 群）", f"{S['similar_units_eval']['n']} 群のうち B0 より上の割合 {pct(S['similar_units_eval']['share_better_than_B0'], 0)}"],
        ["楽観（確定オッズで判定）", f"探す {pct(S['search_optimistic'][best]['roi'])}・評価 {pct(ev['optimistic'][best]['roi'])}"],
        ["多重比較", f"37 候補とも Romano–Wolf の p = 1.00・White の reality check p = {m.get('_white_p', float('nan')):.2f}"],
        ["採否（search_spec）", escape(json.dumps(S["adopt"], ensure_ascii=False))]]))
    # 凍結の後
    if F:
        rows = []
        for k, v in F["groups"].items():
            cc = v["by_gamma"]["0.0"]["cand"]
            rows.append([escape(GROUPS.get(k, k)), escape(v["judge"])] + met_row("", cc)[1:])
        sec.append(f"<h2>凍結の後に 1 回だけ: 2025・2026 年（{escape(F['evaluated_at'] or '')}）</h2>"
                   "<p><b>何度か見た期間での確かめ</b>（rules_overview・重賞 B の評価・第 3 版の評価で成績を見ている）。100% を超えても採用の根拠にしない（事前登録）。</p>"
                   + table(["群", "判定（5 段）"] + HEAD[1:], rows))
        rows = []
        for k, v in F["groups"].items():
            for b, x in v["by_gamma"]["0.0"]["baselines"].items():
                rows.append([escape(GROUPS.get(k, k)), escape(NAMES.get(f"{b}_fuku", b)), f"{x.get('bet_races', 0):,}", ci(x), pct(x.get("roi_ex_top5"))])
        sec.append("<h3>基準（同じ群）</h3>" + table(["群", "基準", "参戦", "回収率（95% 区間）", "上位 5 件除き"], rows))
        rows = [[escape(GROUPS.get(k, k)), escape(json.dumps(v.get("coverage", {}), ensure_ascii=False)), pct(v["optimistic"].get(best, {}).get("roi"))]
                for k, v in F["groups"].items()]
        sec.append("<h3>カバー率と楽観（確定オッズで判定）</h3>" + table(["群", "カバー率", "楽観の回収率"], rows))
    else:
        sec.append("<h2>凍結の後に 1 回だけ: 2025・2026 年</h2><p>まだ流していない。</p>")
    lv = "".join(f"<li><b>{escape(x['label'])}</b>: {escape(x['if'])}</li>" for x in P["criteria"]["levels"])
    sec.append(f"<h2>判定の 5 段（事前登録 prereg_final.json）</h2><ul class=\"t\">{lv}</ul>")
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>◎ の買う／見送る</title><style>{CSS}
ul.t{{margin:4px 0;padding-left:18px}}h2{{margin-top:28px}}
</style></head><body><main>
<h1>ルールの一覧の ◎ の買う／見送る</h1>
<p class="muted">対象: <a href="../../../data/meta/graded/rules_overview.html">ルールの一覧</a>の ◎（通常レースの第 3 版・重賞 B）。
記録（計画・レビュー・事象と根拠）: docs/operations/rules_bet_opt_20261118.md。作り: src/scripts/docs/build_rules_bet_opt_html.py。
最終更新: {datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")}</p>
<div class="panel wrap"><b>結論</b><ul class="t">
<li>walk-forward の ◎ は、どの発走前オッズの帯・確からしさ・券種でも、回収率 100% に届かない（2020〜22 年の 37 候補すべて）。</li>
<li>いちばん良いのは「◎ が T-60 で 5 倍以下のときだけ複勝」で、評価 2023〜24 年は {pct(ev['best']['roi'])}（95% 区間 {pct(ev['best']['roi_lo'])}〜{pct(ev['best']['roi_hi'])}）。
市場の 1 番人気の複勝（{pct(ev['baselines']['B_fav_fuku']['roi'])}）と同程度。</li>
<li>人気薄の ◎（T-60 で 10 倍超）の単勝は 43〜65% で、favourite-longshot bias の向きと一致する。</li>
<li>ロジスティック回帰で、◎ の点は市場の勝率に上乗せする情報を持たなかった（係数 ≈ 0）。</li>
<li>答え: <b>この ◎ は見送り</b>。買うとしても、市場の 1 番人気の複勝と同じ程度。</li></ul></div>
<div class="panel wrap"><b>事象と根拠</b><ul class="t">
<li>事象: 人気側に絞るほど回収率が上がるが、上限は 85% 前後で、控除（20%）を超えない。</li>
<li>根拠 1: favourite-longshot bias（Snowberg &amp; Wolfers 2010）。人気薄ほど過大評価が控除に上乗せされる。</li>
<li>根拠 2: Benter（1994）。市場と独立の情報を市場の確率に足して初めて価値が出る。◎ の点にはその上乗せが無かった。</li>
<li>注意: 2025・2026 年は「未検証」ではない（何度か見た期間）。発走前のオッズは 2026 年の T-60 と確定の比を借りて再現しており、締切前の売れ方の情報（γ）の感度を並べた。</li></ul></div>
{''.join(sec)}
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT))
