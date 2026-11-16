"""うまうまー式の穴馬の分析（6 番人気以下を分母にした条件）を 1 枚の HTML にする。外部の JS・CSS なし、ライト・ダーク両対応、スマホ幅。

使い方: python -m src.scripts.docs.build_graded_longshot_html [置き場] [出力 HTML]
入力: <GRADED_OUT>/prereg_longshot.json・longshot_cells.json・eval_longshot.json（無ければ評価の節を出さない）
出力: data/meta/graded/graded_longshot.html
"""

from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path

from src.scripts.docs.build_graded_handicapper_html import CSS

REPO = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get("GRADED_OUT", "/home/jovyan/work/graded_out_v2"))
OUT = REPO / "data/meta/graded/graded_longshot.html"
TOP_PER_RKEY = 8


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def f2(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def rec(a) -> str:
    return "[" + "-".join(map(str, a)) + "]" if a else "—"


def table(head: list[str], rows: list[str]) -> str:
    return f'<div class="scroll"><table><thead><tr>{"".join(f"<th>{escape(h)}</th>" for h in head)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def cond(c: dict) -> str:
    """条件の日本語の文（主）と、切り口の ID・水準（小さく添える）。"""
    from src.research.graded import level_text as LT
    ids = " × ".join(f"{t} {lv}" for t, lv in zip(c["types"], c["levels"]))
    return f'{escape(LT.describe_combo(list(c["types"]), list(c["levels"])))}<div class="muted" style="font-size:11px">{escape(ids)}</div>'


def build(src: Path = SRC, out: Path = OUT) -> Path:
    P = json.loads((src / "prereg_longshot.json").read_text(encoding="utf-8"))
    cells = json.loads((src / "longshot_cells.json").read_text(encoding="utf-8"))
    ep = src / "eval_longshot.json"
    E = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else None
    c = P["constants"]
    perm = [p["pass"] for p in P["perm"]]
    # 全重賞の条件
    ev_pooled = {tuple(x["types"]) + tuple(x["levels"]): x["eval"] for x in (E["pooled"]["cells"] if E else [])}
    prow = []
    for x in P["pooled"]:
        e = ev_pooled.get(tuple(x["types"]) + tuple(x["levels"]))
        prow.append(f"<tr><td class='wrap'>{cond(x)}</td><td class='n'>{rec(x['rec'])}</td><td class='n'>{pct(x['rate'])}</td><td class='n'>{pct(x['recall'])}</td>"
                    f"<td class='n'>{pct(x['coverage'])}</td><td class='n'>{f2(x['lift'])}</td><td class='n'>{pct(x['roi_t'])}</td><td class='n'>{pct(x['roi_f'])}</td>"
                    f"<td class='n'>{x['pbin']:.1e}</td>"
                    + (f"<td class='n'>{rec(e['rec'])}</td><td class='n'>{pct(e['rate'])}</td><td class='n'>{pct(e['recall'])}</td><td class='n'>{f2(e['lift'])}</td>"
                       f"<td class='n'>{pct(e['roi_t'])}</td><td class='n'>{pct(e['roi_f'])}</td>" if e else "<td colspan='6' class='muted'>—</td>") + "</tr>")
    ptab = table(["条件（AND）", "開発 [1-2-3-着外]", "複勝率", "再現率", "該当率", "リフト", "単回収", "複回収", "二項の確率",
                  "評価 [1-2-3-着外]", "複勝率", "再現率", "リフト", "単回収", "複回収"], prow)
    # 重賞ごと
    by = {}
    for x in cells:
        by.setdefault(x["rkey"], []).append(x)
    rrow = []
    for rk in sorted(P["base"]):
        b = P["base"][rk]
        cs = sorted(by.get(rk, []), key=lambda x: x["pbin"])[:TOP_PER_RKEY]
        top = "<br>".join(f"{cond(x)} {rec(x['rec'])} 再現率 {pct(x['recall'])}・リフト {f2(x['lift'])}・単 {pct(x['roi_t'])}・複 {pct(x['roi_f'])}" for x in cs)
        npass = len(by.get(rk, []))
        rrow.append(f"<tr><td>{escape(rk)}</td><td class='n'>{b['N']}</td><td class='n'>{b['K']}</td><td class='n'>{pct(b['K'] / b['N'] if b['N'] else None)}</td>"
                    f"<td class='n'>{npass}</td><td class='wrap muted'>{top}</td></tr>")
    rtab = table(["重賞", "分母（6 人気以下の走）", "うち 3 着内", "分母の複勝率", "採る条件（同じ馬の集まりをまとめた後）", f"上位 {TOP_PER_RKEY} 条件（二項の確率の小さい順）"], rrow)
    # 記事の例
    exe = {(x["rkey"], x["condition"]): x for x in (E["examples_eval"] if E else [])}
    xrow = []
    for x in P["examples_dev"]:
        e = exe.get((x["rkey"], x["condition"]))
        xrow.append(f"<tr><td>{escape(x['rkey'])}</td><td>{'全出走' if x['pop_min'] is None else str(x['pop_min']) + ' 人気以下'}</td><td class='wrap'>{escape(x['condition'])}</td>"
                    f"<td class='wrap'>{escape(x['note_claim'])}</td><td class='n'>{x['k']}/{x['K']}（{x['races']} 回）</td><td class='n'>{rec(x['hit']['rec'])}</td>"
                    f"<td class='n'>{rec(x['rest']['rec'])}</td><td class='n'>{f2(x['lift'])}</td>"
                    + (f"<td class='n'>{e['k']}/{e['K']}（{e['races']} 回）</td><td class='n'>{rec(e['hit']['rec'])}</td><td class='n'>{f2(e['lift'])}</td>" if e else "<td colspan='3'>—</td>") + "</tr>")
    xtab = table(["重賞", "分母", "条件", "記事の数字", "手元 2016〜2024（再現率）", "満たす馬", "満たさない馬", "リフト", "評価 2025・2026（再現率）", "満たす馬", "リフト"], xrow)
    # 評価のまとめ
    ev_html = '<p class="muted">2025・2026 年の評価はまだ流していない。</p>'
    if E:
        def srow(nm, s, keys):
            out = []
            d = s["denominator"]
            out.append(f"<tr><td>{escape(nm)}（分母）</td><td class='n'>{s['races']}</td><td class='n'>{rec(d['rec'])}</td><td class='n'>{pct(d['rate'])}</td><td>—</td><td>—</td><td>—</td>"
                       f"<td class='n'>{pct(d['roi_t'])}</td><td class='n'>{pct(d['roi_f'])}</td></tr>")
            for k, lab in keys:
                h = s[k]
                out.append(f"<tr><td>{escape(nm)}・{escape(lab)}</td><td></td><td class='n'>{rec(h['rec'])}</td><td class='n'>{pct(h['rate'])}</td><td class='n'>{pct(h['recall'])}</td>"
                           f"<td class='n'>{pct(h['coverage'])}</td><td class='n'>{f2(h['lift'])}</td><td class='n'>{pct(h['roi_t'])}</td><td class='n'>{pct(h['roi_f'])}</td></tr>")
            return out
        rows = []
        rows += srow("重賞ごとの条件・6 番人気以下", E["by_rkey"]["pop6"], [("hit_1", "1 つ以上に当たる"), ("hit_5", "5 つ以上"), ("hit_20", "20 以上")])
        rows += srow("重賞ごとの条件・前日版の上位 5 頭を除く", E["by_rkey"]["preday_out_top5"], [("hit_1", "1 つ以上に当たる"), ("hit_5", "5 つ以上"), ("hit_20", "20 以上")])
        rows += srow("全重賞の条件・6 番人気以下", E["pooled"]["pop6"], [("hit_1", "1 つ以上に当たる"), ("hit_3", "3 つ以上")])
        rows += srow("全重賞の条件・前日版の上位 5 頭を除く", E["pooled"]["preday_out_top5"], [("hit_1", "1 つ以上に当たる"), ("hit_3", "3 つ以上")])
        K = E.get("kill_overlap")
        krows = [] if not K else [f"<tr><td>{escape(nm)}</td><td class='n'>{rec(x['rec'])}</td><td class='n'>{pct(x['rate'])}</td><td class='n'>{pct(x['roi_t'])}</td><td class='n'>{pct(x['roi_f'])}</td></tr>"
                 for nm, x in (("6 番人気以下・消しに当たる", K["pop6_killed"]), ("6 番人気以下・消しに当たらない", K["pop6_not_killed"]),
                               ("全重賞の条件に当たる穴馬・消しに当たる", K["pooled_hit_killed"]), ("全重賞の条件に当たる穴馬・消しに当たらない", K["pooled_hit_not_killed"]),
                               ("重賞ごとの条件 5 つ以上・消しに当たる", K["rkey_hit5_killed"]), ("重賞ごとの条件 5 つ以上・消しに当たらない", K["rkey_hit5_not_killed"]))]
        ev_html = (f"<p>評価: {escape(E['evaluated_at'])} に 1 回だけ流した（この分析では初回。対象は {escape(E['eval_end'])} までの 2025・2026 年の重賞の主の層）。"
                   "ただし 2025・2026 年の重賞そのものは、ほかの分析ですでに 4 度見ている。全重賞の条件は互いに独立ではない（前走 5 着以内を含む組が多く、評価で同じ馬の集まりになる組もある）。</p>"
                   + table(["分母・条件", "レース", "[1-2-3-着外]", "複勝率", "再現率", "該当率", "リフト", "単回収", "複回収"], rows)
                   + ("" if not K else f"<h2>20 サイクルの消しの規則との重なり</h2><p class='muted'>消しの規則: {escape(K['rule'])}（B・全体で 1 つ。1 レース 3 頭を消す）。"
                   f"6 番人気以下のうち消しに当たる割合 {pct(K['pop6_killed_share'])}、全重賞の条件に当たる穴馬のうち消しに当たる割合 {pct(K['pooled_hit_killed_share'])}。</p>"
                   + table(["馬の集まり", "[1-2-3-着外]", "複勝率", "単回収", "複回収"], krows)))
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>重賞の穴馬分析</title><style>{CSS}</style></head><body><main>
<h1>重賞の穴馬分析（うまうまー式、6 番人気以下を分母にした条件）</h1>
<div class="panel wrap"><p>{escape(P['note'])}</p>
<ul class="muted"><li>分母: 重賞ごとに、主の層（同じレース名 × 2016〜2024 年で一番多く行われたコース）の 6 番人気以下の全出走（凡走も含む）。人気は分母を選ぶためだけに使い、条件に使わない。</li>
<li>条件: 切り口 1〜2 個の水準の AND。採る条件: 好走 ≥ {c['K_MIN']} 頭・好走のいたレース ≥ {c['ED_MIN']}・再現率 ≥ {pct(c['RECALL_MIN'])}・リフト ≥ {c['LIFT_MIN']}。
再現率 = 好走した穴馬のうち条件を満たす割合（記事の「何頭中何頭」）。リフト = 該当馬の複勝率 ÷ 分母の複勝率。回収率は物差しだけ。</li>
<li>重賞ごと（{P['n_rkeys']} 重賞、組 {P['n_cells_by_rkey']:,}）: 採る条件を満たした組 {P['pass_real']:,}。結果を入れ替えた偶然のデータでも {min(perm):,}〜{max(perm):,} 組。BH（FDR {pct(c['FDR'])}）で残る組 {P['pass_real_bh']}。
<b>重賞ごとの条件は、偶然の当たりと区別できない。</b></li>
<li>全重賞の分母（{P['pooled_base']['N']:,} 走、うち 3 着内 {P['pooled_base']['K']:,}、組 {P['n_cells_pooled']:,}）: BH で残る組 {P['pooled_real']['bh']:,}（偶然のデータでは {', '.join(str(p['bh']) for p in P['pooled_perm'])}）。
好走 {c['POOLED_K_MIN']} 頭以上で二項の確率の小さい {c['POOLED_TOP']} 組を凍結（開発でまったく同じ馬の集まりになる組だけをまとめた。ほぼ同じ組は残る）。</li>
<li>凍結: {escape(P['frozen_at'])}（2016〜2024 年の中だけで決めた）。評価は 2025・2026 年（{escape(c['EVAL_END'])} まで）で 1 回だけ。</li></ul></div>
<h2>全重賞の条件（BH で残ったもの、上位 {len(P['pooled'])} 組）</h2>{ptab}
<h2>評価（2025・2026 年）</h2>{ev_html}
<h2>記事の例の再現</h2><p class="muted">記事の数字は取得時の要約（docs/ideas/umauma_method_notes.md）。記事は 10〜15 年、手元は 2016〜2024 年で、期間と分母が違う。</p>{xtab}
<h2>重賞ごとの条件（2016〜2024 年。偶然と区別できないので参考）</h2>{rtab}
<p class="muted">戻る: <a href="graded_handicapper.html">重賞のルールベース予想家</a>・<a href="graded_rules_table.html">重賞のルール一覧</a>・<a href="graded_cycles.html">重賞の分析サイクル</a>。
作り: src/research/graded/longshot.py → src/scripts/docs/build_graded_longshot_html.py</p>
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.2f} MB")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    build(Path(a[0]) if a else SRC, Path(a[1]) if len(a) > 1 else OUT)
