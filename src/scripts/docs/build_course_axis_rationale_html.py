"""コース別の走法軸の優位の『根拠』を、実力・人気を除いて言語化した専用 HTML を作る（読むだけ・HTML を書く）。

入力: src/scripts/data/course_axis_rationale.py の出力 JSON、data/meta/course_master/jra_courses.json。
使い方: python src/scripts/docs/build_course_axis_rationale_html.py RATIONALE.json [OUT.html]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.research.race import course_axis_rationale as car  # noqa: E402
from src.scripts.docs.build_course_axis_demand_html import CSS, SORT_JS, load_master, shape_of, table, td  # noqa: E402

OUT = REPO / "docs/html/operations/running_map_course_axis_rationale.html"
MODES = [("raw", "補正なし（現行の定義）"), ("res", "実力を除く（本命）"), ("res_pts", "実力を除く・ポイント制"), ("pop", "参考: 人気で補正")]
EXTRA_CSS = """
.ctl { display:flex; gap:14px; flex-wrap:wrap; align-items:center; margin:10px 0; } .ctl select, .ctl button { background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:5px 10px; font-size:13px; }
.ctl button.on { background:var(--accent); color:#0b1428; font-weight:700; }
.hgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:12px; } .hcell { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:8px 10px; }
.hcell b { font-size:14px; } .hcell svg { width:100%; height:auto; display:block; } .legend span { margin-right:14px; font-size:12px; color:var(--muted); }
.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
td.ok { color:var(--ok); font-weight:700; } td.no { color:var(--muted); }
"""


def corr_fit_chk(stats: dict, a: str = "fit", b: str = "y26") -> float:
    xs, ys = [], []
    for row in stats.values():
        if row.get(a) and row.get(b):
            xs += row[a]["b"]
            ys += row[b]["b"]
    return float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 3 else float("nan")


def same_sign_rate(stats: dict, t_min: float = 2.0) -> tuple[int, int]:
    n = k = 0
    for row in stats.values():
        if row.get("fit") and row.get("y26"):
            for bf, sf, bg in zip(row["fit"]["b"], row["fit"]["se"], row["y26"]["b"]):
                if abs(bf / sf) >= t_min:
                    n += 1
                    k += int(np.sign(bf) == np.sign(bg))
    return n, k


def build(d: dict, mp: dict) -> str:
    K = d["axes"]
    M = d["modes"]
    meta, dg = d["meta"], d["diag"]
    feats = {}
    for ck in M["res"]:
        ve, su, di = ck.split("|")
        f = car.shape_features(mp[(ve, su, int(di))], su, int(di))
        if f:
            feats[ck] = f
    eff = {m: car.shape_effects(M[m], feats, K) for m in ("raw", "res", "res_pts")}
    repl = [e for e in eff["res"] if e["verdict"] == "根拠あり"]
    nul_repl = sum(1 for nu in d["null"] for e in car.shape_effects(nu, feats, K) if e["verdict"] == "根拠あり")
    cc = {m: {p: car.course_counts(M[m], p, K) for p in ("fit", "y26")} for m in M}
    ncc = [{p: car.course_counts(nu, p, K) for p in ("fit", "y26")} for nu in d["null"]]
    null_fit = np.mean([c["fit"]["hits"] for c in ncc])
    null_26 = np.mean([c["y26"]["hits"] for c in ncc])
    n_tests = cc["res"]["fit"]["n"]
    fit_courses = sum(1 for r in M["res"].values() if r.get("fit"))
    chk_courses = sum(1 for r in M["res"].values() if r.get("y26"))
    ne = K.index("endure")
    r2s, r2d = dg["axis_r2_by_ability"]["芝"], dg["axis_r2_by_ability"]["ダート"]

    nm = car.AXIS_NAME
    ts_e = next((e for e in repl if e["axis"] == "ts" and e["feature"] == "距離"), None)
    concl = []
    concl.append(f"""<li><b>実力を除くと、「コースごとに優位な軸」と言える根拠は、個別のコースでは残らない</b>。実力は市場を見ない代理（その馬の出走前の自分の成績）で除いた。
コース × 軸 {n_tests} 件のうち、2024-25 年で |t| ≥ {car.T_COURSE} は {cc['res']['fit']['hits']} 件。着順をレース内で入れ替えた乱数ラベル（{d['meta']['n_shuffle']} 回）でも平均 {null_fit:.1f} 件出るので、偶然の範囲。
コース別の係数を 2024-25 年と 2026 年で比べた相関は {corr_fit_chk(M['res']):.2f}（補正なしでも {corr_fit_chk(M['raw']):.2f}）で、ほぼ再現しない。</li>""")
    if ts_e:
        concl.append(f"""<li><b>形状まで束ねると、言語化できる優位は 1 つだけ残った: 距離が長いコースほど TS（末脚の速さ）が効きやすい</b>
（400m 長いごとに係数 {ts_e['b_fit'] * 100:+.1f} 点／軸 1 標準偏差、t = {ts_e['t_fit']:+.1f}。2026 年の確認でも {ts_e['b_chk'] * 100:+.1f}、t = {ts_e['t_chk']:+.1f} と同じ向き）。
補正なしでは t = +2.8 / +3.2 とより強く、実力を除くと弱まる（実力の代理が一部含まれていた）。ポイント制でも同じ向き（fit t = +1.7、2026 t = +2.5）だが、基準 {car.T_REPL} に僅かに届かない。<b>弱い根拠</b>の位置づけ。
乱数ラベルでは、同じ基準を満たす形状 × 軸は {nul_repl} 件（{d['meta']['n_shuffle']} 回の合計）。</li>""")
    else:
        concl.append("<li>形状まで束ねても、2 つの期間で同じ向きに再現する効果はなかった。</li>")
    concl.append(f"""<li>それ以外（直線長 → TS、小回り → Cr性能・TS持続力、坂 → TS持続力・Hペース耐性など、事前に考えやすい筋）は、<b>2 つの期間で同じ向きには再現しなかった</b>（§3）。
2026 年だけに出るもの（ダートでギアチェンジ・TS持続力が効く、Hペース耐性が効きにくい）は fit では出ておらず、根拠にしない。</li>""")
    concl.append(f"""<li>Hペース耐性が実力の代理だった点は、実力を除く前後で確認できる。軸のばらつきのうち実力で説明される割合は、Hペース耐性が芝 {r2s[ne] * 100:.0f}%・ダート {r2d[ne] * 100:.0f}%（ギアチェンジ {r2s[0] * 100:.0f}% / {r2d[0] * 100:.0f}%、他は 5% 以下）。</li>""")

    parts = [f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>走法軸 × コースの適性：優位の根拠（実力・人気を除く） — keiba-vpn docs</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>走法軸 × コースの適性：優位の根拠（実力・人気を除く）</h1>
<p class="muted">作成: 2026-11-11。依頼「各コースでこの走法軸が優位、と言うなら根拠を言語化したい。人気や実力といった、適性とのつながりが薄い要素は排除か補正する」「人気の要素が強く入らないよう、適性だけで言語化」。
良馬場・2024 年以降・頭数 5 以上の {meta['n_runs']:,} 走（{meta['n_races']:,} レース）。2024-25 年（{meta['n_fit']:,} 走）で推定、2026 年（{meta['n_2026']:,} 走・テストセット②）は<b>確認のためだけ</b>に 1 回見た。
計算は <code>src/scripts/data/course_axis_rationale.py</code>（ホスト）、判定は <code>src/research/race/course_axis_rationale.py</code>、この表は <code>src/scripts/docs/build_course_axis_rationale_html.py</code>。表の見出しをクリックすると並べ替え。
前の測定は <a href="running_map_course_axis.html">走法マップ：良馬場のコース別 走法軸の寄与と根拠</a>。</p>

<div class="box"><b>結論</b><ol>{''.join(concl)}</ol>
<p class="sub">一言で言うと、<b>「このコースはこの軸が優位」と実力・人気を抜いたうえで言えるものは、いまのデータでは「距離が長いほど TS」の 1 点だけ（弱い）</b>。
個別のコースの優位は、言語化できる根拠が立たないので出さない。出すと偶然のぶれを理由づけすることになる。</p></div>

<h2>1. 実力・人気をどう扱ったか</h2>
<ul>
<li><b>人気・オッズは結果の補正にも判定にも使わない</b>（市場の評価が入るため。人気で補正した列は、比較用の診断としてだけ並べた）。</li>
<li><b>実力の代理 = その馬の「そのレースより前の走」の得点の平均</b>（1〜3 着 = 1 … 8 着以下 = 0 の得点。平均 {meta['mean_score']:.2f} へ K = {meta['shrink_k']:.0f} 走分縮小し、レース内で中心化）。結果・市場を見ない。
この代理は市場の実力（人気込みの期待 − 頭数だけの期待）と相関 {dg['ability_vs_pop_corr']['芝']:.2f}（芝）・{dg['ability_vs_pop_corr']['ダート']:.2f}（ダート）で、実力の指標として働いている。</li>
<li><b>補正の仕方</b>: 結果（得点 − 頭数だけの期待）と各軸の z を、それぞれ実力の代理へ回帰した残差にする（芝ダ別）。「強い馬ほど軸が高く、得点も高い」という関係を、軸の側と結果の側の両方から除く。</li>
<li>コース固有の差は、「同じ芝ダ × 同じクラスの平均の係数」を引いた残りを軸へ回帰（前の測定と同じ）。標準誤差はレース単位でまとめて計算。係数は 軸 1 標準偏差あたりの得点（×100 で表示）。</li>
<li><b>ポイント制</b>（ユーザー指定: 1〜3 着 = 3 点、4〜5 着 = 1 点、6 着以下 = 0 点）は、結果の点数を替えた頑健性の確認（期待は 11 ÷ 頭数）と、§5 のヒストグラムの集計に使った。</li>
<li>多重検定への備え: 着順をレース内で入れ替えた乱数ラベルを {meta['n_shuffle']} 回、同じ手順に通して「偶然で出る数」を測った。</li>
<li>2024 年は出走記録の整った走が少なく（コース × 年で 120 走以上のコースは 5 つ）、年別の再現（2024 と 2025）は組めなかった。かわりに <b>2024-25 年で推定 → 2026 年で 1 回確認</b>とした。</li>
</ul>

<h2>2. 絞り込み: どの補正で何件残るか</h2>
<p class="sub">コース × 軸の差の検定のうち |t| ≥ {car.T_COURSE} の件数。2024-25 年（{fit_courses} コース × 6 軸 = {n_tests} 件）と、2026 年（{chk_courses} コース × 6 軸 = {cc['res']['y26']['n']} 件）。
|t| の標準偏差は、偶然だけなら約 1。相関は、同じコース × 軸の係数を 2024-25 年と 2026 年で比べたもの。</p>"""]
    rows = []
    for m, lab in MODES:
        n2, k2 = same_sign_rate(M[m])
        rows.append([td(lab), td(cc[m]["fit"]["hits"], True), td(f"{cc[m]['fit']['sd']:.2f}", True), td(cc[m]["y26"]["hits"], True), td(f"{cc[m]['y26']['sd']:.2f}", True),
                     td(f"{corr_fit_chk(M[m]):+.2f}", True), td(f"{k2}/{n2}", True)])
    rows.append([td("乱数ラベル（実力を除く・平均）"), td(f"{null_fit:.1f}", True), td(f"{np.mean([c['fit']['sd'] for c in ncc]):.2f}", True), td(f"{null_26:.1f}", True), td(f"{np.mean([c['y26']['sd'] for c in ncc]):.2f}", True),
                 td(f"{np.mean([corr_fit_chk(nu) for nu in d['null']]):+.2f}", True), td("—")])
    parts.append(table(["補正", "2024-25 年の |t|≥2.3", "|t|の標準偏差", "2026 年の |t|≥2.3", "|t|の標準偏差", "係数の相関（2024-25 vs 2026）", "2024-25 で |t|≥2 のうち 2026 で同符号"], rows))
    parts.append(f"""<ul>
<li>補正なし 11 件 → 実力を除く 9 件 → ポイント制 6 件と減るが、乱数ラベルでも平均 {null_fit:.1f} 件（{min(c['fit']['hits'] for c in ncc)}〜{max(c['fit']['hits'] for c in ncc)} 件）出る。<b>偶然の数と区別できない</b>。</li>
<li>2026 年は |t| の標準偏差が 1.2 台で、乱数ラベル（1.0〜1.1）よりやや大きい。コースごとに多少の差はあるが、2024-25 年の係数とは相関 0.15 以下で、同じ向きに再現しない。</li>
</ul>
<h3>実力で説明される軸のばらつき</h3>""")
    rows = [[td(s)] + [td(f"{v * 100:.1f}%", True, data=v) for v in dg["axis_r2_by_ability"][s]] + [td(f"{dg['ability_vs_outcome_corr'][s]:+.2f}", True)] for s in ("芝", "ダート")]
    parts.append(table(["芝ダ"] + [f"軸の分散のうち実力で説明（{nm[k]}）" for k in K] + ["実力と結果の相関"], rows))
    parts.append("<p class=\"sub\">Hペース耐性は軸の分散の約 2 割が実力で説明され、ギアチェンジが 1 割前後。この分は、補正前の係数に「強い馬が強い」が混ざっていた部分。</p>")

    parts.append(f"""<h2>3. 形状ごとに束ねた検証（言語化の単位）</h2>
<p class="sub">コースを 1 つずつ見ると偶然と区別できないので、コースの形状（直線長・小回り・ダート・坂・距離）で束ね、「形状が 1 段階違うと、軸の係数がどれだけ変わるか」を、{eff['res'][0]['n_fit']} コース（2024-25 年）と {eff['res'][0]['n_chk']} コース（2026 年）で別々に推定した。
各行は t（括弧外が係数 ×100 ＝ 軸 1 標準偏差あたりの得点）。両期間で同じ向き・|t| ≥ {car.T_REPL} なら<b>根拠あり</b>、片方だけなら<b>片方の期間のみ</b>（根拠にしない）。形状は直線長 +100m、小回り、ダート、坂（ゴール前）、距離 +400m の 5 つ。
基準のコースは直線 400m・大回り・芝・坂なし・1600m。内回り・外回りで直線の長さが違うコースは形状が定まらないため除いた（{len(feats)}/{len(M['res'])} コースを使用）。</p>""")
    rows = []
    for e in eff["res"]:
        ex = {m: next(x for x in eff[m] if x["axis"] == e["axis"] and x["feature"] == e["feature"]) for m in ("raw", "res_pts")}
        rows.append([td(nm[e["axis"]]), td(e["feature"]), td(f"{e['b_fit'] * 100:+.1f} ({e['t_fit']:+.1f})", True, data=f"{e['t_fit']:.2f}"),
                     td(f"{e['b_chk'] * 100:+.1f} ({e['t_chk']:+.1f})", True, data=f"{e['t_chk']:.2f}"),
                     f'<td class="{"ok" if e["verdict"] == "根拠あり" else "no"}">{e["verdict"]}</td>',
                     td(f"{ex['raw']['t_fit']:+.1f} / {ex['raw']['t_chk']:+.1f}", True), td(f"{ex['res_pts']['t_fit']:+.1f} / {ex['res_pts']['t_chk']:+.1f}", True)])
    parts.append(table(["軸", "形状", "2024-25 年 係数×100 (t)", "2026 年 係数×100 (t)", "判定（実力を除く）", "補正なしの t（2024-25 / 2026）", "ポイント制の t（2024-25 / 2026）"], rows))
    parts.append(f"""<p class="sub">30 件の検定を 2 期間で見ているので、片方の期間だけで |t| ≥ 2 になるものが数件は出る（偶然）。乱数ラベルで同じ基準を満たしたのは 5 回の合計 {nul_repl} 件。
ただし事前に仮説を絞らずに 30 件を眺めた結果なので、残った 1 件も<b>確定ではなく「次の期間で再確認する候補」</b>。</p>""")

    parts.append("""<h2>4. 根拠を言語化できるもの／できないもの</h2>""")
    if ts_e:
        parts.append(f"""<div class="box"><b>根拠あり（弱い）: 距離が長いコースほど TS（末脚の速さ）が効きやすい</b>
<ul>
<li>数字: 400m 長いごとに、TS が 1 標準偏差高い馬の得点が +{ts_e['b_fit'] * 100:.1f}（2024-25 年, t = {ts_e['t_fit']:+.1f}）／ +{ts_e['b_chk'] * 100:.1f}（2026 年, t = {ts_e['t_chk']:+.1f}）。実力を除く前（補正なし）は t = +2.8 / +3.2。</li>
<li>理由として考えられる筋（<b>未検証の解釈</b>。この分析はラップを見ていない）: 長い距離ほど道中のペースが緩み、直線の入りからの脚の使い方で差がつきやすい。
一方でスプリント戦は前半から速く、末脚の速さの価値が相対的に小さい。</li>
<li>使い方: 「距離が伸びるほど TS を評価する」という方向の補正の候補。<b>個別のコースの「優位」とは言わない</b>（個別のコースの係数は偶然のぶれが大きい）。大きさも小さい（上記の値）ので、印を動かす根拠にはならない。</li>
</ul></div>""")
    parts.append("""<div class="box"><b>根拠なし（出さない）</b>
<ul>
<li>直線が長いコースで TS が効く、小回りで Cr性能・TS持続力が効く、坂のあるコースで TS持続力・Hペース耐性が効く、といった事前に筋の通る話は、どちらの期間でも再現しなかった。</li>
<li>個別のコースで「実測で目立つ」ものは、形状で説明できず、他の期間でも再現しないため、根拠文にしない。下の表では<b>「実測で目立つが再現なし」</b>と区別して示す。</li>
</ul></div>
<h3>コース別の言語化</h3>
<p class="sub">「形状から言えること」は上の根拠ありの効果だけから作った文。実測の欄は、そのコースの 2024-25 年（t）と 2026 年（t）の TS の係数（実力を除く）。
「実測で目立つが再現なし」は、いずれかの期間で |t| ≥ 2.3 の軸（同符号で両期間とも |t| ≥ 1.9 なら再現あり）。</p>""")
    rows = []
    ref_effects = {e["axis"]: e for e in repl if e["feature"] == "距離"}
    for ck in sorted(M["res"], key=lambda c: (c.split("|")[1] != "芝", c.split("|")[0], int(c.split("|")[2]))):
        row = M["res"][ck]
        if not (row.get("fit") or row.get("y26")):
            continue
        ve, su, di = ck.split("|")
        sh = shape_of(mp[(ve, su, int(di))])
        texts = []
        for ax, e in ref_effects.items():
            if ck in feats:
                shift = e["b_fit"] * feats[ck][5] * 100
                dist_txt = f"{di}m"
                texts.append(f"{dist_txt}で基準（1600m）より{'長い' if feats[ck][5] > 0 else ('短い' if feats[ck][5] < 0 else '同じ')}ため、{nm[ax]}は{'効きやすい' if shift > 0.05 else ('効きにくい' if shift < -0.05 else '基準並み')}（予想 {shift:+.1f}）")
        odd = []
        for j, k in enumerate(K):
            ts = [(p, row[p]["b"][j] / row[p]["se"][j]) for p in ("fit", "y26") if row.get(p)]
            if any(abs(t) >= car.T_COURSE for _, t in ts):
                both = len(ts) == 2 and np.sign(ts[0][1]) == np.sign(ts[1][1]) and all(abs(t) >= car.T_REPL for _, t in ts)
                odd.append(f"{nm[k]}{'（再現あり）' if both else '（再現なし）'} " + "/".join(f"{t:+.1f}" for _, t in ts))
        tsj = K.index("ts")
        tcell = " / ".join(f"{row[p]['b'][tsj] * 100:+.1f} ({row[p]['b'][tsj] / row[p]['se'][tsj]:+.1f})" if row.get(p) else "—" for p in ("fit", "y26"))
        rows.append([td(f"{ve}{su}{di}m"), td(sh["straight"], True, data=sh["st0"] if sh["st0"] == sh["st0"] else 0), td("小回り" if sh["small"] else "—"), td(sh["slope"]),
                     td(" ／ ".join(texts) if texts else ("形状が定まらない（内外回り）" if ck not in feats else "形状から言える優位はない")), td(tcell, True), td("、".join(odd) if odd else "—")])
    parts.append(table(["コース", "直線(m)", "カーブ", "坂", "形状から言えること", "TS の係数×100 (t) 2024-25 / 2026", "実測で目立つが再現なし／ありの軸 (t)"], rows))

    # ---- ヒストグラム
    parts.append(f"""<h2>5. コース × 走法軸の「好走馬の数」の分布（ヒストグラム）</h2>
<p class="sub">依頼「各コース × 各走法軸に対しての好走馬カウントの分布ヒストグラム」。良馬場・2024 年以降（2026 年を含む記述の図）。横軸 = その軸の値の帯（芝ダ別に標準化した z。左ほどその特徴が弱く、右ほど強い）。
棒（青）= その帯の<b>実際の好走数</b>、枠線（灰）= 頭数だけから見込まれる好走数（<b>ランダムな着順なら</b>この数）。青が枠線より高い帯は、その軸の値の馬が期待以上に好走している帯。
数は 3 着以内の頭数、またはポイント制（1〜3 着 = 3 点、4〜5 着 = 1 点、6 着以下 = 0 点）の合計から選べる。帯の下の数字は出走数。<b>走数の少ない帯はぶれが大きい</b>（出走数 30 未満の帯は薄く表示）。</p>
<div class="ctl">
<label>コース <select id="hc"></select></label>
<span><button id="m_good" class="on">3着以内の数</button> <button id="m_pts">ポイント制</button></span>
<span><button id="z_raw" class="on">軸の値そのまま</button> <button id="z_res">実力を除いた軸の値</button></span>
</div>
<div class="legend"><span><i class="sw" style="background:#4f8bff"></i>実際</span><span><i class="sw" style="background:transparent;border:1px solid #9aa7c7"></i>頭数だけから見込まれる数（ランダム）</span></div>
<div id="hg" class="hgrid"></div>
<p class="sub">「全コース（芝）」「全コース（ダート）」は、コースの合計。実力を除いた軸の値では、強い馬ほど軸が高いという偏りが除かれる。ヒストグラムは分布を見るための記述で、検定ではない（検定は §2〜§3）。</p>""")
    hist = {"raw": d["hist"], "res": d["histr"]}
    payload = json.dumps({"axes": K, "names": [nm[k] for k in K], "hist": hist, "edges": d["edges"]}, ensure_ascii=False, separators=(",", ":"))
    parts.append(f"""<script>var HD={payload};</script>
<script>
(function(){{
var cks=Object.keys(HD.hist.raw).sort(function(a,b){{var x=a.split("|"),y=b.split("|");return (x[1]===y[1]?0:(x[1]==="芝"?-1:1))||x[0].localeCompare(y[0],"ja")||(+x[2])-(+y[2]);}});
var sel=document.getElementById("hc"),st={{m:"good",z:"raw"}};
var opts=[["ALL|芝","全コース（芝）"],["ALL|ダート","全コース（ダート）"]].concat(cks.map(function(c){{var p=c.split("|");return [c,p[0]+p[1]+p[2]+"m"];}}));
opts.forEach(function(o){{var e=document.createElement("option");e.value=o[0];e.textContent=o[1];sel.appendChild(e);}});
var lab=["〜-1.5","-1.5〜-1","-1〜-0.5","-0.5〜0","0〜0.5","0.5〜1","1〜1.5","1.5〜"];
function get(ck,ax){{
  var H=HD.hist[st.z],keys=ck.indexOf("ALL|")===0?cks.filter(function(c){{return c.split("|")[1]===ck.split("|")[1];}}):[ck];
  var r={{all:[0,0,0,0,0,0,0,0],obs:[0,0,0,0,0,0,0,0],ex:[0,0,0,0,0,0,0,0]}};
  keys.forEach(function(k){{var h=H[k][ax];for(var i=0;i<8;i++){{r.all[i]+=h.all[i];r.obs[i]+=st.m==="good"?h.good[i]:h.pts[i];r.ex[i]+=st.m==="good"?h.ex[i]:h.expts[i];}}}});
  return r;}}
function svg(r){{
  var W=360,Hh=170,l=34,b=34,t=8,w=(W-l-6)/8,mx=Math.max.apply(null,r.obs.concat(r.ex).concat([1]))*1.12,s="";
  s+='<line x1="'+l+'" y1="'+(Hh-b)+'" x2="'+(W-4)+'" y2="'+(Hh-b)+'" stroke="#243358"/>';
  for(var k=0;k<=2;k++){{var v=mx*k/2,y=Hh-b-(Hh-b-t)*k/2;s+='<text x="'+(l-4)+'" y="'+(y+3)+'" font-size="9" fill="#9aa7c7" text-anchor="end">'+Math.round(v)+'</text>';}}
  for(var i=0;i<8;i++){{
    var x=l+i*w,ho=(Hh-b-t)*r.obs[i]/mx,he=(Hh-b-t)*r.ex[i]/mx,thin=r.all[i]<30;
    s+='<rect x="'+(x+3)+'" y="'+(Hh-b-ho)+'" width="'+(w-6)+'" height="'+ho+'" fill="#4f8bff" opacity="'+(thin?0.35:0.95)+'"/>';
    s+='<rect x="'+(x+1.5)+'" y="'+(Hh-b-he)+'" width="'+(w-3)+'" height="'+he+'" fill="none" stroke="#9aa7c7" stroke-width="1"/>';
    s+='<text x="'+(x+w/2)+'" y="'+(Hh-b+11)+'" font-size="7.5" fill="#9aa7c7" text-anchor="middle">'+lab[i]+'</text>';
    s+='<text x="'+(x+w/2)+'" y="'+(Hh-b+22)+'" font-size="8.5" fill="#e6ecff" text-anchor="middle">'+r.all[i]+'</text>';
  }}
  return '<svg viewBox="0 0 '+W+' '+Hh+'" xmlns="http://www.w3.org/2000/svg">'+s+'</svg>';}}
function draw(){{
  var ck=sel.value,h="";
  HD.axes.forEach(function(ax,j){{var r=get(ck,ax),so=0,se=0;for(var i=0;i<8;i++){{so+=r.obs[i];se+=r.ex[i];}}
    h+='<div class="hcell"><b>'+HD.names[j]+'</b> <span class="muted">計 '+Math.round(so)+'（見込み '+Math.round(se)+'）</span>'+svg(r)+'</div>';}});
  document.getElementById("hg").innerHTML=h;
  ["m_good","m_pts","z_raw","z_res"].forEach(function(id){{var on=(id==="m_"+st.m)||(id==="z_"+st.z);document.getElementById(id).className=on?"on":"";}});}}
sel.onchange=draw;
document.getElementById("m_good").onclick=function(){{st.m="good";draw();}};document.getElementById("m_pts").onclick=function(){{st.m="pts";draw();}};
document.getElementById("z_raw").onclick=function(){{st.z="raw";draw();}};document.getElementById("z_res").onclick=function(){{st.z="res";draw();}};
draw();
}})();
</script>""")

    parts.append(f"""<h2>6. 限界と次の一手</h2>
<ul>
<li>実力の代理は「過去走の得点の平均」で、クラスの違い・馬場・展開の影響が残る。実力の指標を変えると（例: タイム指数）、除かれる量が変わる可能性がある。ここでは市場を見ない代理を優先した。</li>
<li>2024 年は整った走が少なく、2024-25 年の推定は実質 2025 年中心（{meta['n_fit']:,} 走）。コース × 軸の標準誤差は大きく、差が小さいコースは検出できない。<b>「効いていない」ではなく「この量では検出できない」</b>と読む。</li>
<li>形状は 5 つの粗い指標。ゴール前の坂の高さ、初角までの距離、スタート形状、コーナーの数などは、コース数（約 40〜60）に対して多すぎるため入れていない。</li>
<li>2026 年はテストセット②。ここでの使い方は「確認 1 回」で、選択には使っていない。確認後の再推定（2024-26 年をまとめる）は、これから追加される走で改めて再確認してから。</li>
<li>画面（走法マップ・コース特注ゾーン）への反映は今回していない。反映するなら「距離が伸びるほど TS を評価」のような形状レベルの補正が候補で、コース単位の優位は出さない。</li>
<li>再現: ホストで <code>cd /home/jovyan/work/keiba-vpn &amp;&amp; nice -n 19 /opt/conda/bin/python -I src/scripts/data/course_axis_rationale.py OUT.json</code> → 手元で <code>python src/scripts/docs/build_course_axis_rationale_html.py OUT.json</code>。</li>
</ul>
{SORT_JS}
</main></body></html>""")
    return "".join(parts)


def main() -> None:
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out.write_text(build(d, load_master()), encoding="utf-8")
    print("wrote", out, f"{out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
