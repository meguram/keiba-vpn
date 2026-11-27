"""成果の HTML（docs/html/operations/final/day_pedigree.html）を、出力の JSON から作る。着順を新しく読まない。

    python -m src.research.day_pedigree.report
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

from src.research.day_pedigree import data as dd

ROOT = Path(__file__).resolve().parents[3]
DST = ROOT / "docs/html/operations/final/day_pedigree.html"
SURF = {"T": "芝", "D": "ダート"}
FAM = {"A": "A 適性 6 次元（主）", "S_special": "S 特殊な祖先 10 頭", "S_rule": "S 指定どおりの 5 頭", "C": "C 共通の血の濃さ",
       "M": "M 中系統（父の父）5 札", "B": "B 母父の系統 5 札"}


def e(x) -> str:
    return html.escape(str(x))


def nm(s: str) -> str:
    return e(s.replace("\xa0", " "))


def f(x, d=3) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{d}f}" if d else f"{x}"


def width_svg(width: dict, s: str) -> str:
    """k ごとの 80% 区間の幅の「朝の見立て（事前）との比」= √(事後の分散 / 事前の分散)。(a) 表示と (b) T-60 の 2 本。
    width80 の k ごとの値は、k 本以上ある日だけに絞られて事前の幅が変わるので使わない（pedigree-2 の指摘）。"""
    W, H, L, B = 560, 220, 48, 30
    series = [(f"{s}_S_special_a_show", "特殊な祖先・(a) 発走の時点", "var(--c1)", ""),
              (f"{s}_S_special_b_t60", "特殊な祖先・(b) T-60", "var(--c1)", "5 4"),
              (f"{s}_A_a_show", "A 適性・(a) 発走の時点", "var(--c2)", ""),
              (f"{s}_A_b_t60", "A 適性・(b) T-60", "var(--c2)", "5 4")]
    ymin, ymax = 94.0, 100.3
    kx = lambda k: L + (k - 1) * (W - L - 16) / 7
    yy = lambda v: 10 + (ymax - v) * (H - B - 10) / (ymax - ymin)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{SURF[s]}の区間の幅の推移" style="width:100%;max-width:{W}px">']
    for v in (94, 96, 98, 100):
        out.append(f'<line x1="{L}" x2="{W-16}" y1="{yy(v):.1f}" y2="{yy(v):.1f}" stroke="var(--border)"/>'
                   f'<text x="{L-6}" y="{yy(v)+4:.1f}" font-size="11" text-anchor="end" fill="var(--muted)">{v}%</text>')
    for k in range(1, 9):
        out.append(f'<text x="{kx(k):.1f}" y="{H-10}" font-size="11" text-anchor="middle" fill="var(--muted)">{k}R目</text>')
    leg = []
    for i, (key, lab, col, dash) in enumerate(series):
        rows = width.get(key, [])
        if not rows:
            continue
        pts = " ".join(f"{kx(r['k']):.1f},{yy(100 * math.sqrt(r['var_ratio'])):.1f}" for r in rows)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-dasharray="{dash}"/>')
        leg.append(f'<span style="white-space:nowrap"><svg width="26" height="10"><line x1="0" x2="26" y1="5" y2="5" stroke="{col}" '
                   f'stroke-width="2" stroke-dasharray="{dash}"/></svg> {e(lab)}</span>')
    out.append("</svg>")
    return "".join(out) + '<p class="sub" style="display:flex;flex-wrap:wrap;gap:4px 16px">' + "".join(leg) + "</p>"


def main() -> None:
    band = json.loads((dd.OUT / "features_band.json").read_text(encoding="utf-8"))
    chk = json.loads((dd.OUT / "check_select.json").read_text(encoding="utf-8"))
    des = json.loads((dd.OUT / "describe.json").read_text(encoding="utf-8"))
    ev = chk["eval"]
    m = ev["main"]
    pw = chk["power_for_30"]
    # 数字は JSON から出す（固定の文にしない。critic の 4・5・11・13）
    taus = [math.sqrt(t) for v in chk["tau2"].values() if v for t in v if t > 0]
    tau_lo, tau_hi = min(taus), max(taus)
    k6 = [math.sqrt(r["var_ratio"]) for key, rows in des["width"].items() if key.endswith("_a_show") for r in rows if r["k"] == 6]
    w_lo, w_hi = 100 * min(k6), 100 * max(k6)
    nr = {}
    for s_ in ("T", "D"):
        rows = des["width"][f"{s_}_A_a_show"]
        cnt = {r["k"]: r["races"] for r in rows}
        days = cnt[1]
        nr[s_] = sum(cnt.values()) / days
    zero_share = 1 - ev["rows_with_prev"] / ev["rows"]
    a_hi = ev["boot_alpha_ci"][1]
    P = []
    P.append(f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="その日のレースの結果から、その日に好走している血統の特徴を読み、後のレースに当てられるかを確かめた研究（day-pedigree）。">
<title>その日の血統の傾向</title><style>
:root {{ --bg:#f7f8fb; --panel:#fff; --panel2:#eef1f7; --border:#d5dbe7; --text:#1b2333; --muted:#5b6578; --accent:#2f5fc4;
  --ok:#1d7f45; --warn:#9a6200; --bad:#b4232c; --c1:#2f5fc4; --c2:#c2571a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60;
  --text:#e6ecff; --muted:#9aa7c7; --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --c1:#7aa9ff; --c2:#fb923c; }} }}
:root[data-theme="dark"] {{ --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60; --text:#e6ecff; --muted:#9aa7c7;
  --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --c1:#7aa9ff; --c2:#fb923c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.7 "Noto Sans JP",system-ui,sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:28px 16px 80px; }}
h1 {{ font-size:24px; margin:0 0 6px; line-height:1.4; }}
h2 {{ font-size:19px; margin:36px 0 6px; padding-top:14px; border-top:1px solid var(--border); }}
h3 {{ font-size:16px; margin:20px 0 4px; }}
p, li {{ overflow-wrap:anywhere; }}
.sub {{ color:var(--muted); font-size:13.5px; margin:0 0 8px; }}
.scroll {{ overflow-x:auto; margin:8px 0 14px; border:1px solid var(--border); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; min-width:560px; font-size:13px; background:var(--panel); }}
th, td {{ border-bottom:1px solid var(--border); border-right:1px solid var(--border); padding:5px 8px; text-align:left; vertical-align:top; }}
td.n {{ text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }}
th {{ background:var(--panel2); color:var(--muted); font-weight:600; white-space:nowrap; }}
.box {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px 16px; margin:12px 0; }}
.tag {{ display:inline-block; font-size:12px; font-weight:700; padding:0 8px; border-radius:9px; border:1px solid var(--warn); color:var(--warn); }}
.bad {{ color:var(--bad); font-weight:700; }} .mu {{ color:var(--muted); }}
code {{ font-family:ui-monospace,Menlo,monospace; background:var(--panel2); padding:1px 5px; border-radius:5px; font-size:12px; }}
</style></head><body><main>
<h1>その日の血統の傾向（day-pedigree）</h1>
<p class="sub">最終更新: 2026-11-27 ・ 担当 day-pedigree ・ 開発の行 2016〜24 年の探す 70% だけ ・ 費用 0 円（手元のデータだけ）</p>
<div class="box"><b>結論</b>
<ul>
<li>その日の前のレースの結果から読んだ血統の傾向は、後のレースで市場の見込みに上乗せしませんでした。上乗せの係数 α は {m['alpha']:+.3f}（標準誤差 {m['se']:.2f}）で、4 つの合否はすべて不合格です。</li>
<li>確かめの 30% の側は開けずに閉じました。開ける前の検出力が {pw['power']:.2f} で、決まりの 0.5 に届かなかったためです。</li>
<li>「レースが進むほど確度が上がる」は、数字ではほとんど起きません。1 日・1 場・1 面のレースは平均で芝 {nr['T']:.1f} 本・ダート {nr['D']:.1f} 本です。第 6 レースの時点（前のレース 5 本を見た後）でも、80% 区間の幅は朝の見立ての約 {w_lo:.0f}〜{w_hi:.0f}% のままです（日の揺れが 0 でない特徴だけの平均）。</li>
<li>過去の日から見た、血統の日の効きの揺れ（τ）は、0 でない特徴で {tau_lo:.2f}〜{tau_hi:.2f} です（5 着内のロジットで 1 標準偏差の特徴あたり）。この大きさを 1 日の数レースで見分けるには、情報が大きく足りません。</li>
</ul>
<p class="sub">買い目・回収率は出していません（市場の上の情報の検定・rules-council の決まり）。会員の画面・本番には出していません。</p></div>
""")
    P.append("""<h2>何をしたか</h2>
<ul>
<li>ある日・ある場・芝かダート（障害は入れない）で、第 k レースの時点までに確定したレースの結果を見ます。</li>
<li>5 着以内に来た馬の血統の特徴が、市場の見込み（確定の単勝から出した 5 着内の確率）より多く出ていれば、その日はその血が来やすいと読みます。</li>
<li>読み方はベイズの更新です。事前は「その場・芝ダのふだんの値」で、レースを見るたびに事後の平均と区間を直します。</li>
<li>時点は 2 本です。(a) 表示: 第 k レースの発走の時点で確定しているレース全部。(b) 検定: 第 k レースの発走 60 分前（T-60）までに確定したレースだけ（rules-council の決まり）。確定は発走の 15 分後とみなしました。</li>
</ul>
""")
    # 祖先の一覧
    P.append("<h2>見た血統の特徴</h2>")
    P.append('<p>登場率 p は、「母の側の牡の祖先のうち、母父の線を除いた 3〜5 代の位置」に、その祖先を持つ馬の割合です（前の年のその芝ダの出走馬）。'
             'たとえばサンデーサイレンスは、この位置では p 約 0.11 なので「特殊」の側に入ります（父や母父としての割合ではありません）。</p>')
    P.append("<ul><li>共通: 1 レースに平均 2 頭以上いる祖先（p ≥ 約 0.14）。1 本の「共通の血の濃さ」にまとめました。</li>"
             "<li>特殊: それより少なく、1 日の中で見分けられる力（検出力）が一定以上の祖先のうち、上位 10 頭。値は血量 × log(1/p) です。</li>"
             "<li>祖先の一覧は着順を見る前に決めて commit しました。</li></ul>")
    for s in ("T", "D"):
        b = band["S"][s]
        rows = "".join(f"<tr><td>{nm(r['name'])}</td><td class='n'>{r['p_mean']:.3f}</td><td class='n'>{r['pow_log3']:.2f}</td></tr>" for r in b["special"])
        com = "、".join(f"{nm(r['name'])}（{r['p_mean']:.2f}）" for r in b["common_by_count"])
        P.append(f"<h3>{SURF[s]}</h3><div class='scroll'><table><tr><th>特殊な祖先</th><th>登場率 p</th><th>日の検出力（オッズ比 3 倍の効き）</th></tr>{rows}</table></div>"
                 f"<p class='sub'>共通の祖先: {com}</p>")
    P.append("<p>一覧の読み方の注意:</p><ul>"
             "<li>10 頭は互いに独立ではありません。リファール・ヌレイエフ・ダンジグはノーザンダンサーの子で、ニアークティックはノーザンダンサーの父です。「10 頭の別々の傾向」とは読めません。</li>"
             "<li>Understanding は、サンデーサイレンスの母 Wishing Well の父です（血統の記録で確かめました）。サンデーサイレンスが母父の馬では、その 2 代上としてこの位置に入るので、p が 0.34 と高く出ます。</li>"
             "<li>「特殊」は、1 日の中で見分けられる力で選んだ結果です。血統の理論の上の特別な祖先ではありません。サンデーサイレンス（p 0.11）は、共通の線（0.14）のすぐ下にいたので入りました。</li>"
             "<li>位置は「母の側の牡の祖先」で、牝祖（母系の底）ではありません。名前は血統の記録の表記のままです。</li></ul>"
             "<p class='sub'>そのほか、A 適性の 6 次元（道悪・先行・パワー・瞬発・芝ダ寄り・距離。aptitude-group と同じ値）・中系統（父の父）・母父の系統も見ました。</p>")
    # 確度の上がり方
    P.append('<h2>レースが進むと確度は上がるか <span class="tag">記述・未確認</span></h2>')
    P.append("<p>図は、第 k レースの時点の 80% 区間の幅を、朝の見立て（事前）の幅を 100% として並べたものです。実線が (a) 発走の時点、破線が (b) T-60 です。"
             "縦軸は 94〜100% だけを切り出しています。どの線も 100% のすぐ下にとどまり、ほぼ朝の見立てのままです。</p>")
    for s in ("T", "D"):
        P.append(f"<h3>{SURF[s]}</h3>" + width_svg(des["width"], s))
        ra = des["width"][f"{s}_S_special_a_show"]
        rb = {r["k"]: r for r in des["width"][f"{s}_S_special_b_t60"]}
        rm = {r["k"]: r for r in des["width"][f"{s}_M_a_show"]}
        rows = "".join(f"<tr><td class='n'>{r['k']}</td><td class='n'>{r['races']:,}</td><td class='n'>{r['prev_mean']:.1f}</td>"
                       f"<td class='n'>{rb[r['k']]['prev_mean']:.1f}</td><td class='n'>{100 * math.sqrt(r['var_ratio']):.1f}%</td>"
                       f"<td class='n'>{100 * math.sqrt(rb[r['k']]['var_ratio']):.1f}%</td>"
                       f"<td class='n'>{100 * math.sqrt(rm[r['k']]['var_ratio']):.1f}%</td></tr>" for r in ra if r['k'] in rm and r['k'] in rb)
        P.append(f"<div class='scroll'><table><tr><th>第 k レース</th><th>レース数</th><th>使った前のレース (a)</th><th>使った前のレース (b)</th>"
                 f"<th>特殊な祖先の幅（朝の見立てとの比）(a)</th><th>同じ (b)</th><th>中系統の幅（朝の見立てとの比）(a)</th></tr>{rows}</table></div>")
    P.append("<p>読み方:</p><ul><li>幅は、朝の見立てに対する比です。いちばん狭まる中系統（0/1 の札）でも、4〜5 本を見た後で約 97〜98% です。</li>"
             "<li>区間がほとんど狭まらないのは、1 レースの情報がごく小さいからです。1 頭の祖先の血量は小さく（4 代目なら 1/16）、5 着内に入る馬は 1 レースに 5 頭だけです。</li>"
             "<li>区間の幅を半分にするには、同じ場・芝ダの 200〜300 レース（数十日分）が要ります（pedigree-2 の見積もり）。</li>"
             "<li>幅の比は、日の揺れが 0 でない特徴だけで平均しました。揺れが 0 の特徴は、事前で固定されて動きません。</li>"
             "<li>φ（ばらつきの直し）は、A 適性の 6 次元から出した値を全部の族に使いました。0/1 の札や祖先の列のばらつきは別に見ていません。</li>"
             "<li>幅の単位は族ごとに違うので、族の間で幅を比べないでください。</li></ul>")
    # 例の日
    P.append('<h2>画面の形の例 <span class="tag">記述・未確認</span></h2>')
    P.append("<p>探す 70% の側の日から無作為に選んだ日です。値は、その祖先を 4 代目に 1 回持つ馬（血量 1/16）の、5 着内のオッズ比の「その日の分」（ふだんの値を引いた分）と 80% 区間です。"
             "区間が 1.00 だけの祖先は、ふだんの日にも日による揺れが見えなかった祖先（事前の揺れ 0）です。</p>")
    for s in ("T", "D"):
        if s not in des["examples"] or not des["examples"][s]:
            continue
        ex = des["examples"][s][0]
        pm = {r["name"]: r["p_mean"] for r in band["S"][s]["special"]}
        names = ex["names"]
        head = "".join(f"<th>{nm(n_)}</th>" for n_ in names[:6])
        body = []
        for r in ex["races"]:
            cells = []
            for j, n_ in enumerate(names[:6]):
                xt = (1 / 16) * math.log(1 / max(pm.get(n_, 0.1), 1e-3))
                cells.append(f"<td class='n'>{math.exp(r['mean'][j] * xt):.2f}<br><span class='mu'>{math.exp(r['lo'][j] * xt):.2f}〜{math.exp(r['hi'][j] * xt):.2f}</span></td>")
            body.append(f"<tr><td class='n'>{r['k']}R目</td><td class='n'>{r['prev']}</td>{''.join(cells)}</tr>")
        d0 = ex["day"].split("_")
        P.append(f"<h3>{e(d0[0][:4])}-{e(d0[0][4:6])}-{e(d0[0][6:])} {e(d0[1])} {SURF[s]}</h3><div class='scroll'><table><tr><th>時点</th><th>使った前のレース</th>{head}</tr>{''.join(body)}</table></div>")
    # 主の検定
    P.append("<h2>市場に上乗せするか（主の検定）</h2>")
    P.append("<p>A 適性の 6 次元をまとめた「その日の点」を、(b) T-60 の時点で作り、第 2 レース以降の 5 着内を市場の見込みより当てるかを見ました（2017〜24 年・探す 70%）。</p>")
    p1, p2 = ev["period_2017-20"], ev["period_2021-24"]
    ps = ev["pass"]
    ok = lambda b: "通過" if b else "<span class='bad'>不合格</span>"
    P.append(f"""<div class="scroll"><table><tr><th>項目</th><th>値</th><th>合否の線</th><th>判定</th></tr>
<tr><td>上乗せの係数 α（95% 区間）</td><td class="n">{m['alpha']:+.3f}（{ev['boot_alpha_ci'][0]:+.2f}〜{ev['boot_alpha_ci'][1]:+.2f}）</td><td>下限 &gt; 0</td><td>{ok(ps['alpha_lo_gt_0'])}</td></tr>
<tr><td>本物 − 日の入れ替えの帰無（95% 区間）</td><td class="n">{m['alpha'] - ev['null_i']['mean']:+.3f}（{ev['boot_diff_i_ci'][0]:+.2f}〜{ev['boot_diff_i_ci'][1]:+.2f}）</td><td>下限 &gt; 0</td><td>{ok(ps['real_minus_null_i_lo_gt_0'])}</td></tr>
<tr><td>札の入れ替えの帰無の 95% 点</td><td class="n">{ev['null_ii']['q95']:+.3f}</td><td>α がこれを越える</td><td>{ok(ps['gt_null_ii_q95'])}</td></tr>
<tr><td>2 期（2017〜20・2021〜24）</td><td class="n">{p1['alpha']:+.3f}・{p2['alpha']:+.3f}</td><td>両方 &gt; 0</td><td>{ok(ps['both_periods_pos'])}</td></tr>
<tr><td>30% を開ける前の検出力（α を半分に割り引く）</td><td class="n">{pw['power']:.3f}</td><td>0.5 以上</td><td><span class='bad'>開けずに閉じる</span></td></tr>
</table></div>
<p class="sub">判定のレース {ev['races']:,}・行 {ev['rows']:,}。T-60 で使えた前のレースは平均 {ev['mean_prev']:.1f} 本。帰無は (i) {ev['null_i']['n']} 回・(ii) {ev['null_ii']['n']} 回、区間は開催の単位のブートストラップ。</p>""")
    # 族ごとの記述
    P.append('<h2>族ごとの記述 <span class="tag">記述・未確認</span></h2><p>合否には使いません。どの族も 95% 区間が 0 をまたぎます（α は族の値の尺度に依るので、族の間で大きさを比べられません）。</p>')
    rows = []
    for s in ("T", "D"):
        for k_, lab in FAM.items():
            a = des["alpha"].get(f"{s}_{k_}")
            if not a:
                continue
            al = a["all"]
            rows.append(f"<tr><td>{SURF[s]}</td><td>{e(lab)}</td><td class='n'>{al['alpha']:+.2f} ± {1.96 * al['se']:.2f}</td>"
                        + "".join(f"<td class='n'>{a[b]['alpha']:+.2f} ± {1.96 * a[b]['se']:.2f}</td>" for b in ("2〜3", "4〜5", "6〜")) + "</tr>")
    P.append("<div class='scroll'><table><tr><th>面</th><th>族</th><th>α ± 95%</th><th>2〜3R目</th><th>4〜5R目</th><th>6R目〜</th></tr>" + "".join(rows) + "</table></div>"
             "<p class='sub'>区間は全部 0 をまたぎます。約 48 個を並べているので、1〜2 個が大きく見えるのは偶然の範囲です。</p>")
    P.append(EXPERTS_HTML.format(ev=ev, a_hi=a_hi, zero=100 * zero_share, z1=(1 - m['alpha']) / m['se']))
    P.append("""<h2>決まりと置き場</h2><ul>
<li>データは開発の行（2016〜24 年）の探す 70% だけです。2016 年は事前を作るためだけに使いました。2025・26 年は読んでいません。GCS は読まず、費用は 0 円です。</li>
<li>分け方は aptitude-group と同じです（開催の単位・seed 20261127）。30% の側は開けていません。</li>
<li>見た回数の台帳: 系統 day_pedigree（KEIBA_SESSION=day-pedigree）。</li>
<li>道具: <code>src/research/day_pedigree/</code>。記録: <code>docs/operations/day_pedigree/log.md</code>。</li>
<li>確かめの本番は前向きの記録（2026-11-28 以降）だけです。買い方に進めるときは、ルールの会議を通します。</li>
</ul></main></body></html>""")
    DST.write_text("".join(P), encoding="utf-8")
    print("書いた", DST)


EXPERTS_HTML = """<h2>なぜ見えなかったか（専門家の見立て）</h2>
<ul>
<li>主の理由は、1 日の情報の少なさです。T-60 で使える前のレースは平均 {ev[mean_prev]:.1f} 本で、判定の行の約 {zero:.0f}% は前のレースが 0 本です。</li>
<li>「事後をそのまま当てる」（α = 1）は棄てられました（z ≈ {z1:.1f}。α の 95% 区間の上限 {a_hi:.2f}）。朝の見立て（過去の日の揺れの大きさ）が、日の効きを大きく見ていたことになります。</li>
<li>事後が動いても、5 着内の確率は約 0.4 ポイントしか変わりません。市場の見込みを動かす大きさではありません。</li>
<li>血統の先行の次元（a_front）の日の揺れは、芝ではほぼ 0 でした。日の前残りは、血統を通らず、馬自身の脚質と枠で決まるとみられます（pedigree-1）。</li>
<li>日の揺れが 0 と出た次元（芝の a_pace_h、芝・ダートの a_dist など）は、主の検定の点に入りません。実際に動いたのは、芝 3・ダート 4 次元ほどです。</li>
<li>予想との突き合わせ: pedigree-1 の α の予想（+0.2〜+0.4）は、区間の中にあり否定できません。pedigree-2 の τ の予想（0.05〜0.08）は上の側が棄てられました。「閉じる見込み 70%」は当たりました。</li>
</ul>
<h3>流さなかったもの</h3>
<ul>
<li>持ち越し（前の日の結果を事前に足す版）: 最もよい場合でも情報は約 3.8 倍で、30% の検出力は約 0.46 にとどまり、0.5 に届きません（pedigree-2 の見積もり）。今流して採否に使うと選び直しになるので、流していません。</li>
<li>記述の残り（馬場が替わった日・クラス別・勝ちの条件付きロジット・着順の情報を使う更新・騎手と厩舎を除いた版・コースの形の層）: 段 2 で閉じたので、計算していません（未計算）。</li>
</ul>
<h3>ほかの担当に回す切り口（この担当では作り直さない）</h3>
<ul>
<li>日の単位でなく、開催の週（同じ場・芝ダの 2 日〜前後半）の単位で溜める形。前のレースの本数が数倍になります。着順を開く前に固定する新しい段として。</li>
<li>日の前後の偏りは、血統でなく、馬の前走の通過順（脚質）の側で見る（レースレベル・脚質の担当へ）。</li>
<li>ダートの芝ダ寄りの次元の静的な効き（μ が毎年 +0.07〜+0.14）は、日の効きではなく静的な効きです。pedigree-lab の「上乗せはダートに集まる」と同じ向きです（探す 70% で見た値なので、選び直しの扱い）。</li>
</ul>"""

if __name__ == "__main__":
    main()
