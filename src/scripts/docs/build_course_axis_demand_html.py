"""良馬場のコース別 走法軸の寄与を、根拠つきの 1 枚の HTML にまとめる（読むだけ・HTML を書く）。

入力: src/scripts/data/course_axis_demand.py の出力 JSON、data/meta/course_master/jra_courses.json（コース形状）。
使い方: python src/scripts/docs/build_course_axis_demand_html.py CAD.json [OUT.html]
"""
from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs/html/operations/running_map_course_axis.html"
NAME = {"gear": "ギアチェンジ", "ts": "TS", "sustain": "TS持続力", "cr": "Cr性能", "endure": "Hペース耐性", "basic": "基礎スピード"}
CLASS_BANDS = ["未勝利・新馬", "1勝", "2・3勝", "OP以上"]
LEVELS = ["Lv1(低)", "Lv2", "Lv3", "Lv4(高)"]
GROUP_LABEL = {"短": "1400m以下", "マイル": "1500〜1700m", "中": "1800〜2500m", "長": "2600m以上"}
SMALL_CURVE = "小回り"
T_MARK, T_STRONG = 2.3, 3.5


def esc(s: object) -> str:
    return html.escape(str(s))


def pct(v: float) -> str:
    return f"{v * 100:.0f}"


def tcell(b: float, se: float, scale: float = 100.0) -> str:
    t = b / se if se > 0 else 0.0
    cls = "t3" if abs(t) >= T_STRONG else ("t2" if abs(t) >= T_MARK else "")
    return f'<td class="num {cls}" data-v="{t:.3f}">{b * scale:+.1f} <span class="muted">({t:+.1f})</span></td>'


def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    h = "".join(f"<th>{esc(x)}</th>" for x in head)
    return f'<div class="scroll"><table class="sortable {cls}"><thead><tr>{h}</tr></thead><tbody>' + "".join("<tr>" + "".join(r) + "</tr>" for r in rows) + "</tbody></table></div>"


def td(v: object, num: bool = False, strong: bool = False, data: object = None) -> str:
    c = ("num " if num else "") + ("strong" if strong else "")
    dv = f' data-v="{data}"' if data is not None else ""
    return f'<td class="{c.strip()}"{dv}>{v}</td>'


def share_cells(share: list[float]) -> list[str]:
    top = max(share)
    return [td(pct(s), True, s == top and s >= 0.4, f"{s:.3f}") for s in share]


def load_master() -> dict[tuple[str, str, int], list[dict]]:
    mp: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for x in json.loads((REPO / "data/meta/course_master/jra_courses.json").read_text(encoding="utf-8")):
        mp[(x["venue"], x["surface"], x["distance"])].append(x)
    return mp


def shape_of(items: list[dict]) -> dict:
    st = sorted({x["final_straight_m"] for x in items if x.get("final_straight_m")})
    lap = sorted({x["lap_length_m"] for x in items if x.get("lap_length_m")})
    rng = lambda v: ("—" if not v else (f"{v[0]:.0f}" if len(v) == 1 or v[-1] - v[0] < 1 else f"{v[0]:.0f}〜{v[-1]:.0f}"))
    curves = []
    for x in items:
        c = (x.get("corner_radius_class_A") or "—").split("（")[0]
        curves.append(f"{x['variant']}:{c}" if x["variant"] and len(items) > 1 else c)
    slope = []
    for x in items:
        slope.append(f"{x['goal_slope_rise']:.1f}m" if x.get("goal_slope_exists") and x.get("goal_slope_rise") else ("あり" if x.get("goal_slope_exists") else "なし"))
    return {"straight": rng(st), "lap": rng(lap), "curve": " / ".join(dict.fromkeys(curves)), "slope": " / ".join(dict.fromkeys(slope)),
            "variants": "・".join(x["variant"] for x in items if x["variant"]) or "—",
            "mixed": len(st) > 1 and st[-1] - st[0] > 1, "st0": float(np.mean(st)) if st else float("nan"),
            "small": float(any((x.get("corner_radius_class_A") or "").startswith(SMALL_CURVE) for x in items)),
            "slope_on": float(any(x.get("goal_slope_exists") for x in items))}


def course_regression(d: dict, mp: dict, mod: str) -> list[dict]:
    """コースごとの固有の差（係数のずれ）を、直線長・小回り・ダート・坂の有無へ重み付き回帰する（残差のばらつきで標準誤差を広げる）。"""
    keys, X, ys, ses = [], [], [], []
    for ck, v in d["courses"].items():
        ve, su, di = ck.split("|")
        sh = shape_of(mp[(ve, su, int(di))])
        if sh["mixed"]:
            continue
        keys.append(ck)
        X.append([1.0, (sh["st0"] - 400) / 100, sh["small"], 1.0 if su == "ダート" else 0.0, sh["slope_on"]])
        ys.append(v["dev"][mod]["dbeta"])
        ses.append(v["dev"][mod]["dse"])
    X, Y, S = np.array(X), np.array(ys), np.array(ses)
    out = []
    for j in range(Y.shape[1]):
        w = 1 / S[:, j] ** 2
        W = X.T @ (X * w[:, None])
        b = np.linalg.solve(W, X.T @ (w * Y[:, j]))
        res = Y[:, j] - X @ b
        infl = max(float((w * res ** 2).sum() / (len(Y) - X.shape[1])), 1.0)
        se = np.sqrt(np.diag(np.linalg.inv(W)) * infl)
        out.append({"b": b, "se": se, "n": len(Y)})
    return out


SORT_JS = """<script>
document.querySelectorAll("table.sortable").forEach(function(t){
  t.querySelectorAll("thead th").forEach(function(th,i){
    th.style.cursor="pointer"; th.title="クリックで並べ替え";
    th.addEventListener("click",function(){
      var dir=th.dataset.dir==="1"?-1:1; th.dataset.dir=String(dir);
      var rows=Array.from(t.tBodies[0].rows);
      var val=function(r){var c=r.cells[i]; var v=c.dataset.v!==undefined?c.dataset.v:c.textContent.trim(); var n=parseFloat(String(v).replace(/[^0-9.+-]/g,"")); return isNaN(n)?v:n;};
      rows.sort(function(a,b){var x=val(a),y=val(b); return (typeof x==="number"&&typeof y==="number")?(x-y)*dir:String(x).localeCompare(String(y),"ja")*dir;});
      rows.forEach(function(r){t.tBodies[0].appendChild(r);});
    });
  });
});
</script>"""

CSS = """:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --code:#0b1428; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.7 "Noto Sans JP",system-ui,sans-serif; }
main { max-width:1280px; margin:0 auto; padding:36px 28px 80px; }
h1 { font-size:26px; margin:0 0 6px; } h2 { font-size:19px; margin:40px 0 4px; padding-top:16px; border-top:1px solid var(--border); }
h3 { font-size:16px; margin:22px 0 4px; }
.muted { color:var(--muted); font-size:13px; } .sub { color:var(--muted); font-size:13.5px; margin:0 0 8px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; margin:8px 0 14px; font-size:13px; }
th,td { border:1px solid var(--border); padding:5px 9px; text-align:left; vertical-align:top; }
th { background:var(--panel); color:var(--muted); font-weight:600; white-space:nowrap; }
td.num { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
td.strong { color:var(--accent); font-weight:700; }
td.t2 { background:rgba(251,191,36,.16); } td.t3 { background:rgba(248,113,113,.28); font-weight:700; }
code { font-family:"JetBrains Mono",ui-monospace,Menlo,monospace; background:var(--code); padding:1px 6px; border-radius:5px; font-size:12px; }
.box { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px 18px; margin:12px 0; }
.warn { color:var(--warn); } .ok { color:var(--ok); }
"""


def build(d: dict, mp: dict) -> str:
    K = d["axes"]
    names = [NAME[k] for k in K]
    meta = d["meta"]
    sf = d["surface"]
    # ---- 数値の要約（本文に差し込む）
    ends = [c["A"]["share"][K.index("endure")] for c in d["courses"].values()]
    n_top = sum(1 for c in d["courses"].values() if c["A"]["share"][K.index("endure")] == max(c["A"]["share"]))
    allt = [(ck, j, b / s) for ck, c in d["courses"].items() for j, (b, s) in enumerate(zip(c["dev"]["B"]["dbeta"], c["dev"]["B"]["dse"])) if s > 0]
    n_test = len(allt)
    n_hit = sum(1 for _, _, t in allt if abs(t) >= T_MARK)
    n_exp = n_test * 0.0214   # |t|>=2.3 の両側確率
    regB = course_regression(d, mp, "B")
    regA = course_regression(d, mp, "A")
    ia = d["axis_vs_ability"]
    e = K.index("endure")

    out = [f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>走法マップ 良馬場のコース別 軸の寄与と根拠 — keiba-vpn docs</title><style>{CSS}</style></head><body><main>
<h1>走法マップ：良馬場のコース別 走法軸の寄与と根拠</h1>
<p class="muted">作成: 2026-11-10。ユーザー指摘「どのコースでも Hペース耐性の寄与率が高い。東京は早い上がり（TS）、小回りのローカルは TS持続力・Cr性能、のようにコース形状と結びつくはず」を受け、
良馬場・2024 年以降・頭数 5 以上の {meta['n_runs']:,} 走（{meta['n_races']:,} レース）で測り直した。計算は <code>src/scripts/data/course_axis_demand.py</code>（ホストで実行）、この表は
<code>src/scripts/docs/build_course_axis_demand_html.py</code> が JSON から作る。表の見出しをクリックすると並べ替えできる。続きの検証（実力・人気を除いた優位の根拠づけ、好走数ヒストグラム）は <a href="running_map_course_axis_rationale.html">走法軸 × コースの適性：優位の根拠</a>。</p>

<div class="box"><b>結論</b>
<ol>
<li><b>Hペース耐性が高いのは、コースの要求ではなく「実力の代理」になっているから</b>。現行の寄与率は「得点 − 頭数だけの期待」を回帰するので、強い馬ほど得点が高い分がそのまま軸に乗る。
Hペース耐性は市場が見ている実力（人気込みの期待 − 頭数だけの期待）と相関 {ia['芝'][e]:.2f}（芝）・{ia['ダート'][e]:.2f}（ダート）で、6 軸の中で最大。
人気で実力を除くと R² は芝 {sf['芝']['A']['r2']:.3f} → {sf['芝']['B']['r2']:.3f}、ダート {sf['ダート']['A']['r2']:.3f} → {sf['ダート']['B']['r2']:.3f} へ 1/10〜1/20 に落ち、芝では Hペース耐性の寄与率が {pct(sf['芝']['A']['share'][e])}% → {pct(sf['芝']['B']['share'][e])}% になる。</li>
<li><b>コース形状（直線の長さ・小回り・坂）との関係は、このデータでは確認できない</b>。コース固有の差を、直線長・小回り・ダート・坂の有無に重み付き回帰しても、
TS（直線が長いほど +）・TS持続力・Cr性能（小回りほど +）はいずれも |t| &lt; 1.5 で向きも弱い（§4）。東京芝の TS も平均との差は見えない（t = {d['venue_dev']['東京|芝']['B']['dbeta'][1] / d['venue_dev']['東京|芝']['B']['dse'][1]:+.1f}、向きはむしろ逆）。</li>
<li>コースごとの差の検定は {n_test} 件（{len(d['courses'])} コース × 6 軸）で |t| ≥ {T_MARK} が {n_hit} 件。偶然でも約 {n_exp:.0f} 件は出る数なので、個々の印は仮説の材料に留まる。</li>
<li>理由として考えられるのは、(a) 軸が「出走前の過去走のプロファイル」で測りが粗い、(b) 市場がすでに軸の情報を織り込んでいる（人気調整後の数字は下限）、(c) 軸ごとの効果が小さい（人気調整後の R² は 0.3〜1%）。
「効いていない」ではなく「この手法では検出できない」と読む。</li>
<li>クラス・タイム指数の水準で分けても、現行の寄与率では Hペース耐性が首位のまま（§3）。クラスが上がるほど Hペース耐性の係数は小さくなる（芝 0.106 → 0.067）が、順位は変わらない。</li>
</ol></div>

<h2>1. 何を測ったか</h2>
<ul>
<li><b>A: 現行の再現</b>。得点（1〜3 着 = 1、4 着 0.8、…、8 着以下 = 0）− 頭数だけの期待 を 6 軸の z へ線形回帰し、軸ごとに「抜いたときの R² の落ち」を 100% に規格化（<code>running_map_course._importance</code>）。
画面の寄与率と同じ定義だが、ここでは走数 {meta['min_cell']} 以上のコースを出している（画面は 600 走以上のコースで、足りないコースは場・全場のグループの値を使う）。</li>
<li><b>B: 人気調整後</b>。得点 − 「頭数と人気順が同じ馬の平均得点」（2016〜2023 年の走で作成。評価期間と分離）。実力のうち市場が織り込んだ分を除くので、軸が<b>市場の見落とし</b>をどれだけ説明するかを見る。
市場が軸の情報を一部すでに価格に入れているため、B の効果は下限。</li>
<li><b>C: コース固有の差</b>。B の回帰係数から、「同じ芝／ダート × 同じクラスの平均の係数（切片込み）」を引いた残りを軸へ回帰した係数のずれ。単位は 軸 1 標準偏差あたりの得点（×100 で表示）。標準誤差はレース単位でまとめて計算（括弧内は t）。</li>
<li>軸の値は、その走より前の走だけから作ったプロファイルを順位 → 標準正規にそろえた z（<code>running_map</code> と同じ）。良馬場のみ（{meta['since']} 以降、2026 年の走も含む）。</li>
</ul>

<h2>2. なぜどのコースでも Hペース耐性が高いのか</h2>
<p class="sub">芝・ダート全体での A と B の比較と、軸と「市場が見ている実力」の相関。</p>"""]

    rows = []
    for surf in ("芝", "ダート"):
        for mod, lab in (("A", "A 現行（頭数だけの期待）"), ("B", "B 人気調整後")):
            f = sf[surf][mod]
            rows.append([td(surf), td(lab), td(f"{f['n']:,}", True), td(f["races"], True), td(f"{f['r2']:.4f}", True)] + share_cells(f["share"])
                        + [td(f"{b / s:+.1f}", True, data=f"{b / s:.2f}") for b, s in zip(f["beta"], f["se"])])
    out.append(table(["芝ダ", "指標", "走数", "レース", "R²"] + [f"寄与率% {n}" for n in names] + [f"t {n}" for n in names], rows))
    rows = [[td(s)] + [td(f"{v:+.2f}", True, data=v) for v in ia[s]] for s in ("芝", "ダート")]
    out.append(table(["芝ダ"] + [f"相関 {n}" for n in names], rows))
    out.append(f"""<p class="sub">相関 = 軸の z と「人気込みの期待 − 頭数だけの期待」（市場が見ている実力）の相関。ダートでは人気調整後も Hペース耐性は t = {sf['ダート']['B']['beta'][e] / sf['ダート']['B']['se'][e]:.1f} で残る
（市場の見落としにも効いている）一方、Cr性能は負（t = {sf['ダート']['B']['beta'][3] / sf['ダート']['B']['se'][3]:.1f}）。</p>

<h2>3. クラス・メンバーレベル（タイム指数）で分けると</h2>
<p class="sub">クラスは <code>grade</code>（未勝利・新馬 / 1勝 / 2・3勝 / OP以上）。タイム指数の水準は、netkeiba の出走前の「タイム指数の平均（speed_avg）」を出走馬で平均し（0 は欠け、有効な馬が 5 頭以上かつ半数以上のレース）、
レース単位の 4 分位（境界 {', '.join(str(c) for c in meta['level_cuts'])}）で Lv1〜Lv4 とした。出走前に分かる値で、結果は使っていない。ファイルのある 2024〜2025 年だけが対象で、新馬・2026 年は水準なし（{meta['level_missing_runs']:,} 走）。</p>""")
    for title, src, order, pre in (("3.1 クラス別", d["class_surface"], CLASS_BANDS, "class"), ("3.2 タイム指数の水準別", d["level_surface"], LEVELS, "level")):
        out.append(f"<h3>{title}</h3>")
        rows = []
        for surf in ("芝", "ダート"):
            for c in order:
                v = src.get(f"{surf}|{c}")
                if not v:
                    continue
                a, b = v["A"], v["B"]
                rows.append([td(surf), td(c), td(f"{a['n']:,}", True), td(f"{a['r2']:.3f}", True), td(f"{b['r2']:.4f}", True)] + share_cells(a["share"])
                            + [td(f"{x / s:+.1f}", True, data=f"{x / s:.2f}") for x, s in zip(b["beta"], b["se"])])
        out.append(table(["芝ダ", "区分", "走数", "R² A", "R² B"] + [f"A 寄与率% {n}" for n in names] + [f"B t {n}" for n in names], rows))
    rows = [[td(c)] + [td(f"{d['level_by_class'][c][l]:,}", True) for l in LEVELS] for c in CLASS_BANDS]
    out.append("<h3>3.3 クラスと水準の対応（走数）</h3>" + table(["クラス"] + LEVELS, rows) + """<p class="sub">水準はほぼクラスと重なる（未勝利は Lv1〜2、2・3勝は Lv3〜4）ので、水準で分けても同じ傾向が出る。
クラス内の差を見るには水準の方が細かいが、走数は少なくなる。</p>""")

    # ---- §4 コース形状との関係
    out.append("""<h2>4. コース形状との関係（仮説の検定）</h2>
<p class="sub">ユーザーの仮説: 直線が長いコース（東京など）は TS、小回りのローカルは TS持続力・Cr性能。コースごとの固有の差（C）を、
直線の長さ（400m からの差を 100m 単位で）・小回りか（<code>jra_courses.json</code> の <code>corner_radius_class_A</code> が「小回り」始まり）・ダートか・ゴール前の坂の有無へ重み付き回帰した（内外で直線長が違うコースは除く）。
表は 係数×100（括弧内 t）。仮説が正しければ、TS の「直線長」と、TS持続力・Cr性能の「小回り」が正で大きく出る。</p>""")
    head = ["指標", "軸", "直線が 100m 長い", "小回り", "ダート", "ゴール前に坂"]
    rows = []
    for mod, lab, reg in (("B", "B 人気調整後", regB), ("A", "A 現行", regA)):
        for j, k in enumerate(K):
            r = reg[j]
            rows.append([td(lab), td(NAME[k])] + [tcell(r["b"][i], r["se"][i]) for i in (1, 2, 3, 4)])
    n_reg = regB[0]["n"]
    out.append(table(head, rows) + f'<p class="sub">対象 {n_reg} コース。黄 = |t| ≥ {T_MARK}、赤 = |t| ≥ {T_STRONG}。</p>')

    out.append("<h3>4.1 直線の長さでまとめた場合</h3>")
    rows = []
    for k, v in d["straight_dev"].items():
        surf, name = k.split("|")
        m = v["B"]
        rows.append([td(surf), td(name), td(f"{v['n']:,}", True)] + [tcell(b, s) for b, s in zip(m["dbeta"], m["dse"])])
    out.append(table(["芝ダ", "直線の長さ（場）", "走数"] + names, rows) + '<p class="sub">B 人気調整後・係数×100（t）。短い直線 = 札幌・函館・福島・小倉・中山、中間 = 京都・阪神・新潟、長い直線 = 東京・中京（内外コースの混在はあるが場でまとめた）。</p>')
    out.append("<h3>4.2 競馬場別</h3>")
    rows = []
    for k, v in d["venue_dev"].items():
        ve, surf = k.split("|")
        m = v["B"]
        rows.append([td(ve), td(surf), td(f"{v['n']:,}", True), td(v["races"], True)] + [tcell(b, s) for b, s in zip(m["dbeta"], m["dse"])])
    out.append(table(["競馬場", "芝ダ", "走数", "レース"] + names, rows) + '<p class="sub">B 人気調整後・係数×100（t）。</p>')
    out.append("<h3>4.3 芝ダ × 距離帯別（全場）</h3>")
    rows = []
    for gk, v in d["groups"].items():
        s_, b_ = gk.split("|")
        a, b, dv = v["A"], v["B"], v["dev"]["B"]
        rows.append([td(s_), td(GROUP_LABEL[b_]), td(f"{a['n']:,}", True), td(f"{a['r2']:.3f}", True), td(f"{b['r2']:.4f}", True)] + share_cells(a["share"]) + [tcell(x, s) for x, s in zip(dv["dbeta"], dv["dse"])])
    out.append(table(["芝ダ", "距離帯", "走数", "R² A", "R² B"] + [f"A 寄与率% {n}" for n in names] + [f"C 固有の差 {n}" for n in names], rows))

    # ---- §5 コース別
    keys = sorted(d["courses"], key=lambda k: (k.split("|")[0], k.split("|")[1], int(k.split("|")[2])))
    out.append(f"""<h2>5. 良馬場のコース別（{len(keys)} コース）</h2>
<p class="sub">走数 {meta['min_cell']} 以上のコース。形状はコースマスタ（直線長・周長・坂・カーブの大きさは 3 ソース合意の値）。内外で形状が違うコースは「内:… / 外:…」で併記し、直線長・周長は範囲で表示する。</p>
<h3>5.1 現行（A）の寄与率 と コース形状</h3>""")
    rows = []
    for ck in keys:
        ve, su, di = ck.split("|")
        c = d["courses"][ck]
        sh = shape_of(mp[(ve, su, int(di))])
        rows.append([td(ve), td(su), td(di, True), td(f"{c['A']['n']:,}", True), td(sh["straight"], True), td(sh["lap"], True), td(esc(sh["slope"])), td(esc(sh["curve"])), td(sh["variants"])]
                    + share_cells(c["A"]["share"]) + [td(f"{c['A']['r2']:.3f}", True), td(f"{c['B']['r2']:.4f}", True)])
    out.append(table(["競馬場", "芝ダ", "距離", "走数", "直線m", "周m", "坂の高さ", "カーブ", "内外"] + [f"寄与率% {n}" for n in names] + ["R² A", "R² B"], rows))
    out.append("<h3>5.2 人気調整後（B）の寄与率</h3>")
    rows = []
    for ck in keys:
        ve, su, di = ck.split("|")
        c = d["courses"][ck]
        rows.append([td(ve), td(su), td(di, True), td(f"{c['B']['n']:,}", True)] + share_cells(c["B"]["share"]) + [td(f"{c['B']['r2']:.4f}", True)])
    out.append(table(["競馬場", "芝ダ", "距離", "走数"] + [f"寄与率% {n}" for n in names] + ["R² B"], rows)
               + '<p class="sub">R² B が 1% 前後と小さいので、この寄与率は大きくぶれる（走数が少ないコースほど）。目安として読む。</p>')
    out.append("<h3>5.3 コース固有の差（C）— 同じ芝ダ・同じクラスの平均からのずれ</h3>")
    rows = []
    for ck in keys:
        ve, su, di = ck.split("|")
        c = d["courses"][ck]
        sh = shape_of(mp[(ve, su, int(di))])
        m = c["dev"]["B"]
        rows.append([td(ve), td(su), td(di, True), td(sh["straight"], True), td(esc(sh["curve"]))] + [tcell(b, s) for b, s in zip(m["dbeta"], m["dse"])])
    out.append(table(["競馬場", "芝ダ", "距離", "直線m", "カーブ"] + names, rows)
               + f'<p class="sub">B 人気調整後・係数×100（t）。黄 = |t| ≥ {T_MARK}（{n_test} 件中 {n_hit} 件、偶然でも約 {n_exp:.0f} 件）、赤 = |t| ≥ {T_STRONG}。正 = 平均よりその軸が効く。</p>')

    out.append(f"""<h2>6. 読み方と限界</h2>
<ul>
<li><b>現行の寄与率（A）は、コースの要求ではなく実力の代理を含む</b>。{n_top}/{len(d['courses'])} コースで Hペース耐性が首位（最小 {pct(min(ends))}%）。画面の「寄与率」を「このコースで求められる能力」と読まないこと。</li>
<li>B は市場の織り込みを引くため、軸の効果の<b>下限</b>。「A は過大、B は過小」で、真の要求はその間にある。</li>
<li>コース固有の差の基準（芝ダ × クラスの平均）は同じ走で推定しており、その推定誤差は t に入っていない。場別・直線長別のように基準に占める割合が大きい集計ほど、差は 0 側に偏る（検出しにくい側の誤差）。</li>
<li>多重検定（{n_test} 件）。個々のコースの印は仮説の材料であり、確認には別期間（2026 年はテストセット②として温存）での再現が要る。</li>
<li>軸は出走前のプロファイル（過去走から作る）。新馬・履歴の薄い馬は父平均などで埋まるため、軸の測りは粗い。ここで効果が見えないことは、軸そのものが無意味という意味ではない。</li>
<li>良馬場のみ。道悪の地図は基礎スピードの代わりに道悪適性を入れた別の地図で、ここでは扱っていない。</li>
<li>タイム指数の水準は 2024〜2025 年のみ（2026 年のファイルが無い）。クラスの方が走数は多い。</li>
</ul>
<h2>7. 反映（2026-11-10 ユーザー指示「実力の代理はコース適性から排除」）</h2>
<ul>
<li>コース適性（寄与率・地図の重み・好走ゾーン・当日の馬場傾向）の期待得点を、頭数だけから<b>人気込み（B と同じ考え方）</b>に変更した（<code>REFERENCE_VERSION</code> 16、<code>running_map_course.POP_TABLE</code>）。画面に出るのは基準の作り直し後。この表（§2〜§5 の A）は変更前の値。</li>
<li>画面の好走ゾーンの倍率は「人気どおり = 1」になる。寄与率は B 相当（市場の織り込みを除いた下限）で、R² が小さく、コース別にはぶれる。</li>
<li>コース形状から決める方針（直線長・カーブ・坂から「求められる方向」を事前に置く）は、このデータでは裏付けが取れていないため未採用。</li>
</ul>
<p class="muted">再計算: ホストで <code>cd /home/jovyan/work/keiba-vpn &amp;&amp; nice -n 19 /opt/conda/bin/python -I src/scripts/data/course_axis_demand.py OUT.json</code> → 手元で <code>python src/scripts/docs/build_course_axis_demand_html.py OUT.json</code>。</p>
{SORT_JS}
</main></body></html>""")
    return "".join(out)


def main() -> None:
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out.write_text(build(d, load_master()), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
