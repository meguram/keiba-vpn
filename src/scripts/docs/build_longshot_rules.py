"""穴馬 × コースのめぐ分析のルール探し（docs/html/operations/final/longshot_course_rules.html）を、結果の JSON から作る（数字を手で写さない）。

入力:
  docs/operations/retrain_trials/longshot_rules.json（src/research/course/longshot_rules.py describe が書く。母集団の記述・合成の確かめ・較正）
  config/longshot_rules/prereg.json（事前登録と結末）
使い方: python -m src.scripts.docs.build_longshot_rules
費用 0 円（手元のファイルだけ）。
"""

from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.scripts.docs.build_latest_strategy_tables import CSS

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/operations/retrain_trials/longshot_rules.json"
PREREG = ROOT / "config/longshot_rules/prereg.json"
DST = ROOT / "docs/html/operations/final/longshot_course_rules.html"
TITLE = "穴馬 × コースのめぐ分析のルール探し"
JST = timezone(timedelta(hours=9))
VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
CLS = {0: "2勝", 1: "3勝", 2: "OP・L・重賞"}
SMALL_N = 100           # これ未満の走のコースは区間を出さない（偶然の幅が大きすぎる）
OBIN_EDGES: list[float] = []


def esc(x) -> str:
    return html.escape(str(x))


def pc(x, d: int = 1) -> str:
    return "－" if x is None else f"{x * 100:.{d}f}%"


def num(x, d: int = 2) -> str:
    return "－" if x is None else f"{x:.{d}f}"


def ci(v, pct: bool = True) -> str:
    if not v:
        return "－"
    f = (lambda x: f"{x * 100:.0f}%") if pct else (lambda x: f"{x:.2f}")
    return f"{f(v[0])}〜{f(v[1])}"


def table(head: list[str], rows: list[list], num_cols=frozenset()) -> str:
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f'<td class="n">{c}</td>' if i in num_cols else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def fold(summary: str, body: str) -> str:
    return f'<details><summary>{esc(summary)}</summary><div class="dbody">{body}</div></details>'


def unit_key(u: str) -> tuple:
    v = u[:2]
    surf = u[2]
    dist = int("".join(ch for ch in u[3:] if ch.isdigit()) or 0)
    return (VENUES.index(v) if v in VENUES else 99, surf, dist)


def obin_label(r: dict, edges: list[float] | None = None) -> str:
    """確定オッズの帯の名前。母集団は再現オッズで決めるので、端の帯は確定で 10 倍未満・50 倍超を含む。"""
    e = edges or OBIN_EDGES
    b = r["obin"]
    if b == 0:
        return f"確定 〜{e[1]:.1f} 倍（10 倍未満を含む）"
    if b == len(e) - 2:
        return f"確定 {e[b]:.1f} 倍〜（50 倍超を含む）"
    return f"確定 {e[b]:.1f}〜{e[b + 1]:.1f} 倍"


HEAD_T = ["区分", "走", "レース", "勝率", "単勝の回収率", "区間", "市場比", "区間"]
HEAD_F = ["区分", "走", "3 着内", "複勝の回収率", "区間", "市場比", "区間"]


def rows_t(recs: list[dict], label) -> list[list]:
    out = []
    for r in recs:
        small = r["n"] < SMALL_N
        out.append([esc(label(r)), f"{r['n']:,}", f"{r['races']:,}", pc(r["wins"] / r["n"]), pc(r["roi_t"]),
                    "－" if small else ci(r["roi_t_ci"]), num(r["mr_t"]), "－" if small else ci(r["mr_t_ci"], False)])
    return out


def rows_f(recs: list[dict], label) -> list[list]:
    out = []
    for r in recs:
        small = r["n"] < SMALL_N
        out.append([esc(label(r)), f"{r['n']:,}", pc(r["top3"] / r["n"]), pc(r["roi_f"]), "－" if small else ci(r["roi_f_ci"]),
                    num(r["mr_f"]), "－" if small else ci(r["mr_f_ci"], False)])
    return out


def build() -> str:  # noqa: C901
    D = json.loads(SRC.read_text(encoding="utf-8"))
    P = json.loads(PREREG.read_text(encoding="utf-8"))
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()[:12]
    pop = D["population"]
    a = D["all"][0]
    pw, p2, sa, sb = D["power1"], D.get("power2"), D["simA"], D["simB"]
    w, p_ = pw["win"], pw["place"]
    c0 = sa["cands"][0]["mean"]
    OBIN_EDGES[:] = D["obin_edges"]
    last_ob = D["by_obin"][-1]
    S = P["search"]
    ew, ep = S["e_min_win"], S["e3_min_place"]
    kw, kp = S["shrink"]["k_win"], S["shrink"]["k_place"]
    n_units, n_types = len(D["by_unit"]), len(D.get("types") or [])
    gx = lambda v: "－" if v is None else f"{v:g}"   # noqa: E731
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    out = ['<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           f"<title>{esc(TITLE)}</title><style>{CSS}</style></head><body><main>", f"<h1>{esc(TITLE)}</h1>",
           f'<p class="sub">作成 {now}（JST）。2025・2026 年の着順・払戻は開いていない。</p>']
    # 結論
    out.append('<h2>結論</h2><div class="box"><ul>')
    out.append("<li><b>本物の条件の探索は流さずに閉じた。</b>合成データで確かめると、利益に届く大きさの効きを、この作りでは見つけられないと分かった。"
               "「効かない」と分かったのではない（作りの打ち切り）。</li>")
    out.append(f"<li>作り: コースの中で、めぐの水準の条件を探す。条件は見込みの勝ち数（市場のオッズから見込む勝ち数の和）が単勝 {ew:g}・複勝 {ep:g} 以上のものに限る。"
               "最初の関門は、条件ごとの市場比（同じオッズの馬に比べて何倍勝つか）のばらつきが、偶然より大きいかの検定（総合検定）。</li>")
    out.append(f"<li>各コースに市場比 ×1.3 の小さな条件（中央値 {c0['picked_rows_median']:.0f} 走・見込みの勝ち数 {c0['picked_E_median']:.1f}）を 1 つ植えると、"
               f"関門を超える割合は単勝 {pc(w['power_1.3'], 0)}・複勝 {pc(p_['power_1.3'], 1)}（効きなしと同じ）。事前に決めた規則（50% 以上で本物を流す）に届かない。</li>")
    if p2:
        out.append(f"<li>記録として、下限を満たす大きな条件（コースの穴馬の約 {pc(p2['win']['1.3']['share_rows'], 0)}）に植えても、"
                   f"実際の市場比 {num(p2['win']['1.3']['real_ratio'])}（単）・{num(p2['place']['1.3']['real_ratio'])}（複）では {pc(p2['win']['1.3']['power'], 0)}・{pc(p2['place']['1.3']['power'], 1)}。"
                   f"見えるのは、単勝で市場比 {num(p2['win']['2.0']['real_ratio'])}（{pc(p2['win']['2.0']['power'], 0)}）、複勝で {num(p2['place']['2.0']['real_ratio'])}（{pc(p2['place']['2.0']['power'], 0)}）の大きな効きだけ。</li>")
    out.append(f"<li>×1.3 は、100% に届く最小の大きさである（全部の穴馬の単勝の回収率 {pc(a['roi_t'])} × 1.3 ≈ {pc(a['roi_t'] * 1.3, 0)}）。"
               "利益の出る効きを見つけるには、これより大きな効きか、もっと多い件数が要る。</li>")
    out.append(f"<li>母集団（2016〜24、2勝クラス以上・T-60 の再現オッズで 10〜50 倍）は {a['n']:,} 走・{n_units} コース。全部を買うと単勝 {pc(a['roi_t'])}"
               f"（{ci(a['roi_t_ci'])}）・複勝 {pc(a['roi_f'])}（{ci(a['roi_f_ci'])}）。単勝の市場比は {num(a['mr_t'])} で、100% との差はほぼ控除の分。"
               f"市場比が 1 を下回るのは確定 {OBIN_EDGES[-2]:.1f} 倍以上の帯（{num(last_ob['mr_t'])}）だけ。</li>")
    out.append("<li>通った条件は無い。2025・26 年の 1 回の確かめは行っていない（規則で、段 2 を通った券種だけが対象）。</li></ul></div>")
    # 何をしたか
    out.append("<h2>何を、どう決めたか</h2><ul>"
               "<li>母集団: 2勝・3勝・OP・L・重賞、芝ダ、馬場はすべて。穴馬は T-60 の再現オッズ（確定オッズ × 2026 年の T-60 の実測のずれ、種 1000 の 1 本）で 10〜50 倍。"
               "§77 の想定オッズの OOF は 2022 年からしか無く、2016〜24 には使えない。再現は確定オッズ由来なので、先読みは一部しか消えない。</li>"
               f"<li>単位: コース（会場 × 芝ダ × 距離）。切り口は catalog の水準 {n_types} 型（調教・めぐ指数・B07・G08・I09 を除く）。オッズ・人気は、母集団の範囲・帰無の層・物差しにだけ使う。</li>"
               f"<li>分け方: 2016〜24 の開催（会場 × 年 × 回、{pop['kaisai']} 個）を無作為に 5 つに分ける交差検証と、前向き（2016〜21 → 2022〜24）。開催の単位は、ユーザーの指示の開催日より粗く、統括役が承認した。</li>"
               "<li>縮め方: 勝ち数を市場の見込みの比にし、条件 → コース → 全体へ経験ベイズで縮める。比のばらつき τ² は積率法で推す（レースの中の排他を入れた分散で、偶然の分を引く）。</li>"
               "<li>合否の線・模擬の作り・打ち切りの規則は、本物を見る前に事前登録に書き、「絶対回収率プラスにするマン」3 体の議論と 1 号の決定"
               "（<code>docs/operations/roi_plus_council/decisions.md</code> の D1〜R3-4）で直した。</li></ul>")
    # 検出力
    out.append("<h2>検出力の確かめ（合成データ。着順は使わない）</h2>")
    out.append('<p class="sub">合成は、レースの全頭の確定の q × 帯の比から、Plackett-Luce（Harville）で着順を引き直した。2016〜21 の穴馬で行う。'
               "「×1.3」は単勝の勝つ重みに掛けた倍率で、複勝の 3 着内の比は実際にはそれより小さく上がる（下の表の「実際の比」）。</p>")
    out.append(f"<h3>段 1 の検出力（規則に使った模擬。効きなし {pw['n_null']} 回の 97.5% 点を線に、各 {pw['n_eff']} 回）</h3>")
    rows = [[nm, num(r["line_q975"], 4), num(r["null_mean"], 4), pc(r["power_1.3"], 1), pc(r["power_1.15"], 1)] for nm, r in (("単勝", w), ("複勝", p_))]
    out.append(table(["券種", "線", "効きなしの平均", "×1.3 で超える割合", "×1.15 で超える割合"], rows, {1, 2, 3, 4}))
    out.append(f"<p>植えた条件は、コースの穴馬の 5〜50% を覆う深さ 1 の条件から 1 つ選んだ（中央値 {c0['picked_rows_median']:.0f} 走、見込みの勝ち数 中央値 {c0['picked_E_median']:.1f}・90% 点 {c0['picked_E_q90']:.1f}）。"
               "多くは下限より小さく、検定に入らない。下限を満たす大きな条件の中で、効きが薄まる。</p>")
    out.append("<p>事前登録の段 1 は、本物の τ² を本物のデータの並べ替えの帰無と比べる。ここでの検出力は、合成の帰無から作った線で測った（1 号の決定 R3-3 の作り）。"
               "二つの帰無は平均が約 1 SD 違う（単勝で合成 +0.0002・本物 −0.0057）。植えた効きでの τ² の動きは帰無の SD の半分に届かないので、どちらの線でも結論は変わらない。</p>")
    if p2:
        rows = []
        for kind, nm in (("win", "単勝"), ("place", "複勝")):
            for r in ("1.3", "1.5", "2.0"):
                x = p2[kind][r]
                rows.append([nm, f"×{r}", num(x["real_ratio"]), pc(x["share_rows"], 0), pc(x["power"], 1)])
        out.append(f"<h3>記録: 下限を満たす大きな条件に植えた場合（規則の後に、レビューの指摘で追加。各 {p2['n_eff']} 回）</h3>")
        out.append(table(["券種", "植えた倍率", "実際の比", "効きのある行", "線を超える割合"], rows, {2, 3, 4}))
        out.append("<p>深さ 2 の条件・効きの形が他にある場合は試していない（未検証）。</p>")
    body = []
    s0 = [x for x in D["stage0"] if x.get("emin") == pw["emin"]]
    if s0:
        rows = [[num(lv["sigma"]), num(lv["truth"]["tau2_cell_w"], 4), num(lv["est_mean"]["tau2_cell_w"], 4), num(lv["truth"]["tau2_cell_p"], 4),
                 num(lv["est_mean"]["tau2_cell_p"], 4)] for lv in s0[0]["levels"]]
        body.append(f"<h3>段 0: ばらつき τ² の推定は真の値を取り戻すか（下限 単 {ew:g}・複 {ep:g}）</h3>")
        body.append(table(["倍率のばらつき σ", "単勝 真の τ²", "単勝 推定の平均", "複勝 真の τ²", "複勝 推定の平均"], rows, {0, 1, 2, 3, 4}))
        body.append(f"<p>単勝は、真の値が 0 のときは取り戻す。複勝は効きなしでも推定が約 +0.002 高い。効きがあると、どちらも推定は真の値より大きく出る。"
                    f"そのため縮めには「帰無 1〜20 番の平均を引き、係数 k（単 {kw:g}・複 {kp:g}）で割った値」を使う規則にした（1 号の決定 R3-1。値はこの表の前の版の下限で決めた）。"
                    "複勝の真の τ² は、200 回の引き直しの平均の 3 着内の確率から作るので、引き直しの雑音で少し上に出る（σ=0 でも 0 にならない）。</p>")
    rows = [[f"単 {c['e_min_win']:g}・複 {c['e3_min_place']:g}", num(c["mean"]["n_courses_w"], 1), num(c["mean"]["n_courses_p"], 1),
             num(c["mean"]["bets_t"], 1), num(c["mean"]["bets_f"], 1)] for c in sa["cands"]]
    body.append(f"<h3>模擬 A: 下限の決め（各コースに ×1.3 の条件を 1 つ植える、{sa['cands'][0]['reps']} 回。単勝と複勝の下限は組で流した）</h3>")
    body.append(table(["下限の候補", "条件が残るコース（単）", "（複）", "買った点（単）", "（複）"], rows, {1, 2, 3, 4}))
    body.append(f"<p>どの候補もほぼ 1 点も買わない。規則どおりの下限は 単 {gx(sa['win'])}・複 {gx(sa['place'])}。</p>")
    g = sb["grid"]
    rows = [[nm, f"{sb['kinds'][k]['min_points']:,} 点"] + [pc(sb["kinds"][k]["1.15"][g.index(n)], 0) for n in (200, 1000, 1500, 3000)]
            for k, nm in (("t", "単勝"), ("f", "複勝"))]
    body.append(f"<h3>模擬 B: 合否の点数（真の回収率 1.15・0.78、各 {sb['reps']:,} 回）</h3>")
    body.append(table(["券種", "最小の点数", "1.15 で通る 200 点", "1,000 点", "1,500 点", "3,000 点"], rows, {1, 2, 3, 4, 5}))
    body.append(f"<p>最小の点数は、真の回収率 1.15 で通る割合 50% 以上・0.78 で 2.5% 以下の最小。単勝は {sb['kinds']['t']['min_points']:,} 点が要り、模擬 A で買う点数（ほぼ 0）を超えるので、合否から外して参考にする規則になった。</p>")
    out.append(fold("段 0・模擬 A・模擬 B（下限と点数の決め）", "".join(body)))
    # 母集団
    out.append("<h2>母集団の大きさと基準線</h2>")
    out.append('<div class="box"><b>全部の穴馬を買った値（母集団の記述）であり、条件やコースを選ぶ根拠にしない。</b>'
               "コースは会場の順（札幌→小倉）・芝ダ・距離の順に並べる（回収率の順にしない）。2025・26 年は載せない。</div>")
    unit_cnt = sum(1 for r in D["by_unit"] if r["n"] >= SMALL_N)
    out_t = sum(1 for r in D["by_unit"] if r["n"] >= SMALL_N and (r["mr_t_ci"][0] > 1 or r["mr_t_ci"][1] < 1))
    out_f = sum(1 for r in D["by_unit"] if r["n"] >= SMALL_N and (r["mr_f_ci"][0] > 1 or r["mr_f_ci"][1] < 1))
    out.append(f'<p class="sub">区間は開催で束ねたブートストラップの 95%（{SMALL_N} 走未満の行は出さない）。市場比は 勝ち ÷ 確定の q の和。'
               "複勝の市場比は 3 着内 ÷ 帯 × 頭数の群の率の和で、物差しを同じ全体から作るので、全体と帯の行では定義上ほぼ 1 になる。"
               f"コースの表で市場比の区間が 1 を外れるのは、{unit_cnt} コースのうち単勝 {out_t}・複勝 {out_f}（偶然でも 5% ずつ外れる）。</p>")
    out.append(f'<p>再現の帯に入った穴馬 {pop["longshot_pre"]:,} 走（確定オッズで 10〜50 倍は {pop["longshot_final"]:,} 走。確定では帯の中で再現では外 {pop["final_in_pre_out"]:,}・'
               f'確定では外で再現では中 {pop["final_out_pre_in"]:,}）。開催 {pop["kaisai"]}・開催日 {pop["kaisai_days"]:,}。</p>')
    yl = lambda r: "全体" if "k" in r else f"{r['year']} 年"   # noqa: E731
    out.append("<h3>単勝: 全体・年</h3>" + table(HEAD_T, rows_t(D["all"] + D["by_year"], yl), set(range(1, 8))))
    out.append("<h3>複勝: 全体・年</h3>" + table(HEAD_F, rows_f(D["all"] + D["by_year"], yl), set(range(1, 7))))
    cl = lambda r: CLS.get(r["cgrp"], r["cgrp"]) if "cgrp" in r else obin_label(r)   # noqa: E731
    out.append(fold("クラス・確定オッズの帯", "<h3>単勝</h3>" + table(HEAD_T, rows_t(D["by_cls"] + D["by_obin"], cl), set(range(1, 8)))
                    + "<h3>複勝</h3>" + table(HEAD_F, rows_f(D["by_cls"] + D["by_obin"], cl), set(range(1, 7)))))
    units = sorted(D["by_unit"], key=lambda r: unit_key(r["unit"]))
    ul = lambda r: r["unit"]   # noqa: E731
    out.append(fold(f"コースごと（{n_units} コース。会場の順）", "<h3>単勝</h3>" + table(HEAD_T, rows_t(units, ul), set(range(1, 8)))
                    + "<h3>複勝</h3>" + table(HEAD_F, rows_f(units, ul), set(range(1, 7)))))
    # 帰無・見た回数
    cb = D.get("calib") or {}
    gt = cb.get("gate_2016_21", {})
    out.append("<h2>帰無との比べ・見た回数</h2><ul>"
               "<li>本物の探索は流していないので、本物と帰無の比べ・通った条件・2025・26 年の 1 回の結果は無い。</li>"
               f"<li>記録: 本物のデータの並べ替えの帰無 1〜20 番（コース × クラス群 × 時代 × 確定オッズ 8 帯の中で入れ替え）の τ² の平均は、2016〜21 で"
               f"単勝 {num(gt.get('tau2_cell_w'), 4)}・複勝 {num(gt.get('tau2_cell_p'), 4)}。単勝が負に出るのは合成の帰無（約 0）と違う点で、市場の q と本当の勝率の形のずれと見ている（推定）。</li>"
               "<li>見た回数（2016〜24 の本物の着順の使い道）: 本物のデータの並べ替えの帰無（較正の 20 回を 2016〜21 と交差検証の 5 fold で、固定の前の試し 2〜3 回）、"
               "合成の材料（帯の比 = 2016〜21 の穴馬の 勝ち ÷ q、複勝の物差しの率と払戻の尺度）、この頁の母集団の記述（年・クラス・帯・コース別の全部の穴馬の成績）。"
               "コース別の成績を 1 回見たので、次の議題でコースや帯を選ぶ根拠にしない。本物の水準と着順の組で条件を選んだ・確かめた回数は 0。2025・26 年は 0 回"
               "（計画の前に、めぐ分析の 2025・26 年のオッズ帯別の表を読んでいる）。</li></ul>")
    # 課題
    out.append("<h2>課題と次の議題の候補</h2><ul>"
               "<li>どの単位でも、ルールを探す案は、下限を満たす大きさの条件に植えた効き（×1.3 以上の数段）で関門を超える確率を、先に示してから本物を開く（lessons に追記）。"
               "今回の誤りは、植える条件の大きさと下限を合わせずに模擬を決めたこと。</li>"
               "<li>次の議題の候補（どれも検出力の模擬を先にし、新しい事前登録にする）:<ol>"
               "<li>型の効きをコースをまたいで共有する階層の縮め。件数は数十倍になる。ユーザーの指定（コースの単位）から外れるので承認が要る。"
               "全体に寄るほど、打ち切った「水準の AND を結果で探し直す」に近づくので、探す数を絞る作りにする。</li>"
               "<li>Benter の 2 段（水準を、市場を offset にしたロジットに足す。2 号の案）。50 サイクルで唯一効いた「自前の確率 × 市場」と近く、"
               "打ち切った「T-60 を offset にした柔らかい学習（サイクル 9）」との違いを先に書く。</li>"
               f"<li>再現の帯の中身を本物の T-60 で確かめる。再現の帯の {pop['longshot_pre']:,} 走のうち {pop['final_out_pre_in']:,} 走は確定では帯の外で、帯の中身は再現の 1 本に大きく左右される。"
               "本物の発走前オッズの台帳（GCP の外から GCS を繰り返し読まない作り）を使う。</li></ol></li>"
               "<li>再現のずれの元（2026 年の T-60 と確定のオッズの組）を作るのに、2026 年の race_result_flat のオッズ・人気の列だけを読んだ（着順・払戻は読まない）。</li></ul>")
    # 再現
    out.append("<h2>作り方・再現</h2><ul>"
               f"<li>数字は <code>src/research/course/longshot_rules.py</code>・<code>longshot_synth.py</code> の結果の JSON（<code>docs/operations/retrain_trials/longshot_rules.json</code>）から"
               f"<code>src/scripts/docs/build_longshot_rules.py</code> で生成。事前登録 <code>config/longshot_rules/prereg.json</code>（sha256 {sha}…）。</li>"
               "<li><code>python -m src.research.course.longshot_rules frame</code>（母集団）→ <code>tau --seeds 10001-10020 --sets gate|cv</code>（較正）→ <code>describe</code>（この頁の JSON）。</li>"
               "<li><code>python -m src.research.course.longshot_synth stage0 --ew 8 --ep 30</code>・<code>simA --reps 20 --null 20</code>・<code>simB --reps 2000</code>・"
               "<code>power1 --null 200 --neff 40</code>・<code>power2 --neff 40</code>。</li>"
               "<li>重いので <code>nice -n 19</code>。費用 0 円（手元の表だけ。GCS は読まない）。出力は <code>&lt;COURSE_OUT&gt;/longshot_rules</code>。</li></ul>")
    out.append("</main></body></html>")
    return "\n".join(out)


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(build(), encoding="utf-8")
    print("wrote", DST)


if __name__ == "__main__":
    main()
