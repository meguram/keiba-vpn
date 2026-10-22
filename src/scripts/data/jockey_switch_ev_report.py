"""騎手の乗り替わりの期待値分析の章 docs/html/modeling/chapters/jockey_switch_ev.html を、result.json から作る。

python -m src.scripts.data.jockey_switch_ev report   # JSW_OUT/result.json を読む（ホストで作った JSON を JSW_OUT に置いて実行）
"""

from __future__ import annotations

import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/html/modeling/chapters/jockey_switch_ev.html"

CSS = """:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; --good:#34d399; --bad:#f87171; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1150px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1b2a4d; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
.ok { color: var(--good); } .ng { color: var(--bad); } code { background: #0b1428; padding: 1px 5px; border-radius: 4px; }"""

TYPE_LABEL = {"NE": "非有力→有力（NE）", "EN": "有力→非有力（EN）", "EE": "有力→有力（EE）", "NN": "非有力→非有力（NN）", "SAME": "継続（同じ騎手）"}
PERIOD = {"sel": "選択 2021〜24", "2025": "test 2025", "2026": "2026（記述）"}
STRATA_LABEL = {
    "ten_nori": "テン乗り", "modori": "戻り（以前に乗ったことがある）", "prev_good(1-3)": "前走 1〜3 着", "prev_mid(4-9)": "前走 4〜9 着", "prev_bad(10+)": "前走 10 着以下",
    "pop_1-3": "今走 1〜3 番人気", "pop_4-6": "今走 4〜6 番人気", "pop_7+": "今走 7 番人気以下", "turf": "芝", "dirt": "ダート", "maiden": "新馬・未勝利", "win1+": "1 勝クラス以上",
    "dist_short": "短距離（〜1300）", "dist_mile": "マイル（1301〜1900）", "dist_mid+": "中長距離（1901〜）", "to_stable_main": "同厩舎の主戦への乗り替わり",
    "app_to_regular": "減量騎手→一般（代理指標）",
}


def e(x) -> str:
    return html.escape(str(x))


def table(head: list[str], rows: list[list], num_from: int = 1) -> str:
    h = "".join(f"<th{' class=\"n\"' if i >= num_from else ''}>{e(c)}</th>" for i, c in enumerate(head))
    b = ""
    for r in rows:
        b += "<tr>" + "".join(f"<td{' class=\"n\"' if i >= num_from else ''}>{e(c)}</td>" for i, c in enumerate(r)) + "</tr>"
    return f"<table><tr>{h}</tr>{b}</table>"


def label(cell: str) -> str:
    if cell.startswith("type:"):
        return TYPE_LABEL[cell.split(":")[1]]
    ty, st = cell.split("|")
    return f"{TYPE_LABEL[ty]} × {STRATA_LABEL.get(st, st)}"


def roi(x: dict, k: str) -> str:
    return f"{x[k]:.0f}%（{x[k + '_lo']:.0f}〜{x[k + '_hi']:.0f}）"


def ratio(x: dict) -> str:
    return f"{x['win_ratio']:.2f}（{x['win_ratio_lo']:.2f}〜{x['win_ratio_hi']:.2f}）" if "win_ratio" in x else "—"


def stat_rows(cells: dict, names: list[str]) -> list[list]:
    rows = []
    for nm in names:
        for p in ("sel", "2025", "2026"):
            x = cells[nm][p]
            if not x.get("n"):
                rows.append([label(nm), PERIOD[p], 0, "—", "—", "—", "—", "—"])
                continue
            rows.append([label(nm), PERIOD[p], f"{x['n']:,}", f"{100 * x['win_rate']:.1f}%", f"{100 * x['top3_rate']:.1f}%", roi(x, "roi_win"), roi(x, "roi_place"), ratio(x)])
    return rows


def main(res: dict) -> None:
    cells, tests = res["cells"], res["tests"]
    main_names = [f"type:{k}" for k in ("NE", "EN", "EE", "NN")]
    base = res["baseline"]
    same = base["same_j"]
    pos = sum(1 for t in tests if t["t_abs"] == t["t_abs"] and t["t_abs"] > 0)
    rc = res["reality_check_abs"]
    neg = sorted((t for t in tests if t["t_exc"] == t["t_exc"]), key=lambda t: t["t_exc"])[:6]
    neg_cells = []
    for t in neg:
        if t["cell"] not in neg_cells:
            neg_cells.append(t["cell"])
    body = []
    body.append(f"""<a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>騎手の乗り替わり（有力↔非有力）に期待値はあるか</h1>
<p class="muted">2026-10-22、ユーザー質問『有力ジョッキーへの乗り替わり、およびその逆で期待値が取れるものがないか』への回答。
資料: <a href="equipment_and_change.html">馬具・条件替わり（tp-32、JOCKEY 群）</a>・<a href="jockey_trainer_roi.html">騎手・調教師の分析と回収率（tp-59）</a>・<a href="betting_edge_search.html">馬券で回収率 100% を超える余地の探索（tp-36）</a>・<a href="training_pipeline_decisions.html">決定ログ</a>。
回収率は『1 出走（1 点）に 100 円』。区間はレース単位ブートストラップ。選択は 2021〜2024、test 2025 は最良の数セルの評価 1 回、2026 年はテストセット②（選択に使わない。記述のみ）。市場オッズは払戻の計算と比較にだけ使い、特徴量には使わない。</p>

<div class="note"><b>要点（先に答え）.</b>
<ul>
<li><b>期待値が取れる乗り替わりの型は見つからなかった。</b>事前に列挙した 38 セル × 単勝・複勝 = 76 本のうち、選択期間（2021〜2024）で平均利益が正になったものは <b>{pos} 本</b>。White の Reality Check の p = {rc['p_value']:.2f}、Holm 補正後の最小 p = {res['min_holm_abs']:.2f}。回収率の区間の下限が 100% を超えたセルは無い。</li>
<li>主な 4 型（NE・EN・EE・NN）の単勝・複勝回収率は（選択・2025・2026 を通して）65〜87% で、継続騎乗（単勝 {same['sel']['roi_win']:.0f}%・複勝 {same['sel']['roi_place']:.0f}%）や全体（約 70%）と区別できる優位は無い。有力への乗り替わり（NE）は <b>人気になりすぎも過小評価もされていない</b>（オッズ帯の期待払戻との差は単勝 {cells['type:NE']['sel']['ex_win']:+.1f} 円/100 円、複勝 {cells['type:NE']['sel']['ex_place']:+.1f} 円）。</li>
<li>モデル（V4 の勝率）に対する実績の比は、NE が選択 {cells['type:NE']['sel']['win_ratio']:.2f} → 2025 {cells['type:NE']['2025']['win_ratio']:.2f} → 2026 {cells['type:NE']['2026']['win_ratio']:.2f}、EN が {cells['type:EN']['sel']['win_ratio']:.2f} → {cells['type:EN']['2025']['win_ratio']:.2f} → {cells['type:EN']['2026']['win_ratio']:.2f} と、<b>1.0 の周りを年ごとに上下するだけ</b>で、方向が続く型は無い。モデルはすでに乗り替わりの影響を織り込んでいる。</li>
<li>『逆』（有力→非有力）も同様。人気が下がって妙味が出る、という形跡は無い。<b>唯一、選択期間で有意に悪かった層</b>は『有力→非有力で、前走 1〜3 着の馬』（単勝 {cells['EN|prev_good(1-3)']['sel']['roi_win']:.0f}%・複勝 {cells['EN|prev_good(1-3)']['sel']['roi_place']:.0f}%、オッズ帯の期待を単勝 {cells['EN|prev_good(1-3)']['sel']['ex_win']:+.0f} 円下回る。76 本中でも Holm 後に有意）だが、2025・2026 は期待並みに戻った（単勝 {cells['EN|prev_good(1-3)']['2025']['roi_win']:.0f}%・{cells['EN|prev_good(1-3)']['2026']['roi_win']:.0f}%）ので、『避ける』根拠としても採らない。</li>
<li>凍結する候補は無い（tp-59 の手順では、選択期間の Holm 補正後に残った候補だけを凍結して前向き追跡する。今回は残らなかった）。特徴量化も、tp-32 の JOCKEY 群（乗り替わり・格の差）が V4 に対して増分なしだった結論と整合する。</li>
</ul></div>

<h2>1. 事前に固定した設計</h2>
<ul>
<li><b>有力ジョッキー</b>: その日より前の直近 365 日（当日を含まない）の<b>平地の勝利数</b>の上位 N 人（同数は騎乗数、次に ID）。回収率は定義に使わない。N = 10 が主、5 と 20 は感度。jockey_id がキー（名前では結合しない）。</li>
<li><b>前走騎手の格</b>も『今日時点の』順位で測る（現騎手と同じ物差し。tp-32 の <code>jk_rate_prev</code> と同じ考え方）。前走が不明（2020 年以前の履歴）の出走は除く。障害は除く。</li>
<li><b>型</b>: 継続 / 非有力→有力（NE）/ 有力→非有力（EN）/ 有力→有力（EE）/ 非有力→非有力（NN）。</li>
<li><b>仮説の族</b>: 型 4 + NE・EN それぞれの層別 17（テン乗り/戻り、前走着順 3、今走人気 3、芝/ダ、新馬未勝利/1 勝以上、距離 3、同厩舎の主戦への乗り替わり、減量騎手→一般）= 38 セル × 単勝・複勝 = 76 本。結果を見てから増やさない。</li>
<li><b>選択期間</b> 2021〜2024（2020 年は 365 日の窓が埋まらない助走）。<b>test 2025</b> は選んだ最良の数セルを当てて見る 1 回。<b>2026</b> は記述のみ。</li>
<li><b>検定</b>: 買ったレースだけで、レース単位の利益の平均が 0 より大きいか（片側 t）。Holm 補正と White の Reality Check を併用。<b>絶対の回収率</b>（> 100%）と、<b>オッズ帯の期待払戻を引いた超過払戻</b>（市場が値付けを間違えているか）の 2 通り。</li>
<li><b>減量騎手・外国人・短期免許</b>: データに見習いの印（▲△☆）が無いため、見習い減量は『同じ性齢の最頻の斤量より 2kg 以上軽い（8 頭立て以上）』の代理指標。外国人・短期免許は ID で区別せず、勝利数の順位に任せる（短期免許は 365 日の平地勝利数が少なく、ほぼ非有力に入る。通年で上位の外国人騎手は有力に入る）。</li>
<li>モデルの予測は tp-59 の master にある V4（predictions_v3、2022 年以降）。V5/v6e の馬ごとの予測は、同じ形では持っていないので使っていない（V5 の gain の約 1% が騎手・調教師系で、結論を変える大きさではない）。</li>
</ul>

<h2>2. 主な 4 型の成績</h2>
<p class="muted">『勝率の実績/予測』は 実際の勝ち数 ÷ V4 の予測勝率の合計（1.0 より大きいほどモデルより走った）。2022 年以降のみ。</p>
""")
    body.append(table(["型", "期間", "件数", "勝率", "3着内", "単勝回収率（95%区間）", "複勝回収率（95%区間）", "勝率の実績/予測"], stat_rows(cells, main_names), 2))
    sr = []
    for p, v in same.items():
        sr.append([TYPE_LABEL["SAME"], PERIOD[p], f"{v['n']:,}", f"{100 * v['win_rate']:.1f}%", f"{100 * v['top3_rate']:.1f}%", roi(v, "roi_win"), roi(v, "roi_place"), ratio(v)])
    body.append("<h3>参考: 継続騎乗（同じ騎手）</h3>" + table(["型", "期間", "件数", "勝率", "3着内", "単勝回収率（95%区間）", "複勝回収率（95%区間）", "勝率の実績/予測"], sr, 2))

    body.append("<h2>3. 有力の定義を変えても同じか（感度: 上位 5 / 20 人）</h2>")
    srows = []
    for top, d in res["sensitivity"].items():
        for ty in ("NE", "EN"):
            for p in ("sel", "2025", "2026"):
                x = d[ty][p]
                srows.append([f"上位 {top} 人", TYPE_LABEL[ty], PERIOD[p], f"{x['n']:,}", roi(x, "roi_win"), roi(x, "roi_place"), ratio(x)])
    body.append(table(["有力の定義", "型", "期間", "件数", "単勝回収率", "複勝回収率", "勝率の実績/予測"], srows, 3))

    body.append(f"""<h2>4. 76 本の検定（選択期間 2021〜2024）</h2>
<ul>
<li>平均利益が正（回収率 > 100%）になったセル: <b>{pos} / {res['n_hypotheses']}</b>。最大の t 値は {rc['t_max']:.2f}（{e(rc['best_cell'])}）で負。Reality Check p = {rc['p_value']:.2f}。</li>
<li>オッズ帯の期待を上回る（市場の値付け違い）セル: Holm 補正後の最小 p = {res['min_holm_exc']:.2f}。上回ったと言えるセルは無い。</li>
<li>下は、t 値が最大だった 5 本（大半が件数の小さい層）を、そのまま test 2025・2026 に当てた結果。</li>
</ul>""")
    brows = []
    for b in res["best5"]:
        c = cells[b["cell"]]
        col = "roi_win" if b["bet"] == "win" else "roi_place"
        brows.append([label(b["cell"]), "単勝" if b["bet"] == "win" else "複勝", f"{b['t']:.2f}"] + [f"{c[p]['n']:,} / {c[p][col]:.0f}%" if c[p].get("n") else "—" for p in ("sel", "2025", "2026")])
    body.append(table(["セル", "買い方", "選択期間の t", "選択（件数 / 回収率）", "2025", "2026"], brows, 2))

    body.append("<h2>5. 逆向きの手がかり: 選択期間で最も悪かった層（避ける根拠になるか）</h2>")
    nrows = []
    for nm in neg_cells:
        for p in ("sel", "2025", "2026"):
            x = cells[nm][p]
            if x.get("n"):
                nrows.append([label(nm), PERIOD[p], f"{x['n']:,}", roi(x, "roi_win"), roi(x, "roi_place"), f"{x['ex_win']:+.1f} / {x['ex_place']:+.1f}", ratio(x)])
    body.append(table(["セル", "期間", "件数", "単勝回収率", "複勝回収率", "オッズ帯期待との差 単/複（円）", "勝率の実績/予測"], nrows, 2))
    body.append("""<p class="muted">『有力→非有力で前走 1〜3 着』は選択期間では買われすぎ（回収率が帯の期待より 16 円低い）だったが、2025・2026 は期待並みに戻った。市場が学習した可能性と偶然の両方があり得るので、『避ける』規則にもしない。</p>

<h2>6. 限界と、次にやること</h2>
<ul>
<li>テン乗り（その馬への初騎乗）は 2020 年以降の履歴で判定している。それ以前に乗っていた騎手は『初騎乗』と誤判定され得る（戻りの一部が混ざる）。</li>
<li>モデルとの比較は V4。V5/v6e の馬ごとの予測を持つ表ができれば、同じセルで比を取り直せる（結論が動く可能性は小さい）。</li>
<li>ここでの回収率は『セルの馬を全部買う』形で、EV の高い馬だけを選ぶ形ではない。NE/EN に EV 条件を重ねる検証は、tp-59 の凍結規則（単勝 EV ≥ 1.5 と騎手・調教師の残差）が前向き追跡中なので、そちらの結果が出てから重ねる。</li>
<li>『有力』を勝利数にしているので、1 年で急に有力になった騎手（若手の台頭）は遅れて有力に入る。定義を増やすと多重比較が増えるため、今回は変えない。</li>
</ul>

<h2>7. 再現</h2>
<pre>python -m src.scripts.data.jockey_switch_ev build     # tp-59 の master.parquet から as-of の特徴を作る（JSW_OUT/frame.parquet）
python -m src.scripts.data.jockey_switch_ev analyze   # 型ごとの集計・検定（JSW_OUT/result.json）
python -m src.scripts.data.jockey_switch_ev report    # この章を作る</pre>
<p class="muted">関連: <a href="training_pipeline_decisions.html">決定ログ</a>・<a href="equipment_and_change.html">馬具・条件替わり（tp-32）</a>・<a href="jockey_trainer_roi.html">騎手・調教師の分析と回収率（tp-59）</a>・<a href="betting_edge_search.html">馬券で回収率 100% を超える余地の探索（tp-36）</a></p>
</body></html>""")
    head = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>騎手の乗り替わりに期待値はあるか — keiba-vpn</title>
<style>{CSS}</style></head>
<body>"""
    OUT.write_text(head + "".join(body), encoding="utf-8")
    print("wrote", OUT)
