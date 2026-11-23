"""めぐ分析の条件が回収率 100% を超えない理由（docs/html/operations/final/megu_why_below_100.html）を、診断の JSON から作る（数字を手で写さない）。

入力（src/research/course/megu_why.py が書く）:
  docs/operations/retrain_trials/megu_why_flat.json（平場: collect・perm・report）
  docs/operations/retrain_trials/megu_why_graded.json（重賞: graded）
  config/data_access/ledger.jsonl（見た回数。src.ops.data_ledger.tally で数える）
使い方: python -m src.scripts.docs.build_megu_why
費用 0 円（手元のファイルだけ）。
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scipy.stats import chi2

from src.scripts.docs.build_latest_strategy_tables import CSS

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/operations/retrain_trials"
DST = ROOT / "docs/html/operations/final/megu_why_below_100.html"
TITLE = "めぐ分析が 100% を超えない理由"
JST = timezone(timedelta(hours=9))
LOOK = ("2025-01-01", "2026-11-20")


def esc(x) -> str:
    return html.escape(str(x))


def pc(x, d: int = 1) -> str:
    return "－" if x is None else f"{x * 100:.{d}f}%"


def pt(x) -> str:
    return "－" if x is None else f"{x * 100:+.1f}pt"


def num(x, d: int = 2) -> str:
    return "－" if x is None else f"{x:.{d}f}"


def table(head: list[str], rows: list[list], num_cols=frozenset()) -> str:
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f'<td class="n">{c}</td>' if i in num_cols else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ratio_ci(w: float, q: float) -> tuple[float, float, float]:
    """勝ち数 w ÷ 見込み q と、w をポアソンとみた 95% 区間。"""
    lo = chi2.ppf(0.025, 2 * w) / 2 if w > 0 else 0.0
    return w / q, lo / q, chi2.ppf(0.975, 2 * (w + 1)) / 2 / q


def looks() -> dict:
    """台帳（git の写し）で、2025-01-01〜2026-11-20 を見た回数（系統ごと）と、この診断（megu-why）の回数。"""
    from src.ops import data_ledger as DL
    rows = DL.load_rows([DL.GIT_LEDGER])
    mine = [r for r in rows if r.get("session") == "megu-why" or "megu_" in str(r.get("tool", ""))]
    return {fam: {"all": DL.tally(rows, LOOK, fam)["family"], "mine": DL.tally(mine, LOOK, fam)["family"]} for fam in ("course", "graded")}


def _cal(x: dict, key: str = "cal") -> str:
    c = x.get(key) or {}
    return f"{num(c.get('ratio'))}（{num(c.get('lo'))}〜{num(c.get('hi'))}）" if x.get("n") else "－"


def _pass(x: dict) -> bool:
    """確かめと同じ決まり（切り口 対 外の比 > 1 かつ単勝が基準線より高い）。"""
    return bool(x.get("n") and x["cal"]["ratio"] > 1.0 and x["roi_t"] > x["base_t"])


def oikiri_section() -> str:
    """第 2 段: 追い切りの切り口（src/research/course/megu_oikiri.py）。"""
    p = SRC / "megu_oikiri.json"
    if not p.exists():
        return ""
    O = json.loads(p.read_text(encoding="utf-8"))
    fin = O.get("final") or {}
    conf = [k for k, r in O["flat"].items() if r["confirmed"]]
    sel2 = [k for k, r in O["flat"].items() if r["select"].get("n", 0) >= 300 and (r["select"].get("cal_lo_bonf2") or 0) > 1.0]
    o = ["<h2>第 2 段: 追い切りの切り口を足す（調教師ごとの最終追い切りの出し方）</h2>"]
    if fin:
        li = []
        for k in conf:
            x = fin["flat"][k]["2025_2026"]
            need = x["edge"]["ratio"] / x["roi_t"] if x.get("roi_t") else None      # 払戻の形が同じとき、100% に要る市場比
            li.append(f"<li>{esc(O['cuts'][k])}（2025・26 年）: 市場比 {_cal(x, 'edge')}、単勝 {pc(x.get('roi_t'))}。"
                      f"確かめと同じ決まりでは{'通る' if _pass(x) else '落ちる'}（切り口 対 外の比 {_cal(x)}、外の基準線 {pc(x.get('base_t'))}）。"
                      f"100% に要る市場比は約 {num(need)}。</li>")
        mk_in = all((fin["flat"][k]["2025_2026"]["edge"]["lo"] or 0) <= 1.0 for k in conf)
        head = ("追い切りの切り口に、市場より勝つ確かな効きは見つからなかった（確かめを通った切り口の市場比の区間は、どれも 1 を含む）。" if mk_in
                else "確かめを通った切り口のうち、市場比の区間が 1 を超えるものがある。100% に届くかは下の値を見る。")
        thin = all((r["select"].get("cal_lo_bonf") or 0) < 1.01 for r in O["flat"].values() if r["selected"])
        o.append(f"""<div class="box"><b>第 2 段の結論</b>: {head}
<ul>
<li>{O['k']} 本のうち、選ぶ期間（2019〜21 年）の決まり（片側の Bonferroni の下限 > 1）を通ったのは {sum(r['selected'] for r in O['flat'].values())} 本{'で、どれも下限が 1 をわずかに（1% 未満）超えただけ' if thin else ''}。
両側の Bonferroni にすると {len(sel2)} 本になる。</li>
<li>片側で選んだ本のうち、確かめ（2022〜24 年）も通ったのは {len(conf)} 本（{'・'.join(esc(O['cuts'][k]) for k in conf)}）。</li>
{''.join(li)}
<li>「切り口 対 外の比」が 1 を超えるのは、外の馬（加速なら減速の馬）が市場に負けることも含む。2025・26 年の合否の決まりは事前に定めていない（下の表の「確かめの決まり」は参考）。</li>
<li>めぐ分析の B に足しても（B∩切り口）、上積みは確かめられない（下の表）。</li>
<li>2026 年は追い切りの記録がある走が {pc(fin['coverage']['2026']['flat'], 0)}（平場）しかない。2025 年は {pc(fin['coverage']['2025']['flat'], 0)}。
表の終わり（{esc(fin.get('laps_max', '－'))}）だけでは説明できず、原因は確かめていない。2026 年の値は参考にとどめる。</li>
</ul></div>""")
    o.append(f"""<p class="sub">道具: <code>src/research/course/megu_oikiri.py</code>。第 1 段の診断に従い、結果で組を拾う探し方はしない。切り口 {O['k']} 本を先に決めて測った。</p>
<ul>
<li><b>切り口 対 外の比</b>: 勝ち数 ÷ 見込み。見込みは「追い切りの記録があって切り口に当たらない走」の、同じ期間・芝ダ・群・確定の人気の勝率から作る。本命・大穴の偏りと記録の有無の偏りは除くが、比べる相手は切り口ごとに違う（坂路とウッドは裏返し、加速の相手は減速）。行どうしは比べられない。</li>
<li><b>市場比</b>: 勝ち数 ÷ 確定オッズの見込み。</li>
<li><b>決まり</b>: {esc(O['rule'])}。2025・2026 年は、確かめを通った切り口だけを 1 回数える。</li>
<li><b>経緯</b>: {esc(O.get('history', ''))}</li>
<li><b>注</b>: 加速は 1F が前の 1F より速いこと。ウッド・ポリの前の 1F は近似（(3F − 1F) ÷ 2）なので、坂路と意味が少し違う。種別ごとの内訳は出していない。</li>
<li><b>最終追い切り</b>: レース週の平日の 4F が最も速い 1 本（発走より前）。調教師のいつもは、その日より前の日のレースの最終追い切り。記録の無い走は、どの切り口にも当てはまらないとして数える。</li>
</ul>""")
    cov = dict(O.get("coverage") or {}) | dict(fin.get("coverage") or {})
    o.append("<h3>追い切りの記録がある走の割合（年ごと）</h3>")
    o.append(table(["面"] + list(cov), [["平場"] + [pc(v["flat"], 0) for v in cov.values()], ["重賞"] + [pc(v["graded"], 0) for v in cov.values()]], set(range(1, len(cov) + 1))))
    o.append('<p class="sub">2018 年までは記録が薄いので、選ぶ期間は 2019〜21 年にした。</p>')
    rows = []
    for k, lab in O["cuts"].items():
        r = O["flat"][k]
        s_, c_ = r["select"], r["confirm"]
        mark = "通過" if r["confirmed"] else ("選んだが確かめで落ちた" if r["selected"] else "選ばない")
        rows.append([esc(lab), f"{s_.get('n', 0):,}", _cal(s_), num(s_.get("cal_lo_bonf"), 3), num(s_.get("cal_lo_bonf2"), 3), _cal(s_, "edge"),
                     f"{c_.get('n', 0):,}", _cal(c_), _cal(c_, "edge"), pc(c_.get("roi_t")), pc(c_.get("base_t")), mark])
    o.append("<h3>平場: 2019〜21 年で選び、2022〜24 年で確かめる</h3>")
    o.append(table(["切り口", "選ぶ期間の走数", "切り口 対 外の比", "片側の補正の下限", "両側の補正の下限", "市場比", "確かめの走数", "切り口 対 外の比", "市場比", "単勝", "外の基準線", "判定"],
                   rows, set(range(1, 11))))
    grows = []
    for k, lab in O["cuts"].items():
        r = O["graded"][k]
        grows.append([esc(lab), f"{r['select'].get('n', 0):,}", _cal(r["select"]), f"{r['confirm'].get('n', 0):,}", _cal(r["confirm"]), _cal(r["confirm"], "edge"),
                      pc(r["confirm"].get("roi_t")), pc(r["confirm"].get("base_t"))])
    o.append("<h3>重賞: 選ばずに同じ切り口を当てる（参考）</h3>")
    o.append('<p class="sub">件数が少なく、比べる相手（切り口の外）も小さいので、基準線が大きく揺れる。</p>')
    o.append(table(["切り口", "2019〜21 の走数", "切り口 対 外の比", "2022〜24 の走数", "切り口 対 外の比", "市場比", "単勝", "外の基準線"], grows, set(range(1, 8))))
    if fin:
        o.append("<h3>2025・2026 年（確かめを通った切り口だけ、1 回だけ）</h3>")
        fr = []
        for side, sl in (("flat", "平場"), ("graded", "重賞")):
            for k in fin["cuts"]:
                for per in ("2025", "2026", "2025_2026"):
                    x = fin[side][k][per]
                    fr.append([sl, esc(O["cuts"][k]), per.replace("_", "・"), f"{x.get('n', 0):,}", _cal(x, "edge"), _cal(x), pc(x.get("roi_t")), pc(x.get("base_t")),
                               pc(x.get("roi_f")), pc(x.get("base_f")), "通る" if _pass(x) else "落ちる"])
        o.append(table(["面", "切り口", "年", "走数", "市場比", "切り口 対 外の比", "単勝", "外の基準線（単）", "複勝", "外の基準線（複）", "確かめの決まり"], fr, set(range(3, 10))))
        o.append("<h3>めぐ分析の B（単勝）: 追い切りなし／あり</h3>")
        o.append('<p class="sub">B は凍結の第 3 版の買い目（成績表 B と同じ）。「B∩」は B のうち切り口に当たる馬だけ、「B−」は当たらない馬だけを買う場合。切り口は B を見ずに選んだ。'
                 'B− には追い切りの記録が無い馬も入るので、2026 年（記録が少ない）の B∩ と B− の差は、記録の有無の差と混ざる。</p>')
        br = []
        for per, row in fin["b"].items():
            for name, x in row.items():
                br.append([per, esc(name), f"{x.get('n', 0):,}", pc(x.get("roi_t")), _cal(x, "edge")])
        o.append(table(["年", "買い方", "点数", "単勝の回収率", "市場比"], br, {2, 3, 4}))
    return "\n".join(o)


def build() -> str:
    F = json.loads((SRC / "megu_why_flat.json").read_text(encoding="utf-8"))
    G = json.loads((SRC / "megu_why_graded.json").read_text(encoding="utf-8"))
    LK = looks()
    bt, bf, base = F["buys"]["tansho"], F["buys"]["fukusho"], F["base"]
    perms = F["perm"]
    real0 = next(p for p in perms if p["fit_end"] == 2024 and p["seed"] == 0)
    nulls = [p for p in perms if p["fit_end"] == 2024 and p["seed"] > 0]
    refit = next(p for p in perms if p["fit_end"] == 2021 and p["seed"] == 0)
    nulls21 = [p for p in perms if p["fit_end"] == 2021 and p["seed"] > 0]
    n_t = [p["in_sample"]["tansho"]["roi"] for p in nulls]
    n_f = [p["in_sample"]["fukusho"]["roi"] for p in nulls]
    ho_t, ho_f = refit["holdout"]["tansho"], refit["holdout"]["fukusho"]
    nh_t = [p["holdout"]["tansho"]["roi"] for p in nulls21]
    nh_f = [p["holdout"]["fukusho"]["roi"] for p in nulls21]
    cmp_null = ("帰無の幅の中に入る（区別できない）" if min(nh_t) <= ho_t["roi"] <= max(nh_t)
                else ("帰無の幅より低い（偶然以上のものを持たない）" if ho_t["roi"] < min(nh_t) else "帰無の幅より高い"))
    S, R, MC = F["shrink"], F["rules"], F["market_cols"]
    tr_years_main = [y for y in F["by_year"] if y["year"] <= 2024 and y["bets_t"] >= 100]
    # 探した後（2025・26 年）の買い目の勝ち数 ÷ 市場の見込み（馬ごとに重複なし）
    w_post = bt["2025"]["edge_ci"]["w"] + bt["2026"]["edge_ci"]["w"]
    q_post = bt["2025"]["edge_ci"]["q"] + bt["2026"]["edge_ci"]["q"]
    e_post, e_lo, e_hi = ratio_ci(w_post, q_post)
    sp_post = _mean([bt["2025"]["same_pop_roi"], bt["2026"]["same_pop_roi"]])
    need = 1 / sp_post if sp_post else None                                  # 基準線から 100% に要る勝ち数の倍率（払戻の形が同じとき）
    sp_train = bt["train"]["same_pop_roi"]
    gi, ge, g26, pr = G["in_sample_2022_2024"], G["eval_2025_2026"], G["eval_2026"], G["prereg_dev"]
    pc_cells = G["primary_cells"]
    pc_perm = [v for k, v in pc_cells.items() if k.startswith("perm")]
    made = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    flb_gap = base["train"]["payback"] - base["train"]["roi_t"]
    cal_tr = {r["band"]: r for r in F["calibration"]["train"]}

    o = [f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>{TITLE}</h1>
<p class="sub">作成 {made}（JST）。数字は診断の道具から生成（<code>src/research/course/megu_why.py</code>・<code>src/scripts/docs/build_megu_why.py</code>）。
対象は <a href="latest_strategy_tables.html">最新の買い方の成績表</a> の B（平場）と、重賞の好走データ予想の主に見る版。条件・閾値は選び直していない。2025・2026 年は評価だけ（台帳に eval_only）。</p>
"""]

    o.append(f"""<div class="box">
<b>結論</b>
<p>差は 2 つに分かれる。2 つは足し合わない。</p>
<ol>
<li><b>訓練が高く見えた理由は、探し方である</b>（単勝で訓練 {pc(bt['train']['roi'])} → 探した後 {pc(bt['2025']['roi'])}・{pc(bt['2026']['roi'])}）。
<ul>
<li>条件は「訓練で 3 着内率 70% 以上かつ単か複の回収率 100% 超」の組から残した（<code>src/research/graded/buy.py:50</code>）。訓練の値は機械的に高くなる。</li>
<li>切り口と結果の関係を消した帰無の並べ替えで同じ探し方を流すと、訓練の単勝は {pc(min(n_t))}〜{pc(max(n_t))}（{len(nulls)} 回）。本物の {pc(bt['train']['roi'])} はこの幅に入る。</li>
<li>2016〜21 年だけで探し直し、2022〜24 年に当てると {pc(ho_t['roi'])}。帰無の版を同じく当てると {pc(min(nh_t))}〜{pc(max(nh_t))}（{len(nulls21)} 回）で、本物は{cmp_null}。2025・26 年を使わない確かめである。</li>
</ul></li>
<li><b>探した後に 100% に届かない理由は、条件が市場より先の情報を持たず、払戻率の壁を越えられないことである</b>。
<ul>
<li>探した後の市場比は {num(e_post)}（95% 区間 {num(e_lo)}〜{num(e_hi)}、2025・26 年、{w_post:.0f} 勝）。市場どおりである。</li>
<li>回収率は、同じ人気の基準線と同じ水準。2025 年は {pc(bt['2025']['roi'])} 対 {pc(bt['2025']['same_pop_roi'])}、2026 年は {pc(bt['2026']['roi'])} 対 {pc(bt['2026']['same_pop_roi'])}。</li>
<li>壁の 1 つ目は控除。単勝の払戻率は {pc(base['2025']['payback'])}（2025 年）・{pc(base['2026']['payback'])}（2026 年）。</li>
<li>壁の 2 つ目は本命・大穴の偏り。人気薄ほど回収率が低く、全馬を買うと {pc(base['2025']['roi_t'])}・{pc(base['2026']['roi_t'])}。</li>
<li>同じ人気の基準線（約 {pc(sp_post, 0)}）から 100% に届くには、市場の約 {num(need)} 倍勝つ必要がある。探した後の市場比の区間の上限 {num(e_hi)} は、そこに届かない。</li>
</ul></li>
</ol>
<p>時代の変化・配当の偏り・重複・オッズの時点は、どちらの差の主因でもない（仮説ごとの表）。</p>
<p><b>追い切りの切り口（第 2 段）</b>: 調教師ごとの最終追い切りの出し方を切り口に足しても、市場より勝つ確かな効きは見つからなかった（下の第 2 段の節）。</p>
<p><b>重賞</b>: 平場と同じく、探した期間の中では高く、探した後は落ちる。</p>
<ul>
<li>探した期間の中（2022〜24 年）: 単勝 {pc(gi['roi_t'])}・複勝 {pc(gi['roi_f'])}。</li>
<li>開発の確かめ（2016〜21 年で探し 2022〜24 年に当てる）: {pc(pr['roi_t'])}・{pc(pr['roi_f'])}。同じレースの全馬は {pc(pr['all_roi_t'])}・{pc(pr['all_roi_f'])}。</li>
<li>2025・26 年: 単勝 {pc(ge['roi_t'])}・複勝 {pc(ge['roi_f'])}（{ge['bets']} 頭・{ge['bet_races']} レース）。</li>
<li>主に見る版は複勝の版。複勝は同じ人気の基準線（{pc(ge['same_pop_roi_f'])}）より {pt(ge['roi_f'] - ge['same_pop_roi_f'])} 高いが、区間 {pc(ge['roi_f_lo'], 0)}〜{pc(ge['roi_f_hi'], 0)} は基準線も 100% も含む。件数が少なく、判定できない。</li>
</ul>
</div>
<div class="box">
<b>言葉</b>
<ul>
<li><b>市場の見込み</b>: 確定の単勝オッズの逆数を、レースの中で和 1 に直した勝率。</li>
<li><b>市場比</b>: 買った馬の勝ち数 ÷ 市場の見込みの勝ち数の和。1 なら市場どおり。断りが無ければ買い目（馬ごとに重複なし）で数える。</li>
<li><b>同じ人気の基準線</b>: 同じ期間・芝ダ・クラスの群・確定の人気の、買わなかった馬の回収率を、買った馬の人気の分布で重みづけした値。市場どおりに同じ人気の馬を買ったときの値。</li>
<li><b>帰無の並べ替え</b>: 切り口と結果の関係だけを消したデータで、同じ探し方を流し直すこと。偶然だけで出る値の大きさが分かる。</li>
<li><b>当てはめ</b>: 条件を探した期間の値を、同じ期間で数えたもの。</li>
</ul>
</div>
""")

    o.append(f"<h2>平場: 訓練 {pc(bt['train']['roi'])} から探した後までの段</h2>")
    o.append('<p class="sub">どの段も凍結の第 3 版と同じ設定（<code>src/research/course/prereg_course_v3.json</code>）で探す。表示の上位 5 の条件のうち、訓練の回収率 100% 以上に当てはまる馬を、馬ごとに重複なしで 1 点ずつ買う。</p>')
    rows = [
        ["① 凍結の版の当てはめ（2016〜24 年で探し、同じ期間で数える）", pc(bt["train"]["roi"]), pc(bf["train"]["roi"]), f"{bt['train']['bets']:,}", f"同じ人気の基準線 {pc(sp_train)}・{pc(bf['train']['same_pop_roi'])}"],
        [f"② 帰無の並べ替えで同じ探し方（{len(nulls)} 回）", f"{pc(min(n_t))}〜{pc(max(n_t))}", f"{pc(min(n_f))}〜{pc(max(n_f))}", f"{_mean([p['in_sample']['tansho']['bets'] for p in nulls]):,.0f}（平均）", "①はこの幅に入る"],
        ["③ 2016〜21 年で探し直す（当てはめ）", pc(refit["in_sample"]["tansho"]["roi"]), pc(refit["in_sample"]["fukusho"]["roi"]), f"{refit['in_sample']['tansho']['bets']:,}", ""],
        ["④ ③の条件を 2022〜24 年に当てる", pc(ho_t["roi"]), pc(ho_f["roi"]), f"{ho_t['bets']:,}", f"同じ人気の基準線 {pc(ho_t['base_same_pop_roi'])}・{pc(ho_f['base_same_pop_roi'])}（この行は基準線に買った馬を含む）"],
        [f"④' 帰無の版（2016〜21 年）を 2022〜24 年に当てる（{len(nulls21)} 回）", f"{pc(min(nh_t))}〜{pc(max(nh_t))}", f"{pc(min(nh_f))}〜{pc(max(nh_f))}", "", f"④は{cmp_null}"],
        ["⑤ 凍結の版を 2025 年に当てる", pc(bt["2025"]["roi"]), pc(bf["2025"]["roi"]), f"{bt['2025']['bets']:,}", f"区間 {pc(bt['2025']['roi_lo'], 0)}〜{pc(bt['2025']['roi_hi'], 0)}。2025 年は第 3 版を選ぶのに使った期間"],
        ["⑥ 凍結の版を 2026 年に当てる", pc(bt["2026"]["roi"]), pc(bf["2026"]["roi"]), f"{bt['2026']['bets']:,}", f"区間 {pc(bt['2026']['roi_lo'], 0)}〜{pc(bt['2026']['roi_hi'], 0)}"],
        ["⑦ 同じ人気の基準線（2025・2026 年）", f"{pc(bt['2025']['same_pop_roi'])}・{pc(bt['2026']['same_pop_roi'])}", f"{pc(bf['2025']['same_pop_roi'])}・{pc(bf['2026']['same_pop_roi'])}", "", "⑤⑥と同じ水準"],
        ["⑧ 全馬（2025・2026 年）", f"{pc(base['2025']['roi_t'])}・{pc(base['2026']['roi_t'])}", f"{pc(base['2025']['roi_f'])}・{pc(base['2026']['roi_f'])}", "",
         f"単勝の払戻率 {pc(base['2025']['payback'])}・{pc(base['2026']['payback'])}"],
    ]
    o.append(table(["段", "単勝", "複勝", "点数（単勝）", "注"], rows, {1, 2, 3}))
    o.append('<p class="sub">③④は、凍結と同じ設定を 2016〜21 年に当てた。設定（市場比の下限・人気の上限など）そのものは、2016〜24 年の当てはめと 2025 年を見て選んだ。年ごとの安定性の検査は、2021 年までだと塊が 2 つになり、凍結より緩い。どちらも④を高く見せる向きに働く。</p>')

    o.append("<h2>平場: 仮説ごとの確かめ</h2>")
    o.append('<p class="sub">「どの差か」は、訓練が高く見えた差（訓練 → 探した後）と、探した後に 100% に届かない差（探した後 → 100%）のどちらを説明するか。</p>')
    hyp = [
        ["H2 選び過ぎ", "訓練 → 探した後（主因）",
         f"帰無でも訓練の単勝は {pc(min(n_t))}〜{pc(max(n_t))}。2016〜21 年で探すと、2022〜24 年は {pc(ho_t['roi'])}。帰無の版は {pc(min(nh_t))}〜{pc(max(nh_t))} で、本物は{cmp_null}。",
         "<code>src/research/graded/buy.py:50</code>・<code>src/research/course/search.py:25</code>（1 単位で最大 20 万組）"],
        ["H2 縮小（条件ごと）", "訓練 → 探した後",
         f"条件ごと（重複あり）の市場比は、訓練 {num(S['all']['edge_in'])}、探した後 {num(S['all']['edge_post'])}（区間 {num(S['edge_post_ci']['lo'])}〜{num(S['edge_post_ci']['hi'])}）。"
         f"訓練の上乗せのうち残った割合は {pc(S['all']['keep'])}（区間 {pc(S['keep_lo'], 0)}〜{pc(S['keep_hi'], 0)}）。",
         "下の帯の表"],
        ["H2 縮小（2026 年だけ）", "訓練 → 探した後",
         f"条件ごとの市場比は {num(S['edge_2026_ci']['ratio'])}（区間 {num(S['edge_2026_ci']['lo'])}〜{num(S['edge_2026_ci']['hi'])}）。2025 年は第 3 版を選んだ期間なので、分けて示す。", "－"],
        ["H2 前半と後半", "訓練 → 探した後",
         f"訓練の前半（2016〜21 年）と後半（2022〜24 年）の、条件ごとの市場比の相関は {num(F['rules_split']['corr_edge_ab'])}。どちらも当てはめの期間なのに相関が無い。偶然で残った条件である印。", "－"],
        ["H1 控除と本命・大穴の偏り", "探した後 → 100%（壁）",
         f"単勝の払戻率は {pc(base['train']['payback'])}（訓練）・{pc(base['2025']['payback'])}・{pc(base['2026']['payback'])}。全馬の回収率はそれより約 {pc(flb_gap)} 低い（訓練）。人気薄ほど低い（較正の表の全馬の列）。"
         f"1 番人気の勝率は {pc(base['train']['fav_win'])}・{pc(base['2025']['fav_win'])}・{pc(base['2026']['fav_win'])} で、年ごとに安定。",
         "JRA の払戻率（単勝・複勝 80%）"],
        ["H1 較正（買った馬）", "探した後 → 100%",
         f"買い目の市場の見込みは平均 {pc(bt['train']['q_mean'])}（訓練）・{pc(bt['2026']['q_mean'])}（2026 年）。実際の勝率は {pc(bt['train']['win'])}・{pc(bt['2026']['win'])}。訓練の勝率が高いのは、勝った走を選んだから。",
         "下の較正の表"],
        ["H3 時代の変化", "どちらも小",
         f"2016〜21 年で探し 2022〜24 年に当てた {pc(ho_t['roi'])} は、2025・26 年と同じ水準。全馬の回収率・1 番人気の勝率・払戻率は年ごとにほぼ一定。"
         f"訓練の中の年ごとの値（100 点以上の年で {pc(min(y['roi_t'] for y in tr_years_main), 0)}〜{pc(max(y['roi_t'] for y in tr_years_main), 0)}）は当てはめなので、時代の根拠には使わない。",
         "下の年ごとの表"],
        ["H4 配当の偏り", "どちらも小",
         f"上位 1% の当たりが払戻に占める割合は {pc(bt['train']['top1pct_share'], 0)}（訓練）。上位 5% を除いても訓練は {pc(bt['train']['trim5'])}（単）・{pc(bf['train']['trim5'])}（複）。"
         f"訓練の高さは少数の大穴ではなく、勝率（{pc(bt['train']['win'])}）で作られている。人気薄（20〜50 倍）の訓練の単勝は {pc(cal_tr.get('20〜50', {}).get('roi'))}（当てはめ）。",
         "較正の表のオッズ帯"],
        ["H5 オッズの時点", "0",
         f"表示の条件 {MC['rules']} 個が使う切り口の型 {MC['types']} 種のうち、市場の列（<code>{esc(MC['forbidden_pattern'])}</code>）を使う型は {len(MC['market_types'])} 種（道具の report が検査する）。"
         "当日の買う・買わないはオッズで決まらず、確定オッズでも T-60 でも同じ馬を買う。払戻はどちらでも確定。"
         "なお、凍結の条件の絞り込み（市場比の下限 1.2）は訓練の確定オッズを使う。これは訓練の中の選び方で、当日の判定ではない。",
         "<code>src/research/graded/catalog.py:45</code>・<code>:54</code>"],
        ["H6 重複と相関", "どちらも小",
         f"1 点あたりの条件の数は {num(bt['train']['rules_per_bet'])}（2 つ以上は {pc(bt['train']['multi_rule'], 0)}）、1 レースの点数は {num(bt['train']['bets_per_race'])}。"
         f"2026 年の区間は、レースで束ねて {pc(bt['2026']['roi_lo'], 0)}〜{pc(bt['2026']['roi_hi'], 0)}、開催日で束ねて {pc(bt['2026']['roi_lo_day'], 0)}〜{pc(bt['2026']['roi_hi_day'], 0)} で、ほぼ同じ。",
         "－"],
    ]
    o.append(table(["仮説", "どの差か", "事象と根拠", "場所"], hyp))
    o.append("""<p class="sub">帰無の並べ替えの作り:</p>
<ul class="sub">
<li>単位 × 確定の人気（13 番人気以下はまとめる）の中で、切り口の水準の行を入れ替えた（<code>megu_why.py</code> の permute_levels）。人気と結果の関係は残る。</li>
<li>消えるのは、人気の順位の中のオッズの差（頭数など）と相関する分と、レース単位の切り口のレースの中での一貫性と、同じ馬の繰り返し。</li>
<li>そのため、この帰無は「人気の順位以上の情報が無い」の帰無で、「市場以上の情報が無い」ではない。層を単位 × オッズ帯にした版で確かめるのがよい（今回は流していない）。</li>
<li>1 行だけの層は入れ替わらない。水準の並び（方向の検査）は入れ替える前の表で決めた。消しの条件は B に使わないので探していない。</li>
</ul>""")

    o.append("<h3>訓練の回収率の帯ごとの、探した後（2025・2026 年）の成績</h3>")
    o.append(f'<p class="sub">表示の条件 {R["n"]} 個を、訓練の単勝の回収率で同じ数ずつ 5 つに分けた。訓練の値に予測の力があれば、上の帯ほど右の列が高くなる。条件ごとに数えたので、同じ馬を重ねて数える。条件の訓練の走数は中央値 {R["n_median"]:.0f}（10〜90% 点 {R["n_q"]["0.1"]:.0f}〜{R["n_q"]["0.9"]:.0f}）。</p>')
    o.append(table(["帯（訓練の回収率の低い順）", "条件", "訓練の走数", "訓練の単勝", "探した後の走数", "探した後の単勝", "探した後の市場比（条件ごと）"],
                   [[f"{r['q'] + 1}", r["rules"], f"{r['train_n']:,}", pc(r["train_roi_t"]), f"{r['post_n']:,}", pc(r["post_roi_t"]), num(r["post_edge"])]
                    for r in F["rules_by_train_roi"]], {1, 2, 3, 4, 5, 6}))

    o.append("<h3>較正: 買った馬（単勝）の市場の見込みと実際の勝率（オッズ帯ごと）</h3>")
    o.append('<p class="sub">「全馬」は、同じ期間・同じオッズ帯の全馬。</p>')
    cr = []
    for p, lab in (("train", "訓練（当てはめ）"), ("2025", "2025 年"), ("2026", "2026 年")):
        for r in F["calibration"][p]:
            cr.append([lab, r["band"], f"{r['n']:,}", pc(r["q"]), pc(r["win"]), pc(r["roi"]), pc(r["pop_win"]), pc(r["pop_roi"])])
    o.append(table(["期間", "オッズ帯", "点数", "市場の見込み", "実際の勝率", "単勝の回収率", "全馬の勝率", "全馬の回収率"], cr, {2, 3, 4, 5, 6, 7}))

    o.append("<h3>年ごと</h3>")
    o.append('<p class="sub">2016〜2024 年は条件を探した期間なので、高いのは当然。2025・2026 年は探した後。</p>')
    o.append(table(["年", "全馬の単勝", "全馬の複勝", "単勝の点数", "単勝の回収率", "勝率", "市場比", "複勝の点数", "複勝の回収率"],
                   [[y["year"], pc(y["pop_roi_t"]), pc(y["pop_roi_f"]), f"{y['bets_t']:,}", pc(y["roi_t"]), pc(y["win_t"]), num(y["edge_t"]), f"{y['bets_f']:,}", pc(y["roi_f"])] for y in F["by_year"]],
                   {1, 2, 3, 4, 5, 6, 7, 8}))

    o.append("<h3>並べ替え（帰無）と探し直しの一覧</h3>")
    o.append(f'<p class="sub">どれも訓練 2016〜2024 年の中だけで計算した（2025・2026 年は開かない）。seed 0 は入れ替えなし。凍結の条件の数（単 {real0["rules_t"]}・複 {real0["rules_f"]}）と訓練の値を再現した。</p>')
    pr_rows = []
    for p in sorted(perms, key=lambda x: (-x["fit_end"], x["seed"])):
        h = p.get("holdout")
        pr_rows.append([f"2016〜{p['fit_end']}", "本物" if p["seed"] == 0 else f"入れ替え {p['seed']}", p["rules_t"], num(p.get("n_rules_median"), 0),
                        pc(p["in_sample"]["tansho"]["roi"]), pc(p["in_sample"]["fukusho"]["roi"]), pc(p["in_sample"]["tansho"].get("win")),
                        pc(h["tansho"]["roi"]) if h else "－", pc(h["fukusho"]["roi"]) if h else "－"])
    o.append(table(["探した期間", "版", "単勝で買う条件", "走数の中央値", "探した期間の単勝", "探した期間の複勝", "勝率", "2022〜24 の単勝", "2022〜24 の複勝"], pr_rows, {2, 3, 4, 5, 6, 7, 8}))

    o.append("<h2>重賞（好走データ予想の主に見る版）</h2>")
    o.append(f'<p class="sub">版: {esc(G["primary"])}（<code>src/research/graded/prereg_buy.json</code>）。{G["versions"]} 版の中から、2016〜21 年で探し 2022〜24 年の 3 着内率で選んだ。買いの組の基準は平場と同じ（<code>buy.py:50</code>）。</p>')
    grow = [["探した期間の中（2016〜24 年で探し、2022〜24 年で数える）", f"{gi['bets']}／{gi['bet_races']}", pc(gi["win"]), f"{pc(gi['top3'])}（{pc(gi['same_pop_top3'])}）",
             pc(gi["roi_t"]), pc(gi["roi_f"]), num(gi["edge_ci"]["ratio"]), f"{pc(gi['same_pop_roi_t'])}・{pc(gi['same_pop_roi_f'])}"],
            ["開発の確かめ（2016〜21 年で探し、2022〜24 年に当てる）", f"{pr['n']}／{pr['races']}", pc(pr["win"]), pc(pr["top3"]), pc(pr["roi_t"]), pc(pr["roi_f"]), "－",
             f"同じレースの全馬 {pc(pr['all_roi_t'])}・{pc(pr['all_roi_f'])}"]]
    for k, lab in (("eval_2025_2026", "2025・2026 年"), ("eval_2025", "　2025 年"), ("eval_2026", "　2026 年")):
        g = G[k]
        grow.append([lab, f"{g['bets']}／{g['bet_races']}", pc(g["win"]), f"{pc(g['top3'])}（{pc(g['same_pop_top3'])}）", f"{pc(g['roi_t'])}（{pc(g['roi_t_lo'], 0)}〜{pc(g['roi_t_hi'], 0)}）",
                     f"{pc(g['roi_f'])}（{pc(g['roi_f_lo'], 0)}〜{pc(g['roi_f_hi'], 0)}）", f"{num(g['edge_ci']['ratio'])}（{num(g['edge_ci']['lo'])}〜{num(g['edge_ci']['hi'])}）",
                     f"{pc(g['same_pop_roi_t'])}・{pc(g['same_pop_roi_f'])}"])
    o.append(table(["期間", "買った頭数／レース", "勝率", "3 着内率（同じ人気の基準線）", "単勝（95% 区間）", "複勝（95% 区間）", "市場比（勝ち数）", "同じ人気の基準線（単・複）"], grow, {1, 2, 3, 4, 5, 6}))
    o.append(f'<p class="sub">同じ人気の基準線は、同じ期間の対象レースの買わなかった馬で作った。対象レースの全馬の単勝は {pc(ge["all_roi_t"])}（2025・26 年）、払戻率は {pc(1 / ge["overround"])}。開発の確かめの行は <code>prereg_buy.json</code> の記録の値で、基準線は同じレースの全馬。</p>')
    gh = [
        ["H2 選び過ぎ", "訓練 → 探した後（主因）",
         f"単勝は、探した期間の中 {pc(gi['roi_t'])} → 開発の確かめ {pc(pr['roi_t'])}（同じレースの全馬 {pc(pr['all_roi_t'])}）→ 2025・26 年 {pc(ge['roi_t'])}。"],
        ["H2 重賞の帰無", "参考",
         f"主に見る版の組は {pc_cells['real']:,} 個。重賞の帰無（{esc(G['null_kind'])}）では {min(pc_perm)}〜{max(pc_perm)} 個が通った（{len(pc_perm)} 回）。"
         "この帰無は人気の情報も消すので、本物が多く通るのは人気の情報で説明できる。人気以上の情報があることの根拠にはならない。"],
        ["H1 控除と市場", "探した後 → 100%",
         f"単勝の市場比は {num(ge['edge_ci']['ratio'])}（{num(ge['edge_ci']['lo'])}〜{num(ge['edge_ci']['hi'])}）。単勝は同じ人気の基準線（{pc(ge['same_pop_roi_t'])}）と同じ水準。"
         f"複勝は基準線より {pt(ge['roi_f'] - ge['same_pop_roi_f'])}、3 着内率は {pt(ge['top3'] - ge['same_pop_top3'])} 高いが、区間は基準線を含む。"],
        ["H7 件数の少なさ", "判定できない",
         f"2 年で {ge['bet_races']} レース・{ge['bets']} 頭。複勝の区間 {pc(ge['roi_f_lo'], 0)}〜{pc(ge['roi_f_hi'], 0)} は、基準線（{pc(ge['same_pop_roi_f'])}）も 100% も含む。"
         f"2026 年の複勝 {pc(g26['roi_f'])} は {g26['bets']} 頭の値。"],
        ["H7 見た回数", "訓練 → 探した後",
         f"2025・26 年は、台帳の graded の系統で {LK['graded']['all']} 回見られている（この診断の {LK['graded']['mine']} 回を含む）。"
         "主に見る版の凍結の記録の note には「2025・2026 年の結果を三度見た後」とある（<code>prereg_buy.json</code>）。"],
        ["H7 選び直さない決まり", "訓練 → 探した後",
         "重賞で選び直さない決まり（<code>docs/ideas/course_class_handicapper.md</code> §13.9）は 11-18〜19 にできた。重賞の評価の回（11-13〜15）はそれより前。2025・26 年の値は、まっさらな評価ではない。"],
    ]
    o.append(table(["仮説", "どの差か", "事象と根拠"], gh))

    o.append(oikiri_section())
    o.append("<h2>それでも 100% を超える余地があるとすれば（仮説。2025・26 年を見て選び直さない）</h2>")
    o.append(f"""<ul>
<li><b>探し方を変える</b>: 結果（3 着内率・回収率）で拾う探し方は、帰無でも同じ値を作る。そのため、この探し方はやめる。仮説で決めた少数の切り口を市場比で測り、縮小を掛けてから選ぶ。</li>
<li><b>縮小の強さ</b>: 2016〜21 年で探して 2022〜24 年に当てた値で決める。今回の材料では、④（{pc(ho_t['roi'])}）が基準線（{pc(ho_t['base_same_pop_roi'])}。買った馬を含む値）を下回る。訓練の前半・後半の相関も {num(F['rules_split']['corr_edge_ab'])}。どちらからも、ほぼ市場どおりまで寄せる強さが要る。</li>
<li><b>市場との差だけを使う</b>: 100% には、同じ人気の基準線（訓練 {pc(sp_train, 0)}）から約 {num(1 / sp_train)} 倍の勝ち数が要る。条件が人気以上の情報を持つことを、先に示す必要がある。市場が使わない情報が候補になる。追い切りの調教師ごとの出し方は第 2 段で確かめたが、確かな効きは無かった。</li>
<li><b>券種</b>: JRA の払戻率は単勝・複勝が 80% で、控除が最も小さい（データでも {pc(base['train']['payback'])}）。券種を変えて壁が下がる見込みは無い。</li>
<li><b>オッズの動き</b>: 発走直前のオッズの動きと回収率の関係が報告されている（Hanyu 2025、JRA 2004〜23 年、arXiv 2509.14645）。条件の外の話なので、前向きの記録（11/21〜）で確かめる。</li>
<li><b>新しい確かめの期間</b>: 選び直すなら、2016〜21 年で選び、2022〜24 年で確かめる。試した数で割り引いてから凍結し、凍結の後の前向きの記録（実際の T-60）だけで判定する。
2025・26 年は台帳で course の系統が {LK['course']['all']} 回、graded の系統が {LK['graded']['all']} 回見られているので、判定に使わない。</li>
</ul>""")

    o.append(f"""<h2>作り方・再現</h2>
<ul>
<li>平場は次の順に流す。
<ol>
<li><code>KEIBA_SESSION=megu-why python -m src.research.course.megu_why collect</code>。2025・26 年を読むので、台帳に eval_only を申告する（family course）。</li>
<li><code>python -m src.research.course.megu_why perm --seeds 0,1,… --fit-end 2024</code>（2016〜24 年だけ）と <code>--fit-end 2021</code>（2022〜24 年を確かめに使うので、台帳に confirm を申告する。今回は後から申告した）。流した seed は {'、'.join(f'2016〜{fe} 年で探す版 {len(v)} 回（{min(v)}〜{max(v)}）' for fe, v in (F.get('perm_seeds') or {}).items())}（0 は本物）。</li>
<li><code>python -m src.research.course.megu_why report</code>。</li>
</ol></li>
<li>collect の値は成績表 B と一致した（訓練 {bt['train']['bets']:,} 点・{pc(bt['train']['roi'])}、2025 年 {bt['2025']['bets']} 点・{pc(bt['2025']['roi'])}、2026 年 {bt['2026']['bets']} 点・{pc(bt['2026']['roi'])}）。</li>
<li>重賞: <code>KEIBA_SESSION=megu-why python -m src.research.course.megu_why graded</code>（台帳に eval_only、family graded）。基準線の作りを直すために、流し直した。</li>
<li>見た回数は、git の台帳を <code>src/ops/data_ledger.py</code> の tally で数えた（期間 {LOOK[0]}〜{LOOK[1]}）。この診断で course は {LK['course']['mine']} 回、graded は {LK['graded']['mine']} 回見た。</li>
<li>参考: 本命・大穴の偏りは、競馬の市場で広く見つかっている（Snowberg &amp; Wolfers 2010）。JRA の複勝・3 連単でも報告がある。</li>
<li>費用 0 円（手元の表だけ。GCS を読まない）。</li>
</ul>
</main>
</body>
</html>
""")
    return "\n".join(o)


def main() -> None:
    DST.write_text(build(), encoding="utf-8")
    print("書いた", DST)


if __name__ == "__main__":
    main()
