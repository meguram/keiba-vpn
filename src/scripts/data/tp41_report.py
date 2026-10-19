"""tp-41 の結果を docs の 1 章（similar_horses_pairs.html）にする。

python -m src.scripts.data.tp41_report   # ホストで実行（data/local/modeling/training_pipeline_v4/tp41/ の結果を読む）。出力は docs/html/modeling/chapters/
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from src.pipeline.features import tp41_similar as ts
from src.scripts.data import tp29_report as r29

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline_v4/tp41"
CHAPTER = ROOT / "docs/html/modeling/chapters/similar_horses_pairs.html"

AXIS_JP = {"d_str": "力の差（as-of の強さ）", "d_str4": "力の差（V4 予測の複勝 logit）", "d_sty": "位置取りの差（最初のコーナーの相対位置）", "d_gate": "馬番の差（頭数で割る）",
           "d_apt": "適性の距離（距離帯 × 芝割合）", "cos_ped": "血統のコサイン類似度（5 代の祖先 TF-IDF）", "same_trainer": "同じ調教師", "same_sire": "同じ父", "same_dsire": "同じ母父",
           "both_front": "ともに先行型"}
GROUP_JP = {"SIM_pos": "位置取り: 先行馬の数・先行の負荷・同じ位置取りの馬の数・その中での相対的な強さ", "SIM_apt": "適性: 同じ距離・面の馬の数、その中での相対的な強さ",
            "SIM_ped": "血統: 近い血統の馬の数・最大のコサイン・近い血統の中での相対的な強さ", "SIM_stable": "陣営・血縁: 同厩の他馬の数・同厩の中の順位・同じ父/母父の馬の数",
            "SIM_str": "力の近さ: 周りの密度・全体の中の位置・上下の隣との差", "SIM_prod": "位置取り × 適性 × 力の積カーネル: 似た馬の数と相対的な強さ", "ALL_PASSED": "第 1 段を通った群の同時追加"}


def _f(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _jl(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _bin_tbl(t: dict, top2: bool = True) -> str:
    rows = []
    d3, d2 = t["co_top3"], t.get("co_top2") or t["co_top3"]
    dfp = {r["bin"]: r for r in t["dfp"]}
    for a, b in zip(d3, d2):
        sig = "<b>" if (a["hi"] < 1 or a["lo"] > 1) else ""
        sig2 = "</b>" if sig else ""
        rows.append([html.escape(str(a["bin"])), f"{a['n']:,}", f"{sig}{_f(a['ratio'])} [{_f(a['lo'], 2)}, {_f(a['hi'], 2)}]{sig2}", f"{_f(b['ratio'])} [{_f(b['lo'], 2)}, {_f(b['hi'], 2)}]",
                     _f(dfp.get(a["bin"], {}).get("mean", dfp.get(a["bin"], {}).get("dfp")))])
    head = ["区分", "ペア数", "同時3着以内 観測/期待 [95%区間]", "同時2着以内（馬連）観測/期待 [95%区間]", "着順差の平均（0=同着, 1=最大）"]
    if not top2:
        rows = [[r[0], r[1], r[2], r[4]] for r in rows]
        head = [head[0], head[1], head[2], head[4]]
        return r29._tbl(head, rows, ["", "n", "n", "n"])
    return r29._tbl(head, rows, ["", "n", "n", "n", "n"])


def _glm_tbl(g: dict, tgt: str, title: str) -> str:
    d = g[f"全部|{tgt}"]
    rows = [[html.escape(AXIS_JP.get(k, k)), _f(v["beta"], 4, True), _f(v["se"], 4), _f(v["z"], 1, True)] for k, v in d.items() if k not in ("const", "_n")]
    return f"<h3>{title}</h3>" + r29._tbl(["軸（標準化）", "係数", "SE", "z"], rows, ["", "n", "n", "n"])


def _joint_tbl(j: dict, name: str) -> str:
    rows = []
    for tag, v in j[name].items():
        ci = v["ci"]
        sig = "<b>" if ci[1] < 0 else ""
        rows.append([html.escape(tag), _f(v["test2024_nll"][0], 5), _f(v["test2024_nll"][1], 5), f"{sig}{_f(v['d_test'], 5, True)}{'</b>' if sig else ''}", f"[{_f(ci[0], 5, True)}, {_f(ci[1], 5, True)}]"])
    return r29._tbl(["補正に入れる軸", f"2024 の対数損失（補正なし）", "（補正あり）", "差（負 = 改善）", "95% 区間（レース単位）"], rows, ["", "n", "n", "n", "n"])


def _bet_tbl(b: dict, name: str) -> str:
    rows = []
    for k in ("top1", "top3", "top5", "top10", "top20"):
        v = b[name].get(k)
        if v:
            rows.append([k.replace("top", "上位") + "点", f"{v['races']:,}", f"{_f(v['hit_base'], 4)} → {_f(v['hit_tilt'], 4)}", f"{_f(v['roi_base'] * 100, 1)}% [{_f(v['roi_base_ci'][0] * 100, 0)}, {_f(v['roi_base_ci'][1] * 100, 0)}]",
                         f"{_f(v['roi_tilt'] * 100, 1)}% [{_f(v['roi_tilt_ci'][0] * 100, 0)}, {_f(v['roi_tilt_ci'][1] * 100, 0)}]", f"{_f(v['roi_diff'] * 100, 1, True)}pt [{_f(v['roi_diff_ci'][0] * 100, 1, True)}, {_f(v['roi_diff_ci'][1] * 100, 1, True)}]"])
    return r29._tbl(["買い目の点数", "レース数", "的中率 補正なし → あり", "回収率 補正なし [95%区間]", "回収率 補正あり [95%区間]", "差（あり − なし）"], rows, ["", "n", "n", "n", "n", "n"])


def _calib_tbl(rows: list[dict], label: str) -> str:
    out = []
    byb: dict = {}
    for r in rows:
        byb.setdefault(r["bin"], {})[r.get("target", "win")] = r
    for b, v in byb.items():
        w, t3 = v.get("win"), v.get("top3")
        cell = lambda x: "-" if not x else (("<b>" if (x["hi"] < 1 or x["lo"] > 1) else "") + f"{_f(x['ratio'])} [{_f(x['lo'], 2)}, {_f(x['hi'], 2)}]" + ("</b>" if (x["hi"] < 1 or x["lo"] > 1) else ""))
        out.append([html.escape(str(b)), f"{(w or t3)['n']:,}", cell(w), cell(t3)])
    return f"<h3>{html.escape(label)}</h3>" + r29._tbl(["区分", "頭数", "勝率 実/予測 [95%区間]", "複勝率 実/予測 [95%区間]"], out, ["", "n", "n", "n"])


def _grp_tbl(res: dict) -> str:
    rows = []
    for g, s in (res.get("groups") or {}).items():
        c = s["vs_V4"]
        rows.append([g, html.escape(GROUP_JP.get(g, "")), s["cols"], _f(c["Δwin"], 5, True), _f(c["win_hi"], 5, True), _f(c["Δtop2"], 5, True), _f(c["Δtop3"], 5, True), "採用" if c["採用"] else "不採用"])
    return r29._tbl(["群", "内容", "列数", "Δ勝率", "95% 上限", "Δ連対", "Δ複勝", "採否"], rows, ["", "", "n", "n", "n", "n", "n", ""]) if rows else "<p>（未実施: ホストの重い評価の枠待ち。load1&lt;12 かつ実行中の重い評価が 2 本以下になると自動で走る）</p>"


def build() -> None:
    d = _jl(OUT / "describe.json")
    cb = _jl(OUT / "calib.json")
    jt = _jl(OUT / "joint.json")
    bt = _jl(OUT / "bet.json")
    s1, s2 = _jl(OUT / "stage1" / "results.json"), _jl(OUT / "stage2" / "results.json")
    style = re.search(r"<style>.*?</style>", r29.STYLE_SRC.read_text(encoding="utf-8"), re.S).group(0)
    adopted = [g for g, s in (s2.get("groups") or {}).items() if s["vs_V4"]["採用"]]
    b1 = (s1.get("base") or {}).get("levels", {})
    b2 = (s2.get("base") or {}).get("levels", {})
    g3 = jt.get("3連複", {}).get("d_str4抜きの全部", {})          # 類似度の軸だけ（V4 の強さの差は含まない）
    g2 = jt.get("馬連", {}).get("d_str4抜きの全部", {})
    g3a = jt.get("3連複", {}).get("全部", {})                   # + V4 の強さの差
    g2a = jt.get("馬連", {}).get("全部", {})
    glm = d.get("glm", {})
    if not (s1.get("groups") or {}):
        concl = "特徴量（SIM_* の 6 群）の V4 全体への増分評価は、ホストの重い評価の枠待ちでまだ走っていない（枠が空き次第、自動で第 1 段 → 第 2 段が走る）。それまで V4 には何も足さない。"
    elif adopted:
        concl = "第 2 段で採用基準を満たした群: " + "、".join(adopted)
    elif not (s2.get("groups") or {}):
        passed = [g for g, v in (s1.get("groups") or {}).items() if v["vs_V4"]["Δwin"] < -0.00012 and v["vs_V4"]["win_hi"] < 0.0001]
        concl = ("第 1 段を通った群: " + "、".join(passed) + "（第 2 段は未実施）") if passed else "第 1 段で基準（Δ勝率 < −0.00012 かつ上限 < +0.0001）を通った群はなかった。V4 には何も足さない。"
    else:
        concl = "第 2 段で採用基準を満たした群はなかった。V4 には何も足さない。"
    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>似た馬は着順が近いか（tp-41）— keiba-vpn</title>{style}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>似た馬は着順が近いか（tp-41）</h1>
<p class="muted">『似たような適性・位置取り・力の馬は、同じレースで着順が近くなる』という仮説を、同じレースの馬のペア（{d.get('n_pairs', 0):,} ペア、{d.get('n_races', 0):,} レース、2022〜2024 年の V4 の OOF）で検定し、(A) 個々の確率予測、(B) 組合せ券の確率、(C) 買い目の選び方に効くかを評価した。test 2025・2026 年・市場オッズは使っていない。</p>
<div class="note"><b>要点.</b>
 <ul>
 <li>仮説は<b>部分的に成り立つ</b>。ただし主役は『力の差』（V4 が見積もった強さの差）で、これは V4 の確率から導く同時確率にすでに入っている。それでもなお、<b>適性（距離×面）が近い・馬番が近い・位置取りが近い</b>ペアは、割引 Harville の同時確率より少し一緒に上位に来やすい（1 標準偏差あたり同時3着以内のオッズが約 3.5〜5% 変わる、z は −4.6〜−6.6）。<b>ともに先行型</b>のペアは逆に一緒に来にくい（展開の競合、−3.5%、z −6.0）。<b>血統の近さ・同じ調教師・同じ父</b>は効かない（z の絶対値が 1.5 以下）。</li>
 <li>この補正を<b>組合せ券の確率</b>に入れると、2024 年で 3連複の対数損失が {_f(g3.get('test2024_nll', [0, 0])[0], 4)} → {_f(g3.get('test2024_nll', [0, 0])[1], 4)}（差 {_f(g3.get('d_test'), 4, True)}、95% 区間 [{_f((g3.get('ci') or [0, 0])[0], 4, True)}, {_f((g3.get('ci') or [0, 0])[1], 4, True)}]）、馬連で {_f(g2.get('test2024_nll', [0, 0])[0], 4)} → {_f(g2.get('test2024_nll', [0, 0])[1], 4)}（差 {_f(g2.get('d_test'), 4, True)}、95% 区間 [{_f((g2.get('ci') or [0, 0])[0], 4, True)}, {_f((g2.get('ci') or [0, 0])[1], 4, True)}]）（ここまでは類似度の軸だけ）。V4 自身の強さの差（d_str4）も補正に入れると、3連複 {_f(g3a.get('d_test'), 4, True)}、馬連 {_f(g2a.get('d_test'), 4, True)} まで改善する（V4 + 割引 Harville が、力の離れた馬どうしの同時確率を過大に、力の近い馬どうしを過小に見積もっている分）。改善は有意だが小さい（1 レースあたり 1〜2% の確率の見直し）。</li>
 <li><b>買い目の選び方</b>（確率の上位 k 点）に補正を入れても、回収率は良くならなかった（3連複は補正ありのほうが低い傾向、馬連は差なし）。</li>
 <li><b>個々の確率予測</b>には、V4 の勝率・複勝率は同じレースの似た馬の状況別にほぼ較正されている（±2〜4%）。特徴量 {len(ts.GROUPS)} 群の増分: {concl}</li>
 </ul></div>
<h2>1. 方法</h2>
<ul>
 <li><b>ペア</b>: 同じレースの全ペア（i&lt;j）。類似度は『レース日より前の自分の走』と出走表の情報だけから作る（先読みなし）。位置取り = 最初のコーナーの相対位置の指数平滑（半減期 3 走）、力 = 着順位置の平滑とめぐ指数 v3a（予測用の下側圧縮列、low_confidence と障害戦は除外）の 3 走平均の標準化、適性 = 過去 5 走の距離の対数平均と過去 10 走の芝割合、血統 = 5 代の祖先の TF-IDF のコサイン、陣営 = 同じ調教師・同じ父・同じ母父。</li>
 <li><b>期待</b>: V4 の勝率・連対率・複勝率に整合する割引 Harville（<code>exotic_probabilities</code>）が出す、ペアの『ともに3着以内』（ワイド）と『ともに2着以内』（馬連）の確率。観測 ÷ 期待が 1 を超えれば、V4 + 割引 Harville が過小に見積もっていた同時の起きやすさ。全体では観測 {_f(d.get('o3_obs'), 4)} / 期待 {_f(d.get('o3_exp'), 4)}（ワイド）、観測 {_f(d.get('o2_obs'), 4)} / 期待 {_f(d.get('o2_exp'), 4)}（馬連）で、全体は合っている。</li>
 <li><b>検定</b>: ペアは同じレースの中で独立でないため、区間はレース単位ブートストラップ、回帰は『期待の logit をオフセット』にしたロジスティック回帰＋レース単位クラスタ頑健 SE。着順差はレース固定効果の線形回帰。</li>
</ul>
<h2>2. 軸ごとの『観測 / 期待』</h2>
<p class="muted">太字は 95% 区間が 1 を含まないもの。区分は 5 分位（二値の軸は 2 区分）。</p>
{"".join(f"<h3>{html.escape(t['name'])}</h3>" + _bin_tbl(t) for t in (d.get('bins') or {}).values())}
<h3>力が近いペアに限る（V4 予測の複勝 logit の差が下位 3 分の 1）</h3>
<p class="muted">力の差を揃えても、適性・位置取り・馬番が近いペアは一緒に来やすいかを見る。</p>
{"".join(f"<h4>{html.escape(AXIS_JP.get(ax, ax))}</h4>" + _bin_tbl({'co_top3': t['co_top3'], 'dfp': t['dfp']}, top2=False) for ax, t in (d.get('bins_strength_matched') or {}).items())}
<h2>3. 調整後の回帰（全軸を同時に入れる）</h2>
<p class="muted">係数は『1 標準偏差の変化で、期待に対するオッズ比の対数がどれだけ変わるか』。負 = 遠いほど一緒に来にくい（近いほど一緒に来やすい）。</p>
{_glm_tbl(glm, 'o3', '同時3着以内（ワイド）') if glm else ''}
{_glm_tbl(glm, 'o2', '同時2着以内（馬連）') if glm else ''}
<h3>着順差（レース固定効果）</h3>
{r29._tbl(['軸（標準化）', '係数', 'SE', 'z'], [[html.escape(AXIS_JP.get(k, k)), _f(v['beta'], 5, True), _f(v['se'], 5), _f(v['z'], 1, True)] for k, v in (glm.get('全部|dfp') or {}).items()], ['', 'n', 'n', 'n']) if glm else ''}
<h2>4. 組合せ券の確率への補正（2022–23 年で当てはめ、2024 年で評価）</h2>
<p class="muted">各レースの組合せの確率に exp(Σ_軸 γ × ペア類似度の和) を掛けて正規化（条件付きロジット）。γ は 2022–23 年の対数尤度で最大化（L2 小）。2024 年のレースの、実際に起きた組合せの平均対数損失（小さいほど良い）。『d_str4』は V4 自身の複勝 logit の差で、V4 の強さそのものの同時確率の歪みも含む。</p>
<h3>3連複</h3>{_joint_tbl(jt, '3連複') if jt else ''}
<h3>馬連</h3>{_joint_tbl(jt, '馬連') if jt else ''}
<h2>5. 個々の確率の較正（同じレースの似た馬の状況別）</h2>
<p class="muted">V4 の OOF（2022〜24 年）の実/予測。1 に近いほど較正されている。太字は 95% 区間が 1 を含まないもの。</p>
{"".join(_calib_tbl(cb[k], k) for k in ['先行馬: 他の先行馬の数', '差し・追込馬: 他の先行馬の数', 'sp_rel_s', 'sa_rel_s', 'ss_z', 'sp_prod_rel_s', '同厩の他馬の数'] if k in cb)}
{('<h3>先行馬グループの勝利数 / 予測勝率の合計（そのレースの先行馬の数別）</h3>' + r29._tbl(['先行馬の数', 'レース数', '勝利数 / 予測勝率の合計 [95%区間]'], [[html.escape(str(r['bin'])), f"{r['n']:,}", f"{_f(r['ratio'])} [{_f(r['lo'], 2)}, {_f(r['hi'], 2)}]"] for r in cb.get('先行馬グループ: 勝利数/予測勝率の合計（先行馬の数別）', []) if r['n']], ['', 'n', 'n'])) if cb else ''}
<p class="muted">sp_rel_s = 位置取りが似た馬の中での相対的な強さ、sa_rel_s = 適性が似た馬の中での相対的な強さ、ss_z = レース内の強さの標準化。</p>
<h2>6. 買い目の選び方（確定の払戻で精算、2024 年）</h2>
<p class="muted">確率の上位 k 点を買う。補正の γ は 2022–23 年で当てはめ。回収率は 1 点 100 円の確定の払戻（2024 年の払戻の連結は補正テーブルで補正済み、読めないレースは除外）。回収率の区間はレース単位ブートストラップ。</p>
<h3>3連複</h3>{_bet_tbl(bt, '3連複') if bt else ''}
<h3>馬連</h3>{_bet_tbl(bt, '馬連') if bt else ''}
<h2>7. 特徴量の増分（V4 全体に対して）</h2>
<p class="muted">群: """ + "、".join(f"{g} = {GROUP_JP[g]}" for g in ts.GROUPS) + f"""。同じレースの他馬の値はすべてその馬のレース日より前の値なので、先読みは無い（完全一致テストあり）。</p>
<h3>第 1 段（valid 2022+2023、seed 1 本）</h3>
<p class="muted">基準 {_f(b1.get('logloss_win'), 5)}</p>
{_grp_tbl(s1)}
<h3>第 2 段（valid 2024、新 seed 2 本）</h3>
<p class="muted">基準 {_f(b2.get('logloss_win'), 5)}</p>
{_grp_tbl(s2)}
<h2>8. 結論</h2>
<ul>
 <li>{concl}</li>
 <li>『似た馬は近い着順になる』のは、<b>力の差</b>を通じた効果が大半で、それは V4 の確率が持っている。上乗せで残るのは、適性・馬番・位置取りの近さと、先行同士の競合という小さな効果（組合せ券の確率で 1 レースあたり約 1%）。</li>
 <li>この補正は買い目の選び方（確率の上位 k 点）の回収率を上げなかった。組合せ券の確率の較正（期待値の計算）にだけ小さく効く。実際に使うなら『3連複・馬連の確率に γ の補正を掛ける』形で、推論側に載せるのは、2025 年の独立な確認（一度だけ）と実オッズでの期待値の確認の後にする。</li>
 <li>推論側の設計メモ: 必要な状態量は各馬の過去の位置取り（指数平滑）、距離・芝割合、力（着順の平滑とめぐ指数の平均）のみで、すでに V4 の履歴状態に近い（追加はごく小さい）。同じレースの出走表が確定した時点で、全ペアの類似度から組合せ券の確率を 1 レースあたり数十ミリ秒で補正できる。</li>
</ul>
</body></html>
"""
    CHAPTER.write_text(body, encoding="utf-8")
    print("保存:", CHAPTER)


if __name__ == "__main__":
    build()
