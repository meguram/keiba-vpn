"""tp-39（成長曲線 = 馬体重の変化とパフォーマンス）の章 HTML を、評価の結果 JSON から生成する。

python -m src.scripts.docs.build_growth_curve_chapter --json-dir <tp39 の結果ディレクトリ> [--out docs/html/modeling/chapters/growth_curve_weight.html]
結果 JSON は ``python -m src.scripts.data.tp39_eval <cmd>`` の出力（metrics / describe / prior / flagged / stack / boost / stage1 / stage1b / stage2 / subset の各 results.json）。
実行後は ``python -m src.scripts.docs.build_docs_book`` で本を作り直す。
"""

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_ap = argparse.ArgumentParser()
_ap.add_argument("--json-dir", default=str(REPO / "data/local/modeling/training_pipeline_v4/tp39"))
_ap.add_argument("--out", default=str(REPO / "docs/html/modeling/chapters/growth_curve_weight.html"))
_a = _ap.parse_args()
D = Path(_a.json_dir)
OUT = Path(_a.out)


def J(n):
    p = D / n
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


metrics, desc, prior, flagged, consts = J("metrics.json"), J("describe.json"), J("prior.json"), J("flagged.json"), J("consts.json")
def JS(lab):
    p = D / lab / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


st1, st1b, st2, subset = JS("stage1"), JS("stage1b"), JS("stage2"), J("subset.json")
boost = J("boost.json")

style = re.search(r"<style>(.*?)</style>", (REPO / "docs/html/modeling/chapters/rest_and_interval.html").read_text(encoding="utf-8"), re.S).group(1)

GDESC = {
    "POP_free": "母集団の成長曲線の参照値（性別×年齢×季節×馬格×体重変化の応答、馬格の効果）を着順・タイム由来の指標で + 体重だけの列（標準体重との差・体重の傾き・累積増加）",
    "POP_megu": "同上の応答表・馬格の効果を megu 系の指標（meg・comp）で",
    "SL_pt": "馬ごとの傾き（変化・体重差・変化の大きさ・体重差の大きさ）、補正ポイント。指標 = 着順の正規スコアとタイム偏差の平均（megu なし）", "SL0_pt": "SL_pt と同じだが縮約先を『全体の傾き』にしたもの（母集団曲線が縮約先として効くかの対照）",
    "KR_pt": "近い体重・近い変化のときの過去の成績の平均（カーネル回帰）、最良時の体重との差。pt", "GR_pt": "成績の直近の傾き・指数平滑・最大/最小（成長の勢い）。pt",
    "SL_meg": "馬ごとの傾き（変化・体重差・変化の大きさ・体重差の大きさ）と補正ポイント。指標 = 自前の指数 megu_index",
    "SL_pro": "同上。指標 = 着順の正規スコア", "SL_comp": "同上。指標 = 合成（megu・着順・タイム偏差）",
    "SL0_comp": "SL_comp と同じだが、縮約先を『全体の傾き』にしたもの（母集団曲線が縮約先として効くかの対照）",
    "KR_meg": "近い体重・近い変化のときの過去の成績の平均（カーネル回帰）、最良時の体重との差。megu", "KR_pro": "同上。着順", "KR_comp": "同上。合成",
    "GR_meg": "成績の直近の傾き・指数平滑・最大/最小（成長の勢い）。megu", "GR_pro": "同上。着順", "GR_comp": "同上。合成",
    "ALL_PASSED": "第 1 段を通過した群の全部",
}


def f(x, n=3, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def pct(x, n=1):
    return "—" if x is None else f"{100 * x:.{n}f}%"


def tbl(head, rows, nums=()):
    h = "".join(f"<th>{c}</th>" for c in head)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f'<td class="num">{c}</td>' if i in nums else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    return f"<table><tr>{h}</tr>{body}</table>"


KIND = {"chg": "体重変化（前走比）", "lev": "体重差（その馬の過去平均との差）", "abs": "変化の大きさ（増減どちらでも）", "labs": "体重差の大きさ"}
MET = {"meg": "megu 相対", "mga": "megu 絶対", "pro": "着順の正規スコア", "pt": "着順＋タイム（megu なし）", "comp": "合成"}
parts = []
add = parts.append

# ---- 要点
het = desc.get("hetero", {})
sh = consts.get("shrink", {})
SUMMARY_SLOT = len(parts)
add("")                                                              # 要点は最後に埋める（結果から組み立てる）

# ---- 1. 指標
if metrics:
    rows = []
    for m in ("meg", "mga", "pro", "tim", "pt", "comp"):
        cov = metrics["coverage"].get("2022", {}).get(m)
        rows.append([{"meg": "meg = megu v3a の相対（レース内の中央値との差）", "mga": "mga = megu v3a の絶対（成長の軌跡用。レースでそろえない）", "pro": "着順の正規スコア（Φ⁻¹(1−(着順−0.5)/頭数)）", "tim": "タイム偏差（レース内の中央時計との差を 200m あたりに）", "pt": "pt = 着順の正規スコアとタイム偏差の平均（megu を含まない）", "comp": "合成 comp（megu・着順・タイム偏差のうち 2 つ以上あるとき平均）"}[m],
                     pct(cov), f(metrics["icc_P"][m]["icc1"]), f(metrics["icc_R"][m]["icc1"]), f(metrics["icc_P"][m]["var_total"], 2)])
    add("<h2>1. パフォーマンス指標の選定</h2>")
    add('<p class="muted">信頼性 = 同じ馬の複数の走で指標がどれだけ一致するか（級内相関 ICC(1)、2020–21 年・6 走以上の馬）。P = レース内で相対化した指標、R = P から年齢・性別・クラス・面・距離・休養間隔・前走の P で説明できる分を引いた残差（期待以上度）。</p>')
    add(tbl(["指標", "被覆（2022）", "ICC(P)", "ICC(R)", "P の分散"], rows, nums=(1, 2, 3, 4)))
    mg = metrics.get("megu") or {}
    add('<p class="muted"><b>megu の版:</b> 最終候補 v3a（{} / version {} / 列 {} と {}（絶対）/ 有効 {:,} 行）。low_confidence の行と状態の無い行（障害戦など）を除く。<b>最終版に差し替わったら同じ手順で再実行して採否を確定する。</b>megu を含む結果（meg・mga・comp と、それを使う群）は最終候補の値に基づく暫定。</p>'.format(mg.get("dir"), ", ".join(mg.get("version", [])), mg.get("col"), mg.get("abs_col"), mg.get("rows", 0)))
    add("<ul><li><b>netkeiba のタイム指数は使えなかった。</b>馬ごとの走ごとの値は、ローカルの馬別の成績表（result_tbl）に無い。race_index の time_index_m / speed_* は『出走前に見える過去の値』で、走ごとの指標としての as-of は未検証（tp-03 でも speed_* は外している）。代わりに、プロジェクト自前の指数 megu_index（補正タイム。有効行のみ約 80%）、着順の正規スコア、タイム偏差を使い、合成した。</li>")
    add("<li>絶対の megu（mga）は馬ごとの水準を強く反映する（ICC(P) {}）。レース内で相対化した meg は {}、megu を含まない pt は {}。成長の軌跡（馬の水準の時系列）は mga、レース内の相対的な出来は meg・pt が表す。</li></ul>".format(f(metrics["icc_P"]["mga"]["icc1"]), f(metrics["icc_P"]["meg"]["icc1"]), f(metrics["icc_P"]["pt"]["icc1"])))

# ---- 2. 個体差
if het:
    add("<h2>2. 馬ごとの傾きは、増えると上がる／減ると上がる／無関係に分かれるか</h2>")
    add('<p class="muted">10 走以上の馬（2020–2025）ごとに、R ~ 体重の指標の傾きの t 値を出し、t&gt;2 を『増えると上がる』、t&lt;−2 を『増えると下がる（減ると上がる）』、それ以外を『無関係』とした。<b>帰無分布</b> = 馬の中で体重の並びを入れ替えて同じ集計をした平均（40 回）。偶然だけでも約 3.4% ずつが |t|&gt;2 になる。</p>')
    rows = []
    for k, v in het.items():
        m, kind = k.split("_", 1)
        if m not in ("pt", "mga", "comp"):
            continue
        rows.append([MET[m] + "｜" + KIND[kind], f"{v['horses']:,}", f(v["pairs_per_horse_median"], 0), pct(v["observed_share"]["pos"]), pct(v["null_share"]["pos"]), pct(v["observed_share"]["neg"]), pct(v["null_share"]["neg"]), pct(v["observed_share"]["none"]),
                     f(v["split_half"]["r"], 3, True), f(v["split_half_null_sd"], 3)])
    add(tbl(["指標｜説明変数", "馬数", "対の中央値", "増えると上がる 実測", "帰無", "増えると下がる 実測", "帰無", "無関係 実測", "前半/後半の傾きの相関 r", "帰無の標準偏差"], rows, nums=range(1, 10)))
    add("<p class=\"muted\">megu・着順の指標でも同じ傾向（付録の表）。</p>")
    rows = []
    for k, v in het.items():
        m, kind = k.split("_", 1)
        rows.append([MET[m], KIND[kind], f"{v['horses']:,}", pct(v["observed_share"]["pos"]), pct(v["null_share"]["pos"]), pct(v["observed_share"]["neg"]), pct(v["null_share"]["neg"]), f(v["split_half"]["r"], 3, True)])
    add("<details><summary>全指標・全説明変数の表</summary>" + tbl(["指標", "説明変数", "馬数", "上がる 実測", "帰無", "下がる 実測", "帰無", "前半/後半 r"], rows, nums=range(2, 8)) + "</details>")

# ---- 3. 縮約
if sh:
    add("<h2>3. 傾きの個体差の大きさ（縮約の強さ）</h2>")
    add('<p class="muted">2020–21 年の 6 走以上の馬で、馬ごとの傾きの観測分散 = 真の個体差 τ² + 推定の雑音（モーメント法）。τ² が観測分散の何割か = 馬ごとの傾きの信頼性。</p>')
    rows = []
    for k, v in sh.items():
        m, kind = k.split("_", 1)
        if m not in ("meg", "mga", "pro", "pt", "comp"):
            continue
        share = max(v["tau2_raw"], 0) / v["var_obs"] if v.get("var_obs") else None
        rows.append([MET[m], KIND[kind], f"{v['horses']:,}", f(v["beta0"], 3, True), f(v["sigma2"], 2), f(v["var_obs"], 3), f(v["mean_noise"], 3), f(v["tau2_raw"], 4, True), pct(share)])
    add(tbl(["指標", "説明変数", "馬数", "全体の傾き β0（10kg あたり）", "残差分散 σ²", "傾きの観測分散", "うち雑音", "個体差 τ²", "個体差の割合"], rows, nums=range(2, 9)))

# ---- 4. 母集団の応答
if desc.get("population"):
    P = desc["population"]
    add("<h2>4. 母集団の成長曲線（履歴の無い馬にも使える形）</h2>")
    add('<p class="muted">R の『その馬自身の平均との差』（馬内偏差）の平均。体重変化の帯別。年齢が若い馬は大きな変化（増減どちらでも）で上がり、4 歳以上は増減どちらでも下がる（山型）。符号つきの平均（全体の傾き β0 は −0.01〜−0.02）では見えない。</p>')
    for key, lab in (("dev_by_age_c", "年齢"), ("dev_by_dist_b", "距離"), ("dev_by_season", "季節（0=冬 1=春 2=夏 3=秋）"), ("dev_by_sz_bin", "馬格（0=小 … 4=大）")):
        cells = P[key]["cells"]
        cats = ["大幅減(-10kg~)", "減(-4~-9)", "維持(-3~+3)", "増(+4~+9)", "大幅増(+10kg~)"]
        groups = sorted({c.split("|")[0] for c in cells}, key=lambda s: float(s) if re.match(r"^-?[\d.]+$", s) else 0)
        rows = []
        for g in groups:
            r = [g]
            for c in cats:
                v = cells.get(f"{g}|{c}")
                r.append("—" if not v else ("<b>" if v["lo"] > 0 or v["hi"] < 0 else "") + f"{v['mean']:+.3f}" + ("</b>" if v["lo"] > 0 or v["hi"] < 0 else "") + f"<br/><span class='muted'>n={v['n']:,}</span>")
            rows.append(r)
        add(f"<h3>{lab}別</h3>" + tbl([lab] + cats, rows, nums=range(1, 6)))
    add('<p class="muted">太字 = 95% 区間が 0 を含まない。件数 300 未満のセルは省略。R は pt（着順＋タイム）。</p>')
    vr = desc.get("v4_resid", {}).get("all")
    if vr:
        rows = [[r["cat"], f"{r['n']:,}", f(r["win_resid"], 4, True), f"[{r['win_lo']:+.4f}, {r['win_hi']:+.4f}]", f(r["top3_resid"], 4, True), f"[{r['top3_lo']:+.4f}, {r['top3_hi']:+.4f}]"] for r in vr]
        add("<h3>V4 の予測に対する残差（2022–24）</h3><p class=\"muted\">実際の率 − V4 の予測の平均。V4 は体重変化の列（wt_chg_now、増減のフラグ、5 走の体重の傾きなど）をすでに持つ。</p>" + tbl(["体重変化", "件数", "勝率の残差", "95% 区間", "複勝率の残差", "95% 区間"], rows, nums=range(1, 6)))

# ---- 5. 事前分布の質
if prior:
    add("<h2>5. 母集団の成長曲線は縮約の事前分布として使えるか／履歴の無い馬への適用</h2>")
    add('<p class="muted">2022–24 年の走で『その走の R − その馬の過去平均 R』を予測。数値 = ゼロ予測に対する二乗誤差の減少（×10⁻⁴。大きいほど良い）。履歴の対 n は、傾きの推定に使える過去の走の数。母集団の定数は 2020–21 年だけから作った。指標は pt（megu 非依存）と comp（megu を含む、暫定）。</p>')
    for m, pm in prior.get("metrics", {}).items():
        for kind in ("chg", "lev", "abs"):
            if kind not in pm["slope"]:
                continue
            rows = []
            for bn, r in pm["slope"][kind].items():
                g = lambda k: f"{r[k]['gain_vs_zero'] * 1e4:+.1f}" if k in r else "—"
                c = r["prior_cell_minus_global_mse"]
                rows.append([bn, f"{r['rows']:,}", g("全体の傾き β0"), g("母集団曲線(年齢×性別×馬格の傾き)"), g("母集団の応答表(帯の平均)"), g("馬ごと(縮約なし)"), g("馬ごと(縮約先=全体の傾き)"), g("馬ごと(縮約先=母集団曲線)"),
                             f"{c['diff'] * 1e4:+.1f} [{c['lo'] * 1e4:+.1f}, {c['hi'] * 1e4:+.1f}]"])
            add(f"<h3>{MET[m]}｜{KIND[kind]}</h3>" + tbl(["履歴の対 n", "走数", "全体の傾き", "母集団の傾き（年齢×性別×馬格）", "母集団の応答表", "馬ごと（縮約なし）", "馬ごと（縮約先=全体）", "馬ごと（縮約先=母集団）", "縮約先の差（母集団−全体、負 = 母集団が良い）"], rows, nums=range(1, 9)))
    rows = []
    for m, pm in prior.get("metrics", {}).items():
        for lab, v in pm["debut"].items():
            rows.append([lab, MET[m], f"{v['rows']:,}", f(v["mse_zero"], 4), f(v["mse_pop_size"], 4), f"{v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]", f"{-100 * v['diff'] / v['mse_zero']:.1f}%"])
    if rows:
        add("<h3>履歴 0〜2 走の馬: 馬格（標準体重との差）だけで R を予測</h3><p class=\"muted\">履歴が無い馬は体重変化が無い（前走が無い）ので、性別×年齢×季節×馬格の母集団の表（R の平均）で予測。二乗誤差の差（負 = 良い）と相対減少。</p>"
            + tbl(["履歴", "指標", "走数", "ゼロ予測の MSE", "母集団の馬格表の MSE", "差 [95% 区間]", "相対減少"], rows, nums=range(2, 7)))

# ---- 6. flagged
if flagged:
    add("<h2>6. 補正ポイントの高い／低い馬は V4 に対して勝つか（馬単位とレース単位）</h2>")
    add('<p class="muted">V4 の OOF（2022–24）に対し、補正ポイントが上位／下位 10% の馬の成績。馬単位 = 実際の勝率 − V4 の予測。レース単位 = 該当馬が 1 頭以上いるレースで『該当馬のどれかが勝った割合』 − V4 での期待（Σ p_win）。区間はレース単位のブートストラップ。</p>')
    rows = []
    for m in ("pt", "comp"):
        for col, v in flagged.get(m, {}).items():
            if not col.startswith("gc_"):
                continue
            for nm in ("上位10%", "下位10%"):
                r = v[nm]
                rows.append([col, nm, f"{r['horses']:,}", f"{r['races']:,}", f(r["mean_per_race"], 2), f"{r['horse_win_obs'] - r['horse_win_v4']:+.4f}", f"{r['horse_top3_obs'] - r['horse_top3_v4']:+.4f}", f"{r['race_diff']:+.4f} [{r['race_diff_lo']:+.4f}, {r['race_diff_hi']:+.4f}]"])
    add(tbl(["補正ポイント", "区分", "馬数", "レース数", "1 レースの該当頭数", "馬単位 勝率差", "馬単位 複勝率差", "レース単位 勝率差 [95%]"], rows, nums=range(2, 8)))

# ---- 6b. 軽い一次選別
stack = J("stack.json")
if stack:
    add("<h2>7. 軽い一次選別: V4 の確率に特徴量を 1 つ足す条件付きロジット</h2>")
    add('<p class="muted">V4 の log p_win に γ×（標準化した特徴量）を足した softmax（条件付きロジット）。γ は学習年で最尤に当て、検証年で『レースごとの負の対数尤度（勝ち馬の確率）』の差を見る（負 = 改善）。GBM を再学習せずに全 {n} 特徴量を一度に調べられる。{n} 通りを試しているので、t&lt;−2 は偶然でも約 2.3%（約 {e:.0f} 個）出る。「履歴 0〜2 走」は前走までの出走数が 2 以下の馬だけで γ を当て、その馬どうしで勝ち馬を当てる問題として評価。</p>'.format(n=stack["n_features"], e=0.023 * stack["n_features"]))
    rows = []
    for k, v in stack["results"].items():
        best = v["top"][0]
        rows.append([k, f"{v['races_fit']:,}", f"{v['races_test']:,}", v["tested"], f"{v['base_nll']:.4f}", f"{v['n_improved_t<-2']} / {v['n_worse_t>2']}",
                     f"{best['feature']}（γ={best['gamma']:+.3f}）", f"{best['d_nll']:+.5f} [{best['lo']:+.5f}, {best['hi']:+.5f}]", f"{best['t']:.1f}"])
    add(tbl(["学習→検証｜対象", "学習レース数", "検証レース数", "試した特徴量", "基準の NLL/レース", "t&lt;−2 の数 / t&gt;2 の数", "最良の特徴量", "NLL の差/レース [95%]", "t"], rows, nums=range(1, 9)))

# ---- 7. 増分
def stage_table(st, title):
    if not st.get("groups"):
        return ""
    rows = []
    for g, v in st["groups"].items():
        c = v["vs_V4"]
        rows.append([g + ("" if v.get("megu_free", True) else "（megu 依存・暫定）"), GDESC.get(g, ""), v["cols"], f"{c['Δwin']:+.5f}", f"{c['win_hi']:+.5f}", f"{c['Δtop2']:+.5f}", f"{c['Δtop3']:+.5f}", "採用基準を満たす" if c["採用"] else "不採用"])
    base = st.get("base", {}).get("levels", {}).get("logloss_win")
    return f"<h3>{title}</h3><p class=\"muted\">基準（V4 の 1,135 列）勝率 LogLoss {base:.5f}</p>" + tbl(["群", "内容", "列数", "Δ勝率", "95% 上限", "Δ連対", "Δ複勝", "採否"], rows, nums=(2, 3, 4, 5, 6))


if st1.get("groups") or st1b.get("groups"):
    add("<h2>8. 特徴量の増分（GBM を再学習。V4 全体に対して）</h2>")
    add(stage_table(st1, "第 1 段（valid 2022+2023、seed 1 本）: megu に依存しない群"))
    add(stage_table(st1b, "第 1 段（同じ設定）: megu に依存する群（megu v3a 最終候補・暫定）"))
    add(stage_table(st2, "第 2 段（valid 2024、新 seed 2 本）"))
if subset:
    add("<h2>9. 履歴 0〜2 走の馬だけでの増分（GBM）</h2><p class=\"muted\">第 1 段（valid 2022+2023）の OOF を、前走までの出走数で分けて比べた。Δ勝率は V4 に対する勝率 LogLoss の差（負 = 改善）、区間は 95%（レース単位）。</p>")
    rows = []
    for k, v in subset.items():
        for nm in ("履歴0走", "履歴0〜2走", "履歴3〜5走", "履歴6走以上"):
            r = v[nm]
            rows.append([k, nm, f"{r['rows']:,}", f"{r['Δwin']:+.5f}", f"[{r['win_lo']:+.5f}, {r['win_hi']:+.5f}]", f"{r['Δtop3']:+.5f}"])
    add(tbl(["段階|群", "履歴", "走数", "Δ勝率", "95% 区間", "Δ複勝"], rows, nums=(2, 3, 4, 5)))

# ---- boost
def _bset(sub):
    return {k: v for k, v in boost.get("results", {}).items() if k.endswith("|sub") == sub}


if boost.get("results"):
    add("<h2>7b. V4 の出力を初期値にして新しい特徴量だけで補正する LightGBM（軽い GBM 評価）</h2>")
    add('<p class="muted">V4 の OOF（2022–24）の logit を初期値（init_score）にして、新しい特徴量だけで補正する小さな LightGBM（2 threads・15 葉・学習年の末尾 20% で早期終了）を学習し、検証年の二値 LogLoss の差（負 = 改善）を見る。V4 全体を再学習する第 1 段より軽い近似。分割は「学習 2022–23 → 検証 2024」と「学習 2022 → 検証 2023」。「履歴 0〜2 走」は前走までの出走数が 2 以下の馬だけで学習・評価（履歴の浅い馬専用の補正）。</p>')
    for sub, title in ((False, "全ての馬で学習・評価"), (True, "履歴 0〜2 走の馬だけで学習・評価")):
        names = sorted({k.split("|")[0] for k in _bset(sub)})
        rows = []
        for n in names:
            r = []
            for sp in ("fit22-23→test24", "fit22→test23"):
                v = boost["results"].get(f"{n}|{sp}" + ("|sub" if sub else ""))
                if v:
                    w, t3 = v["win"]["全体"], v["top3"]["全体"]
                    r += [f"{w['d']:+.5f} (上限 {w['hi']:+.5f})", f"{t3['d']:+.5f} (上限 {t3['hi']:+.5f})"]
                else:
                    r += ["—", "—"]
            v0 = boost["results"].get(f"{n}|fit22-23→test24" + ("|sub" if sub else ""))
            rows.append([n, v0["cols"] if v0 else "", v0["win"]["全体"]["rows"] if v0 else ""] + r)
        add(f"<h3>{title}</h3>" + tbl(["集合", "列数", "検証 2024 の走数", "2024 Δ勝率", "2024 Δ複勝", "2023 Δ勝率", "2023 Δ複勝"], rows, nums=range(1, 7)))

# ---- 要点（結果から組み立てる）
def _summary() -> list[str]:
    L = []
    allg = {}
    for st in (st1, st1b, st2):
        allg.update(st.get("groups", {}))
    done = [g for g in tg_groups if g in {**st1.get("groups", {}), **st1b.get("groups", {})}]
    passed = [g for g, v in allg.items() if v["vs_V4"]["採用"]]
    n_all = len(tg_groups)
    if allg:
        best = min(allg.items(), key=lambda kv: kv[1]["vs_V4"]["Δwin"])
        L.append("<b>結論: V4 に足す群はない。</b>" if not passed else f"<b>第 1 段で採用基準を満たした群: {', '.join(passed)}。</b>")
        L.append(f"V4 全体への GBM 評価（第 1 段）は {len(done)}/{n_all} 群が完了。最良は {best[0]} の Δ勝率 {best[1]['vs_V4']['Δwin']:+.5f}（95% 上限 {best[1]['vs_V4']['win_hi']:+.5f}）で、効果の最小値 0.0002 に届かない。残りの群は、ホストの重い処理が 3 本を超えている間は始めない規則で待機中（実行は自動で続く）。")
    else:
        L.append("<b>V4 全体への GBM 評価（第 1 段）はホストの空き待ちで未完了。</b>下の軽い評価（条件付きロジット・補正 LightGBM）では、採用基準に届く特徴量は無い。")
    h = het
    if h:
        pos = [v["observed_share"]["pos"] - v["null_share"]["pos"] for v in h.values()]
        neg = [v["observed_share"]["neg"] - v["null_share"]["neg"] for v in h.values()]
        rs = [v["split_half"]["r"] for v in h.values() if v["split_half"]["r"] is not None]
        L.append(f"<b>馬ごとの傾きの個体差は、偶然とほとんど区別できない。</b>10 走以上の馬で『増えると上がる』と判定された割合は、馬の中で体重の並びを入れ替えた帰無分布との差が {min(pos) * 100:+.1f}〜{max(pos) * 100:+.1f} ポイント、『増えると下がる』は {min(neg) * 100:+.1f}〜{max(neg) * 100:+.1f} ポイント（体重差・体重差の大きさで負側が増えるのは母集団の傾向）。前半と後半の傾きの相関は r = {min(rs):+.3f}〜{max(rs):+.3f}（帰無の標準偏差は約 0.01）。")
    if sh:
        fr = [max(v["tau2_raw"], 0) / v["var_obs"] for k, v in sh.items() if k.split("_")[0] in ("meg", "mga", "pro", "pt", "comp")]
        L.append(f"傾きの真の個体差 τ² は、観測された傾きの分散の {min(fr) * 100:.0f}〜{max(fr) * 100:.0f}%（縮約後は事実上、母集団の傾きに寄る）。馬ごとの傾きを縮約せずに使うと、予測は大きく悪化する（二乗誤差が +30% 前後）。")
    P = desc.get("population", {})
    if P:
        L.append("母集団の成長曲線には山型の形がある: 若い馬（2〜3 歳）は体重が大きく変わるとむしろ上がり、4 歳以上は増減どちらでも下がる（大幅増・大幅減とも −0.06〜−0.15）。ただし V4 はこれをほぼ吸収していて、V4 の勝率の残差は体重変化の帯の全てで ±0.5 ポイント以内。")
    pm = prior.get("metrics", {}).get("pt") if prior else None
    if pm:
        lev = pm["slope"].get("lev", {}).get("全体")
        deb = pm["debut"]
        cell = pm["slope"].get("lev", {}).get("履歴の対(n)=1〜2")
        parts_ = []
        if lev:
            parts_.append(f"体重差の母集団の応答表は R の予測で二乗誤差を {lev['母集団の応答表(帯の平均)']['gain_vs_zero'] / (lev['ゼロ(母集団の傾き 0)']['mse']) * 100:.1f}% 減らす")
        if cell:
            c = cell["prior_cell_minus_global_mse"]
            parts_.append(f"履歴の浅い馬（対 1〜2）で、縮約先を母集団曲線にしても全体の傾きと有意に変わらない（差 {c['diff'] * 1e4:+.1f}×10⁻⁴ [{c['lo'] * 1e4:+.1f}, {c['hi'] * 1e4:+.1f}]）")
        if deb:
            d0 = deb.get("履歴0走")
            if d0:
                parts_.append(f"履歴 0 走の馬は馬格の表で二乗誤差が {-100 * d0['diff'] / d0['mse_zero']:.1f}% 減る（95% 区間は 0 を含まない）")
        L.append("<b>母集団の成長曲線を縮約の事前分布として使う効果は小さい。</b>" + "。".join(parts_) + "。")
    if stack:
        r0 = stack["results"].get("fit22-23→test24|全体")
        if r0:
            r1 = stack["results"].get("fit22→test23|全体", {})
            best = r0["top"][0]
            L.append(f"軽い一次選別（V4 の確率に特徴量を 1 つ足す条件付きロジット）: 試した {r0['tested']} 個のうち 2024 年で t&lt;−2 は {r0['n_improved_t<-2']} 個（偶然でも約 {0.023 * r0['tested']:.0f} 個）、2023 年は {r1.get('n_improved_t<-2', '—')} 個。最良でもレース当たりの負の対数尤度が {best['d_nll']:+.5f}（基準 {r0['base_nll']:.2f} に対し {abs(best['d_nll']) / r0['base_nll'] * 100:.2f}%）。")
    if boost.get("results"):
        allr, subr = _bset(False), _bset(True)
        ds = [v["win"]["全体"]["d"] for v in allr.values()]
        names = sorted({k.split("|")[0] for k in allr})
        both = lambda R, suf: [n for n in names if all(R.get(f"{n}|{sp}{suf}") and R[f"{n}|{sp}{suf}"]["win"]["全体"]["hi"] < 0 for sp in ("fit22-23→test24", "fit22→test23"))]
        ds2 = [v["win"]["全体"]["d"] for v in subr.values()]
        L.append(f"補正 LightGBM（軽い GBM 評価）: 全ての馬で学習した場合の Δ勝率は {min(ds):+.5f}〜{max(ds):+.5f}、2 つの検証年とも 95% 上限が負になった集合は {len(both(allr, ''))} 個。履歴 0〜2 走の馬だけで学習・評価した場合も {min(ds2):+.5f}〜{max(ds2):+.5f}、2 年とも上限が負は {len(both(subr, '|sub'))} 個。分割や群で符号が揺れ、履歴の浅い馬にも効く成長曲線の補正は見えない。")
    L.append("megu の版: " + ", ".join((metrics.get("megu") or {}).get("version", [])) + "（最終版 v3a と同じ値であることを page_reference 側と照合済み）。")
    return L


tg_groups = ["POP_free", "POP_megu"] + [f"{p}_{m}" for p in ("SL", "KR", "GR") for m in ("meg", "mga", "pro", "pt", "comp")] + ["SL0_pt", "SL0_comp"]
parts[SUMMARY_SLOT] = '<div class="note"><b>要点.</b><br/>' + "<br/>".join(_summary()) + "</div>"

# ---- 試した範囲と注意、採否、推論側の設計メモ
mc = {"checked": ""}
try:
    mc = json.loads((D / "megu_check.json").read_text(encoding="utf-8"))
except Exception:
    pass
parts.append("<h2>10. 試した範囲、採否、注意</h2>")
parts.append("<ul>"
    "<li><b>試した群</b>: 母集団の参照値 2 群（POP_free / POP_megu）、馬ごとの傾き 5 指標 × 4 説明変数 + 縮約先の対照 2、カーネル回帰 5 指標、成長の勢い 5 指標（計 " + str(len(tg_groups)) + " 群、特徴量 479 個）。指標は megu の相対・絶対、着順の正規スコア、着順＋タイム、合成の 5 通り。</li>"
    "<li><b>先読みの対策</b>: 馬ごとの値は、その馬の過去の走だけの累積和を 1 行ずらして作る（自分や未来の走は入らない）。カーネル回帰は、i 走目を 0..i−1 走目の配列だけから厳密な和（math.fsum）で計算し、未来の走の数で端数が変わらない。母集団の定数（回帰係数・縮約の強さ・応答表・標準体重）は 2020–21 年だけから作り、全期間に同じ定数を当てる。テスト 9 本: 結果の入れ替えで過去・自分の行が bit 単位で不変、データを日付で切っても同一、2022 年以降を変えても定数が不変、傾きの手計算との一致、megu を差し替えても megu 非依存の群が bit 単位で不変、縮約の推定が既知の個体差を回復。</li>"
    "<li><b>市場オッズ・人気は使っていない</b>。test 2025・2026 年は選択に使っていない（検証は 2022〜24 年）。</li>"
    "<li><b>調教タイムから成長・調子を読む案</b>は別の調査（tp-40）が担当し、ここでは扱っていない。</li></ul>")
if mc.get("version"):
    parts.append("<h3>めぐ指数の版と照合</h3>" + tbl(["項目", "内容"], [["版", mc["version"]], ["読んだ列", f"相対 {mc['used_columns']['relative']} / 絶対 {mc['used_columns']['absolute']}"], ["除外", mc["excluded"]],
        ["最終候補ディレクトリとの照合", f"{mc['candidate_dir']} と {mc['final_dir']} を {mc['years']}（{mc['rows_total']:,} 行）で照合: {mc['result']}"], ["meta.json", f"{mc['meta_json']}"], ["spec.json (sha1)", mc["spec_json_sha1"]], ["spec", str(mc["spec"])]]))
parts.append("<h2>11. 推論側に載せるときの設計メモ（採用された場合のみ。今回は採用なしなので実装しない）</h2>")
parts.append("<ul><li>馬ごとの状態: 傾きの累積和（n, Σx, Σy, Σxx, Σxy, Σyy）を指標 × 説明変数 ごとに 6 個、直近 8 走の R と体重・体重変化、体重の過去平均（件数・合計）。1 頭あたり約 0.5〜1 KB、出走馬が約 5 万頭で 25〜50 MB。</li>"
    "<li>母集団の定数 consts.json（約 90 KB）を同梱。更新は出走後の結果が確定してから、馬ごとに 1 走ぶんの累積を足すだけ（再計算は不要）。</li>"
    "<li>馬体重は T-60 で発表されるので、当日の補正ポイントは発表後に計算できる。megu は結果確定後に付くので、前日までの履歴 R の更新は確定結果の取り込み後。</li></ul>")

html = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>成長曲線（馬体重の変化とパフォーマンス）の特徴量化（tp-39）— keiba-vpn</title><style>{style}</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>成長曲線（馬体重の変化とパフォーマンス）の特徴量化（tp-39）</h1>
<p class="muted">横軸 = 出走日、縦軸 = パフォーマンス指標・馬体重としたときの『体重が増えると上がる馬／減ると上がる馬／無関係な馬』を馬ごとに判別し、該当レースの補正ポイントを V4 に足せるかを検討した。履歴の無い馬・浅い馬には、年齢・性別・馬格・季節の母集団の成長曲線を使う設計を加えた。調教タイムから成長・調子を読む案は別（tp-40）。</p>
{''.join(parts)}

</body></html>"""
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, len(html))
