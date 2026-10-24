"""tp-76: results.json から章 feature_screening.html を作り、決定ログ §71 を追記する。

python -m src.scripts.data.tp76_report <results.json のあるディレクトリ>
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CH = ROOT / "docs/html/modeling/chapters"
E4 = 1e4

COL_DESC = {
    "A_fit_front": "日内の前残り度 × その馬の普段の前向きさ", "A_fit_inner": "日内の内有利度 × 今回の枠の内さ",
    "B_m": "潜在能力の予測（年齢×性別の曲線＋種牡馬の事前分布のカルマン）", "B_sd": "同・不確実性", "B_m_megu": "同・めぐ指数に適用",
    "C_elo": "対戦関係の強さ（Elo 型）", "C_rel": "同・出走馬の平均との差", "C_unc": "同・不確実性（出走数が少ないほど大）",
    "D_dist": "距離帯別の適性（馬内の予測誤差、種牡馬の事前へ縮約）", "D_venue": "競馬場×面別の適性", "D_going": "馬場状態別の適性", "D_cush": "クッション値帯別の適性",
    "D_all": "4 軸の適性の合計", "D_raw_all": "同・種牡馬の事前なし", "D_ag_all": "上がり順位の条件別適性の合計",
    "E_abs": "上がりのコース基準との差の平滑", "E_day": "同・同日の馬場水準で補正", "E_sustain": "前にいたときの上がり", "E_accel": "後ろにいたときの上がり",
    "E_unlucky_prev": "前走の不利度（着順−上がり順位）×後方度", "E_unlucky_m3": "同・直近 3 走平均", "E_std5": "上がり順位の安定性（直近 5 走の標準偏差）",
    "F_n30": "直近 30 日の出走数", "F_n90": "直近 90 日の出走数", "F_since": "前走からの日数（対数）", "F_dist90": "直近 90 日の走行距離", "F_burden90": "直近 90 日の負担（距離×着差）",
    "F_tataki": "休み明けからの叩き回数", "G_n_front": "出走馬中の先行馬の数", "G_front_rank": "普段の位置取りの前からの順位", "G_inner_front": "内側の先行馬の数", "G_fit": "ペース圧力×自馬の前向きさ",
}


def fmt(x, d=2, sign=True):
    return f"{x * E4:+.{d}f}" if sign else f"{x * E4:.{d}f}"


def ci(lo, hi, d=2):
    return f"<span class='muted'>[{lo * E4:+.{d}f}, {hi * E4:+.{d}f}]</span>"


def table(head, rows, cls=None):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = ""
    for r in rows:
        b += "<tr>" + "".join(f"<td{' class=num' if (cls and cls[i]=='n') else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    return f"<table><tr>{h}</tr>{b}</table>"


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def figures(r: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {}
    cur = r["bundle_curve"]
    ks = [c["k"] for c in cur]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(ks, [c["d3"] * E4 for c in cur], "o-", color="#1d4ed8", label="real bundle, ridge 200 (LOYO)")
    for lam, col in (("2000.0", "#0891b2"), ("20000.0", "#059669")):
        cc = r["bundle_lambda_sensitivity"][lam]["curve"]
        ax.plot([c["k"] for c in cc], [c["d3"] * E4 for c in cc], "s--", ms=3, color=col, label=f"real bundle, ridge {int(float(lam))} (post hoc)")
    sh = np.array(r["bundle_shadow_curves"]) * E4
    ax.fill_between(ks, sh.min(0), sh.max(0), color="#f97316", alpha=0.25, label="shadow bundles (range of 3)")
    ax.plot(ks, sh.mean(0), color="#ea580c", lw=1.2, label="shadow mean")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("number of columns added (ordered by sign consistency, then low redundancy)")
    ax.set_ylabel("dLogLoss x1e-4 per horse (negative = better)")
    ax.legend(fontsize=7)
    out["curve"] = fig_b64(fig); plt.close(fig)

    us = r["units"]
    names = list(us)
    d = np.array([us[k]["d3"] for k in names]) * E4
    lo = np.array([us[k]["ci95"][0] for k in names]) * E4
    hi = np.array([us[k]["ci95"][1] for k in names]) * E4
    sh_d = np.array([v["d3"] for v in r["shadow_units"].values()]) * E4
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x = np.arange(len(names))
    ax.axhspan(np.percentile(sh_d, 5), np.percentile(sh_d, 95), color="#f97316", alpha=0.22, label="shadow 5-95% (81 fake units)")
    ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt="o", color="#1d4ed8", capsize=2, label="real units (LOYO, 95% CI)")
    d25 = np.array([us[k]["d3_2025"] for k in names]) * E4
    ax.plot(x, d25, "d", color="#dc2626", ms=5, label="test 2025 (once)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7, rotation=60)
    ax.set_ylabel("dLogLoss x1e-4 per horse")
    ax.legend(fontsize=7)
    out["units"] = fig_b64(fig); plt.close(fig)
    return out


def priority(u: dict) -> str:
    p = u["pass"]
    if p["S5a"] and u["signs_neg"] == 3 and u["d3_2025"] <= 0:
        return "高"
    if p["S4"] or p["S5b"] or u["signs_neg"] == 3:
        return "中"
    return "低"


def combined(pri: str, cost: str, u: dict) -> str:
    if pri == "高" or (pri == "中" and cost == "低" and u["d3_2025"] <= 0):
        return "最優先"
    if pri == "中":
        return "優先"
    return "後回し"


def build(rdir: Path) -> None:
    r = json.loads((rdir / "results.json").read_text(encoding="utf-8"))
    lk = json.loads((rdir / "leak_check.json").read_text(encoding="utf-8"))
    figs = figures(r)
    us = r["units"]
    crit = r["criteria"]
    cn = r["composite_nested"]
    cs = r["composite_sensitivity"]
    be = r["bundle_eval_2025_at_kbest"]
    bs = r["bundle_lambda_sensitivity"]
    rb = r["rules_bundle"]
    rl = rb["lambda_sensitivity"]
    sh_med = r["shadow_d3_q"]["median"]
    best_u = min(us, key=lambda k: us[k]["d3"])
    s4_n3 = [k for k, u in us.items() if u["pass"]["S4"] and u["signs_neg"] == 3]
    s4_n3_ok = [k for k in s4_n3 if us[k]["d3_2025"] <= 0]

    rows_u = []
    for k, u in us.items():
        pri = priority(u)
        passed = "・".join(c for c, v in u["pass"].items() if v) or "-"
        rows_u.append([k, u["family"].split()[0], "・".join(u["cols"]), f"{u['missing_rate']*100:.0f}%", f"{u['corr_base_logit']:.2f}",
                       f"{fmt(u['d3'])} {ci(*u['ci95'])}", f"{fmt(u['d3_2022'],1)} / {fmt(u['d3_2023'],1)} / {fmt(u['d3_2024'],1)}", fmt(u["d3_2025"]),
                       f"{fmt(u['d_win'],1)} / {fmt(u['d_top2'],1)} / {fmt(u['d_top3'],1)}", passed, pri, u["infer_cost"], combined(pri, u["infer_cost"], u)])
    t_units = table(["単位", "族", "列", "欠損", "基準 logit との相関", "Δ3（LOYO）", "年別 22/23/24", "2025", "勝/2着/3着 Δ", "通る基準", "本試験の優先度", "推論コスト", "費用便益"],
                    rows_u, ["", "", "", "n", "n", "n", "n", "n", "n", "", "", "", ""])

    rows_c = []
    for k in ("S1", "S2", "S3", "S4", "S5a", "S5b"):
        t = crit[k]
        rows_c.append([{"S1": "S1 区間上限&lt;0 かつ ≤−2×10⁻⁴", "S2": "S2 3 年同符号 かつ ≤−2×10⁻⁴", "S3": "S3 3 年同符号 かつ ≤−1×10⁻⁴",
                        "S4": "S4 経験ベイズ P(効果&lt;0)≥0.8 または 局所 FDR≤0.2", "S5a": "S5a 片側 90% 上限&lt;0", "S5b": "S5b 片側 80% 上限&lt;0"}[k],
                       f"{t['real_pass']} / {t['n_real']}", f"{t['shadow_pass']} / {t['n_shadow']}", f"{t['fpr']*100:.1f}%"])
    t_crit = table(["基準", "本物が通る数", "シャドウが通る数", "偽陽性率"], rows_c, ["", "n", "n", "n"])

    cur = r["bundle_curve"]
    kb = r["bundle_k_best"]
    sh = np.array(r["bundle_shadow_curves"])
    rows_b = [[c["k"], c["col"], fmt(c["d3"]), fmt(sh[:, i].mean()), f"{fmt(sh[:, i].min())} 〜 {fmt(sh[:, i].max())}", fmt(c["d3_2025"]), c["signs_neg"]] for i, c in enumerate(cur)]
    t_curve = table(["束の列数", "足した列", "実 Δ3（LOYO）", "シャドウ平均", "シャドウ範囲", "2025（参考）", "年の符号一致"], rows_b, ["n", "", "n", "n", "n", "n", "n"])

    t_sens = table(["縮約の強さ（L2）", "最小の列数", "LOYO の Δ3", "2025（1 回）", "同じ列数のシャドウ束 Δ3"],
                   [[f"200（事前登録）", kb, fmt(cur[kb - 1]["d3"]), fmt(be["d3_2025"]), " / ".join(f"{x*E4:+.2f}" for x in sh[:, kb - 1])]] +
                   [[f"{int(float(l))}（事後の感度）", v["k_best"], fmt(v["d3_at_kbest"]), fmt(v["d3_2025_at_kbest"]), " / ".join(f"{x*E4:+.2f}" for x in v["shadow_d3_at_kbest"])] for l, v in bs.items()],
                   ["", "n", "n", "n", "n"])

    t_comp = table(["合成スコア（係数は A 年、較正は B 年、評価は C 年）", "LOYO の Δ3", "年別 22/23/24", "2025（1 回）"],
                   [["事前登録（L2=200）", f"{fmt(cn['d3'])} {ci(*cn['ci95'])}", f"{fmt(cn['d3_2022'],1)} / {fmt(cn['d3_2023'],1)} / {fmt(cn['d3_2024'],1)}", fmt(cn["d3_2025"])]] +
                   [[f"L2={int(float(l))}（事後の感度）", fmt(v["d3"]), "（3 年とも負）" if v["signs_neg"] == 3 else "-", fmt(v["d3_2025"])] for l, v in cs.items()] +
                   [["シャドウの合成スコア（3 本）", " / ".join(fmt(x) for x in r["composite_shadow_d3"]), "-", "-"]], ["", "n", "n", "n"])

    def rrow(nm, key, d):
        return [nm, d["n_flags"] if "n_flags" in d else "-", fmt(d["d3"]), ci(*d["ci95"]) if "ci95" in d else "", fmt(d["d3_2025"])]
    t_rules = table(["ルール台帳の束（L2=400、事前登録）", "フラグ数", "LOYO の Δ3", "95% 区間", "2025"],
                    [rrow("全ルール", "all", rb["all"]), rrow("採用済み（発見期間で選んだもの）", "adopted", rb["adopted"]), rrow("未採用のみ", "non_adopted", rb["non_adopted"])] +
                    [["日内並べ替えのシャドウ（全ルール）", "-", " / ".join(fmt(x) for x in rb["shadow_all_d3"]), "", "-"]], ["", "n", "n", "", "n"])
    rows_rs = []
    for l, v in rl.items():
        for nm, key in (("全ルール", "all"), ("採用済み", "adopted"), ("未採用のみ", "non_adopted")):
            rows_rs.append([f"{int(float(l))}", nm, fmt(v[key]["d3"]), fmt(v[key]["d3_2025"]), v[key]["signs_neg"]])
        rows_rs.append([f"{int(float(l))}", "シャドウ（全ルール 2 本）", " / ".join(fmt(x) for x in v["shadow_all_d3"]), "-", "-"])
    t_rules_s = table(["L2（事後の感度）", "対象", "LOYO の Δ3", "2025", "年の符号一致"], rows_rs, ["", "", "n", "n", "n"])

    top = sorted(us, key=lambda k: us[k]["d3"])[:6]
    rows_sub = [[k, f"{fmt(us[k]['sub']['run0']['d3'])} / {fmt(us[k]['sub']['run0']['d3_2025'])}", f"{fmt(us[k]['sub']['run1_2']['d3'])} / {fmt(us[k]['sub']['run1_2']['d3_2025'])}",
                 f"{fmt(us[k]['sub']['run3p']['d3'])} / {fmt(us[k]['sub']['run3p']['d3_2025'])}"] for k in top]
    t_sub = table(["単位（Δ3 の大きい順）", "データ内で初出走（run_no=0）：LOYO / 2025", "1〜2 走：LOYO / 2025", "3 走以上：LOYO / 2025"], rows_sub, ["", "n", "n", "n"])

    cols = r["columns"]
    order = r["bundle_order"]
    rows_col = [[c, COL_DESC.get(c, ""), cols[c]["signs_neg"], fmt(cols[c]["d3"]), f"{cols[c]['redundancy']:.2f}", f"{cols[c]['resid_corr_win']:+.3f}", f"{cols[c]['resid_corr_top3']:+.3f}",
                 "・".join(f"{x*100:+.2f}" if x is not None else "-" for x in cols[c]["resid_bins_top3"]), "弱" if c in r["weak_signal_cols"] else ""] for c in order]
    t_col = table(["列（束の順）", "意味", "年の符号一致", "単独 Δ3", "冗長性（他列との平均|相関|）", "残差との相関（勝）", "（3 着内）", "3 着内の残差：五分位（pt）", "弱いシグナル"], rows_col,
                  ["", "", "n", "n", "n", "n", "n", "n", ""])

    dsc = r["desc_2026"]
    rows_d = [[c, f"{dsc[c]['2024']['mean']:.3f}", f"{dsc[c]['2025']['mean']:.3f}", f"{dsc[c]['2026']['mean']:.3f}", f"{dsc[c]['2025']['missing']*100:.0f}% / {dsc[c]['2026']['missing']*100:.0f}%"] for c in order if c in dsc]
    t_desc = table(["列", "2024 平均", "2025 平均", "2026 平均（記述のみ）", "欠損率 2025 / 2026"], rows_d, ["", "n", "n", "n", "n"])
    lk_rows = [[x["cut"], f"{x['rows']:,}", f"{x['max_abs_diff']:.1e}"] for x in lk["truncation"]]
    lk_rows2 = [[x["cut"], f"{x['max_abs_diff']:.1e}"] for x in lk["future_perturb"]]
    sd = lk["sameday_perturb"][0]

    unit_defs = "".join(f"<tr><td>{k}</td><td>{u['family']}</td><td>{'・'.join(u['cols'])}</td><td>{u['infer_cost']}</td><td>{u['cost_note']}</td></tr>" for k, u in us.items())
    best = us[best_u]

    html = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>学習前の特徴量スクリーニング（tp-76）— keiba-vpn</title>
<style>:root {{ --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; }}
* {{ box-sizing: border-box; }} body {{ font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1180px; margin: 0 auto; }}
a {{ color: var(--accent-2); }} h1 {{ font-size: 26px; }} h2 {{ font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; }} h3 {{ font-size: 16px; margin: 22px 0 8px; }}
table {{ border-collapse: collapse; margin: 12px 0 18px; font-size: 12.5px; background: var(--panel); }} th, td {{ border: 1px solid var(--border); padding: 5px 9px; text-align: left; vertical-align: top; }} th {{ background: #1b2a4d; }}
td.num {{ text-align: right; font-family: monospace; }} .muted {{ color: var(--muted); font-size: 12.5px; }} .note {{ background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }}</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>学習前の特徴量スクリーニング（tp-76）</h1>
<p class="muted">ユーザーの問い（2026-10-22）: 『学習の前に、採用可能かどうかの特徴量エンジニアリングを実施できないか』『採用基準の厳しさを甘くする作戦も考えてほしい』『とても小さい効果の特徴量でも、束になれば効果が出る可能性がある』。候補 A〜G の {len(us)} 単位（{len(r['columns'])} 列）を as-of で作り、基準モデル（v6e の LOYO OOF 2022〜24・test 2025）の logit に少数の係数で足したときの LogLoss の増分を測った。<b>モデルの学習（GBM・NN）はしていない</b>。市場オッズ・人気は特徴量にも係数にも使っていない。2026 年は選択に使わず、記述統計だけ。</p>
<div class="note"><b>要点.</b>
<ul>
<li><b>個別では、効果はどれも極小.</b> 候補の 1 頭あたり LogLoss の増分（Δ3、3 目的の平均、負が改善）で最良は {best_u} の {fmt(best['d3'])}×10⁻⁴ {ci(*best['ci95'])}（2025 年 {fmt(best['d3_2025'])}×10⁻⁴）。従来の基準 S1・S2・S3 を通る候補は 0 / {crit['S1']['n_real']}。</li>
<li><b>シャドウ特徴量で測ると、緩めた基準も使える段階がある.</b> 偽の特徴量（同じ日の中で値を並べ替え／同じレース内で馬をずらす）81 単位を同じ手順に通した偽陽性率は、S4（経験ベイズ・局所 FDR）で {crit['S4']['fpr']*100:.1f}%（本物 {crit['S4']['real_pass']} 単位が通過）、S5a（片側 90%）で {crit['S5a']['fpr']*100:.1f}%（{crit['S5a']['real_pass']} 単位）、S5b（片側 80%）で {crit['S5b']['fpr']*100:.1f}%（{crit['S5b']['real_pass']} 単位）。本物の最良 {fmt(r['best_real_d3'])}×10⁻⁴ は、偽の最良 {fmt(r['best_shadow_d3'])}×10⁻⁴ を上回る。偽の特徴量を足すと中央値で {fmt(sh_med)}×10⁻⁴ 悪化する（足すだけの過学習のコスト）ので、増分はこの分を差し引いて読む。</li>
<li><b>推奨: 特徴量の候補入りは S4 かつ LOYO 3 年同符号.</b> 偽陽性率が 2.5% 以下で、通る本物が 13 単位ある。S2・S3 は偽陽性率 0% だが、要求する効果（−1〜−2×10⁻⁴）が候補の実力より大きく、通る本物が 0（検出力が足りない）。馬券の規則の基準（回収率の下限 &gt; 100% など）は緩めない。</li>
<li><b>個別では不採用でも、束にすると第 2 基準で『候補』になる.</b> 弱いシグナル（LOYO 3 年同符号）を束ねた合成スコアは、LOYO で {fmt(cn['d3'])}×10⁻⁴ {ci(*cn['ci95'])}、2025 年（1 回）で {fmt(cn['d3_2025'])}×10⁻⁴。同じ手順の偽の合成スコアは {fmt(np.mean(r['composite_shadow_d3']))}×10⁻⁴（悪化側）で、区間は 0 を外れる。ただし大きさは 1 頭あたり約 1×10⁻⁴ で、従来の採用基準 2×10⁻⁴ の半分。束の列数に応じた曲線は第 5 節。</li>
<li><b>ルール台帳（tp-75）の弱いシグナルも束になる.</b> 1,070 型のうち未採用のルールだけを束ねても改善しない（{fmt(rl['4000.0']['non_adopted']['d3'])}×10⁻⁴ で悪化側）。発見期間で採用した 178 フラグの束は、2025 年（1 回）で {fmt(rl['4000.0']['adopted']['d3_2025'])}×10⁻⁴（L2=4000、事後の感度）。事前登録の弱い縮約（L2=400）では過学習する。</li>
<li><b>注意.</b> 少数パラメータの加法的な増分は、GBM が使う非線形・交互作用を過小評価する。判定は『不採用』ではなく、学習再開後に本試験へ進める優先度。血統（種牡馬）を使う B・D は、現状の不完全な 10 世代表に基づくので、完全な表が揃った後に再実行する。</li>
</ul></div>

<h2>1. 候補と作り方（事前登録 edccb96）</h2>
<p>候補は {len(us)} 単位（列 {len(r['columns'])}）、シャドウは 1 単位につき 3 本の計 {len(us)*3} 単位を、評価の実行前に登録した（<code>src/scripts/data/tp76_prereg.json</code>）。すべて as-of（その走・その日の自分より後の結果を使わない）で、作成コードは <code>src/pipeline/features/tp76_asof.py</code>。母集団の定数（年齢×性別の平均、ノイズ）は 2020〜2021 年の走だけから作る。</p>
<table><tr><th>単位</th><th>族</th><th>列</th><th>推論コスト</th><th>コストの理由</th></tr>{unit_defs}</table>
<ul>
<li><b>A 同日バイアス</b>: 同じ日・同じ場・同じ面の、ラウンドが自分より小さいレースの『前残り（初角位置と着順の順位相関）』『内有利（枠と着順の相関）』を、レースの頭数で重み付けして縮約。馬の普段の位置取り・今回の枠と掛けて『適合度』にする。</li>
<li><b>B 潜在能力</b>: 局所水準モデル（カルマン）。y = 着順の正規スコア（またはめぐ指数のレース内 z）、事前平均は年齢×性別の曲線＋同じ種牡馬の厳密に前の産駒の初出走の平均（縮約）。休養が長いほど能力の揺れを大きくする。</li>
<li><b>C 対戦関係</b>: レースごとに『負かした割合』と期待（ロジスティック）の差で馬の強さを更新（出走数に応じて更新幅を減らす）。</li>
<li><b>D 条件別適性</b>: B の予測誤差（その走で期待より上か下か）を、条件（距離帯・競馬場×面・馬場状態・クッション値帯）ごとに馬自身で集計し、同じ種牡馬の産駒の同条件の平均（事前）へ縮約。『D5r』は事前なし。『D6』は上がり順位の予測誤差版。</li>
<li><b>E〜G</b>: 上がり（コース基準・同日水準の補正、前後の位置での上がり、前走の不利、安定性）、負荷・疲労（直近の出走数・距離・負担・叩き回数）、メンバー構成（先行馬の数、自分の順位、内側の先行馬、ペース圧力）。</li>
</ul>
<h3>測り方</h3>
<p>offset = logit(v6e の確率)（勝ち・2 着内・3 着内）に、候補の標準化した列（欠損率 2% 超は欠損フラグつき）の係数を足す L2 ロジスティックを当てはめる。基準は切片だけ再較正した v6e。LOYO は 2022/23/24 を 1 年ずつ保留して他の 2 年で当てはめ、<b>2025 は 2022〜24 で当てはめて 1 回だけ評価</b>。Δ3 = 3 目的の 1 頭あたり 2 値 LogLoss の増分の平均。区間はレース単位（{r['n_races']:,} レース・{r['n_rows']:,} 頭）。</p>

<h2>2. リーク検査（結果）</h2>
<p>tp-65 の方式を流用。（1）切り詰め: ある日までの表だけで作り直した特徴量が、全期間で作った同じ行の値と一致するか。（2）未来の摂動: ある日より後の結果（着順・位置取り・上がり・タイム・めぐ指数）を並べ替えても、その日までの特徴量が変わらないか。（3）同日の摂動: 同じ日のあるレース以降の結果を並べ替えても、そのレースまでの特徴量が変わらないか。</p>
{table(['切り詰め：切る日', '行数', '最大の差'], lk_rows, ['', 'n', 'n'])}
{table(['未来の摂動：切る日', '最大の差'], lk_rows2, ['', 'n'])}
<p>同日の摂動（{sd['day']}、レース {sd['race_id']} 以降）: 最大の差 {sd['max_abs_diff']:.1e}。結果: <b>{'すべて一致（先読みなし）' if lk['pass'] else '不一致あり'}</b>。欠損の位置が違う場合は不一致として扱う。単体テスト（<code>tests/pipeline/test_tp76_asof.py</code>）でも同じ 3 種を確認している。</p>

<h2>3. 候補ごとの結果</h2>
<p>Δ3 の単位は ×10⁻⁴（1 頭あたり、負が改善）。『基準 logit との相関』は、基準モデルの勝ちの logit とその列の最大の絶対相関で、冗長性の目安（高いほど基準がすでに持っている情報）。『本試験の優先度』: 高 = S5a を通り LOYO 3 年同符号で 2025 も負、中 = S4・S5b・3 年同符号のいずれか、低 = それ以外。『費用便益』: 優先度が高、または優先度が中でコストが低く 2025 も負なら最優先。</p>
<img alt="Δ3 by unit" style="max-width:100%" src="data:image/png;base64,{figs['units']}" />
{t_units}
<h3>初出走と履歴のある馬（部分集団）</h3>
{t_sub}
<p class="muted">『データ内で初出走』は 2020 年以降のデータでの最初の出走（run_no=0）。数 {us[top[0]]['sub']['run0']['n']:,} 頭走。</p>

<h2>4. シャドウ較正と基準の段階</h2>
<p>偽の特徴量（P: 同じ日の中で値を並べ替え、H: 同じレース内で馬を 1 つずらす、P: 別の乱数）を、候補の単位ごとに 3 本、計 {len(r['shadow_units'])} 本作り、同じ手順に通した。分布と欠損の構造は本物と同じで、情報だけが壊れている。偽陽性率 = 偽の単位のうち基準を満たした割合。</p>
{t_crit}
<ul>
<li>偽の単位の Δ3: 最小 {fmt(r['shadow_d3_q']['min'])}、5% 点 {fmt(r['shadow_d3_q']['p05'])}、中央値 {fmt(r['shadow_d3_q']['median'])}、95% 点 {fmt(r['shadow_d3_q']['p95'])}（×10⁻⁴）。偽の特徴量は平均して悪化側に寄る（足すだけで過学習する分）。本物の最良 {fmt(r['best_real_d3'])} は偽の最良 {fmt(r['best_shadow_d3'])} を上回る（Boruta 型の判定）。</li>
<li>S4 の経験ベイズ: 候補の効果の分布を正規 N({r['eb']['mu']*E4:+.2f}, {np.sqrt(r['eb']['tau2'])*E4:.2f}²)×10⁻⁴ と推定（モーメント法）し、各候補の事後 P(効果&lt;0) を出した。局所 FDR の帰無はシャドウの z 分布（平均 {r['eb']['null_mean']:+.2f}、標準偏差 {r['eb']['null_sd']:.2f}）。</li>
<li><b>推奨.</b> 特徴量の『候補入り』は <b>S4 かつ LOYO 3 年同符号</b>（偽陽性率 {crit['S4']['fpr']*100:.1f}% 以下）。これを満たす本物は {len(s4_n3)} 単位（{'・'.join(s4_n3)}）で、さらに 2025 年の点推定が負だったのは {len(s4_n3_ok)} 単位（{'・'.join(s4_n3_ok)}）。保守的な代替は S5a（偽陽性率 {crit['S5a']['fpr']*100:.1f}%、{crit['S5a']['real_pass']} 単位）。S2・S3 は偽陽性率 0% だが本物が 0 で、今の候補群には厳しすぎる（個別の増分の最大が −1×10⁻⁴ 前後）。馬券の規則の基準は緩めない。</li>
</ul>

<h2>5. 束（弱いシグナルを捨てずに束ねる）</h2>
<p>列ごとに LOYO の Δ3 を測り、|相関| &gt; 0.85 の列はクラスタに束ねて代表を 1 列選ぶ（二重に数えない。束ねたクラスタ: {'; '.join('・'.join(c) for c in r['clusters'])}）。順位は<b>効果の大きさではなく</b>、符号の一貫年数（多い順）と冗長性（低い順）。LOYO の 3 年すべて負の列を『弱いシグナル』とした（{len(r['weak_signal_cols'])} 列: {'・'.join(r['weak_signal_cols'])}）。</p>
<img alt="bundle curve" style="max-width:100%" src="data:image/png;base64,{figs['curve']}" />
{t_curve}
<p>束の列数に対する LOYO の Δ3 は k={kb} で最小の {fmt(cur[kb-1]['d3'])}×10⁻⁴ {ci(*be['ci95'])} に達し、その後は飽和して悪化する（足しすぎの過学習）。同じ列数のシャドウ束は {' / '.join(fmt(x) for x in sh[:, kb-1])}（悪化側）で、本物の束が上回る。事前登録の弱い縮約（L2=200）では 2025 年（1 回）が {fmt(be['d3_2025'])}×10⁻⁴ と確認できなかったが、縮約を強めた事後の感度分析では 2025 年も負だった。</p>
{t_sens}
<h3>合成スコアへの圧縮</h3>
<p>全代表列をまとめて L2 ロジスティックに当てはめた線形予測を、勝ち用・3 着内用の 2 本の合成スコアにする。<b>係数は発見期間の外側の年に当てる</b>（A 年で係数、B 年で切片と傾きを較正、C 年で評価。2025 は A=2022〜23・B=2024）。</p>
{t_comp}
<p><b>結論（個別では不採用でも束にすると採用可能になるか）.</b> 答え: 従来の採用基準では『いいえ』、第 2 基準では『候補になる』。 弱いシグナルの束は、1 頭あたり約 −1×10⁻⁴ の改善として、偽の束と区別できる（LOYO の区間が 0 を外れ、偽は悪化側）。2025 年も点推定は負（縮約の強さによらず）で、{'区間は 0 を跨ぐ' if abs(cn['d3_2025']) < 1.96 * cn['se3_2025'] else '区間も 0 を外れる'}（標準誤差 {cn['se3_2025']*E4:.2f}×10⁻⁴）。従来の採用基準（−2×10⁻⁴）には届かないが、第 2 基準（3 年同符号・点推定が負）では『候補』。合成スコアを学習再開後に特徴量として入れる案とし、<b>GBM に入れたときの増分は別途測る</b>。</p>

<h2>6. ルール台帳の弱いシグナルの束</h2>
<p>ルールペルソナの台帳（<code>docs/operations/tp75_rule_ledger.csv</code>、1,070 型、市場非依存）の各ルールの該当フラグを、同じ手順で束にした。式を評価できたのは {r['rules_meta']['used']} 型（失敗 {r['rules_meta']['failed']}、該当が少なすぎる・多すぎるものは除外）。第 2 ラウンド（tp-75b、連言ルール）の結果が出たら追加で取り込む。</p>
{t_rules}
{t_rules_s}
<p><b>読み方.</b> 事前登録の L2=400 では 854 フラグが過学習し、偽の束と同じく悪化する。縮約を強めた事後の感度分析（L2=4,000・40,000）では、採用済みルールの束は LOYO と 2025 年の両方で負、未採用ルールの束は改善せず（正または 0 付近）、偽の束（全ルールの日内並べ替え）は悪化側。<b>採用済みの束の LOYO は、そのルールを発見期間の同じ年で選んだ選択バイアスを含む</b>ので、公平な値は 2025 年の 1 回（約 {fmt(rl['4000.0']['adopted']['d3_2025'])}×10⁻⁴）。個別には基準を満たさないルールの合計の増分は、見つからなかった。</p>

<h2>7. 冗長性・残差との関連</h2>
{t_col}
<p class="muted">『残差との相関』は発見期間の、基準モデルの残差（実績−予測）とその列（標準化）の相関。五分位の残差（3 着内）は、列の低い方から高い方への 5 区分の平均残差（pt）。</p>

<h3>2026 年の記述統計（選択に使っていない）</h3>
{t_desc}
<p class="muted">2026 年の平均・欠損率が 2025 年と大きくずれる列は、推論側で作るときの分布の確認対象。</p>

<h2>8. 検出力の改善: 共通乱数と複数 seed（学習再開後の本試験の設計）</h2>
<p>GBM の増分は −1〜−3×10⁻⁴ 程度を見分ける必要があり、seed 間のばらつきが同じ桁になりうる。次の設計にする（<b>実行は学習再開後</b>。seed のばらつきは未測定で、以下は設計と見積り）。</p>
<ul>
<li><b>共通乱数（common random numbers）</b>: 基準モデルと候補モデルで、fold の分け方・bagging と特徴量サンプリングの seed・データの順序を同じにし、対応のある差（候補 − 基準）だけを集計する。各 seed の差の分散は 2σ²(1−ρ)（σ = 1 モデルの seed 間の標準偏差、ρ = 基準と候補の相関）。ρ が高いほど差の分散が小さくなる。</li>
<li><b>複数 seed の平均</b>: S 本の seed の差を平均すると分散は 1/S。目標は、差の標準誤差 ≤ 3×10⁻⁵（−1×10⁻⁴ を約 3σ で見分ける）。S は先行実験で決める: v6e の基準を 3 seed 回して σ と ρ を測る（約 4 時間 × 2 並列で 6〜8 時間）。</li>
<li><b>見積り</b>: 候補 1 組あたり LOYO（7 メンバー）が 1 seed で約 4 時間（v6 監査の見積り）。基準は v6e を再利用できるので、S=3 で約 12 時間（2 並列で約 6 時間）、S=5 で約 20 時間（2 並列で約 10 時間）。</li>
<li>レース単位ブートストラップは今回と同じ。seed 間と fold 間のばらつきを合わせた階層ブートストラップも併用する。</li>
</ul>

<h2>9. 束と後ろ向き除去（学習再開後）</h2>
<ol>
<li><b>一括投入</b>: 弱いシグナルの代表列（{len(r['bundle_order'])} 列）と合成スコア 2 本を v6e に足した 1 モデルを学習（LOYO 約 4 時間 + 最終学習 2 回で約 3.4 時間 = 約 7〜8 時間）。</li>
<li><b>後ろ向き除去</b>: (a) L2 正則化の強さを上げて係数が落ちる順に見る（学習済みモデルの入力に対する permutation importance は約 30 分）。(b) 族ごとの ablation（A〜G の 7 族を 1 つずつ外す）: 7 回 × 約 4 時間 = 約 28 時間（3 並列で約 10 時間）。(c) 残った束の合計の増分を <b>test 2025 で 1 回だけ</b>評価する。</li>
<li><b>シャドウ束</b>: 同じ本数の偽の列を足した学習を 1 回（約 4 時間）。本物の束の増分が、偽の束を上回る点を示す（Boruta 型。学習内の重要度の比較も出す）。</li>
<li>合計の見積り: 一括投入 + シャドウ + 族ごとの ablation で、並列 3 本なら約 1.5 日。</li>
</ol>

<h2>10. 学習再開後の進め方（優先順位）</h2>
{table(['順位', '対象', '理由'],
       [['1', '合成スコア（弱いシグナルの束）と、D2・D5・D5r（競馬場×面と 4 軸の適性）', 'S4 かつ 3 年同符号、2025 年も負。ただし完全な 10 世代表が揃ってから血統の事前を作り直す'],
        ['2', 'A2（日内の内有利×今回の枠）、B3（めぐ指数のカルマン）', 'S5a を通る。2 年は負。推論側は日内の結果の取り込み、めぐ指数の履歴状態が必要'],
        ['3', 'E1・E2（上がりのコース基準・同日補正）、D4・D6', 'コースの基準表だけで作れる（コース低）／D4・D6 は LOYO では負だが 2025 は未確認'],
        ['4', 'C（Elo）、F、G', '基準 logit と相関が高い（C）、または効果が見えない（F・G）。コスト低なので束に入れて除去で判断']],
       ['', '', ''])}
<p>推論側の整備: 採用した列は、学習側と同じ値になることを v6_parity と同じ手順で確認してから shadow に入れる。A は同日の結果の取り込み、B・C は日次の状態、D は馬×条件と種牡馬×条件の事前表が必要（推論コスト欄）。復活セット（<a href="revival_feature_set.html">revival_feature_set.html</a>）の段階・偽陽性率の見積りは、この章の第 4 節を取り込む。</p>

<h2>11. 限界</h2>
<ul>
<li>血統（種牡馬）を使う B・D は、現状の不完全な 10 世代表（5 世代＋牡アンカー）に基づく。完全な表が揃ったら B1・B2・B3・D1〜D6 を再実行する。</li>
<li>少数パラメータの加法的な増分は、GBM の非線形・交互作用を過小評価する。『効果が見えない』は『本試験に進めない』ではない。</li>
<li>基準モデルは、2022〜24 が LOYO の OOF、2025 が bundle_v6_2024 の test 予測で、モデルが違う。基準は各評価で切片だけ再較正した。</li>
<li>束・合成スコア・ルールの縮約（L2=2,000 以上）は事後の感度分析で、事前登録は L2=200（束）と 400（ルール）。事後の値で最良を選んでいない（3 値を並べて示した）。</li>
<li>第 2 ラウンドの連言ルール（tp-75b）は未取り込み。2026 年は記述のみ（<code>results.json</code> の desc_2026）。</li>
<li>seed のばらつき（第 8 節）は未測定。</li>
<li>評価コードは最初の実行の後に、部分集団・残差の五分位・2026 の記述・縮約の感度分析を足した（主な数値は決定的で、最初の実行と一致。最初の出力は <code>results_run1_before_additions.json</code> に残してある）。</li>
</ul>
<p class="muted">再現: <code>python -m src.scripts.data.tp76_screen prereg → build → eval</code>、<code>python -m src.scripts.data.tp76_report &lt;出力ディレクトリ&gt;</code>（ホストでは heavy_run.sh 経由）。結果は <code>/home/jovyan/work/tp76_out/</code>。</p>
</body></html>"""
    (CH / "feature_screening.html").write_text(html, encoding="utf-8")
    print("chapter written", len(html))
    log_section(r, crit, cn, be)


def log_section(r: dict, crit: dict, cn: dict, be: dict) -> None:
    p = CH / "training_pipeline_decisions.html"
    t = p.read_text(encoding="utf-8")
    t = re.sub(r"<h2>71\. 以降の節</h2>", "<h2>72. 以降の節</h2>", t)
    if "<h2>71. 学習前の特徴量スクリーニング" in t:
        return
    sec = f"""<h2>71. 学習前の特徴量スクリーニング（tp-76）</h2>
<p class="muted">章: <a href="feature_screening.html">feature_screening.html</a>。コード: src/pipeline/features/tp76_asof.py・src/scripts/data/tp76_screen.py・tp76_report.py・tests/pipeline/test_tp76_asof.py。事前登録は評価の実行前にコミット（edccb96、候補 27 単位・32 列・シャドウ 81 単位）。モデルの学習はしていない。</p>
<ul>
<li><b>問い（ユーザー）.</b> 学習の前に、採用可能かどうかの特徴量エンジニアリングをする。採用基準の厳しさを緩める作戦（シャドウ特徴量による較正）と、とても小さい効果を束ねる作戦も考える。</li>
<li><b>結果 1（個別）.</b> 候補の個別の増分は極小（Δ3 の最良 {r['best_real_d3']*E4:+.2f}×10⁻⁴）。従来の基準 S1・S2・S3 を通る候補は 0/27。リーク検査（切り詰め一致・未来の摂動・同日の摂動）はすべて一致。</li>
<li><b>結果 2（基準の段階とシャドウ）.</b> 偽の特徴量 81 単位の偽陽性率は S4 で {crit['S4']['fpr']*100:.1f}%（本物 {crit['S4']['real_pass']}）、S5a で {crit['S5a']['fpr']*100:.1f}%（{crit['S5a']['real_pass']}）、S5b で {crit['S5b']['fpr']*100:.1f}%（{crit['S5b']['real_pass']}）。<b>推奨: 特徴量の候補入りは S4 かつ LOYO 3 年同符号</b>。馬券の規則の基準は緩めない。</li>
<li><b>結果 3（束）.</b> 弱いシグナルの束と合成スコアは、LOYO で {cn['d3']*E4:+.2f}×10⁻⁴（区間 {cn['ci95'][0]*E4:+.2f}〜{cn['ci95'][1]*E4:+.2f}）、2025 年（1 回）で {cn['d3_2025']*E4:+.2f}×10⁻⁴。偽の束は悪化側。個別では不採用でも、束にすると第 2 基準では候補になる（従来の −2×10⁻⁴ には届かない）。ルール台帳の未採用ルールの束は改善せず、採用済みの束は 2025 年で負（事後の縮約）。</li>
<li><b>申し送り.</b> 学習再開後は、共通乱数と複数 seed、一括投入と後ろ向き除去の設計（章の第 8・9 節）で本試験へ。血統を使う B・D は完全な 10 世代表の後に再実行。連言ルール（tp-75b）の結果が出たら束に追加。</li>
</ul>

<h2>72. 以降の節</h2>"""
    t = t.replace("<h2>72. 以降の節</h2>", sec, 1)
    p.write_text(t, encoding="utf-8")
    print("decision log §71 added")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
