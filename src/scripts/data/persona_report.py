"""persona_gating.py の結果（persona_gating.json）から、章 persona_ensemble_survey.html を生成する。

python -m src.scripts.data.persona_report --src persona_gating.json [--out docs/html/modeling/chapters/persona_ensemble_survey.html]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

PERSONA_JA = {"ABILITY": "能力重視家", "LAP": "ラップ予想家", "TRACKQ": "馬場質予想家", "PED": "血統予想家", "BIAS": "トラックバイアス予想家", "FIELD": "（参考）メンバー構成"}
EN_P = {"ABILITY": "ability", "LAP": "lap", "TRACKQ": "track quality", "PED": "pedigree", "BIAS": "track bias", "FIELD": "field (ref.)"}
PERSONA_DEF = {
    "ABILITY": "HIST_RESULT・STATE・EXP・CONN_JKTR（過去の成績・状態・経験・騎手と調教師の事前成績）",
    "LAP": "STAGE_AUX（前段の走破時間・着順の予測、位置取り・上がりの補助ヘッド）・FIELD_PACE（展開の圧力）・HIST_SPEED（上がり・速度の履歴）",
    "TRACKQ": "AFFINITY（競馬場・馬場状態・コース別の適性）・FIELD_ENV（開催時期・環境）。実測のクッション値・含水率は V4 に無く、代理は弱い",
    "PED": "PED（種牡馬・母父×条件の成績、sire_x_*）",
    "BIAS": "COND（枠・馬番・頭数・当日と前日の勝ち馬上がり差 track_bias_* ほか）",
    "FIELD": "FIELD_MEM・FIELD_DEMO・FIELD_CONN（出走メンバーの構成）",
}
INCL = [
    ("ラップ予想家", "前段 s_time_pred / s_rank_pred / mtl_*（位置取り・上がりの補助ヘッド）、FIELD_PACE（f1〜f5）、HIST_SPEED。ラップ予想・追走難度 v2・位置取り・想定上がりは v6e の前段 e の入力",
     "高", "STAGE_AUX が全列 LightGBM の gain の 52%、FIELD_PACE 3%、HIST_SPEED 2%（tp-51）。展開タイプ・展開利は tp-61・tp-64 で追加なし"),
    ("能力重視家", "HIST_RESULT・HIST_SPEED・EXP・STATE、めぐ指数 v3a、v6e の RB 16 列", "高", "HIST_RESULT 6%・STATE 2%・EXP 0.4%。単体の専門家でも最良に近い（ABILITY 2.17 / LAP 2.10）"),
    ("血統予想家", "PED 32 列、tp-38 の ALL_BEST 43 列（V5）、血統キー復旧、sire_x_*", "中〜高", "PED 2.5%。tp-60 の血統×レース質・血統クラスタは V5 への累積で不採用。推論側の血統名は 10/22 に不一致を見つけて修正中"),
    ("馬場質予想家", "AFFINITY（b_*）、FIELD_ENV（時期）。実測の馬場質（クッション値・含水率・同日の馬場実測の偏り）は V4/V5 に無い", "低〜中",
     "クッション値は tp-16b で不採用、tp-72 の 73 型（脚質×枠×クッション値）も通過なし。『当日・前日の実測の偏り』は棚卸しの新規候補（見込み −0.0000〜−0.0003）"),
    ("トラックバイアス予想家", "COND の track_bias_*（同日・前日の勝ち馬上がり差）、枠・馬番、FIELD_PACE", "中", "COND 1%。逃げはさみ・ブリンカー×脚質・脚質×枠は tp-72 で優位なし。位置×コース×馬場×ペースは tp-74 で織り込み済み"),
]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def _fonts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = [n for n in ("Noto Sans CJK JP", "Noto Sans JP", "Noto Serif CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "Yu Gothic", "Meiryo", "DejaVu Sans") if n in have][:1] or ["DejaVu Sans"]
    return plt


def make_figs(r: dict) -> list[tuple[str, str]]:
    plt = _fonts()
    figs = []
    fs = 13.6
    # 図 1: ペルソナ単体と全列モデルの差
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ks = list(r["persona_alone"])
    m = [r["persona_alone"][k]["d_vs_full"][0] for k in ks]
    lo = [r["persona_alone"][k]["d_vs_full"][1] for k in ks]
    hi = [r["persona_alone"][k]["d_vs_full"][2] for k in ks]
    ax.barh([EN_P[k] for k in ks], m, xerr=[[a - b for a, b in zip(m, lo)], [b - a for a, b in zip(m, hi)]], color="#4f8bff")
    ax.set_xlabel("race-level NLL minus full model (larger = worse)")
    ax.invert_yaxis()
    figs.append(("ペルソナ単体は全列モデルより悪い（横軸: 全列モデルとのレース単位 NLL の差。大きいほど悪い）", _png(fig)))
    plt.close(fig)
    # 図 2: ゲーティング（固定重みとの差）
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    lams = [3, 30, 300, 3000]
    for rg, lab in (("sd", "surface x dist"), ("going", "going"), ("cls", "class"), ("fsz", "field size"), ("season", "season"), ("venue", "venue"), ("sg", "surface x going")):
        ys = [r["gated"][f"{rg}|lam{l}"]["d_vs_fixed"][0] for l in lams]
        ax.plot(lams, ys, marker="o", label=lab)
    ax.axhline(0, color="#999", lw=1)
    ax.axhline(-2.7e-3, color="#fbbf24", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("shrinkage lambda (larger = closer to fixed weights)")
    ax.set_ylabel("vs fixed weights (negative = better)")
    ax.legend(fontsize=7, ncol=2)
    figs.append(("条件別の重みは、弱い縮約では過学習、強い縮約では固定重みに収束（縦軸: 固定重みとの差、負=改善。凡例: 面×距離帯, 馬場状態, クラス, 頭数, 季節, 競馬場, 面×馬場。破線=採用基準 0.0002/頭 ≈ 0.0027/レース）", _png(fig)))
    plt.close(fig)
    # 図 3: 不一致の三分位ごとの本命の的中率 - 予測
    fig, ax = plt.subplots(figsize=(5, 3.2))
    d = r["disagreement"]
    ks = ["低", "中", "高"]
    ax.bar(["low", "mid", "high"], [(d[k]["top1_hit"] - d[k]["top1_pred"]) * 100 for k in ks], color="#7aa9ff")
    ax.axhline(0, color="#999", lw=1)
    ax.set_ylabel("favorite hit rate - predicted (pt)")
    ax.set_xlabel("persona disagreement tercile (std of log-prob)")
    figs.append(("ペルソナが割れているレースほど、本命は予測より来る（横軸: ペルソナ間の不一致の三分位 low/mid/high）", _png(fig)))
    plt.close(fig)
    return figs


def f(x: float, nd: int = 4) -> str:
    return f"{x:+.{nd}f}"


def build(r: dict) -> str:
    figs = make_figs(r)
    fs = 13.6
    fx = r["fixed_stack"]
    cl = r["gated"]["cls|lam30"]
    head = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>予想家ペルソナ別サブモデル＋条件別ゲーティングは有用か（調査） — keiba-vpn</title>
<style>:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1100px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1b2a4d; }
td.num { text-align: right; font-family: monospace; } .muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>予想家ペルソナ別サブモデル＋条件別ゲーティングは有用か（調査）</h1>
<p class="muted">ユーザーの問い（2026-10-22）: 『予想重視が違うペルソナごとにサブモデル（スタッキングあり）を複数算出し、コースや馬場、条件に応じてサブモデルの重みをチューニングする作戦は有用か。ラップ予想家・馬場質予想家・能力重視家・血統予想家・トラックバイアス予想家を考えている』。
調査と設計まで（モデルの学習はしていない）。コード: src/scripts/data/persona_gating.py（run）・persona_report.py。入力は tp-51 のグループ別専門家の OOF（valid 2022・2023、94,022 行・6,905 レース）。市場オッズは使わない。test 2025・2026 年は使っていない。</p>
"""
    note = f"""<div class="note"><b>要点.</b>
<ul>
<li><b>判断: 精度向上の手段としては、現行の形では見送り（no-go）。</b> 5 つの観点は V4/V5 の特徴量と前段・補助ヘッドにすでに内包されている（第 1 章）。代理のペルソナ（専門家の対数確率の平均）の単体は、全列モデルより一貫して悪い（NLL が +0.10〜+0.60/レース）。</li>
<li><b>固定重みのスタック（全列 + 6 ペルソナ）でも改善は基準未満。</b> 交差年で {f(fx[0])}/レース（≈ {f(fx[0]/fs)}/頭）、95% 区間 [{f(fx[1])}, {f(fx[2])}] で 0 を跨ぐ。採用基準 0.0002/頭（≈ 0.0027/レース）に届かない。</li>
<li><b>条件別の重み（ゲーティング）は、弱い縮約では過学習で悪化し、強い縮約では固定重みに収束する。</b> 面×距離帯 λ=3 で {f(r['gated']['sd|lam3']['d_vs_fixed'][0])}、競馬場 λ=3 で {f(r['gated']['venue|lam3']['d_vs_fixed'][0])}（どちらも悪化、区間は 0 の上）。λ=3000 ではどの条件も ±0.0003 以内。唯一の兆候はクラス別（λ=30: 固定重みとの差 {f(cl['d_vs_fixed'][0])} [{f(cl['d_vs_fixed'][1])}, {f(cl['d_vs_fixed'][2])}] ≈ {f(cl['d_vs_fixed'][0]/fs)}/頭、区間は 0 を跨ぐ）。</li>
<li><b>条件による『最良ペルソナ』の入れ替わりは見えない。</b> 7 つの条件軸すべてで、どのセルでも単体最良は LAP（ラップ予想家）で、2022・2023 年のセル別プロファイルの相関は 0.97〜0.998。tp-51 の 面×距離帯の重み（S8 +0.00001）やグループ別専門家（±0.00003）と同じ結論。</li>
<li><b>別の使い道は残る:</b> ペルソナ間の不一致が大きいレースほど、全列モデルの本命は予測より来る（低 {f((r['disagreement']['低']['top1_hit']-r['disagreement']['低']['top1_pred'])*100,1)}pt / 中 {f((r['disagreement']['中']['top1_hit']-r['disagreement']['中']['top1_pred'])*100,1)}pt / 高 {f((r['disagreement']['高']['top1_hit']-r['disagreement']['高']['top1_pred'])*100,1)}pt）。較正と確信度の補助には使える見込みだが、予測の四分位ごとに見ると一貫しない（暫定）。</li>
<li><b>この結果は下限の見積り.</b> 専門家は V4 の 2 年・fast 設定で、純粋なペルソナではない。馬場質・バイアスの実測列は V4 に無く、各方向の学習は 1 年分（約 3,400 レース）のため、ゲーティングの検出力は低い。目的に合わせて作ったペルソナ（実測のクッション値・同日の馬場の偏り・位置取り予想・ラップ予想を入力）で 5 年分の OOF を作れば、結論が変わる余地は残る（第 5 章）。</li>
</ul></div>
"""
    s = head + note
    s += "<h2>1. 台帳: 各ペルソナは、すでに特徴量・前段・補助として入っているか</h2>\n<table><tr><th>ペルソナ</th><th>入力の候補（既存）</th><th>内包の度合い</th><th>根拠</th></tr>\n"
    for a, b, c, d in INCL:
        s += f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>\n"
    s += "</table>\n<p class='muted'>全列モデルは 5 観点の入力をすべて持つ（単一モデルが内包している）。ペルソナ別に分けるのは、情報を足すのではなく『見る列を絞った別の見方』を作ることで、多様性の価値は tp-51 で測った: グループ別専門家の OOF の logit 相関は 0.15〜0.9 と多様だが、単体が弱く（勝率 LogLoss 0.224〜0.252）、V4 の 7 メンバーに足した効果は ±0.00003 以内。</p>\n"
    s += "<h2>2. ペルソナの定義（代理）と、重なり</h2>\n<table><tr><th>ペルソナ</th><th>代理に使ったグループ（V4 の 16 グループ）</th></tr>\n"
    for k in PERSONA_JA:
        s += f"<tr><td>{PERSONA_JA[k]}</td><td>{PERSONA_DEF[k]}</td></tr>\n"
    s += """</table>
<p class="muted">各グループの専門家は『そのグループ + COND（距離・頭数・馬番・枠・track_bias_*）だけ』の LightGBM（tp-51）。ペルソナの出力 = 該当専門家の勝つ確率の対数の平均をレース内で正規化。<b>重なり:</b> 全専門家が COND を共有するため、BIAS の入力はすべてのペルソナに含まれる。LAP の STAGE_AUX は『能力』（走破時間の予測）と『位置取り・上がり』を同時に持つ前段の出力で、能力と分離できない。馬場質の実測列が無いため、TRACKQ は『条件別の適性』の代理にとどまる。市場オッズ・人気は入力にしない。</p>
"""
    s += "<h2>3. 上限の見積り</h2>\n<h3>3.1 ペルソナ単体と全列モデル</h3>\n"
    s += f"<p>全列モデル（fast 設定）のレース単位 NLL は {r['full_nll']:.4f}（勝ち馬の対数確率の符号反転。1 頭あたりの勝率 LogLoss に直すと約 1/{fs}）。</p>\n"
    s += "<table><tr><th>ペルソナ</th><th>NLL</th><th>全列との差 [95% 区間]</th></tr>\n"
    for k, v in r["persona_alone"].items():
        s += f"<tr><td>{PERSONA_JA[k]}</td><td class='num'>{v['nll']:.4f}</td><td class='num'>{f(v['d_vs_full'][0])} [{f(v['d_vs_full'][1])}, {f(v['d_vs_full'][2])}]</td></tr>\n"
    s += "</table>\n"
    s += "<h3>3.2 固定重みのスタックと、条件別の重み（交差年）</h3>\n<p>条件付きロジット（全列モデルの対数確率 + 6 ペルソナの対数確率の線形結合）を、一方の年で当てはめ、もう一方の年で評価（2 方向を連結）。条件別の重みは θ + δ_cell（δ に ridge λ）。区間はレース単位ブートストラップ（2,000 回）。</p>\n"
    s += f"<p>固定重み（全列のみとの差）: {f(fx[0])} [{f(fx[1])}, {f(fx[2])}]。</p>\n"
    s += "<table><tr><th>条件</th><th>λ</th><th>固定重みとの差 [95% 区間]</th><th>全列のみとの差 [95% 区間]</th></tr>\n"
    names = {"sd": "面×距離帯", "going": "馬場状態", "cls": "クラス", "fsz": "頭数", "season": "季節", "venue": "競馬場", "sg": "面×馬場"}
    for key, v in r["gated"].items():
        rg, lam = key.split("|lam")
        a, b = v["d_vs_fixed"], v["d_vs_full"]
        s += f"<tr><td>{names[rg]}</td><td class='num'>{lam}</td><td class='num'>{f(a[0],5)} [{f(a[1],5)}, {f(a[2],5)}]</td><td class='num'>{f(b[0],5)} [{f(b[1],5)}, {f(b[2],5)}]</td></tr>\n"
    s += "</table>\n"
    for cap, b64 in figs:
        s += f"<p><img alt='{cap}' style='max-width:100%' src='data:image/png;base64,{b64}'><br><span class='muted'>{cap}</span></p>\n"
    q = r.get("disagreement_by_pred_quartile", {})
    if q:
        s += "<p class='muted'>予測の水準（本命の予測確率の四分位）ごとの『本命の的中率 − 予測』（pt）を、不一致の三分位で見ると:</p>\n<table><tr><th>予測の四分位（0=低）</th><th>不一致 低</th><th>中</th><th>高</th></tr>\n"
        for k, v in q.items():
            s += f"<tr><td>{k}</td>" + "".join(f"<td class='num'>{v.get(t, 0)*100:+.1f}</td>" for t in ("低", "中", "高")) + "</tr>\n"
        s += "</table>\n<p class='muted'>全体では単調（低 −1.3 → 高 +2.1pt）だが、四分位ごとには一貫しない（第 1・2 四分位は差なし）。暫定。</p>\n"
    s += """<h3>3.3 最良ペルソナの入れ替わり（異質性）</h3>
<p>単体 NLL が最小のペルソナを、条件のセル（2 年合計で 150 レース以上）ごとに 2022・2023 年で比べた。面×距離帯 8、馬場状態 3、クラス 3、頭数 3、季節 4、競馬場 9、面×馬場 7 のすべてのセルで、最良は両年とも LAP。セル別プロファイル（ペルソナ NLL − セル内平均）の年をまたぐ相関は 0.971〜0.998。<b>単体の優劣に条件による入れ替わりは無い</b>（優劣の差は条件間の差より大きく、スタックでの限界的な寄与の違いは、上の gating 結果が示す）。</p>
<h3>3.4 現実的な上限</h3>
<p>採用基準（0.0002/頭 ≈ 0.0027/レース）に対し、固定重みのスタックで ≈ 0.0002/頭に届かず、条件別重みの上乗せは区間が 0 を跨ぐ（最良のクラス別 λ=30 で ≈ 0.00012/頭）。弱い基準（全列 fast 単体）に対する値であり、V4/V5 の 7〜8 メンバーのスタック（すでに多様性がある）に足した場合の上乗せは、tp-51 の実測（専門家・面×距離帯の重みで ±0.00003 以内）から <b>0〜0.0001/頭</b> と見込む。</p>
<h2>4. リスク</h2>
<table><tr><th>リスク</th><th>内容と対策の案</th></tr>
<tr><td>データ量</td><td>年あたり約 3,400 レースに対し、セル数×ペルソナ数（例: 競馬場 10 × 6 = 60 係数）。λ=3 で悪化（+0.0055〜+0.0082/レース）。対策: 階層縮約（全体 → 面×距離帯 → 競馬場）、条件は 2〜3 個のレジーム変数（クラス・面・馬場）に限る、λ は年をまたぐ検証で決める。</td></tr>
<tr><td>重みの不安定</td><td>セル別の係数は年をまたいで入れ替わりやすい（弱い縮約）。対策: 時系列の前進評価、係数に年間の平滑化。</td></tr>
<tr><td>既存アンサンブルとの重複</td><td>ペルソナ代理は V4 の 7 メンバー・補助ヘッド・前段の出力の部分集合で、情報は新しくない。tp-51 でも専門家・面×距離帯の重みは効果なし。</td></tr>
<tr><td>代理の弱さ</td><td>馬場質・バイアスの実測列（クッション値・含水率・同日の馬場の偏り）が V4 に無い。作る場合はその入力が要る。</td></tr>
<tr><td>解釈のわな</td><td>ペルソナの不一致は『モデルが割れている』ではなく、確率の水準と相関する（本命の確率が高いほど不一致の絶対値も大きい）。対数確率の標準偏差に直し、予測の水準別に見る必要がある。</td></tr></table>
<h2>5. 提案する設計（Phase 1。軽量版）</h2>
<ol>
<li><b>ペルソナを目的に合わせて作る（学習が要る。学習の再開の合図を待つ）。</b> 入力を重ならないように割り当て、5 年分（2020〜2024）の LOYO OOF を作る。ラップ予想家=位置取り予想・ラップ予想・追走難度 v2・想定上がり・前段 e、馬場質予想家=実測のクッション値・含水率・同日と前日の馬場の偏り・馬場状態、能力重視家=めぐ指数・RB・履歴、血統予想家=PED・ALL_BEST・母系の適性、トラックバイアス予想家=枠・馬番・同日の勝ち馬の通過順と上がり・脚質別の成績の偏り。各ペルソナは LightGBM 1 本（fast 設定で足りる）。</li>
<li><b>スタッキング:</b> 全列モデル + ペルソナの対数確率の条件付きロジット（固定重み）を基準にし、条件別の重みは階層縮約（クラス・面×馬場の 2 レジーム変数から開始）。λ は年をまたぐ検証で決める。</li>
<li><b>評価:</b> LOYO（2020〜2024）と前進評価、採否は『固定重みとの差の 95% 上限 &lt; 0、効果 ≥ 0.0002/頭』（ゲーティング分のみ）。test 2025 は選んだ構成で 1 回、2026 は参照のみ。</li>
<li><b>go の条件:</b> 目的に合わせたペルソナの固定重みスタックが、V5（可能なら v6e）に対して LOYO で −0.0002/頭を超えること（そうでなければゲーティングは試さない）。所要: OOF 作成 約 6 本 × 5 年（fast 設定で約 6〜9 時間）、スタッキングは数分。</li>
<li><b>馬券の観点:</b> ペルソナ間の不一致を確信度の補助にできるかを、市場オッズ付きで検証する（不一致の三分位 × 予測の水準別の較正差・複勝回収率）。今回は市場を使っておらず、EV の絞りに使えるとは言えない。</li>
</ol>
<h2>6. 結論（go / no-go）</h2>
<ul>
<li><b>no-go（現行の入力と代理のままのゲーティング）:</b> 精度向上の根拠が無い。固定重みの上乗せも基準未満、条件別の重みは過学習または収束、最良ペルソナの入れ替わりも見えない。</li>
<li><b>条件付き go（軽量版の Phase 1 の最初の一歩）:</b> 学習の再開後、馬場質と位置取りの実測列を入力にした 2 ペルソナ（馬場質予想家・トラックバイアス予想家）だけを 5 年分の OOF で作り、V5 に対する固定重みの上乗せを測る。上乗せが −0.0002/頭に届かなければ、ペルソナ路線は打ち切る。</li>
<li><b>学習なしでできる次の一歩:</b> ペルソナの不一致と較正差の関係を、市場オッズ付きで検証する（本命の複勝回収率・EV の絞り）。</li>
</ul>
<p class="muted">限界: 専門家は V4 の fast 設定・2 年分の OOF で、V5/v6e の馬ごとの予測ではない。評価は勝率の条件付きロジット（連対・複勝は未評価）。2 方向の交差年（学習 1 年）で、実運用の学習量（5〜6 年）より検出力が低い。</p>
</body></html>
"""
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="docs/html/modeling/chapters/persona_ensemble_survey.html")
    a = ap.parse_args()
    r = json.loads(Path(a.src).read_text(encoding="utf-8"))
    Path(a.out).write_text(build(r), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
