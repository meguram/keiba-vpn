"""tp-58 の章（docs/html/modeling/chapters/feature_cross_search.html）を結果ファイルから生成する。

python -m src.scripts.data.tp58_report --res /tmp/tp58_res
結果ディレクトリ: summary.json / hand_score.csv / named_score.csv / nn_blend*.json / tree_pairs.json / singles.csv / lgb_*/result.json / resid.json
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/html/modeling/chapters/feature_cross_search.html"
TITLE = "特徴量の掛け合わせ（交互作用・組合せ）の探索（tp-58）"

CSS = """:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; --ok:#4ade80; --bad:#f87171; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1100px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1b2a4d; }
td.num { text-align: right; font-family: monospace; } .muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
.key { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--ok); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
.pos { color: var(--ok); } .neg { color: var(--bad); } code { font-family: ui-monospace, Menlo, Consolas, monospace; background: #0b1428; padding: 1px 5px; border-radius: 4px; }"""


def e(x) -> str:
    return html.escape(str(x))


def table(head, rows, num_from=1) -> str:
    h = "".join(f"<th>{e(c)}</th>" for c in head)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td class='{'num' if i >= num_from else ''}'>{e(c)}</td>" for i, c in enumerate(r)) + "</tr>"
    return f"<table><tr>{h}</tr>{body}</table>"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f5(x) -> str:
    return f"{x:+.5f}".replace("-", "−")


def build(res: Path) -> str:
    S = jload(res / "summary.json")
    tp = jload(res / "tree_pairs.json") or []
    hand = pd.read_csv(res / "hand_score.csv") if (res / "hand_score.csv").exists() else pd.DataFrame()
    named = pd.read_csv(res / "named_score.csv") if (res / "named_score.csv").exists() else pd.DataFrame()
    sing = pd.read_csv(res / "singles.csv", index_col=0) if (res / "singles.csv").exists() else pd.DataFrame()
    resid = {"V5 の LightGBM 単体": jload(res / "resid_lgb.json") or jload(res / "resid.json"), "現行の合成 S0": jload(res / "resid_S0.json")}
    resid = {k: v for k, v in resid.items() if v}
    lgb = {p.parent.name: jload(p) for p in sorted(res.glob("lgb_*/result.json"))}
    p = []
    p.append(f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>{TITLE}</title><style>{CSS}</style></head><body><h1>{TITLE}</h1>")
    p.append("<p class='muted'>ユーザーの問い（2026-10-21）: 各特徴量の効果的な掛け合わせ（交互作用・組合せ）も使えないか、検証する価値がありそう。もうやっているか。対象: V5（V4 の 1,062 列 + tl_ 再取得 + CY 37 列 + ALL_BEST 43 列 = 1,142 列）の上に、掛け合わせが上乗せになるか。検証年 2022・2023・2024（OOF のある年）。test 2025・2026 年は選択に使っていません。市場オッズ・人気は使っていません。</p>")
    p.append("<h2>1. 要点</h2><div class='key'><ul>")
    p.append(f"<li><b>これまでにやっていたこと</b>: 手設計の掛け合わせは G3 cross_misc（採用）、tp-37 の交互作用 11 群（不採用）、tp-38 の祖先×位置×条件のセル、tp-29/31 のコース適性、PED_sire_cond（採用）、tp-51 の特徴量グルーピング型。V5 の列にも、距離変化（<code>chg_dist*</code>）・間隔（<code>rest_*</code>）・脚質×ペース（<code>frp_style_x_pace*</code>）・体重変化（<code>d_wt_chg*</code>）などの掛け合わせが既に入っています。<b>未実施だったのは『自動の総当たり』と、交互作用を表現できるモデル構造の検証</b>で、今回それを行いました。</li>")
    if S:
        n = S["n_candidates"]
        sd = S["null_sd_mad"]
        p.append(f"<li><b>A 自動の総当たり: 上乗せなし。</b>V5 の重要度上位 {S['k']} 列とそのレース内偏差 {S['rel']} 列（基 {S['bases']} 列）から、積・差・対数比 <b>{n:,} 候補</b>を、V5 の OOF 確率に対するスコア検定（再学習なし）で全て調べました（3 目的で検定数 {S['n_tests']:,}）。発見期間（2022–23）で Bonferroni（z={S['bonferroni_z']:.2f}、帰無のばらつき {min(sd.values()):.1f}〜{max(sd.values()):.1f} 倍に補正）を超えたものは <b>0</b> でした。上位の候補は『単独列の残差の足し算』にすぎません（§4）。</li>")
    p.append("<li><b>B 手設計の掛け合わせ（as-of 目的変数統計 12 種 + 数値 30 種）: 基準を満たすものなし。</b>最も一貫したのは調教師×競馬場（発見・確認とも z=2.5〜4）ですが、検定数を考えると有意ではありません（§5）。</li>")
    p.append("<li><b>C 交互作用を表現できる NN（DCN-v2・FM・DeepFM）: 構造の上乗せはなし。</b>効いたのは『掛け合わせの構造』ではなく『NN という別系統 + レース内ソフトマックス』で、構造なしの MLP が最良でした。V5 の LightGBM 単体にブレンドすると勝率 −0.0013 ですが、<b>現行の合成（S0）に足すと勝率 −0.00022（上限 −0.00011）・連対と複勝は変化なし</b>、L3 を含む合成の上では −0.00014 で基準（0.0002）に届きません（§6）。</li>")
    p.append("<li><b>D 残差ブースティング: 現行の合成にはもう学べる構造がほぼ残っていない。</b>V5 の OOF ロジットを土台に浅い木（切り株 = 加法のみ / 深さ 3 = 3 列までの交互作用）を LOYO で学習すると、現行の合成（S0）に対しては深さ 3 でも勝率 −0.00002（上限 +0.00003）・連対 −0.00012・複勝 −0.00005。V5 の LightGBM 単体に対しては勝率 −0.00019（切り株）→ −0.00027（深さ 3）で、<b>交互作用で取れる分は −0.00008</b>、残りは加法で、合成が既に取っています（§8）。</li>")
    p.append("<li><b>結論: 掛け合わせに上乗せなし。</b>追加する列・表はありません。NN メンバーは現行の合成へ勝率のみ −0.0002 で、L3 と重なるため単独の採用基準に届かず、v5 最終学習で L3 を足すか決めるときに同条件の候補として残します（§9）。</li>")
    p.append("</ul></div>")
    # 2 method
    p.append("<h2>2. 方法</h2><ul><li><b>スコア検定</b>: 学習済みモデルの OOF 確率を offset として、列 z を 1 本足したときの対数尤度の傾き（score）を、再学習なしで測ります。勝ちはレース内ソフトマックス（offset = log p_win）、連対・複勝は二値ロジット。z = g/√I は『上乗せなし』で標準正規に近くなります。実装は有限差分と一致することをテストで確認（<code>tests/pipeline/test_tp58_score.py</code>）。レースの中で動かない列は 0 になるよう、レース内で中心化してから情報量を計算します。</li>")
    p.append("<li><b>発見期間と確認期間</b>: 候補の選別は 2022–23 年、確認は 2024 年で別に行い、選別バイアスを避けます。OOF が存在するのは 2022〜2024 年だけです。</li>")
    p.append("<li><b>多重検定</b>: 試した候補数を全て記録（A: 44,631、B: 12 種×4 統計 + 30 種）。帰無の分布が標準正規より広い（OOF の較正のずれ・列間の相関）ため、全候補の MAD から帰無のばらつきを推定して z を割り、そのうえで Bonferroni を使いました。</li>")
    p.append("<li><b>確認</b>: スコア検定で有望なものだけを、V5 と同じ設定の LightGBM（検証 2022/23/24・seed 0,1・同じパラメータ）に列として足して再学習し、<code>feature_bakeoff.compare_oof</code>（レース単位のブートストラップ。採用 = 勝率の差の上限 &lt; 0・効果 ≥ 0.0002・連対と複勝が悪化しない）で V5 と比べます。</li></ul>")
    # 3 prior
    p.append("<h2>3. 木の経路で続けて分割される組（SHAP 交互作用の代わり）</h2><p>V5 の列で LightGBM を 2021 年以前のデータだけで学習し、木の中で祖先と子の分割に現れる列の組を、子の分割 gain で重み付けして数えました（SHAP 交互作用値は計算量の都合で見送り、同じ目的の経路頻度で代替）。</p>")
    p.append(table(["組", "gain 合計", "回数"], [[f"{a} × {b}", f"{g:.0f}", n] for a, b, g, n in tp[:12]]))
    p.append("<p class='muted'>上位は前段モデルの出力 <code>mtl_t_rank_norm_gap</code>（最重要列）と、頭数・騎手数・クラス・性齢・調教・坂路の組です。つまり『前段の強さ × 状況』の形で木が分かれていますが、これらの積・差を明示列にしても上乗せは出ませんでした（次節）。</p>")
    # 4 discovery
    p.append("<h2>4. 自動の総当たり（積・差・対数比）</h2>")
    if S:
        rows = []
        for t in ("win", "top2", "top3"):
            rows.append([t, f"{S['null_sd_mad'][t + '_disc']:.2f}", f"{S['null_sd_mad'][t + '_conf']:.2f}", S[f"n_pass_{t}_disc_bonf"], S[f"n_pass_{t}_disc_conf"], S[f"n_pass_{t}_disc_conf_over_singles"]])
        p.append(table(["目的", "帰無のばらつき（発見）", "帰無のばらつき（確認）", "発見期間で Bonferroni 超え", "かつ確認期間で同符号 z&gt;2.58", "かつ単独列より強い"], rows))
        top = S["top30_by_disc"]["win"][:8]
        p.append("<h3>勝率で発見期間の上位（z は補正前）</h3>")
        p.append(table(["a", "操作", "b", "z 発見", "z 確認", "単独 a", "単独 b"], [[r["a"], r["op"], r["b"], r["z_win_disc"], r["z_win_conf"], r["za_win_disc"], r["zb_win_disc"]] for r in top], num_from=3))
        p.append("<p>上位は <code>wc_c1_best − fm_a_l3fz_ewm2_own_mean</code>（z=7.9）のような差ですが、<b>2 本それぞれの単独の z（+5.8 と −5.4）の足し算</b>です。補正後（÷1.96）は 4.0 で、Bonferroni の 5.08 に届きません。確認期間でも符号は同じ（+3.1）ですが、これは次の『単独列の残差』と同じ現象です。</p>")
    if len(sing):
        import numpy as np

        rows = []
        for c in ["tl_best_t1z_14", "tl_f_t1z", "wc_c1_best", "wc_c1_last", "wc_cc_last"]:
            if c in sing.index:
                rows.append([c] + [f"{sing.loc[c, k]:+.2f}" for k in ("win_disc", "win_conf", "top2_disc", "top2_conf", "top3_disc", "top3_conf")])
        p.append("<h3>副次の発見: 調教の時計の列は、単独列としての残差が一貫して残っている</h3>")
        p.append(table(["列", "勝率 発見", "勝率 確認", "連対 発見", "連対 確認", "複勝 発見", "複勝 確認"], rows))
        p.append("<p class='muted'>符号は発見・確認で一致します（調教の時計 <code>tl_*</code>・坂路/ウッド <code>wc_*</code> の列を、V5 がわずかに使い足りない向き）。これは掛け合わせではなく加法の主効果です。大きさは §8 の残差ブースティングで見積もります。</p>")
    # 5 hand
    p.append("<h2>5. 手設計の掛け合わせ</h2><h3>5.1 カテゴリの掛け合わせ（as-of 目的変数統計 + 階層縮約）</h3>")
    p.append("<p>『子キー（例: 騎手×コース）の過去の勝率・複勝率』を、親キー（例: 騎手）へ縮約した値と、その差（交互作用そのもの）。同日のレースは集計に入れません（brute-force 一致テストと、将来の着順を書き換えても過去行が変わらないテストで確認: <code>tests/pipeline/test_tp58_cross.py</code>）。父・母父は学習表で空だったため <code>ped_tbl</code> の世代 1・2 から作りました。</p>")
    if len(hand):
        rows = []
        for cr in hand["cross"].unique():
            sub = hand[hand["cross"] == cr].set_index("stat")
            for st in ("win_m", "win_dev", "top3_m", "top3_dev"):
                r = sub.loc[st]
                rows.append([cr if st == "win_m" else "", st, r["z_win_disc"], r["z_win_conf"], r["ez_win_disc"], r["ez_win_conf"], r["z_top3_disc"], r["z_top3_conf"]])
        p.append(table(["掛け合わせ", "統計", "勝率 z 発見", "勝率 z 確認", "親を除いた z 発見", "親を除いた z 確認", "複勝 z 発見", "複勝 z 確認"], rows, num_from=2))
        p.append("<p class='muted'>12 種 × 4 統計 × 3 目的 = 144 検定。帰無のばらつきが 1.4〜2 倍あるため、|z| が 3〜4 でも有意とは言えません。発見・確認の両方で |z|≥2.5 だったのは調教師×競馬場（勝率・複勝）だけです。</p>")
    p.append("<h3>5.2 数値の掛け合わせ（距離変化×前走着順、斤量×馬体重×距離、枠×脚質×頭数、年齢×間隔 など 30 種）</h3>")
    if len(named):
        mx = max(named[["z_win_disc", "z_win_conf", "z_top2_disc", "z_top2_conf", "z_top3_disc", "z_top3_conf"]].abs().max())
        both = named[((named["z_win_disc"].abs() > 2.5) & (named["z_win_conf"].abs() > 2.5)) | ((named["z_top3_disc"].abs() > 2.5) & (named["z_top3_conf"].abs() > 2.5))]
        p.append(f"<p>30 種を 3 目的で検定（90 検定）。<b>|z| の最大は {mx:.2f}</b>、発見・確認の両方で |z|&gt;2.5 になったものは {len(both)} 件でした。元の 2 列の一次項を除いた有効スコアでも同様です。</p>")
        rows = [[f"{r.a} {'×' if r.op == 'prod' else '/'} {r.b}", r.z_win_disc, r.z_win_conf, r.z_top3_disc, r.z_top3_conf] for r in named.itertuples()][:30]
        p.append(table(["組", "勝率 発見", "勝率 確認", "複勝 発見", "複勝 確認"], rows))
    # 6 NN
    p.append("<h2>6. 交互作用を表現できるモデル構造（NN メンバー）</h2>")
    p.append("<p>V5 と同じ入力（1,142 列 + 欠損指示 + 小さなカテゴリの one-hot = 1,660 次元）で、MLP（対照）・DCN-v2（クロス層 3）・FM（二次項のみ、線形 + FM）・DeepFM を学習。損失は勝ちがレース内ソフトマックス、連対・複勝が二値 CE。検証年ごとに『それ以前（検証の 10 日前まで）』で学習し、最後の 1 年で早期終了 → 最良エポックで全期間を学習し直し。ブレンド重みは検証年以外の 2 年で選ぶ（LOYO）。</p>")
    rows = []
    for tag, base in (("nn_blend", "V5 LightGBM 単体（r0 設定）"), ("nn_blend_lgb_r2", "V5 LightGBM 単体"), ("nn_blend_S0_r0", "現行の合成 S0（r0）"), ("nn_blend_S0_r1", "現行の合成 S0"), ("nn_blend_S0_r2", "現行の合成 S0（新 seed・広い MLP）"), ("nn_blend_S11f", "L3 を含む合成 S11f")):
        d = jload(res / f"{tag}.json")
        if not d:
            continue
        for v, r in d.items():
            c = r["compare"]
            rows.append([base, v, f5(c["Δwin"]), f5(c["win_hi"]), f5(c["Δtop2"]), f5(c["Δtop3"]), "採用" if c["採用"] else "—"])
    p.append(table(["ブレンド先", "NN", "Δ勝率", "上限", "Δ連対", "Δ複勝", "基準"], rows, num_from=2))
    p.append("<p class='muted'>Δ は (ブレンド − 先) の対数損失、負が改善。r0 = lr 1e-3・エポック 1〜2 で停止（過学習）、r1 = lr 3e-4・wd 1e-3・dropout 0.4・入力 dropout 0.2・隠れ 256-128・seed 4、r1s2 = r1 の別 seed、r2 = 隠れ 512-256・dropout 0.5・入力 0.3・seed 8。dcn/fm2/deepfm の _r1 は同じ正則化。試した NN 設定は r0 を含め 4 系統 × 4 種 + 組合せ。</p>")
    # 7 lgb
    p.append("<h2>7. LightGBM に列として足した確認</h2>")
    if lgb:
        rows = []
        for k, r in lgb.items():
            c = r["compare"]
            rows.append([k, r["n_added"], f5(c["Δwin"]), f5(c["win_hi"]), f5(c["Δtop2"]), f5(c["Δtop3"]), "採用" if c["採用"] else "—"])
        p.append(table(["追加", "列数", "Δ勝率", "上限", "Δ連対", "Δ複勝", "基準"], rows, num_from=1))
    else:
        p.append("<p class='muted'>（実行待ち／未完了）</p>")
    # 8 resid
    p.append("<h2>8. 残差ブースティング（V5 の OOF にまだ学べる構造があるか）</h2>")
    p.append("<p>V5 の OOF ロジットを init_score にして、浅い木のブースティングを『検証年以外の 2 年』で学習し検証年に当てる（LOYO）。深さ 1（切り株 = 列ごとの加法効果のみ）と深さ 3（3 列までの交互作用）の差が『交互作用で取れる分』です。ラウンド数は学習 2 年の内側で決め、検証年は見ません。</p>")
    if resid:
        rows = []
        for base, rr in resid.items():
            for k, r in rr.items():
                rows.append([base, k, f5(r["delta"]), f5(r["hi"]), ", ".join(f"{y}:{v['rounds']}R/{v['delta']:+.5f}" for y, v in r["per_year"].items())])
        p.append(table(["残差を取る先", "目的_構造", "Δ（新−先）", "上限", "年ごと（ラウンド/Δ）"], rows, num_from=2))
    else:
        p.append("<p class='muted'>（実行待ち／未完了）</p>")
    p.append("<h2>9. 判定と v6-features との統合方針</h2>")
    if lgb:
        c = next(iter(lgb.values()))["compare"]
        lgb_txt = f"LightGBM の確認（上位 40 組 + 手設計 60 列を同時に追加）は 勝率 {f5(c['Δwin'])}（上限 {f5(c['win_hi'])}）・連対 {f5(c['Δtop2'])}・複勝 {f5(c['Δtop3'])} で、基準を{'満たしました' if c['採用'] else '満たしませんでした'}（§7）。"
    else:
        lgb_txt = "LightGBM での確認は実行待ちで未了です（§7）。"
    p.append("<div class='key'><ul>")
    p.append("<li><b>掛け合わせの列: 採用なし。</b>自動の総当たり（44,631 候補）・手設計のカテゴリ交差（12 種）・手設計の数値交差（30 種）のいずれも、発見期間で多重検定を超えるものがなく、残差ブースティング（深さ 3）でも現行の合成に勝率 −0.00002（上限 +0.00003）しか残っていません。最も一貫した調教師×競馬場（発見・確認とも z=2.5〜4）は、検定数と帰無の広がりを考えると有意とは言えません。" + lgb_txt + "</li>")
    p.append("<li><b>NN メンバー: 掛け合わせの構造（DCN・FM）は効かず、効くのは別系統としての MLP のみ。</b>現行の合成（S0）に足すと勝率 −0.00018〜−0.00026（上限 −0.00008〜−0.00013）で連対・複勝は変わらない（新 seed でも再現）。ただし L3 を含む合成（S11f）の上では −0.00014〜−0.00015（基準 0.0002 未満）で、L3 と同じ信号に重なります。tp-51 の結論『別系統メンバーは足しても効果なし』と矛盾せず、NN は勝率の 0.0002 を L3 と分け合う関係です。<b>採用しない</b>（v5 最終学習で L3 の採否を決めるとき、重みつき MLP 1 本を同条件の候補にする）。</li>")
    p.append("<li><b>v6-features との統合方針</b>: 追加採用がないため、v6 の列に足す作業はありません。将来採用する場合は、積・差・対数比の列 <code>tp58_pairs.pair_column</code>（定数は 2021 年以前の行のみ、列名 <code>x58_&lt;op&gt;__&lt;a&gt;__&lt;b&gt;</code>）と as-of 交差 <code>tp58_cross.shrunk_cross</code>（列名 <code>x58_&lt;名&gt;_&lt;目的&gt;_{m,dev,n}</code>）を v6 の表に結合するだけで、列名の接頭辞 <code>x58_</code> は v6 の列と衝突しません。</li>")
    p.append("<li><b>副次の知見（掛け合わせではない）</b>: 調教の時計（<code>tl_*</code>・<code>wc_*</code>）の単独列に、発見・確認で同符号の残差が残る（§4）。残差ブースティングの切り株（加法）で −0.00019 取れるのは、この種の主効果が V5 の LightGBM 単体にはまだ残っていることを示しますが、合成には入っています。調教の列を作り直す v6 側の仕事で扱うのが筋です。</li>")
    p.append("</ul></div>")
    p.append("<h3>試した数（多重検定の分母）</h3>")
    p.append(table(["段", "試した数", "備考"], [["A 総当たり（積・差・対数比）", "44,631 候補 × 3 目的 = 133,893 検定", "基 210 列（重要度上位 150 + レース内偏差 60）。Bonferroni z=5.08"], ["B1 カテゴリ交差", "12 種 × 4 統計 × 3 目的 = 144 検定", "親との差・縮約平均 × 勝率/複勝率"], ["B2 数値の手設計", "30 種 × 3 目的 = 90 検定", "積/対数比"], ["C NN", "設定 4 系統 × 4 構造 + 組合せ、ブレンド先 3 通り", "r0, r1, r1s2, r2"], ["D 残差ブースティング", "2 土台 × 3 目的 × 2 構造", "上位 150 列"], ["確認 LightGBM", "1 回（上位 40 組 + 手設計 60 列を同時に追加）", "V5 と同設定"]], num_from=9))
    p.append("</body></html>")
    return "".join(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    a = ap.parse_args()
    OUT.write_text(build(Path(a.res)), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
