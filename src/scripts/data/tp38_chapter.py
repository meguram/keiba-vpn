"""tp-38 の章（docs/html/modeling/chapters/pedigree_position_cross.html）を、評価結果の JSON から生成する。

python -m src.scripts.data.tp38_chapter <結果ディレクトリ>   # ホストの tp38 出力（stage1/stage2/phi_summary.json など）を手元にコピーしたもの
"""

from __future__ import annotations

import html as H
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHAPTER = ROOT / "docs/html/modeling/chapters/pedigree_position_cross.html"
HEAD_FROM = ROOT / "docs/html/modeling/chapters/lap_groups_and_trainer_stats.html"
TITLE = "血統の位置・存在割合・クロス・類似種牡馬（tp-38）"

DESC = {
    "PX_cell_sd": "祖先 × 位置の区分 × (面×距離帯) の産駒成績を階層縮約し、位置の重み 2^-gen で合算（17 列）",
    "PX_cell_vn": "同上、条件を (面×競馬場)",
    "PX_cell_gd": "同上、条件を (面×馬場の良/悪)",
    "PX_cellw_sd": "同上、成績を勝ったかどうか（1/0）にした版",
    "PX_cell_all": "同上、条件なし（位置の区分だけ）",
    "PX_cell_sf": "同上、条件を面（芝/ダート）だけ",
    "PX_cross": "近交係数（Wright の経路和）、クロスの数・強さ、クロス祖先の『クロスしたときの上乗せ』と条件別の適性（19 列）",
    "PX_emb": "血統表の TF-IDF ベクトルを SVD で圧縮した座標の上位 16 次元（静的）",
    "PX_emb32": "同上、32 次元",
    "PX_clu": "血統クラスタ（k=64, 256）× 条件の産駒成績",
    "PX_sim": "父・父父・母父の産駒成績を、血統構成が近い種牡馬の産駒成績で縮約（42 列）",
    "PX_ridge_fix": "リッジ回帰（祖先の効果 + 条件別の上乗せ）、位置の重みを理論値 2^-(gen-1) に固定（10 列）",
    "PX_ridge_als": "同上、位置の重み φ をデータから推定（交互最小二乗）",
    "PX_ridge_als2": "同上、φ の推定を理論値へ強く縮約した版",
    "ALL_BEST": "各系統で第 1 段が最良だった PX_cell_sd + PX_emb + PX_ridge_als の同時追加",
    "ALL_PASSED": "第 1 段を通過した全群の同時追加",
}
ORDER = ["PX_cell_sd", "PX_cell_vn", "PX_cell_gd", "PX_cell_sf", "PX_cellw_sd", "PX_cell_all", "PX_cross", "PX_emb", "PX_emb32", "PX_clu", "PX_sim", "PX_ridge_fix", "PX_ridge_als", "PX_ridge_als2"]


def jl(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def f5(x, nd=5):
    return "-" if x is None or x != x else f"{x:+.{nd}f}"


def build(d: Path) -> str:
    s1, s2 = jl(d / "stage1/results.json"), jl(d / "stage2/results.json")
    g1, g2 = s1.get("groups", {}), s2.get("groups", {})
    phi, sim, meta, scr = jl(d / "phi_summary.json"), jl(d / "simcheck.json"), jl(d / "static/static_meta.json"), jl(d / "screen.json")
    pos = jl(d / "position.json")
    fin = jl(d / "final.json")
    adopted = [g for g, v in g2.items() if v.get("vs_V4", {}).get("採用")]
    rows = ""
    for g in ORDER + ["ALL_BEST", "ALL_PASSED"]:
        a, b = g1.get(g), g2.get(g)
        if not a and not b:
            continue
        ca, cb = (a or {}).get("vs_V4", {}), (b or {}).get("vs_V4", {})
        rows += (f"<tr><td>{g}</td><td>{H.escape(DESC.get(g, ''))}</td><td class=num>{(a or b)['cols']}</td>"
                 f"<td class=num>{f5(ca.get('Δwin'))}（{f5(ca.get('win_hi'))}）</td><td class=num>{f5(ca.get('Δtop3'))}</td>"
                 f"<td class=num>{(f5(cb.get('Δwin')) + '（' + f5(cb.get('win_hi')) + '）') if b else '-'}</td><td class=num>{f5(cb.get('Δtop2')) if b else '-'}</td><td class=num>{f5(cb.get('Δtop3')) if b else '-'}</td>"
                 f"<td>{('採用' if cb.get('採用') else '不採用') if b else ('第 1 段で不通過' if a and not (a['vs_V4']['Δwin'] < -0.00012 and a['vs_V4']['win_hi'] < 0.0001) else '第 2 段待ち')}</td></tr>")
    n_groups = len([g for g in g1 if g != "ALL_PASSED"])
    ph_rows = ""
    tot = {v: sum(phi.get(f"cov_{v}", [0])) or 1.0 for v in ("fix", "als", "als2")} if phi else {}
    for r in phi.get("positions", [])[:14]:
        ph_rows += (f"<tr><td>{r['gen']}</td><td>{r['name']}</td><td class=num>{r['prior']:.3f}</td><td class=num>{r['phi_als']:.2f}</td><td class=num>{r['phi_als2']:.2f}</td>"
                    f"<td class=num>{r['cov_fix'] / tot['fix']:.3f}</td><td class=num>{r['cov_als'] / tot['als']:.3f}</td></tr>")
    sh_rows = ""
    if phi:
        import numpy as np

        gen = np.concatenate([np.full(2 ** g, g) for g in range(1, 6)])
        side = np.concatenate([np.arange(2 ** g) >> (g - 1) for g in range(1, 6)])
        for v, nm in (("fix", "固定（理論値 2^-(gen-1)）"), ("als", "推定（交互最小二乗）"), ("als2", "推定（理論値へ強く縮約）")):
            c = np.array(phi[f"cov_{v}"])
            sh_rows += f"<tr><td>{nm}</td>" + "".join(f"<td class=num>{c[gen == g].sum() / c.sum():.3f}</td>" for g in range(1, 6)) + f"<td class=num>{c[side == 0].sum() / c.sum():.3f}</td><td class=num>{c[side == 1].sum() / c.sum():.3f}</td></tr>"
    sim_rows = "".join(f"<tr><td>{H.escape(o['stallion'])}</td><td class=num>{o['runs']}</td><td>" + "、".join(f"{H.escape(n['name'])}（{n['sim']}）" for n in o["neighbors"][:4]) + "</td></tr>" for o in sim[:16])
    nbm = meta.get("neighbors", {})
    sire_n, sire_nb = nbm.get("sire", {}).get("stallions", "-"), nbm.get("sire", {}).get("with_neighbors", "-")
    pos_rows = ""
    for ref, lst in pos.items():
        for r in lst:
            pos_rows += f"<tr><td>{'父' if ref == 'sire' else '母父'}</td><td>{r['gen']} 世代 側{r['side']}{'・父系直系' if r['tm'] else ''}{'・母系直系' if r['tf'] else ''}</td><td class=num>{r['n_anc']}</td><td class=num>{r['slope']:+.2f}</td><td class=num>{r['corr']:+.2f}</td></tr>"
    ab = g2.get("ALL_BEST", {}).get("vs_V4", {})
    next_txt = ""
    if ab:
        next_txt = (f"<p>3 系統（PX_cell_sd・PX_emb・PX_ridge_als、43 列）の同時追加 ALL_BEST は、勝率 Δ{f5(ab['Δwin'])}（95% CI 上限 {ab['win_hi']:+.1e}）、連対 Δ{f5(ab['Δtop2'])}、複勝 Δ{f5(ab['Δtop3'])}（上限 {f5(ab['top3_hi'])}）。"
                    "勝率の CI 上限が 0 にわずかに届かず（採用基準は上限 &lt; 0）、<b>採用基準は満たさない</b>。複勝は有意に改善しているが、単独の群は第 2 段で再現しなかった（下表）ので、3 系統が部分的に同じ情報を拾って合算で効いている可能性と、valid 2024 の偶然の可能性を分けられていない。</p>")
    summary = ("第 2 段（valid 2024・新 seed）で採用基準を満たした群はありませんでした。V4 には何も足しません。" if not adopted and g2 else (f"採用: {', '.join(adopted)}。" if adopted else "第 2 段は未実施です。"))
    if fin:
        summary += " " + fin.get("summary", "")
    head = HEAD_FROM.read_text(encoding="utf-8").split("<body>")[0]
    import re

    head = re.sub(r"<title>.*?</title>", f"<title>{TITLE} — keiba-vpn</title>", head)
    body = f"""<body><a href="../../index.html" class="muted">← ドキュメント一覧</a><h1>{TITLE}</h1>
<p class="muted">血統表のどの位置に誰がいるか（存在割合）、同じ祖先が両側に出るか（クロス）、構成の似た種牡馬は同じように効くか、を統計量の出し方から設計し、すべて『そのレース日より前の産駒成績だけ』で作って、V4（1,062 列の基準）への増分を測った。キーは馬・祖先の ID。市場オッズ・人気は使っていない。</p>
<div class="note"><b>要点.</b> {summary}</div>

<h2>1. 既存の資産と、今回の未実施部分</h2>
<table><tr><th>既存（V4 に入っている / 不採用だった）</th><th>今回の新規</th></tr>
<tr><td><b>PED_sire_cond</b>（採用済み）: 父・母父・父父・母母の ID × 条件（距離帯・面・馬場・競馬場）の産駒成績。位置は 4 か所に固定。</td><td>全ての祖先（5 世代 62 位置）を対象に、位置の区分ごとの効果を分けて推定し、合算する。</td></tr>
<tr><td>PED_anc_apt（G4・不採用）: 祖先の産駒の平均成績を世代ごとに集約。条件なし・位置は世代だけ。</td><td>条件別・位置の区分別（父系/母系の直系を含む 18 区分）・階層縮約・馬自身を引く。</td></tr>
<tr><td>PED_blood（G4・不採用）: 上位 30 祖先の血量、血の集中度。</td><td>近交係数（経路和）とクロス祖先の上乗せ。</td></tr>
<tr><td>PED_nick（父×母父の相性）、半きょうだい（G4・不採用）</td><td>血統空間の座標と、構成が近い種牡馬による縮約。</td></tr>
<tr><td>血統適性システム（研究層: 役割別 lift・maternal_deep・L2 クラスタ）</td><td>V4 の学習表には入っていない。今回の血統クラスタ（k-means）は PX_clu で比較した。</td></tr></table>

<h2>2. 設計した統計量</h2>
<h3>2.1 位置と存在割合</h3>
<p>祖先 a が馬 h の血統表の世代 g・位置 p にいるとき、遺伝子の期待割合は 2^-g（同じ祖先が複数回出るときは合算）。位置は 62 か所だが、推定を安定させるため <b>18 の区分</b>にまとめた: 父・母（gen1）、父父・父母・母父・母母（gen2）、gen3〜5 は (世代, 父側/母側, 直系か) の 4 通りずつ。直系 = 父父父…（tail-male）か 母母母…（tail-female）。理論上、祖先の効果は直接の産駒に対して 1、孫 1/2、曾孫 1/4…と薄まる（産駒が祖先の遺伝子の半分、孫が 1/4 を持つため）。ただし父と母父は同じ世代でも異なる経路・異なる育ち方をするので、位置ごとの重みはデータで確かめる（4 節）。</p>
<h3>2.2 F1: 祖先 × 位置の区分 × 条件の階層縮約（PX_cell_*）</h3>
<p>走 i の成績 y（レース内の着順を [−0.5, 0.5] にした値。レース内の平均は 0）について、</p>
<ul><li>μ_c = 条件 c の全走の平均（その日より前、全体平均へ縮約）</li>
<li>m_a = (S_a,c − S_own + k₁·μ_c) / (N_a,c − N_own + k₁)（祖先 a の産駒の条件 c での成績、k₁=30）</li>
<li>m_a,p = (S_a,p,c − S_own + k₂·m_a) / (N_a,p,c − N_own + k₂)（さらに位置の区分 p を分ける、k₂=30）</li>
<li>馬の特徴量 = Σ_(a,p) w_(a,p) · (m_a,p − μ_c) / Σ w（w は 2^-gen の合計）。世代別の平均、父側・母側、直系、信頼度で重み付けた版、位置固有の上乗せ（m_a,p − m_a）なども作る。</li></ul>
<p>S・N は『その日より前の走』だけの合計・件数、S_own・N_own はその馬自身の過去走（同じ条件）で、祖先の成績に自分が混ざらないよう引く（引くのは日付が前の走だけなので、今回の結果は入らない）。条件は (面×距離帯)・(面×競馬場)・(面×馬場)・面・なし を比べた。</p>
<h3>2.3 F2: リッジ回帰と位置の重みの推定（PX_ridge_*）</h3>
<p>y_i ≈ Σ_p φ_p · (δ_G[a(p)] + δ_C[k(i), a(p)])。δ_G は祖先の全体の効果、δ_C は条件 k（面×距離帯）ごとの上乗せ（δ_G より強く縮める）、φ_p は位置 p の重み。φ を理論値に固定して δ をリッジで解き（PX_ridge_fix）、δ を固定して φ を 62 次元の回帰で解く、を 3 回繰り返す（PX_ridge_als。φ は理論値へ縮約。als2 は強く縮約）。作り直しは半年ごと（2021-01-01 から）。<b>馬を 3 つの fold に分け、その fold の馬を除いた走で学習したモデルで予測する</b>ので、その馬自身の過去の結果は δ に入らない（リッジが血統の組み合わせから馬を『覚えて』しまうのを防ぐ）。</p>
<h3>2.4 クロス（PX_cross）</h3>
<p>近交係数 F = Σ 0.5^(g₁+g₂−1)（同じ祖先が父側の g₁ 世代・母側の g₂ 世代に出るとき、全ての組で足す。祖先自身の近交は無視）。クロスした祖先の数、最大の強さ、最も近い世代、片側で重複する量、クロスの強さの帯ごとの全体成績、クロス祖先ごとの『クロスしたときの上乗せ』（(祖先, クロス) の産駒成績 − (祖先) の産駒成績）と、クロス祖先のその条件での適性を作った。</p>
<h3>2.5 血統空間と類似種牡馬（PX_emb / PX_clu / PX_sim）</h3>
<p>馬の血統表を祖先ベクトル（Σ 2^-gen、IDF で希少な祖先を重く）にして SVD（32 次元）、k-means（64 / 256）。<b>種牡馬の類似度</b>は、その種牡馬自身の血統表（産駒の血統表の中の位置から復元。相対世代 1〜4、重み 2^-j）のコサイン類似度で、父・父父・母父それぞれの役割で産駒を持つ種牡馬の中から上位 10 頭（類似度 0.2 以上）を近傍とする（カーネル κ = sim²）。産駒の少ない種牡馬の成績は、近傍の産駒成績のカーネル加重平均を縮約先にした経験ベイズで補う: m̃ = (S_自分 − S_own + k·m_近傍) / (N_自分 − N_own + k)。父として産駒を持つ {sire_n} 頭の種牡馬のうち、近傍が付いたのは {sire_nb} 頭。</p>

<h2>3. 類似種牡馬の検出例（人が見て妥当かの確認用）</h2>
<p>産駒数の多い種牡馬の最近傍（コサイン類似度）。全兄弟（同じ父母）は類似度 1.0 になる。</p>
<table><tr><th>種牡馬</th><th>走数</th><th>近い種牡馬（類似度）</th></tr>{sim_rows}</table>

<h2>4. 位置ごとの重み（どの位置がどれだけ効くか）</h2>
<p>リッジの交差予測（学習に使っていない馬・日付）の予測と実際の成績との共分散のうち、各位置が占める割合。2024 年以降の作り直し（学習データが多い）の平均。φ は父 = 1 に正規化した推定値（φ は祖先ごとの縮約の強さと対になる量なので、母系の位置では大きく出やすく、寄与の割合の方を見るのが確実）。</p>
<table><tr><th>世代</th><th>位置</th><th>理論値 2^-(g-1)</th><th>φ（推定）</th><th>φ（強く縮約）</th><th>寄与の割合（固定）</th><th>寄与の割合（推定）</th></tr>{ph_rows}</table>
<table><tr><th>φ の扱い</th><th>1 世代</th><th>2 世代</th><th>3 世代</th><th>4 世代</th><th>5 世代</th><th>父側</th><th>母側</th></tr>{sh_rows}</table>
<p class="muted">参考: 祖先 a が『直接の父として持つ効果』を軸に、同じ祖先が別の位置にいるときの産駒平均成績の回帰の傾き（件数の重み付き）。理論の減衰（2 世代 0.5、3 世代 0.25）より小さく出るのは、軸の側の測定誤差による希釈を含むため（大まかな目安）。</p>
<table><tr><th>軸</th><th>位置の区分</th><th>祖先数</th><th>傾き</th><th>相関</th></tr>{pos_rows}</table>

<h2>5. 特徴量としての増分（V4 全体に対する勝率 LogLoss の差、95% CI 上限）</h2>
<p>第 1 段 = valid 2022+2023・seed 1 本、第 2 段 = valid 2024・新 seed 2 本（通過した群と、その同時追加）。採用基準: 勝率の CI 上限 &lt; 0、効果 0.0002 以上、連対・複勝の悪化 0.0002 以下。試した群は {n_groups} 群。</p>
<table><tr><th>群</th><th>内容</th><th>列数</th><th>第 1 段 Δwin（上限）</th><th>Δtop3</th><th>第 2 段 Δwin（上限）</th><th>Δtop2</th><th>Δtop3</th><th>採否</th></tr>{rows}</table>
<p class="muted">注意: 第 1 段の基準は valid 2022+2023 で選んだ値なので選択バイアスを含む（第 2 段で確かめる）。複数の群を試しているため、1 群だけが通過した場合は偶然の可能性を割り引く。</p>

<h2>6. 採否と次の一手</h2>{next_txt}
<ul><li><b>V4 には何も足さない</b>（採用基準を満たした群なし）。</li>
<li><b>確かめる順番</b>: ALL_BEST を、次の再学習（v5）の候補群と一緒に累積（tp-27）とアンサンブル（LOYO）で評価する。ここで勝率の上限が 0 を下回り、複勝の改善も保たれれば採用する。</li>
<li><b>採用する場合の推論側の状態量</b>（V4 のサービング状態に足す）: ① PX_cell_sd: (祖先, 位置の区分, 条件) 約 34 万セルと (祖先, 条件) 約 21 万セルの (合計, 件数)（約 10 MB）、馬ごとの条件別の過去走の (合計, 件数)（既存の馬の履歴に同居）。レース確定ごとに出走馬 1 頭あたり約 50 セルを加算。② PX_emb: 祖先の IDF（3.8 万）と SVD の成分（3.8 万 × 32、約 5 MB）。出走馬の 5 世代血統から 16 次元に射影するだけ（静的）。③ PX_ridge_als: δ_G（約 1.4 万）・δ_C（11 条件 × 約 3,300）と φ（62）を、馬の fold ごと（3 通り）に半年ごとに作り直す（約 1 MB）。作り直しは学習表の更新ジョブで行い、予測時は 62 位置の内積のみ。</li>
<li>出走馬の 5 世代血統（horse_pedigree_5gen）が前日夕方までに取れている前提（ped_tbl の増分と同じ経路）。無い馬は特徴量が欠損になる。</li></ul>

<h2>7. 先読みの防止</h2>
<ul><li>全ての成績の統計は『日付が厳密に前』の走だけ（同じ日・同じレースの他の馬は入らない）。累積和の引き算で自分の今回の結果の端数が残らないよう、馬自身の過去走の合計は別に累積して引く。</li>
<li>リッジは作り直し日より前の走だけで学習。馬の fold 分けは馬 ID から決まる静的な割り当てで、成績は使わない。</li>
<li>血統空間・クラスタ・種牡馬の近傍は血統表だけから作る静的な量で、成績を使わない。</li>
<li>テスト: 定義との総当たり一致、1 レースの結果をずらしても『そのレースの日より前の行』『そのレース自身の行』の特徴量が bit 単位で変わらないこと（F1・クロス・クラスタ・類似・リッジ）。</li></ul>
</body></html>"""
    return head + body


def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tp38_local")
    CHAPTER.write_text(build(d), encoding="utf-8")
    print("章を書いた:", CHAPTER)


if __name__ == "__main__":
    main()
