# 市場と独立の列（穴馬 AI の MF 版・2026-11-27。model-relabel にも渡す）

## 結論

- めぐ AI の v5 の 1,083 列に、市場（オッズ・人気）に由来する列は無かった。判定は作りのコードを読んで決めた（market 0・suspect 0）。
- v5 は、名前の正規表現 `odds|pop|ninki|tansho|fukusho|payoff|market|ret_` で市場の列を既に外している（`src/research/megu_cycles/mc_data.py` の `_rich_matrix`）。それをすり抜けた列は無かった。
- 主の MF 版は「v5 − 追い切り評価の 3 列（oikiri_eval・oikiri_eval_rank・oikiri_eval_gap）」とする。
  - 追い切り評価は netkeiba の A〜D で、市場ではない。ただし公開の専門家の意見で、人気を通して市場と重なりうる。ユーザーの指定「市場のオッズを反映しそうな領域は確実に反映されないように」に合わせ、方針で外す。
- MF − M（v5 のまま）の差は、この 3 列の分だけ。もともと小さい見込み。差が 0 でも「市場と独立」の主張が通ったとは読まない（v5 は既に名前で外している）。
- 列ごとの判定は `market_audit.csv`（column・family・verdict・source_path_line・reason。1,083 行）。

## オッズの扱い（入力ではなく、学ぶ馬の選び）

- 母集団の絞り込み（単勝 10 倍以上）にはオッズを使う。これは「どの馬を学ぶか」の選びで、入力の列ではない。
- 母集団の中のオッズの高さ（10〜15 倍と 50 倍以上の違い）は、どの列にも入れない。入力の列はすべて上のとおり市場を読まない。
- 市場は学習の後に、Benter の 2 段（副の記述）と、比べ先（母集団の中の最低オッズ）・A/E の分母（較正した市場の q3）だけで使う。

## 疑われた族の判定

| 族 | 判定 | 根拠 |
|---|---|---|
| `*_res`・`c_res_*`・`e_res_*` | clean | Elo の S−E（着順のスコア − Elo の期待スコア）。市場の見込みからの残差ではない（horse_history/elo.py:16-17,46） |
| `o_*`・`ob_*` | clean | 馬主・生産者の成績。`o_res` は着順の正規化 0.5−(着順−1)/(頭数−1)（factory_conn.py:68）。入力に odds は無い（:20-22） |
| `f3_E` | clean | Elo の期待スコア（match.py:13） |
| `track_bias_*` | clean | 勝ち馬の上がり 3F の当日・前日の値（track_bias_pedigree.py:408,453-504）。確定オッズの S−E を使うトラックバイアスの画面の列とは別物 |
| `fc_*` | clean | 騎手・調教師の通算成績（着順だけ）のレース内の集計。人気の重みは無い（factory_field.py:125-142） |
| `jockey_*`・`trainer_*`・`horse_prior_*` | clean | 勝ち・複勝・件数だけ（historical_entity_stats.py:54-87・serving_layer_a.py:100-129）。出馬表の odds・popularity は layer_a_dataset.py:132-137 で落とす |
| `exp_*` | clean | 同じ条件の出走数・勝ち数（factory_self.py:386-405）。期待値ではない |
| `s_*` | clean | 前段のモデル。目的は走破時計の偏差と正規化着順（stage_models.py:73-74）。入力に市場の列は無い（v6_stage.py:66-91） |
| `z*`・`pxc_*`・`pxe_*`・`pd_*`・`ped_*` | clean | 血統の列。目的は着順（tp38_ridge.py:114,119-133 ほか） |
| `sr_`・`a_`・`b_`・`f2_`・`f5_`・`pc_` | clean | 時計・上がり・通過順・ラップと Elo（horse_history/common.py:22 の RUN_COLS に odds は無い） |
| `oikiri_*` | 方針で外す | 公開の専門家の意見（上の結論） |

- 確かめきれていない所（推測）: LAP のモデル本体は宣言と出力のスキーマだけで確かめた。`tl_*`（s_ の入力）の作りは読んでいない。どちらも判定は変わらない見込み。
- v5 の外で市場に近いもの（v5 には入っていない）: tp30_volatility の `mk_*`（確定オッズの構造）・horse_ease.parquet の odds 列・X_xgb の v5 以外の 148 列（tl_・wc_・mtl_・rb_ など。対象外として調べていない）。

## 判定の規則（model-relabel にも渡す）

1. 列の名前でなく、作りのコードが実際に読む列（RUN_COLS・HIST_COLS・`columns=`）で判定する。元の表（race_result_flat）には odds・popularity・payoff があるので、読み込みの一覧に無ければ clean。
2. res・E・exp・residual は「何からの見込みか」で決める。Elo・レーティング・基準タイム・着順の正規化なら clean。オッズ・人気・予測オッズ q̂ からなら market。
3. 上流のモデルの列（s_・EASE2 など）は、入力と目的変数をあわせて判定する。列を絞らない merge は、そのファイルの全列を入力とみなしてスキーマを確かめる。
4. レース内の集計（fm_・fc_ など）と相対化（_rz・_rrank・_rgap）は、元の列の判定を受け継ぐ。元が market なら market。
5. 接頭辞の衝突に注意する。o_ は馬主で odds ではない。exp_ は経験数で期待値ではない。族は作りの関数で特定する。
6. 人気との相関の高さは根拠にしない。能力の列は作りの上で人気と相関する。
7. 学習側と推論側の両方で式を確かめる。片方しか読めないときは推測として残す。
8. 市場ではない公開の意見（追い切り評価・予想印など）は market と分け、方針で扱いを決める（この研究では外す）。
9. 着順を使った重要度・相関で列を選び直さない（la-2 の決定。2019〜21 の着順で列を選ぶと選び直しになる）。
