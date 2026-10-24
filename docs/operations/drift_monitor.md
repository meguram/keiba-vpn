# ドリフトの週次監視（直近 8 週の「予測 vs 結果」と、これまでの予測分布）

2026 年はテストセット②として持つ（学習・検証・選択・再学習に含めない）。その前提で、**毎週、週末の結果（着順・払戻・確定オッズ）が揃った後に 1 回**、前週分を足して再計算し、ドリフトが起きていないかを見る。
診断の結果と原因は [ドリフト診断の章](../html/modeling/chapters/drift_diagnosis.html)（`docs/html/modeling/chapters/drift_diagnosis.html`）。

**この監視の指標は『報告・警報のみ』に使う。モデル・閾値・特徴量の選択や学習には使わない**（`config/drift_periods.json` の `usage`）。モニタ画面（/monitor）に載せるのは別作業で、
データは `data/local/meta/drift_monitor/` に出す（latest.json / latest.html / history.jsonl / reports/<時刻>/ / backtest/<時刻>/）。

## 期間の定義（`config/drift_periods.json`）

| 区分 | 期間 | 役割 |
|---|---|---|
| train | 2020-01-01〜2025-12-31 | v4/v5 の最終バンドル（`bundle_*_2025`）の学習 |
| test① | 2025 年 | v1〜v5 の評価に使った（`bundle_*_2024` の 1 回評価）。選択には使わない |
| test② | 2026 年 | 学習・検証・選択・再学習に含めない。ドリフト検知と独立確認用（モニタ用の閲覧のみ） |
| monitor | 2027-01-01〜 | テストセット③の候補。2026 年を再学習に含めない限り 2027 年も学習外。含めたら `train.to` を更新し、`test2`・`monitor` の区切りを動かす |
| baseline | 2022-01-01〜2025-12-31 | 管理図の基準（モデル−市場の差の窓ごとの変動）。2022〜2024 年は LOYO の OOF、2025 年は `bundle_*_2024` の 1 回評価 |

2027 年をテストセット③として扱うときは、`config/drift_periods.json` の `monitor`・`prediction_sources` だけを更新する。コードは変えない。

## 何を見るか

**主指標は『モデルと市場（確定単勝オッズ）の勝率 LogLoss の差』**（レースの難しさの変動を打ち消す）。直近 8 週のローリング窓と、監視開始（2026-01）からの累積で測る。

### ラベルあり（結果が揃った週）

| 指標 | 内容 |
|---|---|
| 差（勝率） | モデル LL − 市場 LL。窓の値に、レースでクラスタした標準誤差とブートストラップ 95% 区間を併記。窓内が `min_races_window`（150）レース未満は判定保留 |
| 差（連対・複勝） | 市場の勝率から Harville で作った連対・複勝を参考の基準にした差 |
| LogLoss・Brier・ECE・較正曲線 | 勝率・連対・複勝。較正曲線は十分位（基準期間と直近窓を重ねる）。logit 傾き |
| 的中の順位相関 | レースごとの、予測が高い順と着順のスピアマン相関（モデル／市場）。その差 |
| 予測の尖り | モデルの勝率の対数オッズの SD ÷ 市場のそれ（`sd_ratio`）。低下は入力の情報の欠落を疑う |
| 管理図 | 窓の差（基準の平均と ±2σ・±3σ）、週次の差の EWMA（λ=0.2）と CUSUM（k=0.5, h=5） |
| 区分 | `ばらつきの範囲`（窓の差 ≤ 基準+2σ）／`境界`／`有意な悪化`（窓の差−2×SE が基準の平均を超え、かつ基準+3σ 超え）／`保留` |
| 基準モデルとの比較 | 結果の表だけから作った固定の簡素なモデル（`drift_benchmark.py`、2021-2024 年で学習、調教・血統・オッズなし）の『市場との差』の直近窓と、2025 年の同じ月との変化。**これが動いていれば市場側・世の中の変化、動かず自前の差だけが上がっていれば自前のモデル・入力側**（2026 年は 6〜10 月の変化 +0.0025±0.0023、直近窓 −0.0002±0.0035 で動いていない） |
| 層別 | 芝/ダ・距離帯・頭数・競馬場・新馬/未勝利・年齢・月。直近窓の差を基準期間の同じ層と比べる片側 z 検定を BH で FDR 補正（q<0.10 を強調）。**参考扱い**（多重比較のため原因の手がかり） |

### ラベルなし（先行指標。結果が出る前にも出せる）

| 指標 | 内容 |
|---|---|
| 予測分布 | 勝率・複勝率の PSI・KS・分位点、最大勝率、エントロピー、対数オッズ SD、頭数別。基準＝学習の OOF と test①。最大勝率・SD・エントロピーは基準の窓の ±3σ で見る |
| 入力データの質 | 初出走の馬・学習期間に出ていない騎手/調教師の割合。**調教ページの被覆**（保存記録に『そのレース日の調教の行』がある出走馬の率。学習側はほぼ 100%、下限 0.90） |
| 特徴量の PSI | 上位 50 列（v5 バンドルの木モデル gain）の PSI と欠損率の跳ね。抜き取った日の特徴量を、2025 年の同じ推論経路の抜き取りと比べる（`scripts/cron/drift_monitor_weekly.sh` が抜き取る） |
| v4/v5 shadow | `predictions_v3` と `predictions_v5` を同じ馬で突き合わせ、LogLoss の差（v5−v4）の累積と直近窓。約 2,000 レースまで区間は 0 を跨ぐ |

## 警報の段階と対処

窓の指標は 4 つの独立な群に分ける: **性能**（差・連対/複勝の差・順位相関・EWMA）、**較正**（ECE・傾き）、**予測の尖り/分布**（`sd_ratio`・PSI）、**入力**（調教の被覆など）。

| 段階 | 条件 | 対処 |
|---|---|---|
| 緑（ok） | どの群も基準の 2σ 以内 | なし |
| 注意 | 1 群が基準期間の 2σ を超えた | 次週も出るか見る。**入力の群なら原因（データの供給）を先に調べる**。表示は変えない |
| 警告 | 独立な 2 群以上が、連続 2 週で 2σ を超えた | 層別・特徴量の PSI・調教の被覆で原因を切り分ける（モデルの劣化かデータの不足か）。データの不足なら取り直し→同じ日を再予測して差を測る。予測の表示に「精度が落ちている可能性」の注意を出すか検討 |
| 要対応 | 差が +3σ を **4 週連続**で超えた／または較正のずれが回収率に効く水準（ECE 勝率>0.012、logit 傾きが 0.85〜1.15 の外）が連続 2 週 | (1) 入力の不足が原因なら取り直して予測を作り直す（再学習は不要）。(2) 較正のずれなら較正の再当てはめ（`config/ev_calibration.json`、複勝 EV の較正の見直し）と、期待値表示を目安扱いにする。(3) 入力が正常でも差が続くなら再学習の検討（下記） |
| 保留 | 窓内が 150 レース未満 | 判定しない |

**再学習のトリガーは単発の悪化では動かさない**: 要対応が 4 週以上続き、かつ入力の供給（調教の被覆・特徴量の欠損）が正常で、ECE/傾きを直しても差が残るときに、再学習の検討に上げる。再学習に 2026 年を含めるかは別途ユーザー判断
（2026 年はテストセット②。含めるなら、最終評価をやり終えたあとで、2027 年をテストセット③に切り替える）。

CUSUM は一度鳴ると指標が戻っても鳴り続ける。原因に対処したあとは `--ack-reset YYYY-MM-DD --note "…"` でその週以降からやり直す（`ack.json` と `ack_history.jsonl` に記録）。

## 管理限界の決め方と検証（バックテスト）

限界は基準期間（2022-2025）の窓ごとの値の平均±kσ（k=2 が注意、3 が要対応）。CUSUM/EWMA は週ごとの値の基準（平均・σ）。値は式と慣例で決め、2026 年の結果を見て選んでいない。
ただし診断で 2026 年のデータは見ているので、『2026 年に鳴らなかった閾値』ではなく『式で決めた閾値』である。

`python -m src.scripts.data.drift_backtest` が、2025〜2026 年の各週にさかのぼって同じ判定を再現する（2025 年の週は 2022-2024 の OOF だけを基準にし、先読みしない）。
結果（窓 8 週・3σ。詳細は診断の章）: **2025 年（ドリフトの無かった年）52 週のうち、注意が 2 週（3.8%、ECE がわずかに 2σ 超え）、警告・要対応は 0 週**。
2026 年は、調教の被覆の下落（4/20 の週）で入力の群が注意を出し、6/22 の週に警告、8/17 の週に要対応（差が 3σ 超えを 4 週連続）。感度表（窓 4/8/12 週 × 2/2.5/3σ）は診断の章にある。

## 実行

```bash
# 手動（環境変数に依らず実行）。結果は <out-dir>/latest.json・latest.html・history.jsonl（追記のみ）・reports/<時刻>/（上書きしない）
python -m src.scripts.data.drift_monitor [--out-dir data/local/meta/drift_monitor] [--no-features] [--no-shadow] [--no-coverage] [--dry-run]
python -m src.scripts.data.drift_monitor --ack-reset 2026-11-02 --note "調教ページの取り直しを完了"
python -m src.scripts.data.drift_backtest            # 過去の各週にさかのぼる（約 10 分）

# 特徴量の抜き取り（重い。1 日 10 レースで約 8 分。nice 19・load>14 で待つ）→ 監視（特徴量の PSI まで見る）
bash scripts/cron/drift_monitor_weekly.sh
```

- 実行時刻: **月曜の朝 1 回**（週末の結果・払戻・確定オッズが揃った後）。`scripts/cron/install_drift_monitor_cron.sh` が登録する行は月曜 10:00 JST（月曜 01:00 UTC、タグ `KEIBA-DRIFT-MONITOR`）。**登録はしていない（承認後に登録）**。
- 週の途中で実行すると、結果がまだ揃っていない最新週（レース数が直近の中央値の 6 割未満）は判定から外す（`partial_week_excluded`）。
- **自動実行（既定オフ）**: `.env` に `KEIBA_DRIFT_MONITOR=1` を足すと、金曜の weekly-update の後（`auto_scrape.finalize_weekly_update`）に `drift_monitor_hook.py` が別プロセスで `drift_monitor --no-features` を起動する。待たない・失敗しても本体は成功。**cron と併用しない**（同じ週に 2 回履歴が足される）。
- 出力先は既定 `data/local/meta/drift_monitor/`（`KEIBA_DRIFT_MONITOR_DIR` で変更）。読み取りと `data/local` への書き込みのみで、GCS・DB・netkeiba へはアクセスしない（調教の被覆は `storage.load` で GCS を読むだけ）。
- 特徴量の PSI の基準は `data/local/inference/drift_features/2025/`（`python -m src.scripts.data.drift_dump_features --years 2025 --bundle …bundle_v5_2024 --stage …v4_verify_2024`。2025 年の補助ヘッドは OOF に差し替える）。

## 部品

| 部品 | ファイル |
|---|---|
| 純関数（PSI・KS・較正・ECE・Brier・順位相関・差の区間・層別分解・FDR・管理限界・EWMA・CUSUM） | `src/scripts/data/drift_metrics.py` |
| 評価表（結果＋市場＋モデル予測）の読み込み | `src/scripts/data/drift_data.py` |
| 週次監視の本体・警報の段階・バックテスト | `src/scripts/data/drift_monitor.py`、`drift_backtest.py` |
| レポート HTML（スコアカード・管理図・較正曲線） | `src/scripts/data/drift_report.py` |
| ラベルなしの先行指標 | `src/scripts/data/drift_labelfree.py` |
| 調教ページの被覆 | `src/scripts/data/drift_coverage.py` |
| 特徴量の抜き取り（推論経路） | `src/scripts/data/drift_dump_features.py` |
| 一回きりの診断 | `drift_diagnose.py`・`drift_diagnose_features.py`・`drift_mask_experiment.py`・`drift_top_features.py`・`drift_benchmark.py` |
| 自動起動 | `src/pipeline/inference/drift_monitor_hook.py`（auto_scrape から） |
| テスト | `tests/scripts/test_drift_metrics.py`・`test_drift_monitor.py`・`test_drift_diagnose.py` |

予測の出どころは `config/drift_periods.json` の `prediction_sources`: 2022〜2024 年＝v4 アンサンブルの LOYO OOF（`tp27_v5/ens/v5.parquet`）、2025 年＝`final_eval_v4/test2025_predictions.parquet`、2026 年以降＝バックフィル（`predictions_backfill/bundle_v4_2025`）＋shadow（`predictions_v3`）。

## 注意

- 2026 年のバックフィル予測は、保存済みの `horse_training` を使って作るため、調教ページの取得が止まった 4 月以降は調教系の特徴量（tl_・wc_）が欠損になる。**差・尖りが悪化したときは、まず調教の被覆を見る**（入力の不足を除いてから、概念ドリフトを疑う。2026 年は入力の不足が尖りの低下の一部を説明したが、差の拡大の大半は説明できなかった）。
- 市場の確定オッズは Σ(1/オッズ) が 1.0〜1.6 の外のレースを除く。着順が負（中止・取消）の行は除く。
- 窓は 8 週なので、段差の検出は約 1.5 か月遅れる（入力の被覆と CUSUM は数週で出る）。`sd_ratio` と予測分布は結果が不要なので、予測が出た時点で先に出せる。
- `shadow_v5` は v5 が `predictions_v5/` に溜まってから出る（10/24 以降）。約 2,000 レースで区間が狭まる。
