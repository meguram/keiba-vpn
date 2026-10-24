# v6 の推論用の参照データ: 更新するもの・しないもの（2026-10-22）

v5 / v6 の本番・shadow 経路が、学習時に作って固定した（または状態として持つ）参照データを読む。更新が要るものだけ cron にし、
モデルが固定の水準を前提にしているものは更新しない（再学習と同時に更新する）。

## 更新する（cron 登録済み。ホスト root の crontab、タグ KEIBA-V6-STATE）

| ジョブ | 時刻（UTC） | 内容 | 負荷 | 出力 |
|---|---|---|---|---|
| rb-state-daily（`scripts/cron/rb_state_daily.sh`） | 毎日 23:20（08:20 JST。export_tables_daily 22:00・feature-accumulate 22:40 の後、export のロックも待つ） | 前日までに確定した走で RB（めぐ指数/netkeiba 指数の履歴）の履歴状態を作り直す。検査に通らなければ古い版を残す | 約 6 秒・1GB 未満 | `data/local/inference/state/rb_v6/{hist.parquet,params.json}`（ファイル単位で原子的に差し替え） |
| px-ridge-window（`scripts/cron/px_ridge_window.sh`） | 毎週日曜 18:30（月曜 03:30 JST）。新しい半年の 30 日前から、リッジが無いときだけ作る | PX（血統 ALL_BEST）のリッジを次の半年ぶん作る。普段は何もしない | 約 50 秒・heavy_run 経由 | `state/px_v5/ridge_<T0>.npz` |

登録: `sudo bash scripts/cron/install_v6_state_cron.sh`（冪等。登録前に crontab を `data/local/meta/cron_backups/` へ退避）。確認: 手動実行で
rb-state-daily は 312,817 走・最終日 2026-10-18 の状態を新規作成し、再実行で同じ件数（差 0）。px-ridge-window は一時の場所で ridge_2026-01-01 を作り直し、既存と
**完全一致（最大差 0.0）**、本番の状態には触れていない。
注意: リッジは T0 より前の確定結果で学習する（as-of。特徴量パイプラインの一部）。2027-01-01 の窓は 2026 年の結果を含む（既存の ridge_2026-07-01 も 2026 年前半を含む）。

## 更新しない（理由つき）

| 参照データ | 理由 |
|---|---|
| 祖先表・SVD の成分（px_v5/anc・ancestors・horses・svd） | モデルが学習時の基底を前提にしている。新しい馬は `Ext`（血統 JSON から推論時に足す）が扱うので、状態の馬の表が古くても 2026 年の出走馬は欠けない |
| attrs_frozen.json・horse_attrs・final_stage・final_aux・bundle | 学習時の定数・モデル。再学習と同時に更新する |
| tl_reference_v2（追い切りの基準表） | as-of の月次表で、無い月は直近の月で補う。作り直しは再取得した全追い切りからの組み直しが要り、v5 の tl_ の定義が変わりうる。再学習と同時 |
| train_table 由来の母集団（V4Features の field_hist ほか） | 推論時に学習表（〜2025 年）に 2026 年以降の出馬表・結果を足して作る（固定ではない） |
| PedState（ped_*）・LayerA の履歴 | 起動時に最新の履歴から作る（固定ではない）。食い違いの原因は固定ではなく、血統名を空にする既定（`blank_pedigree=True`） |
| horse_profile（馬主） | 日次の serving-state-profile が足している（既存） |
| ラップ予想・追走難度 v2 の保存値 | 既存の cron（lap-forecast-eve/morning・raceday-eve・tracking-v2-weekly） |
