# 取得・取り込みの欠けの調べ（data-intake、2026-11-24）

GCS は読んでいない（手元の表・ログ・launch_watch だけ）。費用は 0 円。
2025・26 年の race_result_flat・race_shutuba_flat からは race_id・horse_id・日付と weight・weight_change の欠けの割合だけを読み、着順・払戻は読んでいない。
台帳の申告（config/data_access/ledger.jsonl、session data-intake、family data_intake、purpose reproduce）の reads は declare の既定（着順・払戻）のまま。実際には読んでいない。

## 1. 2026 年の追い切りの記録が 32% しかない（megu_oikiri）

- 原因: megu_oikiri が読む `data/local/modeling/training_pipeline_v4/tp35/training_laps_fresh.parquet` は、過去分を取り直したときの固定の表（10/19 作成）。
  2026-05 以降の行がほとんど無い（2026-04 は 9,062 本、2026-05 は 839 本、以降は月 500〜700 本）。
- 取得の失敗・取り込みの漏れ・列の名前の変更ではない。手元の調教の記録（`data/local/horse_training/`）は、2026-04 以降の出走のほぼ 100% を持つ（2026-01〜03 は 85〜89%）。
- 出走のうち、レース週（1〜6 日前）の追い切りが表にある割合（race_result_flat の出走と突き合わせ）:

| 表 | 2025 年 | 2026 年 | 2026-05〜11 の月ごと |
|---|---|---|---|
| 固定の表（training_laps_fresh） | 99.0% | 31.9% | 0〜7.6% |
| 重ねた表（training_laps_merged） | 99.5% | 98.9% | 99.1〜100% |

- 直した: `src/scripts/data/build_training_laps_merged.py` で、固定の表に手元の記録の行を重ねた `tp35/training_laps_merged.parquet` を作った（ホストに置いた。固定の表は上書きしない）。
  馬 × レースの日の組が手元にあれば手元の行だけを使い、無ければ固定の表の行を残す。費用 0 円、取り直しは不要。
- 手元の記録だけでは 2025 年が欠ける。出走のうち手元に記録ファイルがある割合は、月ごとに 2025-01 の 45% から 2025-12 の 80%。
- megu_oikiri final は流し直さない（統括役の判断、11/24。条件は凍結済みで、結論は「確かな効きなし」）。megu_why の HTML に注を足した。
- 注意: 直した表の 2026 年の値も、発走後に取った追い切り。発走前に使える情報より多い（2 の構造上のずれと同じ根）。

## 2. 推論の入力の欠け（本番 v6e）

### 追い切り（tl_ の列）: 今も起きている。構造上のずれが有力

- 影のランナーの記録（`data/local/inference/launch_watch/2026112[1-3].jsonl` の event=v6c）で、na_tl の中央値は 0.49（新馬戦以外。新馬戦は 0.10〜0.15）。
  学習の表（cand_TL_A_v2）の同じ 50 列の空の割合は、1〜3R で 2024 年 0.26・2025 年 0.26。差は約 23pt。
- 手元の調教の記録から `training_serving.tl_a_for_race` を作り直すと、影の値と一致した（202605050501: 0.485、202608050601: 0.522、202605050502: 0.512）。取り込みの経路は記録どおりに動いている。
- 欠けているのは w1_（1 週前）・w2_（2 週前）の列で、全馬が空。
- 発走前に取った調教ページの記録では、そのレースの追い切りは最終追い切りだけのことが多い。
  - 1 週前の追い切りがある馬: 発走前に取った記録 16%（1,040 件）、発走後に取った記録 83%（9,327 件）。
  - 1 レースあたりの追い切りの数: 発走前 1.9 本、発走後 4.3 本。
  - 例: 2024104937 は 11/21 18:38 に取得。11/22 のレースは 11/18 の 1 本だけで、11/07 のレースは 10/18〜11/04 の 6 本。
- 学習の表は発走後に取った記録で作っている。推論では見えない 1・2 週前の追い切りを学習で使っている（構造上のずれ）。
- 未確認の点: netkeiba が発走前に載せていないのか、こちらの取り方で落ちているのか。HTML の控えは GCS にしかないので開いていない。
  確かめ方（費用 0、統括役の承認済み）: 11/27 の前日取得の記録を 1 頭分写して残し、レースの後に取り直した記録と並べる（取得は 1 ページ）。
- 直す担当: 構造上のずれなら model-improve-cycles（学習の X の w1_・w2_ を発走前の見え方に合わせる、または外す）。本番 v6e に効いているが、効きの大きさは測っていない。
- 2025-06-28 の欠けを「同じ根」とは言えない。2025 年の記録はどれも発走後に取ったもの。
  候補は、v6_parity が読む GCS・キャッシュの記録が、手元で取り直した記録と違うこと（2025-06-28 の出走馬 469 頭のうち、手元に記録があるのは 223 頭）。未確認。

### 馬体重: 11/21〜23 に取り込みの欠けは無い

- na_weight は weight・weight_change・d_wt_* の空の割合。
- 新馬戦は 0.6（d_wt_ の 3 列が履歴なしで空。weight はある）。それ以外は 0〜0.16（d_wt_ の前走との差が足りない馬だけ）。どちらも学習の側と同じ形。
- 2025-08-02 の 5〜8% の欠けは再現できず、根は未確認。

## 3. sire_smartrc_lines.json が共有の作業ツリーで差分になる

- 書き手: entries-sync（例: 11/23 08:30）→ `src/scraper/entries_sync.py:279` → `refresh_for_races` → `build_horse_sire_map.line_labels` → `smartrc_sire_lines.build()`。export_tables_daily の precompute_race_analysis も同じ経路。新しい種牡馬が出た日だけ書く（11/23 は 2 頭増）。
- 直し: 書き出し先の既定を `data/local/meta/sire_lines/sire_smartrc_lines.json` に移した。手元に無い・壊れたときだけ git の種の版（data/meta）を読む。data/local が無い場所（CI・リンクを張る前の worktree）では書かない。
- 種の版を新しくするときは、ホストの手元の台帳を data/meta へ写して commit する（worktree・CI の研究の道具は種の版を読む）。
- 反映の手順（共有の作業ツリー）: 配備の後に `data/local/meta/sire_lines/sire_smartrc_lines.json` ができたのを確かめてから、data/meta の版を `git checkout --` で戻す。先に写してもよい。

## 4. profile-local（e8d51c13）の効き

- 11/21〜23 は、本体の weight-runner（`logs/raceday_weight_runner.log`）・影のランナー（`logs/candidate_runner.log`）のどちらにも「新しい馬の補い」の行が無い。
  この行は表に無い馬が 1 頭でもいれば必ず出る（同じ logger の「手元の profile_dyn を読んだ」は出ている）。profile の GCS 読みは 0 回。
- 前の値（11/15 の影で 282 回）は commit の記載。測り直していない。
- precompute（serving-state-precompute）の出力はログに残っていない。profile_dyn は 717 → 780 → 823 行と増えており、補いは precompute の側で起きている。
- 課金エクスポート（BigQuery）は、この環境に認証が無く見ていない。
- 11/23 の影の gcs_mb は中央値 1.25MB で、前の 2 日（0.3〜0.4MB）より多い。月 1 円未満。
