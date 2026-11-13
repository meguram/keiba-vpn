# 金曜の GCS の大量の読み出しを減らす計画（2026-11-13・統括役の依頼）

## 今の読み（費用の監査役の集計と、コード・ログでの確認）
| 処理 | 呼ばれ方 | 読むもの | 1 回の量 |
|---|---|---|---|
| 馬名インデックス（1 回目） | 18:00 weekly-update の中（`auto_scrape_queue.py:748-756`、キューを使わない側は `auto_scrape.py:1362-1370`）→ `horse_name_index.run_weekly_horse_name_index_update` | race_shutuba の全年を list_keys で一覧し、約 3.5 万件を 1 件ずつ load（`race_shutuba` は L2 の保持期間が 300 秒に切り詰められ、実質毎回 GCS） | 約 3.5 万件・約 0.2GiB |
| 馬名インデックス（2 回目） | 18:30 の cron horse-name-index（`task_horse_name_index`、`auto_scrape.py:1008-1021`）→ 同じ関数 | 同上（11/13 は 1 回目と時間が重なって 2 回読んだ） | 約 3.5 万件・約 0.2GiB |
| 成長曲線 | 同じ cron の後半（`auto_scrape.py:1025-1028`）→ `growth_curve_service.run_weekly_growth_curve_update` | horse_result の年ごとの一覧と、全頭 約 5.8 万件の load。`max_age_days=7.0` が週の周期をわずかに超えるため、毎回 skip=0 | 約 5.8 万件・約 1.2〜1.5GiB |

合計は週 約 1.5〜1.9GiB、月 約 7〜8GiB。

## 直し方
1. **馬名インデックスをローカルの flat から作る**（`horse_name_index.rebuild_horse_name_index_from_local_flat`。既定の source を `flat` にする）。
   - `data/page_reference/tables/<年>/race_shutuba_flat.parquet` と `race_result_flat.parquet` の horse_id・horse_name を、全年分あわせる。
   - 今のインデックスにある馬は残す（和集合）。そのため、馬の取得のたびの upsert（`run.py`）で入った馬も消えない。
   - 実データで、今の horse_name_index.json（58,701 頭）と馬の集合が完全に一致することを確認済み。
   - 出馬表の flat には、毎朝 7:00 の書き出しで 14 日先までの出馬表が入る。そのため、週末の新馬も金曜の時点で入る。
   - GCS の読みは 0。shutuba・horse_result の版は、環境変数で選べるように残す。
   - 2 回呼ばれること自体は残す（どちらも 0 件の読みになるため。週次の経路は VPS 取得の作業で触っているので変えない）。
2. **成長曲線を「変わった馬」だけにする**（`run_weekly_growth_curve_update`）。
   - 対象は、ローカルの race_result_flat で直近 14 日に走った馬と、成長曲線のファイルがまだ無い馬の和集合（horse_result が変わるのは走ったときなので）。
   - 年ごとの horse_result の一覧（`filter_horse_ids_with_horse_result`）は、対象を絞った後の馬にだけ当てる。または名前で読んで無ければ飛ばす（一覧は取らない）。
   - 最後に計算した日時は `_index_meta.json` に残る。
   - 1 回の読みは、約 5.8 万件から約 1,000〜2,000 件（その週の出走馬）に減る。
3. テストを足し、critic・CI と同じテスト・push・CI の見届けまで行う。

## 見積もり（1 回の件数・量 × 週 1 × 4.3）
- 直す前: 馬名 3.5 万件 × 2 ＋ 成長曲線 5.8 万件 ≒ 12.8 万件・約 1.7GiB/週 → 月 約 7.3GiB（約 130 円。無料枠を超えた月）、Class B 約 55 万回（約 33 円）。
- 直した後: 馬名 0 ＋ 成長曲線 約 1,500 件 × 約 25KB ≒ 40MB/週 → 月 約 0.17GiB（約 3 円）、Class B 約 6,500 回（0.4 円）。

## 実施の結果（2026-11-13 夜）
- 1 の照合: flat から作った版は 58,701 頭。今の horse_name_index.json（58,701 頭）と比べ、消えた馬 0・増えた馬 0。既存の馬は和集合で残る（テスト `tests/pipeline/test_weekly_gcs_reads_local.py`）。
- 2 の対象の決め方は、日数の窓ではなく**前回からの差分**にした。
  - 対象: ローカルの race_result_flat の最後の完走の日付が、保存済みの曲線の最後のレースより新しい馬と、曲線の無い馬。
  - 週次の取得が数日遅れても、結果が flat に入った次の回に拾う。
- 11/13 の実データでの対象は 2,348 頭（変わった 1,798 ＋ 曲線なし 550）。今までは 58,701 頭。
- 取りこぼしの条件: horse_result（GCS）の更新が flat より遅れている馬は、まだ新しいレースを含まない曲線になる。その場合も、曲線の最後のレースが flat より古いままなので、次の週も対象に残る（毎週 1 件読む）。
- 対象から外れた馬の無作為の 30 頭を作り直し、保存済みの曲線と比べた。30 頭とも一致した（GCS の読み 29 件）。
- critic の指摘で 2 点を足した。
  - 今日以降の出馬表に載っている馬も対象にした（地方・海外のレースは flat に無いため、中央に戻る時点で作り直す）。
  - 作れなかった馬（horse_result が無い・レースが空）は `data/calculated_data/growth_curve/_weekly_failed.json` に覚え、28 日あけて試す（出馬表に載ったら待たない）。
  - 11/13 の実データでの対象は 3,049 頭（出馬表 1,041 頭を含む）。
- 見積もり（直した後）: 成長曲線 約 3,000 件 × 約 25KB ≒ 75MB/週 → 月 約 0.32GiB（約 6 円）、Class B 約 1.3 万回（0.8 円）。馬名インデックスは 0。
  直す前の月 約 7.3GiB（約 130 円＋Class B 約 33 円）から、約 7 円になる。
