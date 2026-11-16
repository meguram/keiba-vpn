# 結果の表（race_result_flat）の race_class が空のレースが増えた件（2026-11-16）

## 何が起きたか
- 2026 年の race_result_flat で race_class が空のレースが増えた（2025 年は 3〜4%、2026 年 10 月 36%・11 月 59%）。出馬表の表は 3〜4% のまま。
- netkeiba のページの書き方の変化（e9939d63・75b89c0e の RaceData01 の件）とは別の原因。**表の書き出し（`src/scraper/export_tables.py`）の不具合**。

## 原因
1. 確定結果（db.netkeiba.com、`RaceResultParser`）と速報（race.netkeiba.com、`RaceResultOnTimeParser`）の解析は race_class を出さない。
   flat の race_class は、書き出しのときの出馬表のメタの補い（`_extract_race_meta` の fallback ＝ `meta_index`）だけで入る。
2. 2026-10-14 から日次の書き出しに速報（race_result_on_time）が入った。速報は確定結果より先に書き出され、`meta_index`
   （race_result_flat から組む）にそのレースがまだ無いので、race_class が空のまま速報の flat に入った（10/17〜11/08 の速報は全レース空）。
3. 2026-10-18 から、確定結果が未公開の間は速報の行を race_result_flat に橋渡しする（`bridge_on_time_results`）。空の race_class の行が
   race_result_flat に入った。
4. 次の日の書き出しは `_build_meta_index_from_race_result_flat` で、橋渡しの行（空の race_class）をそのままメタにした。
   確定結果が出たとき、そのレースは「メタあり」とみなされ、出馬表からの補い（`_add_shutuba_meta`）を飛ばした。
   確定結果の行も空の race_class で書かれた（10/24・10/25・10/31・11/01・11/07・11/08 の全レース。ログの `bridged_races` と一致）。
5. 11/14・11/15 が空でないのは偶然。確定結果を 1 件も読めなかった日は、出馬表から補ったメタが速報の書き出しまで残る。
   1 件でも読めた日は `_build_meta_index_from_race_result_flat(year) or meta_index` でそれを捨てるので、また空になる。
- 5/24・6/06・6/07 の固まり（約 60 レース）は 10 月より前で、橋渡しの前。原因は追っていない（同じ埋め直しで埋まる）。

## 直したこと
- `_build_meta_index_from_race_result_flat`: 同じレースに確定結果と橋渡しの行があるときは確定結果の行を採る。空の race_class を、
  手元の `race_shutuba_flat.parquet` で補う（`_fill_index_from_local_shutuba`。GCS・netkeiba は読まない。出馬表にだけあるレースは足さない）。
- race_result を追記した後のメタの作り直しを `_rebuild_meta_index` にした（flat にまだ無いレースの、出馬表から補ったメタを捨てない）。
- 橋渡しの行から組んだメタで、手元の出馬表でも race_class が埋まらないもの（手元の表に無い・枠順確定前の版のまま）は、
  `_needs_shutuba_meta` で出馬表を名前で 1 件読んで補う（一覧なし）。障害レースは確定結果が出るまで毎日読み直す。
  最悪の見積もり: 障害 3 レース/日 × 7 日 × 30 日 ≒ 630 回/月の読み（Class B 約 0.04 円）＋転送 約 20MB（約 0.4 円）。月 1 円未満。
- 埋め直しの CLI `python -m src.scraper.export_tables --years 2026 --fill-race-class [--apply]` を足した（下）。
- テスト: `tests/scraper/test_export_race_class_fill.py`。
- 日次の書き出しの費用はほぼ増えない（主に手元の race_shutuba_flat を読む。GCS の一覧は 0 回、読みは上の月 1 円未満）。

## 空のレースの数（2026-11-16、手元の表で数えた。`--fill-race-class` を `--apply` なしで実行）
| 年 | 表 | レース | 空 | 手元の出馬表で埋められる | 埋めた後も空 |
|---|---|---|---|---|---|
| 2025 | race_result_flat | 3,456 | 121（3.5%） | 0 | 121 |
| 2026 | race_result_flat | 3,072 | 354（11.5%） | 242 | 112（3.6%） |
| 2026 | race_result_on_time_flat | 3,042 | 417 | 312 | 105 |
- 埋めた後も空の 112 レースのうち、約 106 は障害レース（出馬表でも空。2025 年の 3〜4% と同じもの）。
  残る 6 レース（2/07 東京 8〜12R・202610010604）は手元の出馬表の表に無い。
- 月別（2026 race_result_flat の空）: 10 月 103/288、11 月 99/168（うち確定結果 96/96、速報の橋渡し 3/72）。

## 埋め直しの手順（推奨: 手元の表だけで直す。実行はユーザーの判断）
本番の表（ホストの `data/page_reference/tables/`）を書き換え、GCS へミラーされるので、このセッションでは実行していない。
1. 日次の書き出し（07:00 JST、`/tmp/keiba_export_tables_daily.lock`）と重ならない時刻に、ホストのリポジトリで:
   ```
   # 数えるだけ（書き換えない）
   python -m src.scraper.export_tables --years 2026 --fill-race-class
   # 書き換える（race_result_flat と race_result_on_time_flat の空の race_class だけ）
   python -m src.scraper.export_tables --years 2026 --fill-race-class --apply   # 日次と同じロックを取る（動いていれば止まる）
   ```
2. もう一度 `--fill-race-class`（`--apply` なし）を実行し、race_result_flat の「埋められる」が 0、「埋めた後も空」が約 112 になることを確かめる。
3. race_class を使う集計を作り直す（次の定期の作り直しを待ってもよい）: 騎手・調教師の成績（`python -m src.pipeline.build_jockey_trainer_stats`）、
   めぐ指数（`src/pipeline/megu_index/`）、レースの成績の特徴量（`src/pipeline/features/race_performance.py`）。
4. 残る 6 レース（障害でないもの）を埋めるなら、その race_id の出馬表を GCS から名前で読む（`_add_shutuba_meta` と同じ。最大 6 回）。
   急がない。

### 見積もり（一度きり）
| 項目 | 回数・量 | 円 |
|---|---|---|
| netkeiba への取得 | 0 回 | 0 |
| GCS の読み（手順 1〜3） | 0 回 | 0 |
| GCS の書き（race_result_flat 約 2.9MB をミラーの push で 1 回。差分は手元の MD5 で判定） | Class A 1〜2 回 | 0.01 未満 |
| 残る 6 レースを埋める場合 | GCS の読み 6 回（Class B）・転送 約 0.3MB | 0.01 未満 |

### やらないこと
- `export_tables --force` で作り直さない。flat を消すと `_build_meta_index`（race_shutuba の一覧 ＋ 約 3,000 件の読み込み）に落ち、
  GCP の外から GCS を大量に読む（転送料）。また確定結果・速報の文書を取り直しても race_class は入らない（文書に無い）。
