# 2026-11-13 の持ち越し: 用意済みの変更と適用の手順（一覧 0・cron の修正・job_map）

ユーザーの指示で、今日の作業は 18:00 JST で締めた。週次（18:00〜）の後に予定していた commit と crontab の変更は、11/14 に行う。

## 用意した変更の場所
どちらも、2026-11-13 17:10 時点の develop（de3d68ca）に `git apply --check` で当たることを確かめてある。

| 変更 | 置き場（ホスト。消えない） | 作業用の写し（/tmp。再起動で消えうる） |
|---|---|---|
| ① 毎朝の書き出しの GCS 全件一覧を 0 にする | `data/local/meta/pending_patches/20261113/nolist.patch` | scratchpad の `repo/`（`/tmp/claude-1000/-home-jovyan-work-keiba-vpn/0545e650-9e50-4b8a-a970-9ef25b037940/scratchpad/repo`） |
| ② ホストの cron の修正（共通ログインの時刻・sporthorse を外す・serving-state のタグ） | `data/local/meta/pending_patches/20261113/cron.patch` | 同上 |
| ③ job_map を crontab の実態に合わせる | `data/local/meta/pending_patches/20261113/apply_jobmap.py`（今の `config/gcp_job_map.json` に差分だけを当てるスクリプト。冪等） | `scratchpad/apply_jobmap.py` |

- 計画は `/home/jovyan/.claude-jovyan/plans/polymorphic-frolicking-flame.md`（②③）。①の中身は `docs/operations/flat_gaps_20261112.md` §6（パッチに含まれる）に書いた。
- critic の確認は済んでいる（①は 3 往復、②③は 1 往復で、指摘はすべて反映済み）。

### ① の中身（nolist.patch）
- `src/scraper/export_tables.py`: 本体の 6 カテゴリ（race_result・出馬表・ラップ・速報・オッズ 2 種）も `export_year_local` に載せる。
  - 母集合は `local_universe`（race_lists ＋ 出馬表・速報・確定結果の flat。欠けた回は 01〜12R を補う。確定結果だけは同じ開催の次の 2 日も補う）。
  - 窓と試行は `LOCAL_POLICY` で決める。
- `scripts/cron/export_tables_daily.sh`: 13 カテゴリを `--refresh --from-local-results` の 1 回の呼び出しにまとめる。
- `src/scraper/parse_drift_docs.py`: 標本をローカルの表から選ぶ。基準の年の割合は 30 日保存して使い回す。
- テスト: `tests/scraper/test_export_local_candidates.py`・`tests/scraper/test_parse_drift_docs.py`。
- 照合: 11/07・08 を消した表の写しで、11/09・10 の候補を作った。6 カテゴリとも取りこぼしは 0。
- 費用: 月 約 50 円 → 最悪 約 10 円・見込み 約 4 円（一覧 0 回）。

### ② の中身（cron.patch）
- `install_shared_login_cron.sh`: `5 0`（UTC＝JST 09:05）を `5 15`（JST 00:05）にする。
- `install_sporthorse_births_cron.sh`: `--remove`（安全策つき）を足す。
- 新規 `install_serving_state_cron.sh`（タグ KEIBA-SERVING-STATE）。
- `config/decoupling_roles.json`: 4 つの役割を足す。
- docs: `shared_login.md`・`cron_catchup.md`。
- テスト: `tests/scripts/test_install_cron_lines.py`。

## 11/14 の適用の手順
1. 11/13 の週次の結果を確かめる（`docs/operations/flat_gaps_20261112.md` §3 の手順。めぐ指数の 10/31・11/01・11/07・11/08 が埋まったか）。
2. ①: `git apply data/local/meta/pending_patches/20261113/nolist.patch` → CI と同じテスト → commit・push → CI が緑になるまで見る。翌朝（11/15）7:00 の書き出しから効く。
3. ②③:
   1. `git apply .../cron.patch` と `python3 .../apply_jobmap.py config/gcp_job_map.json` を当てる。
   2. `python -m src.ops.job_map validate` を通す。
   3. `python -m src.scripts.docs.build_scheduled_jobs_html` で一覧の HTML を作り直す（`--check` で確かめる。持ち主は monitor-scheduled-jobs-portal）。
   4. テスト（tests/ops・tests/scripts）→ commit・push → CI。
4. root の crontab を変える（前の crontab は各スクリプトが `data/local/meta/cron_backups/` に退避する）。
   - `sudo bash scripts/cron/install_shared_login_cron.sh`
   - `sudo bash scripts/cron/install_sporthorse_births_cron.sh --remove`。jovyan の crontab からも外す: `bash scripts/cron/install_sporthorse_births_cron.sh --remove`
   - `sudo bash scripts/cron/install_serving_state_cron.sh`
   - 権限の確認で止められたら、回り込まずに統括役へ返す。
   - 変えたら、monitor-scheduled-jobs-portal に内容を知らせる（config/scheduled_jobs.json への反映のため）。
5. 統括役の追加の指示: `sync_pg_from_gcs`（`src/scripts/data/etl_stg_db.py` の毎晩の全件一覧 2 回）を、直近のレースだけを名前で読む作りに直す。短い計画を立て、見積もりをコミットメッセージに書く。

## 11/14 朝 7:00 の書き出しで確かめること（①はまだ入っていない。今の作り）
- `logs/export_tables_daily.log` の最後の回について:
  - `LOCAL_EXPORT_FAIL`・`RUN_AXES_FAIL` が無いこと。
  - 付随 7 カテゴリの「一覧なしの追記: 候補 N・読めた M・無い K」の行（11/13 から入った作り）。今朝の予想は、パドック・バロメーターが 11/07・08 の 72 件ずつで、ほかは 0 件前後。
  - `update_run_axes` の出力（`{"candidates": …}`）。11/14 からは 11/07・08 の 72 レースが馬別ラップの公開の窓に入る。ただし、GCS の horse_lap はバロメーターの取得が保存してから入るので、最初の日は not_found が多くてよい。
  - parse_drift の比較の対象に、2026 年の馬柱・SmartRC（11/13 に作った flat）が加わり、時間が延びていないか。
- `data/page_reference/tables/2026/_export_local_report.json` の errors が 0 であること。
- 走法の軸の穴埋め（`backfill_horse_lap --years 2026`、ユーザーが 11/13 に起動。調教の過去分の取得が終わるまで待機中）が終わっていれば、翌朝の `update_run_axes` が run_axes を作り直していること（jsonl が run_axes より新しければ作り直す）。

## 11/13 に済んだこと（参考）
- b867161c・3d32e6bb:
  - 付随 7 カテゴリを毎朝一覧なしで書き出す。
  - 走法の軸の日次更新（バロメーターの取得から馬別ラップを保存し、update_run_axes で追記）。
  - 天気の取得の止まりを見えるようにした。
- 3a86b5ff・42f48f22: 天気の取得の費用の超過の修正と、定期取得の廃止（crontab から削除済み）。
- 2026 年の flat 7 種類を 1 回作った（馬柱・指数・バロメーター・SmartRC・追い切り・パドック・厩舎コメント）。
- 追いつき実行の cron を登録した（KEIBA-CRON-CATCHUP）。
