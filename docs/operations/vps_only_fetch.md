# VPS だけでデータ取得する（2026-11〜）

ユーザーの方針（2026-11-12）: データ取得は VPS の中だけで行い、GCP との連携を外す。保存先は今どおり GCS（正本）。
取得の後の処理（予測・馬場速度・成績の集計など）はホストが行う。ホストの取得の cron は、VPS で問題なく取れることを確かめてから止める。
計画の全体: VPS（取得）→ GCS → ホスト（後処理と予測）。

## 引き継ぎ（2026-11-13 18:00 JST で今日の作業を締めた。明日ここから再開）
### 済んだこと（すべて develop。master への昇格はまだ。6bdcfbeb の CI の赤は他セッションの vps_alive.py が原因で、de3d68ca で直った）
- 第 1 段（30097498・f5f969ac ほか）: VPS のタスク表をローカルにする、予約を VPS の systemd timer で行う、ホストの常駐ランナーに「取得しない」モードを足す。
- 第 2 段・段 1〜3（04cb7383・3ba60dcf・4211b34a・2e6c8066 ほか）: レース一覧・時刻表を VPS→GCS→ホストに渡す（push-vps / pull-vps）、
  ホストの予測の前に VPS の T-15 の保存を確かめる、オッズのキャッシュを短くする。
- 段 4 の前半（host_postprocess.py）: 前夜の計算・馬場速度・予測の成績を、ホストの後処理だけのジョブにする。cron は未登録。
- 段 4 の後半（2026-11-13 夜）: VPS の weekly-update は後処理（めぐ指数・PG 同期・予測の成績・追走難度・ドリフト）をせず、
  GCS の `ops/mirror_vps/_weekly_finalize.json` に目印（書いた時刻・対象の開催日）を書く。ホストの `host_postprocess weekly-finalize`
  （`src/ops/weekly_finalize.py`、土 00:07〜月 08:07 JST の毎時）が目印を名前指定で 1 件読み、未処理の回なら finalize_weekly_update を動かす。
  キュー待機が打ち切られた回（終了コード 7）は目印が書かれないので、ホストも後処理をしない。失敗は 3 回まで取り直す。
  済んだ回は `logs/weekly_finalize_done.json`（回の鍵は目印の written_at）。
- 段 5（2026-11-13 夜）: backfill-all-horse は過去の馬を ID 順に回る補填（11/13 に取ったのは 2013 年生まれの 2 頭）。
  新馬の父（sire）はプロフィールのページからも取れるので、VPS の前日夕の新馬の取得で足りる → **ホストに残す**（新しいタスクは作らない）。
  あわせて、新馬の血統（new-horse-pedigree）の取りこぼしを直した: VPS には手元の ped_tbl が無く全馬が「血統なし」に見え、
  上限 150 頭を GCS に 5 世代表がある馬で使い切っていた → 上限は実際に通信した頭数で数え（飛ばす経路の祖先・種付けの取得も数える）、
  新しい ID（新馬）から確かめ、GCS の確認は 1 回 600 頭まで（馬 ID の名前指定で読む。一覧しない。最悪 月 約 5 円）。
- 費用の対策:
  - GCP の並走の Jobs を止めた。費用の歯止め（cost_guard、cron 登録済み）を入れた。
  - VPS の取得役を一時停止した（15:00 JST）。
  - 温め役を合図方式にした（ユーザーが 16:30 JST に入れ替えた）。
- 方法 A（6bdcfbeb）: root 所有の ops ファイルを、deploy から決まった操作で入れ替える仕組み。下の節を参照。

### 残り（明日以降、上から順に）
1. **費用の確認（11/14・11/15 の午前）**: 予約はセッションを閉じると消えるので、手で行う。
   - 方法: `GCP_BILLING_BQ_TABLE=horse-racing-466415.billing_export.gcp_billing_export_v1_01B0D4_CDCC8C_CBA87A KEIBA_ENV=stg PYTHONPATH=. /opt/conda/bin/python3` で
     `load_project_dotenv()` の後、`src.scripts.cloud_jobs.gcp_daily_cost_report` の fetch_cost_history / fetch_sku_breakdown を使う（読み取りのみ）。
   - 11/14: 11/13 の日額を、11/08〜11/12 と比べる（Cloud Run の SKU、月末の見込み、月 2,000 円との関係）。
     11/13 は horse-name-index（m サイズ、見込み 1 回約 13 円）が初めて新方式で動いた日。
   - 11/15: 11/14 の日額を確かめる。温め役の一覧が 1 日 300 回以下かも確かめる。
   - 費用の監査役のセッションも 11/14 に確かめる。
2. ~~段 4 の後半（週次の後処理）~~ 済み（上）。ホストの cron（`install_host_postprocess_cron.sh`）の登録は段 6 で担当を渡すとき。
3. ~~方法 A の最初の root 作業~~ 済み（2026-11-13 17:11 JST）。入れ替えに使えるのは、6bdcfbeb 以降の取得の配布物が VPS に取り込まれた後（master への昇格の後）。
4. **段 6（VPS に配備して段階的に渡す）**（2026-11-13 18:3x にユーザー承認。master への昇格（11/14 朝、monitor-cloudrun-handoff が担当）の後に始める）:
   - 配備: 第 1 段・第 2 段のコードを master に昇格する（ユーザーの指示が要る）。release-pull が 03:00〜04:00 に取り込む。
   - 準備: VPS の fetch.env に KEIBA_FETCH_TASK_STORE=file を入れ、fetch-timers（方法 A）と push-vps の timer を入れる。
     取得役のユニット（keiba-fetch-agent.service。タスク表を file に固定し、設定の誤りの 78 で再起動しない）と fetch_agent_ctl.sh は
     方法 A の許可リストの外（root 所有）なので、root で入れ替える: `install -m 644 templates/keiba-fetch-agent.service /etc/systemd/system/`・
     `systemctl daemon-reload`・`systemctl try-restart keiba-fetch-agent`（動いていれば新しいユニットで起動し直す。今は止めてある）・
     `bash vps_08_fetch_ctl.sh`（fetch_agent_ctl.sh を入れ直す）。vps_03_units.sh --enable-fetch-agent でも同じことをする。
     コードでも、KEIBA_RUN_LOCATION=vps で gcs のタスク表を作ろうとすると断る（取得役・予約・計画のどの入口でも。src/ops/fetch_task.py の task_store_from_env）。
     ただし KEIBA_RUN_LOCATION を付けずに VPS で手で動かすと素通りするので、手で動かすときもユニットと同じ環境変数を付ける。shadow-on と vps_03_units.sh は、
     fetch.env の KEIBA_FETCH_TASK_STORE=file・取得のコードの FileTaskStore・ユニットの file の固定の 3 つがそろわなければ断る（GCS のタスク表に戻さない）。
   - 影モード: `fetch_agent_ctl.sh shadow-on` で 2〜3 日動かし、250MB の枠・件数・エラーを確かめる。
   - 担当を渡す: 軽いものから、`config/fetch_owner.json` の agent_tasks に足していく。
   - ホストの cron: `install_vps_mirror_pull_cron.sh` と `install_host_postprocess_cron.sh` を root で登録する。
   - 仕上げ: 開催週を 1 回通してから、ホストの取得の cron を止める。
   - ssh keiba-vpn での作業は、その都度ユーザーに確かめる。
5. **IAM の案 B**（2026-11-13 18:3x にユーザー承認。付けるのは統括役。付ける直前に下の最小のバインディングを統括役に送る）:
   VPS の取得役（KEIBA_RUN_LOCATION=vps・影モードなし・KEIBA_FETCH_TASK_STORE=file）が書く・消す場所をコードで洗い出した（2026-11-13）。
   すべて `roles/storage.objectUser`＋`scripts/gcp/make_iam_condition.sh <接頭辞> <タイトル>` の条件（上書きに delete が要るため objectCreator では足りない）。
   | 接頭辞 | 書くもの | 書けないと |
   |---|---|---|
   | `chuou/data/preprocessed/netkeiba/pc/` | 取得データの本体（HybridStorage.save。出馬表・オッズ・結果・馬・血統など） | 致命的（403 の後 60 秒は黙って捨てる） |
   | `ops/sire_lines/` | 系統台帳 sire_line_map.json（sire-line のモード） | 致命的（タスクが失敗） |
   | `ops/cache_dirty/` | 温め役の目印（画面に関わる保存のたび。同じキー 60 秒に 1 回） | 非致命的（温め役が取り直さない） |
   | `ops/page_quality/` | latest.json（page-quality-check） | 非致命的 |
   | `chuou/data/raw/html/` | HTML の見本（カテゴリごと 10 件、11 件目以降を**削除**） | 非致命的。削除させたくなければ付けない |
   - 付いていて使うもの: `ops/mirror_vps/`（受け渡し・週次の目印）、`ops/heartbeat/vps/`（定期実行の見張り）、バケット全体の objectViewer（読み・一覧）。
   - 付いていて、この構成では使わないもの: `ops/fetch_tasks/`・`ops/events/`（タスク表がファイルのため）、`shadow/`・`ops/fetch_tasks_shadow/`（影モードを外すため）。
     影モードの間（段 6 の最初の 2〜3 日）は shadow/ を使う。影を終えたら外してよい。
   - `chuou/data/others/` は VPS の取得の経路で誰も書かないので付けない（docs/operations/gcp_user_steps_20261026.md の想定と違う）。
   - 費用（最悪・月）: VPS から GCS への書き込みは転送料 0。Class A は、取得データの保存（開催日 1 日 数千件）と ops/cache_dirty/ の目印（1 日 約 1,000 件）で
     月 約 10 万件 → 約 80 円。ホストが今書いている分（同じ件数）が VPS に移るだけで、全体では増えない。
   - 付ける前の確認: バケットのソフト削除が有効か（`gcloud storage buckets describe gs://$BUCKET --format='value(soft_delete_policy.retentionDurationSeconds)'`）。
6. **調教の夜間の補填**: ホストの調教の過去分の取り直し（別セッション）が終わってから VPS に移す。

### 再開の手順
- `git log --oneline -20 -- src/ops src/scraper/auto_scrape_queue.py docs/operations/vps_only_fetch.md` で、この後に入った変更を見る。
- 状態を確かめる:
  - `config/fetch_owner.json` の agent_tasks が空 → まだ何も渡していない。
  - `ssh keiba-vps systemctl is-active keiba-fetch-agent` が inactive → 一時停止のまま（実行の前にユーザーに確かめる）。
  - `crontab -l`（root）で cost_guard の行がある。
- 上の「残り」の 1 から進める。

## できたもの（2026-11-13）
- VPS のタスク表をローカルに置く（`KEIBA_FETCH_TASK_STORE=file`、`/var/lib/keiba-fetch/tasks`。`src/ops/fetch_task.py` の FileTaskStore）。
- 予約を VPS の systemd timer で行う（`keiba-fetch-reserve.timer` 毎時 50 分、`keiba-fetch-raceday-tomorrow.timer` 20:30、`-today.timer` 07:20。
  `scripts/ops/vps/templates/`、導入は `vps_03_units.sh`）。GCP の reserve-fetch-tasks / reserve-race-day は使わない。
- VPS の fetch.env の雛形に、予測に要る取得のフラグ（直前オッズ・3 連単・前日夕の調教・新馬の基本情報）。既存の VPS の fetch.env は上書きしないので手で足す。
- ホストの常駐ランナー（raceday-runner・raceday-weight-runner・raceday-result-runner）の「取得しない」モード。
  置き換え先の取得（`src/ops/fetch_owner.py` の RUNNER_CORE）をすべて VPS が担当したとき、または `KEIBA_HOST_NO_FETCH=1` のとき、
  ランナーは取得をせず保存済みのデータで後処理（レース前の予測・馬体重の判定と予測・馬場速度と成績の集計）だけをする。
  今は担当（`config/fetch_owner.json` の agent_tasks）が空なので、どれも動き方は変わらない。

## 2026-11-13: VPS の取得役を一時停止
- 理由: 担当（agent_tasks）が空なのに、影モードの実行役が GCS のタスク表（ops/fetch_tasks・ops/fetch_tasks_shadow の各状態）を毎分 6 回一覧していた
  （1 日 8,640 回・月 約 210 円＋転送。GCS の監査ログで見つけた）。本番の取得はホストなので、止めても影響は無い。
- 操作（ユーザー承認）: 15:00:58 JST に `sudo /opt/keiba-ui/bin/fetch_agent_ctl.sh shadow-off`（deploy ユーザー、パスワード不要の決められた操作）。
  keiba-fetch-agent を stop・disable し、fetch.env の KEIBA_FETCH_SHADOW_ENABLED を 0 にした。ほかのユニット（画面・キャッシュの温め・配備の取り込み）は触っていない。
  停止前: enabled/active、KEIBA_FETCH_AGENT_ENABLED=1・KEIBA_FETCH_SHADOW_ENABLED=1、版 fetch-b316ae59。
- 再開: 段 6 で、VPS の fetch.env に KEIBA_FETCH_TASK_STORE=file を入れ（タスク表を VPS のローカルにする。GCS の一覧は 0 件になる。
  tests/ops/test_fetch_local_store.py で確認）、第 1 段・第 2 段のコードを配備してから、`fetch_agent_ctl.sh shadow-on` か vps_03_units.sh で有効にする。

## 第 2 段（2026-11-13〜）
計画: 段 1 受け渡し → 段 2 T-15 の保存の確認 → 段 3 ホストのキャッシュの鮮度 → 段 4 後処理の分離 → 段 5 馬の成績の毎日の取得 → 段 6 配備と段階的な受け渡し。

### 段 1. レース一覧・発走時刻表の受け渡し（VPS → GCS → ホスト）
- VPS: `python -m src.ops.gcs_mirror push-vps`（systemd timer 5 分おき）。**担当が VPS に移った木**（race_lists・race_day_schedule ← daily-race-lists）のうち
  中身が変わったものだけを `ops/mirror_vps/` に上げ、目録 `ops/mirror_vps/_index.json` を書き直す（変わったときだけ）。差分は VPS の手元の記録（MD5）で判定し、GCS は一覧しない。
  担当が無ければ何もしない（影モードの取得分を上げない）。
- ホスト: `python -m src.ops.gcs_mirror pull-vps --trees race_lists,race_day_schedule`（5 分おき）。目録 1 件を読み、ローカルと中身が違うファイルだけ取る。
  担当が移った木だけを取り込み、移っていなければ GCS に一切触れない（今は担当が空なので費用 0）。最後に取り込めてから 2 日（目録の保持期間）を超えたら、
  取りこぼしの恐れとして終了コード 2。取り込んだファイルはホストの通常の push で ops/mirror/ に上がり、Cloud Run の API に今どおり届く。
  cron の登録は `scripts/cron/install_vps_mirror_pull_cron.sh`（root で実行）。
- **JRA の馬場（cushion/）は受け渡さない（当面ホストで取る）**: cushion_values.json は手元のファイルに追記していく作りで、VPS のファイルには VPS が取った分しか無い。
  丸ごと取り込むとホストの 2018 年からの履歴を消す。開催カレンダー（race_calendar_YYYY.json）は多くのタスクがホストと VPS の両方で作り直す。
  書き込みが原子的でない。予測に届くのが遅れる。レコード単位で合わせる設計を別に作るまで、jra-baba-morning と当日の取り直しはホストに残す。
- **IAM**: VPS の取得役のサービスアカウント（keiba-fetch@horse-racing-466415.iam.gserviceaccount.com）に、`ops/mirror_vps/` 以下だけの
  `roles/storage.objectUser`（条件タイトル `fetch-mirror-vps`、`scripts/gcp/make_iam_condition.sh ops/mirror_vps/` と同じ式）を **2026-11-13 に付けた**
  （ユーザー承認。バケットの IAM のバインディング 17 → 18 件、ほかは変わらず）。戻し方: 同じ条件ファイルで
  `gcloud storage buckets remove-iam-policy-binding gs://<バケット> --member serviceAccount:keiba-fetch@… --role roles/storage.objectUser --condition-from-file …`。
  本番のデータの置き場所（`chuou/data/preprocessed/netkeiba/pc/`・`chuou/data/others/`）への書き込み（案 B）は、段 6 で担当を渡す直前に、
  VPS が実際に書くパスを確かめてから、あらためてユーザーに確認する。

### 段 4. 後処理をホストに残す（src/ops/host_postprocess.py、cron は scripts/cron/install_host_postprocess_cron.sh）
- 対応する取得の担当が VPS に移ったジョブだけ動く（移っていなければ何もしない・GCS に触れない）: 前夜の計算（← raceday-eve-fetch。19:10・19:40）、
  馬場速度（← raceday-evening か recent-retry。翌朝 07:20。開催日の夜の当日分は翌朝に出る）、予測の成績（← recent-retry。土日月の 21:50）。
  ロックで重ならないようにし、GCP Jobs と同じ上限時間で止める。VPS の raceday-evening は馬場速度を計算しない（post_processing=host）。
- 費用（最悪・月。ホストは GCP の外なので VPS が保存したデータを読む分が転送料）:
  予測の成績 1 回 = 11 日分 × 36 レース × 4 種 × 約 30KB ≒ 47MB × 月 13 回 ≒ 0.6GB → 約 11 円。
  前夜の計算 1 回 = 翌日 36 レース × 約 5 種 × 約 30KB ≒ 5MB × 2 回 × 30 日 ≒ 0.3GB → 約 6 円。
  馬場速度は手元の flat から計算（export_tables_daily が新しいレースを読む分は既存）→ 1 円未満。合計 約 20 円。
- 週次の後処理（めぐ指数・PG 同期・tracking_v2_eval・ドリフト）は、VPS の weekly-update では飛ばし、ホストの weekly-finalize に移した（2026-11-13 夜）。
  費用（最悪・月）: 目印 1 件（数百バイト）の読みが週 約 56 回 → 1 円未満。後処理がホストで GCS から読む分は、対象の開催日（多くて 4 日 × 36 レース）の
  確定結果・指数など 1 回 約 20MB × 月 4〜5 回 ≒ 100MB → 転送 約 2 円。

## 担当を VPS に渡す前に、そろえる必要があるもの（段 2 以降で対応）
1. ~~**local_only のデータの受け渡し**~~（段 1 でレース一覧・時刻表に対応。JRA の馬場は未対応・ホストに残す）: `race_lists`・`race_day_schedule`（HybridStorage の local_only。その機械のディスクだけ）と、
   JRA の馬場（`data/page_reference/cushion/`）。VPS が取るようになったら、VPS → GCS（ミラーの束・目録の方式）→ ホストに渡す。
   無いと、ホストのランナーは no-schedule で何もしない。
2. **VPS の T-15 の保存を確かめてから予測する**: 取得しないモードは、予定 + 猶予（`KEIBA_HOST_NO_FETCH_GRACE_SEC`、既定 180 秒）の後に必ず予測する。
   VPS の保存が遅れると、T-60 のオッズなどで予測してしまう。保存の時刻を確かめる仕組みが要る。
3. **ホストのディスク L2 キャッシュ**: 以前はホストが自分で保存して更新していた。VPS が保存するだけだと、ホストの L2 に古いオッズなどが残る。
4. **後処理だけのジョブの一覧**: raceday-eve（前夜の計算）・recent-retry の後処理（馬場速度・予測の成績）など、取得のジョブの中にある後処理を、
   取得と分けてホストに残す（VPS の recent-retry は後処理を GCP に任せる作りで、GCP Jobs は停止中）。
5. **毎日の馬の成績の一括（backfill-all-horse）と調教の夜間の補填（horse-training-backfill）**:
   - backfill-all-horse（毎日 06:00 JST）は、出馬表・確定成績の**全件を一覧して読む**ところから始まる（src/scraper/backfill.py の _run_phase_horse）。
     ホストは手元のキャッシュから読むので軽いが、VPS で動かすと GCS の全件を読むことになり転送料が大きい。中身はほとんど「すでにある馬は飛ばす」
     （2026-11-13 の実績: 500 頭中 498 頭を飛ばし、取ったのは 2 頭）。予測に要る出走馬の成績は、VPS の前日夕の新馬の取得
     （KEIBA_EVE_HORSE_PROFILE=1）と週次の取得（金曜）で取れる。**過去分の補填としてホストに残し**、VPS に移すのは、直近のレース一覧から
     対象の馬だけを選ぶ形を作ってから（全件の一覧をしない）。
   - 調教の夜間の補填は、ホストの調教の過去分の取り直し（別セッション）が終わってから VPS に移す（二重のアクセスを避ける）。
6. GCP 側（停止中）の `requires_fetch_done`・`vps_watchdog` は GCS のタスク表を見ている。GCP Jobs を再開する場合は合わせて見直す。

## 費用の見積もり（CLAUDE.md の決まり。最悪の回数で）
- 減る: VPS の GCS のタスク表の一覧 → 0。実測（2026-11-13、GCS の監査ログ）は、担当が空でも影モードで毎分 6 回（1 日 8,640 回・月 約 26 万回）、一覧だけで月 約 210 円＋応答の転送。最初の見積もり（月 約 8.6 万回・−150 円）の約 3 倍だった。
- 予約（VPS）: 開催日の時刻表・出馬表を読む。1 日 数十件 × 約 6KB。月 1 円未満。
- ホストの馬体重の判定（取得しないモード）: GCS の出馬表を 90 秒おきに読み直す。1 日最大 36 レース × 24 回 × 約 10KB
  ≒ 864 回・8.6MB。× 30 日（最悪）＝ 月 約 2.6 万回・260MB → Class B 約 1.6 円 + 転送 約 4.6 円 ≒ 月 6 円。
- ホストのレース前の予測（取得しないモード）: VPS が保存したデータ（出馬表・馬柱・オッズなど 1 レース約 10 件・約 0.5MB）を GCS から読む。
  1 日 36 レース ≒ 18MB × 30 ＝ 540MB → 転送 約 10 円 + 操作 約 1 円 ≒ 月 11 円。
- 合計で月 約 −130 円（増えない）。どれも対象のレースのファイルだけを読み、全件の一覧はしない。

## 記録だけ（費用は上限内・直していない）
- 新馬の血統（new-horse-pedigree）は VPS で毎日、出走予定の馬を最大 600 頭、GCS の 5 世代表と血統 HTML の有無で確かめ直す
  （手元に ped_tbl が無いため、取得済みの馬も毎日読み直す）。最悪 月 約 5 円。気になれば、確かめ済みの馬を VPS の手元に記録して翌日は飛ばす。

## 取得の網羅で気づいたこと
- 2026-09-22（祝日開催、24 レース）が、oikiri・paddock・trainer_comment・barometer の 4 カテゴリで同時に GCS に無い（fetch-status-unified-display の調査、2026-11-13）。祝日開催の取りこぼしとみられる。

## 関連: VPS の API キャッシュの温め役の費用（2026-11-13）
- 温め役（keiba-cache-warm、scripts/ops/vps/cache_warm.py）が 60 秒おきに GCS の ops/cache_dirty/ を一覧していた（月 約 265 円）。
  合図 ops/cache_dirty/_beacon.json（ホスト・Cloud Run Jobs の push と配備のときだけ書く）の世代を見て、変わったとき（最短 5 分）・
  合図が無いとき（10 分）・安全網（30 分）だけ一覧する形にした（05f35916。見積もり 最悪 月 約 40 円）。
- VPS への配備: 2026-11-13 16:30 JST ごろ、ユーザーが root で /opt/keiba-ui/bin/cache_warm.py を入れ替えた
  （sha256 467dac79… が develop の 05f35916 と一致。keiba-cache-warm.timer は active、16:30:59 の実行は終了コード 0）。
  1〜2 日後に、1 日の一覧が 300 回以下になったかを Monitoring で確かめる。戻すときは `git show 05f35916^:scripts/ops/vps/cache_warm.py` の版に入れ替える。
- 今後の入れ替えを root に頼らずに行う方法（方法 A）を作った。下の節を参照。

## VPS の root 所有の ops ファイルを deploy から入れ替える（方法 A、2026-11-13）
- 取得の配布物（kind=fetch）に ops ファイル（`src/ops/release_publish.py` の `OPS_FILES`: cache_warm.py と、温め役・定期実行の見張り・
  VPS の取得のタイマーのユニット）と、その sha256 の目録 `ops_manifest.json` を入れる。VPS の root のタイマー（release-pull）が
  latest.json の sha256 で確かめてから、root 所有の `/opt/keiba-fetch/releases/<版>/` に展開する。
- deploy は `sudo -n /opt/keiba-ui/bin/install_ops_files.sh install <版> <名前>` だけで入れ替えられる（名前: cache-warm・cache-warm-units・
  sched-status-units・fetch-timers）。スクリプトは版の形・許可リストの名前・版のディレクトリとその親が root 所有で書けないこと・目録の sha256 を毎回確かめ、
  1 つでも違えば何もしない。今のファイルは `/var/lib/keiba-ops/backup/<名前>/<時刻>/` に退避し、`rollback <名前>` で戻せる。記録は `/var/log/keiba-ops-install.log`。
  確認は `status [<名前>]`。fetch-timers は入れるだけで有効化しない（有効化は担当を渡すときに別途）。
- 費用: GCS に触れない（取り込み済みの配布物だけを使う）。配布物が数十 KB 増えるだけで、月 1 円未満。
- **最初の root 作業は済み**（2026-11-13 17:11 JST、install_ops_files.sh と sudoers を VPS に導入。sha256 は develop と一致）。手順の記録: develop の 3 ファイル（`scripts/ops/vps/install_ops_files.sh`・`scripts/ops/vps/vps_12_ops_installer.sh`・
  `scripts/ops/vps/templates/sudoers-keiba-ops-install`）を VPS の /tmp に置き、ConoHa のコンソールで root でログインするか `su -` で root になってから
  `bash /tmp/vps_12_ops_installer.sh`（sudoers は visudo -cf で検査してから入る）。確認: `sudo -u deploy sudo -n /opt/keiba-ui/bin/install_ops_files.sh status`。
  外すときは `bash /tmp/vps_12_ops_installer.sh --remove`。
