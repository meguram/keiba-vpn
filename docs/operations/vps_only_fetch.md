# VPS だけでデータ取得する（2026-11〜）

ユーザーの方針（2026-11-12）: データ取得は VPS の中だけで行い、GCP との連携を外す。保存先は今どおり GCS（正本）。
取得の後の処理（予測・馬場速度・成績の集計など）はホストが行う。ホストの取得の cron は、VPS で問題なく取れることを確かめてから止める。
計画の全体: VPS（取得）→ GCS → ホスト（後処理と予測）。

## できたもの（2026-11-13）
- VPS のタスク表をローカルに置く（`KEIBA_FETCH_TASK_STORE=file`、`/var/lib/keiba-fetch/tasks`。`src/ops/fetch_task.py` の FileTaskStore）。
- 予約を VPS の systemd timer で行う（`keiba-fetch-reserve.timer` 毎時 50 分、`keiba-fetch-raceday-tomorrow.timer` 20:30、`-today.timer` 07:20。
  `scripts/ops/vps/templates/`、導入は `vps_03_units.sh`）。GCP の reserve-fetch-tasks / reserve-race-day は使わない。
- VPS の fetch.env の雛形に、予測に要る取得のフラグ（直前オッズ・3 連単・前日夕の調教・新馬の基本情報）。既存の VPS の fetch.env は上書きしないので手で足す。
- ホストの常駐ランナー（raceday-runner・raceday-weight-runner・raceday-result-runner）の「取得しない」モード。
  置き換え先の取得（`src/ops/fetch_owner.py` の RUNNER_CORE）をすべて VPS が担当したとき、または `KEIBA_HOST_NO_FETCH=1` のとき、
  ランナーは取得をせず保存済みのデータで後処理（レース前の予測・馬体重の判定と予測・馬場速度と成績の集計）だけをする。
  今は担当（`config/fetch_owner.json` の agent_tasks）が空なので、どれも動き方は変わらない。

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
- 週次の後処理（めぐ指数・PG 同期・tracking_v2_eval・ドリフト）は、VPS の weekly-update では飛ばし、ホストの後処理だけのジョブに移す（未着手。2026-11-13 の週次の取得の後に）。

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
- 減る: VPS の 60 秒おきの GCS のタスク表の一覧（prod と影で月 約 8.6 万回）→ 0。月 約 −150 円。
- 予約（VPS）: 開催日の時刻表・出馬表を読む。1 日 数十件 × 約 6KB。月 1 円未満。
- ホストの馬体重の判定（取得しないモード）: GCS の出馬表を 90 秒おきに読み直す。1 日最大 36 レース × 24 回 × 約 10KB
  ≒ 864 回・8.6MB。× 30 日（最悪）＝ 月 約 2.6 万回・260MB → Class B 約 1.6 円 + 転送 約 4.6 円 ≒ 月 6 円。
- ホストのレース前の予測（取得しないモード）: VPS が保存したデータ（出馬表・馬柱・オッズなど 1 レース約 10 件・約 0.5MB）を GCS から読む。
  1 日 36 レース ≒ 18MB × 30 ＝ 540MB → 転送 約 10 円 + 操作 約 1 円 ≒ 月 11 円。
- 合計で月 約 −130 円（増えない）。どれも対象のレースのファイルだけを読み、全件の一覧はしない。

## 取得の網羅で気づいたこと
- 2026-09-22（祝日開催、24 レース）が、oikiri・paddock・trainer_comment・barometer の 4 カテゴリで同時に GCS に無い（fetch-status-unified-display の調査、2026-11-13）。祝日開催の取りこぼしとみられる。
