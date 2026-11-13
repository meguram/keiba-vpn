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

## 担当を VPS に渡す前に、そろえる必要があるもの（未着手）
1. **local_only のデータの受け渡し**: `race_lists`・`race_day_schedule`（HybridStorage の local_only。その機械のディスクだけ）と、
   JRA の馬場（`data/page_reference/cushion/`）。VPS が取るようになったら、VPS → GCS（ミラーの束・目録の方式）→ ホストに渡す。
   無いと、ホストのランナーは no-schedule で何もしない。
2. **VPS の T-15 の保存を確かめてから予測する**: 取得しないモードは、予定 + 猶予（`KEIBA_HOST_NO_FETCH_GRACE_SEC`、既定 180 秒）の後に必ず予測する。
   VPS の保存が遅れると、T-60 のオッズなどで予測してしまう。保存の時刻を確かめる仕組みが要る。
3. **ホストのディスク L2 キャッシュ**: 以前はホストが自分で保存して更新していた。VPS が保存するだけだと、ホストの L2 に古いオッズなどが残る。
4. **後処理だけのジョブの一覧**: raceday-eve（前夜の計算）・recent-retry の後処理（馬場速度・予測の成績）など、取得のジョブの中にある後処理を、
   取得と分けてホストに残す（VPS の recent-retry は後処理を GCP に任せる作りで、GCP Jobs は停止中）。
5. **毎日の馬の成績の一括（backfill-all-horse）と調教の夜間の補填（horse-training-backfill）** を VPS のタスクにする（250MB の枠で測る）。
6. GCP 側（停止中）の `requires_fetch_done`・`vps_watchdog` は GCS のタスク表を見ている。GCP Jobs を再開する場合は合わせて見直す。

## 費用の見積もり（CLAUDE.md の決まり。最悪の回数で）
- 減る: VPS の 60 秒おきの GCS のタスク表の一覧（prod と影で月 約 8.6 万回）→ 0。月 約 −150 円。
- 予約（VPS）: 開催日の時刻表・出馬表を読む。1 日 数十件 × 約 6KB。月 1 円未満。
- ホストの馬体重の判定（取得しないモード）: GCS の出馬表を 90 秒おきに読み直す。1 日最大 36 レース × 24 回 × 約 10KB
  ≒ 864 回・8.6MB。× 30 日（最悪）＝ 月 約 2.6 万回・260MB → Class B 約 1.6 円 + 転送 約 4.6 円 ≒ 月 6 円。
- ホストのレース前の予測（取得しないモード）: VPS が保存したデータ（出馬表・馬柱・オッズなど 1 レース約 10 件・約 0.5MB）を GCS から読む。
  1 日 36 レース ≒ 18MB × 30 ＝ 540MB → 転送 約 10 円 + 操作 約 1 円 ≒ 月 11 円。
- 合計で月 約 −130 円（増えない）。どれも対象のレースのファイルだけを読み、全件の一覧はしない。
