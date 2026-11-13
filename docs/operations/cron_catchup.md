# 毎日の定時ジョブの追いつき実行（PC が止まっていた日）

## なぜ
ホストの cron は、予定時刻に PC（WSL・コンテナ）が止まっていると、その日は走らない。
2026-11-12 は、PC の起動が 08:44 JST だった。そのため、07:00 JST の `export_tables_daily` をはじめ、朝の毎日のジョブが走らなかった。

## 仕組み（`src/ops/cron_catchup.py`）
15 分おきに `python -m src.ops.cron_catchup run` を呼ぶ。
許可リスト（`config/cron_catchup.json`）のジョブのうち、次のすべてを満たすものを、crontab のその行のコマンドで 1 回だけ実行する。
コマンドは `cron_run.sh` 経由なので、実行記録も通常と同じく残る。

1. **時刻の窓:** 直前の予定時刻 pf から 20 分以上たち、18 時間以内で、いまと pf が JST で同じ日付である。
   - pf は crontab の式から求める。cron の時刻（ホストは UTC）で評価する。
   - 日付が変わったら、翌日の予定に任せる。日付を見て動くジョブが、翌日の分を二重に処理しないようにするため。
2. **始まっていない:** pf 以降に、そのジョブが始まった記録（`data/local/meta/cron_runs/<job>.tsv`）が無く、いま動いてもいない。
   - 動いているかは、`.running` の印と ps（`cron_run.sh <job> --`）で見る。
   - `.running` の印の開始が PC の起動より前なら、停止で残った残骸とみなす（PID の再利用で誤認しない）。
3. **cron が止まっていた:** pf の時刻に cron が動いていなかった。判定は次のどちらか。
   - cron のデーモンの起動時刻が、pf より後（PC・コンテナの再起動）。起動から 5 分は待つ。
   - 15 分おきの通知（`slack-scrape-alerts`）の記録に、閉じた空白がある（スリープなど）。
     - 判定は `scrape_status_notify.host_down_windows` と同じものを使う。
     - 空白が閉じる（通知が再開する）まで待つので、復帰時に cron 自身が飛んだ予定を実行する分と重ならない。
   - cron が動いていたのに始まっていないジョブは追いつかない。Slack の「未開始」の通知で人が見る。
4. **まだ追いついていない:** この pf の分の印（`data/local/meta/cron_catchup/<job>.json`）が無い。
   - 印は実行の前に書く。失敗しても、同じ予定の分は二度と走らせない。

- **実行の順:** 複数のジョブは、予定時刻の順に 1 つずつ実行する。
- **直前の見直し:** 各ジョブの実行の直前に、2〜4 を見直す。前のジョブの実行中に本来の cron・手動で始まった分は走らせない。
- **重なり:** 前の追いつきが動いている間に次の 15 分が来ても、何もしない。
  - crontab の行の `flock -n` で、cron_run.sh にも入らない。入ると `cron-catchup.running` の印を上書きして消し、長い追いつきが監視から見えなくなるため。
  - 中のロック（`data/local/meta/cron_catchup/.lock`）は、手で `run` したときの二重起動も防ぐ。
- **環境:** ジョブに渡す環境は cron と同じ作りにする。既定の SHELL・PATH に、HOME・LOGNAME と、crontab のその行より前の環境変数の行を足す。
- **依存の順の限界:** 依存の順が保たれるのは、追いつき同士の間だけ。
  - たとえば 07:10 JST に起動すると、07:15 の `update_jockey_trainer_stats` は本来の cron で走る。
  - そのとき、`export_tables_daily` の追いつき（猶予の 20 分のあと）より先に走るので、前日のデータで集計される。翌日の予定で直る。

### 許可リスト（`config/cron_catchup.json`）
| ジョブ | 予定（JST） | 理由 |
|---|---|---|
| daily-race-lists | 07:00・17:00 | 開催一覧。後から取っても意味がある |
| export_tables_daily | 07:00 | 推論の履歴の flat の増分 |
| update_jockey_trainer_stats | 07:15 | 騎手・調教師の集計 |
| feature-accumulate | 07:40 | 履歴の特徴量（export の後） |
| rb-state-daily | 08:20 | v6 の日次状態 |

入れていないもの:
- **`serving-state-precompute`・`serving-state-profile`:** 2026-11-13 から `scripts/cron/install_serving_state_cron.sh`（タグ `KEIBA-SERVING-STATE`）で登録する。追いつきに入れるかは別に判断する（precompute は開催日の朝 07:35。無くても最初のレースで作られるので、追いつかなくても壊れない）。
- **開催の時刻に合わせて動くランナー（`raceday-*`）:** 遅れて起動すると、発走に合わせた取得の前提が崩れる。
- **ホストから GCS を読む量の多い `sync_pg_from_gcs`:** 必要なら、上の表に足すかどうかを判断する。

## 費用
- 追いつきが走るのは、逃した予定の代わりだけ。印で、同じ予定の分は 1 回に限る。
  そのため、各ジョブが 1 日に走る回数は予定どおり（1 日 1 回）を超えず、GCS・API の量は増えない。
- 最悪の場合として、追いつきが走った後に、本来の cron が同じ日にもう一度走ることを考える。予定は 1 日 1 回なので、同じ予定時刻はその日にもう来ない。
- 15 分おきの判定そのものは、ローカルのファイル（crontab・実行記録・/proc）を読むだけで、0 円。
  1 日 96 回 × 30 日 = 2,880 回だが、どれもローカルの読み込みだけ。

## 登録の手順（crontab の書き換え。ユーザーの承認後に行う）
```bash
cd /home/jovyan/work/keiba-vpn
sudo /opt/conda/bin/python3 -m src.ops.cron_catchup plan        # 何を追いつくかを、実行せずに確かめる（root の crontab を読む）
sudo bash scripts/cron/install_cron_catchup.sh                   # 15 分おきの行を登録（前の crontab は data/local/meta/cron_backups/ に控える）
sudo bash scripts/cron/install_cron_catchup.sh --remove          # 外すとき
```
- 登録後の動きは `logs/cron_catchup.log` で確かめる。
- 実行記録は `data/local/meta/cron_runs/cron-catchup.tsv` に残る。
- **登録済み（2026-11-13 10:19 JST、ユーザーの承認後）**。root の crontab に `KEIBA-CRON-CATCHUP` の行が 1 行ある。
  - 登録した時点で、11/13 の朝のジョブは定時に走っていた。11/12 の取りこぼしは日付が変わっていて対象外なので、追いつきで走ったものは無い。
  - 最初の回（10:30 JST）は終了コード 0 で、何も実行しなかった（`cron-catchup.tsv` で確認）。

## 対象のジョブを足すとき
- `config/cron_catchup.json` の `jobs` に足す。
- 毎日の定時（`分 時 * * *`）で、`raceday-` で始まらないことは、`tests/ops/test_cron_catchup.py` が確かめる。
- ホストから GCS を読む・書く量の多いジョブを足すときは、上の費用の考え方（1 日 1 回を超えない）を、そのジョブについて確かめる。

## weekly-update（金 18:00 JST）を許可リストに入れない理由（2026-11-12 の判断）
「同じ JST の日付のうち 1 回だけ」なので、回数と netkeiba へのアクセスの総量は変わらない。それでも、次の理由で今は入れない。

1. **並走の相手と時間帯が変わる。**
   - 週次は、実績で 10〜324 分かかる（10/30 は 18:00〜23:24）。キュー待ちの上限は 48 時間ある。
   - 追いつきは 23:59 JST まで始まりうるので、遅く始まると、本来の 18:00 の回なら重ならない処理と重なる。
     - 00:00〜04:00 の `backfill-20xx-fast`（auto_scrape には譲らない）
     - 00:00 の `sync_pg_from_gcs`
     - 06:00 の `backfill-all-horse`、06:10 の `recent-retry`
   - 補填（`--phase past`）は、既定で譲らずに並走する（`BACKFILL_PAST_YIELD=0`）。抑えているのは取得スロットだけ。
   - 調教の取得の `wait_ok` は 1 時間で諦めるので、その夜の取得が打ち切られる回も増える。
2. **週次に依存する金曜のジョブとの順序が崩れる。**
   - `horse-name-index`（金 18:30。週次の開始後が前提。金曜以外は何もしない）は、週次の追いつきより先に走るか、PC が止まっていれば走らない。その週の成長曲線の更新が抜ける。
   - `barometer-retry`（金 19:30）も、週次より前に走ると取り直しの意味が薄れる。
3. **追いつきのロックを長く持つ。**
   - 週次の間（最大 48 時間）、ほかのジョブの追いつきが止まる。
   - その間に JST の日付が変わると、翌朝の毎日のジョブの追いつきは失われる。

### 金曜に PC が止まっていたときの対応（当面の手順）
- 週次を手で 1 回だけ回す（その夜のうち、早い時刻に）。
  ```bash
  cd /home/jovyan/work/keiba-vpn && TZ=Asia/Tokyo sudo bash scripts/cron/cron_run.sh weekly-update -- bash scripts/cron/yield_to_rescrape.sh weekly-update -- bash scripts/cron/run_auto_scrape_logged.sh /home/jovyan/work/keiba-vpn weekly-update logs/weekly_update.log
  ```
- 回せなかった週のめぐ指数は、次の週次の欠けの補完（`megu_missing_dates`、直近 60 日）で埋まる。
- 確定結果の取り直しは、次の週次の対象（直近 10 日）から外れると行われない。

### 入れるなら必要なこと
- ジョブごとに、追いつきの締め切りの時刻を設定できるようにする（例: weekly-update は 21:00 JST まで）。
- 依存するジョブ（`horse-name-index`・`barometer-retry`）も、週次の後に追いつく。
- 長いジョブは追いつきのロックの外で動かす（ほかのジョブの追いつきを止めないように）。
