# 出馬表の段階的な収集と「常に最新」の表示（entries-sync）

JRA の公示に沿って、未来のレースの出馬表を**段階ごとに**取り、画面には**いまの段階と最終更新**を付けて出す。
（2026-10-21 に「10/24・25 の出馬表が何も表示されない」報告を受けて整備した。）

## 1. 空表示の原因（2026-10-21 水曜時点の調査）

| 層 | 状態 |
|---|---|
| レース一覧 `race_lists/20261024.json`・`20261025.json` | **あった**（各 36 レース、頭数 `entries_count` つき。`daily-race-lists` が 07:00/17:00 JST に取得） |
| 出馬表 `race_shutuba` | **無かった**。取得は前日 18:00 の `raceday-eve` と各 R T-15 だけで、週の途中（日曜夕の特別登録〜木曜）には取りに行く仕組みが無かった |
| API `/api/race/{id}/predictions` | 出馬表が無いので `no_entries`。さらに、出馬表があっても**馬番が 0 の段階の馬を捨てていた**（`entries_only_predictions._entries` が馬番必須） |

→ 画面には馬が 1 頭も出なかった。

## 2. JRA の公示の流れと、netkeiba に出るもの

| 段階 | 時期（既定。実測で補正） | 出馬表に入っているもの | 内部名 |
|---|---|---|---|
| 特別登録 | 開催の約 6 日前の日曜夕（18 時ごろ） | 重賞・特別競走の登録馬（馬名・一部の騎手）。平場は未定 | `registered` |
| 登録馬の更新 | 月〜水 | 登録馬・騎手・斤量が順次入る | `registered` |
| 出走馬の決定 | 木曜（12 時以降を既定） | 平場も出る。出走馬が決まる。**枠番・馬番はまだ** | `entries` |
| 出馬表の確定 | 前日: 金曜 16 時（土曜開催）／土曜 10 時（日曜開催） | **枠番・馬番が入る**。取消・騎手変更は以後も起きうる | `final` |
| 馬体重の発表 | 当日（発走の約 1 時間前） | 馬体重・増減 | `weighed` |

### 実測（2026-10-21 水曜 19:30 JST。ホストで保存せずに取得して集計だけ確認）

| レース | 頭数 | 騎手 | 斤量 | 発走時刻 | 枠・馬番 | 馬体重 |
|---|---|---|---|---|---|---|
| 京都 1R 2歳未勝利（平場） | 14 | 8/14 | 0/14 | なし | なし | なし |
| 京都 11R 菊花賞（G1） | 20 | 19/20 | 20/20 | 15:40 | なし | なし |
| 新潟 12R 3歳以上1勝クラス（平場） | 34（登録。抽選前） | 23/34 | 0/34 | なし | なし | なし |

→ **水曜時点で、netkeiba はすでに登録馬の出馬表を返している**（枠・馬番だけが未確定）。木曜・金曜・土曜の中身は、最初の週の運用で実測して
このページの表を更新する（`src/scripts/data/entries_probe.py` で件数だけ確認できる）。

## 3. 収集スケジュール（JST。設定: `config/entries_schedule.json`）

cron は **10 分おき**に `entries-sync` を起動する。起動時に「いまの時刻が checkpoint の窓（25 分）に入っていて未実施か」だけを判定し、
該当しなければ数秒で終了する（netkeiba には触れない）。

| 曜日 | checkpoint（JST） | critical（確定した出馬表も取り直す） |
|---|---|---|
| 日 | 18:30、19:30 | — |
| 月・火・水 | 08:30、20:00 | — |
| 木 | 12:30、15:30、17:30、19:30、21:30 | 12:30、15:30、17:30 |
| 金 | 10:00、16:05、16:30、17:15、18:30 | 16:05、16:30、17:15 |
| 土 | 10:05、10:30、11:15、12:30 | 10:05、10:30、11:15 |

- 対象日: 今日の**翌日から 14 日先まで**の開催日（`race_lists` にある日）。**今日のレースは対象外**（開催日ランナー `raceday-runner` が担当）。
- 取り直す条件（`entries_phase.should_refetch`）: 未取得／レース一覧の頭数が変わった／段階が `final` 未満（毎回）／
  `final`・`weighed` は**前日・当日の critical** でだけ（取消・騎手変更の反映）。
- 一覧が不完全な日（特別登録の段階など）と critical では、`race_list` も取り直す（1 日 1 リクエスト）。
- 1 回の所要: 36R×2日 ≒ 72 リクエスト ≒ 7 分（キュー経由。DEC-023・クライアントのスロット・間隔はキュー側が守る）。
- 既存の `raceday-eve`（前日 18:00）は**最終確認**として残す（追走難度の前夜計算などはこちら）。

## 4. 保存するもの（最新＋履歴）

- `race_shutuba`（GCS／ローカル）: 最新の出馬表で上書き（従来どおり）。馬番が 0 の段階でもスキーマに通る（`horse_number` は int）。
- `data/local/meta/entries_state/<race_id>.json`（**履歴はここ**）:
  `phase`・`label`・`updated_at`（中身が変わった時刻）・`checked_at`（最後に確認した時刻）・`signature`・
  `n_entries / n_numbered / n_jockey / n_weight`・`history`（段階と内容の変化。最大 40）・`changes`（取消・追加・騎手変更・斤量変更。最大 60）。
- `data/local/meta/entries_sync_state.json`: 実施済みの checkpoint（二重実行の防止）と直近の結果。
- 内容の署名は馬体重・オッズを含めない（取消・騎手・斤量・馬番の変化だけを検知）。

## 5. 画面での見え方

- 「今週の AI 予測」: レース一覧の上に**その日の進み具合**（例: `出馬表: 特別登録（暫定） 12 / 出走馬決定 24`）、
  選んだレースの見出しに**段階バッジ**（特別登録（暫定）／出走馬決定／出馬表確定／馬体重発表済み／出馬表待ち）と
  **更新 10/23 16:05**、確定前は「頭数・騎手・枠順は確定前です」、取消・騎手変更があれば「取消 1頭・騎手変更 2件」。
- 馬番が無い段階の表は、枠・馬番を「-」で出し、**予測は出さない**（右上は「確定後に予測」。予測は出走馬と枠順の確定後）。
- レース詳細: 見出しに同じバッジ。出馬表タブは馬番なし（—）で馬が並ぶ。「AI予測を実行」は確定後まで押せない。
- API: `entries_status`（`/api/race/{id}/predictions` に追加。後方互換）と、`/api/race-list/{date}` の各行の `entries_phase / entries_label / entries_updated_at`。
  馬番なしの応答は `provisional_entries: true`、各馬に `horse_number: null` と `horse_number_provisional: true`。

## 6. 導入（ホスト）

反映ファイル（ホストでは `git pull` しない。ファイル単位で配置し、退避を取る）:

- 収集: `src/scraper/entries_phase.py`、`src/scraper/entries_sync.py`、`src/scraper/auto_scrape.py`（`entries-sync` タスク）、`config/entries_schedule.json`、
  `scripts/auto_scrape.sh`、`scripts/cron/install_entries_sync_cron.sh`、`src/scraper/sla_spec.py`、`src/scripts/data/entries_probe.py`
- API: `src/api/entries_status.py`、`src/api/entries_only_predictions.py`、`src/api/app.py`（差分適用。3 か所）
- フロント: `frontend/components/EntriesStatus.tsx`、`frontend/lib/entriesStatus.ts`、`frontend/app/weekly-predictions/page.tsx`、`frontend/app/race/[id]/page.tsx`

手順:

1. 上のファイルを配置 → API を再起動（キュー running/queued=0 のとき）→ フロントを `frontend_build.sh stg` でビルドして切替。
2. **今週分の穴埋め**（10/24・25）: 登録後すぐ 1 回、手動で実行する。
   ```
   ssh keiba-remote 'cd <HOST_REPO> && sudo -n bash -c "set -a; . ./.env; set +a; KEIBA_ENV=stg PYTHONPATH=. /opt/conda/bin/python3 -m src.scraper.entries_sync --force"'
   ```
   先に `--dry-run` で取り直す予定のレースだけ確認できる（取得しない）。
3. cron 登録（root）: `sudo -n bash scripts/cron/install_entries_sync_cron.sh <HOST_REPO>`（タグ `KEIBA-ENTRIES-SYNC`、UTC `*/10 23,0-13`）。外すときは `--remove`。
4. 確認: `data/local/meta/entries_state/` にレースごとのファイルができる、`logs/entries_sync.log`、
   `curl localhost:23220/api/race-list/20261024` の各行に `entries_phase` が付く。

ロールバック: cron を `--remove`、`app.py` の差分を戻して再起動、フロントを戻して再ビルド（`entries_status` が無くても画面は従来どおり動く）。

## 7. 障害時の挙動

- netkeiba が出馬表を返さない／パース 0 件: 保存せず、状態も更新しない（次の checkpoint で再試行）。
- キューが詰まって待ちきれない（25 分）: その checkpoint は実施済みにして次へ（`queue_ok: false` を記録）。画面は前回の状態のまま。
- 取消などで頭数が減る: 取り直した出馬表が最新。差分は `changes` に残り、バッジに「取消 N頭」が出る。
- 状態ファイルが無いレース: 保存済みの出馬表から段階だけ判定して表示する（更新時刻は無いことがある）。
- 二重起動: ファイルロック（`entries_sync.lock`）で見送り。

## 8. 未確認（最初の週に実測して更新する）

- 木曜の出走馬決定が netkeiba に出る時刻（既定は 12:00 以降を `entries`）。
- 金曜 16 時／土曜 10 時の確定で、枠番・馬番が入る正確な時刻と、netkeiba の反映の遅れ。
- 特別登録の公示直後（日曜夕）の、登録馬の数と騎手の有無。
- `entries_decided_at`・checkpoint の時刻は `config/entries_schedule.json` で変更できる。
