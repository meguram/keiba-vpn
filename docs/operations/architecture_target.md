# 将来構成と移行設計: UI は ConoHa VPS、常駐処理は GCP、ホスト PC はモデリングのみ

**作成**: 2026-10-21 / **種別**: 設計書（調査と文書のみ。ホスト・GCP・ConoHa・Cloudflare の設定は変更していない）
**予算制約（ユーザー回答 2026-10-21）**: GCP + Cloudflare + ConoHa の合計で **月 2,000 円以内**。計算とモニタリングは続ける。
**為替の仮定**: 1 ドル = 150 円（最新値は要確認。`config/cost_assumptions.json` の `fx_jpy_per_usd` を変えて再計算できる）
**金額の再計算**: `python -m src.scripts.ops.cost_estimate --scenario all --format md`（前提は `config/cost_assumptions.json`）
**関連**: [vps-gcp-responsibilities.md](./vps-gcp-responsibilities.md)・[cost-estimate.md](./cost-estimate.md)・[deployment-vps-vs-gcp.md](./deployment-vps-vs-gcp.md)（2026-10-02 の旧設計。本書が**上書き**する点は 1.3）・[go_live.md](./go_live.md)・[ads_architecture.md](./ads_architecture.md)・[maintenance_deploy.md](./maintenance_deploy.md)・[gcp-cloud-run-jobs.md](./gcp-cloud-run-jobs.md)

---

## 0. 結論（要約）

1. **月 2,000 円以内で実現できる最小構成（第一案 A）は、計算の下限で約 260 円、上限で約 2,050 円**（為替 150 円）。コストのほぼ全部が「GCS（東京）の保存量と書き込み回数」で、Cloud Run の計算は**無料枠にほぼ収まる**。上限が予算を僅かに超えるのは GCS の保存を 100GB・書き込み 60 万回/月と置いた場合で、書き込みをまとめる（Parquet/JSONL 化）・古い生データを間引く、で約 1,500 円に収まる（4.3）。
2. **予算を超える項目は 4 つ**: API の最小インスタンス 1（約 2,900 円〜）、Cloud SQL（約 1,800 円）、東京の e2-micro 常駐機（約 1,800 円）、固定 IP の Cloud NAT（約 4,800 円）。**ユーザー方針を文字どおり（API 常時起動 + DB + 取得用常駐機 + 固定 IP）に実装すると約 6,800〜13,000 円/月**で、予算の 3〜6 倍になる。
3. **「GCP 上で常駐」は予算と両立しにくい。** GCP の常時起動は費用が時間で積み上がる（e2-micro の Always Free は米国 3 リージョンのみで東京は有料）。前払い済みで追加費用 0 の **ConoHa VPS（2GB）が「常駐」の最安の置き場所**。そこで本書は、**GCP を「使った時間だけ課金のサーバーレス（Cloud Run の API は最小 0、ジョブ、GCS）」に使い、軽い常駐（UI、cloudflared、スクレイピングの runner、cron 司令塔）は VPS に置く**構成を第一案にする。ユーザー方針との差は 1.2 に明記する。
4. **最大のリスクはスクレイピング**（netkeiba / SmartRC / JRA）。クラウドの IP は遮断されやすく、固定 IP（Cloud NAT）は予算外。ConoHa VPS の IP は **netkeiba に対して一度も試験していない**。推奨は「取得系は VPS（固定 IP・常駐・前払い済み）に置く。**事前に VPS で小規模な試験を行い、通れば移す。通らなければ取得だけホスト PC（自宅 IP）に残す**」。GCP の Cloud Run Jobs に取得を置く案 B は、遮断の恐れと同時 1 本ルール（DEC-023）の実現が難しく、第一案にしない（4.2）。
5. **最初の一歩（低リスク・追加費用 0）**: VPS に cloudflared と Next.js（ホストで作ったビルド成果物）を載せ、`app.megu-keiba.com` の向き先を VPS に切り替える。API は当面ホストのトンネル経由のまま。詳細は 7.1。
6. **ホスト PC が止まっても UI・API・予測・取得が続く**ことが目的。そのために (a) 予測の生成を GCP の Jobs（Cloud Tasks の予約実行）に移し、(b) 学習済みモデルとデータの正本を GCS に置き、(c) ホスト PC はモデリングと GCS への配布だけにする（フェーズ 3〜5）。取得だけは VPS が担う（VPS の稼働が可用性の前提になる）。
7. **決めていただくこと**は 9 章（予算の上限の扱い、GCP プロジェクトとリージョン、取得を残す場所、可用性の目標、データ保持など）。

> 金額の根拠: Cloud Run の無料枠（月 200 万リクエスト・18 万 vCPU 秒・36 万 GiB 秒）、R2 の料金（10GB 無料・egress 無料）は公式ページで 2026-10-21 に確認した。**Cloud Run / GCS / Scheduler の単価は公式ページの本文を取得できなかったため作成者の知識ベース（要確認）**。確認できていない前提は `--list-unverified` で一覧できる（付録 A）。

---

## 1. 前提と旧設計との関係

### 1.1 ユーザー方針（2026-10-21）
1. ページ表示（UI）は ConoHa VPS（160.251.250.252、2GB / 3 コア / SSD 100GB、36 か月前払い済み、2028/05/18 満了、停止しても料金は変わらない）で常駐させる。
2. UI 以外の常駐処理は GCP 上で常駐させる。
3. ホスト PC（`keiba-remote`）はローカルで実行するモデリングの更新などだけを行う。
4. 予算: GCP + Cloudflare + ConoHa の合計で月 2,000 円以内が望ましい。

### 1.2 本書が方針から外れる点（理由つき）
| 方針 | 本書の第一案 | 理由 |
|---|---|---|
| 「UI 以外の常駐処理は GCP」 | **取得（スクレイピング）の runner と cron の司令塔は VPS に残す**。API・予測生成・集計・通知は GCP（サーバーレス） | ① GCP の常時起動は予算を食う（e2-micro 東京 約 1,800 円、API 常時起動 約 2,900 円〜）。② 取得は送信元 IP が固定でないと遮断されやすく、固定 IP（Cloud NAT）は約 4,800 円。③ VPS は前払い済みで追加費用 0、メモリ 2GB でも取得 runner（約 150〜300MB）は載る |
| 「API は GCP」 | **API は Cloud Run（最小インスタンス 0）**。コールドスタート（数秒）は VPS 側のキャッシュで隠す | 最小インスタンス 1 は約 2,900 円〜で予算外 |
| 「DB は GCP」 | **Cloud SQL は使わない**（約 1,800 円で予算の 9 割）。GCS（Parquet / JSON）+ 必要なら Firestore 無料枠 | 現行も実質ファイルが正本。PostgreSQL は `sync_pg_from_gcs` の補助用途 |

この差が受け入れられない場合の比較は 4.2（案 B〜F）。**予算を優先するか、方針の文字どおりを優先するかは 9 章で決めていただく。**

### 1.3 旧設計（2026-10-02）から変わる点
| 旧設計 | 本書での扱い |
|---|---|
| Cloud SQL（stg/prod 共用 1 台、約 1,650〜1,950 円/月）を確定 | **取りやめ**（予算超過）。GCS + ファイルで代替 |
| 推論は「まず VPS、メモリ不足なら GCP」 | **GCP（Cloud Run Jobs / Cloud Tasks 予約）に移す**。ホスト PC 停止中も予測を作り続けるため。低メモリ構成（約 340MB・0.6 秒）で成立 |
| 学習 PC = データ準備の全部 + 学習 | **ホスト PC = モデリング（特徴量の再構築・学習・評価）のみ**。日々の増分特徴量は GCP の Jobs |
| 取得は VPS（当日分）+ 学習 PC（過去分） | 当日分は VPS のまま。過去分の長期バックフィルは**ホスト PC の回線**（夜間）。取得の置き場所は VPS 試験の結果で最終決定（フェーズ 4） |
| 合計 約 $18/月（約 2,700 円） | **約 260〜2,050 円/月**（4 章） |

---

## 2. 現状の棚卸し（ホスト PC の責務）

ホスト（`keiba-remote`）は 24 コア / RAM 90GB / ディスク 7.3TB（空き 6.0TB）で、**すべての常駐処理と定期実行を 1 台で担当**している。以下は 2026-10-21 時点の実測（読み取りのみ）。

### 2.1 常駐サービス
| サービス | ポート / 実体 | メモリ（RSS 目安） | 備考 |
|---|---|---|---|
| Next.js（本番ビルド） | :23200 `next-server` | 約 115MB | UI。`/api/*` を FastAPI/Flask へ rewrites（`frontend/next.config.js`） |
| FastAPI | :23220 `main.py`（uvicorn **3 ワーカー**） | 1 ワーカー約 270〜540MB（実測 2026-10-21）→ **合計 約 1.1GB** | 予測・期待値・推奨などの API、スクレイピングキューのワーカー、各種 daemon スレッド。**VPS の 2GB に同居させると Next.js や取得 runner と合わせて余裕がなくなる**。Cloud Run では 1 ワーカー・2GiB にする |
| Flask `/api/v1` | :23210 `main.py --flask-api` | 約 87MB | 分析系の API |
| 監視ポータル | :23900 `src.monitor.app` | 約 76MB | 開発者専用 |
| cloudflared | トンネル `meguai-stg` | 約 40MB | `app` / `monitor` を Cloudflare Access 配下で公開 |
| PostgreSQL 14 | ローカル | 共有メモリ込み約 0.8GB | `sync_pg_from_gcs` の補助。FastAPI の一部が参照 |
| MLflow | 127.0.0.1 | 約 480MB | 学習の記録。**ホスト PC に残す** |
| ウォッチドッグ / ensure_monitor / ensure_postgres | cron（3 分 / 5 分） | — | 死活監視 |
| `keiba-remote-access` | cron（2 分） | — | sshd / chisel / ngrok（ホストへの遠隔アクセス用） |
| 開発ツール（Cursor サーバーなど） | — | 約 1.5GB | 運用外 |

### 2.2 定期ジョブ（root の crontab、JST）
CPU・メモリは「重い / 軽い」で示す。外部アクセスの凡例: **NK**=netkeiba（ログイン依存あり）、**SR**=SmartRC、**JRA**=JRA 公式、**GCS**=Cloud Storage。

| 分類 | ジョブ | 時刻（JST） | 所要・負荷 | 入力 → 出力（`data/local/` 他） | 外部 | 依存 |
|---|---|---|---|---|---|---|
| **取得（時間厳守）** | `raceday-runner` | 開催日 07:30 起動、数時間常駐 | 軽い（sleep と取得） | 発走時刻表 → T-60 / T-15 のオッズ・バンドル、`KEIBA_ODDS_LATE_SNAPSHOTS` で T-10/5/2、3連単 | NK | 出馬表 |
| | `raceday-result-runner` | 開催日 07:30 | 軽い | 確定結果・払戻 | NK | — |
| | `raceday-weight-runner` | 開催日 08:00〜 | 軽い + 予測 | 馬体重の発表を検知 → **予測**（v3/v4/v5 shadow）→ `predictions_*` | NK + GCS | モデルバンドル |
| | `raceday-eve` | 毎日 18:00 | 中（約 50 分: 調教の取り直し約 500 頭を含む） | 出馬表・馬柱・調教・前夜の precompute | NK, SR | — |
| | `raceday-evening` / `raceday-comment` | 17:30 / 20:00・07:40 | 軽い | 速報まとめ、厩舎コメント | NK | — |
| | `weekly-update` / `horse-name-index` | 金 18:00 / 18:30 | 中 | 週次更新、馬名インデックス | NK | — |
| | `daily-race-lists` | 07:00 / 17:00 | 軽い | レース一覧 | NK | — |
| | `jra-baba-morning` | 05:00〜08:50 の 10 分おき | 軽い | 馬場情報（含水率・クッション値） | JRA | — |
| | `recent-retry` / `barometer-retry` | 06:10・19:40 / 08:30（金 19:30） | 軽い | 未取得の取り直し | NK | — |
| **取得（長時間・夜間）** | `backfill-*`（約 12 本） | 00:00〜09:00 ほか | 軽い（CPU）・長時間 | 年度別の過去データ | NK | — |
| | `new-horse-pedigree` | 21:00 | 軽い | 新馬の 5 世代血統 | NK | — |
| | `horse-remarks-daily` | **03:00** | 軽い | 馬の備考 | NK | メンテ窓と重なる |
| | `sporthorse-births-nightly` | 22:00 | 軽い | 誕生日データ | 外部 | — |
| | `horse-training-backfill` | 月〜木 22:00（1 晩 2,200 頭） | 軽い | 調教の過去分の取り直し | NK | 他の取得中は待つ |
| **予測・特徴量** | `export_tables_daily` | 07:00 | 中 | 推論用の履歴テーブル | GCS | — |
| | `feature-accumulate` | 07:40 | 中 | 履歴特徴の蓄積（skew 記録） | — | 学習は 2026 年を使わない |
| | `serving-state-precompute` / `-profile` | 07:35 / 08:10 | 中 | `data/local/inference/state/`（約 62MB）の更新 | — | features |
| | `lap-forecast`（eve / morning） | 19:30 / 07:30 | 中 | `inference/lap_forecast/`（約 11MB） | — | 出馬表 |
| | `tracking-v2`（前夜保存の週次突き合わせ） | 月 08:30 | 中 | `inference/tracking_v2/`（約 18MB） | — | — |
| | `update_jockey_trainer_stats` | 05:30 | 中 | 騎手・調教師統計 | GCS | — |
| | `sync_pg_from_gcs` / `ensure_postgres` | 00:00 / 5 分おき | 中 | PostgreSQL 補助 | GCS | — |
| **集計・通知・監視** | `slack-results` / `scrape-alerts` / summary / coverage | 土日 15:00〜20:59 の 20 分おき / 15 分おき / 21:30・07:30 / 日 08:00 | 軽い | `prediction_results/`（約 11MB）→ Slack | Slack | — |
| | `drift-monitor-weekly` / `frozen-tracking-weekly` / `odds-drift-weekly` | 月 10:00 / 月 10:30 / 日 22:00 | 中（特徴量の抜き取りあり） | `meta/drift_monitor/` ほか | — | 予測・確定結果 |
| | `page_quality_check` / `rotate_logs` | 08:00 / 04:30 | 軽い | — | — | — |
| | `maintenance-deploy` | 毎日 03:05 | 軽い（pending が空なら即終了） | `data/local/deploy/` | — | メンテナンス窓 |
| **モデリング（ホスト PC に残す）** | 学習・評価・特徴量の再構築（ノートブック・`src/scripts/data/tp*`） | 随時（重い） | 24 コア・数十 GB | `data/local/modeling/`（約 13GB）、`features/`（約 2.0GB） | — | — |

### 2.3 データ量（概算）
| 領域 | サイズ | 移行後の置き場所 |
|---|---|---|
| `data/local/modeling`（学習・評価・バンドル。`training_pipeline` 5.0GB、`final_v4` 321MB、`final_v5` 約 330MB） | 約 13GB | **ホスト PC**（正本）。配布する推論バンドルだけ GCS（約 170〜320MB/版） |
| `data/local/features`（特徴量ストア） | 約 2.0GB | ホスト PC（再構築）＋ GCS（Jobs が読む増分・必要列のみ） |
| `data/local/horse_pedigree_5gen / 10gen` | 約 0.47GB / 1.0GB | GCS（Jobs が読む） |
| `data/page_reference`（UI ポータブルバンドル） | 約 3.0GB | GCS（正本）＋ VPS に必要分のキャッシュ |
| `data/local/inference`（状態・予測・ラップ・追走難度） | 約 144MB | GCS（正本）＋ 公開スナップショットは R2 |
| `data/local/horse_training` | 約 135MB | GCS |
| `data/local/meta`（data_health、監査ログなど） | 約 125MB | ホスト PC（運用ログ）。必要なものは GCS |
| GCS 上の生データ（取得 JSON の正本。派生 11 カテゴリ含む） | **実サイズは未確認**（GCP に接続していないため）。見積りは 20GB（下限）〜100GB（上限） | GCS（東京） |

---

## 3. 移動先の割り当て

### 3.1 割り当て表（第一案 A）
| 機能 | ConoHa VPS（前払い済み） | GCP（サーバーレス） | Cloudflare（無料枠） | ホスト PC（残す） |
|---|---|---|---|---|
| UI（Next.js 本番ビルド） | **◎ 常駐**（成果物をホストで作って配布） | — | Access（メール認証）・キャッシュ | ビルド |
| トンネル・認証 | cloudflared（`app.`・`monitor.`） | — | Tunnel・Access（50 ユーザーまで無料） | — |
| API（FastAPI/Flask） | リバースプロキシ（ローカル nginx/caddy）のみ | **Cloud Run（最小 0、1 ワーカー、2GiB、最大 3 インスタンス）** | api.（任意） | 開発・検証用に残す |
| 予測生成（前夜保存・馬体重検知後の予測・ラップ・追走難度・shadow v4/v5） | 起動の合図（取得 runner が検知 → Cloud Tasks / Jobs を呼ぶ） | **Cloud Run Jobs（Cloud Tasks の予約実行で T-60/T-15 を厳守）** | — | — |
| 集計（予測成績、騎手・調教師統計、ドリフト、凍結追跡、データヘルス） | — | **Cloud Run Jobs** | — | — |
| 通知（Slack） | cron（軽い通知） | Jobs からも可 | — | — |
| 取得（開催日 runner・前日取得・夜間・調教取り直し） | **◎ 常駐（固定 IP）** ※試験が前提 | （案 B: Jobs）。JRA 馬場など IP 非依存の軽い取得は Jobs も可 | — | 長期バックフィル（夜間・自宅 IP）、VPS が遮断された場合の代替 |
| データの正本 | 必要分のキャッシュ | **GCS（東京）**（生データ・特徴量・血統・予測・バンドル） | R2: 公開用スナップショットのみ | 作業用コピー |
| DB | なし（PostgreSQL は廃止） | Firestore 無料枠（任意）／ファイル | — | — |
| 監視ポータル | 軽いので VPS に移す（約 76MB） | — | `monitor.`（Access） | — |
| 学習・評価・特徴量の再構築・ノートブック・MLflow | — | — | — | **◎** |
| 学習済みモデルの配布 | 取得のみ（pull） | GCS から読む | — | **GCS へアップロード** |
| 公開サイト（広告用・将来） | — | — | Pages（`www.`）+ R2 スナップショット | スナップショット生成の元データ |

### 3.2 VPS（2GB）のメモリ予算
| 項目 | 見積り |
|---|---|
| OS・SSH・cron | 約 250MB |
| Next.js 本番（`next start`） | 約 120〜200MB |
| cloudflared | 約 40MB |
| リバースプロキシ（caddy） | 約 30MB |
| 取得 runner（Python、`nice`、同時 1 本） | 約 150〜300MB |
| 監視ポータル | 約 80MB |
| swap（SSD 上に 2GB を追加 → 費用 0） | 余裕 |
| **合計** | **約 0.8〜1.1GB（余裕あり）** |

> **`next build` を VPS で実行しない**（2GB ではメモリ不足の恐れ）。ビルドはホスト PC（または CI）で行い、成果物を R2/GCS に置いて **VPS が自分で取りに行く（pull 型）**。受信の SSH を開けずに済む。

### 3.3 ホスト PC が止まっても続くか
| 機能 | ホスト停止中 | 条件 |
|---|---|---|
| UI の表示 | 続く | VPS とトンネルが生きている |
| API・予測の閲覧 | 続く | Cloud Run + GCS |
| 予測の生成 | 続く | 取得 runner（VPS）→ Cloud Tasks → Jobs |
| 取得 | 続く | VPS が遮断されていない（試験で確認）。遮断時はホスト PC に依存 |
| モデルの更新 | 止まる（想定どおり） | ホスト PC のみ |

---

## 4. 予算制約つきの構成と月額

### 4.1 第一案 A: 月 2,000 円以内（GCP + Cloudflare + ConoHa 前払い済み）
計算は `python -m src.scripts.ops.cost_estimate`（前提は `config/cost_assumptions.json`）。為替 1 ドル = 150 円。

| 項目 | 下限（円/月） | 上限（円/月） | 前提（下限 → 上限） |
|---|---:|---:|---|
| Cloud Run（API） | 0 | 約 110 | 3 万 → 40 万リクエスト、平均 0.3 → 0.5 秒、1 vCPU / 2GiB、コールドスタート 150 → 600 回（1 回 8〜10 秒）。**無料枠（18 万 vCPU 秒・36 万 GiB 秒・200 万リクエスト）にほぼ収まる** |
| Cloud Run Jobs（予測生成・集計） | 0 | 約 560 | 5 万 → 16 万 vCPU 秒、12 万 → 42 万 GiB 秒（開催日 8 日/月 × 予測 3 回 + 日次・週次の集計）。無料枠は API と共通で消費 |
| GCS（東京） | 約 250 | 約 1,180 | 保存 20 → 100GB、書き込み 15 万 → 60 万回、読み取り 80 万 → 400 万回、外部への送信 1 → 8GB。**東京は無料枠の対象外**（Always Free の 5GB は米国 3 リージョンのみ） |
| Cloud Scheduler | 0 | 約 135 | 3 → 12 ジョブ（無料 3 ジョブ。VPS の cron を司令塔にすれば 3 のまま） |
| Secret Manager / Artifact Registry / Logging | 約 10 | 約 70 | 有効バージョン 6〜10、イメージ 1〜2.5GB、ログ 5〜20GiB（無料 50GiB） |
| Cloudflare（R2・Pages・Tunnel・Access） | 0 | 0 | R2 は 10GB・Class A 100 万・Class B 1,000 万まで無料。egress 無料。Access は 50 ユーザーまで無料 |
| ConoHa VPS | 0 | 0 | 前払い済み。停止しても料金は変わらない（月額換算は契約金額 ÷ 36 を別途記入） |
| **合計** | **約 260** | **約 2,050** | 下限は予算の 13%、上限は予算を僅かに超える |

**コストの 6 割は GCS**（保存量・書き込み回数・送信量）。上限を予算内に収めるレバー（`--set` で試算）:
| 対策 | 上限の月額 |
|---|---:|
| そのまま | 約 2,050 円 |
| 書き込みをまとめる（小さな JSON を日次の Parquet / JSONL にして 60 万 → 20 万回） | 約 1,750 円 |
| + 保存量を 100 → 60GB に（古い生データの圧縮・間引き） | 約 1,610 円 |
| + Scheduler を VPS の cron で代替（3 ジョブのまま） | **約 1,480 円** |
| 為替が 1 ドル = 160 円のとき（そのまま） | 約 2,190 円 |

### 4.2 比較: 超過するが利点がある案（同じ前提で計算）
| 案 | 内容 | 下限 | 上限 | 予算 | 利点 / 欠点 |
|---|---|---:|---:|---|---|
| **A**（第一案） | 上記 | 260 | 2,050 | ○（レバーで内） | 最安。API のコールドスタート（数秒）を許容 |
| **B** | A + 取得も Cloud Run Jobs（GCP の IP） | 260 | 2,830 | △ | 全部 GCP。**IP 遮断のリスク**・同時 1 本ルールの実現が難しい |
| **C** | A + API の最小インスタンス 1（1 vCPU / 2GiB） | 3,160 | 4,720 | × | コールドスタートなし。アイドル課金が月 約 2,900 円 |
| **D** | A + Cloud SQL（db-f1-micro） | 2,060 | 3,850 | × | DB が欲しい場合。現状は不要 |
| **E** | A + 東京の e2-micro 常駐機（外部 IPv4 込み） | 2,050 | 3,910 | × | 『GCP の常駐機』が欲しい場合。取得を置くには固定 IP が不確実 |
| **F** | C + D + E + Cloud NAT（方針を文字どおりに実装） | 6,760 | 13,180 | × | 予算の 3〜6 倍 |

> Always Free の e2-micro（月 1 台）は **us-west1 / us-central1 / us-east1 のみ**で東京は対象外。米国に置くと日本の取得先への遅延と IP の相性が悪く、取得用には向かない。

### 4.3 有料になりやすい項目（避ける設計）
| 項目 | 月額の目安 | 避け方 |
|---|---|---|
| Cloud SQL | 約 1,800 円〜 | 使わない。ファイル（Parquet / JSON）+ Firestore 無料枠 |
| Memorystore（Redis） | 約 5,000 円〜 | 使わない（VPS 上の軽い Redis / メモリキャッシュ） |
| Cloud NAT / 固定 IP | 約 4,800 円〜 | 使わない。取得の固定 IP は VPS |
| 常時起動の GCE / Cloud Run 最小インスタンス | 約 1,800〜2,900 円 | サーバーレス（最小 0）+ VPS 側キャッシュ |
| ネットワーク送信 egress | 約 18 円/GB | 公開スナップショットは R2（egress 無料）。GCS → 外部は最小に |
| Cloud Logging / Monitoring の取り込み超過 | 約 75 円/GiB 超過 | ログを絞る（無料 50GiB） |
| 別リージョン間の転送 | 約 1〜18 円/GB | **GCS と Cloud Run を同一リージョン（asia-northeast1）に** |
| BigQuery のクエリ・保存 | 無料枠 1TiB / 10GiB | 課金エクスポートだけなら無料枠内 |

---

## 5. リスクと論点

### 5.1 スクレイピングを GCP に移す問題（最大のリスク）
| 論点 | 内容 | 推奨 |
|---|---|---|
| IP | クラウド事業者の IP 範囲は遮断されやすい。Cloud Run Jobs の送信元は共有・変動。固定は Cloud NAT（約 4,800 円）が要る | **取得は VPS に置く**（固定 IP・前払い済み）。**ConoHa の IP は netkeiba で未試験**なので、移す前に 7.4 の試験をする |
| ログイン | netkeiba のログイン依存の取得（`stg_rescrape_login_dependent_20261014.md`）。Cookie をファイルに保持 | 秘密は Secret Manager（または VPS の `.env`、権限 600）。セッションの同時使用による無効化を避ける（先に別セッションでログインが衝突した前例） |
| 同時 1 本（DEC-023）・NetkeibaClient のプロセス横断スロット | 現状は**ファイルロック**（root 所有のスロットディレクトリ、キューは `job_queue.py` のローカル JSON）。Cloud Run の複数インスタンスでは共有できない | 取得を VPS の**単一ホスト・単一ランナー**に集約すれば現状の仕組みがそのまま使える（案 A の利点）。案 B にするなら GCS の世代条件付き書き込み（対象オブジェクトのロック）か Firestore のトランザクションで分散ロックを実装（工数大） |
| 夜間の長時間バックフィル | 数百時間規模 | ホスト PC の自宅回線で夜間実行（Cloud Run Jobs だと vCPU 秒が無料枠を超える） |

**代替の比較**
| 置き場所 | 費用 | IP | 可用性 | 評価 |
|---|---|---|---|---|
| **VPS（推奨、要試験）** | 0 | 固定 | 24 時間 | 第一候補 |
| ホスト PC（自宅） | 0 | 実績あり（現状動作中） | PC 稼働に依存 | 試験で VPS が通らなかった場合の代替 |
| Cloud Run Jobs | 約 0〜780 円 | 変動・遮断リスク | 高い | 取得先が IP 非依存の軽いもの（JRA 馬場など）のみ |
| GCE（東京）+ 固定 IP | 約 1,800〜2,200 円 | 固定（予約 IP） | 高い | 予算の大半を使う |
| Cloud NAT 経由の Jobs | 約 4,800 円〜 | 固定 | 高い | 予算外 |

### 5.2 開催日の時間厳守ジョブ（T-60 / T-15 / 直前オッズ / 馬体重検知 → 予測）
- 取得 runner（VPS）が各レースの T-60・T-15・T-10・T-5・T-2 に自分で取得する（現行と同じ）。**予測**だけを GCP に出す。
- 呼び出しは **Cloud Tasks の予約実行**（無料: 月 100 万オペレーション、`predict_race_day --mode enqueue` が実装済み）。VPS の cron でも秒単位で起動できる。
- 遅延の見積り: Jobs のコールドスタート 10〜30 秒 + モデル読み込み（バンドル 160〜170MB、`serving` 状態 62MB）+ 推論（低メモリ構成で約 0.6 秒/レース）。**合計 1 分以内**。馬体重が出てから予測を出す要件（T-60）に対して十分。
- 同じ時刻帯のレースは**1 回の起動にまとめて**モデル読み込みを共有する（コスト・遅延の両方で有利）。
- Cloud Scheduler の最小粒度は 1 分で、Always Free は 3 ジョブ。**司令塔は VPS の cron**にして Scheduler を使わない（または 3 ジョブに収める）。

### 5.3 データ同期
- 正本は GCS（東京）。Jobs は GCS から必要な列・馬のシャードだけを読む（`FeatureStore` の行フィルタ読み込み、`serving` 状態 62MB の低メモリ構成）。**全件ロードは禁止**。
- ホスト PC のモデリングが作った特徴量・バンドルは `gs://<bucket>/models/ensemble/<version>/` と `latest.json`（sha256 検証つき）で配布。
- as-of の保証: 特徴量は「日付が厳密に前」だけで作る既存の規則を維持。学習側と推論側が同じビルダーを使い、列名・順序を `manifest.json` で検証する（旧設計の方針を継続）。
- 費用: GCS（東京）の保存は **月 約 345 円/100GB**。同一リージョン（Cloud Run / Jobs ↔ GCS）の読み出しは無料。外部への送信（GCS → ホスト PC・VPS など）は約 18 円/GB なので、**ホスト PC → GCS のアップロードは無料、GCS → ホスト PC のダウンロードが有料**になる点に注意（差分だけ取得する）。

### 5.4 公開スナップショット（go_live.md）との整合
- 予測生成 Jobs の最後に、**許可リスト方式**で公開用の静的 JSON を作り、R2 に置く（R2 は 10GB・egress 無料）。限定側（Access で保護）の `app.` とは別バケット・別ホスト名（`www.` / `pub.`）。
- 禁止項目（オッズの数値、過去走の表、調教、期待値の数値）が混ざったら公開せず止まるテストをジョブに入れる（`go_live.md` 4 章）。

### 5.5 認証とネットワーク（UI(VPS) → API(Cloud Run)）
```
ブラウザ → Cloudflare(Access: メール認証) → Tunnel → VPS(Next.js) →(rewrites)→ ローカル caddy →(HTTPS)→ Cloud Run
                                                                         └ 共有ヘッダ X-Keiba-Proxy-Key を付与
```
- 推奨: Cloud Run は**共有ヘッダ（Secret Manager と VPS の `.env`）+ Cloudflare Access のサービストークン**の二重で保護し、`*.run.app` の直接アクセスは無効に近づける（ヘッダ無しは 401）。Cloud Run のインスタンス上限（最大 3）・同時実行 20〜40・リクエスト上限 60 秒で暴走を抑える。
- 代替: IAM 認証（ID トークン）は VPS にサービスアカウントキーを置く必要があり、鍵の管理が増える。
- **VPS の公開面**: 受信ポートは全閉（cloudflared のみ）。SSH は閉じる（Tunnel 経由か、パスキーと送信元 IP 制限）。デプロイは pull 型（VPS が R2/GCS の `latest.json` を見て成果物を取りに行く）。
- 秘密情報: Secret Manager（有効バージョン 6 個まで無料）。Slack Webhook、Cloudflare トークン、netkeiba のログイン、R2 のキー、共有ヘッダ。ホストの `.env` は 640 にしてある。

### 5.6 運用・信頼性
| 項目 | 方針 |
|---|---|
| デプロイ | **UI**: ホストでビルド → R2 に成果物 + `latest.json` → VPS がメンテナンス窓（毎日 03:00〜04:00 JST）に取得して入れ替え。**API / Jobs**: ホスト PC から `gcloud run deploy`（ユーザーが実行、または窓内に自動化）。ロールバックは前のリビジョンへトラフィックを戻す |
| 監視 | UI はトンネルと `/api/health`（Access の背後なので内部監視）。Cloud Run は組み込みのメトリクス。Slack の既存通知（スクレイピング状況、ドリフト、凍結追跡）を継続 |
| ログ | Cloud Logging は無料枠 50GiB に収まるよう INFO を絞る。VPS は `rotate_logs` |
| バックアップ | GCS のバージョニング（任意、保存量が増える）。ホスト PC の `data/local` の重要部分は GCS に同期。VPS は自動バックアップ無効のため、**VPS 上にしかない状態は持たない** |
| 障害時 | トンネル断 → 監視が検知して Slack。VPS 停止 → ConoHa のコンソールから再起動。Cloud Run 障害 → 公開スナップショット（R2）で閲覧を継続 |

### 5.7 セキュリティ
- 2GB の VPS は DoS に弱い。Cloudflare の前段（キャッシュ・レート制限・Bot 対策）を前提にする。
- VPS の OS 更新（自動更新）、ファイアウォール全閉（受信は cloudflared の外向き接続のみ）、SSH は閉じる、鍵は VPS に置かない。
- 取得の秘密（netkeiba のログイン）は VPS のみ。Cloud Run には渡さない（案 A）。

---

## 6. コストの計算方法とモニタリング

### 6.1 計算を再現する
```
python -m src.scripts.ops.cost_estimate --scenario all                     # 全案(円/月)
python -m src.scripts.ops.cost_estimate --scenario A_budget_core --format md
python -m src.scripts.ops.cost_estimate --scenario A_budget_core --fx 160  # 為替を変える
python -m src.scripts.ops.cost_estimate --scenario A_budget_core \
  --set scenarios.A_budget_core.usage.high.gcs.class_a_ops=200000          # 使用量を変える
python -m src.scripts.ops.cost_estimate --list-unverified                  # 公式未確認の単価・無料枠
```
`config/cost_assumptions.json` に、単価・無料枠・使用量の仮定（`verified` つき）がある。**実績が出たら単価と使用量を実績で置き換える**（毎月の見直しを推奨）。テスト: `tests/scripts/test_cost_estimate.py`。

### 6.2 GCP の予算アラート（ユーザーが Console で設定する手順）
1. Google Cloud Console → **お支払い**（Billing）→ 対象の請求先アカウント → **予算とアラート** → **予算を作成**。
2. 名前: `keiba-monthly`。範囲: すべてのプロジェクト・すべてのサービス（クレジットは含めない）。
3. 金額: **指定額 2,000 円**（請求先アカウントの通貨に合わせる。USD なら 約 13 ドル）。
4. しきい値: **50% / 80% / 100%**（実績）と **120%**（実績）。加えて **100%（予測）**を追加すると月の途中で超過見込みを検知できる。
5. 通知: メール（請求先アカウント管理者）を有効にし、必要なら **Pub/Sub トピック**を接続（6.4 のブレーカー用）。
6. 保存後、テスト通知を確認。

> 予算アラートは**通知のみで、請求を止めない**。止めるのは 6.4 のブレーカー。

### 6.3 請求データの分析と既存の通知
- **課金データのエクスポート（BigQuery、標準の使用量コスト）**を有効化（お支払い → 課金データのエクスポート）。データセットは `asia-northeast1`。エクスポート自体は無料で、保存・クエリは BigQuery の無料枠（10GiB / 1TiB）に収まる。テーブル名を環境変数 `GCP_BILLING_BQ_TABLE` に設定。
- 既存: `python -m src.scripts.cloud_jobs.gcp_daily_cost_report`（前日のサービス別コストを Slack に毎朝送る。BigQuery を読む）。**拡張案（未実装）**: ① 月累計と**月末予測**（日次の平均 × 残日数）を予算 2,000 円と比べる、② `cost_estimate` の見積りとの差を出す、③ しきい値超過時は『注意 / 警告』で通知、④ 週次サマリー（Slack）。ジョブは Cloud Run Jobs（無料枠内）で実行。
- 月次の見直し: 実績で `config/cost_assumptions.json` を更新し、本書の表を再生成する（`--format md`）。

### 6.4 暴走防止
| 対策 | 設定 |
|---|---|
| Cloud Run のインスタンス上限 | API: 最大 3 インスタンス・同時実行 20〜40・タイムアウト 60 秒 |
| Cloud Run Jobs | タスクのタイムアウト（予測 15 分、集計 60 分）・リトライ 1 回まで |
| Scheduler / Tasks | リトライ上限 1〜2・最大試行時間を設定（無限リトライしない） |
| リクエスト上限 | リクエストサイズ・レスポンスサイズ。共有ヘッダ無しは即 401（課金を発生させない） |
| egress の監視 | 課金エクスポートの SKU（Network / Data Transfer）を日次で確認。急増で警告 |
| **ブレーカー（設計案のみ、未実装）** | 予算アラートの Pub/Sub → Cloud Run functions（無料枠）で、**API の最大インスタンスを 0 にする / 受信を内部のみにする**。請求先アカウントの無効化は**行わない**（データ損失の恐れ） |

### 6.5 費用の更新カレンダー
| 項目 | 日付 | 備考 |
|---|---|---|
| ドメイン `megu-keiba.com` | **2027-08-21 更新**（自動更新 ON） | 年額は要記入（ConoHa 契約画面）。Cloudflare 側は Free で 0 円 |
| ConoHa VPS | **2028-05-18 満了** | 自動更新は **OFF を推奨**（使い続けるかを満了前に判断）。旧サイトは停止（シャットダウン） |
| GCP | 毎月 | 予算アラート・日次通知 |
| Cloudflare | Free プラン | 年次の更新なし。R2 が 10GB を超えたら課金（月 約 2 円/GB） |
| Slack | Free | — |

---

## 7. 段階的な移行計画

月額は `cost_estimate` の案 A の増分（下限〜上限、円）。フェーズ 0 は現状（GCP の実績は日次通知の値で要記入）。

| フェーズ | 内容 | 追加月額（円） | 累計月額（円） |
|---|---|---:|---:|
| **0** 現状 | ホスト PC に全部。GCS は既存（正本）、公開は Cloudflare Tunnel + Access | 0 | **GCP 実績を記入**（GCS の実績） |
| **1** UI を VPS へ | Next.js と cloudflared を VPS に。API は当面ホスト（トンネル経由） | **0** | 0 + 現状 |
| **2** API とデータの GCP 化（読み取り系） | Cloud Run の API（最小 0）+ GCS を正本に + VPS のキャッシュ。公開スナップショットを R2 | 0〜120 | 〜 約 500 |
| **3** 予測生成の GCP 化 | Cloud Tasks + Cloud Run Jobs。バンドルを GCS に配布。shadow v4/v5 も | 0〜560 | 〜 約 1,100 |
| **4** 取得の置き場所を決定 | VPS に移す（試験が通れば）/ ホスト PC に残す。**GCP に移さないのが推奨** | 0（VPS）／〜780（Jobs） | 〜 約 1,100（Jobs 案は 〜 約 1,900） |
| **5** ホストはモデリングのみ | cron の整理、常駐の停止、モデルの GCS 配布の自動化 | 0 | 案 A の合計（260〜2,050） |

### 7.1 最初にやる小さな一歩（フェーズ 1。低リスク・追加費用 0）
**目的**: UI の常駐を VPS に移し、ホスト PC を止められる範囲を広げる。
1. （ユーザー）ConoHa のコンソールで VPS を**起動**し、OS を最新に更新、**swap 2GB を追加**、ファイアウォールを受信全閉にする（SSH は一時的に自分の IP のみ）。
2. （ユーザー）VPS に Node.js 20 LTS と cloudflared を入れる。Cloudflare の Zero Trust で**新しいトンネル `meguai-vps`**を作成し、トークンを VPS の `.env`（権限 600）に保存。
3. （ホスト）`next build` で成果物を作り、VPS に配置（初回は手動、以降は pull 型）。`next start -p 23200` を systemd で常駐。
4. Tunnel の Public Hostname に `app-new.megu-keiba.com → http://localhost:23200`（VPS 側のトンネル）を登録、Access に同じ `owner` ポリシーを適用。**ホストのトンネルは残したまま**並行稼働して確認。
5. VPS の Next.js の `rewrites` の向き先を、ホストの API（トンネル経由の内部ホスト名）にする。API の応答遅延が増えるので、Cloudflare / Next のキャッシュを確認。
6. 問題なければ `app.megu-keiba.com` を VPS のトンネルに切り替える（ホストのトンネル側のホスト名を外す）。
7. **ロールバック**: `app.` の向き先を元のトンネル（ホスト）に戻す。数分。
8. **完了条件**: 1 週間、`app.` が VPS 経由で安定。VPS のメモリ使用が 1.2GB 以下。ホスト PC の Next.js を止められる。

### 7.2 フェーズ 2（API とデータの GCP 化）の作業
GCS を正本に固定（バケットの整理・ライフサイクル・リージョン確認）→ API のコンテナ化（1 ワーカー、起動時間の短縮、必要ファイルを GCS から遅延取得）→ Cloud Run へデプロイ（最小 0、最大 3）→ VPS の caddy から呼ぶ → 公開スナップショットを R2 へ。**所要の目安**: 1〜2 週間。**ロールバック**: rewrites の向き先をホストの API に戻す。**ユーザー作業**: GCP プロジェクト・請求先アカウントの確認、予算アラート（6.2）、課金エクスポート（6.3）、Secret Manager の登録。

### 7.3 フェーズ 3（予測生成の GCP 化）の作業
`predict_race_day --mode enqueue`（Cloud Tasks）と Jobs のイメージ化（ML ライブラリ約 255MB + バンドル）、バンドルの GCS 配布と sha256 検証、shadow v4/v5 の並走、予測成績の集計 Jobs、通知。**検証**: 同じ日をホストと GCP で予測して一致を確認（数日間並走）。**ロールバック**: ホストの cron を戻す。

### 7.4 フェーズ 4 の事前試験（VPS から netkeiba への取得）
- VPS で `tests/scraper/manual/netkeiba_horse_page_smoke.py` 等を**低頻度（4.5〜7 秒間隔、50 リクエスト以内）**で実行し、HTTP 403 / 429、ログインの可否、応答内容を確認する。**ホストの取得（DEC-023）と同時に実行しない**。
- 通れば runner を VPS に移し、ホストの cron を止める。通らなければ取得はホスト PC に残し、可用性はホスト PC の稼働に依存することを受け入れる。

---

## 8. 既存資産の再利用
| 資産 | 状態 | 使い道 |
|---|---|---|
| `src/config/gcp_guard.py` | dev PC からの GCP 接続を禁止（`KEIBA_ENV=dev`） | **この設計の作業もホスト（stg）か Console で行う**。dev PC では GCP に触れない（AGENTS.md） |
| `HybridStorage`（`src/scraper/storage.py`） | GCS が正本。stg は `GCS_BUCKET` 設定済み | Cloud Run / Jobs でもそのまま使える（サービスアカウント認証） |
| Cloud Tasks（`predict_race_day --mode enqueue`） | 実装済み | T-60 / T-15 の予約実行 |
| `scripts/gcp/deploy_cloud_run_jobs.sh`・`docs/operations/gcp-cloud-run-jobs.md` | 設計図（未デプロイ） | Jobs の登録（15 ジョブのうち必要な分） |
| `gcp_daily_cost_report.py` | 実装済み（BigQuery 課金エクスポートが前提） | 6.3 |
| `model_sync.py`・`publish_model`（未実装） | 旧設計 | モデル配布の自動化（GCS `latest.json`） |
| 低メモリ構成（`serving` 状態 約 62MB、バンドル約 160〜170MB、約 0.6 秒/レース） | 実装済み | Jobs / Cloud Run の 2GiB で足りる |
| prod プロファイル（`config/ports.env` の 233xx、`KEIBA_ENV=prod`） | 定義のみ。現在の **stg がホストで本番相当**として動いている | 移行後の『本番』は VPS（UI）+ GCP（API/Jobs）。環境名は `KEIBA_ENV=prod` を使うか、stg のまま運用名にするかを決める（9 章） |

注意: `KEIBA_ENV` が未設定だと prod 扱いになる（起動拒否の安全装置あり）。Cloud Run では必ず明示する。

---

## 9. 決めていただくこと
1. **予算の扱い**: 月 2,000 円を『上限』と『目安』のどちらにするか。API の最小インスタンス 1（約 2,900 円）を許容するか。
2. **方針の文字どおり vs 予算優先**: 『取得を GCP に移す』は予算外（案 B は最大 約 2,830 円、案 F は約 6,800〜13,000 円）。**取得を VPS に置く第一案 A を採ってよいか**。
3. **取得の置き場所**: VPS で試験してよいか（7.4）。通らなかったときにホスト PC に残す（PC の常時稼働が前提）ことを受け入れるか。
4. **GCP のプロジェクトとリージョン**: 既存のプロジェクト ID（請求先アカウントの通貨が JPY か USD か）。リージョンは **asia-northeast1（東京）**でよいか。GCS の既存バケットのリージョン・サイズ・ライフサイクルは未確認。
5. **可用性の目標（SLO）と停止許容時間**: 例『開催日の 08:00〜20:00 は 99%、それ以外は 95%』『メンテナンス窓 03:00〜04:00 は停止可』。VPS 1 台を単一障害点とするか。
6. **データ保持**: GCS の生データの保持期間（古い生データの圧縮・Nearline 化）、予測のスナップショットの世代数、ログの保持。
7. **環境名**: 移行後の本番を `KEIBA_ENV=prod` にするか、stg を本番名として使い続けるか。
8. **旧 VPS（旧サイト）**: 停止後の VPS をそのまま新しい UI に再利用してよいか（OS を作り直すか）。
9. **公開側（広告）サイトの計画との整合**: 公開スナップショットを R2 に置く方針でよいか（go_live.md）。
10. **通貨・為替**: 見積りの為替（150 円）と、請求先アカウントの通貨。

---

## 付録 A: 公式ページで確認した前提 / 未確認の前提
- **確認済み（2026-10-21、公開ページ）**: Cloud Run 無料枠（リクエスト課金: 200 万リクエスト・36 万 GiB 秒・18 万 vCPU 秒）、Cloud Storage 無料枠（5GB-月、**米国 3 リージョンのみ**）、Secret Manager（6 バージョン・1 万アクセス）、Firestore（1GiB、読み取り 5 万 / 書き込み 2 万 / 削除 2 万 per 日）、BigQuery（1TiB / 10GiB）、e2-micro 無料枠（**米国 3 リージョンのみ**）、Artifact Registry（0.5GB）、Cloud Logging（50GiB）、Pub/Sub（10GiB）、Cloud Build（2,500 分）、Cloudflare R2（保存 $0.015/GB-月、Class A $4.50/百万、Class B $0.36/百万、egress 無料、無料枠 10GB / 100 万 / 1,000 万）。
- **未確認（作成者の知識ベース。公式ページの本文を取得できなかった）**: Cloud Run の単価（vCPU 秒・GiB 秒・リクエスト・アイドル・Jobs）、東京が Tier 1 か、GCS 東京の単価と操作課金、Cloud Scheduler（無料 3 ジョブ・$0.10）、e2-micro 東京の単価・外部 IPv4 の課金、Cloud SQL・Cloud NAT の額（旧 `cost-estimate.md` の値）、Cloudflare Access の無料 50 ユーザー。**採用前に公式ページと [GCP 料金計算ツール](https://cloud.google.com/products/calculator) で確認**。`--list-unverified` で一覧を出せる。
