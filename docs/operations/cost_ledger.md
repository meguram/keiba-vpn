# 費用の台帳（GCP + Cloudflare）

最終更新: 2026-11-14 14:40 JST（費用の監査役。GCS の DATA_READ 監査ログの最終集計 11/13 14:40〜11/14 14:30 JST を 3 章 #7 に追加。課金エクスポートは GCS の転送が 11/13 21:00 JST まで反映）

上限は **GCP + Cloudflare の合計で月 2,000 円**（`config/cost_assumptions.json` の `budget_jpy_per_month`、CLAUDE.md の「費用の上限」）。
この台帳は、費用に効くものを 1 か所で管理する。費用に効く変更を入れるとき・計画を審査するときは、ここに行を足して合計を直す。
試算の道具（`python -m src.scripts.ops.cost_estimate`）のシナリオは 10 月の設計の想定で、今の構成とは合っていない。今の数字はこの台帳を正とする。

## 0. 結論（2026-11-13 時点）

### 11/14 14:40 JST の更新（監査ログの最終集計。詳細は 3 章 #7）
- **朝の大量読みの正体が分かった: ホストの root cron `backfill-all-horse`（毎日 06:00 JST、`python -m src.scraper.backfill --phase horse`）**。
  `src/scraper/backfill.py:347-355` が馬 ID を集めるために race_result と race_shutuba を**年を絞らずに全件**（2016〜2026、約 8 万件）GCS から 1 件ずつ読む。06:00〜07:00 に race_result、07:00〜07:50 に race_shutuba。
  呼び手は `megu-keiba@`・IP 175.177.45.24・UA gcloud-python 3.10.1 / Python 3.11.6（ホスト）で、**GCP の外**。1 日 約 8 万回・約 0.65〜0.8GiB。毎日 1 時間 50 分かけて、処理は 0〜2 頭（11/06〜11/14 の `logs/backfill_horse.log`）。
  費用: Class B 月 約 145 円（いつもかかる）＋転送 月 約 21〜25GiB（請求月の無料の 100GiB を超えた月は 約 400〜470 円）。**新しく始まったものではなく、前からの見込みに入っている**。
- 11/13 10:25〜10:45 JST の 約 1.4GiB は、別のセッションが 10:21 JST に手で流した 2026 年の書き出し（`logs/export_tables_2026_once_20261113.log`、13 カテゴリ × 約 3,000 レース）。**一度きり**（11/14 の同じ時間は 95MiB）。無料枠を超えた月なら 約 26 円。
- **24 時間（11/13 14:40〜11/14 14:30 JST）の GCP の外への転送の推定は 約 3.6GiB**（読み出し 2.9GiB・一覧の応答 0.7GiB）。うち金曜の週次の全件読み 約 1.9GiB、朝の backfill 約 0.8GiB、調教の dry-run（一度きり・957858ad で修正済み）約 0.3GiB。
  **金曜以外のふだんの日は 約 1.0〜1.1GiB/日、週の平均 約 1.3GiB/日**（fetch-agent の停止・cache-warm の合図方式・verify_past_values の一覧 0・調教の補いの修正が入った後の姿）。21:00 の見込み（3.2〜3.5GiB/日）より小さい。
- **12 月の見込みを直す: 転送は 1 日 約 1.3GiB × 31 ≒ 40GiB で無料の 100GiB に収まり 0 円**（手で流す一度きりの全件読みに 約 60GiB の余裕）。**合計 約 1,400〜1,480 円 → backfill を直せば 約 1,250〜1,340 円**。2.6 の表の転送の行は 0 円が中央になる。
- **11 月の無料枠**: 11/13 21:00 JST までで 70.7GiB。残り 約 17.8 日 × 1.3GiB ≒ 23GiB で **月末 約 94GiB の見込み（ぎりぎり収まる）**。手の一度きりの読みが 6GiB を超えると超える（超えても 1GiB 18.86 円）。backfill を早く直せば 約 83GiB で余裕ができる。
- 対策の第一は backfill の馬 ID の集め方を手元の flat に替えること（3 章 #7。取得〔scraper〕の担当セッションへ）。

### 21:00 JST の更新（引き継いだ監査役）
- **12 月の見込み（今の構成。21:30 に 280a6ab0・4d424c7a・7e1cc81f を入れて直した）: 転送を除いて 月 約 1,400〜1,480 円**。GCS の外への転送は、1 日の量が約 3.2GiB 未満なら請求月の無料の 100GiB に収まって 0 円、超えると 1GiB ごとに 18.86 円が乗る（12 月は 31 日。1 日 3.5GiB なら 約 +160 円、4GiB なら 約 +450 円、4.5GiB なら 約 +750 円）。
  **中央の見込みは 1 日 約 3.2〜3.5GiB（verify_past_values の一覧 0 で 1 日 0.3〜0.4GiB 減る）→ 合計 約 1,400〜1,640 円（上限との差 −360〜−600 円、上限内）**。1 日 4.5GiB に戻ると 約 2,200 円で超過する。2.6 を参照。
  変更後の転送の 1 日分（11/14 JST）は、11/15〜16 の課金エクスポートで確定させる。
- **11 月の転送の無料枠**: 請求月 202611 の無料の段（`tier_start_amount=0`）は、11/12 21:00 JST までで **65.9GiB**（`tier_start_amount=100` の 12.5GiB は 10/31〜11/01 朝の超過分で、無料の 100GiB のカウンタには入っていない）。
  残り 約 34GiB。1 日 3〜4GiB なら **11/21〜24 ごろに 100GiB を超え**、月末（12/01 17:00 JST）までに 約 +400〜750 円が 11 月に乗る（下の「11/18〜19」は 6.5GiB/日の前提で、今は外れた）。
- **最も重いリスクだった: master への昇格で、止めた Scheduler 13 件が作り直される**（3 章 #6）。**21:30 の追記: 7e1cc81f（job_map の 13 グループに retired、scheduler_gen が作らない）が develop に入った**。昇格はこのコミットを含む版でだけ行い、昇格の後に Scheduler が keiba-g-0730-shadow の 1 件のままであることを確かめる。7e1cc81f より前の版を master に上げると作り直される。作り直されて動き出すと、月 約 +1,700 円（Jobs の CPU・メモリ。GCS の操作 約 +1,000 円を足すと 約 +2,700 円）。止めたままでも Scheduler の固定費が 13 件 × 15 円 ≒ 約 +195 円かかる。
- **一覧（G1）は 1 日 約 1.2 万回 → 約 5,000〜6,000 回**に下がった（15:00 の fetch-agent の停止で −8,640 回、16:30 の cache-warm の合図方式で VPS の一覧が 1 時間 6〜9 回）。今の一覧の約 6 割は Cloud Run の keiba-api（東京の中なので転送 0）。
- **新しく見つけた周期の大量読み**（監査ログ）: 金曜 18:00 の weekly-update と 18:30 の horse-name-index が、**どちらも race_shutuba の全年（約 3.5 万件）をホストから GCS で読み直している（二重）**。続けて成長曲線の週次更新が horse_result を 5.8 万頭読む。1 回 約 1.5〜1.9GiB・月 約 7〜8GiB（無料枠を超えた月で 約 130〜150 円）。CLAUDE.md の決まり 2 に反する。3 章 #2。
- **まだ呼び手が分からない読み出し**: 朝 06〜08 JST の ReadObject 約 9.6 万回・約 0.9GiB と、10〜11 JST の 約 1.5GiB（監査ログの開始 14:44 より前）。ただし 11/12 の転送の実績 2.7GiB（00:00〜21:00 JST）は朝の時間を含むので、**毎日のものなら見込みの 1 日 3.2〜3.5GiB にもう入っている**。上乗せになるのは 11/13 から新しく始まった読みだった場合だけ（1 件 約 95KB と大きく、Cloud Run の起動時の束や配備〔転送 0〕の可能性もある）。**11/14 の朝の分を、監査ログを戻す 14:44 JST の前に集計して確かめる**（3 章 #2 の「監査ログの集計」）。

### 14:50 JST 時点の記述（以下、経緯として残す）

- **11 月はすでに上限を超えている**。請求月（太平洋時間で区切る。`invoice.month=202611`）の 11/12 午前までの請求は **1,783 円**。
  cost_guard（JST で区切る）の今月の累計は 2,634 円（10/31 夜〜11/01 朝の超過分を含む）。11 月の残りで上限内に戻すことはできないので、**目標は 12 月の月額**とする。
- **直近 7 日（11/05〜11/11、全部そろった日）の実績は 1 日平均 151 円 → 月 約 4,540 円**。最大は Cloud Run Jobs（51 円/日）と GCS の操作（62 円/日）。
- 11/12 17:00 JST に GCP の並走 Jobs の Scheduler（keiba-g-* 13 件）が止まった。止めた後の 24 時間（Cloud Monitoring）から見込むと、天気の取得（下の緊急）を除いても **12 月の見込みは月 約 1,900 円＋GCS の外への転送料（約 1,800 円）＝ 約 3,700 円**。転送料を入れると上限の約 1.8 倍。
- **最大の隠れた項目は GCS の外への転送（Download APAC）**。1 日 4〜10GiB 出ているが、11/02 以降の課金エクスポートでは 0 円に見える。
  これは無料ではなく、**請求月ごとに最初の 100GiB が 0 円（`price.tier_start_amount=0`）で、100GiB を超えると 18.86 円/GiB** という段階の単価のため（10/31・11/01 朝の行は `tier_start_amount=100`・18.86 円/GiB）。
  11 月は 11/12 午前までに **65GiB** を使った。1 日 約 5〜6GiB のままだと **11/18〜19 ごろに 100GiB を超え**（21:00 の更新: 今の 1 日 3〜4GiB なら 11/21〜24 ごろ）、そこから 1 日 約 100 円**が上乗せされる。cost_guard の見込みは過去の 0 円の日を平均に使うので、超える前には気づけない。
- **【解決: 14:08 に停止、42f48f22 で廃止】緊急（11/13 13:40 から）: 天気の取得の再開で、ホストが毎時 約 345MiB を GCS から読むようになった**（1 日 約 8.1GiB、月 約 4,400〜4,600 円）。
  `scripts/cron/jma_weather_run.sh` が毎時 `jma_weather fetch --from 20160101` を回し、`src/scraper/jma_weather.py:242`（`load_month`、`:209` で `bypass_cache=True`）が
  10 競馬場 × 131 か月 ＝ 1,310 件の月ファイル（GCS 上で合計 345MiB、1 件 最大 376KiB）を毎回全部読んで「取得済みか」を確かめている。13:40 の回は `fetched 190, skipped 39,500`。
  このままだと 11 月の残りの無料の 35GiB は約 4 日で尽き、既存の 1 日 6.5GiB と合わせて 1 日 約 14.6GiB → **12 月は転送だけで月 約 6,400 円**。3 章 #1。
- **天気（気象庁）の定期取得は廃止（2026-11-13、ユーザーの判断）**。14:08 に止めの印で止め、**ホストの root の crontab の KEIBA-JMA-WEATHER の行は 42f48f22 で外した**（`install_jma_weather_cron.sh --remove`、退避は `data/local/meta/cron_backups/`。その前の 3a86b5ff で 1 日 1 回・前回からの範囲だけに直していた）。GCP 側に専用の Scheduler は無く、`config/gcp_job_map.json` の `jma-weather` は enabled=false（g-hourly のグループの 1 つ。keiba-g-hourly は 14:4x に削除済み）。下の表の G5 は合計から外した。
- 11/13 の追加の調べ: VPS の nginx の上流は本番・stg とも **Cloud Run**（`/etc/nginx/sites-enabled/keiba-ui:17` `keiba-api-…run.app`、`keiba-ui-stg:18` `rstg---keiba-api-…`）。cache_warm の取り直しがホストの API で GCS を読む経路は**無い**。
  VPS で GCS を周期的に読むのは cache_warm（60 秒おき）と release-pull（3 分・5 分おきに latest.json 1 件）だけ。
- 上限内に戻すには、(0) 天気の取得の毎時の全件読みを止める、(1) GCS の外への転送を月 100GiB 未満にする（＝ホスト・VPS から GCS を読む量を 1 日 3GiB 未満に）、(2) GCS の一覧（1 日 約 1.2 万回・約 3.4GiB）を減らす、(3) 固定費（Scheduler・Artifact Registry）を削る、の 3 つが要る。詳細は 3 章。
  （21:00 の時点: (0) は廃止で済み。(2) は fetch-agent の停止・cache-warm の合図方式・毎朝の書き出しの一覧 0（3fcb2a2b）で 1 日 約 5,000〜6,000 回に。(3) は Scheduler 13 件の削除・AR の保持 1 日で済み。残るのは (1)。）

## 1. 単価（2026-11 の課金エクスポートの実単価。円）

| 品目 | 実単価 | 無料枠 | 根拠 |
|---|---|---|---|
| GCS の外への転送（Download APAC） | **18.86 円/GiB**（100GiB を超えた分） | **請求月ごとに 100GiB**（太平洋時間の月。超えた分から課金） | 課金エクスポートの `price.tier_start_amount`（0 → 0 円、100 → 18.86 円）。10/31 と 11/01 の行 |
| GCS Class A（書き込み・一覧） | 0.79〜0.80 円/千回 | なし | 11/10: 34,200 回 → 27 円 |
| GCS Class B（読み出し・メタデータ） | 0.063 円/千回 | なし | 11/10: 191,000 回 → 12 円 |
| GCS 保存（東京 Standard） | 約 3.6 円/GiB-月 | なし（無料枠は米国のみ） | 1 日 1.7GiB-月の計上で 6.2 円 → 保存 約 52GiB |
| Cloud Run（サービス、リクエスト課金）CPU | 約 4.1 円/千 vCPU 秒 | 月 18 万 vCPU 秒・36 万 GiB 秒（**サービスと Jobs で共有**） | 11/12: 1,460 秒 → 6 円。無料枠は 11 月は 11/07〜08 に使い切った |
| Cloud Run Jobs CPU / メモリ | 約 2.9 円/千 vCPU 秒 / 約 0.32 円/千 GiB 秒 | 同上（共有） | 11/10: 17,500 秒 → 50 円 |
| Cloud Run の外への送信（API の応答） | 約 20 円/GiB | — | 11/06: 0.51GiB → 10 円 |
| Cloud Scheduler | 約 15 円/件・月（0.1 ドル） | 請求先アカウントで 3 件（**実績では効いていないように見える**） | 17 件で 1 日 約 9 円（= 17 件 × 15 円 ÷ 30）。**停止中のジョブも数えられている** |
| Artifact Registry 保存 | 約 15 円/GB-月 | 0.5GB | 16GiB 保存で 1 日 約 6 円 |
| Secret Manager アクセス | 1 万回まで無料 | 月 1 万回 | Jobs の停止で 1 日 700 回前後 |
| Cloudflare（Tunnel・Access・DNS） | 0 円（無料プラン） | Access 50 ユーザー | R2 は使っていない（コードに接続なし） |

注意:
- **無料枠は月の前半で尽きる**。Cloud Run の無料枠（サービスと Jobs で共有）は 11 月は 11/07〜08 に尽き、そこから Jobs が 1 日 50〜140 円になった。GCS の転送の 100GiB も月の途中で尽きる。月初の数日の 0 円を見て「安い」と判断しない。
- 課金エクスポートは約 1 日遅れ、GCS は 6 時間ごとの束で入る。今日と昨日は不完全（11/12 は 14 時以降の束が無い）。

## 2. 項目ごとの表（月額、円）

列の意味: 「実績」は課金エクスポートか Cloud Monitoring の実測。「見込み（今）」は 11/12 の Jobs 停止後の今の構成。「最悪」は CLAUDE.md の決まり 1（最悪の回数）で見積もった値で、無料枠を使わない側に寄せる。
「今日の予定がすべて入った後」は 4 章の変更を足した値。

### 2.1 固定費（何もしなくてもかかる）

| # | 項目 | 動く場所 | 頻度（最悪） | 1 回の量 | 実績 | 見込み（今） | 最悪 | 今日の予定の後 | 根拠 | 状態 |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | GCS 保存 | GCS（東京） | 常時 | 約 52GiB（10/28 実測 39GiB から増加） | 6.2 円/日 | 186 | 215（60GiB） | 186 | 課金 `Standard Storage Tokyo` | 稼働中 |
| F2 | Artifact Registry 保存 | GCS 相当（asia-northeast1 `keiba-vpn`） | 配備ごとに API の版 0.87GB・Jobs の版 1.13GB が増える | 変更前 16GiB（API 13 版・Jobs 6 版。2 日の保持で残っていた） | 6〜10 円/日 | 160（保持 1 日。11/13 14:4x に変更済み） | 220（1 日に配備 10 回） | 165（hotfix の本番配備で 1 版増） | 掃除の設定（live-・新しい 3 版・**1 日**）。`scripts/gcp/deploy_cloud_run_api.sh:103` ほか。Jobs の `live-jobs`・`live-jobs-a/b` で 3.4GB が Jobs 停止中も残る | 稼働中 |
| F3 | Cloud Scheduler | Scheduler | 常時 | **1 件**（keiba-g-0730-shadow。停止中の keiba-g-* 13 件は 11/13 14:4x JST に削除、smoke-auto 3 件も 11/13 に削除） | 17 件で 9 円/日（停止中も課金されていた） | 0〜15 | 15 | 0〜15 | 課金 `Cloud Scheduler Jobs` の数量が 11/06〜11/12 ずっと 17。**11/14〜15 の課金で数量が 1〜4 件に下がることを確かめる** | 稼働中 |
| F4 | Secret Manager | Cloud Run・Jobs | アクセスごと | — | 0〜2 円/日 | 10 | 30 | 10 | 課金 `Secret access operations`（Jobs 停止で 4,000 → 700 回/日） | 稼働中 |
| F5 | Cloudflare | Cloudflare | — | 無料プラン（Tunnel・Access・DNS） | 0 | 0 | 0 | 0 | R2 の接続なし | 稼働中 |
| | **固定費の計** | | | | | **約 360〜375** | **約 475** | **約 365〜380** | | |

### 2.2 Cloud Run のサービス（API・監視）

| # | 項目 | 動く場所 | 頻度（最悪） | 1 回の量 | 実績 | 見込み（今） | 最悪 | 今日の予定の後 | 根拠 | 状態 |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | keiba-api（本番・stg を同じサービスのタグで） | Cloud Run（最小 0・最大 3、1 vCPU/2GiB、リクエスト課金、起動時 CPU ブースト） | 起動し直し 96 回/日（15 分ごと、決まり 1）＋要求 1 日 5,000〜10,000 件 | 起動 1 回 約 50 秒 × ブーストで 2 vCPU ＝ 100 vCPU 秒、起動時に束 `ops/mirror/_bundle/base.tar.gz` を 1 回読む（東京の中なので転送 0） | CPU 1,100〜3,700 vCPU 秒/日（Jobs 停止後は無料枠の中） | 0〜60 | 約 700（96 × 100 × 30 ＝ 28.8 万 vCPU 秒＋要求 6 万 − 無料 18 万 ＝ 17 万 × 4.1 円/千） | 0〜60 | `gcloud run services describe keiba-api`（maxInstanceCount 3、cpuIdle、startupCpuBoost） | 稼働中 |
| R2 | keiba-api の外への送信 | Cloud Run → VPS（Next.js）・利用者 | 要求ごと | 0.05〜0.5GiB/日 | 1〜10 円/日（7 日平均 4 円） | 120 | 300（0.5GiB/日） | 120 | 課金 `Cloud Run Network Internet Data Transfer Out` | 稼働中 |
| R3 | keiba-monitor | Cloud Run（最小 0・最大 1、0.5GiB） | 閲覧時のみ（Scheduler からの点検は今は無い） | — | ほぼ 0 | 0 | 30 | 0〜3（d6c72260 の /schedules。ふだん 約 3 円・最悪 約 35 円） | `gcloud run services describe keiba-monitor`、d6c72260 | 稼働中（d6c72260 の版は**未配備**） |
| | **Cloud Run サービスの計** | | | | | **120〜180** | **約 1,030** | **120〜180** | | |

### 2.3 Cloud Run Jobs

| # | 項目 | 動く場所 | 頻度（最悪） | 1 回の量 | 実績 | 見込み（今） | 最悪 | 今日の予定の後 | 根拠 | 状態 |
|---|---|---|---|---|---|---|---|---|---|---|
| J1 | 並走の Jobs（keiba-dispatch-s/m/l/xl、keiba-g-* 13 件の Scheduler。**13 件は 11/13 14:4x に削除。master への昇格で作り直されるリスクあり → 3 章 #6**） | Jobs | 平日 1 日 約 130 回・開催日 約 670 回（11/08〜09 は s が 541 回） | m は 1 回 約 1.7 分。1 回ごとに state_in・mirror_in を GCS から取り込む（決まり 5） | 11/05〜11/11 平均: CPU 51 円＋メモリ 5 円/日＋GCS の操作の大半 | 0（停止中） | 再開すると 約 1,700＋GCS の操作 約 1,000 | 0 | `src/ops/cost_guard.py:10`（keiba-g- を止める）、実行履歴（11/13 は shadow 1 回だけ） | **停止（11/12 17:00 JST）** |
| J2 | 費用の日報（keiba-g-0730-shadow → keiba-dispatch-s） | Jobs | 1 日 1 回 | 約 4 分・1 vCPU | 無料枠の中 | 0 | 15 | 0 | `src/ops/cost_guard.py:39` KEEP | 稼働中 |

### 2.4 GCS の操作と転送（Cloud Monitoring の実測。Jobs 停止後の 24 時間 = 11/12 13:54〜11/13 13:54 JST）

| # | 項目 | 動く場所 | 頻度（実測／最悪） | 1 回の量 | 実績 | 見込み（今） | 最悪 | 今日の予定の後 | 根拠 | 状態 |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | GCS の一覧（ListObjects、Class A） | 呼び手は 3 章 #2 | **1.2 万回/日 → 11/13 15:00 以降 約 5,000〜6,000 回/日**（1 時間 約 250 回。keiba-api 約 6 割・ホスト 約 3 割・VPS 1 時間 6〜9 回）。開催日の Jobs 稼働時は 1.7〜2.6 万 | 応答 平均 約 280KB（変更前 1 日 3.4GiB） | 1.2 万 × 0.8 円/千 = 9.6 円/日 → 約 4 円/日 | **約 110**（5,000 × 30 × 0.8 円/千 ≒ 120。21:30 に 280a6ab0 の verify_past_values の一覧 0（1 日 約 480 回）で −10。変更前 290） | **約 1,300**（Cloud Run の API の一覧の最悪 約 1,250〔#2、起動し直し 96 回/日〕＋ホスト・VPS 約 2,000 回/日 ≒ 50。Jobs 再開ならさらに +330） | 110 | Monitoring `storage.googleapis.com/api/request_count` method=ListObjects | 稼働中 |
| G2 | GCS の書き込み（WriteObject、Class A） | ホストの取得・補填・ミラーの push、Cloud Run | 1.5 万回/日（11/11 は 7,200） | — | 約 12 円/日 | 350 | 600（2.5 万/日） | 360（VPS の push・run_axes の horse_lap 保存で 1 日 300〜600 回増） | Monitoring method=WriteObject | 稼働中 |
| G3 | GCS の読み出し（ReadObject・GetObjectMetadata・GetBucketMetadata、Class B） | ホスト・VPS・Cloud Run | 21.7 万回/日（開催日 31〜38 万）。Jobs 停止後: 11/12 約 19 万、金曜 11/13 約 37 万（週次の全件読み） | — | 約 14 円/日 | **約 450**（週の平均 約 24 万/日。21:00 に直した） | 720（38 万/日） | 450 | Monitoring method=ReadObject 20.3 万・GetObjectMetadata 1.2 万・GetBucketMetadata 2,000 | 稼働中 |
| G4 | **GCS の外への転送（Download APAC）** | **ホスト・VPS（GCP の外）が GCS を読む分** | 1 日 4〜10GiB（11/05〜11/11 平均 6.5GiB）。11/12 は 21 時間で 2.7GiB（fetch-agent・cache-warm の変更前） | — | 11/02〜 0 円（請求月の 100GiB の中）。11/12 21:00 JST までに 65.9GiB | **約 0〜160**（21:30 に直した。1 日 3.2〜3.5GiB × 31 − 100GiB。1 日 3.2GiB 未満なら 0、4.5GiB なら 約 750。変更前の見込みは 6.5GiB/日で 約 1,800） | **約 1,500**（6GiB/日 × 31 − 100）／10GiB/日なら 3,960 | 約 0〜160（11/15〜16 の課金で 11/14 の 1 日分を見て確定） | 課金 `Download APAC` の `price.tier_start_amount` | 稼働中 |
| G5 | **天気の取得の毎時の全件読み（11/13 13:40 再開 → 14:08 停止）** | **ホスト**（root cron `40 * * * *` → `scripts/cron/jma_weather_run.sh`） | 24 回/日 | 月ファイル 1,310 件・**345MiB**（Class B 1,310 回） | 13:40 の 1 回で確認（`logs/jma_weather.log`: skipped 39,500） | **約 4,400**（8.1GiB/日 × 30 × 18.86 円。Class B 月 約 60 円を含む） | 約 4,640（無料枠なし） | 同左（計画 polymorphic の ① に見積もりなし） | `src/scraper/jma_weather.py:242`・`:209`、`jma_weather_run.sh` の `run fetch --from 20160101` | **廃止（2026-11-13）** |
| | **GCS の操作と転送の計（G5 を除く）** | | | | | **約 2,850** | **約 5,710** | **約 2,900** | | |

参考（GCS 全体の送信量。東京の中の Cloud Run・Jobs への分を含む）: ReadObject 6〜12GiB/日、ListObjects 3〜3.4GiB/日（Jobs 停止後）。
課金の Download APAC（GCP の外への分）は 1 日 4〜7GiB で、全体の約半分が外に出ている。

### 2.5 合計

| | 見込み（今） | 最悪 | 今日の予定の後（見込み） |
|---|---|---|---|
| 固定費（2.1） | 360〜375 | 475 | 365〜380 |
| Cloud Run サービス（2.2） | 120〜180 | 1,030 | 120〜180 |
| Cloud Run Jobs（2.3） | 0 | 15（再開すれば +2,700） | 0 |
| GCS の操作（2.4 の G1〜G3） | 910（21:30 に直した。変更前 1,050） | 2,620（G1 の最悪を API の一覧の最悪で直した） | 910 |
| GCS の外への転送（G4） | 0〜160（1 日 3.2〜3.5GiB。0〜750 の幅） | 1,500 | 0〜160 |
| 天気の取得（G5） | 0（廃止） | 0（廃止） | 0（廃止） |
| 今日の変更の残り（4 章。監視 d6c72260・hotfix・run_axes。VPS 取得の段 1〜5 は取得役の停止中で 0） | +10 | +50 | +10（段 6 で取得を渡すと +約 50） |
| **合計（21:30 に直した）** | **約 1,400〜1,640** | **約 5,690** | **約 1,400〜1,690** |
| 上限 2,000 円との差 | **−360〜−600（上限内）** | +3,690 | −310〜−600 |
| 参考: 14:50 の時点の合計（G1・G3・G4 を直す前） | 約 3,330〜3,410（差 +1,330〜1,410） | 約 7,220 | 約 3,360〜3,460 |
| 参考: G5 が毎時のまま動いた場合 | 約 8,000〜8,100 | 約 12,100 | |
| G4・G5 を除いた合計（Scheduler・AR の変更前） | 約 1,820〜1,880 | 約 3,740 | 約 1,840〜1,900 |
| G4・G5 を除いた合計（11/13 の変更後） | 約 1,530〜1,610 | 約 3,450 | 約 1,560〜1,650 |
| G4・G5 を除いた合計（21:00、fetch-agent の停止・cache-warm の合図方式の後） | 約 1,400〜1,480 | 約 4,190 | 約 1,400〜1,530 |

直近の実績（課金エクスポート、JST の日付、クレジット差し引き後）:

| 日付 | 合計 | GCS | Cloud Run | AR | Scheduler | 備考 |
|---|---|---|---|---|---|---|
| 10/30 | 1,459 | 1,246 | 160 | 52 | 0 | ホストの事前生成の周回で転送 約 60GiB/日 |
| 10/31 | 1,566 | 1,299 | 213 | 55 | 0 | 同上（Download APAC 1,131 円） |
| 11/01 | 1,152 | 969 | 138 | 45 | 0 | 朝まで転送の超過 |
| 11/02 | 150 | 105 | 2 | 43 | 0 | 転送は 100GiB の無料の中に入った |
| 11/03 | 91 | 62 | 0 | 29 | 0 | |
| 11/04 | 93 | 74 | 2 | 17 | 0 | |
| 11/05 | 74 | 58 | 4 | 12 | 0 | |
| 11/06 | 176 | 145 | 10 | 16 | 5 | 書き込み 12 万回（Jobs） |
| 11/07 | 106 | 75 | 6 | 15 | 9 | |
| 11/08 | 199 | 53 | 127 | 9 | 9 | Cloud Run の無料枠が尽きる（開催日） |
| 11/09 | 246 | 62 | 168 | 6 | 9 | 開催日。Jobs CPU 143 円 |
| 11/10 | 128 | 45 | 68 | 6 | 9 | |
| 11/11 | 131 | 40 | 76 | 6 | 9 | |
| 11/12 | （不完全） | | | | | 17:00 JST に Jobs の Scheduler 停止。GCS の 14 時以降が未反映 |

- 直近 7 日（11/05〜11/11）の平均: **151 円/日 → 月 約 4,540 円**（内訳: Jobs CPU 51・Class A 40・Class B 22・AR 10・Scheduler 7・保存 6・Jobs メモリ 5・API CPU 5・API 送信 4）。
- ただしこの 7 日の GCS の転送は無料の 100GiB の中なので 0 円で入っている。転送料が付く月後半はこれより高くなる。
- 11/12 の Jobs 停止の効果は、課金エクスポートでは 11/14〜15 に確かめる（1 日 約 60 円 → 約 0 円になっていること、Class A・B が減ること）。

### 2.6 12 月の見込み（2026-11-13 21:00、今の構成）

| | 月額（円） | 根拠 |
|---|---|---|
| 固定費（GCS 保存・AR・Scheduler 1 件・Secret） | 360〜375 | 2.1。Scheduler の数量が 1 件に下がったことは 11/14〜15 の課金で確かめる |
| Cloud Run サービス（API の CPU・送信、監視） | 120〜180 | 2.2。Jobs を止めたので API の CPU は無料枠の中 |
| Cloud Run Jobs | 0 | 毎朝の g-0730-shadow 1 回だけ（無料枠の中） |
| GCS の一覧（G1） | 約 110 | 1 日 約 5,000 回（15:00〜20:50 の実測、1 時間 約 250 回）から verify_past_values の 約 480 回を引いた値 |
| GCS の書き込み（G2） | 約 350 | 1 日 1.3〜1.6 万回 |
| GCS の読み出し（G3） | 約 450 | 週の平均 1 日 約 24 万回（平日 約 19 万、金曜 約 37 万） |
| 今日の変更の残り（監視 d6c72260・hotfix など） | 約 10 | 4 章 |
| **小計（転送を除く）** | **約 1,400〜1,480** | |
| GCS の外への転送（G4） | 0〜750（中央 0〜160） | 12 月は 31 日。1 日 3.2GiB 未満なら 0（無料の 100GiB の中）、3.5GiB で 約 160、3.8GiB で 約 330、4GiB で 約 450、4.5GiB で 約 750 |
| **合計** | **約 1,400〜1,640（中央）／最大 約 2,230** | |
| **上限 2,000 円との差** | **−360〜−600（上限内）／最大 +230** | 1 日の転送が 約 4.1GiB を超えると上限を超える |

- 転送の 1 日の量の根拠: 11/12（木、fetch-agent・cache-warm・verify_past_values の変更前、21 時間分）で 2.7GiB。verify_past_values の一覧 0（280a6ab0）で 1 日 0.3〜0.4GiB 減る。そこから VPS の一覧の応答が減り、週次の全件読み（週 約 1.5〜1.9GiB ≒ 1 日 約 0.25GiB）が乗る。開催日（土日）は多め。**11/14（土）の 1 日分を 11/15〜16 の課金エクスポートで見て確定させる**。
- 見込みを崩すもの: (1) 7e1cc81f より前の版を master に昇格して Scheduler が作り直される（#6、+1,700〜2,700）、(2) 朝の呼び手の分からない大量読みが 11/13 から新しく始まったホストの読みだった場合（#2。毎日のものなら 11/12 の実績に入っている）、(3) VPS の取得を渡す段 6（+約 50。ホストが GCS から読み直す量は段ごとに実測する）、(4) API の起動し直しの最悪の回数（#4、無料枠を超えると +700）、(5) Cloud Run の API に残る一覧（#2 の「Cloud Run の API に残っている一覧」。今は 月 約 85 円、最悪 約 1,250 円）。

## 3. 監査の結果（重さの順）

### #0 【解決: 廃止 2026-11-13】天気の取得の再開で、ホストが毎時 345MiB を GCS から読んでいた（月 約 4,400 円）
- 根拠: `scripts/cron/jma_weather_run.sh` が毎時（root cron `40 * * * *`）`daily` の後に `fetch --from 20160101` を回す。`fetch_range`（`src/scraper/jma_weather.py:225-266`）は 2016-01-01〜昨日の全日 × 10 競馬場を回り、
  `(競馬場, 月)` ごとに `load_month`（`:209`、`bypass_cache=True` でホストのキャッシュを使わない）で GCS の月ファイルを読んでから「取得済みか」を判定する。取得済みの過去の月も毎回読む。
- 実測: GCS の `chuou/data/others/jma_weather_hourly/` は 1,310 件・345MiB（1 件 最大 376KiB）。13:40 の回は `{"fetched": 190, "skipped": 39500}`（`logs/jma_weather.log`）。毎時 345MiB → 1 日 8.1GiB → 月 約 243GiB。
- 費用: 18.86 円/GiB で 月 約 4,600 円（Class B 月 3.9 万回 約 2 円）。単独で上限の 2.3 倍。11 月の無料の残り 35GiB は約 4 日で尽きる。
- 10/26 以前（中断印で止まる前）も同じ作りで毎時動いていた。10 月の転送の超過の一部だった可能性がある。
- 計画（`polymorphic-frolicking-flame.md` の ①）には費用の見積もりが無く、CLAUDE.md の決まり 1・2（周期ごとに GCS を大量に読まない）と「ホストから GCS を繰り返し読まない」に反する。
- 直し方の案（統括役へ）: (a) 今すぐ、毎時の `fetch --from 20160101` を外して `daily`（直近 3 日、月ファイル 10〜20 件・約 5MB）だけにする。過去分の補いは 1 回だけ手で回す。
  (b) 取得済みの (競馬場, 月) をホストの手元の記録（決まり 6 と同じ考え方）に持ち、埋まった月は読まない。(c) `daily` も毎時でなく 1 日 1〜2 回にする（毎時だと 5MB × 24 ＝ 月 約 3.6GiB・約 68 円）。

### #1 GCS の外への転送が 1 日 4〜10GiB あり、月の後半に 1 日 約 100 円ずつ乗る（最大の項目・cost_guard が事前に気づけない）
- 課金エクスポートの `Download APAC` は 11/02 以降 0 円だが、使用量は 11/05〜11/11 で 1 日 3.9〜9.6GiB（平均 6.5GiB）。0 円は請求月ごとの最初の 100GiB の段階の単価のため。11/12 午前で 65GiB を使った。
- 見込み: 11/18〜19 に 100GiB を超え、残りの約 12 日で約 1,100〜1,400 円。12 月以降は 6.5GiB/日のままなら月 約 1,800 円（上限の 9 割）。
  **21:00 の更新**: 天気の廃止・調教の判定（957858ad）・fetch-agent の停止・cache-warm の合図方式の後は、1 日 約 3〜4GiB の見込み。11 月は 11/21〜24 ごろに 100GiB を超えて 約 +400〜750 円、12 月は 約 0〜160 円（0〜750 の幅。21:30 に verify_past_values の一覧 0 を入れて直した）。確定は 11/15〜16 の課金エクスポート。
- cost_guard（`src/ops/cost_guard.py:6-9`）は直近 3 日の平均で月末を見込むので、0 円の日が平均に入っている間は超過を予告できない。止められる対象も Scheduler だけ（`:14` に明記）で、ホスト・VPS の読み出しは止められない。
- 出どころ: 3 章の #2 と下の「出どころの調べ」を参照。CLAUDE.md の「GCP の外から GCS を繰り返し読まない」に反している処理がある。

### #2 GCS の一覧（ListObjects）が Jobs 停止後も 1 日 約 1.2 万回・約 3.4GiB（Class A 月 約 290 円＋外に出る分の転送）
- 1 回の応答が平均 約 280KB と大きく、大きな接頭辞（年ごとのカテゴリなど）を何度も一覧している形。決まり 2「GCS を全件一覧する処理を周期ごとに入れない」に反するものが残っている。
- 【解決: 3fcb2a2b】ホストの毎朝の書き出し（`scripts/cron/export_tables_daily.sh` → `src/scraper/export_tables.py` の `storage.batch_list_blobs`）が、カテゴリ 6 つ × 年の全件一覧を毎日行っていた（parse_drift_docs と合わせて 1 日 約 30 回・約 90MB）。3fcb2a2b で 13 カテゴリを手元の母集合（race_lists・flat）から名前で読む作りにし、**一覧は 0 回**（月 約 50 円 → 最悪 約 10 円・見込み 約 4 円）。毎晩の PG 同期の一覧（毎晩 2 回）も c03b86f7 で 0 回にした。
- **11/13 15:00 以降の一覧（G1）は 1 日 約 5,000〜6,000 回**（下の「監査ログの集計」）。見出しの 1.2 万回は変更前。
- 出どころの調べ（11/13、コードの調べ＋目印の件数の 1 回の数え）:
  | 呼び手 | 動く場所・頻度 | 1 日の量（推定） | 根拠 | egress_billed |
  |---|---|---|---|---|
  | 【済み: 16:30 に合図方式】VPS の `keiba-cache-warm.timer` が `ops/cache_dirty/` の昨日以降を毎回一覧 | VPS・60 秒おき（1,440 回/日）→ 配備後は一覧 1 時間 6〜9 回（1 日 約 150〜220 回）、合図のメタデータ読み 毎分 1 回 | 一覧 1,440 回（Class A 月 約 35 円）・目印 約 500 件 × 約 0.6KB × 1,440 ≒ **0.4GiB**（月 約 230 円の転送） | `scripts/ops/vps/cache_warm.py:254-261`（fields の指定なし）。目印は 11/02 の bbc45db6 から馬の保存ごとに増える（`src/utils/cache_marker.py:89-94` ← `src/scraper/storage.py:543`）。1 日 300〜550 件（11/13 実測） | 見ていない |
  | 【解決: 280a6ab0】ホストの `verify_past_values` が 10 年分を一覧 | ホスト・2 時間おき（12 回/日） | 約 3.45 万件 × 12 回 ≒ 0.3〜0.4GiB → **一覧 0 回**（対象はローカルの race_result_flat、基準年の値は 30 日使い回し、転送課金の場所では 1 回 500 件まで）。コミットの見積もり: 月 約 200 円 → 最悪 約 11 円 | `src/scripts/data/verify_past_values.py:151,163` → `storage.py:1448-1459`（fields の指定なし） | 見ていない |
  | 【解決: 3fcb2a2b】ホストの毎朝の書き出しの全件一覧 | ホスト・1 日 1 回 | 変更前 1 日 約 30 回・約 90MB → **0 回** | 3fcb2a2b（`export_year_local` を 13 カテゴリに）、PG 同期は c03b86f7 | — |
  | **VPS の `keiba-fetch@` サービスアカウント（取得の実行役 fetch_agent と思われる）** | **VPS（呼び元 IP 2400:8500:…:250:252）・60 秒ごとに一覧 6 回** | **一覧 8,640 回/日**（Class A 月 約 210 円）＋応答の転送 | 監査ログ（11/13 14:44〜15:00 JST で 90 回、毎分 :52 に 6 回まとめて）。30097498 の見積もり（月 8.6 万回）の 3 倍。担当表は空でも一覧はしていた。**11/13 15:00 JST に統括役が keiba-fetch-agent を止め、15:05 以降は 0 回**（20:50 まで全件の数で確かめた）。再開するなら `KEIBA_FETCH_TASK_STORE=file` にしてから（しないと 8,640 回/日が戻る） | 見ていない |
  | Cloud Run の `keiba-api@`（IP 2600:1900:…、Google の中） | Cloud Run | 14:44〜15:00 で 37 回 → 1 日 約 3,300 回（Class A 月 約 80 円。転送は東京の中なので 0） | 監査ログ | — |
  | VPS の `keiba-pull@`（cache_warm の一覧・release-pull の取得） | VPS | 一覧 毎分 1 回 → 1 日 1,440 回、取得（latest.json）1 日 約 1,200 回 | 監査ログ（15:05〜15:17 で一覧 11・取得 10） | 見ていない |
  | **ホストの `megu-keiba@`（IP 175.177.45.24）が `horse_training/2016〜2018/` を一括で読む** | **ホスト・まとまって（15:05〜15:06 に 4,100 件、15:16〜15:17 に 7,300 件）** | **12 分で 1.15 万件 × 平均 約 65KB ≒ 0.7GB**（1 件 15〜115KB を 5 件測った）。続く頻度はまだ分からない | 監査ログ。UA は gcloud-python 3.10.1 / Python 3.11.6（ホスト）。**原因（11/13 特定）**: `backfill_horse_training_recent --dry-run`（別のセッションが 3 回流した）の対象判定 `src/scripts/data/drift_coverage.default_loader` が、HybridStorage.load で GCS から 1 頭ずつ読む作り。同じ読み方を**夜間の調教の補い（root cron `0 13 * * 1-4` UTC ＝ 月〜木 22:00 JST）が毎回行っていた**：週 4 回 × 約 0.5GB ＝ 月 約 8GB・**約 150 円**（100GiB を超えた月）。**【解決: 957858ad（11/13 15:29 JST）】手元の複製（`data/local/horse_training`）を先に読み、GCS は GCP の中でだけ読む作りに直した → 月 約 150 円 → 0 円**。15:00〜15:17 の 約 1.9 万件の読みは直す前の dry-run で、一度きり | 見ていない |
  | **ホストの週次の全件読み: weekly-update（金 18:00 JST）と horse-name-index（金 18:30 JST）が、どちらも race_shutuba の全年を読み直す（二重）。続けて成長曲線の週次更新が horse_result を全頭読む** | ホスト・週 1 回（金曜。root cron `0 9 * * 5`・`30 9 * * 5` UTC） | race_shutuba 約 3.5 万件 × 2 回（18:20〜18:50）＋ horse_result 約 5.8 万件（18:50〜19:45）≒ **1 回 約 1.5〜1.9GiB → 月 約 7〜8GiB**（無料枠を超えた月で 約 130〜150 円）。Class B 月 約 60 万回 ≒ 40 円 | 監査ログ（2% の標本、11/13 18:20〜19:50）。`src/utils/horse_name_index.py:339-392`（`rebuild_horse_name_index_from_gcs_race_shutuba` が年ごとに `list_keys` して全件 `load`）、`src/scraper/auto_scrape.py:1008-1028`（horse-name-index）と `:1355-1358`（weekly-update の中でも同じ関数を呼ぶ）。`logs/weekly_update.log` 18:21〜18:45 と `logs/horse_name_index.log` 18:30〜18:49 に同じ「馬名インデックス shutuba: year=…」が出ている。成長曲線は `GrowthCurveService`（58,417 頭）。**CLAUDE.md の決まり 2 に反する。直し方の案: weekly-update 側の再構築をやめて 1 回にし、手元の flat（race_shutuba_flat）か horse_result の保存時の upsert（`horse_name_index.py:155`）に寄せる** | 見ていない |
  | **【11/14 に特定】朝 06〜08 JST の大量読み = ホストの backfill-all-horse（毎日 06:00 JST）。10〜11 JST の 1.5GiB = 11/13 だけの手の書き出し** | ホスト（`megu-keiba@`・175.177.45.24）・毎日 06:00〜07:50 JST | 約 8 万回・約 0.65〜0.8GiB/日（Class B 月 約 145 円＋転送 月 約 21〜25GiB） | `src/scraper/backfill.py:347-355`（race_result・race_shutuba を年を絞らず全件 `list_keys`＋`load`）。10〜11 JST は `logs/export_tables_2026_once_20261113.log`（一度きり）。3 章 #7 | 見ていない |
  | 残り | 監査ログを 24 時間ためてから集計（11/14） | | | | — | Cloud Run の中の一覧なら転送 0 円。VPS の nginx の上流がホストの API なら、cache_warm の取り直しがホストの API で GCS を読む経路がある（`scripts/ops/vps/nginx-keiba-ui.conf:15`） | — |
  - `egress_billed()` を見て止めているのは `src/api/prerender.py:66-68` の 1 か所だけ。上の周期処理はどれも見ていない。
  - 監査ログ（14:44 JST から）の最初の 5 分: GCS の読み出し 25 件はすべて VPS から（keiba-fetch@ の一覧 18、keiba-pull@ の一覧 3・取得 4 ＝ release-pull と cache_warm）。
    集計の道具は監査役の scratchpad の logs.py（呼び手・IP・メソッド・接頭辞ごとの件数）。応答のバイト数は監査ログに無いので、件数 × 1 回の量で見積もる。
  - **監査ログの集計（11/13 14:44〜20:50 JST の約 6 時間。24 時間の一部）**: get（ReadObject と GetObjectMetadata を合わせた `storage.objects.get`）は `sample(insertId, 0.02)` の 2% の標本 × 50 の推定、list は全件の数。
    読む量を小さくするため、`entries:list` に `fields=entries(timestamp,protoPayload),nextPageToken` と標本の条件を付けた（1 回 約 4,500 件）。

    | 呼び手（principal） | 呼び元 IP | メソッド | 件数（6 時間） | 主な中身 | 転送料 |
    |---|---|---|---|---|---|
    | `megu-keiba@`（ホスト） | 175.177.45.24 | get | **約 15.7 万**（標本 3,140） | 金曜の週次の全件読み（race_shutuba 約 7.6 万・horse_result 約 5.8 万）、15:00〜15:17 の調教の dry-run 約 1.9 万（957858ad の前、一度きり） | **かかる** |
    | `keiba-api@`（Cloud Run） | 2600:1900:…（Google の中） | get | 約 6.5 万（標本 1,293） | 16〜19 時に集中。horse_result・horse_pedigree_5gen・horse_training・race_result など preprocessed、ops/mirror の目録・束 | 0（東京の中）。Class B のみ |
    | `keiba-api@` | 同上 | list | 887（全件） | 1 時間 約 150〜220 回 | 0。Class A のみ |
    | `megu-keiba@` | 175.177.45.24 | list | 394（全件） | 18 時台に 233 回（週次の `list_keys`）、ほかは verify_past_values（2 時間おき）など 1 時間 10〜50 回 | かかる（応答） |
    | `keiba-pull@`（VPS） | 2400:8500:…:250:252 | list | 124（全件） | cache-warm。16:30 の合図方式の後は 1 時間 6〜9 回 | かかる（小） |
    | `keiba-pull@` | 同上 | get | 約 550（標本 11） | 合図 `ops/cache_dirty/_beacon.json`（毎分）と release-pull の `ops/releases/ui*/latest.json` | かかる（小） |
    | `keiba-fetch@`（VPS） | 同上 | list | 90（全件） | 14:44〜15:00 だけ。**15:05 以降 0** | — |

    まとめ: GCP の外への転送になるのはほぼホスト（`megu-keiba@`）の読み。周期のものでは金曜の週次の全件読みが最大。VPS の分は fetch-agent の停止と cache-warm の合図方式で小さくなった。
    11/14 の朝（06〜11 JST）の分を、監査ログを戻す 14:44 JST の前に同じやり方で集計し、まだ分からない朝の大量読みの呼び手を確かめる。
  - **Cloud Run の API に残っている一覧（統括役の依頼、21:30。直すのはユーザーが戻ってから）**
    前提: `HybridStorage.batch_list_blobs`（`src/scraper/storage.py:1157`）は、当年の一覧を 60 秒（`:1104`）、過去の年を 1 時間（`:1103`。ただし同じプロセスで保存があった後は 60 秒）、メモリとディスク（Cloud Run では揮発）に持つ。
    起動し直すとどちらも空になる。1 年分の一覧は 1 カテゴリ 約 3,000〜3,500 件 → 1,000 件ずつで **約 4 ページ ＝ Class A 4 回**（監査ログの 1 件も 1 ページ）。東京の中なので転送 0、Class A 0.8 円/千回。
    最悪は CLAUDE.md の決まり 1 に合わせ、「キャッシュの無い一人目」が 1 日 96 回（15 分ごと）来るとみなした。

    | 場所 | 経路 | 呼ばれ方 | 1 回の一覧 | 最悪の回数 × 30 | 最悪の月額 | 備考 |
    |---|---|---|---|---|---|---|
    | `src/api/app.py:2004` | `GET /api/row-data-coverage`（`:2501`） | 要求ごと（監視の画面で日付を開いたとき） | race_shutuba の当年 1 カテゴリ ≒ 4 回 | 4 × 96 × 30 ≒ 1.2 万回 | 約 9 円 | 60 秒より後の要求はまた一覧 |
    | `src/api/app.py:2141` | `GET /api/date-race-matrix`（`:2086`） | 要求ごと | `TRACK_CATEGORIES` 17 カテゴリ × 当年 ≒ 約 68 回 | 68 × 96 × 30 ≒ 19.6 万回 | 約 160 円 | 監視の日付の行列。当年は 60 秒で切れるので、開くたびにほぼ毎回一覧 |
    | `src/api/monitor_coverage.py:114` | `GET /api/date-raw-matrix`（`app.py:2220`）、品質の検査 `quality_health.run_presence_check`（`quality_health.py:232`） | 要求ごと | 17 カテゴリ × 当年 ≒ 約 68 回 | 68 × 96 × 30 ≒ 19.6 万回 | 約 160 円 | 上と同じ一覧を別のキャッシュの鍵なしで取り直す（同じ HybridStorage のキャッシュは共有） |
    | `src/api/app.py:2448` | `POST /api/scrape-missing`（`:2397`） | 要求ごと（手で押す操作） | 指定カテゴリ × 年 ≒ 4 回 × カテゴリ数 | 手の操作なので 1 日 数回 | 1 円未満 | 周期では呼ばれない |
    | `src/api/app.py:3242` | `GET /api/scrape-summary-all`（`:3148`。監視の一覧画面、`src/monitor/app.py` から） | 要求ごと、ただし結果を 5 分持つ（`:3163`） | `_SUMMARY_SOURCES` 9 カテゴリ × race_lists の全年（2016〜2026 の 11 年）≒ 99 一覧 × 4 ≒ **約 400 回**（起動直後。過去の年はその後 1 時間持つ） | 400 × 96 × 30 ≒ 115 万回 | **約 920 円** | 最大の項目。加えて race_lists を全日付読む（Class B 約 1,000 件 × 96 × 30 ≒ 290 万回 ≒ 180 円） |
    | `src/data_health/coverage.py:197`（`collect_presence`） | `data_health.runner`（`runner.py:81`）、`horse_coverage`（`coverage.py:358`）。`scrape_from_health_plan` から | 手で回す検査（root の crontab・job_map に定期の登録は無い） | カテゴリ × 年の全件。毎回 `invalidate_blob_cache` で取り直す | 定期では 0 | 0（1 回 数百回 ≒ 1 円未満） | ホストで回すと応答は転送料（1 回 数十〜数百 MB）になる。定期に入れるときは要審査 |

    監査ログとの照らし合わせ: keiba-api@ の一覧は 14:44〜20:50 JST で **887 回**（1 時間 150〜220 回、16〜19 時に集中）→ 1 日 約 3,300〜3,500 回 ≒ **月 約 85 円**。
    最悪の合計は 約 1,250 円（上の表の和）で、実績はその 1 割未満。実績の多くは、監視の画面を開いている間の date-race-matrix・date-raw-matrix（1 回 約 68）と、起動直後の scrape-summary-all（約 400）で説明できる大きさ。
    直し方の案（ユーザーが戻ってから）: (a) 当年の一覧の 60 秒を 5〜15 分にする（保存のときの `invalidate_blob_cache` で新しさは保てる）、(b) scrape-summary-all と行列は、ホストが毎朝作る flat・coverage（`data/local/meta/coverage`）か ops/mirror の目録から作り、GCS の一覧をやめる、(c) 過去の年の一覧は束（`ops/mirror/_bundle/base.tar.gz`）に入れて起動時に読む。
  - 前の確かめ方の案（済み）: VPS の `/etc/nginx/sites-enabled/keiba-ui` の上流を見る。GCS のデータアクセスの監査ログを 1 日だけ有効にして呼び手（IP・サービスアカウント）を数える（Cloud Logging の無料 50GiB の範囲）。

### #3 固定費だけで月 約 650 円（上限の 3 分の 1）。止めている Scheduler も課金されている
- Cloud Scheduler: 課金の数量は 11/06〜11/12 ずっと 17 件（smoke-auto の一度きりの 3 件・停止中の keiba-g-* 13 件を含む）。停止中でも 1 件 約 15 円/月かかり、無料の 3 件も効いていないように見える。
  smoke-auto 3 件の削除（11/13）で −45 円。11/14 の課金で数量が 14 になったか確かめる。停止中の keiba-g-* 13 件を消せば、さらに 約 −195 円（cost_guard が止める対象が無くなるので、作り直しの手順を残してから）。
- Artifact Registry 16GiB（月 約 240 円）: stg への配備ごとに API の版 0.87GB が増え、2 日の保持で 1 日の配備回数 × 2 日分が残る（11/11〜12 で 13 版）。Jobs の版（`live-jobs`・`live-jobs-a/b` で 3.4GB、月 約 50 円）は Jobs 停止中も残る。
  `setup_project.sh:35` の保持を 2 日 → 1 日にすると 約 −80 円、Jobs を当面使わないなら live-jobs-a/b のタグを外して 約 −35 円。

### #4 最悪の回数で見積もると、API の起動し直しだけで無料枠を超える（実績は無料枠の中）
- keiba-api は起動時 CPU ブースト（startupCpuBoost）あり・1 回 約 50 秒。決まり 1 の最悪 96 回/日では、起動だけで月 28.8 万 vCPU 秒になり、無料の 18 万を超えて 約 700 円。
- 実績（11/05〜11/12）は 1 日 1,100〜3,700 vCPU 秒で無料枠の中。ただし無料枠は Jobs と共有なので、**Jobs を 1 本でも再開すると API の分も月の途中から課金になる**（11/08 以降がそれ）。
- stg と本番が同じサービスの別のタグで動くので、起動し直しは 2 系統分になりうる。

### #5 今日の計画の見積もりの抜け
- **天気の取得の再開（polymorphic-frolicking-flame の ①）に費用の見積もりが無かった**。調べた結果が #0（毎時 345MiB、月 約 4,400 円）。
- **VPS の取得の計画（snoopy-wondering-wilkes）の「後処理がホストで読む VPS の保存分」（36 レース × 8 種 × 20KB、月 約 3 円）は小さすぎる可能性**。担当を VPS に渡すと、今ホストが取って手元に持っているもの（オッズ・馬体重・結果・払戻・馬の成績・調教）を、ホストは GCS から読み直すことになる。
  ホストの取得量（1 日 数十〜数百 MB）がそのまま転送になると、月 1〜6GiB・20〜110 円。すでに 100GiB の無料を超える月なので、全額が課金になる。段 1〜5 の各段で「ホストが GCS から読む 1 日の量」を実測して台帳に入れることを条件にする。
- **30097498 の「月 約 −150 円」は、担当を VPS に渡して `KEIBA_FETCH_TASK_STORE=file` にしたときに初めて効く**（コミットメッセージにもそう書いてある）。それまでは VPS の fetch_agent の 60 秒おきの一覧（月 約 4.3 万回 × 0.8 円/千 ＝ 約 35 円＋応答の転送）がそのまま続く。−150 円は一覧の回数（8.6 万回）を 2 倍に見ている可能性がある。
  **21:00 の更新**: 監査ログの実測では fetch_agent の一覧は 1 日 8,640 回（毎分 6 回）で、見積もりの 3 倍だった。**11/13 15:00 JST に統括役が keiba-fetch-agent を止めたことで、一覧 −8,640 回/日（Class A 月 約 −210 円）が実現した**（file モードによるものではない）。file モードにしないまま再開すると、この分は戻る。
- 077bc683 の hotfix を master に入れると本番の配備が 1 回増え、Artifact Registry に 0.87GB が 2 日残る（約 +5 円）。読み出しは減るだけなので問題なし。

### #6 【重大 → 7e1cc81f で修正（develop）】master への昇格のたびに、止めた Scheduler（keiba-g-* 13 件）が作り直される（月 約 +1,700 円）
- リポジトリの変数 `ENABLE_JOBS_DEPLOY=true` のため、master への昇格（`.github/workflows/promote.yml:121` の「Jobs の更新」）で `scripts/ci/deploy_gcp.sh:172`（`jobs` の段の `deploy_cloud_run_jobs.sh apply scheduler`）が動く。
  `src/ops/scheduler_gen.py:94-111` は、`config/gcp_job_map.json` の中で ready かつ enabled のジョブを 1 つでも持つグループの Scheduler を `describe` して、無ければ `create`・あれば `update` する。
  11/13 14:4x に消した keiba-g-* 13 件のうち ready なグループは、昇格のたびに**作り直されて ENABLED で動き出す**（cost_guard は 1 日 2 回なので、止めるまで最大 半日 動く）。
- 費用: 動き出すと 2.3 の J1 のとおり、Jobs の CPU・メモリで 月 約 +1,700 円（GCS の操作 約 +1,000 円を足すと 約 +2,700 円）。cost_guard が止めても、PAUSED のまま 13 件 × 15 円 ≒ 約 +195 円/月の固定費が戻る。
- 対応: **7e1cc81f（cron-catchup-weekly-decision のセッション、11/13 20:47 JST）**で、`config/gcp_job_map.json` の 13 グループ（g-0530・g-0700・g-0730・g-0730-dayprep・g-0800・g-1000-mon・g-1910-eve・g-2130・g-5min・g-hourly・g-pump・g-results・g-sun-2200）に retired を付け、`scheduler_gen` は retired のグループを `--include-unready` でも作らないようにした（テスト `tests/ops/test_scheduler_gen_retired.py`）。コミットの確認では、出るコマンドは keiba-g-0730-shadow の update（無ければ create）だけ。
  **残る条件: master への昇格は 7e1cc81f を含む版でだけ行う。昇格の後に `gcloud scheduler jobs list` が keiba-g-0730-shadow の 1 件のままであることを確かめる**。
  別の手として、リポジトリの変数 `ENABLE_JOBS_DEPLOY` を false にすれば `deploy_gcp.sh:170-172` で Jobs の更新ごと飛ばせる（Jobs の版の `live-jobs` タグの付け替えも止まる点に注意）。

### #7 監査ログの最終集計（11/13 14:40 〜 11/14 14:30 JST の 24 時間。2026-11-14 の監査役）

集め方: GCS の DATA_READ 監査ログ（11/13 14:44 JST 有効化、11/14 14:44 すぎにユーザーが無効化）。`storage.objects.get` は `sample(insertId, 0.02)` の 2% の標本 × 50、`storage.objects.list` は全件（3,416 件）。
監査ログには応答のバイト数が無いので、Cloud Monitoring の `storage.googleapis.com/network/sent_bytes_count`（10 分刻み、method 別）を、同じ 10 分の呼び手の件数の比で割り振った（1 件の大きさが違う呼び手が同じ 10 分に混ざると誤差が出る。下の backfill は 06 時台の単独の時間から 1 件の大きさを出して幅で示した）。
GCP の内か外かは呼び元 IP で分けた（`2600:1900:…` は Google の中 ＝ Cloud Run・Jobs、東京の中なので転送 0）。道具は監査役の作業場所（`/tmp/claude-1000/audit/` の dump.py・est.py・mon_r.py）。GCS 自体は読んでいない。

**24 時間の合計（Monitoring）**: ReadObject 35.4 万回・4.89GiB、ListObjects 3,461 回・2.05GiB。

| 呼び手（principal） | 場所 | 呼び元 IP・UA | 読み出し（回・量） | 一覧（回・応答の量） | 外への転送 |
|---|---|---|---|---|---|
| `megu-keiba@` | **ホスト（GCP の外）** | 175.177.45.24・gcloud-python 3.10.1 / Python 3.11.6 | **約 24.4 万回・約 2.8GiB** | 718 回・約 0.52GiB | **約 3.3GiB** |
| `keiba-api@` | Cloud Run（内） | 2600:1900:…・Python 3.11.18 | 約 11.4 万回・約 1.95GiB | 2,338 回・約 1.37GiB | 0（Class A 月 約 56 円・Class B 月 約 200 円） |
| `keiba-pull@` | VPS（外） | 2400:8500:…:250:252・gcloud-python 3.17.1 / Python 3.10 | 約 3,100 回・約 0.07GiB（`ops/cache_dirty` の合図・`ops/releases` の latest.json） | 264 回・約 0.13GiB（合図方式の後は 1 時間 4〜16 回） | 約 0.2GiB |
| `keiba-fetch@` | VPS（外） | 同上 | 0 | 90 回（11/13 15:00 まで。止めた後は 0） | 約 0.03GiB |
| `keiba-deployer@` | GitHub Actions（外） | 20.3.167.84 | 約 100 回・約 0.02GiB（`ops/releases`・`models/fitted_artifacts`） | 0 | 約 0.02GiB |
| `keiba-jobs@` | Cloud Run Jobs（内） | 2600:1900:… | 約 1,200 回（`ops/fetch_tasks_shadow`、07:30 の shadow の 1 回） | 6 回 | 0 |
| **外への転送の計** | | | | | **約 3.6GiB/24 時間** |

ホスト（`megu-keiba@`）の読み出しの中身（カテゴリ × 時間帯）:

| 時間帯（JST） | 中身 | 回数 | 量 | 原因 | 頻度 |
|---|---|---|---|---|---|
| 11/14 06:00〜07:50 | race_result 2016〜2026 全件（06 時台）→ race_shutuba 2016〜2026 全件（07 時台） | 約 8 万 | 約 0.65〜0.8GiB（06 時台は単独で 362MiB・1 件 約 9KB、07 時台の race_shutuba は 1 件 約 7KB） | **backfill-all-horse**（下の 1） | **毎日** |
| 11/13 18:00〜19:50 | race_shutuba 全年 × 2 回・horse_result 約 5.8 万頭 | 約 17 万 | 約 1.6GiB（一覧の応答 約 0.3GiB を足すと 約 1.9GiB） | 金曜の weekly-update・horse-name-index・成長曲線（3 章 #2 の表） | 週 1 回（金曜） |
| 11/13 15:00〜15:17 | horse_training 2016〜2018 | 約 2.1 万 | 約 0.3GiB | 調教の dry-run（957858ad で修正済み） | 一度きり |
| 11/14 03:00 | 一覧 70 回 | — | 約 0.06GiB | backfill-2020-fast（UTC 火・金 18:00 ＝ JST 水・土 03:00）の年の一覧 | 週 2 回 |
| ほかの時間 | race_result_on_time・race_barometer・race_odds など当年の少量 | 1 時間 数百 | 1 時間 10〜50MiB | entries-sync・raceday 系 | 毎日 |

時間帯ごとの外への読み出し（ホスト、MiB）: 06 時 361・07 時 約 270〜430・08 時 48・10〜12 時 各 15〜20・15 時 301・18 時 876・19 時 733・21 時 53。

**1. 朝の大量読みの正体（11/13 の「まだ分からない」の答え）**
- 06〜08 JST の 約 9.6 万回・約 0.9GiB は、**ホストの root cron `0 21 * * *`（UTC ＝ 06:00 JST）の backfill-all-horse**（`python -m src.scraper.backfill --phase horse`、`logs/backfill_horse.log`）。
  - `src/scraper/backfill.py:347-355`（`_run_phase_horse`）が、馬 ID を集めるために `runner.storage.list_keys(category)` を**年を付けずに**呼び（`src/scraper/storage.py:1450-1460` は year が None だと `<category>/` 全体を一覧）、race_result・race_shutuba を 1 件ずつ `storage.load`（`storage.py:749`、L2 の TTL が切れていれば GCS）している。対象年度のログは [2020..2026] だが、ここは `years` を受け取らない（`backfill.py:892-894`）ので 2016〜2019 も読む。
  - そのあと `pending[:500]` の 500 頭について `storage.load("horse_result", hid)`（`:371`）を読み、ほぼ全部が sire ありで skipped。**ここ 9 日の horses_processed は 0〜2 頭**で、1 日 1 時間 50 分・約 8 万回の読みのほとんどが「未処理の馬を探す」ためだけに使われている。
  - Cloud Run（keiba-api）の 07:10〜07:50 の読み（horse_result・horse_pedigree_5gen・horse_training 約 2.2 万回）と 08:30〜08:40 の 409MiB（`ops/mirror/data/local/race_analysis` 約 1,300 件、1 件 約 300KB）は GCP の中なので転送 0。
- 11/13 の 10〜11 JST の 約 1.5GiB（10:25〜10:45 に 約 9,100 回・1 件 約 150KB）は、別のセッションが 10:21 JST に手で流した**2026 年の書き出し**（`logs/export_tables_2026_once_20261113.log`、13 カテゴリ × 約 3,000 レース）。11/14 の同じ時間は 95MiB で、**毎日のものではない**。
- どちらも 11/12 以前の課金の実績（1 日 3.7〜4.8GiB）に含まれていたもので、見込みへの上乗せではない。

**2. ふだんの日の外への転送（今の構成）**

| 項目 | 1 日の量 | 根拠 |
|---|---|---|
| backfill-all-horse（毎日） | 約 0.7〜0.8GiB（一覧 76 回 約 0.04GiB を含む） | 上の表 |
| ホストのほかの読み・一覧（entries-sync・raceday 系など） | 約 0.1〜0.15GiB | 上の表の「ほかの時間」 |
| VPS（keiba-pull@） | 約 0.1GiB | 合図方式の後の一覧 1 時間 4〜16 回＋合図・latest.json |
| GitHub Actions の配備 | 約 0.02GiB/回 | keiba-deployer@ |
| **ふだんの日の計** | **約 1.0〜1.1GiB** | |
| 金曜の週次（weekly-update・horse-name-index・成長曲線） | ＋約 1.9GiB/週 | 3 章 #2 |
| backfill-2020-fast・2021-fast の年の一覧 | ＋約 0.06GiB × 週 2〜4 回 | 11/14 03:00 |
| **週の平均** | **約 1.3GiB/日（月 約 40GiB）** | |

- 課金エクスポートの実績: 11/12 は 3.70GiB、11/13 は 3.71GiB（21:00 JST までの一部。手の書き出し 1.4GiB・調教の dry-run 0.3GiB・金曜の週次を含む）。どちらもこのあと止めた・直した処理（fetch-agent・cache-warm の毎分の一覧・verify_past_values・調教の補いの夜の読み）を含むので、ふだんの姿は上の表のほうに近い。**11/15〜16 の課金エクスポートで 11/14（土）・11/15（日）の 1 日分を見て確かめる**（土日は raceday 系が多いので 1.0〜1.5GiB を見込む）。
- 費用: 12 月は 31 日 × 1.3GiB ≒ 40GiB で、請求月の無料の 100GiB に収まり**転送は 0 円**。手で流す一度きりの全件読み（11/13 の書き出しのような 1〜2GiB）は月 30 回ぶん入る余裕がある。上限に効くのは「全件読みを周期に入れる」変更だけ。
  11 月は 11/13 21:00 JST までで 70.7GiB。残り 約 17.8 日 × 1.3GiB ≒ 23GiB → **月末 約 94GiB**。手の一度きりの読みが合わせて 6GiB を超えると、超えた分が 1GiB 18.86 円で 11 月に乗る。

**3. 対策（コードはこの作業では変えない。担当の案つき）**

| # | 対策 | 場所 | 効果の見込み | 担当の案 |
|---|---|---|---|---|
| A | **backfill の馬 ID の集め方を、GCS の全件読みから手元の flat に替える**。`collect_horse_ids_from_tables(TABLES_DIR, years)`（`src/pipeline/build_horse_entity_store.py:90`、race_result_flat・race_shutuba_flat の horse_id 列だけ）を使う。2026 の flat が古い日のために、当年だけは GCS の年つき一覧（`batch_list_blobs`、`storage.py:1157`）か race_lists を足す。少なくとも `src.config.gcp_guard.egress_billed()`（`src/config/gcp_guard.py:40`）が True のときは GCS を全件読まない（957858ad の `src/scripts/data/drift_coverage.py:38-42` と同じ形） | `src/scraper/backfill.py:347-355`、呼び出し `:892-894` | **Class B −約 8 万回/日（月 約 −145 円、確実）・転送 −約 0.7GiB/日（月 約 −21GiB。無料枠を超える月なら −約 400 円）**。06:00〜07:50 の 1 時間 50 分も数分に縮む | 取得（scraper・backfill）の担当セッション。CLAUDE.md の決まり 2 に反する周期処理なので、統括役の優先度は高め |
| A' | （A の前の応急）処理する頭数が 0 の日が続くなら、毎日 → 週 1 回に下げる（root cron `0 21 * * *` → 曜日を 1 つ。`scripts/cron/` の登録スクリプトを lib_crontab で入れ替え） | root crontab `KEIBA_BACKFILL` の backfill-all-horse | 上の 6/7（月 約 −125 円＋転送 −18GiB） | 同上（cron の変更はユーザーの OK を得てから） |
| A'' | 500 頭の sire の確かめ（`backfill.py:371`、`storage.load("horse_result")` 500 回/日）も、手元の horse_result（`data/local`）を先に読む | `src/scraper/backfill.py:371-375` | Class B −500 回/日・転送 −約 8MB/日（小） | A と一緒に |
| B | 金曜の週次の二重の全件読みを 1 回にし、手元の flat に寄せる（3 章 #2 の表の直し方の案） | `src/utils/horse_name_index.py:394`（年ごとの `list_keys` `:428`＋1 件ずつ `load` `:410`）、呼び出し `src/scraper/auto_scrape.py:1016,1021`（horse-name-index）と `:1364,1366`（weekly-update の中） | 転送 −約 0.8〜1.0GiB/週（月 約 −4GiB）・Class B −約 15 万回/月（約 −10 円）。片方をやめるだけでも半分 | 馬名インデックス・週次更新の担当セッション（取得の担当と同じでよい） |
| C | Cloud Run の API の一覧（1 日 約 2,300〜3,500 回）を減らす | 3 章 #2 の「Cloud Run の API に残っている一覧」の表（(a)〜(c)） | 実績 月 約 56〜85 円 → 数十円。最悪の 1,250 円の芽を摘む | API の担当セッション（ユーザーが戻ってから） |
| D | 手で全件を読む一度きりの処理（書き出し・dry-run）は、GCP の中（Cloud Run Jobs）か手元の flat で行う。ホストで流すときは事前に「件数 × 1 件の大きさ」を見積もる | 運用の決まり（CLAUDE.md の決まり 2・3 の補い） | 11/13 は 1.7GiB（無料枠を超えた月なら 約 32 円）。11 月の残りの無料枠を守る | 統括役から各セッションへの周知 |

- 見積もりの直し（2.6 の 12 月の見込みへの反映）: 転送の行は**中央 0 円**（1 日 1.3GiB）、最悪は全件読みの周期処理が新しく入った場合。GCS の読み出し（G3）は A を入れると 約 450 → 約 305 円。**12 月の合計は 約 1,400〜1,480 円（A なし）／約 1,250〜1,340 円（A あり）**。上限 2,000 円との差 −520〜−750 円。
- 監査ログを無効にした後は、同じ問いを Monitoring の時間ごとの ReadObject（回数・量）と課金エクスポートの `Download APAC` で見る。朝 06〜08 JST の 1 時間 4 万回・360MiB 前後が消えれば A が効いた印。

## 4. 今日（2026-11-13）入った・入る予定の変更の見込み

| 変更 | 状態 | 月額の増減（見込み） | 最悪 | 判定 | 根拠・条件 |
|---|---|---|---|---|---|
| 30097498・f5f969ac（VPS の取得の土台、取得しないモード）、1da83d99（再開しても GCS のタスク表に戻らない） | 入った（担当は空）。**VPS の fetch-agent は 11/13 15:00 JST に統括役が停止**。1da83d99 で、再開しても GCS のタスク表の一覧には戻らない | **−210（停止で実現。一覧 8,640 回/日 → 0）** | 再開して file モードでないなら ±0（−210 が戻る） | 減る | 実測の一覧は 1 日 8,640 回（見積もりの 3 倍、#5）。再開は `KEIBA_FETCH_TASK_STORE=file` にしてから。馬体重の判定の読み直し 月 約 6 円・予測の読み 約 11 円は転送料（100GiB 超えの月は全額） |
| VPS の取得の残り（snoopy-wondering-wilkes、段 1〜6） | 段 1〜5 は入った・**段 6（担当を渡す・cron の登録）は未**。VPS の取得役は 11/13 15:00 から停止中 | **今は 0**（担当が空で、どの段も GCS に触れない）。渡した後 約 +48（下の 5 行の合計） | +80〜+160（後処理の読み直しを多めに見た値） | **条件つき** | 段ごとにホストの GCS 読み出し量を実測して台帳を直す。`pull-index` は目録だけ読む（全件一覧しない）ことをテストで確かめる。cushion の毎時の 1.5MB は、目録で変わったときだけ取るなら 1 日数回 |
| 　段 1: 04cb7383（VPS が取ったレース一覧・発走時刻表を `ops/mirror_vps/` 経由でホストへ） | 入った（未切り替え） | 0（渡した後 約 +1.5） | 約 1.5 | 上限内 | ホストの目録の読み 1 件・約 5KB × 288 回/日 × 30 ≒ 43MB。VPS は差分を手元の MD5 で判定し、一覧しない |
| 　段 2・3: 4211b34a（T-15 の保存を確かめてから予測、オッズの鮮度） | 入った（未切り替え） | 0（渡した後 約 +19） | 約 19 | 上限内 | 待ちの問い合わせ 約 1.1 万件（Class B）≒ 0.7 円＋オッズの読み直し 約 1GB ≒ 18 円（開催日は月 8〜10 日なので実際は少ない） |
| 　段 4 前半: 2e6c8066（取得の後処理をホストに残す入口） | 入った（未切り替え） | 0（渡した後 約 +20） | 約 20 | 上限内 | 予測の成績 47MB × 13 回 ≒ 0.6GB → 約 11 円、前夜の計算 0.3GB → 約 6 円 |
| 　段 4 後半: 14be06ee（週次の後処理をホストに残す、目印方式） | 入った（未切り替え。cron の登録は段 6） | 0（渡した後 約 +2） | 約 2 | 上限内 | 目印を名前で 1 件読む（一覧しない）。後処理の読み 月 約 100MB |
| 　段 5: d086cc39（新馬の血統の上限と順序） | 入った（VPS の取得役の停止中は動かない） | 0（動けば 約 +5） | 約 5 | 上限内 | VPS → GCS の名前指定の読み 1 回 600 頭まで。月 約 5.4 万回 ≒ 3 円＋転送 90MB ≒ 2 円 |
| 毎朝の書き出しへの 7 カテゴリ（polymorphic-frolicking-flame の ③） | 入った | +1 未満（最悪 11） | 11 | 上限内 | 一覧 0 回。ホストからの名前指定の読み 1 日 最大 1,500 件 × 10KB |
| 3fcb2a2b（毎朝の書き出しの全件一覧を 0 に。本体 6 カテゴリ・parse_drift_docs） | 入った | **約 −46**（月 約 50 → 見込み 約 4） | 約 10 | 減る | 一覧 1 日 約 30 回・約 90MB → 0 回。書き出しの読み 最悪 1 日 約 10MB・Class B 約 1,300 回、parse_drift_docs 1 日 480 件・約 3MB |
| c03b86f7（毎晩の PG 同期の全件一覧を 0 に） | 入った | 約 −1（月 約 0.3〜3 → 約 0.5） | 約 0.5 | 減る | 一覧 毎晩 2 回 → 0 回。flat に無いものだけ名前で確かめる（最悪 1 日 約 300 回の Class B） |
| 957858ad（調教の被覆の判定をホストの手元の複製だけで） | 入った（11/13 15:29 JST） | **約 −150 → 0**（100GiB を超えた月の額） | 0 | 減る | 夜間の補い（月〜木 22:00）・drift_monitor の判定で毎回 約 1 万頭・0.5〜0.6GB を GCS から読んでいた → 0。週 4 回 × 0.5GB ≒ 月 約 8GB |
| 7c4a8adb（cron の修正: 共通ログインの時刻、sporthorse の外し方、serving-state のタグ、job_map を crontab の実態に） | 入った | 0（+0.6 未満） | 約 0.6 | 上限内 | serving-state は開催日の出馬表 約 36 件で月 0.1 円未満、profile は最悪 月 約 0.5 円。GCS の読みは増えない |
| 7e1cc81f（費用のため消した Scheduler 13 件を、昇格で作り直さない。job_map に retired） | 入った（develop、11/13 20:47 JST） | 0（作り直されたときの +1,700〜2,700 と固定費 +195 を防ぐ） | 0 | 減る（リスクの除去） | #6。昇格はこのコミットを含む版でだけ |
| 280a6ab0（verify_past_values の全件一覧を 0 に） | 入った（develop、11/13 20:50 JST） | **約 −200**（コミットの見積もり。一覧 1 日 約 480 回・転送 1 日 0.3〜0.4GiB） | 約 11 | 減る | 対象はローカルの flat、基準年の値は 30 日使い回し、転送課金の場所では 1 回 500 件まで。cron を 1 日 1 回にするのはユーザーの作業（handoff の ④） |
| 4d424c7a（開催日の後処理の馬場速度の race_result_flat の増分で一覧しない） | 入った（develop、11/13 20:52 JST） | 約 −1（1 回 約 3MB・Class A 約 4 回 × 月 約 10 回） | 0.1 未満 | 減る | その日のレースを `local_universe` から作って名前で読む |
| 3a86b5ff → 42f48f22（天気: 1 日 1 回・前回からの範囲に直した後、定期取得を廃止） | 入った（cron は 42f48f22 で外した） | 0（直す前の作りなら 約 4,400、1 日 1 回の作りでも 約 1.5〜3） | 0 | — | GCP の jma-weather は enabled=false。過去分の補いは手で 1 回だけ（約 345MiB ≒ 6 円） |
| run_axes の毎日更新（同 ②） | 実装中 | +1 未満 | 5 | 上限内 | 新しい GET 0、horse_lap の保存 1 日 約 10 件（Class A）、読みは名前指定 |
| 天気の取得の再開（同 ①） | **廃止（2026-11-13）** | 0 | 0 | — | #0。毎時 1,310 件・345MiB を読む作りだったため、ユーザーの判断で定期取得をやめた |
| 077bc683 と master への hotfix | 作業中 | +5（Artifact Registry） | +10 | 上限内 | 読み出しは減る |
| 定期ジョブの監視（d6c72260。計画 cozy-booping-marshmallow、keiba-vpn-b0） | **実装済み（d6c72260）・配備は一部**: バケットの条件付き権限（vps-sched-status）とホストの cron（vps-alive 15 分おき）は 11/13 18:4x に実施済み。**keiba-monitor の Cloud Run への配備と、プロジェクトの run.viewer・cloudscheduler.viewer の付与は未実施** | 約 +3 | 約 +35（起動し直し 最悪 約 31 円、無料枠の中なら 0） | 上限内 | 審査の条件 1（Cloud Run を起こさない）を採った: VPS の keiba-sched-status.timer（15 分おき）が GCS に直接書く（Class A 月 2,880 回 ≒ 2.3 円）。ポータルの読み 約 0.5 円。新しい Scheduler・15 分未満の ping・ホストからの GCS の読みは無し |
| VPS の cache-warm の一覧を合図方式に（計画 `/tmp/claude-1000/work/plan_cache_warm_20261113.md`） | **実施済み（VPS に配備 11/13 16:30 JST）** | 約 −220（今 約 265 → 約 20〜45） | 約 45 | **上限内（条件つき）** | 下の「審査: cache-warm の合図方式」。11/14 の監査ログ 24 時間分で確定 |
| smoke-auto-* の Scheduler 3 件の削除 | 済み | −45 | −45 | 減る | `docs/operations/deleted_scheduler_smoke_auto_20261113.md` |
| **合計（21:30 に直した）** | | **約 −860**（verify_past_values −200・fetch-agent の停止 −210・cache-warm −220・調教の判定 −150・書き出し −46・smoke-auto −45・ほか +10。VPS 取得の段 1〜5 は今 0、渡すと +48） | 約 +200 | | 減った分は 2.4 の G1・G4 に入れた（verify の分は 21:30 に反映）。残る未知は朝の大量読みと金曜の週次の全件読み（#2）。#6 は 7e1cc81f で修正、昇格の版に注意 |

### 審査: cache-warm の合図方式（2026-11-13）
判定: **上限内（条件つき）**。月 約 265 円 → 最悪 約 45 円（−220 円）。
- 毎分の合図のメタデータ読み: Class B 月 4.3 万回 ≒ 2.6 円。応答は約 1KB なので転送は月 約 45MB ≒ 1 円。問題なし。
- 合図を書くのをホストの push（5 分おき）と配備のときだけにすれば、上書きは最悪 288 回/日 → 月 8,640 回 ≒ 7 円。保存ごとに書く案（最悪 月 13 万回・約 100 円）は採らない。
- 合図が無いときの 10 分おきの一覧: 月 4,320 回 × 応答 約 300KB（今の実測は 1 回 約 500 件 × 約 0.6KB）＝ 約 1.3GB → 転送 約 25 円＋Class A 約 3 円。最悪でも約 30 円。
- 条件:
  1. 一覧に `fields="items(name,generation),nextPageToken"` を付けて応答を小さくする（今の `cache_warm.py:254-261` は指定が無く、1 件ごとに全メタデータが返る）。転送が約 5 分の 1 になる。
  2. 保存ごとの目印（`mark_from_save`。ホストの取得の保存など）は合図を変えない。push で 1 件も上げない回は合図も変わらないので、それらの目印が温められないまま残りうる。安全網として、合図が変わらなくても 30〜60 分に 1 回は一覧する（月 720〜1,440 回・数円）。
  3. 配備の 1〜2 日後に、監査ログ（11/14 14:44 JST までの分）か Monitoring の ListObjects の回数で、1 日の一覧が 300 回以下になったことを確かめる。
- 配備後の途中の確認（監査ログ、keiba-pull@）: 配備前は 10 分ごとに一覧 9 回（1 日 約 1,300 回）。配備後の 16:32〜16:45 JST の 13 分は**一覧 2 回**（1 時間 約 9 回 → 1 日 約 220 回の割合）、合図のメタデータ読み 12 回（毎分 1 回）、release-pull の取得 12 回。目安（1 日 300 回以下）に入っている。11/14 に 24 時間分で確定する。

### 審査: 定期ジョブの監視（2026-11-13、計画 `~/.claude-jovyan/plans/cozy-booping-marshmallow.md`）
判定: **条件つきで上限内**。条件を満たさないと、CLAUDE.md の「15 分未満の間隔で API を叩く ping を足さない」に反するか、起動し直しの費用が乗る。
- **結果（21:00 の追記）**: d6c72260 は下の条件 1（ハートビートは Cloud Run を起こさない）を採った。VPS の keiba-sched-status.timer が 15 分おきに `ops/heartbeat/vps/status.json` を GCS に直接書き、VPS 全体の停止はホストの cron（install_vps_alive_cron.sh、15 分おき）が公開の /version.json で確かめる。見積もりは ふだん 約 3 円・最悪 約 35 円/月。バケットの条件付き権限とホストの cron は 11/13 18:4x に実施済み、keiba-monitor の配備とプロジェクトの viewer 権限は未実施。
- **前提の誤り: 5 分おきのハートビートは今は動いていない**。VPS の `keiba-heartbeat.timer` は disabled・inactive、`/etc/keiba-ui/heartbeat.env` は 0 バイト（11/13 に読むだけで確認）。
  「既存に相乗りするので呼び出し回数は今と同じ」ではなく、**5 分おきの ping を新しく足す**ことになる。
- 受け手は `keiba-api`（`src/api/cloudrun_entry.py:202-203`。2GiB・起動 約 50 秒・起動時 CPU ブースト）。
  - 5 分おき: インスタンスが落ちきらず起動し直しは少ないが、CLAUDE.md の決まりに反する。要求 8,640 回/月 × 約 1 秒 ＋ latest.json の読み書き（Class A・B 各 8,640 回）で 約 40 円。
  - 15 分おき: 決まりは守れるが、アイドルで落ちる間隔とほぼ同じなので、**最悪は毎回起動し直し**（96 回/日 × 100 vCPU 秒 ＝ 月 28.8 万 vCPU 秒、無料枠なしで 約 1,000 円）。決まり 1 の最悪の回数そのもの。
- 画面を開く・点検の見積もり（月 約 21 円）は、`keiba-monitor` の**起動し直しの分が入っていない**。最悪 96 回/日がすべて起動し直し（数秒 × ブースト 2 vCPU）なら 約 +100 円（無料枠の中なら 0）。GCS の読みは Cloud Run の中なので転送 0 で、回数の見積もりは妥当。
- 毎朝の g-0730-shadow に 1 手順: GCS の読み 1 回・数秒で、無料枠の中（最悪でも 月 1 円）。問題なし。
- 条件（どれか 1 つ。推奨は 1）:
  1. **ハートビートは Cloud Run を起こさない**: VPS が 15 分おきに `ops/heartbeat/vps.json` を GCS に直接書く（GCP への書き込みは転送 0、Class A 月 2,880 回 ≒ 2 円）。評価は画面を開いたとき（5 分キャッシュ）と毎朝のジョブ。赤の即時の Slack は VPS が自分のタイマーを見て送る（GCP の費用 0）。合計 月 約 5 円。
  2. 受け手を `keiba-monitor`（512Mi・起動が速い）に移し、15 分おきにする（10/28 の試算の案 I）。最悪 月 約 100〜150 円（無料枠の中なら 0）。
  3. keiba-api のまま 5 分おき: 決まりに反するので、ユーザーの明示の承認が要る。
- 監視の判定の「15 分（5 分 × 3）来なければ途絶」は、15 分おきにするなら 45 分にする。

## 5. 上限内に戻すための案（統括役への提案）

11/13 のユーザーの承認と状況（記録: `docs/operations/deleted_scheduler_keiba_g_20261113.md`、`docs/operations/cost_changes_20261113.md`）:
- 0（天気）: 定期取得を廃止（14:08 に停止、ホストの cron は 42f48f22 で外した）。
- **C（Scheduler 13 件の削除）: 実施済み**（11/13 14:4x JST、統括役がサービスアカウントで。各件 PAUSED を確かめてから削除。残りは keiba-g-0730-shadow ENABLED の 1 件）。記録 `docs/operations/deleted_scheduler_keiba_g_20261113.md`。月 約 −195 円。
- **D（Artifact Registry の保持 2 日 → 1 日）: 実施済み**（11/13 14:4x JST、統括役。keep-live・keep-recent 3 は不変を確認）。配備スクリプト（`scripts/gcp/deploy_cloud_run_api.sh:103` など）も 1 日にそろえた（5d81383e）。月 約 −80 円。
- **GCS の DATA_READ 監査ログ（24 時間）: 有効にした（2026-11-13 14:44 JST / 05:44 UTC、ユーザーのアカウントで。auditConfigs = storage.googleapis.com DATA_READ）**。
  サービスアカウントには `getIamPolicy` の権限が無く 403 だったため、ユーザーが実施。**戻すのは 11/14 14:44 JST 以降**（統括役がユーザーに案内）。呼び手ごとの集計は費用の監査役が出して、3 章 #1・#2 に書く（11/13 20:50 までの分は 3 章 #2 の「監査ログの集計」に書いた）。
  **戻すのは 11/14 14:44 JST に 24 時間になってから、ユーザーのアカウントで**（コマンドは前の監査役の scratchpad の `cmds/3_revert_gcs_data_read_audit.sh`。auditConfigs を消して変更前に戻す）。その前に、監査役が朝 06〜11 JST の分を集計する。
- 確かめる予定: 11/14〜15 の課金エクスポートで、`Cloud Scheduler Jobs` の数量が 1〜4 件に下がったこと、`Artifact Registry Storage` が 1 日 約 0.3GiB-月以下に下がったこと（費用の監査役）。


| 案 | 減る額（月） | 手間・影響 |
|---|---|---|
| 0. 天気の取得を `daily` だけ・1 日 1〜2 回にする（#0） | 約 −4,400 | 過去分の補いは 1 回だけ別に回す |
| A. ホスト・VPS から GCS を読む処理を特定して、1 日の読み出しを 3GiB 未満にする（#1・#2） | 約 −1,800（転送を無料の 100GiB 内に） | 呼び手ごとに直す。周期処理は `egress_billed()` で止めるか、手元の記録で差分を判定する（決まり 6） |
| B. 大きな一覧を差分の目録か手元の記録に置き換える（#2） | 約 −200〜−290（Class A）＋転送の一部 | 毎朝の書き出しは 3fcb2a2b、PG 同期は c03b86f7、verify_past_values は 280a6ab0、馬場速度は 4d424c7a で一覧 0 にした（済み）。残りは週次の horse-name-index の `list_keys` と、Cloud Run の API の一覧（#2、今 月 約 85 円・最悪 約 1,250 円） |
| F. 週次の全件読み（weekly-update と horse-name-index の race_shutuba の二重読み、成長曲線の horse_result 全頭）を 1 回・手元の flat に寄せる（#2） | 約 −130〜−150（無料枠を超えた月）＋Class B 約 −40 | ホストの手元の複製・flat を使う。二重の片方をやめるだけでも半分近く減る |
| C. 停止中の keiba-g-* 13 件の Scheduler を、設定を docs に残して削除する | 約 −195 | cost_guard の止める対象が無くなる（再開するときは作り直す） |
| D. Artifact Registry の保持を 2 日 → 1 日、使わない Jobs の版のタグを外す | 約 −80〜−115 | 戻すときの版が減る |
| E. 0・A〜D をすべて入れた後の見込み | 約 1,300〜1,500（C・D だけなら 約 3,300〜3,400。A・B で転送と一覧を減らさないと上限内に戻らない）。**21:30 の時点の見込みは 約 1,400〜1,640（2.6）** | 上限内（ただし最悪の回数では 2,000 を超えうるので、API の起動回数と Jobs の再開に注意） |

## 6. 更新の手順

- **いつ**: (1) 費用に効く変更を commit する前（審査として行を足す）、(2) 毎週金曜（課金エクスポートの直近 7 日で「実績」列を直す）、(3) cost_guard が「上限に近い」を通知したとき、(4) 月初の 3 日目（前の月の確定額を書く）。
- **誰が**: 費用の監査役（統括役が立ち上げるエージェント）。監査役がいないときは、変更を入れたセッションが自分の行を足し、統括役が確かめる。
- **何を見るか**:
  - 課金エクスポート（BigQuery `horse-racing-466415.billing_export.gcp_billing_export_v1_*`）: 日付で絞って、日 × サービス × SKU の金額と数量（`usage.amount_in_pricing_units`）。GCS の転送は `price.tier_start_amount` で無料の段か課金の段かを見る。請求月は `invoice.month`（太平洋時間）。
  - Cloud Monitoring: `storage.googleapis.com/api/request_count`（method 別の回数）と `storage.googleapis.com/network/sent_bytes_count`（method 別の送信量）の直近 7 日。課金エクスポートより半日早く分かる。
  - `logs/cost_guard.log`（cost_guard の判定）、Cloud Run の実行履歴（Jobs の回数）、Scheduler の一覧（件数と状態）、Artifact Registry の容量。
- **どう直すか**: 2 章の該当の行の「実績」「見込み」を直し、2.5 の合計と上限との差を更新する。新しい項目は「動く場所・最悪の頻度・1 回の量・根拠（path:line かコミット）」をそろえて足す。最終更新の日付を直す。
