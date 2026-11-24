# 計画: レースごとの「予測・推奨が画面に出た」Slack 通知（stg・本番）

担当: slack-notify。統括役: claude-orchestrator-handover。状態: **実装済み・既定オフ**（統括役の承認 2026-11-19: §8 は推奨どおり。1 レース 1 回・今週末は入れない・本番は app.megu-keiba.com）。

ユーザーの依頼（原文）: 「stg/prodそれぞれで、レース開催日にAI予測結果、買い目推奨がUI上に反映されたタイミングで、そのレースIDが指定された対象リンク付きでslackに通知を飛ばすようにしてほしいです。」

## 1. 結論

- **案 A（ホストの手元の記録で判定して送る）を推奨する。** GCS・Cloud Run の操作は 0 回で、費用は 0 円。
- 足すもの:
  - 新しいスクリプト 1 本と、root の cron 1 行。
  - push の中で、上げた時刻を手元の記録に 1 行足す処理。今ある `_note_stage2_push` と同じ作りで、GCS の操作は 0 回。
- 送るのは、`KEIBA_SLACK_NOTIFY=1` と専用の目印の両方があるときだけ。既定はオフ。
- 今週末（11/21〜23）は送らない。送り始めるのは 11/28（土）から。今週末の扱いは統括役が選ぶ（§8）。
- **本番のリンクの基点は `https://app.megu-keiba.com`。** 依頼文の `https://megu-keiba.com` は API の例にしか出てこない。根拠は `scripts/cron/env_parity_check.sh:12` と `config/billing.json:7-10`。stg は `https://stg.megu-keiba.com`。

## 2. 今の流れ（stg と本番は同じミラーを読む）

| 段 | 何が | 保存先 | 根拠 |
|---|---|---|---|
| 1 | 本体の推論（weight-runner、08:00〜）。T-65〜T-30 に馬体重がそろうか、T-30 を過ぎたら推論する | `data/local/inference/predictions_v6/<日>/<rid>.parquet`、`events/<日>.jsonl`（`status`・`v6_status`・`sec77`） | `src/pipeline/inference/weight_triggered_predict.py:339-345, 385-387` |
| 2 | 推奨（target）の元になる §77 の想定オッズ | **GCS へ直接**（`final_odds_prediction`。ミラーは通らない） | 同 `:410-443` |
| 3 | 未勝利・新馬だけ、影のランナーの第 1 段（v6c）と第 2 段の判定 | `predictions_v6c/…`・`bets_stage2/<日>/<rid>.json`・`launch_watch/<日>.jsonl` | `src/pipeline/inference/candidate_runner.py:513-514, 597-614` |
| 4 | push（root の cron、5 分おき） | `gs://…/ops/mirror/…`。手元の記録 `gcs_mirror_push_manifest.json` の `recent`（2 日で刈る）。第 2 段だけ `launch_watch/push_<日>.jsonl` にも | `src/ops/gcs_mirror.py:159, 309-314, 386-395, 410-431` |
| 5 | Cloud Run（stg の rstg タグ・本番）が目録を読み直す | リクエストがきっかけで、前回から 60 秒以上たっていれば読み直す | `src/ops/gcs_mirror.py:783-827` |
| 6 | 応答の使い回しと画面のキャッシュ | prerender 120 秒、画面のキャッシュ 45 秒 | `scripts/gcp/deploy_cloud_run_api.sh:74`、`frontend/lib/httpCache.ts:8` |

- stg と本番は、同じバケットの同じ `ops/mirror` を読む（`scripts/ci/deploy_gcp.sh:35-43`）。**データの事象は 1 つで、送り先が 2 つ**になる。
- **動くコードは環境で違う**（stg は develop、本番は master）。環境で違う点は次のとおり。
  - 第 2 段を読み始める日 `KEIBA_STAGE2_START`（stg は 20261114、本番は 20261121。`deploy_gcp.sh:130-139`、`src/api/stage2_live.py:39-44`）
  - 未昇格の表示の有無
- 画面に出す中身の決まり（`src/api/stage2_live.py`、`src/api/local_predictions.py:70-75, 228-232`、`src/api/v1/delegates.py:82-93, 276-313`）:
  - 予測: 未勝利・新馬で v6c があり、止めの目印が無ければ v6c。それ以外は v6e（predictions_v6）。
  - 推奨: 第 2 段の判定があり、開始日を過ぎていれば第 2 段を出す。`_dev_only` が有効なら、開発者だけが第 2 段で、会員は target のまま。それ以外は target で、§77 が要る。
  - §77 は、リクエストのたびに `HybridStorage()` を作って読む（delegates.py:297）。このため、GCS に保存した直後から読まれる。この作りを共有の storage に変えると、最大 3600 秒「準備中」が残る（storage.py:337, 1007）。この前提が変わったら判定を見直す。
- レースの画面は `/race/<race_id>`（`frontend/app/race/[id]/page.tsx`）。推奨馬券はそのページのタブにある。

## 3. 「画面に出た」をどこで判定するか

| 案 | 判定 | 確からしさ | 費用（最悪） | 評価 |
|---|---|---|---|---|
| **A** ホストの手元 | 予測ファイルを上げた時刻と、推奨の元（§77 または第 2 段の判定）の両方がそろってから 4 分後 | 推定。push から画面までは 60＋120＋45 秒で最悪約 4 分 | 0 円 | **推奨** |
| B Cloud Run の中 | 目録を読み直して、新しいレースを見つけた時点 | 誰も開かないと読み直しが起きない（最小 0・request モード） | 複数のインスタンスで重ねて送る。防ぐには GCS に送信済みを書く（月 約 0.2 円）。stg と本番の Webhook の秘密の追加と、配備が要る | 不採用 |
| C 外から API を叩く | レースごとに 1 回、予測と推奨の API を呼ぶ | 実際の応答で確かめられる | 72 回/日。起動し直しが増える。推奨の API は会員以上で、開発者の認証情報を外で使うことになる | 不採用（秘密の危険） |
| 既存の 15 分おきの alerts（scrape_status_notify）に相乗り | — | 最大 15 分遅れる | 0 円 | 不採用（遅い） |

案 A を推奨する理由:
- 判定に要る事象（推論・§77・判定・push の時刻・発走時刻）は、すべてホストの手元にある。
- 送り先は、ホストの `.env` の `SLACK_WEBHOOK_URL_STG` と `_PROD` で分けられる。3 つの値がすべて別であることを、値を出さずに確かめた。
- master への昇格にも、Cloud Run の配備にも依らない。

案 A の弱点と扱い:
- 「出た」は推定になる。文面は「画面に出た見込み」と書き、上げてから 4 分以上たってから送る。
- push の目録（`_index`）の書き込みに失敗すると、Cloud Run が気付くのは 20〜30 分後になる（gcs_mirror.py:160-161, 386-395）。
  - そのときは送らずに、遅れの警告に回す。
  - 見分け方: push のログ（`logs/gcs_mirror_push.log`）の「更新の目録を書けませんでした」を、手元で見る。
- 実際に開かれたかどうかは、これまでどおり VPS の nginx と Cloud Run のリクエストのログで見る（weekend_launch §5）。周期の ping は足さない。

## 4. 作り

### 4a. push の時刻を手元に残す（`src/ops/gcs_mirror.py`、小さな変更）

- 今ある `_note_stage2_push` を一般にして、`predictions_v6`・`predictions_v6c`・`bets_stage2` を上げた時刻を、レースごとに `launch_watch/push_<日>.jsonl` に足す。
  - 既存の行の形 `{"at","event":"pushed","race_ids"}` に `tree` を足す。`launch_watch_report` は `tree` が無い行を第 2 段とみなすので、互換を保てる。
- 最善努力で書く。例外は握りつぶし、push を止めない。GCS の操作は 0 回。
- 理由: manifest の `recent` は 2 日で刈られるので、後から当て直すときに上げた時刻が残らない。

### 4b. 新しいスクリプト `src/scripts/ops/race_ready_notify.py`

読むのは手元のファイルだけで、`HybridStorage`・GCS・API は使わない（テストで呼び出しが 0 回であることを確かめる）。

1. 今日（`ZoneInfo("Asia/Tokyo")` で求める）の `data/page_reference/race_lists/<日>.json` を見る。無ければ、重い import の前に終える。
2. レースごとに、次を集める。
   - 予測・判定を上げた時刻: `launch_watch/push_<日>.jsonl`（4a）。
   - 推奨の元: `events/<日>.jsonl` のうち、そのレースの最後の `status=ok` の行。`v6_status=ok` と `sec77=ok` で判定する。
   - 発走時刻: 推論に使った出馬表の写し `snapshots/<日>/<rid>.json` を先に使う。無ければ `entries_state/<rid>.json` の `start_time`。
     - entries_state は特別登録の時点の値が残ることがある。実装のときに、2 つの出所が食い違う日を数える。
   - 開催場・R・レース名・対象（未勝利・新馬）: race_lists と手元の出馬表の写しから取る。
3. 環境ごと（stg・本番）に、画面の中身を `stage2_live` の関数（`disabled`・`dev_only`・`v6c_disabled`・開始日）で決める。目印の解釈は自前で書き直さない。
   - 予測の段: v6c が出る環境・レースなら v6c の push、それ以外は v6 の push と `v6_status=ok`。
   - 推奨の段:
     - 第 2 段が会員に出る環境・レースなら、第 2 段の判定の push。
     - それ以外は §77（`sec77=ok`）。
     - `_dev_only` のときは、会員の推奨は target として判定する。第 2 段の状態は「開発者だけ」の 1 行で添える。
   - 本番は、master に昇格していない表示を出さない。設定 `features_prod`（例: `stage2: true`）で、本番に入っている機能だけを見る。
4. 送る条件（1 レース・1 環境に 1 回）:
   - 予測の段と推奨の段の両方がそろってから 4 分がたった。
   - 発走前である。
   - そろってから 15 分以内である。それより古いものは送らず、「見送り」として記録する。目印を後から置いたときに、溜まった分が一度に出ないようにするため。
5. 遅れの警告（1 レース 1 回）:
   - 発走 10 分前を過ぎても 4 を満たさないときに送る。発走後は送らない。
   - 中止・取消のレース（race_lists や出馬表の写しで分かるもの）は除く。
   - 同じ原因（例: §77 が `no_model`・`disabled`、推論なし、目録の書き込みの失敗）の警告は、1 回の実行で 1 通にまとめる。
   - 送るのは stg のチャンネルだけ。本番のチャンネルには「出た」の通知だけを送る。
6. 目印が無い間（dry-run）は、条件を満たしたレースを「見送り（dry-run）」として記録し、後からは送らない。送るはずだった内容はログに残す。Webhook の値は出さない。

### 4c. 送り先の設定 `config/race_ready_notify.json`（秘密は書かない）

```json
{
  "lead_sec": 240,
  "max_age_min": 15,
  "late_warn_min": 10,
  "targets": {
    "stg":  {"base_url": "https://stg.megu-keiba.com", "webhook_env": "SLACK_WEBHOOK_URL_STG",  "keiba_env": "stg"},
    "prod": {"base_url": "https://app.megu-keiba.com", "webhook_env": "SLACK_WEBHOOK_URL_PROD", "keiba_env": "prod",
             "features": {"stage2": true}}
  }
}
```

- 第 2 段の開始日は設定に持たない。`stage2_live` の既定（本番 20261121）に従い、stg だけ `stage2_start` で上書きできるようにする。stg の上書きは週末の後に外す予定（deploy_gcp.sh:130-139）。
- 宛先の取り違えを防ぐため、次のようにする。
  - Webhook は `os.environ[webhook_env]` を直接読む。空なら、その環境には送らない。汎用の `SLACK_WEBHOOK_URL` には落とさない。
  - `send_payload` に渡す辞書は、`{"KEIBA_ENV": "<stg|prod>", "SLACK_WEBHOOK_URL_<STG|PROD>": url}` だけにする。`KEIBA_NOTIFY_ENV_LABEL` などを引き継がないので、見出しの【STG】【PROD】が環境ごとに正しく付く。
  - テストでは、「_PROD が空なら本番に送らない」「stg の通知が stg の URL にだけ行く」「リンクの文字列（`https://app.megu-keiba.com/race/<rid>`）」を固定する。
- `.env` は本体が `_load_env`（`dotenv_values`。既にある環境変数・`.env.stg` などで上書きしない）で読む（`cron_run.sh` は `.env` を読まないため。shell で source すると値の `$`・空白を解釈するので、2026-11-24 に source をやめた）。
- Webhook の値は、ログ・記録・テスト・dry-run の出力に出さない（`redact` を通す）。

### 4d. 通知の中身（見本）

```
【PROD】予測・推奨が画面に出ました（見込み 10:41）
東京 5R 2歳未勝利 ・ 発走 11:50 ・ race_id 202605050505
・AI 予測: 会員に出た（v6e）
・推奨馬券: 会員に出た（いまの推奨）
・第 2 段（試行中）: 開発者だけ・単勝 1 点（判定 10:38）   ← 未勝利・新馬で、送る時点で上がっていれば
<https://app.megu-keiba.com/race/202605050505|レースを開く>
```

- 遅れの警告の見本: 【STG】【注意】「発走 10 分前を過ぎても、予測・推奨が出ていません（n レース）」。各レースについて、止まっている段（推論なし／§77 なし／判定待ち／push 待ち／目録の書き込みの失敗）と、リンクを付ける。

### 4e. 重ねて送らない記録

- 置き場は `data/local/meta/slack_sent/race_ready_<日>.json`。鍵は `(環境, race_id, 種類=ready|late)` で、値は `sent`・`skipped_dryrun`・`skipped_old`・`failed:n`。
- 送れたら、すぐ記録する（一時ファイル → `os.replace`）。
- 再試行は実行の単位で数える。`send_payload(retries=1)` で送り、送れなければ次の分に送り直す。1 件につき最大 3 回まで。
  - Webhook の設定が無い（`webhook_not_configured`）ときは、回数に数えない。
- 重ねて送るのは、最大 1 通まで許す。次の 2 つは防げないため。
  - 時間切れ: Slack は受け取っていたが、こちらには応答が届かなかった。
  - 送れた後、記録を書く前に落ちた。
- 二重起動は `flock -n` で防ぐ（重なったら、その回は何もしない）。
- 1 回の実行で送るのは最大 20 通。1 通ごとに 1.2 秒あける（Webhook は 1 秒に約 1 通が上限）。
- 今日の分だけを送る。日をまたいだ記録は見ない。

### 4f. 動かし方

- root の cron に 1 行足す（push・ランナーと同じ持ち主）。
  - **crontab の時刻は UTC で書く**: `* 22-23,0-8 * * *`（JST の 07:00〜17:59、毎分）。
  - `nice -n 19` で動かす。開催日でなければ、race_lists を見た時点で終える。開催日の 1 回は 0.5〜1 秒の見込み。
  - 毎分にする理由: 5 分おきだと、送るのが最大 9 分遅れる。読むのは手元のファイルだけなので、毎分でも費用は 0 円。
- 送るのは「`KEIBA_SLACK_NOTIFY=1` かつ目印 `data/local/meta/race_ready_notify.on` がある」ときだけ。
  - `KEIBA_SLACK_NOTIFY` は、通知全体を止めるスイッチ（`src/ops/alerting.py:4, 34`）。これを外せば、この通知も止まる。
  - 目印は、コードが共有の作業ツリーに入った時点で送り始めないためにある。`.env` は既に `KEIBA_SLACK_NOTIFY=1` なので、目印が無いと止められない。
- 入れ方・外し方は `scripts/cron/install_race_ready_notify_cron.sh [--remove]`。`install_candidate_runner_cron.sh` と同じ作りにする。時刻の欄はテストで確かめる。

## 5. 費用（最悪の回数）

| 項目 | 回数 | 月額 |
|---|---|---|
| GCS の読み出し・一覧・書き込み | 0 回（手元のファイルだけ。4a も手元に書くだけ） | 0 円 |
| Cloud Run のリクエスト | 0 回 | 0 円 |
| Slack の送信 | 「出た」が 36R × 2 環境 × 開催 9 日 ＝ 648 通。警告は、まとめるので 1 日数通 | 0 円（Incoming Webhook は無料） |
| ホストの計算 | 毎分 1 回・0.5〜1 秒（nice 19）。開催日でなければほぼ 0 | 0 円 |
| 合計 | | **0 円** |

## 6. 確かめ方

1. 単体テスト（`tests/ops/test_race_ready_notify.py`、`tests/ops/test_gcs_mirror*.py` に 4a の分）で、次を確かめる。
   - 判定（v6e・v6c・§77・第 2 段・`_dev_only`・開始日）
   - 発走後と 15 分超は送らない
   - 重ねて送らない
   - 警告のまとめ
   - stg・本番の送り先・見出し・リンク
   - 「_PROD が空なら送らない」
   - Webhook の値が出ない
   - storage・GCS の呼び出しが 0 回
   - cron の時刻の欄
2. 過去の開催日（11/15）で当て直す: `--day 20261115 --dry-run --replay`。
   - 11/15 には、予測と判定の push の記録が残っていない（第 2 段の判定は当て直しの分だけ）。そこで、ファイルの更新時刻に 5 分を足した値を、上げた時刻の代わりに使う。
   - 推定どうしの比べになる。このため、見るのは判定の論理（どのレースが、どの段で、いつ「そろった」か）と文面にとどめる。
   - 見た目は `render_preview_html` で確かめる（送らない）。
3. 統括役の確認の後に、stg の Webhook へ接続テストを 1 通送る（`send_test`）。
4. 開催日の実データで確かめる（§8）。

## 7. 段取り

| いつ | 何を | 条件 |
|---|---|---|
| 11/19〜20 | 実装・テスト・critic・develop への push・CI | 統括役の承認の後 |
| 11/21〜23 | 既定は「入れない」。§8 の選択による | 統括役が選ぶ |
| 11/24〜 | 4a（push の時刻の記録）をホストに入れる → 11/28 の朝より前に cron と目印を入れる | 統括役・ユーザーの確認 |
| 11/28（土）〜 | stg・本番の両方へ送る。初日は送った時刻を launch_watch_report・nginx のログと比べる | — |

- 金（11/20）昼の昇格との関係: 案 A はホストだけで動くので、**昇格に間に合わせる必要はない**。本番の通知の文面は、本番に入っている機能に合わせる（4c の `features`）。
- 開催日の朝には、ホストの当日の処理を変えない。新しい cron も入れない。
- 4a は push（今週末の本番載せの通り道）に触れる。このため、ホストへ入れるのは早くて 11/24。今週末の早送り（11/20 夜）には入れない。

## 8. 統括役に決めてほしいこと

1. 未勝利・新馬で、第 2 段の判定が本体より後に上がるときの扱い。
   - **推奨: 1 レース 1 回。** 会員に出る中身（今週末なら v6e と target）がそろった時点で送る。第 2 段がその時点で上がっていれば、1 行足す。上がっていなければ「判定待ち」と書く。
   - 別案: 第 2 段が上がった時点（または T-20 の system_skip）で、もう 1 通送る。1 レースに最大 2 通になる。
2. 今週末（11/21〜23）の扱い。
   - **推奨: 何も入れない。** 今週末の本番載せの見張りを優先する。
   - この場合、今週末の push の時刻は残らない（manifest の `recent` は 2 日で刈られる）。当て直しは、ファイルの更新時刻からの推定になる。
   - 別案: 金の夜に新しいスクリプトと cron だけを、目印なし（送らずにログだけ）で入れる。実データで判定の時刻を確かめられる。4a は入れない。push の時刻は、毎分 manifest を読んで手元に写す。ホストに足すのは、読むだけの処理が 1 つ。
3. 本番のリンクの基点を `https://app.megu-keiba.com` にしてよいか（依頼文は `https://megu-keiba.com`）。

## 9. 別のエージェントのレビュー（2026-11-19、critic。16 件）

| # | 指摘 | 対応 |
|---|---|---|
| 1 | 本番の画面は `app.megu-keiba.com` | 直した（1・4c 節）。確かめた: `env_parity_check.sh:12`・`billing.json:7-10` |
| 2 | root の crontab は UTC | 直した（4f 節: `* 22-23,0-8 * * *`。今日は Asia/Tokyo で求める） |
| 3 | 未勝利・新馬は、v6c・第 2 段が画面の中身になりうる | 直した（4b-3。`stage2_live` の関数を使い回す） |
| 4 | Webhook が汎用に落ちる・`.env` を読まない・見出しの取り違え | 直した（4c 節） |
| 5 | 溜まった分が一度に出る・発走後の ready | 直した（4b-4・4b-6: 発走前かつ 15 分以内。dry-run の分は見送りとして記録） |
| 6 | `KEIBA_SLACK_NOTIFY` が効かない | 直した（4f 節: 両方がそろったときだけ送る） |
| 7 | 目録の書き込みの失敗で、Cloud Run が気付くのは 20〜30 分後 | 直した（3 節の弱点。ログで見分けて警告に回す） |
| 8 | 当て直しの材料（`recent` は 2 日で刈る） | 直した（4a で push の時刻を手元に残す。§8-2 の根拠を書き直した） |
| 9 | 警告がうるさい | 直した（4b-5: 原因ごとにまとめ、stg だけに送り、中止は除く） |
| 10 | 再試行の重なり・設定が無いとき | 直した（4e 節: retries=1、実行の単位で数える、`flock -n`、重ねて送るのは最大 1 通） |
| 11 | stg と本番は動くコードが違う | 直した（2 節・4c の `features`） |
| 12 | 発走時刻の出所 | 直した（4b-2: 出馬表の写しを先に使う。食い違いは実装のときに数える） |
| 13 | 開催場・クラスを GCS から取らない | 直した（4b-2。テストで storage の呼び出し 0 回を確かめる） |
| 14 | events の複数行 | 直した（4b-2: 最後の `status=ok` の行。`v6_status=ok` も見る） |
| 15 | 第 2 段の開始日を二重に持たない | 直した（4c 節） |
| 16 | 1 回の実行の重さ | 直した（4f 節: nice 19。開催日でなければ早く終える） |

## 10. 実装（2026-11-19）

| 何 | どこ |
|---|---|
| push の時刻の手元の記録（`data/local/meta/race_push_log/<日>.jsonl`。ミラーの木の外で GCS へは上げない。`index_ok` 付き） | `src/ops/gcs_mirror.py` の `_note_race_push`（host の push だけ） |
| 判定・送信・記録 | `src/scripts/ops/race_ready_notify.py` |
| 設定 | `config/race_ready_notify.json` |
| cron | `scripts/cron/race_ready_notify.sh`（flock -n・nice 19。.env は本体が読む）、`scripts/cron/install_race_ready_notify_cron.sh [--no-mark\|--remove]` |
| テスト | `tests/ops/test_race_ready_notify.py`・`tests/ops/test_gcs_mirror.py` |

当て直し（`python -m src.scripts.ops.race_ready_notify --day 20261115 --replay`。push の時刻はファイルの更新時刻＋5 分で代用）:

| 日 | レース | 「出た」の通知 | 遅れの警告 | 画面に出た見込み（発走の何分前） |
|---|---|---|---|---|
| 11/14 | 36 | 72（36 × 2 環境） | 0 | 44〜52 分前（中央 48） |
| 11/15 | 36 | 72 | 0 | 44〜54 分前（中央 48） |

weekend-launch の 11/15 の実測（API で見えるのは最も遅くて発走 44 分前）と合う。

入れ方（11/24 以降。統括役の合図の後、共有の作業ツリーにこのコードが入ってから、root で）:
1. `bash scripts/cron/install_race_ready_notify_cron.sh --no-mark`（送らずにログだけ。`logs/race_ready_notify.log` で判定の時刻を見る）
2. 11/28 の朝より前に `bash scripts/cron/install_race_ready_notify_cron.sh`（目印を置いて送り始める）
3. 止める: `rm data/local/meta/race_ready_notify.on`（すぐ効く）か、`--remove`
