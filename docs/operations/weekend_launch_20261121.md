# 今週末（11/21 土・11/22 日）の本番載せ: 新しい候補＋第 2 段＋凍結した買い方（試行中）

担当: weekend-launch。統括役: claude-orchestrator-handover。関係する担当:
- model-improve-cycles: 候補・前段の保存値・RB
- raceday-weight-runner: 本体の当日の推論・早送り
- bet-optimizer: 凍結した買い方

公開するかは 11/20（金）15:00 の判定で決める（§9）。費用の増分は最悪で月 約 25 円（ユーザーの了承は約 20 円。記録の予備の約 4.4 円を 11/18 夜に足した。統括役に報告済み）。

## 1. 何を・どこに載せるか

- **本番の v6e・§77（想定オッズ）・今の推奨（target）は一切触らない。** 本体の raceday-weight-runner はそのまま動く。
- 次の 3 つは、ホストの**別のプロセス（候補の影のランナー `src/pipeline/inference/candidate_runner.py`）**で動かす。
  - 新しい候補: cycles16_fix3/final の束（2016〜2025 年で学習）
  - 第 2 段: noise_fit2025（T-60 の単勝オッズを offset）
  - 凍結した買い方: `config/bet_cycles/frozen_20261117_r2.json`（hash 70ba29fa7b55）
- 別のプロセスにする理由は 2 つ。
  - 2016 年起点の設定一式（`KEIBA_FIRST_YEAR=2016` は import の時点で決まる値。表の付け替え redirect_serving・layer_a）を、そのプロセスの中だけで付けるため。
  - 新しい束では §77 の要る列のうち 36 列が無いため。本体を新しい束に替えると、今の推奨まで全レースで「準備中」になる。
- **対象は未勝利・新馬のレースだけ**（ユーザーの判断 2026-11-18）。ほかのレースは、確率の欄・推奨馬券・レース一覧の推奨度・印のどれも今の表示のまま。
- 出力の決まり:
  - 1 レース 1 回だけ書く。発走後と判定済みのレースには書かない。
  - 規則による見送り（頭数・市場とのずれが小さい・候補なし）は、判定ファイルに書く。
  - **システムの原因**（T-60 の写しが無い・推論の失敗・メモリや GCS の上限・前段の保存値が無い・オッズの崩れ）のときは、**ファイルを書かない**。画面は今の target のまま。見張りの記録に `system_skip` を残す。

| 木 | 中身 | 画面での使われ方 |
|---|---|---|
| `data/local/inference/predictions_v6c/<日>/<rid>.parquet` | 新しい候補の第 1 段の確率 | 未勝利・新馬のレース詳細の確率の欄。札「新しいモデル（試行中）」を付け、v6e の市場補正・EV の較正は当てない |
| `data/local/inference/predictions_stage2/<日>/<rid>.parquet` | 第 2 段の p・q・r・T-60 オッズ・人気 | 記録用 |
| `data/local/inference/bets_stage2/<日>/<rid>.json` | 買い目の判定（買う／規則による見送りと理由） | 未勝利・新馬の推奨馬券のタブ。札「試行中」を付ける |

- 3 つの木は gcs-mirror-push（5 分おき）で上がる。**stg（rstg）と本番は同じミラーを読むので、両方に同時に出る。**
- 実オッズ由来の値は、開発者（view=full）だけに出す（`actual_odds_public: false`）。
  - 対象: T-60 の単勝オッズ・人気・q・r と、それを offset にした第 2 段の勝率・買いの線
  - 会員に出すのは、◎ の馬番と「試行中」だけ
- 成績の記録（prediction_results）は、target の仮想精算のまま（v6e）。第 2 段の試行の成績は、`bets_stage2` と確定払戻から週明けに別に精算する。

## 2. 当日の流れ（JST）

| 時刻 | 何が | どこで |
|---|---|---|
| 前日 夜 | 前段 S3 の保存値（tracking_v2・lap_forecast）を 2016 年版で翌日分作る（s12/phase1.sh）。RB の状態（base_2016/rb_state）を作り直す | model-improve-cycles |
| 07:30 | raceday-runner。T-60 で単複を保存した直後に、手元の写し `odds_t60/<日>/<rid>.json` を書く（目印 odds_t60_copy.on があるとき） | ホスト（本体） |
| 07:45 | 影のランナーを起動 → 対象（未勝利・新馬）を数える → 事前の確認（preflight・鮮度）→ warm（約 5〜8 分、RSS 約 3〜10GB） | ホスト（cron `KEIBA-CANDIDATE-RUNNER`） |
| 08:00〜16:45 | 15 分おきに起動し直す。落ちたときだけ動き、flock で二重にならない | 同上 |
| 08:00 | 本体の weight-runner（v6e・§77） | ホスト（本体・変更なし） |
| 各 R | 本体の推論が済む（events の status=ok）か T-30 を過ぎたら、候補の第 1 段を流す（発走の早い順に 1 レースずつ）。T-60 の写しが届いたら、第 2 段と判定 | 影のランナー |
| 各 R の T-20 | 判定が無いレースは `system_skip`。ファイルは書かない（今の表示のまま） | 影のランナー |
| 5 分おき | push → API の目録（60 秒）→ prerender（120 秒）→ 画面（45 秒） | ホスト → Cloud Run → VPS |

- 第 1 段の時間（11/15 の当て直し。ホストの load は 20 台）: 1 レース 120〜140 秒。その日の 1 レース目だけ約 300 秒。
- 判定から画面までの遅れは、最悪で約 9 分。

## 3. 有効にする手順（ユーザーの承認の後。統括役が合図）

追跡されるファイル（config など）は書き換えない。書き換えると、共有の作業ツリーの毎時の pull が止まる。

1. master に昇格する（API と画面）。画面は VPS が 03:10 に取り込むので、金の夕方までに。昇格しても、木にファイルが無い間は今の表示のまま。
2. 共有の作業ツリーに、このコードが入っていることを確かめる（raceday-weight-runner の 11/20 夜の早送り）。
3. root で `bash scripts/cron/install_candidate_runner_cron.sh` を流す。次の 2 つが入る。
   - 目印 `data/local/meta/candidate_runner.on`・`odds_t60_copy.on`
   - cron の行（07:45 の起動と、15 分おきの起動し直し。`KEIBA_CANDIDATE_RUNNER=1` 付き）
4. 影の作業場所 `/home/jovyan/work/retrain_out/candidate_live/inf` を用意する。
   - `tracking_v2`・`lap_forecast` は、base_2016/s12/inf へのリンク
   - `state/px_v5` は、本番と同じ中身のコピー
5. model-improve-cycles が、前日の夜に次の 2 つを作る。
   - 前段の保存値（翌日分）
   - RB の状態（cut が、履歴の表の最終日以降になるように）

## 4. 事前の確認（金の夜〜土の朝）

| いつ | 何を見る | 正常 |
|---|---|---|
| 金 夜 | `base_2016/s12/inf/tracking_v2/2026-11-21/` のレース数 | 開催の全レース（障害は lap_forecast に無くてよい） |
| 金 夜 | `cycles16_fix3/rb_live/state_live/params.json` の cut（`rb_live/eve.log`） | 履歴の表の最終日（2026-11-16）以降。古いと、影のランナーは warm の後に止まる（stale_stop）。そのときは全レースが今の表示 |
| 金 夜 | めぐ指数の週次（金 18:00 開始、書き終わりは 18:45〜23:24）の後に、rb_live.sh を流し直したか | 土の朝の `launch_watch` の rb_state の megu_last が 11/15 以降（cut では分からない。本体の v6 も同じ条件なので、止めずに記録する） |
| 土 07:55 | `logs/candidate_runner.log` の「対象 n / m レース」と「warm 済み」 | FIRST_YEAR=2016・束 cycles16_fix3/bundle_v6_2025・付け替え base_2016_fix2。preflight_missing・stale_stop が無い |
| 土 1R の T-55 | `data/local/inference/odds_t60/20261121/` | 1R の写しがある |
| 土 最初の未勝利・新馬の T-50 | `launch_watch/20261121.jsonl` | decision か system_skip がある |
| 同 T-40 | 画面のレース詳細・推奨馬券のタブ | 「試行中」の札と、買い目か見送りの理由 |

## 5. 見張り（レースごとに T-10 に間に合ったか）

- 記録はホストの手元だけ（GCS を読まない・API を叩かない）。
  - `data/local/inference/launch_watch/<日>.jsonl`: scope・warm・v6c・odds_t60_seen・decision・system_skip・end
  - 同 `push_<日>.jsonl`: 判定ファイルを上げ終えた時刻
  - 本体の `events/<日>.jsonl`
- まとめ: `python -m src.scripts.data.launch_watch_report --day 20261121`。レースごとに次を出す。
  - 発走・本体の推論・T-60・判定・push の完了の時刻
  - 「API で見える見込みの時刻」（push の完了＋60 秒）と、T-10 に間に合ったか
- 「画面に出たか」は、次の 2 つのログで見る（読むだけ・無料の範囲）。
  - VPS の nginx のアクセスログ
  - Cloud Run のリクエストのログ
  - 見るのは、そのレースの `/api/v1/betting/optimize` の 200 の有無と時刻。誰も開かなかったレースは「推定（push の時刻から）」と書く。会員の Cookie は使わない。
- API を周期で叩く ping は足さない（CLAUDE.md）。

## 5b. 前向きの記録（試行の成績を後から集計する元。集計は bet-optimizer）

11/21 からの試行で、レースごとに次の 4 種類を残す。どれも race_id で突き合わせられる。

| 種類 | 置き場（ホスト） | 形式・主な項目 | GCS |
|---|---|---|---|
| 1. 実際の T-60 のオッズ | `data/local/inference/odds_t60/<YYYYMMDD>/<race_id>.json` | `phase=t60`・`snapshot_at`・`entries[]`（horse_number・win_odds・popularity） | 予備（上げるだけ） |
| 1. 同上（第 2 段の入力のまま） | `data/local/inference/predictions_stage2/<YYYYMMDD>/<race_id>.parquet` | horse_id・horse_number・p_win・p_top2・p_top3・q・r・odds・pop_t60・odds_t60_at・odds_phase・model | ミラー |
| 2. 判定（買い・見送りと理由・◎・p・q・r） | `data/local/inference/bets_stage2/<YYYYMMDD>/<race_id>.json` | status・skip_reason・skip_label・honmei・bets・horses[]（p_win・q・r・pop・odds）・r_max・q_r_max・c_class・field・frozen_hash・odds_phase | ミラー |
| 2. システムの原因の見送り | `data/local/inference/launch_watch/<YYYYMMDD>.jsonl` の `event=system_skip` | race_id・reason・detail・at | 予備（上げるだけ） |
| 3. 判定の時刻 | 判定ファイルの `as_of` | ISO8601（+09:00） | ミラー |
| 3. 判定・push の時刻 | `launch_watch/<YYYYMMDD>.jsonl` の `event=decision`、`launch_watch/push_<YYYYMMDD>.jsonl` | ISO8601（+09:00） | 予備（上げるだけ） |
| 4. 確定の着順と払戻 | 週明けに、`data/page_reference/tables/2026/race_result_flat.parquet`・payoff と race_id でつなぐ | — | 既存の表 |

- odds_t60 と launch_watch は、ほかに元が無い。**ホストの手元から消さない**（掃除の対象にしない）。
- 予備として、gcs_mirror の BACKUP_TREES で GCS へ上げるだけにしている（Cloud Run には降ろさない。画面のキャッシュの目印も立てない）。開催日の 5 分おきの push に乗る。
- cron の push は `--recent-days 2`（更新時刻が 2 日以内のファイルだけ）。前向きの記録は 11/21 から書かれるので、開催日のうちに上がる。リハーサル（11/08・11/15）の記録は対象外で、上げない。
- 試行の成績の集計（参戦・的中・回収率・見送りの理由ごとの内訳）は、週明けに bet-optimizer が行う。prediction_results（target の仮想精算）とは別に数える。

## 6. 戻し方と判断の基準

| 対象 | 戻し方 | 効くまで |
|---|---|---|
| 予測の表示と推奨馬券の両方 | ホストで root として `sudo env PYTHONPATH=$PWD /opt/conda/bin/python -m src.pipeline.inference.candidate_runner --set-disabled on --day 20261121`（共有の作業ツリーで） | 約 9〜11 分（push 5 分＋目録 60 秒＋prerender 120 秒＋画面 45 秒）。その日の判定ファイルを書き直すので、レースごとのキャッシュの目印も立つ |
| 予測の表示だけ（推奨は第 2 段のまま） | 同じコマンドに `--v6c-only` を付ける | 同じ |
| また有効にする | `--set-disabled off`。ミラーは削除を伝えないので、目印は消さずに `{"disabled": false}` で上書きする | 同じ |
| 影のランナーを止める | `pkill -f src.pipeline.inference.candidate_runner`。止めた後のレースは今の表示。15 分おきに起動し直すので、止め続けるなら cron の行を消す | すぐ |
| 翌日から止める | `bash scripts/cron/install_candidate_runner_cron.sh --remove`（行と目印を消す） | — |
| **予備: ホストの push が動かないとき** | 開発者の PC から `gcloud storage cp` で、`gs://<GCS_BUCKET>/ops/mirror/data/local/inference/bets_stage2/_disabled.json` に `{"disabled": true}` を直接置く。Cloud Run は、目録か 30 分ごとの全体の一覧で拾う | 最大 約 30 分 |
| **予備: それでも戻らないとき** | `scripts/gcp/rollback_cloud_run_api.sh` で、前の版（第 2 段を読まない API）に戻す。ユーザーの承認が要る | 約 2〜3 分 |

- 本番の v6e・§77・target は動き続けているので、戻した先は常に生きている。
- stg と本番は同じミラーを読むので、目印は両方に同時に効く。
- 実測（2026-11-18 の stg、11/15 の判定で。VPS の中から 1 分おきに 1 回だけ問い合わせ）:
  - 1 回目（17:00）: on で推奨馬券は 9 分 20 秒、確率の欄は 10 分 40 秒で今の表示に戻った。
  - 2 回目（19:24）: on で推奨馬券は 6 分 10 秒、off で第 2 段の表示に 4 分で戻った。
  - 確率の欄は、2 回目は 25 分たっても戻らなかった。終わった開催日の予測の応答は 6 時間使い回す決まりのため（`prerender.race_max_age`）。
    開催当日のレースは 120 秒で作り直すので、当日は推奨馬券と同じく約 9 分以内に戻る見込み。過去の日の表示まで急いで戻すときは、配備で世代を進める。

戻す基準:
- `system_skip` が、1 日の対象レースの 2 割を超える → 原因を調べる（T-60 の取得の失敗なら統括役へ）。失敗したレースには今の target が出るので、画面に空欄はできない。
- T-15 までに判定ファイルも `system_skip` も無いレースが 2 つ続く → 影のランナーが止まっている。ログを見て、起動し直しを待つ。
- 候補の予測で、次のどちらかが起きる → `--set-disabled on --v6c-only`。
  - 全頭が同じ確率
  - 予測 1 位＝1 番人気の割合が、未勝利・新馬の 1〜6 レースで極端（40% 未満か 90% 超）
- 本体の推論が遅れる（events の T-30 超え）か、メモリが詰まる → 影のランナーを止める。
  - 影のランナーは、RSS 20GB で推論をやめ、24GB で自分から終わる。`oom_score_adj=1000` で、本体より先に落ちる側にしてある。
- 開催中（土日 07:30〜17:00）は、ほかのセッションの重い計算を止めてもらう（統括役が各担当に伝える）。

## 7. 費用（最悪の回数で。CLAUDE.md の再発防止 1・5）

| 項目 | 回数・量 | 月額 |
|---|---|---|
| ホストの計算 | ホストの CPU | 0 円 |
| push の書き込み（3 つの木） | 36R × 3 ファイル × 1 回 × 開催 9 日 ＋ 目録 ≒ 1,000 回 | 約 0.8 円 |
| 30 分ごとの全体の一覧への接頭辞の追加 | 48 回 × 30 日 × 3 ＝ 4,320 回（Class A） | 約 3.5 円 |
| 新しい木が空の間の起動時の一覧 | 最悪 96 回 × 30 日 × 3 木 | 最大 約 7 円 |
| 影のランナーの GCS の読み出し（出馬表など） | 1 日 50MB を上限に止める（11/08 の 2 レースで 0.6MB・226 回）。開催 9 日で最大 0.45GiB | 最大 約 8.5 円 |
| 見張り | ホストの記録・ログを読むだけ | 0 円 |
| 記録の予備（odds_t60・launch_watch を GCS へ上げるだけ） | 最悪: launch_watch の 2 ファイル × 1 日 288 回（5 分おきの push のたびに中身が変わるとみなす）＋ T-60 の写し 36 ＝ 612 回/日 × 開催 9 日 ≒ 5,500 回（Class A）。画面のキャッシュの目印は立てないので、VPS の取り直しは増えない。GCS の外への転送は無い | 約 4.4 円 |
| 合計 | | **最悪 月 約 25 円**（通常は 7 円未満） |

## 8. 事象と根拠

- 凍結した買い方:
  - 規則: 未勝利・新馬のレースで、2 番人気以下・勝率 5% 以上の馬の r=log(p/q) の最大が、訓練の 5 分位の 3 段目以上（0.4917）なら、その馬の単勝を 100 円買う。
  - 2025 年 test（2 回目の確かめ）: 回収率 107.0%（区間 94.0〜119.6%）。上位 5 件を除くと 94.4%。**回収率は確かめ中**なので、画面に「試行中」と出す。
  - 障害の未勝利も、凍結のとおり対象。2025 年 test で、障害の参戦は年約 30 回・回収率 83%。
- 当日の判定は、バックテストの関数（bet_cycles の play・_r_features・apply_qcuts）をそのまま呼ぶ。2025 年 test の正解 19,602 行と、全件一致を確認した。
- 第 2 段の当日の計算は、model_cycles_market.predict_full と同じ。2026 年の 89 レースで、差の最大 1e-16・◎ の一致 89/89。

## 9. 金 15:00 の公開の判定（すべて満たせば公開。1 つでも欠ければ影だけ）

1. 列の一致: 2025 年の 1 日で、影のランナー自身の X（`x_dir`）と学習の行列を列の群ごとに比べ、不一致率がほぼ 0（model-improve-cycles）。
2. 第 2 段の p と判定の一致（済み: 差 1e-16・19,602 行）。
3. preflight と鮮度の検査が、実物の作業場所で通る。止める側で動くことも確かめる。
4. 11/15 の全レースを本体と同時に当て直し、次を数字で出す。
   - 最悪の遅れ（本体の推論 → 判定 → API で見える見込み）が T-15 より前
   - RSS の最大
   - GCS の全読み出しの量・回数と、一覧の回数
   - 保存の試みが 0 件
5. stg で失敗の演習をする（T-60 の写しなし・1R の失敗・全部の失敗）。どれでも画面が今の表示に戻ること。止めの目印を往復させ、各画面が何分で戻るかを測る。
6. 有効化の手順を流しても、共有の作業ツリーの pull が止まらない（git status が汚れない）。
