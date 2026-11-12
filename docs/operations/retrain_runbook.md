# 再学習ランブック（別セッション用）

利用者が **「再学習を進めて」**（または「再学習の続きをやって」）と言ったら、このファイルに従って、順番に作業し、状態を記録する。
手順の中身は `docs/html/operations/retrain_order.html`（読み物）と `config/retrain_plan.json`（機械が読む定義）にある。
モデルの版と採用状況は `docs/html/operations/model_versions.html`。

## 1. セッションを始めたら最初にやること

1. 必ず日本語で答える。作業はホスト経由（`ssh keiba-remote`）で行う。dev PC から GCP に直接触らない。
2. ホストで状態を見る。無ければ作る。

```bash
ssh keiba-remote 'python3 /home/jovyan/work/retrain_state/retrain_ctl.py status'
```

- ホストに `retrain_ctl.py` が無いとき: dev PC のリポジトリから `scp scripts/dev/retrain_ctl.py config/retrain_plan.json keiba-remote:/home/jovyan/work/retrain_state/`（先に `ssh keiba-remote mkdir -p /home/jovyan/work/retrain_state`）。
- 定義（`retrain_plan.json`）を直したら、同じ場所へ送り直す。状態ファイルは `state.json`（消さない）。
- 別セッションが動いていないか確認する: `status` に `▶`（実行中）の段があれば、その段のログとプロセスを先に見る。二重に始めない。

3. `next` を実行し、出力に従う。1 段ずつ進める（並列にしない）。

```bash
ssh keiba-remote 'python3 /home/jovyan/work/retrain_state/retrain_ctl.py next'
```

以降 `ctl` と書くのは上のコマンドのこと。

## 2. 1 段ごとの進め方

1. `ctl next` で、次の段・コマンド・完了の確認を読む。
2. `ctl start <段>` で開始を記録する（依存の段が終わっていない・他の段が実行中・OK 待ちの段は拒否される）。
3. コマンドを実行する。重い処理は `/tmp/heavy_run.sh`（同時 3 本・load が 14 未満になるまで待つ）経由で、ログは `/home/jovyan/work/retrain_state/logs/<段>.log` に出す。ホストの Python は `/opt/conda/bin/python3`。
4. 長い処理は始めた時と終わった時だけ利用者に伝える（定期報告はしない）。待つときは完了を待つ作業を背景で行う。
5. 完了の確認（`done_check`）を実際に見て、満たしていれば `ctl done <段> --note "確認した内容と数値"`。満たさなければ `ctl fail <段> --note "原因"`。
6. 次の段へ。

## 2.5 上流（s3 → s4 → s7）は切り離しドライバで動いている

2026-11-06 に起動済み。`scripts/dev/retrain_upstream.py`（ホストの `/home/jovyan/work/retrain_state/`）が s3 追走難度 → s4 ラップ予想 → s7 前段 S3 を直列に実行し、段の状態を `retrain_ctl` に自動で記録する。出力は `/home/jovyan/work/retrain_out/`（本番の tp54_out・tp55_out3・final_stage・config は上書きしない）、ログは `retrain_state/logs/{s3,s4,s7}.log` と `upstream.out`。

- 動いているか: `ssh keiba-remote 'pgrep -af retrain_upstream.py'`。`status` が `▶` ならドライバが担当中なので、同じ段を手で始めない。
- 失敗したら: ドライバは止まり、段は `✘`。原因を直して `ctl reset <段>` → 上と同じ起動コマンド（`cd /home/jovyan/work/keiba-vpn && setsid nohup /opt/conda/bin/python3 /home/jovyan/work/retrain_state/retrain_upstream.py > .../logs/upstream.out 2>&1 < /dev/null &`）で再開する（済みの段は飛ばす）。
- 完了後: 旧版との比較が `retrain_out/parity_{s3,s4,s7}.json` にある。相関が低い列があれば、入力データの変化（2025 の passing_order など）が原因か確認してから s9 以降へ進む。s8（RB）以降はドライバの対象外で、この手順に沿って 1 段ずつ進める。

## 3. 最初に利用者へ伝える既定値（確認は 1 回だけ）

最初のセッションで、次の既定のまま進めてよいかを一言で伝え、反対されなければ `ctl decide <項目> <値> --confirm` で確定する。止まって待たない（反対が来るまで既定で進める）。

| 項目 | 既定 | 意味 |
|---|---|---|
| scope | models | めぐ指数 v3a は変えず、追走難度・ラップ予想 → S3 → 親モデル → 較正・馬券 → 配備 |
| train_years | 2020-2025 | **2016-2025 は選ばない**。選ぶならコード固定の年・位置取り予想の代替の一致確認が要るので、必ず止まって利用者に確認する |
| with_optional | no | 展開タイプ・ペース予測は実行しない |
| feature14 | no | 風で補正した履歴は同梱しない（v6e と比べられなくなる） |

伝えるべき注意: 定義を変えずに 2020〜2025 で学び直すと、結果は v6e とほぼ同じになる。

## 4. 利用者の OK が要る段（ゲート）

次の段は、実行前に利用者の明示の OK を得る。OK の言葉を `ctl approve <段> --by "発言の要約"` に記録してから `start` する。

- s1: DB・GCS への反映を伴うめぐ指数の再計算（scope が full のときだけ出る）
- s12: 過去分の予測の差し替え
- s18: 配備（stg → prod）

OK が出ていなければ、そこまでの結果を報告して待つ。

## 5. 守ること（例外なし）

- **費用**: GCP + Cloudflare は月 2,000 円以内（CLAUDE.md）。常駐・周回・GCS の読み出しを足さない。s18 の配備で繰り返す処理を足すときは、最悪の回数で月額を見積もり、円で示して確認する。
- **test 2025 は親モデル本体（s10）で 1 回だけ評価する**。結果を見て手順や特徴を変えて再評価しない。やり直したくなったら止まって利用者に聞く。
- **2026 年は学習・選択に使わない**（テストセット②）。
- 市場オッズ・人気は親モデルの特徴量にしない。
- 平地と障害を混ぜない。ID（horse_id / jockey_id / trainer_id）をキーにする。
- ホストの資源を超えない: 6 コア空ける・重い処理は直列・load < コア数。他の Claude が取得中なら `/tmp/when_idle.sh` で空きを待つ。
- 馬券の結果に「100% 超」と書かない。区間の下限が 100% を超える方策は確認できていない。
- 勝手に採用を変えない。新しい版が v6e より悪いときは**採用しない**で、結果を報告する。
- prod / master への反映・秘密情報・破壊的な操作は、必ず確認してから。

## 6. 失敗したとき

1. `ctl fail <段> --note "原因"`。ログの末尾を見て原因を特定する（メモリ不足・load・入力の欠け・コード固定の年など）。
2. 直せる（再実行・環境変数・入力の補完）なら直して `ctl reset <段>` → `start` からやり直す。
3. 判断が要る（精度が基準に届かない・設計の変更・費用・年の拡張）なら `ctl block <段> --reason "..."` で止め、利用者に報告して待つ。
4. 同じ段が 2 回失敗したら止めて報告する（3 回目を黙って回さない）。

## 7. 完了したら

- `ctl status` が全段 ✔／－ になったら、結果を `docs/html/operations/model_versions.html`（採用一覧・履歴）と `retrain_order.html`、必要なら `training_pipeline_decisions.html` に反映する（採用しなかった場合も理由つきで）。
- 記録（`ctl log` と state.json の history）の要点を利用者に報告する。報告の形式は、段ごとの表（段 / 状況 / 数値 / 次）。
- メモリ（`project_retrain_runbook.md`）の状態を更新する。

## 8. 状態の見方

`docs/html/operations/retrain_order.html` の §2.1 のチャートにも、箱ごとに同じ記号の状態が付き、実行中の段の箱は橙の太枠になる（`retrain_ctl.py` が記録のたびに書き直す。手動で直すときは `retrain_ctl.py html`。ホストの `retrain_state/retrain_ctl.py` は `scripts/dev/retrain_ctl.py` の写しなので、直したら送り直す）。

| 記号 | 意味 |
|---|---|
| ・ | 未着手 |
| ▶ | 実行中 |
| ✔ | 完了 |
| ✘ | 失敗（原因を調べる） |
| ■ | 止め（利用者の判断待ち） |
| － | 対象外（scope・任意で除外） |
| ★要OK | 実行前に利用者の OK が要る |

コマンド一覧: `status` / `next` / `decide` / `approve` / `start` / `done` / `fail` / `block` / `skip` / `reset` / `log`（詳しくは `python3 retrain_ctl.py -h`）。
