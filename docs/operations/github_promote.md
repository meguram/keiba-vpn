# GitHub Actions による昇格（develop → main のマージで prod の画面を公開）

`main` に push（= `develop` からのマージ）されると、`.github/workflows/promote.yml` が **テスト → 画面のビルド → 承認 → GCS に公開** まで進める。
VPS は、公開された版を `pull_release.py` が窓（毎日 JST 03:10）に取りに行って入れ替える（失敗時は自動で戻る）。SSH は開けない。

## 1. 全体の流れ

```
develop ──(マージ)──▶ main ──push──▶ promote.yml
                                      │
   ① test        ci.yml と同じテスト＋秘密の検査（secret_scan --tracked）
   ② build-ui    node 20 / next build（standalone）/ version.json（commit・source_digest・NEXT_PUBLIC_*）
                 → tar.gz ＋ .sha256 ＋ .manifest.json（artifact に 14 日保存）
   ③ 【承認】    Environment「production」の Required reviewers が承認するまで、④ は始まらない
   ④ publish-ui  WIF で GCP に認証（鍵なし）→ sha256 を検算 → GCS の ops/releases/ui/ に置く → latest.json を最後に書く
                 │
                 ▼  GCS: ops/releases/ui/{<版>.tar.gz, <版>.tar.gz.sha256, <版>.manifest.json, latest.json, previous.json}
   VPS（pull_release.py）毎日 JST 03:10 に latest.json を見て取り込み → install_release.sh → healthcheck → 失敗なら自動で元の版
```

- **取得（スクレイピング）は CI で一切しない。**
- **当てはめ済みの係数・重み（`config/fitted_artifacts.txt`）は CI のビルドに含まれない。** 画面のビルドには不要で、API やジョブが実行時に GCS（`models/fitted_artifacts/`）から取得する。
- GCS の接頭辞は既存の `ops/releases/`（`src/ops/release_publish.py` の `PREFIX`）。VPS の取り込みと同じ場所で、ホストからの `publish_release.sh` とも共存する（`latest.json` を書いた方が「prod に出す版」）。

### API（Cloud Run 移行後）
`workflow_dispatch` で `deploy_target=api`、かつリポジトリの Variables に `ENABLE_API_PROMOTE=true` のときだけ動くスケルトン（既定は無効）。
`api-revision`: ビルド → **通信 0% のリビジョン**を作る → スモーク。`api-traffic`: **別の承認**（Environment `production-traffic`）の後に通信を切り替える。戻すのは `scripts/gcp/rollback_cloud_run_api.sh`。

## 2. あなたの作業（この順に。コマンドの値は文書に書かない）

1. **Cloud Shell で WIF を用意する**（鍵は作らない）
   ```bash
   cd ~/keiba-vpn && git pull
   GCS_BUCKET=<バケット名> bash scripts/gcp/setup_github_wif.sh            # 表示だけ（dry-run が既定）
   GCS_BUCKET=<バケット名> bash scripts/gcp/setup_github_wif.sh --apply    # 内容を確認してから実行
   ```
   - 作るもの: Workload Identity Pool と Provider（入れる条件: **リポジトリ `meguram/keiba-vpn`、ブランチ `main`、Environment `production`**）、`keiba-deployer` への `roles/iam.workloadIdentityUser`、バケットの **`ops/releases/` だけ**への `roles/storage.objectUser`（IAM Conditions）。
   - 前提: バケットが **uniform bucket-level access**（IAM Conditions が使える）。確認: `gcloud storage buckets describe gs://<バケット名> --format="value(uniform_bucket_level_access)"` が `True`。
   - 最後に、次の手順で入れる値（`WIF_PROVIDER`、`WIF_SERVICE_ACCOUNT`、`GCS_BUCKET`）が表示される。
2. **GitHub: Environment を作る**（Settings → Environments → New environment）
   - `production`: **Required reviewers に本人**、**Deployment branches = Selected branches → `main` のみ**。
   - API を有効化するときだけ、`production-traffic` も同様に作る（承認者は別の人でもよい。一人運用なら同じ本人で、承認を 2 回に分ける目的）。
3. **GitHub: `main` のブランチ保護**（Settings → Branches → Add rule、対象 `main`）
   - 直接 push 禁止（PR 経由）、必須のステータスチェックに CI の `lint-and-test` を指定。PR 必須は任意（一人運用なら外してもよいが、直接 push 禁止は付ける）。
4. **GitHub: Variables を入れる**（Settings → Secrets and variables → Actions → **Variables**）
   | 名前 | 内容 | 秘密か |
   |---|---|---|
   | `WIF_PROVIDER` | 手順 1 の最後に表示された値 | いいえ（Variables でよい） |
   | `WIF_SERVICE_ACCOUNT` | `keiba-deployer@<プロジェクト>.iam.gserviceaccount.com` | いいえ |
   | `GCS_BUCKET` | 公開先のバケット名 | いいえ |
   | `NEXT_PUBLIC_DEFAULT_RACE_DATE` `NEXT_PUBLIC_PICKER_PAST_DAYS` `NEXT_PUBLIC_ENABLE_FEATURES` | stg のビルドと同じ値（`scripts/ops/vps/build.env.example` 参照） | いいえ |
   - **Secrets に入れるものは、ありません**（鍵ファイルを作っていないため）。
5. **アクションを SHA 固定する**（任意だが推奨。変更を勝手に取り込まない）
   ```bash
   bash scripts/ci/pin_actions.sh           # 解決結果を表示
   bash scripts/ci/pin_actions.sh --write   # promote.yml の `# PIN` 付きを @<40桁> に書き換え
   ```
   （`git ls-remote` で GitHub の公開リポジトリを読むだけ。書き換え後はコミットして push。）
6. **初回は手動で試す**: Actions → promote → Run workflow（`main`、`skip_tests` はオフ）。承認の画面で **Review deployments → production → Approve**。完了後、`python -m src.ops.release_publish status --kind ui` で `latest.json` を確認。VPS は次の JST 03:10 に取り込む。

## 3. すぐに反映したいとき（選択肢）
既定は『次の窓（03:10）』。急ぐ場合の案（いずれも SSH は開けない）。
- **A. 従来どおり `promote.sh`**（ホストから SSH で即時）。ホストを切り離した後は使えない。
- **B. VPS の pull を手動で起動**: ConoHa のコンソール、または `sudo systemctl start keiba-release-pull@ui.service`（VPS に入れる手段がある間）。
- **C. GCS の変更通知で起動**: `latest.json` の更新を Pub/Sub → Cloud Tasks/Run で受け、VPS が定期的に見る間隔を短くする（例: 窓の外では 5 分おきに `latest.json` の鮮度だけ確認）。実装は別作業。

## 4. ロールバック
- **公開を取り消す（VPS がまだ取り込む前）**: GCS の `ops/releases/ui/latest.json` を `previous.json` の内容に戻す（`gcloud storage cp gs://<バケット>/ops/releases/ui/previous.json gs://<バケット>/ops/releases/ui/latest.json`）。
- **VPS が取り込んだ後**: VPS で `sudo /opt/keiba-ui/bin/rollback.sh`（直前の版に戻る）。`pull_release.py` は失敗した版を繰り返さない（`failed_ids`）。
- **API**: `scripts/gcp/rollback_cloud_run_api.sh`（前のリビジョンへ戻す）。

## 5. 費用
公開リポジトリの GitHub Actions は無料。GCS への置き場所は数十 MB/版で、無視できる額。

## 6. セキュリティ
- **鍵ファイルを作らない・Secrets に GCP の鍵を置かない。** WIF の条件（リポジトリ・`main`・`production` の承認を通った job）に合わない実行は、GCP に入れない。
- **フォークからのプルリクエストには秘密を渡さない**（GitHub の標準の挙動）。`pull_request_target` は使わない（テストで禁止を検査）。
- `permissions` は既定 `contents: read`。`id-token: write` は公開の job だけ。
- サードパーティの action は **コミット SHA 固定**（上の手順 5）。更新は手動で確認してから。
- 公開前に `scripts/dev/secret_scan.sh --tracked` を CI のテスト job で実行する（秘密・内部情報の混入を止める）。
- 成果物は承認の間に差し替えられないよう、公開の job で sha256 を検算する。

## 7. よくある失敗
| 症状 | 原因と対処 |
|---|---|
| 認証で `unauthorized_client` | Provider の条件に合っていない（ブランチが `main` でない／job が `environment: production` でない／リポジトリ名の違い）。`WIF_PROVIDER` の値も確認 |
| 認証できたが `403` で書けない | 条件付きの権限が `ops/releases/` に限られている。接頭辞が違う／uniform bucket-level access が無効 |
| `build.env` の項目数が 0 | Variables の `NEXT_PUBLIC_*` が未設定。stg と揃っていないと、`source_digest` で差分が出る |
| 承認の画面が出ない | Environment `production` に Required reviewers が未設定、または Deployment branches が `main` でない |
| VPS が取り込まない | 窓（03:10）前／`pull_release.py` が同じ版の失敗を記録済み（`--force` で解除）。`release_publish status --kind ui` の `applied/vps.json` を見る |
| テストが CI で落ちる | 係数ファイル（git 管理外）が無い環境でも skip する作り。新しいテストで係数を前提にしていないか確認 |
