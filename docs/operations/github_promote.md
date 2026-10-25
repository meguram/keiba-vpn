# GitHub Actions による反映（develop → stg → master。push で自動）

ブランチは **develop → stg → master** の順にマージする。**push（= マージ）で次が自動で動く**。

| push 先 | ワークフロー | 反映先 | 承認 |
|---|---|---|---|
| `stg` | `.github/workflows/deploy-stg.yml` | stg 環境: API の通信 0% のタグ付きリビジョン（`rstg`）＋スモーク。画面は STG バッジつきでビルドして `ops/releases/ui-stg/` に公開（VPS の 2 つ目の画面 `stg.megu-keiba.com` が 5 分おきに取り込む。**8 章の準備が済んでから**） | なし（Environment `stg`） |
| `master` | `.github/workflows/promote.yml` | 本番: API の通信を新リビジョンへ切り替え／Jobs の更新／画面を GCS に公開（VPS が窓で取得） | **あり**（Environment `production` の Required reviewers、1 回） |

共通の部品: `_test.yml`（テスト）・`_build-ui.yml`（画面のビルド）。手順の本体は `scripts/ci/deploy_gcp.sh`（手元で `--dry-run` して確かめられる）。
VPS は、公開された画面を `pull_release.py` が窓（毎日 JST 03:10）に取りに行って入れ替える（失敗時は自動で戻る）。SSH は開けない。

## 1. 全体の流れ

```
develop ──マージ──▶ stg ──push──▶ deploy-stg.yml（変更されたファイルで動く job を選び、並走。画面だけの変更は約 2 分、API の変更は約 4 分。
                                    文書・テストだけの変更は何も動かさない。判定は scripts/ci/stg_changes.sh。手動実行・判定不能のときは両方動かす。
                                    テストは同じ push の ci.yml が行うので、ここでは行わない）
                                    build-ui          画面が standalone でビルドできる（STG バッジつき）
                                    check-jobs-image  Jobs のイメージを作り、全ジョブの実行モジュールを読み込めるか（push しない・GCP に触れない）
                                    deploy（環境 stg） WIF で GCP に認証（鍵なし）
                                         build-api-image  API のイメージを作って push
                                         api-revision     通信 0% のタグ付きリビジョン rstg（stg だけテストユーザー test/test が有効）
                                         smoke            鍵なし 401・鍵あり 200・ハートビート（失敗なら失敗のまま）
                                    publish-stg-ui    build-ui が通った後だけ、画面を ops/releases/ui-stg/ に公開（API の deploy は待たない）
        stg ──マージ──▶ master ──push──▶ promote.yml
                                    ① verify-stg        このコミットが stg に含まれること（含まれなければ止める）
                                    ② test + build-ui
                                    ③ 【承認】           Environment「production」の Required reviewers が承認するまで ④ は始まらない
                                    ④ deploy（1 つの job。途中で承認待ちにならない）
                                         build-images → api-revision（rc<コミット8桁>）→ smoke → promote（通信を 100% に）
                                         → jobs（ENABLE_JOBS_DEPLOY=true のときだけ。JOBS_PARALLEL=1 の間は並走=影モード）
                                         → 画面を GCS の ops/releases/ui/ に公開（latest.json を最後に書く）
```

- **スモークに失敗したら通信は切り替わらず、以降（Jobs・画面の公開）も進まない。**
- **取得（スクレイピング）は CI で一切しない。**
- **当てはめ済みの係数・重み（`config/fitted_artifacts.txt`）は git に無い。** API イメージのビルド直前に、`deploy_gcp.sh` の `fetch_fitted` が GCS（`models/fitted_artifacts/`）の `config/` 配下だけを検証つきで取り、`./config` に重ねてからビルドする（約 6MB。血統集計 `data/` と `models/` は入れない）。取れなかったとき、stg は警告して係数なしで続行し、prod は中止する。ホストで係数を更新したら `fitted_artifacts.sh pack` → `model_publish publish --component fitted_artifacts --from <dir>` で GCS に上げ、次のデプロイで入る。Jobs のイメージは実行時取得。
- マージは **マージコミットか fast-forward**（スカッシュ不可）。スカッシュすると「stg に含まれる」ことを確認できず、`verify-stg` で止まる。
- Jobs の更新は、Cloud Shell でキュー・Run Job・IAM を整えるまでは無効（`ENABLE_JOBS_DEPLOY` が `true` でない）。完全移行（並走を終える）ときに `JOBS_PARALLEL=0` にする。
- VPS の nginx の向き先（`switch_api_upstream.sh cloudrun`）は 1 回きりの作業で、この自動化の対象外。

## 2. あなたの作業（この順に。コマンドの値は文書に書かない）

1. **Cloud Shell で WIF を用意する**（鍵は作らない。再実行で、既にある Provider の条件も更新される）
   ```bash
   cd ~/keiba-vpn && git pull
   GCS_BUCKET=<バケット名> bash scripts/gcp/setup_github_wif.sh            # 表示だけ（dry-run が既定）
   GCS_BUCKET=<バケット名> bash scripts/gcp/setup_github_wif.sh --apply    # 内容を確認してから実行
   ```
   - 入れる条件: **リポジトリ `meguram/keiba-vpn`、かつ「`master` + Environment `production`」または「`stg` + Environment `stg`」**。
   - `keiba-deployer` への `roles/iam.workloadIdentityUser`、バケットの **`ops/releases/` だけ**への `roles/storage.objectUser`（IAM Conditions）。
   - 前提: バケットが uniform bucket-level access。確認: `gcloud storage buckets describe gs://<バケット名> --format="value(uniform_bucket_level_access)"` が `True`。
   - 最後に、次で入れる値（`WIF_PROVIDER`、`WIF_SERVICE_ACCOUNT`、`GCS_BUCKET`）が表示される。
2. **デプロイ用 SA の権限を足す**（初期設定の再実行。何度流しても同じ）
   ```bash
   BUCKET=<バケット名> bash scripts/gcp/setup_project.sh
   gcloud secrets add-iam-policy-binding KEIBA_PROXY_KEY --member "serviceAccount:keiba-deployer@<プロジェクト>.iam.gserviceaccount.com" --role roles/secretmanager.secretAccessor
   ```
   （CI のスモークが、鍵つきで API を叩くために、`KEIBA_PROXY_KEY` を読む。Scheduler／Tasks の管理権限もここで付く。）
3. **GitHub: Environment を 2 つ作る**（Settings → Environments → New environment）
   - `stg`: **Deployment branches = Selected branches → `stg` のみ**。承認者は付けない。
   - `production`: **Required reviewers に本人**、**Deployment branches = Selected branches → `master` のみ**。
4. **GitHub: ブランチ保護**（Settings → Branches → Add rule）
   - 対象 `stg` と `master`: 直接 push 禁止（PR 経由）、必須のステータスチェックに CI の `lint-and-test` を指定。一人運用なら PR の承認は任意。
   - マージ方法は **Create a merge commit**（Squash/Rebase は外す）。`master` へは `stg` からだけ PR を作る。
5. **GitHub: Variables を入れる**（Settings → Secrets and variables → Actions → **Variables**）
   | 名前 | 内容 |
   |---|---|
   | `WIF_PROVIDER` | 手順 1 の最後に表示された値 |
   | `WIF_SERVICE_ACCOUNT` | `keiba-deployer@<プロジェクト>.iam.gserviceaccount.com` |
   | `GCP_PROJECT_ID` | GCP のプロジェクト ID（`horse-racing-466415`） |
   | `GCS_BUCKET` | バケット名（`magu-keiba-horse-racing-ai`） |
   | `CLOUD_RUN_REGION` | `asia-northeast1` |
   | `CLOUD_RUN_SERVICE` | `keiba-api` |
   | `SMOKE_RACE_ID` | スモークで使うレース ID（データのある直近のレース。例 `202604040301`） |
   | `ENABLE_JOBS_DEPLOY` | 最初は未設定。Jobs の準備ができたら `true` |
   | `JOBS_PARALLEL` | 未設定のままでよい（既定 `1` = 並走）。完全移行のときに `0` |
   | `NEXT_PUBLIC_PICKER_PAST_DAYS` `NEXT_PUBLIC_ENABLE_FEATURES` | stg の画面と同じ値（`150` / 空でよい。`scripts/ops/vps/build.env.example` 参照） |
   - **`NEXT_PUBLIC_DEFAULT_RACE_DATE` は設定しない。** 週間予測の既定日は、画面が自動で選ぶ（今日が開催日なら今日、無ければ次回の開催の最初の開催日。`frontend/lib/weekDates.ts` の `pickTwoWeeks`／`firstDateOf`）。固定値は dev で実データの日を見るための上書き用で、本番のビルドに入れると開催週が替わるたびに古くなる。
   - **Secrets に入れるものは、ありません**（鍵ファイルを作っていないため）。
6. **アクションを SHA 固定する**（任意だが推奨）
   ```bash
   bash scripts/ci/pin_actions.sh           # 解決結果を表示
   bash scripts/ci/pin_actions.sh --write   # `# PIN` 付きを @<40桁> に書き換え（コミットして push）
   ```
7. **初回は stg から試す**: `stg` ブランチへ develop をマージ → Actions の deploy-stg が緑になること、`rstg` の URL が出ること。次に `stg → master` をマージ → 承認の画面で **Review deployments → production → Approve**。

## 3. すぐに反映したいとき（選択肢）
画面の既定は『次の窓（03:10）』。急ぐ場合の案（いずれも SSH は開けない）。
- **A. 従来どおり `promote.sh`**（ホストから SSH で即時）。ホストを切り離した後は使えない。
- **B. VPS の pull を手動で起動**: ConoHa のコンソール、または `sudo systemctl start keiba-release-pull@ui.service`。
- API は、`master` の承認後すぐ（通信の切り替えまで同じ job で行う）。

## 4. ロールバック
- **API**: `bash scripts/gcp/rollback_cloud_run_api.sh --apply`（1 つ前のリビジョンへ。数秒）。
- **画面（VPS が取り込む前）**: GCS の `ops/releases/ui/latest.json` を `previous.json` の内容に戻す（`gcloud storage cp gs://<バケット>/ops/releases/ui/previous.json gs://<バケット>/ops/releases/ui/latest.json`）。
- **画面（VPS が取り込んだ後）**: VPS で `sudo /opt/keiba-ui/bin/rollback.sh`（直前の版へ）。`pull_release.py` は失敗した版を繰り返さない。
- **Jobs**: `JOBS_PARALLEL` / `ENABLE_JOBS_DEPLOY` を戻して再実行、または前のイメージ（コミット 8 桁のタグ）で `deploy_cloud_run_jobs.sh apply jobs`。

## 5. 費用
公開リポジトリの GitHub Actions は無料。イメージの置き場所（Artifact Registry）は 1 版数百 MB で、古い版は自動掃除（setup_project.sh の設定）。Cloud Run は最小インスタンス 0 のため、反映のたびに増えるのはリビジョンだけ。

## 6. セキュリティ
- **鍵ファイルを作らない・Secrets に GCP の鍵を置かない。** WIF の条件（リポジトリ・`master`+`production`／`stg`+`stg`）に合わない実行は GCP に入れない。
- **フォークからのプルリクエストには秘密を渡さない。** `pull_request_target` は使わない（テストで禁止を検査）。
- `permissions` は既定 `contents: read`。`id-token: write` は反映の job だけ。
- サードパーティの action は **コミット SHA 固定**（手順 6）。更新は手動で確認してから。
- 公開前に `scripts/dev/secret_scan.sh --tracked` を CI のテスト job で実行する。
- 画面の成果物は承認の間に差し替えられないよう、反映の job で sha256 を検算する。
- stg と本番は同じ SA（`keiba-deployer`）。stg の job は `promote` と `jobs` を呼べない作り（`deploy_gcp.sh` が拒否、テストで検査）だが、権限は同じなので、Environment とブランチの条件で分けている。

## 7. よくある失敗
| 症状 | 原因と対処 |
|---|---|
| 認証で `unauthorized_client` | Provider の条件に合っていない（ブランチと Environment の組が違う／リポジトリ名の違い）。手順 1 を再実行して条件を更新。`WIF_PROVIDER` の値も確認 |
| `verify-stg` で止まる | `master` のコミットが `stg` に含まれていない（スカッシュマージ、または `stg` を通さずに `master` へ push）。`develop → stg → master` の順にマージし直す |
| 認証できたが `403` で書けない | 条件付きの権限が `ops/releases/` に限られている。接頭辞が違う／uniform bucket-level access が無効 |
| `build-images` で `Repository ... not found` | Artifact Registry の `keiba-vpn` が無い。Cloud Shell で一度 `deploy_cloud_run_api.sh all --apply` を実行して作る |
| `build-images` のモジュール読み込みで落ちる | Jobs のイメージに足りないライブラリがある（`constraints-jobs.txt` と `requirements.txt` を確認。`python -m src.scripts.ops.autonomy_check deps`） |
| スモークで `SMOKE_RACE_ID` のエラー | Variables に `SMOKE_RACE_ID` を設定する |
| スモークが 401/403 | `keiba-deployer` が `KEIBA_PROXY_KEY` を読めない（手順 2 の 2 本目のコマンド） |
| `build.env` の項目数が 0 | Variables の `NEXT_PUBLIC_*` が未設定 |
| 承認の画面が出ない | Environment `production` に Required reviewers が未設定、または Deployment branches が `master` でない |
| VPS が取り込まない | 窓（03:10）前／同じ版の失敗を記録済み（`--force` で解除）。`python -m src.ops.release_publish status --kind ui` の `applied/vps.json` を見る |

## 8. VPS の取り込みと、stg 環境（stg.megu-keiba.com）をホストから独立させる

**現状の事実**: 本番の画面を VPS に入れる pull 型の仕組み（`pull_release.py`・タイマー・取り込み用の資格情報）は、VPS に**まだ入っていない**。`master` の反映が `ops/releases/ui/` に公開した画面は、これを入れるまで VPS に届かない（それまでは `promote.sh` の SSH で入れていた）。`stg.megu-keiba.com` も今はホスト PC の画面を指している。以下で、どちらも VPS が GCS から自分で取り込む形にする。順番に、VPS は **root** で行う（VPS の SSH ユーザー `deploy` はリリースの導入しか `sudo` できない）。

1. **取り込み用の資格情報**（Cloud Shell）。読み取り専用で、`ops/releases/` と `applied/vps.json` の書き込みだけ。
   ```bash
   cd ~/keiba-vpn && git pull
   GCS_BUCKET=<バケット名> bash scripts/gcp/setup_pull_sa.sh                        # 表示だけ
   GCS_BUCKET=<バケット名> bash scripts/gcp/setup_pull_sa.sh --apply --make-key /tmp/keiba-pull-sa.json
   ```
   鍵（JSON）は VPS が GCP の外にあるため必要。画面に出さず、手元の PC にダウンロード（Cloud Shell のメニュー → ダウンロード）して VPS へ渡し、**渡し終えたら Cloud Shell と手元の両方から消す**。
2. **VPS に仕組みを入れる**（VPS で root）。
   ```bash
   git clone --depth 1 --branch stg --filter=blob:none --sparse https://github.com/meguram/keiba-vpn.git ~/keiba-vpn && cd ~/keiba-vpn   # 既定ブランチ(main)は古く VPS 用スクリプトが無い。master に載った後は --branch master でよい
   git sparse-checkout set scripts/ops/vps
   bash scripts/ops/vps/vps_03_units.sh --dry-run          # 内容の確認（何も変更しない）
   bash scripts/ops/vps/vps_03_units.sh --with-pull-venv
   ```
   続けて、`/etc/keiba-pull/pull.env` に `GCS_BUCKET=<バケット名>` と `GOOGLE_APPLICATION_CREDENTIALS=/etc/keiba-pull/sa.json` を書き、手順 1 の鍵を `/etc/keiba-pull/sa.json`（権限 600）に置く（`install -m 600 鍵.json /etc/keiba-pull/sa.json`）。置いた後に `vps_03_units.sh --with-pull-venv` をもう一度実行すると、取り込みタイマーが有効になる。
   WSL から渡すときは、Cloud Shell の『ダウンロード』は `/home/<ユーザー>/` 配下しか選べないので、先にホームへ移す。手元の保存先は Windows 側（`/mnt/c/Users/<名前>/Downloads/`）。渡したら両方消す（`shred -u`）。
3. **本番の画面を 1 回取り込む**（VPS で root）。以後は毎日 JST 03:10 に自動。
   ```bash
   set -a; . /etc/keiba-pull/pull.env; set +a     # 手で実行するときだけ必要（タイマーは systemd が読む）
   /opt/keiba-pull/venv/bin/python /opt/keiba-ui/bin/pull_release.py --kind ui --force
   curl -s http://127.0.0.1:3000/version.json        # 公開した版（release が keiba-ui-<コミット8桁>-…）になること
   ```
4. **stg 用の 2 つ目の画面を入れる**（VPS で root）。本番の画面・nginx の本番の設定には触れない。
   ```bash
   bash scripts/ops/vps/vps_05_stg_instance.sh --dry-run
   bash scripts/ops/vps/vps_05_stg_instance.sh --cloudrun-host rstg---keiba-api-244891111662.asia-northeast1.run.app --pull-now
   curl -s http://127.0.0.1:3002/version.json        # env が stg の版になること（stg ブランチの push で公開済みのとき）
   ```
   前提: `/etc/nginx/keiba-proxykey.conf` があること（`docs/operations/cloud_run_api.md` の手順 11）。無ければ、Cloud Shell で `(umask 077; gcloud secrets versions access latest --secret=KEIBA_PROXY_KEY > ~/pk.txt)` → 手元経由で VPS の `/tmp/pk.txt` へ → `bash scripts/ops/vps/set_secret.sh /etc/keiba-ui/proxykey.env KEIBA_PROXY_KEY < /tmp/pk.txt && bash scripts/ops/vps/render_nginx_proxykey.sh && shred -u /tmp/pk.txt`（値は画面に出さない。渡したら全コピーを消す）。
5. **Cloudflare で `stg.megu-keiba.com` の転送先を付け替える**（ダッシュボード）。Zero Trust → Networks → Tunnels。
   - ホスト PC のトンネル（`meguai-stg`）の Routes から `stg.megu-keiba.com`（元の転送先 `http://localhost:23200`）を**外す**。`monitor` と `api-internal` は残す。
   - VPS のトンネル（`meguai-prod`）に Public hostname `stg.megu-keiba.com` → Service `HTTP` / `localhost:8081` を**足す**。
   - Access のアプリケーション（`stg.megu-keiba.com` の認証）はそのまま。
6. **確認**: https://stg.megu-keiba.com/ を開く（Access で認証）→ 画面端に **STG** のバッジ。以後、`stg` ブランチへの push で、この画面が最新になる（公開から最大 5 分）。
   戻すとき: 手順 5 を逆にする（ホストのトンネルに `stg.megu-keiba.com` を戻す）。

メモリ: 2 つ目の画面の上限は 160MB（常駐の合計に足しても、2GB の VPS に十分な余裕。`tests/ops/test_vps_units.py` が上限を検査）。
