# stg と prod の挙動をそろえる設計（監査と検査）

特徴量・モデル・config・フラグを直したとき、**stg と prod に同じ挙動を同時に反映できるか**の監査結果と、そのための検査の仕組み。
移行後の構成（UI=VPS / API=Cloud Run / Jobs=GCP / ホスト=モデリング）は `architecture_target.md`、昇格手順は `vps_ui_deploy.md`、モデル変更の標準手順は `model_release_checklist.md`。

## 1. 要点

- **今の構成では、バックエンド（API・モデル・特徴量・config・フラグ）は 1 か所（ホスト）で、stg の UI も prod の UI（VPS）も同じ API を見る。** そのため API／モデル側の変更は構造上 stg と prod に同時に出る。裏を返すと『stg で先に確認してから prod』が API 側ではできない（UI だけ段階昇格できる）。
- 揃わなくなるのは (a) UI のビルド（stg と prod が別々に作られる）、(b) 移行後に API・Jobs・ホストが別環境になったとき、(c) フラグが `.env` ごとに別管理になったとき。
- そこで **『実効リリース・マニフェスト』** を各環境が返し、`env_parity_check` で差を機械的に検出する。ホストは git の HEAD と実ファイルがずれた継ぎはぎ運用なので、commit ではなく**ファイル内容のハッシュ**で比べる。

## 2. 挙動を決める要素と、stg/prod が分かれうる点（重大度順）

| # | 要素 | 今（UI=VPS、API=ホスト） | 移行後に分かれうる点 | 対策（実装済み／運用） |
|---|---|---|---|---|
| 1 | **API を stg と prod で別系統にするか** | 同一（ホストの 1 つ）。変更は即・両方に出る | Cloud Run を 2 サービスにすると別々に更新される | **API は 1 サービス**にし、新リビジョンをタグ（`stg`）付きで無トラフィックでデプロイ→`parity` 確認→トラフィックを切替（＝昇格）。UI だけが stg/prod 別 |
| 2 | **挙動フラグ**（`KEIBA_PROB_CALIBRATION` `KEIBA_PACE_TYPE` `KEIBA_LAP_FORECAST` `KEIBA_TRACKING_V2` `KEIBA_PACE_COHERENCE` ほか） | ホスト `.env` のみ | ホスト `.env`・VPS `.env`・Cloud Run の環境変数・Jobs の環境変数に**二重管理** | `config/behavior_flags.json`（リポジトリ管理）を正にし、`.env` が食い違うと `flag_drift` で検知。環境ごとに違ってよい名前だけ `env_overridable` / `env_overrides` に宣言 |
| 3 | **モデルバンドル／前段／追い切り基準**（`KEIBA_V3_BUNDLE` `KEIBA_V3_STAGE` `KEIBA_V5_*`、MLflow、GCS `latest.json`） | ホストのローカルパス | Jobs・Cloud Run は GCS `latest.json`、ホストはローカルパス → **別々の版**になりうる | マニフェストの `models`（`manifest.json` の sha256 から作る digest）で比較。配布は `publish_model`（`model_registry`）を唯一の入口にし、`latest.json` を切り替えるだけで昇格・ロールバック |
| 4 | **コード**（`src/api` `src/pipeline/{inference,models,features,betting}` `src/utils`） | ホストの継ぎはぎ（git HEAD と不一致） | Cloud Run のイメージに焼いた版／VPS 実行役の版／ホストの版 | `code.tree`（内容ハッシュ）と `code.dirs`（ディレクトリ別）で比較。イメージは同じコミットから 1 回だけビルドして全環境へ |
| 5 | **config/*.json**（較正・印・閾値・展開・めぐ指数） | ホストにファイル単位で置換 | イメージに焼いた版とホストの版がずれる | `configs.digest`／ファイル別ハッシュ。運用専用の config（メンテ窓・スクレイプ防御・費用・会員名簿）は比較対象外 |
| 6 | **特徴量ストア・serving 状態**（`data/local/features`、低メモリ serving 約 62MB） | ホストのローカル | GCS 正本 vs ホストのコピーの更新タイミング差 | 特徴量の列順・定義は学習側と推論側の `manifest.json` 検証（`model_release_checklist.md`）。データは GCS 正本に固定し、Jobs は GCS から読む |
| 7 | **保存済みの予測**（`race_predictions`、`data/local/inference/*`） | ホストと GCS | 変更後に古いモデルの予測が残り、環境で見え方が違う | 変更後に対象日の予測を再生成（チェックリスト）。予測にはモデル版名を刻む（`model_version`） |
| 8 | **フロントのビルド**（ソース差・`NEXT_PUBLIC_*`） | stg はホストでビルド、prod は `build_release.sh` で別ビルド | stg と prod のビルドが別コミット・別環境変数 | `frontend.source_digest`（`version.json` に入る）と `frontend.public_env` で比較。`promote.sh` が**昇格前検査**で stg のソースと昇格コミットが違えば止める |
| 9 | **キャッシュ**（API 30 秒キャッシュ、Cloudflare、Next.js） | 窓で再起動 | 反映直後の見え方の差 | 反映後に `parity` を窓の後（JST 04:10）に実行。Cloudflare のキャッシュは必要なら Purge |
| 10 | データ（生データ・結果・オッズ） | GCS を stg/prod で共有（同一バケット） | 環境ごとに別バケットにすると差が出る | 共有を維持（`deployment.py` の方針）。dev だけが `data/dev_mock` |

**意図した差**（比べない）: `KEIBA_ENV`、`KEIBA_DEPLOYMENT_LABEL`（STG バッジ）、`KEIBA_PROFILE`、`KEIBA_STG_MOCK_PREDICTIONS`、`KEIBA_TEST_USER`、ホスト名、生成時刻、`NEXT_PUBLIC_ENV_LABEL`、コミット ID（ホストが継ぎはぎのため）。ほかに違ってよいものはフラグ表で宣言する。

## 3. 仕組み

| 部品 | 役割 |
|---|---|
| `src/ops/release_manifest.py` | マニフェストの生成と比較（標準ライブラリのみ。VPS・Cloud Run・ホストで動く）。含めるのは allowlist のみ（ハッシュ・版名・挙動フラグの値。秘密・URL・バケット名は含まない。パス系は末尾名、`KEIBA_MODEL_STORE` は短いハッシュ） |
| `GET /api/ops/release-manifest` | 環境が自分の実効マニフェストを返す。**トークン（`KEIBA_MANIFEST_TOKEN`、ヘッダ `X-Keiba-Manifest-Token`）一致のときだけ**。未設定なら 404（無効）。`?detail=1` でファイル別ハッシュ |
| `version.json`（prod UI） | `build_release.sh` が `source_digest` と `public_env` を追記 |
| `config/behavior_flags.json` + `src/utils/behavior_flags.py` | 挙動フラグの既定（共通）と環境別上書き。明示済みの環境変数は上書きしない。**表が空なら従来どおり**（既定は現状維持） |
| `python -m src.scripts.ops.env_parity_check compare` | stg と prod のマニフェストを取得して差を表示。意図しない差があれば終了コード 1、取得できなければ 2。`--slack` で通知 |
| `python -m src.scripts.ops.env_parity_check promote-precheck` | `promote.sh` が呼ぶ昇格前検査（stg のソースと昇格コミットの frontend が同じか、フラグのずれが無いか） |
| `scripts/cron/env_parity_check.sh` / `install_env_parity_cron.sh` | 毎日 JST 04:10（窓の自動反映の後）に照合し、差があれば Slack。登録はユーザーの承認後（タグ `KEIBA-ENV-PARITY`） |

### 3.1 使い方

```bash
# この環境のマニフェストを見る
python -m src.ops.release_manifest print

# stg と prod を照合（トークンは環境変数。値は表示されない）
python -m src.scripts.ops.env_parity_check compare \
  --stg-api https://<STG_HOST> --prod-api https://app.megu-keiba.com \
  --stg-ui https://<STG_HOST>/version.json --prod-ui https://app.megu-keiba.com/version.json
# ネットワークを使わず、ssh などで取った JSON 同士でも比べられる
python -m src.scripts.ops.env_parity_check compare --stg-file stg.json --prod-file prod.json
```

Cloudflare Access の背後にあるので、機械から呼ぶときはサービストークン（`CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET`）を環境変数で渡す（`api-internal` と同じ方式。値は書かない）。

### 3.2 フラグを `config/` に寄せる手順（二重管理をやめる）

1. ホストで `python -m src.ops.release_manifest suggest-flags` を実行すると、今 `.env` で設定されている挙動フラグから `config/behavior_flags.json` の案が出る。
2. 内容を確認して `defaults` に入れ、コミットする（stg/prod/Cloud Run/Jobs で共通になる）。
3. 環境ごとに違ってよいものだけ `env_overrides.<env>`（その環境の値）か `env_overridable`（`.env` で違ってよい名前）に書く。
4. `.env` から同じ値の行を消す（残っても値が同じなら問題ない。違う値が残ると `flag_drift` で検知される）。

## 4. 同じ窓・同じマニフェストで更新する流れ

1. 変更をコミットし、`stage_deploy create --ref <commit>` で**ホスト（stg）**にステージ（JST 03:05 に自動反映）。
2. JST 04:10 の `env_parity_check` が、反映後の実効マニフェストを照合して結果を Slack に出す。
3. 翌日までに stg で確認（`model_release_checklist.md` の推論一致・見え方）。
4. **同じコミット**で `promote.sh --commit <commit>` を実行（prod の UI へ。前検査が stg のソースとの一致を確認）。
5. 昇格後に `env_parity_check compare` を再実行し、`prod` の `version.json` の `source_digest` が stg と一致していることを確認。

API・モデル・config はホスト（共有）なので手順 1 の時点で prod の API にも効く。**『API は stg で確認してから prod』にしたい変更は、移行後に Cloud Run のタグ付きリビジョン（トラフィック 0）で行う**（`architecture_target.md` 5.6）。

## 5. 決めていただくこと

1. 移行後の API を『1 サービス（共有）＋タグ付きリビジョンで事前確認』にするか、『stg 用と prod 用の 2 サービス』にするか。**推奨は 1 サービス**（費用・保守が少なく、常に同じ挙動）。
2. フラグを `config/behavior_flags.json` に寄せる実施時期（`suggest-flags` の出力を確認してコミットするだけ）。
3. `KEIBA_MANIFEST_TOKEN` の設定（ホスト `.env` とチェックを走らせる場所）と、毎日の照合 cron の登録。

## 6. 環境変数（値は書かない）

| 名前 | 場所 | 用途 |
|---|---|---|
| `KEIBA_MANIFEST_TOKEN` | API を動かす環境（ホスト `.env`、将来は Cloud Run の Secret）／照合を走らせる側 | マニフェスト API の合言葉。未設定なら API は無効（404）。環境ごとに分けるなら `KEIBA_MANIFEST_TOKEN_STG` / `_PROD`（照合側） |
| `CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET` | 照合を走らせる側 | Cloudflare Access のサービストークン（`api-internal` 用と同じ方式） |
| `KEIBA_PARITY_STG_API` / `KEIBA_PARITY_PROD_API` / `KEIBA_PARITY_STG_UI` / `KEIBA_PARITY_PROD_UI` | `scripts/cron/env_parity_check.sh` | 照合先 URL（既定は <STG_HOST> / app.megu-keiba.com。UI は `…/version.json`） |
