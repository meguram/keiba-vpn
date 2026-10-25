# publish 専用のサービスアカウント `keiba-publisher`（D6 の権限分離）

目的（`decoupling_plan.md` 9.4）: ホスト（研究専用になる）が運用に渡せるのは **`models/` への公開だけ**にし、運用の正本（`race_predictions`・`ops/`・`state/` など）には書けないようにする。研究の誤操作が公開に影響しない。

## 手順（ユーザー。Cloud Shell。1 つずつ）
1. SA を作る:
   ```bash
   PROJECT=$(gcloud config get-value project); BUCKET=<運用のバケット>
   gcloud iam service-accounts create keiba-publisher --display-name "keiba publisher (models/ only)"
   ```
2. `models/` の接頭辞だけに限った書き込み権限を付ける（上書きと読み取りを含む `objectUser`。`latest.json` の上書きに要る）:
   ```bash
   BUCKET=$BUCKET bash scripts/gcp/make_iam_condition.sh models/ models-write > /tmp/cond_publisher.yaml
   gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
     --member "serviceAccount:keiba-publisher@$PROJECT.iam.gserviceaccount.com" --role roles/storage.objectUser --condition-from-file /tmp/cond_publisher.yaml
   ```
   ※ 読み書き（get/create/delete）は `resource.name`、一覧（list）は `objectListPrefix` で別々に条件が要るので、式に両方を入れている。`models/` の外は一覧も書き込みもできない。
3. 鍵を作ってホストに置く（ホストは GCP の外なので鍵ファイルが要る。置く場所は決め打ち）:
   ```bash
   gcloud iam service-accounts keys create publisher.json --iam-account keiba-publisher@$PROJECT.iam.gserviceaccount.com
   # 手元（ホストの jovyan）へ安全に移して: ~/.config/keiba-gcp/publisher.json  （chmod 600）。Cloud Shell 側の控えは削除
   ```
4. 動作確認（ホスト）: `ssh keiba-remote 'bash -s' < scripts/gcp/publish_models_from_host.sh` に引数なしで流すと「publish 専用の鍵を使います」と出る。公開は `... bash -s -- tl_reference_v5` など（同じ内容なら再アップロードなし）。
5. **確認（越境できないこと）**: ホストで publisher の鍵を使い、`models/` の外に書こうとして 403 になること。
   ```bash
   KEIBA_PUBLISHER_KEY=~/.config/keiba-gcp/publisher.json python3 - <<'PY'
   import json, os
   from google.cloud import storage
   from google.oauth2 import service_account
   k = os.environ["KEIBA_PUBLISHER_KEY"]; c = storage.Client(credentials=service_account.Credentials.from_service_account_file(k), project=json.load(open(k))["project_id"])
   b = c.bucket(os.environ.get("GCS_BUCKET", "<バケット>"))
   try: b.blob("ops/_publisher_test.txt").upload_from_string("x"); print("NG: ops/ に書けてしまった")
   except Exception as e: print("OK: 403 になる", type(e).__name__)
   PY
   ```
6. D6 で、ホストの `.env` から運用の GCS 鍵（書き込み可）・Slack・netkeiba・Cloudflare を外す。読み取りは `keiba-viewer`（読み取り専用）に替える。

## ローテーション
鍵は 90 日で作り直す: `keys create` で新しい鍵を置き、公開を 1 回通してから、古い鍵を `gcloud iam service-accounts keys delete` で無効化する。
