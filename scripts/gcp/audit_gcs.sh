#!/usr/bin/env bash
# GCS バケットの現状を『読み取りだけ』で表にする（何も変更しない）。
#
#   bash scripts/gcp/audit_gcs.sh                  # プロジェクト内の全バケットの一覧
#   bash scripts/gcp/audit_gcs.sh <バケット名>     # 1 つの詳細（リージョン・クラス・版管理・ライフサイクル・権限）
#   DU=1 bash scripts/gcp/audit_gcs.sh <バケット名>  # 使用量も数える（オブジェクト数が多いと時間がかかる。一覧の読み取り課金は微小）
#   PREFIXES=1 DU=1 bash scripts/gcp/audit_gcs.sh <バケット名>   # 先頭の接頭辞ごとの容量
#
# 判断の目安は docs/operations/gcp_setup_runbook.md 6 章（東京 asia-northeast1 か、Standard か、ライフサイクルの有無）。
set -uo pipefail

BUCKET="${1:-${BUCKET:-}}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || { echo "プロジェクト未設定: gcloud config set project <ID>" >&2; exit 1; }

if [ -z "$BUCKET" ]; then
  echo "== プロジェクト $PROJECT のバケット一覧"
  gcloud storage buckets list --project "$PROJECT" \
    --format="table(name,location,location_type,default_storage_class,versioning_enabled,uniform_bucket_level_access)"
  echo "（詳細: bash scripts/gcp/audit_gcs.sh <バケット名>）"
  exit 0
fi

command -v python3 >/dev/null || { echo "python3 が必要です" >&2; exit 1; }
JSON="$(gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" --format=json)" || { echo "バケットを読めません: $BUCKET" >&2; exit 1; }
JSON="$JSON" python3 - "$BUCKET" <<'PY'
import json, os, sys
d = json.loads(os.environ["JSON"])
loc = d.get("location", "?")
rows = [
    ("リージョン", f"{loc}（{d.get('location_type', '?')}）", "東京は ASIA-NORTHEAST1。違えば転送費・遅延が増える" if loc.upper() != "ASIA-NORTHEAST1" else "OK（東京）"),
    ("ストレージクラス", d.get("default_storage_class", "?"), "Standard 以外なら取り出し課金に注意" if d.get("default_storage_class") != "STANDARD" else "OK"),
    ("バージョニング", str(bool(d.get("versioning_enabled"))), "有効だと古い版の分も保存料金。要否を決める"),
    ("一様なアクセス制御", str(bool(d.get("uniform_bucket_level_access"))), "True 推奨"),
    ("公開アクセス防止", str(d.get("public_access_prevention", "?")), "enforced 推奨"),
    ("ライフサイクル", json.dumps((d.get("lifecycle_config") or d.get("lifecycle") or {}).get("rule", "なし"), ensure_ascii=False)[:300], "古い生データの Nearline 化・削除はここで設定"),
    ("保持ポリシー", json.dumps(d.get("retention_policy") or "なし", ensure_ascii=False), ""),
    ("作成日", str(d.get("creation_time", "?")), ""),
    ("ラベル", json.dumps(d.get("labels") or {}, ensure_ascii=False), ""),
]
w = max(len(r[0]) for r in rows)
print(f"== gs://{sys.argv[1]} の設定")
for k, v, note in rows:
    print(f"  {k.ljust(w)} : {v}" + (f"   ← {note}" if note else ""))
PY
echo "== IAM（誰が何をできるか）"
gcloud storage buckets get-iam-policy "gs://$BUCKET" --project "$PROJECT" --format="table(bindings.role,bindings.members.flatten())" 2>/dev/null \
  || gcloud storage buckets get-iam-policy "gs://$BUCKET" --project "$PROJECT"
if [ "${DU:-}" = "1" ]; then
  echo "== 使用量（読み取りのみ）"
  gcloud storage du --summarize --readable-sizes "gs://$BUCKET"
  if [ "${PREFIXES:-}" = "1" ]; then
    echo "-- 先頭の接頭辞ごと"
    gcloud storage du --summarize --readable-sizes "gs://$BUCKET/*" 2>/dev/null | sort -h | tail -40
  fi
else
  echo "（使用量は DU=1 を付けて再実行）"
fi
