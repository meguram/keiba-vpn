#!/usr/bin/env bash
# 本番の API（Cloud Run keiba-api）の最大台数だけを、急ぎで変える（例: 開催中に 429 や長い待ちが出たので 1 → 3 台へ戻す）。
# 本番で通信 100% の版と同じ設定（イメージの digest・環境・秘密・資源・起動プローブ）で、最大台数だけを変えた版を作り、通信を移す。
# services update は使わない（最後に作られた版 = stg の版のひな形を使うことがあるため。scripts/gcp/api_revision_clone.py）。
#
#   bash scripts/gcp/set_api_max_instances.sh 3            # 作る版の要点を見せるだけ（変更しない）
#   bash scripts/gcp/set_api_max_instances.sh 3 --apply    # 版を作り（通信 0%）、確かめてから通信を 100% 移す
#
# ホストで root の gcloud の認証を使うときの例:
#   sudo -n env CLOUDSDK_CONFIG=/home/jovyan/.config/gcloud bash scripts/gcp/set_api_max_instances.sh 3 --apply
# 次の promote でも同じ台数にするなら: gh variable set CLOUD_RUN_MAX_INSTANCES --body 3（しないと次の promote で 1 台に戻る）
# イメージは今の本番の版と同じ digest なので、Artifact Registry の live-prod の印がそのまま守る。
set -euo pipefail
N="${1:?最大台数（1〜10）}"; MODE="${2:-}"
PROJECT="${PROJECT_ID:-horse-racing-466415}"; REGION="${REGION:-asia-northeast1}"; SERVICE="${SERVICE:-keiba-api}"
HERE="$(cd "$(dirname "$0")" && pwd)"
case "$N" in [1-9]|10) ;; *) echo "最大台数は 1〜10: $N" >&2; exit 2 ;; esac
G=(gcloud --project "$PROJECT")
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
"${G[@]}" run services describe "$SERVICE" --region "$REGION" --format=json > "$TMP/svc.json"
CUR="$(python3 -c 'import json,sys; sys.path.insert(0, sys.argv[2]); import api_revision_clone as c; print(c.serving_revision(json.load(open(sys.argv[1]))))' "$TMP/svc.json" "$HERE")"
"${G[@]}" run revisions describe "$CUR" --region "$REGION" --format=json > "$TMP/rev.json"
NEW="${SERVICE}-max${N}-$(date +%m%d%H%M%S)"
python3 "$HERE/api_revision_clone.py" "$TMP/svc.json" "$TMP/rev.json" "$N" "$NEW" > "$TMP/new.json"
python3 - "$TMP/rev.json" "$TMP/new.json" <<'EOF'
import json, sys
rev, new = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
t = new["spec"]["template"]; c = t["spec"]["containers"][0]
env = [e["name"] for e in c.get("env", [])]
print(f"今の本番の版: {rev['metadata']['name']}（最大 {rev['metadata']['annotations'].get('autoscaling.knative.dev/maxScale')} 台）")
print(f"作る版: {t['metadata']['name']}・最大 {t['metadata']['annotations']['autoscaling.knative.dev/maxScale']} 台・イメージ {c['image'][-20:]}")
print(f"  資源 {c.get('resources', {}).get('limits')}・同時 {t['spec'].get('containerConcurrency')}・環境 {len(env)} 個（KEIBA_TEST_USER={'あり' if 'KEIBA_TEST_USER' in env else 'なし'}）")
EOF
if [ "$MODE" != "--apply" ]; then echo "（見せるだけ。変えるときは --apply）"; exit 0; fi
"${G[@]}" run services replace "$TMP/new.json" --region "$REGION" >/dev/null
# 作った版が、今の本番の版と同じイメージ・同じ環境の名前で、最大台数だけ違うことを確かめてから通信を移す
"${G[@]}" run revisions describe "$NEW" --region "$REGION" --format=json > "$TMP/made.json"
python3 - "$TMP/rev.json" "$TMP/made.json" "$N" <<'EOF'
import json, sys
a, b, n = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
ca, cb = a["spec"]["containers"][0], b["spec"]["containers"][0]
names = lambda c: sorted(e["name"] for e in c.get("env", []))
ok = (a["status"]["imageDigest"] == b["status"].get("imageDigest") and names(ca) == names(cb)
      and ca.get("resources") == cb.get("resources") and b["metadata"]["annotations"].get("autoscaling.knative.dev/maxScale") == n)
if not ok:
    sys.exit("作った版が今の本番の版と違います。通信は移していません（作った版は 0% のまま。消すなら gcloud run revisions delete）")
print("作った版を確かめました（同じイメージ・環境・資源。最大台数だけ違う）")
EOF
"${G[@]}" run services update-traffic "$SERVICE" --region "$REGION" --to-revisions "${NEW}=100" >/dev/null
echo "通信を ${NEW} に 100% 移しました。戻すときは: ${G[*]} run services update-traffic $SERVICE --region $REGION --to-revisions ${CUR}=100"
