#!/usr/bin/env bash
# VPS → Cloud Run の共有鍵（X-Keiba-Proxy-Key）を、値を画面・ログ・引数・手元のファイルに出さずに作り直す（ホストで、ユーザーの gcloud の認証で流す）。
# 値は Secret Manager（KEIBA_PROXY_KEY_NEXT）にだけ置き、VPS と KEIBA_PROXY_KEY へはパイプで渡す。
# 2026-11-23: nginx -T の出力に鍵が平文で出た（手元の記録だけ。外には出ていない）ため。手順の全体と副作用は docs/operations/edge_fix_20261123.md の「H」。
# 前提: この道具と workflow の WITH_NEXT_KEY が master まで入っていること（gh workflow run は、そのブランチの workflow のファイルを使う）。
#
#   bash scripts/ops/rotate_proxy_key.sh next          # 1) 新しい鍵を作り、KEIBA_PROXY_KEY_NEXT に新しい版として入れる
#   （GitHub の変数 WITH_NEXT_KEY=1 → stg・本番・監視ポータルを配備し直す。Cloud Run が古い鍵と新しい鍵の両方を通す）
#   bash scripts/ops/rotate_proxy_key.sh vps           # 2) 本番・stg・監視ポータルに NEXT が載ったのを確かめてから、VPS の nginx を新しい鍵にする
#   bash scripts/ops/rotate_proxy_key.sh promote-key   # 3) KEIBA_PROXY_KEY に新しい値を入れる（CI のスモーク・監視ポータルが使う）
#   （GitHub の変数 WITH_NEXT_KEY を消して、stg・本番・監視ポータルを配備し直す）
#   bash scripts/ops/rotate_proxy_key.sh retire        # 4) KEIBA_PROXY_KEY と KEIBA_PROXY_KEY_NEXT の古い版を無効にする（最新の 1 版は残す）
#
# 環境変数: PROJECT（既定は gcloud の設定）・REGION（既定 asia-northeast1）・SERVICE（既定 keiba-api）・MONITOR（既定 keiba-monitor）
#           VPS_HOST（既定 keiba-vps。ssh の設定名）・FORCE=1（next の二重実行の防ぎを外す）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-asia-northeast1}"; SERVICE="${SERVICE:-keiba-api}"; MONITOR="${MONITOR:-keiba-monitor}"
VPS_HOST="${VPS_HOST:-keiba-vps}"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || { echo "プロジェクト未設定（PROJECT=… を付ける）" >&2; exit 1; }
export PROJECT

next_value() {   # KEIBA_PROXY_KEY_NEXT の最新の版の値を標準出力へ（パイプの先にだけ渡す）
  gcloud secrets versions access latest --secret KEIBA_PROXY_KEY_NEXT --project "$PROJECT"
}

# 名前の付いた版（リビジョン）の環境に KEIBA_PROXY_KEY_NEXT があるか（値は見ない。名前だけ）
rev_has_next() {
  gcloud run revisions describe "$1" --region "$REGION" --project "$PROJECT" --format=json \
    | python3 -c 'import json,sys; c=json.load(sys.stdin)["spec"]["containers"][0]; sys.exit(0 if any(e.get("name")=="KEIBA_PROXY_KEY_NEXT" for e in c.get("env",[])) else 1)'
}

# 本番で通信を受けている版・stg（rstg のタグ）の版・監視ポータルの通信を受けている版の全部に NEXT があるか
check_next_loaded() {
  local revs r ok=1
  revs="$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format=json \
    | python3 -c 'import json,sys; t=json.load(sys.stdin)["status"].get("traffic",[]); print("\n".join(sorted({x["revisionName"] for x in t if x.get("percent",0)>0 or x.get("tag")=="rstg"})))')"
  revs="$revs
$(gcloud run services describe "$MONITOR" --region "$REGION" --project "$PROJECT" --format=json \
    | python3 -c 'import json,sys; t=json.load(sys.stdin)["status"].get("traffic",[]); print("\n".join(sorted({x["revisionName"] for x in t if x.get("percent",0)>0})))')"
  for r in $revs; do
    if rev_has_next "$r"; then echo "  NEXT あり: $r"; else echo "  NEXT なし: $r" >&2; ok=0; fi
  done
  [ "$ok" = 1 ]
}

case "${1:-}" in
  next)
    if [ "${FORCE:-0}" != 1 ] && [ "$(gh variable get WITH_NEXT_KEY 2>/dev/null || true)" = 1 ]; then
      echo "WITH_NEXT_KEY=1 のまま（作り直しの途中）です。NEXT の最新を変えると台ごとに受ける鍵がばらばらになります。やり直すなら FORCE=1" >&2; exit 1
    fi
    # 40 文字の 16 進（render_nginx_proxykey.sh が拒む文字を含まない）
    head -c 20 /dev/urandom | od -An -tx1 | tr -d ' \n' | bash "$ROOT/scripts/gcp/put_secret.sh" KEIBA_PROXY_KEY_NEXT
    echo "次: gh variable set WITH_NEXT_KEY --body 1 → stg・本番・監視ポータルを配備し直す（docs の H の手順）"
    ;;
  vps)
    echo "== Cloud Run の版に NEXT が載っているか"
    check_next_loaded || { echo "NEXT の無い版があります。VPS は切り替えません（切り替えると、その版は全部 401 になる）" >&2; exit 1; }
    echo "== VPS の道具と権限"
    ssh "$VPS_HOST" 'test -x /opt/keiba-ui/bin/set_secret.sh && test -x /opt/keiba-ui/bin/render_nginx_proxykey.sh && sudo -n true' \
      || { echo "VPS に set_secret.sh・render_nginx_proxykey.sh が無いか、sudo -n が使えません（切り替えていません）" >&2; exit 1; }
    # VPS では値を標準入力で受けてファイルに書き（set_secret.sh）、nginx の include を作り直して reload する（値は表示しない）
    next_value | ssh "$VPS_HOST" 'sudo -n bash /opt/keiba-ui/bin/set_secret.sh /etc/keiba-ui/proxykey.env KEIBA_PROXY_KEY && sudo -n bash /opt/keiba-ui/bin/render_nginx_proxykey.sh'
    echo "== 確かめ（Cookie 付きはキャッシュを通らず上流へ行く。401 なら鍵が通っていない）"
    ssh "$VPS_HOST" 'for p in 8080 8081; do printf "port %s: " $p; curl -s -o /dev/null -w "%{http_code}\n" -H "Host: app.megu-keiba.com" -H "Cookie: keiba_rotate_check=1" "http://127.0.0.1:$p/api/scrape-dates?filter=meeting"; done'
    ;;
  promote-key)
    next_value | bash "$ROOT/scripts/gcp/put_secret.sh" KEIBA_PROXY_KEY
    echo "次: gh variable delete WITH_NEXT_KEY → stg・本番・監視ポータルを配備し直す"
    ;;
  retire)
    bash "$ROOT/scripts/gcp/put_secret.sh" KEIBA_PROXY_KEY --retire
    bash "$ROOT/scripts/gcp/put_secret.sh" KEIBA_PROXY_KEY_NEXT --retire
    ;;
  *)
    sed -n '2,16p' "$0"; exit 2 ;;
esac
