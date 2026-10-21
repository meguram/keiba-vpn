#!/usr/bin/env bash
# prod（VPS）の版と、直近の昇格履歴、手元の HEAD を並べて表示する。 bash scripts/ops/vps/compare_versions.sh [ssh のホスト名]
SSHH="${1:-keiba-vps}"
echo "== prod (VPS) の版"; ssh "$SSHH" "curl -s -m 5 http://127.0.0.1:3000/version.json; echo; ls -1t /srv/keiba-ui/releases | head -3" 2>&1 | sed 's/^/  /'
echo "== 直近の昇格履歴"; tail -3 data/local/meta/vps_promote_history.jsonl 2>/dev/null | sed 's/^/  /' || echo "  （なし）"
echo "== 手元 HEAD"; git log -1 --format='  %H %s' | cut -c1-120
