#!/usr/bin/env bash
# =============================================================================
# 取得の実行役（keiba-fetch-agent）を、決まった操作だけで扱う（VPS・root で実行）。
#
# deploy ユーザーは sudoers（templates/sudoers-keiba-fetch-ctl）で、このスクリプトだけをパスワードなしで実行できる。
# 置き場は /opt/keiba-ui/bin（root 所有。deploy は書き換えられない）。設置: vps_08_fetch_ctl.sh
#
#   sudo /opt/keiba-ui/bin/fetch_agent_ctl.sh status       # 実行役の状態・入っている版・設定の有無（値は出さない）
#   sudo /opt/keiba-ui/bin/fetch_agent_ctl.sh logs [行数]   # 実行役のログ（既定 50 行、最大 500 行）
#   sudo /opt/keiba-ui/bin/fetch_agent_ctl.sh selfcheck    # 取得に要るものが揃っているか（fetch_child selfcheck）
#   sudo /opt/keiba-ui/bin/fetch_agent_ctl.sh pull         # 公開済みの取得のコードを今すぐ取り込む（窓の外でも）
#   sudo /opt/keiba-ui/bin/fetch_agent_ctl.sh shadow-on    # 影モードを始める（実行役を有効にして起動）
#   sudo /opt/keiba-ui/bin/fetch_agent_ctl.sh shadow-off   # 影モードをやめる（実行役を止めて無効にする）
#
# 秘密（netkeiba のログイン・鍵）の値は表示しない。本番の取得を VPS に任せる指定（KEIBA_FETCH_AGENT_TASKS）は
# ここでは扱わない（あれば shadow-on を断る）。担当の移譲は docs/operations/vps_agent.md 4 章の手順で人が行う。
# =============================================================================
set -euo pipefail

# 置き場はテストのときだけ差し替える（sudo は既定で環境変数を引き継がないので、deploy からは変えられない）
ENV_FILE="${KEIBA_FETCH_ENV_FILE:-/etc/keiba-fetch/fetch.env}"
UNIT=keiba-fetch-agent
CURRENT="${KEIBA_FETCH_CURRENT:-/opt/keiba-fetch/current}"
SA_FILE="${KEIBA_FETCH_SA_FILE:-/etc/keiba-fetch/sa.json}"
REQUIRED_KEYS=(GCS_BUCKET)

die() { echo "fetch_agent_ctl: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "root で実行してください（sudo）"

has_key() { grep -qE "^${1}=" "$ENV_FILE" 2>/dev/null; }
key_value() { grep -E "^${1}=" "$ENV_FILE" 2>/dev/null | tail -n1 | cut -d= -f2- | tr -d '"'"'"' '; }

# 1 行だけ書き換える（無ければ足す）。持ち主・権限はそのまま
set_flag() {
    local key="$1" val="$2" tmp
    [ -f "$ENV_FILE" ] || die "$ENV_FILE がありません（vps_agent.md 4 章の 7）"
    tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
    grep -vE "^${key}=" "$ENV_FILE" > "$tmp" || true
    printf '%s=%s\n' "$key" "$val" >> "$tmp"
    chown --reference="$ENV_FILE" "$tmp"; chmod --reference="$ENV_FILE" "$tmp"
    mv "$tmp" "$ENV_FILE"
}

show_status() {
    echo "unit: $(systemctl is-enabled "$UNIT" 2>&1 || true) / $(systemctl is-active "$UNIT" 2>&1 || true)"
    echo "release: $(readlink "$CURRENT" 2>/dev/null || echo なし)"
    if [ -f "$ENV_FILE" ]; then
        for k in KEIBA_FETCH_AGENT_ENABLED KEIBA_FETCH_SHADOW_ENABLED; do echo "$k=$(key_value "$k" || true)"; done
        for k in "${REQUIRED_KEYS[@]}"; do echo "$k: $(has_key "$k" && echo あり || echo なし)"; done
        echo "netkeiba のログイン（netkeiba_id・netkeiba_pw）: $(grep -cE '^netkeiba_(id|pw)=.+' "$ENV_FILE" || true)/2 行"
        echo "KEIBA_FETCH_AGENT_TASKS: $(has_key KEIBA_FETCH_AGENT_TASKS && echo "あり（$(key_value KEIBA_FETCH_AGENT_TASKS)）" || echo なし)"
    else
        echo "$ENV_FILE: なし"
    fi
    echo "sa.json: $([ -f "$SA_FILE" ] && echo あり || echo なし)"
    if [ -f "$CURRENT/config/fetch_owner.json" ]; then
        echo "agent_tasks: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("agent_tasks") or [])' "$CURRENT/config/fetch_owner.json")"
    fi
    echo "fetch_shadow.json: $([ -f "$CURRENT/config/fetch_shadow.json" ] && echo あり || echo なし)"
}

cmd="${1:-status}"
case "$cmd" in
    status)
        show_status
        ;;
    logs)
        n="${2:-50}"
        [[ "$n" =~ ^[0-9]+$ ]] || die "行数は数字で"
        [ "$n" -le 500 ] || n=500
        journalctl -u "$UNIT" -n "$n" --no-pager
        ;;
    selfcheck)
        sudo -u keibafetch env KEIBA_ENV=prod PYTHONPATH="$CURRENT" /opt/keiba-fetch/venv/bin/python -m src.ops.fetch_child selfcheck
        ;;
    pull)
        # タイマー（keiba-release-pull@.service）と同じ環境で動かす。sudo は環境を引き継がないので、ここで読む
        ( set -a; . "${KEIBA_PULL_ENV_FILE:-/etc/keiba-pull/pull.env}"; set +a
          KEIBA_ENV=prod KEIBA_RUN_LOCATION=vps PYTHONUNBUFFERED=1 /opt/keiba-pull/venv/bin/python /opt/keiba-ui/bin/pull_release.py --kind fetch --force )
        echo "release: $(readlink "$CURRENT" 2>/dev/null || echo なし)"
        ;;
    shadow-on)
        [ -f "$CURRENT/config/fetch_shadow.json" ] || die "取得のコードに fetch_shadow.json がありません（古い版）。先に pull"
        for k in "${REQUIRED_KEYS[@]}"; do has_key "$k" || die "$ENV_FILE に $k がありません"; done
        [ -f "$SA_FILE" ] || die "$SA_FILE がありません"
        if has_key KEIBA_FETCH_AGENT_TASKS && [ -n "$(key_value KEIBA_FETCH_AGENT_TASKS)" ]; then
            die "KEIBA_FETCH_AGENT_TASKS が設定されています（影ではなく本番の取得を VPS が始めます）。人が確認して消してから"
        fi
        set_flag KEIBA_FETCH_AGENT_ENABLED 1
        set_flag KEIBA_FETCH_SHADOW_ENABLED 1
        systemctl enable "$UNIT" >/dev/null
        systemctl restart "$UNIT"
        sleep 5
        show_status
        journalctl -u "$UNIT" -n 20 --no-pager
        ;;
    shadow-off)
        set_flag KEIBA_FETCH_SHADOW_ENABLED 0
        owned="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("agent_tasks") or []))' "$CURRENT/config/fetch_owner.json" 2>/dev/null || echo 0)"
        if [ "$owned" != 0 ] || { has_key KEIBA_FETCH_AGENT_TASKS && [ -n "$(key_value KEIBA_FETCH_AGENT_TASKS)" ]; }; then
            systemctl restart "$UNIT"          # 本番の取得を担当している。影だけやめて、実行役は止めない
        else
            systemctl disable --now "$UNIT" >/dev/null 2>&1 || true
        fi
        show_status
        ;;
    *)
        die "使い方: status | logs [行数] | selfcheck | pull | shadow-on | shadow-off"
        ;;
esac
