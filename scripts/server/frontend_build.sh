# shellcheck shell=bash
# Next.js 本番ビルド（next build + next start）の「要否判定」と環境シグネチャ。
# server_watchdog.sh から source される。単体でも source してテストできる（副作用なし）。
#
# 方針: distDir に BUILD_ID が無い／入力ファイルが BUILD_ID より新しい／ビルド時の環境が変わった → 再ビルド。
# NEXT_PUBLIC_* と KEIBA_API_URL / KEIBA_FASTAPI_URL / KEIBA_ENV はビルド時に固定されるため、シグネチャに含める。

# 入力として見るパス（frontend/ からの相対。存在するものだけ）
FRONTEND_BUILD_INPUTS=(app components lib styles public hooks middleware.ts next.config.js package.json package-lock.json tsconfig.json postcss.config.js tailwind.config.js ../config/ports.env)

# ビルド時に固定される環境のシグネチャ（1 行の文字列）
frontend_env_signature() {
    local k out=""
    for k in KEIBA_ENV KEIBA_API_URL KEIBA_FASTAPI_URL NODE_ENV; do
        out+="${k}=${!k-}|"
    done
    # NEXT_PUBLIC_* は値ごと（名前順）
    local names
    names=$(compgen -v | grep '^NEXT_PUBLIC_' | LC_ALL=C sort || true)
    for k in $names; do
        out+="${k}=${!k-}|"
    done
    printf '%s' "$out"
}

# 再ビルドが必要なら 0（真）、不要なら 1 を返し、理由を stdout に 1 行出す。
#   frontend_build_needed <frontend_dir> <dist_dir 名> <env シグネチャ>
frontend_build_needed() {
    local fdir="$1" dist="$2" sig="$3"
    local dist_path="${fdir}/${dist}"
    if [ ! -f "${dist_path}/BUILD_ID" ]; then
        echo "BUILD_ID が無い（未ビルドまたは開発モードの残骸）"
        return 0
    fi
    if [ ! -f "${dist_path}/.build-env" ] || [ "$(cat "${dist_path}/.build-env" 2>/dev/null)" != "$sig" ]; then
        echo "ビルド時の環境（API の URL・NEXT_PUBLIC_* など）が前回と違う"
        return 0
    fi
    local p newer
    for p in "${FRONTEND_BUILD_INPUTS[@]}"; do
        [ -e "${fdir}/${p}" ] || continue
        newer=$(find "${fdir}/${p}" -type f -newer "${dist_path}/BUILD_ID" -print -quit 2>/dev/null || true)
        if [ -n "$newer" ]; then
            echo "ソースがビルドより新しい: ${newer#"${fdir}"/}"
            return 0
        fi
    done
    return 1
}

# frontend/ 直下の .next-* だけを消す（変数が空・想定外のパスなら何もしない）
frontend_remove_dist() {
    local target="$1"
    case "$target" in
        */frontend/.next-*) ;;
        *) echo "frontend_remove_dist: 想定外のパスのため削除しません: ${target}" >&2; return 1 ;;
    esac
    [ -e "$target" ] || return 0
    rm -rf -- "$target"
}

# 1 分平均の負荷（/proc/loadavg の先頭）
frontend_current_load1() {
    awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0
}

# load1 がコア数×係数を超えていれば 0（真）
#   frontend_load_too_high <load1> <cores> [factor=0.8]
frontend_load_too_high() {
    awk -v l="$1" -v c="$2" -v f="${3:-0.8}" 'BEGIN { exit !(l > c * f) }'
}

# 単体実行: 稼働中のサーバを止めずに別ディレクトリへ本番ビルドする（切替は --switch-frontend）
#   bash scripts/server/frontend_build.sh [stg|prod]   強制: KEIBA_FORCE_BUILD=1
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    _d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    exec bash "${_d}/server_watchdog.sh" --profile "${1:-stg}" --build-frontend
fi
