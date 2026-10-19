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
