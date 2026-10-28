# shellcheck shell=bash
# Next.js 本番ビルド（next build + next start）の「要否判定」と環境シグネチャ。
# server_watchdog.sh から source される。単体でも source してテストできる（副作用なし）。
#
# 方針: distDir に BUILD_ID が無い／入力ファイルが BUILD_ID より新しい／ビルド時の環境が変わった → 再ビルド。
# NEXT_PUBLIC_* と KEIBA_API_URL / KEIBA_FASTAPI_URL / KEIBA_ENV はビルド時に固定されるため、シグネチャに含める。

# 入力として見るパス（frontend/ からの相対。存在するものだけ）
FRONTEND_BUILD_INPUTS=(app components content lib styles public hooks middleware.ts next.config.js package.json package-lock.json tsconfig.json postcss.config.js tailwind.config.js ../config/ports.env)

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

# ── プロセス停止（ss / netstat / lsof に依存しない。/proc だけで動く）──────────────────────────
# ホストには ss / netstat / lsof が無く、親(sh -c / npm)を止めても子の next-server（プロセス名が next-server で
# "-p" を含まない、親が 1 になって残る）が旧ビルドのままポートを掴み続け、新ビルドの切替が失敗した（2026-10-19）。

_fb_log() { if declare -F log >/dev/null 2>&1; then log "$@"; else echo "$@" >&2; fi; }

# 全プロセスの「pid ppid」を 1 行ずつ出す（/proc/<pid>/stat。comm に空白や ")" が入っても最後の ")" で切る）
_fb_pid_ppid_list() {
    local f line rest pid
    for f in /proc/[0-9]*/stat; do
        line=$(cat "$f" 2>/dev/null) || continue
        [ -n "$line" ] || continue
        pid="${f#/proc/}"; pid="${pid%/stat}"
        rest="${line##*) }"                     # "S <ppid> ..."
        rest="${rest#* }"                       # "<ppid> ..."
        echo "${pid} ${rest%% *}"
    done
}

# 指定 PID の子孫（子・孫…）を再帰的に列挙する（自分は含めない）
frontend_descendants() {
    local root="$1" out="" frontier="$1" next pid ppid list
    list=$(_fb_pid_ppid_list)
    while [ -n "$frontier" ]; do
        next=""
        for pid in $frontier; do
            while read -r c p; do
                if [ "$p" = "$pid" ]; then next="$next $c"; fi
            done <<< "$list"
        done
        next="${next# }"
        out="$out $next"
        frontier="$next"
    done
    echo "${out# }" | tr -s ' '
}

# 指定ポートで LISTEN しているソケットの inode（IPv4/IPv6）
frontend_listen_inodes() {
    local port="$1" hp f
    hp=$(printf '%04X' "$port")
    for f in /proc/net/tcp /proc/net/tcp6; do
        [ -r "$f" ] || continue
        awk -v hp="$hp" 'NR > 1 && $4 == "0A" { n = split($2, a, ":"); if (a[n] == hp) print $10 }' "$f"
    done
}

# ポートが LISTEN されていれば 0（真）
frontend_port_in_use() {
    [ -n "$(frontend_listen_inodes "$1")" ]
}

# 指定ポートを LISTEN しているプロセスの PID（root なら全プロセス、一般ユーザーなら自分のプロセスのみ見える）
frontend_listener_pids() {
    local port="$1" inode p
    for inode in $(frontend_listen_inodes "$port"); do
        find /proc -maxdepth 3 -path '/proc/[0-9]*/fd/*' -lname "socket:\[${inode}\]" -print 2>/dev/null \
            | awk -F/ '{print $3}'
    done | sort -un | tr '\n' ' ' | sed 's/ $//'
}

# 起動元がこのリポジトリの frontend か（cwd で判定）
_fb_pid_cwd_is() {
    local pid="$1" want="$2" cwd
    cwd=$(readlink "/proc/${pid}/cwd" 2>/dev/null) || return 1
    [ "$cwd" = "$want" ]
}

# 停止対象 PID を集める。
#   1) PID ファイルのプロセス（npm / next の親。コマンドラインに npm / next / node を含むもののみ）と、その子孫すべて
#   2) ポートを LISTEN しているプロセス（next-server）で、cwd が frontend のもの、と、その子孫
# dev の別プロファイル（別ポート）は、ポートで限定されるので巻き込まない。
frontend_collect_stop_pids() {
    local port="$1" cwd_dir="$2" pid_file="${3:-}" roots="" p out="" r
    if [ -n "$pid_file" ] && [ -f "$pid_file" ]; then
        p=$(tr -dc '0-9' < "$pid_file" 2>/dev/null)
        if [ -n "$p" ] && [ -d "/proc/$p" ] && tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -Eq 'npm|next|node'; then
            roots="$p"
        fi
    fi
    for p in $(frontend_listener_pids "$port"); do
        if _fb_pid_cwd_is "$p" "$cwd_dir"; then
            roots="$roots $p"
        else
            _fb_log "  ポート ${port} を掴むプロセス ${p} は cwd が ${cwd_dir} ではないため止めません"
        fi
    done
    for r in $roots; do
        out="$out $r $(frontend_descendants "$r")"
    done
    echo "$out" | tr ' ' '\n' | grep -E '^[0-9]+$' | grep -vx "$$" | sort -un | tr '\n' ' ' | sed 's/ $//'
}

# 停止して、ポートが空くまで待つ。
#   frontend_stop_all <port> <frontend の絶対パス> [pid_file]
#   TERM → 最大 10 秒待つ → 残っていれば KILL → 最大 5 秒待つ。ポートが空かなければ 1 を返す。
frontend_stop_all() {
    local port="$1" cwd_dir="$2" pid_file="${3:-}" pids p i
    pids=$(frontend_collect_stop_pids "$port" "$cwd_dir" "$pid_file")
    if [ -n "$pids" ]; then
        _fb_log "  フロントエンドのプロセスを停止 (TERM): ${pids}"
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
    fi
    local wait_sec="${KEIBA_STOP_WAIT_SEC:-10}"
    for i in $(seq 1 $((wait_sec * 2))); do
        frontend_port_in_use "$port" || break
        sleep 0.5
    done
    if frontend_port_in_use "$port"; then
        pids=$(frontend_collect_stop_pids "$port" "$cwd_dir" "$pid_file")
        _fb_log "  ${wait_sec} 秒待ってもポート ${port} が空かない — KILL: ${pids:-（対象なし）}"
        for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
        for i in $(seq 1 10); do
            frontend_port_in_use "$port" || break
            sleep 0.5
        done
    fi
    [ -n "$pid_file" ] && rm -f "$pid_file"
    if frontend_port_in_use "$port"; then
        _fb_log "  ERROR: ポート ${port} が空かない（掴んでいるプロセス: $(frontend_listener_pids "$port")）"
        return 1
    fi
    return 0
}

# 起動した next が、期待したビルドで動いているか: ビルド固有の URL が 200 を返し、リスナーが起動した PID の子孫である
#   frontend_verify_build <port> <build_id> <started_pid>
frontend_verify_build() {
    local port="$1" bid="$2" started="$3" code l desc ok=0
    code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "http://127.0.0.1:${port}/_next/static/${bid}/_buildManifest.js" 2>/dev/null || true)
    if [ "$code" != "200" ]; then
        _fb_log "  ビルド確認に失敗: /_next/static/${bid}/_buildManifest.js が HTTP=${code}（別のビルドのサーバが応答している）"
        return 1
    fi
    desc=" $started $(frontend_descendants "$started") "
    for l in $(frontend_listener_pids "$port"); do
        case "$desc" in *" $l "*) ok=1 ;; esac
    done
    if [ "$ok" != "1" ] && [ -n "$(frontend_listener_pids "$port")" ]; then
        _fb_log "  ビルド確認に失敗: ポート ${port} のリスナー($(frontend_listener_pids "$port"))が起動した PID ${started} の子孫ではない（旧プロセスが残っている）"
        return 1
    fi
    return 0
}

# 単体実行: 稼働中のサーバを止めずに別ディレクトリへ本番ビルドする（切替は --switch-frontend）
#   bash scripts/server/frontend_build.sh [stg|prod]   強制: KEIBA_FORCE_BUILD=1
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    _d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    exec bash "${_d}/server_watchdog.sh" --profile "${1:-stg}" --build-frontend
fi
