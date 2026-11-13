#!/usr/bin/env bash
# =============================================================================
# VPS の root 所有の ops ファイル（温め役・systemd ユニット）を、取り込み済みの取得の配布物から入れ替える（方法 A、2026-11-13）。
# deploy ユーザーが `sudo -n /opt/keiba-ui/bin/install_ops_files.sh <操作>` で使う（sudoers: templates/sudoers-keiba-ops-install）。
#
#   install_ops_files.sh install <版> <名前>   # 配布物 /opt/keiba-fetch/releases/<版>/ の中のファイルを、許可リストの <名前> の分だけ入れる
#   install_ops_files.sh rollback <名前>       # 直前の退避に戻す
#   install_ops_files.sh status [<名前>]       # 入っているファイルの sha256 と、取り込み済みの各版の sha256
#
# 安全の決まり:
# - 置き場所とその親もリンクでなく root 所有で書けないことを確かめる。置き場所・権限・入れた後の操作は、下の許可リストに固定（引数では変えられない）。名前は許可リストのキーだけ。
# - 版は fetch-<commit 8 桁>-<12 桁の日時> の形で、今の版（/opt/keiba-fetch/current）と同じものだけ（古い版へは rollback で戻す）。配布物の版のディレクトリとその親が、リンクでなく root 所有で、
#   group/other から書けないことを毎回確かめる（書ければ中止）。配布物は release-pull（root）が latest.json の sha256 で確かめてから展開したもの。
# - 配布物の中の ops_manifest.json の sha256 と、入れるファイルの sha256 が 1 つでも違えば何もしない。
# - 入れる途中・入れた後の操作で失敗したら、退避から戻して中止する。退避は名前ごとに新しい 5 つまで。
# - ユニットは systemd-analyze verify の出力を見せてから daemon-reload（依存先が未設置の警告があるため止めはしない）。退避は /var/lib/keiba-ops/backup、記録は /var/log/keiba-ops-install.log。
# - GCS には触れない（費用 0）。
# 試験用: root 以外が KEIBA_OPS_TEST_ROOT を指定したときだけ、置き場所をその下に差し替え、所有者の確認を自分の uid にし、
#   systemctl・systemd-analyze を呼ばない（root で動くときは環境変数をすべて無視する）。
# =============================================================================
set -euo pipefail
umask 022
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [ "$(id -u)" = 0 ]; then
    TEST_ROOT=""
else
    TEST_ROOT="${KEIBA_OPS_TEST_ROOT:-}"
    [ -n "$TEST_ROOT" ] || { echo "root で実行してください（sudo -n $0 ...）" >&2; exit 2; }
fi
OWNER_UID="$(id -u)"
RELEASES="$TEST_ROOT/opt/keiba-fetch/releases"
BIN_DIR="$TEST_ROOT/opt/keiba-ui/bin"
UNIT_DIR="$TEST_ROOT/etc/systemd/system"
BACKUP_DIR="$TEST_ROOT/var/lib/keiba-ops/backup"
LOG="$TEST_ROOT/var/log/keiba-ops-install.log"
# 親の確認をどこまで遡るか（本番は / まで。試験は TEST_ROOT まで）
CHECK_TOP="${TEST_ROOT:-/}"

# --- 許可リスト（名前 → 「配布物の中の相対パス|置き場所|権限」をスペース区切り） ---
T=scripts/ops/vps/templates
declare -A FILES=(
    [cache-warm]="scripts/ops/vps/cache_warm.py|$BIN_DIR/cache_warm.py|755"
    [cache-warm-units]="$T/keiba-cache-warm.service|$UNIT_DIR/keiba-cache-warm.service|644 $T/keiba-cache-warm.timer|$UNIT_DIR/keiba-cache-warm.timer|644"
    [sched-status-units]="$T/keiba-sched-status.service|$UNIT_DIR/keiba-sched-status.service|644 $T/keiba-sched-status.timer|$UNIT_DIR/keiba-sched-status.timer|644"
    [fetch-timers]="$T/keiba-fetch-reserve.service|$UNIT_DIR/keiba-fetch-reserve.service|644 $T/keiba-fetch-reserve.timer|$UNIT_DIR/keiba-fetch-reserve.timer|644 $T/keiba-fetch-raceday@.service|$UNIT_DIR/keiba-fetch-raceday@.service|644 $T/keiba-fetch-raceday-tomorrow.timer|$UNIT_DIR/keiba-fetch-raceday-tomorrow.timer|644 $T/keiba-fetch-raceday-today.timer|$UNIT_DIR/keiba-fetch-raceday-today.timer|644 $T/keiba-mirror-vps-push.service|$UNIT_DIR/keiba-mirror-vps-push.service|644 $T/keiba-mirror-vps-push.timer|$UNIT_DIR/keiba-mirror-vps-push.timer|644"
)
# 入れた後の操作（reload = daemon-reload、enable:<timer> = その timer を enable --now）
declare -A AFTER=(
    [cache-warm]=""
    [cache-warm-units]="reload"
    [sched-status-units]="reload enable:keiba-sched-status.timer"
    [fetch-timers]="reload"
)
NAMES="cache-warm cache-warm-units sched-status-units fetch-timers"

die() { echo "中止: $*" >&2; log_line "${OP:-?}" "${VER:--}" "${NAME:--}" "-" "中止: $*"; exit 1; }
log_line() {
    mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\t%s\t%s\tby=%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" "$4" "$5" "${SUDO_USER:-$(id -un)}" >>"$LOG" 2>/dev/null || true
}
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
check_name() { case " $NAMES " in *" $1 "*) ;; *) echo "中止: 許可リストに無い名前: $1（使えるもの: $NAMES）" >&2; exit 2;; esac; }
check_version() { [[ "$1" =~ ^fetch-[0-9a-f]{8}-[0-9]{12}$ ]] || { echo "中止: 版の形が不正: $1" >&2; exit 2; }; }

# パスがリンクでなく、所有者が root（試験では自分）で、group/other から書けないこと
safe_node() {
    local p="$1" uid mode
    [ -L "$p" ] && return 1
    [ -e "$p" ] || return 1
    uid="$(stat -c %u "$p")"; mode="$(stat -c %a "$p")"
    [ "$uid" = 0 ] || [ "$uid" = "$OWNER_UID" ] || return 1
    (( (8#$mode & 8#022) == 0 )) || return 1
    return 0
}
# 版のディレクトリから CHECK_TOP まで、すべて safe_node
check_release_dir() {
    local d="$1"
    [ -d "$d" ] && [ ! -L "$d" ] || die "配布物の版がありません（リンク不可）: $d"
    local cur="$d"
    while :; do
        safe_node "$cur" || die "root 所有でない・リンク・deploy などから書ける場所: $cur"
        [ "$cur" = "$CHECK_TOP" ] || [ "$cur" = "/" ] && break
        cur="$(dirname "$cur")"
    done
}
# 配布物の中のファイル（版のディレクトリの中に収まり、途中にリンクが無いこと）
check_release_file() {
    local rel_dir="$1" rel="$2" cur
    case "$rel" in /*|*..*) die "不正な相対パス: $rel";; esac
    cur="$rel_dir/$rel"
    [ -f "$cur" ] || die "配布物に無いファイル: $rel"
    while [ "$cur" != "$rel_dir" ]; do
        safe_node "$cur" || die "リンク・root 所有でない・書ける: $cur"
        cur="$(dirname "$cur")"
    done
}
manifest_sha() {   # ops_manifest.json から 1 ファイルの sha256（無ければ空）
    python3 -I -c 'import json,sys
d=json.load(open(sys.argv[1]))
v=(d.get("files") or {}).get(sys.argv[2], "")
print(v if isinstance(v,str) else "")' "$1" "$2"
}

take_lock() {   # 入れ替えは同時に 1 つだけ（stderr を差し替えないよう、exec のリダイレクトは {} の中で）
    { exec 9>"${TEST_ROOT:-}/run/keiba-ops-install.lock"; } 2>/dev/null || exec 9>/dev/null
    flock -n 9 || { echo "中止: 別の入れ替えが実行中" >&2; exit 1; }
}
systemctl_() {
    if [ -n "$TEST_ROOT" ]; then
        echo "systemctl $*" >>"$TEST_ROOT/systemctl.calls"
        [ "$1" != "${KEIBA_OPS_TEST_FAIL_SYSTEMCTL:-}" ]     # 試験: 指定した操作だけ失敗させる
    else
        systemctl "$@"
    fi
}
verify_unit() {   # 書式の確認（依存先が未設置のときの警告もあるので、出力を見せるだけで止めない）
    [ -n "$TEST_ROOT" ] && { echo "verify $1" >>"$TEST_ROOT/systemctl.calls"; return 0; }
    local out; out="$(systemd-analyze verify "$1" 2>&1)" || true
    [ -z "$out" ] || { echo "systemd-analyze verify の出力（$1）:"; echo "$out" | head -5; } >&2
    return 0
}
run_after() {
    local name="$1" a
    for a in ${AFTER[$name]}; do
        case "$a" in
            reload) systemctl_ daemon-reload ;;
            enable:*) systemctl_ enable --now "${a#enable:}" ;;
        esac || return 1
    done
}

# 置き場所のディレクトリから CHECK_TOP まで safe_node、置き場所のファイルはリンク不可
check_dest() {
    local dst="$1" cur
    [ -L "$dst" ] && die "置き場所がリンク: $dst"
    cur="$(dirname "$dst")"
    while [ ! -e "$cur" ] && [ ! -L "$cur" ]; do cur="$(dirname "$cur")"; done
    while :; do
        safe_node "$cur" || die "置き場所が root 所有でない・リンク・書ける: $cur"
        [ "$cur" = "$CHECK_TOP" ] || [ "$cur" = "/" ] && break
        cur="$(dirname "$cur")"
    done
}
# 退避から戻す（install の途中の失敗と rollback で使う）。戻した後に、入れる前に無かったユニットは止める
restore_from() {
    local name="$1" bdir="$2" e src dst mode b rc=0
    for e in ${FILES[$name]}; do
        IFS='|' read -r src dst mode <<<"$e"
        b="$bdir/$(basename "$dst")"
        if [ -f "$b" ]; then
            install -m "$mode" "$b" "$dst.ops-new" && { [ -n "$TEST_ROOT" ] || chown root:root "$dst.ops-new"; } && mv -f "$dst.ops-new" "$dst" \
                || { echo "戻せなかった: $dst" >&2; rc=1; }
        elif [ -f "$b.absent" ] && [ -e "$dst" ]; then
            case "$dst" in *.timer|*.service) case "$dst" in *@.service) ;; *) systemctl_ disable --now "$(basename "$dst")" || true;; esac;; esac
            rm -f "$dst" || { echo "消せなかった: $dst" >&2; rc=1; }
        fi
    done
    return "$rc"
}
run_after_rollback() {   # rollback の後: daemon-reload と、まだ置いてある timer だけ有効化し直す
    local name="$1" a
    for a in ${AFTER[$name]}; do
        case "$a" in
            reload) systemctl_ daemon-reload || true ;;
            enable:*) if [ -f "$UNIT_DIR/${a#enable:}" ]; then systemctl_ enable --now "${a#enable:}" || true; fi ;;
        esac
    done
}
abort_restore() {   # install の失敗: 退避から戻して中止。戻せたらその回の退避は消す（中身は今の状態と同じ）。戻せなければ退避を残す
    local name="$1" bdir="$2" why="$3"
    if restore_from "$name" "$bdir"; then
        run_after_rollback "$name"
        rm -rf "$bdir"
        die "$why。退避から戻した"
    fi
    run_after_rollback "$name"
    die "$why。退避から戻しきれなかった（退避 $bdir を残した。rollback $name でもう一度戻す）"
}
prune_backups() {   # 退避は名前ごとに新しい 5 つまで
    local name="$1"
    find "$BACKUP_DIR/$name" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -n -5 | while read -r d; do rm -rf "$d"; done || true
}
current_version() {   # release-pull が切り替えた今の版（/opt/keiba-fetch/current のリンク先の名前）
    local cur="$TEST_ROOT/opt/keiba-fetch/current"
    if [ -L "$cur" ]; then basename "$(readlink -f "$cur")"; fi
}

do_install() {
    local ver="$1" name="$2" rel_dir entries e src dst mode want got stamp bdir cur
    rel_dir="$RELEASES/$ver"
    cur="$(current_version)"
    [ "$ver" = "$cur" ] || die "今の版（/opt/keiba-fetch/current = ${cur:-なし}）以外からは入れない: $ver（古い版へ戻すときは rollback）"
    check_release_dir "$rel_dir"
    check_release_file "$rel_dir" ops_manifest.json
    entries="${FILES[$name]}"
    # 1) 全ファイルを先に確かめる（1 つでも違えば何もしない）
    for e in $entries; do
        IFS='|' read -r src dst mode <<<"$e"
        check_release_file "$rel_dir" "$src"
        check_dest "$dst"
        want="$(manifest_sha "$rel_dir/ops_manifest.json" "$src")" || die "ops_manifest.json が読めない"
        [[ "$want" =~ ^[0-9a-f]{64}$ ]] || die "ops_manifest.json に無い: $src"
        got="$(sha_of "$rel_dir/$src")"
        [ "$want" = "$got" ] || die "sha256 の不一致: $src"
        case "$dst" in *.service|*.timer) verify_unit "$rel_dir/$src";; esac
    done
    # 2) 今のファイルを退避
    stamp="$(date -u +%Y%m%dT%H%M%S%N)"
    bdir="$BACKUP_DIR/$name/$stamp"
    install -d -m 700 "$bdir"
    for e in $entries; do
        IFS='|' read -r src dst mode <<<"$e"
        if [ -f "$dst" ]; then cp -p "$dst" "$bdir/$(basename "$dst")"; else : >"$bdir/$(basename "$dst").absent"; fi
    done
    echo "$ver" >"$bdir/FROM_VERSION"
    # 3) 一時名で入れて mv（途中で失敗したら退避から戻して中止）
    for e in $entries; do
        IFS='|' read -r src dst mode <<<"$e"
        { install -d -m 755 "$(dirname "$dst")" && install -m "$mode" "$rel_dir/$src" "$dst.ops-new" \
            && { [ -n "$TEST_ROOT" ] || chown root:root "$dst.ops-new"; } && mv -f "$dst.ops-new" "$dst"; } \
            || { rm -f "$dst.ops-new"; abort_restore "$name" "$bdir" "入れる途中で失敗: $dst"; }
        log_line install "$ver" "$name" "$(sha_of "$dst")" "ok $(basename "$dst")"
    done
    run_after "$name" || abort_restore "$name" "$bdir" "入れた後の操作（${AFTER[$name]}）に失敗"
    prune_backups "$name"
    log_line install "$ver" "$name" "-" "完了（退避 $bdir）"
    echo "入れました: $name（版 $ver）。退避: $bdir"
}

do_rollback() {
    local name="$1" bdir e src dst mode
    bdir="$(find "$BACKUP_DIR/$name" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
    [ -n "$bdir" ] || { echo "退避がありません: $name（何もしません）"; log_line rollback - "$name" - "退避なし"; return 0; }
    for e in ${FILES[$name]}; do
        IFS='|' read -r src dst mode <<<"$e"
        check_dest "$dst"
    done
    restore_from "$name" "$bdir" || { run_after_rollback "$name"; die "戻しきれなかった（退避 $bdir は残した）"; }
    run_after_rollback "$name"
    for e in ${FILES[$name]}; do
        IFS='|' read -r src dst mode <<<"$e"
        log_line rollback "$(cat "$bdir/FROM_VERSION" 2>/dev/null || echo -)" "$name" "$( [ -f "$dst" ] && sha_of "$dst" || echo absent)" "ok $(basename "$dst")"
    done
    # 使った退避は外す（もう一度 rollback すると、さらに 1 つ前に戻る）
    rm -rf "$bdir"
    echo "戻しました: $name（$bdir の内容）"
}

do_status() {
    local names="${1:-$NAMES}" n e src dst mode v
    for n in $names; do
        echo "== $n"
        for e in ${FILES[$n]}; do
            IFS='|' read -r src dst mode <<<"$e"
            if [ -f "$dst" ]; then echo "  入っている $(sha_of "$dst")  $dst"; else echo "  未設置                                                              $dst"; fi
            for v in "$RELEASES"/fetch-*; do
                [[ "$(basename "$v")" =~ ^fetch-[0-9a-f]{8}-[0-9]{12}$ ]] || continue
                [ -f "$v/$src" ] && echo "    版 $(basename "$v") $(sha_of "$v/$src")"
            done
        done
    done
    [ -f "$LOG" ] && { echo "== 直近の記録（$LOG）"; tail -5 "$LOG"; }
    return 0
}

OP="${1:-}"; VER="-"; NAME="-"
case "$OP" in
    install)
        [ $# = 3 ] || { echo "使い方: $0 install <版> <名前>" >&2; exit 2; }
        VER="$2"; NAME="$3"; check_version "$VER"; check_name "$NAME"
        take_lock
        do_install "$VER" "$NAME" ;;
    rollback)
        [ $# = 2 ] || { echo "使い方: $0 rollback <名前>" >&2; exit 2; }
        NAME="$2"; check_name "$NAME"
        take_lock
        do_rollback "$NAME" ;;
    status)
        [ $# -le 2 ] || { echo "使い方: $0 status [<名前>]" >&2; exit 2; }
        if [ $# = 2 ]; then check_name "$2"; do_status "$2"; else do_status; fi ;;
    *)
        echo "使い方: $0 install <版> <名前> | rollback <名前> | status [<名前>]（名前: $NAMES）" >&2; exit 2 ;;
esac
