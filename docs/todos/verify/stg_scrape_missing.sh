#!/usr/bin/env bash
# stg（学習PC）: データチェック → 不足データの特定 → スクレイピング → 再チェック。
#
#   bash docs/todos/verify/stg_scrape_missing.sh                              # ドライラン（何を取得するかを表示するだけ）
#   EXECUTE=1 MAX_JOBS=50 bash docs/todos/verify/stg_scrape_missing.sh        # まず少数で実行
#   EXECUTE=1 ROUNDS=5 YES=1 bash docs/todos/verify/stg_scrape_missing.sh     # チェック→取得→再チェックを最大 5 回（進捗が無ければ自動停止）
#   RESUME=1 CLEAR_PAUSE=1 bash docs/todos/verify/stg_scrape_missing.sh       # アクセス制限で中断した後の再開
#
# 環境変数: EXECUTE / YES / MAX_JOBS / ROUNDS / STATUS(missing|invalid|calendar) / CATEGORY(カンマ区切り) / SINCE / UNTIL /
#           NO_INFERRED=1 / REQUEUE_FAILED=1 / STOP_ON_STATUS(既定 any) / STRICT_NOT_FOUND=1 / STOP_AFTER / RESUME / CLEAR_PAUSE / SKIP_PROBE
#
# 方針: スクレイピング中のエラーには敏感に対応する。「ページが存在しない」以外のエラー（HTTP 400 以上・404 を含む、通信エラー、
#       アクセス制限のページ）は基本的にアクセス制限とみなし、検知したら直ちに全リクエストを止めて実行を終了する。
#       終了前に「実行中ジョブを待機に戻す → アクセス一時停止フラグ → その時点のデータチェック → 再開用の状態と計画を保存」を行う。
#       制限が解けたら RESUME=1 で続きから。
#
# 注意: GCS に書き込む＝stg と prod は同一バケットなので prod が読むデータも変わる。EXECUTE なしなら何も書かない。
# 結果: data/local/meta/data_health/stg/{latest.html, scrape_plan.json, scrape_runs/*.json, access_restriction.json}
source "$(dirname "$0")/_lib.sh"
TITLE="stg: 不足データのスクレイピング（データチェック連携）"
export KEIBA_ENV=stg

args=()
[ "${EXECUTE:-0}" = "1" ] && args+=(--execute)
[ "${YES:-0}" = "1" ] && args+=(--yes)
[ -n "${MAX_JOBS:-}" ] && args+=(--max-jobs "$MAX_JOBS")
[ -n "${ROUNDS:-}" ] && args+=(--rounds "$ROUNDS")
[ -n "${STATUS:-}" ] && args+=(--status "$STATUS")
[ -n "${SINCE:-}" ] && args+=(--since "$SINCE")
[ -n "${UNTIL:-}" ] && args+=(--until "$UNTIL")
[ -n "${STOP_ON_STATUS+x}" ] && args+=(--stop-on-status "$STOP_ON_STATUS")
[ -n "${STOP_AFTER:-}" ] && args+=(--stop-after "$STOP_AFTER")
[ "${NO_INFERRED:-0}" = "1" ] && args+=(--no-inferred)
[ "${REQUEUE_FAILED:-0}" = "1" ] && args+=(--requeue-failed)
[ "${STRICT_NOT_FOUND:-0}" = "1" ] && args+=(--strict-not-found)
[ "${RESUME:-0}" = "1" ] && args+=(--resume)
[ "${CLEAR_PAUSE:-0}" = "1" ] && args+=(--clear-pause)
[ "${SKIP_PROBE:-0}" = "1" ] && args+=(--skip-probe)
if [ -n "${CATEGORY:-}" ]; then
  IFS=',' read -ra cats <<<"$CATEGORY"
  for c in "${cats[@]}"; do args+=(--category "$c"); done
fi

echo "$TITLE  (KEIBA_ENV=stg execute=${EXECUTE:-0} resume=${RESUME:-0})"
LOG="$HOME/keiba-scrape-$(date +%Y%m%d-%H%M%S).log"
echo "  コンソールの全出力（スクレイパーの『取得失敗 […]: スキーマ違反の値』の行を含む）を $LOG にも保存します"
python3 -m src.scripts.scraping.scrape_from_health_plan "${args[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

case "$rc" in
  0) pass "対象範囲のすべてが揃い、スキーマに適合している（不足なし）" ;;
  3) if [ "${EXECUTE:-0}" = "1" ]; then fail "まだ不足が残っている（scrape_runs のログと race_ids/ を確認。進捗が無ければ取得先に存在しない可能性）"
     else info "ドライランです。上の対象を確認し、EXECUTE=1 で実行してください"; pass "ドライラン完了（何も書き込んでいません）"; fi ;;
  4) fail "事前確認で中止（dev/prod・GCS 接続・netkeiba 認証情報・ワーカー稼働中・アクセス制限中のいずれか）" ;;
  5) fail "アクセス制限の疑いで中断。制限が解けたことを確認し、RESUME=1 CLEAR_PAUSE=1 で再開（状態: data/local/meta/data_health/stg/access_restriction.json）" ;;
  6) fail "確認が取れず中止（非対話で実行するなら YES=1）" ;;
  *) fail "スクリプトが失敗（終了コード $rc）" ;;
esac
finish
