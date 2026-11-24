#!/usr/bin/env bash
# 影のランナー（candidate_runner）の材料の予約。開催の暦（祝日・月曜の開催を含む）で動く。登録は install_candidate_rb_live_cron.sh（root の crontab）。
#   eve     … 毎日 18:30 JST。翌日が開催日なら、前段 S3 の保存値（tracking_v2・lap_forecast）と RB を作る（rb_live/eve.sh）。
#               週の最初の開催日の前夜は、めぐ指数の週次が前の週の開催日を書くまで待つ（期限は対象日の 03:00）。
#   morning … 開催日の 07:05〜07:40 JST に 5 分おき。export_tables_daily（07:00）が前の開催日の全レースの結果を表に入れたら、RB を作り直す（rb_live/rb_live.sh）。
#               07:45 に影のランナーが RB を読み始めるので、07:40 より後には作り直さない。export と同じロックを取ってから読む（書き出し中の表を読まない）。
#               作り直しに失敗したら、その日は目印（fail_<YYMMDD>m）を置いて繰り返さない。
# 判断は src/pipeline/inference/candidate_rb_schedule.py（手元のファイルだけ。GCS は読まない・費用 0）。二重起動しない（flock）。
# 置き場（rb_live）か、rb_live.sh が使う道具の作業場所（.claude/worktrees/model-cycles）が無いホストでは、記録して何もしない。
set -u
MODE="${1:?eve|morning}"
PROJECT_DIR="${2:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
PY="${KEIBA_PYTHON:-/opt/conda/bin/python3}"
O="${CANDIDATE_RB_LIVE_DIR:-/home/jovyan/work/retrain_out/cycles16_fix3/rb_live}"
[ -f "$O/eve.sh" ] && [ -f "$O/rb_live.sh" ] || exit 0
log() { echo "[$(TZ=Asia/Tokyo date '+%m-%d %H:%M')] $*" >> "$O/eve.log"; }
if [ ! -d "$PROJECT_DIR/.claude/worktrees/model-cycles/src" ]; then
  log "警告: rb_live.sh の道具の作業場所（.claude/worktrees/model-cycles）が無い。予約（$MODE）を流さない"; exit 1
fi
cd "$PROJECT_DIR" || exit 1
exec 9>"/tmp/keiba-candidate-rb-live.lock"
flock -n 9 || exit 0
export TZ=Asia/Tokyo PYTHONPATH="$PROJECT_DIR"
case "$MODE" in
  eve)
    OUT=$("$PY" -m src.pipeline.inference.candidate_rb_schedule eve 2>>"$O/eve.log") || { log "予約の判断に失敗（eve）"; exit 1; }
    [ -n "$OUT" ] || exit 0
    D=$(echo "$OUT" | cut -f1); W=$(echo "$OUT" | cut -s -f2); DL=$(echo "$OUT" | cut -s -f3)
    if [ -n "$W" ]; then bash "$O/eve.sh" "$D" "$(date -d "$D" +%y%m%d)e" "$W" "$DL"; else bash "$O/eve.sh" "$D" "$(date -d "$D" +%y%m%d)e"; fi
    log "前夜の予約 rc=$?（対象日 $D）"
    ;;
  morning)
    late() { [ "$(date +%H%M)" -ge 0741 ]; }          # 07:45 に影のランナーが RB を読み始める。それより前に作り終える（作り直しは 1 分ほど）
    # export_tables_daily と同じロック。待たずに取る（取れなければ書き出し中なので次の回へ）。export は -n で取るので、こちらが長く持つと書き出しが飛ぶ。
    # 取れたら作り直しの終わりまで持つ（書き出しの途中の表を読まない）。07:05 より後に起きる export は無い前提（07:00 起動）
    exec 8>"/tmp/keiba_export_tables_daily.lock"
    if ! flock -n 8; then
      if late; then log "警告: 朝の RB: export が 07:40 を過ぎても終わらない。作り直さない。影のランナーは StaleInputs で止まる。手で rb_live.sh を流す"
      else log "朝の RB: 待つ（export の書き出し中）"; fi
      exit 0
    fi
    OUT=$("$PY" -m src.pipeline.inference.candidate_rb_schedule morning 2>>"$O/eve.log") || { log "警告: 予約の判断に失敗（morning）"; exit 1; }
    ACT=$(echo "$OUT" | cut -f1); N="$(echo "$OUT" | cut -f2)m"; WHY=$(echo "$OUT" | cut -f3-)
    case "$ACT" in
      skip) exit 0 ;;
      wait)
        if late || [ "$(date +%H%M)" -ge 0740 ]; then log "警告: 朝の RB を作り直せないまま窓が終わる（$WHY）。影のランナーは StaleInputs で止まる。手で rb_live.sh を流す"
        else log "朝の RB: 待つ（$WHY）"; fi
        exit 0 ;;
      rebuild)
        if [ -e "$O/fail_$N" ]; then exit 0; fi
        if late; then log "警告: 朝の RB: 07:40 を過ぎたので作り直さない（$WHY）。影のランナーは StaleInputs で止まる"; exit 0; fi
        log "朝の RB: 作り直す（$WHY）"
        bash "$O/rb_live.sh" "$N" >> "$O/eve_rb_$N.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then date '+%F %T' > "$O/fail_$N"; log "警告: 朝の RB の作り直しに失敗 rc=$rc（$O/eve_rb_$N.log）。今日は繰り返さない。影のランナーは StaleInputs で止まる"; exit 1; fi
        log "朝の RB rc=0 $(tail -n 1 "$O/eve_rb_$N.log")"
        # 古い作り直しを消す（14 日より前。state_live の向け先は残す）。state_*・prep_* は 1 回で約 40MB
        live=$(readlink "$O/state_live")
        find "$O" -maxdepth 1 -type d \( -name 'state_[0-9]*' -o -name 'prep_[0-9]*' \) -mtime +14 ! -name "$live" -exec rm -rf {} + 2>/dev/null
        find "$O" -maxdepth 1 -type f \( -name 'state_[0-9]*.check.txt' -o -name 'eve_rb_[0-9]*.log' -o -name 'eve_phase1_[0-9]*.log' -o -name 'eve_megu_[0-9]*.log' -o -name 'fail_[0-9]*' \) -mtime +14 -delete 2>/dev/null
        ;;
    esac
    ;;
  *) echo "eve|morning" >&2; exit 2 ;;
esac
