#!/usr/bin/env bash
# VPS のメモリの実測を Markdown の表で出す（architecture_target.md 3.2 の想定値を実測で置き換えるため）。読み取り専用。root 不要（cgroup が読めれば値が出る）。
#   bash measure_memory.sh [--label "通常時"|"取得の実行役の動作中"] > mem_<label>.md
# 目標: used <= 600MB（通常時）、MemAvailable >= 1000MB。ユニットごとの現在値・ピーク・MemoryHigh/Max と、プロセス合計 RSS を並べる。
set -u
LABEL="通常時"
[ "${1:-}" = "--label" ] && LABEL="${2:-通常時}"
UNITS="${KEIBA_MEASURE_UNITS:-nginx cloudflared-prod keiba-ui keiba-fetch-agent keiba-heartbeat keiba-release-pull@ui earlyoom}"
mb() { awk -v b="${1:-}" 'BEGIN{ if (b ~ /^[0-9]+$/) printf "%.0f", b/1048576; else printf "-" }'; }
lim() { case "${1:-}" in infinity|""|"[not set]") printf "-";; *) mb "$1";; esac; }
read -r total used free shared buff avail < <(free -m | awk '/^Mem:/ {print $2,$3,$4,$5,$6,$7}')
read -r st stu < <(free -m | awk '/^Swap:/ {print $2,$3}')
echo "### VPS メモリ実測（$LABEL）  $(date '+%Y-%m-%d %H:%M %Z')"
echo
echo "- \`free -m\`: total ${total}MB / **used ${used}MB** / **available ${avail}MB** / swap ${stu}/${st}MB / load $(cut -d' ' -f1-3 /proc/loadavg)"
ok_used="NG"; [ "${used:-9999}" -le 600 ] && ok_used="OK"
ok_av="NG"; [ "${avail:-0}" -ge 1000 ] && ok_av="OK"
echo "- 目標: used ≤ 600MB → **$ok_used** / MemAvailable ≥ 1000MB → **$ok_av** / swap 使用 ≤ 200MB → $([ "${stu:-0}" -le 200 ] && echo OK || echo NG)"
echo
echo "| ユニット | 状態 | 現在(MB) | ピーク(MB) | MemoryHigh | MemoryMax | プロセス RSS 合計(MB) | 再起動 |"
echo "|---|---|---:|---:|---:|---:|---:|---:|"
for u in $UNITS; do
  show="$(systemctl show "$u" -p ActiveState -p MemoryCurrent -p MemoryPeak -p MemoryHigh -p MemoryMax -p NRestarts -p ControlGroup 2>/dev/null)"
  g() { printf '%s\n' "$show" | sed -n "s/^$1=//p" | head -1; }
  cg="$(g ControlGroup)"; peak="$(g MemoryPeak)"
  [ -z "$peak" ] || [ "$peak" = "[not set]" ] && [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.peak" ] && peak="$(cat "/sys/fs/cgroup$cg/memory.peak" 2>/dev/null)"
  rss="-"
  if [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/cgroup.procs" ]; then
    rss="$(for p in $(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null); do awk '/^VmRSS:/ {print $2}' "/proc/$p/status" 2>/dev/null; done | awk '{s+=$1} END{ if (NR) printf "%.0f", s/1024; else printf "-" }')"
  fi
  echo "| $u | $(g ActiveState) | $(mb "$(g MemoryCurrent)") | $(mb "$peak") | $(lim "$(g MemoryHigh)") | $(lim "$(g MemoryMax)") | $rss | $(g NRestarts) |"
done
echo
echo "上位プロセス（RSS）:"
echo '```'
ps -eo rss,comm --sort=-rss 2>/dev/null | head -8 | awk 'NR==1 {print "RSS(MB) COMMAND"; next} {printf "%7.0f %s\n", $1/1024, $2}'
echo '```'
