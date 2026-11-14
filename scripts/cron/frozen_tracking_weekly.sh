#!/usr/bin/env bash
# 凍結した馬券候補（tp-59・tp-62）の前向き追跡を、毎開催週の確定後（月曜朝）に 1 回回す。
# 判定対象は凍結日（2026-10-22）以降の確定したレース。履歴は追記のみ（data/local/meta/frozen_tracking/ledger.jsonl）で、同じ日を何度回しても変わらない（冪等）。
# 追跡は読み取りと data/local への書き込みのみ（pre オッズの読み出しと、最後の pre_odds_table だけが race_odds の文書を GCS から少量読む）。DB/netkeiba へはアクセスしない。判定結果は報告専用で、モデルや閾値の選択・学習には使わない。
# 止める: KEIBA_FROZEN_TRACKING=0 を付けて実行（または crontab の該当行 # KEIBA-FROZEN-TRACKING を外す）。
# 前提: export_tables_daily（毎日 07:00 JST）で race_*_flat が更新済み。drift_monitor_weekly.sh（月曜 10:00 JST）の後に回す。
# Usage: bash scripts/cron/frozen_tracking_weekly.sh [PROJECT_DIR]
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
if [ "${KEIBA_FROZEN_TRACKING:-1}" = "0" ]; then echo "frozen tracking: 停止中（KEIBA_FROZEN_TRACKING=0）"; exit 0; fi
export TZ="${TZ:-Asia/Tokyo}" PYTHONPATH=. KEIBA_ENV="${KEIBA_ENV:-stg}"
PY="${PYTHON:-/opt/conda/bin/python3}"; [ -x "$PY" ] || PY="$(command -v python3)"
# 二重起動の防止（前の週の実行が長引いた場合など）
LOCK="${KEIBA_FROZEN_TRACKING_LOCK:-/tmp/keiba_frozen_tracking.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "frozen tracking: 別のプロセスが実行中のため中止"; exit 0; }
# ホストの負荷が高い間は待つ（最大 30 分。重い処理を優先）
waited=0
while [ "$(awk '{print int($1)}' /proc/loadavg)" -gt 14 ] && [ "$waited" -lt 1800 ]; do sleep 60; waited=$((waited + 60)); done
echo "frozen tracking: 開始 $(date '+%F %T') (負荷待ち ${waited}s)"
nice -n 19 "$PY" -m src.scripts.data.frozen_tracking track --from 20261022 --models "${KEIBA_FROZEN_MODELS:-v4,v5}" --variants "${KEIBA_FROZEN_VARIANTS:-final,pre}"
rc=$?
echo "frozen tracking: rc=${rc} 終了 $(date '+%F %T')"
# 購入時点のオッズ（t60〜t02）の実測表を更新する（ペルソナ 1 の検証用。失敗しても追跡の終了コードには影響させない）。
# 読むのは結果が確定した新しいレースの race_odds 文書だけ（週 約 70 件・約 1 MB、スナップショット無しの再試行を含めても月 1 円未満。GCS の一覧は取らない）。
nice -n 19 "$PY" -m src.scripts.data.pre_odds_table || echo "pre_odds_table: 失敗（次回に回す）"
# 買い目の決まりの前向き追跡（docs/operations/rule_tracking.md）。上の pre_odds_table の手元の表を読むので、その後に回す。
# 読むのは手元の予測・race_result_flat・pre_odds.parquet だけ（GCS・GCP は読まない。費用 0 円）。失敗しても追跡の終了コードには影響させない。
# 止める: KEIBA_RULE_TRACKING=0
if [ "${KEIBA_RULE_TRACKING:-1}" != "0" ]; then
  nice -n 19 "$PY" -m src.scripts.data.rule_tracking track
  rrc=$?
  if [ "$rrc" = 3 ]; then
    echo "rule_tracking: 停止（事前登録の決まり・閾値が変わっている。直すまで毎週止まる。docs/operations/rule_tracking.md）" >&2
  elif [ "$rrc" != 0 ]; then
    echo "rule_tracking: 失敗 rc=${rrc}（次回に回す）"
  fi
fi
exit "$rc"
