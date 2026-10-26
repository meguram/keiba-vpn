#!/usr/bin/env bash
# ホスト（keiba-remote）の実ファイルを、Jobs の入力 state（GCS の state/）へ上げる（種まき）。ホストの古い git は触らず、origin/develop のコードで state_sync を動かす。
#   手元から: ssh keiba-remote 'bash -s -- predict-race rb-state-daily' < scripts/gcp/seed_state_from_host.sh
#   日次更新の Jobs（export-tables-daily feature-accumulate parse-drift-daily px-ridge-window）の種まきもここから。include の {year} などは現在の年で置換する
#   引数なし: 何も上げず、各ジョブの state_in の実サイズだけを表示する（plan）
# 上げるのは job_map の state_in（include で絞った分だけ）。state/ は Jobs 専用の置き場で、ホストの本番データ・画面が読む ops/mirror には触れない。
# 日次で更新する Jobs（export-tables-daily・feature-accumulate・serving-state-precompute・px-ridge-window・rb-state-daily）が ready になるまで、履歴は種まきの時点で止まる。
set -euo pipefail
[ "${1:-}" != "--" ] || shift
REPO="${KEIBA_HOST_REPO:-/home/jovyan/work/keiba-vpn}"; PY="${PYTHON:-/opt/conda/bin/python3}"
NEW="$(mktemp -d /tmp/keiba_seed.XXXXXX)"; trap 'rm -rf "$NEW"' EXIT
cd "$REPO"; sudo -n git fetch -q origin; sudo -n git archive origin/develop src config | tar -x -C "$NEW"
[ ! -f "$REPO/.env" ] || ln -sf "$REPO/.env" "$NEW/.env"
cd "$NEW"
Y="$(TZ=Asia/Tokyo date +%Y)"; PARAMS=(--param "year=$Y" --param "prev_year=$((Y - 1))")
if [ "$#" -eq 0 ]; then
  for j in prediction-results-refresh predict-race day-prep eve-precompute rb-state-daily export-tables-daily feature-accumulate parse-drift-daily px-ridge-window; do echo "== $j (plan)"; PYTHONPATH=. "$PY" -m src.ops.state_sync plan --job "$j" --which in --root "$REPO" "${PARAMS[@]}" 2>&1 | grep -v -i warn; done
  echo "上げるには、ジョブ名を引数に渡してください（例: predict-race rb-state-daily）。"; exit 0
fi
for j in "$@"; do
  echo "== 種まき: $j"; PYTHONPATH=. "$PY" -m src.ops.state_sync push --job "$j" --which in --root "$REPO" "${PARAMS[@]}" 2>&1 | grep -v -i warn
done
