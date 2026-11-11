#!/usr/bin/env bash
# 開発 PC の分析コード（走法マップ・Best5・血統・事前計算）をホストへ揃える。ホストは git 管理ではないので、版（ANALYSIS_VERSION）を上げる・分析コードを直したら必ずこれで送る。
#   bash scripts/dev/sync_host_analysis_code.sh          # 差分を表示するだけ
#   bash scripts/dev/sync_host_analysis_code.sh --apply  # 差分を送り、ホストで試し計算（Best5・血統が出るか）まで確かめる
# 送る対象は git で追跡している分析関連のファイルだけ（他セッションの作業中ファイルには触れない）。費用: ホスト内の計算と ssh のみ（GCS・GCP は読まない）。
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=${KEIBA_HOST:-keiba-remote}
REMOTE=${KEIBA_HOST_REPO:-/home/jovyan/work/keiba-vpn}
PY=${KEIBA_HOST_PY:-/opt/conda/bin/python3}
EXTRA="src/pipeline/features/tp28_workout_stats.py src/pipeline/features/tp85_pedprior.py src/scripts/data/tp85_ped_reference.py src/scripts/data/precompute_race_analysis.py"

mapfile -t FILES < <(git ls-files src/research/race src/research/race_analysis $EXTRA | grep '\.py$' | sort -u)
FILES+=(data/meta/trainer_pattern/trainer_pattern_2020_2025.json)   # 調教師×パターンの凍結表（めぐ分析の trainer_pattern が読む）
dirty=$(git status --porcelain -- "${FILES[@]}" | grep -v '^??' || true)
if [ -n "$dirty" ]; then echo "注意: 未コミットの変更を含むファイルを送ります:"; echo "$dirty"; fi

# 手元の md5 をホストで照合し、違う・無いものを出す
diff_list=$(md5sum "${FILES[@]}" | ssh "$HOST" "cd $REMOTE && sudo md5sum -c --quiet 2>&1 | sed -E 's/^md5sum: //; s/: (FAILED.*|No such file.*)//; /WARNING/d' | sort -u" || true)
if [ -z "$diff_list" ]; then echo "ホストの分析コードは手元と一致しています（${#FILES[@]} ファイル）。"; else echo "ホストと違うファイル:"; echo "$diff_list"; fi
[ "${1:-}" = "--apply" ] || exit 0

if [ -n "$diff_list" ]; then
  # shellcheck disable=SC2086
  tar cf - $diff_list | ssh "$HOST" "cd $REMOTE && sudo tar xf - --overwrite --no-same-owner"
  echo "送りました。"
fi
echo "ホストで試し計算します（保存先は /tmp/ra_check。本番の結果は変えません）..."
ssh "$HOST" "cd $REMOTE && sudo rm -rf /tmp/ra_check && sudo env PYTHONPATH=. KEIBA_ENV=stg KEIBA_RACE_ANALYSIS_DIR=/tmp/ra_check $PY -m src.scripts.data.precompute_race_analysis --check > /tmp/ra_check.log 2>&1; rc=\$?; tail -8 /tmp/ra_check.log; exit \$rc" \
  || { echo "NG: 試し計算で Best5・血統が欠けました（上のログを確認）。版を上げたり、cron に任せたりしないでください。"; exit 1; }
echo "OK: 試し計算で Best5・血統の欠けはありません。"
