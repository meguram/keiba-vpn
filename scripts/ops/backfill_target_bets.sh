#!/usr/bin/env bash
# stg ホストで実行: 保存済みの予測成績の日次記録に、画面の推奨馬券(target)の仮想購入 bets_t（推奨度つき）を追記する。
#   bash scripts/ops/backfill_target_bets.sh --from 20260101 --to 20261025 [--dry-run] [--ref stg]
# ホストのリポジトリは develop より古く（git pull できない状態のことがある）ので触らない。GitHub の指定ブランチを /tmp に取り込み、
#   データ(data/)・.env・学習済みの係数(config/fitted_artifacts.txt の分)はホストのものをそのまま使う。書き込み先はホストの
#   data/local/prediction_results（と、.env で有効なら GCS の ops/prediction_results）。印・着順・既存の数値は変えない。何度実行しても同じ結果。
# 直近 10 日 → それ以前 の順に、nice 19 で 1 本ずつ。load が高いときは始めない。
set -euo pipefail
REPO="${KEIBA_HOST_REPO:-/home/jovyan/work/keiba-vpn}"; REF="stg"; WORK="${WORK:-/tmp/keiba-backfill-target}"; PY="${PYTHON:-/opt/conda/bin/python3}"
FROM=""; TO=""; DRY=""
while [ $# -gt 0 ]; do case "$1" in --from) FROM="$2"; shift 2;; --to) TO="$2"; shift 2;; --ref) REF="$2"; shift 2;; --dry-run) DRY="--dry-run"; shift;; *) echo "不明な引数: $1" >&2; exit 1;; esac; done
[ -n "$FROM" ] && [ -n "$TO" ] || { echo "--from YYYYMMDD --to YYYYMMDD が要ります" >&2; exit 1; }
[ -d "$REPO/data" ] || { echo "ホストのリポジトリが見つかりません: $REPO" >&2; exit 1; }
load1=$(cut -d' ' -f1 /proc/loadavg); ok=$("$PY" -c "print(1 if $load1 < 14 else 0)")
[ "$ok" = 1 ] || { echo "load が高い($load1)ので中止します。下がってからもう一度" >&2; exit 1; }

echo "== 1/3 コードを取り込む: GitHub の $REF → $WORK"
rm -rf "$WORK"; git clone -q --depth 1 --branch "$REF" https://github.com/meguram/keiba-vpn.git "$WORK"
echo "   $(git -C "$WORK" log --oneline -1)"

echo "== 2/3 データ・設定をつなぐ（ホストのものをそのまま使う）"
rm -rf "$WORK/data"; ln -s "$REPO/data" "$WORK/data"
for f in .env .env.stg .env.prod; do [ -e "$REPO/$f" ] && ln -sf "$REPO/$f" "$WORK/$f" || true; done
n=0
while IFS= read -r line; do
  case "$line" in ''|\#*) continue;; esac
  line="${line%/}"; src="$REPO/$line"
  if [ -e "$src" ]; then mkdir -p "$WORK/$(dirname "$line")"; rm -rf "$WORK/$line"; cp -a "$src" "$WORK/$line"; n=$((n+1)); fi
done < "$WORK/config/fitted_artifacts.txt"
echo "   学習済みの係数 $n 件をコピー"

echo "== 3/3 追記（$FROM〜$TO。直近 10 日 → それ以前。${DRY:-書き込みあり}）"
cd "$WORK"; export PYTHONPATH="$WORK" KEIBA_ENV="${KEIBA_ENV:-stg}" TZ=Asia/Tokyo
RECENT_FROM=$(date -d "${TO:0:4}-${TO:4:2}-${TO:6:2} -9 days" +%Y%m%d)
[ "$RECENT_FROM" \< "$FROM" ] && RECENT_FROM="$FROM"
nice -n 19 "$PY" -m src.scripts.data.prediction_results --from "$RECENT_FROM" --to "$TO" --add-target-bets $DRY
if [ "$RECENT_FROM" \> "$FROM" ]; then
  EARLY_TO=$(date -d "${RECENT_FROM:0:4}-${RECENT_FROM:4:2}-${RECENT_FROM:6:2} -1 day" +%Y%m%d)
  nice -n 19 "$PY" -m src.scripts.data.prediction_results --from "$FROM" --to "$EARLY_TO" --add-target-bets $DRY
fi
echo "完了。確認: https://stg.megu-keiba.com/races?date=$TO"
