#!/usr/bin/env bash
# stg ホストで実行: 保存済みの予測成績の日次記録に、画面の推奨馬券(target)の仮想購入 bets_t（推奨度つき）を追記する。
#   bash scripts/ops/backfill_target_bets.sh --from 20260101 --to 20261025 [--dry-run] [--sec77] [--ref stg]
#   --sec77: 先に想定オッズを §77 のモデルで作る(odds_sec77_backfill)。sec77 が無い日の bets_t は、検証と基準が違うため集計に入らない
# ホストのリポジトリは develop より古く（git pull できない状態のことがある）ので触らない。GitHub の指定ブランチを /tmp に取り込み、
#   データ(data/)・.env・学習済みの係数(config/fitted_artifacts.txt の分)はホストのものをそのまま使う。書き込み先はホストの
#   data/local/prediction_results（と、.env で有効なら GCS の ops/prediction_results）。印・着順・既存の数値は変えない。何度実行しても同じ結果。
# 直近 10 日 → それ以前 の順に、nice 19 で 1 本ずつ。load が高いときは始めない。
set -euo pipefail
REPO="${KEIBA_HOST_REPO:-/home/jovyan/work/keiba-vpn}"; REF="stg"; WORK="${WORK:-/tmp/keiba-backfill-target}"; PY="${PYTHON:-/opt/conda/bin/python3}"
FROM=""; TO=""; DRY=""; SEC77=""
while [ $# -gt 0 ]; do case "$1" in --from) FROM="$2"; shift 2;; --to) TO="$2"; shift 2;; --ref) REF="$2"; shift 2;; --dry-run) DRY="--dry-run"; shift;; --sec77) SEC77=1; shift;; *) echo "不明な引数: $1" >&2; exit 1;; esac; done
[ -n "$FROM" ] && [ -n "$TO" ] || { echo "--from YYYYMMDD --to YYYYMMDD が要ります" >&2; exit 1; }
[ -d "$REPO/data" ] || { echo "ホストのリポジトリが見つかりません: $REPO" >&2; exit 1; }
load1=$(cut -d' ' -f1 /proc/loadavg); ok=$("$PY" -c "print(1 if $load1 < 14 else 0)")
[ "$ok" = 1 ] || { echo "load が高い($load1)ので中止します。下がってからもう一度" >&2; exit 1; }

# 安全: 消すのは /tmp/keiba-backfill-* の作業ディレクトリだけ。ホストの data/ は、最後に『リンク』を張るだけで、中身を消す・書き換える操作はしない
case "$WORK" in /tmp/keiba-backfill-*) ;; *) echo "WORK は /tmp/keiba-backfill-* にしてください: $WORK" >&2; exit 1;; esac

echo "== 1/3 コードを取り込む: GitHub の $REF → $WORK"
[ -L "$WORK" ] && { echo "WORK がリンクです。中止: $WORK" >&2; exit 1; }
[ -e "$WORK/data" ] && [ -L "$WORK/data" ] && rm -f "$WORK/data"      # 前回のリンクは、リンクだけを消す（中身には触れない）
rm -rf "$WORK"; git clone -q --depth 1 --branch "$REF" https://github.com/meguram/keiba-vpn.git "$WORK"
echo "   $(git -C "$WORK" log --oneline -1)"

echo "== 2/3 学習済みの係数をコピーし、データ・設定をつなぐ（ホストのものをそのまま使う。data/ は触らずリンクだけ）"
n=0
while IFS= read -r line; do
  case "$line" in ''|\#*|data/*) continue;; esac           # data/ の行は、リンクで済むのでコピーしない
  line="${line%/}"; src="$REPO/$line"
  if [ -e "$src" ]; then mkdir -p "$WORK/$(dirname "$line")"; cp -a "$src" "$WORK/$(dirname "$line")/"; n=$((n+1)); fi
done < "$WORK/config/fitted_artifacts.txt"
echo "   学習済みの係数 $n 件をコピー"
for f in .env .env.stg .env.prod; do if [ -e "$REPO/$f" ]; then ln -sf "$REPO/$f" "$WORK/$f"; fi; done
rm -rf "$WORK/data"                       # ここはクローンした実体のディレクトリ（まだリンクではない）
ln -s "$REPO/data" "$WORK/data"

echo "== 3/3 追記（$FROM〜$TO。直近 10 日 → それ以前。${DRY:-書き込みあり}）"
cd "$WORK"; export PYTHONPATH="$WORK" KEIBA_ENV="${KEIBA_ENV:-stg}" TZ=Asia/Tokyo
RECENT_FROM=$(date -d "${TO:0:4}-${TO:4:2}-${TO:6:2} -9 days" +%Y%m%d)
[ "$RECENT_FROM" \< "$FROM" ] && RECENT_FROM="$FROM"
if [ -n "$SEC77" ] && [ -z "$DRY" ]; then       # 検証と同じ想定オッズ(§77)を過去日分まとめて作る（1 本だけ・nice 19・load を見ながら。重い: 1 日あたり約 10 分）
  echo "== 3a/3 想定オッズ(sec77)の作成 $FROM〜$TO"
  KEIBA_INFERENCE_ENGINE=full nice -n 19 "$PY" -m src.scripts.data.odds_sec77_backfill --from "$FROM" --to "$TO" --max-load 12
fi
nice -n 19 "$PY" -m src.scripts.data.prediction_results --from "$RECENT_FROM" --to "$TO" --add-target-bets $DRY
if [ "$RECENT_FROM" \> "$FROM" ]; then
  EARLY_TO=$(date -d "${RECENT_FROM:0:4}-${RECENT_FROM:4:2}-${RECENT_FROM:6:2} -1 day" +%Y%m%d)
  nice -n 19 "$PY" -m src.scripts.data.prediction_results --from "$FROM" --to "$EARLY_TO" --add-target-bets $DRY
fi
echo "完了。確認: https://stg.megu-keiba.com/races?date=$TO"
