#!/usr/bin/env bash
# ホスト: 配備中の版で API の応答を手元のデータだけで作り（src/ops/edge_render.py）、VPS のフォルダへ送る。docs/operations/edge_cache_plan_20261121.md §10。
#
#   bash scripts/ops/edge/edge_sync.sh stg            # 作る → 送る → current を付け替える
#   bash scripts/ops/edge/edge_sync.sh prod
#   EDGE_DRY=1 bash scripts/ops/edge/edge_sync.sh stg # 作るだけ（送らない）
#
# - jovyan で流す（root の cron からは runuser -u jovyan -- で）。ssh keiba-vps・gh の認証は jovyan のもの。
# - GCS を読まない（作る処理は data/local/mirror の写しだけを読む）。ホスト → VPS は ssh の tar（GCP の外どうし。課金なし。ホストに rsync は無い）。
# - 版: 最後に成功した配備（prod は promote.yml、stg は deploy-stg.yml）の commit。ブランチの先頭ではない（承認待ちで先に進むため）。
#   その commit を別のツリー（$EDGE_HOME/<env>/src）に取り出して作る。共有の作業ツリー（ホストの cron が動く）は使わない。
# - 止める: config/edge_render.json の enabled を false（作らない）、VPS で touch /srv/keiba-edge/<env>/_ctl/OFF（返さない）。
set -euo pipefail

ENV_NAME="${1:-}"
case "$ENV_NAME" in prod) WF=promote.yml; BR=master ;; stg) WF=deploy-stg.yml; BR=stg ;; *) echo "使い方: edge_sync.sh stg|prod" >&2; exit 2 ;; esac
GH_REPO_NAME="${EDGE_GH_REPO:-meguram/keiba-vpn}"
BASE="${EDGE_BASE:-/home/jovyan/work/keiba-vpn}"           # データの置き場（data/local・data/page_reference）
EDGE_HOME="${EDGE_HOME:-/home/jovyan/work/keiba-edge}"
TREE="$EDGE_HOME/$ENV_NAME/src"
OUT="$EDGE_HOME/out"
REMOTE="${EDGE_REMOTE:-keiba-vps}"
REPO_URL="${EDGE_REPO_URL:-https://github.com/meguram/keiba-vpn.git}"
PY="${EDGE_PYTHON:-/opt/conda/bin/python3}"
GIT=(git -c credential.helper= -c credential.helper='!gh auth git-credential')
[ "$(id -u)" != 0 ] || { echo "root では流さない（jovyan で。手元の状態に root の持ち物を作らない）" >&2; exit 2; }
set -o pipefail
mkdir -p "$EDGE_HOME" "$OUT"
log() { echo "$(date '+%F %T') [edge-sync $ENV_NAME] $*"; }
exec 9>"$EDGE_HOME/.lock-$ENV_NAME"
case " ${EDGE_RENDER_ARGS:-} " in
  *" --full "*) flock -w 1800 9 || { log "前の回が 30 分終わらないので、深夜の全部の作り直しを抜かした"; exit 1; } ;;   # 深夜の回は待つ（黙って抜けない）
  *) flock -n 9 || { log "前の回が動いている"; exit 0; } ;;
esac

# 1) 配備中の版
# cron の cwd はリポジトリの外なので -R で名指しする。そのブランチの、成功した配備の workflow の commit
SHA="${EDGE_SHA:-}"                                   # 試しのときだけ（EDGE_DRY=1 と組みで、版を名指しする）
if [ -z "$SHA" ]; then
  # API に変更が無い push では deploy のジョブが skip されても run は success になる。deploy のジョブが成功した run だけを採る
  for rid in $(gh run list -R "$GH_REPO_NAME" --workflow "$WF" --branch "$BR" --status success -L 10 --json databaseId -q '.[].databaseId' 2>/dev/null || true); do
    if [ "$(gh run view -R "$GH_REPO_NAME" "$rid" --json jobs -q '[.jobs[] | select(.name == "deploy") | .conclusion] | first' 2>/dev/null || true)" = success ]; then
      SHA="$(gh run view -R "$GH_REPO_NAME" "$rid" --json headSha -q .headSha 2>/dev/null || true)"; break
    fi
  done
fi
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { log "配備中の版が分からない（gh run list $WF）。作らない"; exit 1; }

# 2) 別のツリーを、その版に
if [ ! -d "$TREE/.git" ]; then
  mkdir -p "$(dirname "$TREE")"
  "${GIT[@]}" clone -q --no-checkout "$REPO_URL" "$TREE"
fi
if [ "$(git -C "$TREE" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$SHA" ]; then
  git -C "$TREE" cat-file -e "$SHA^{commit}" 2>/dev/null || "${GIT[@]}" -C "$TREE" fetch -q origin
  git -C "$TREE" checkout -q --detach "$SHA"
  log "ツリーを ${SHA:0:12} にした"
fi
rm -f "$TREE/.env" "$TREE/.env.stg" "$TREE/.env.prod"           # ホストの .env を読ませない（作る処理も KEIBA_SKIP_DOTENV=1）
mkdir -p "$TREE/data"
link() {   # link <ホストの実物> <ツリーの中の場所>。実物が無ければ何もしない。空のフォルダ（何かの処理が作った）は消してリンクし直す
  [ -e "$1" ] || return 0
  if [ -L "$2" ]; then return 0; fi
  [ -d "$2" ] && rmdir "$2" 2>/dev/null || true
  [ ! -e "$2" ] || { log "$2 がリンクでない（中身のあるフォルダかファイル）。作らない"; exit 1; }
  mkdir -p "$(dirname "$2")"; ln -s "$1" "$2"
}
for d in local calculated_data cache; do link "$BASE/data/$d" "$TREE/data/$d"; done
# data/page_reference は git の管理（BUNDLE.md）と未追跡の項目が混ざる。未追跡の項目を 1 つずつリンクする（直書きのパスが黙って空を読まない）
for p in "$BASE"/data/page_reference/*; do
  b="$(basename "$p")"
  git -C "$TREE" ls-files --error-unmatch "data/page_reference/$b" >/dev/null 2>&1 && continue
  link "$p" "$TREE/data/page_reference/$b"
done
[ -L "$TREE/data/local" ] || { log "data/local のリンクが無い。作らない"; exit 1; }

# 3) git に無い当て込みの係数（Cloud Run は配備のときに GCS の models/fitted_artifacts/ から重ねる。ホストの同じファイルを重ねる）
while IFS= read -r f; do
  case "$f" in config/*|models/*) ;; *) continue ;; esac
  src="$BASE/$f"
  [ "$f" = config/odds_pred/odds_pred_config.json ] && [ ! -f "$src" ] && src="$BASE/data/local/modeling/odds_pred/odds_pred_config.json"
  if [ -f "$src" ]; then
    mkdir -p "$TREE/$(dirname "$f")"
    # 大きなモデルを毎回読まない: 大きさと更新時刻で比べる（cp -p で更新時刻も写す）
    [ "$(stat -c %s.%Y "$src")" = "$(stat -c %s.%Y "$TREE/$f" 2>/dev/null || true)" ] || cp -p "$src" "$TREE/$f"
  fi
done < <(grep -v '^#' "$TREE/config/fitted_artifacts.txt" 2>/dev/null | sed 's/[[:space:]]*$//')

# 4) 作る（低い優先度）
cd "$TREE"
KEIBA_PAGE_REFERENCE_DIR="$BASE/data/page_reference" PYTHONPATH="$TREE" PYTHONPYCACHEPREFIX="$EDGE_HOME/pyc" \
  nice -n 19 ionice -c3 "$PY" -m src.ops.edge_render render --env "$ENV_NAME" --out "$OUT" --base "$BASE" ${EDGE_RENDER_ARGS:-} \
  > "$EDGE_HOME/render-$ENV_NAME.log" 2>&1 || { log "作る処理が失敗（$EDGE_HOME/render-$ENV_NAME.log）"; tail -n 5 "$EDGE_HOME/render-$ENV_NAME.log"; exit 1; }
tail -n 1 "$EDGE_HOME/render-$ENV_NAME.log"
S12="${SHA:0:12}"
[ -f "$OUT/$ENV_NAME/$S12/_meta/heartbeat.json" ] || { log "作った結果が無い"; exit 1; }
[ "${EDGE_DRY:-0}" = 1 ] && { log "EDGE_DRY=1 なので送らない"; exit 0; }

# 5) 送る（手元の記録で変わったファイルだけ。ssh の tar。VPS では 1 ファイルずつ置き換え）→ current を付け替える → 古い版は 1 つだけ残す
RESYNC=""; case " ${EDGE_RENDER_ARGS:-} " in *" --full "*) RESYNC=--resync ;; esac   # 深夜は全部送り直す（VPS の中身との食い違いを直す）
PYTHONPATH="$TREE" "$PY" -m src.ops.edge_render push --env "$ENV_NAME" --out "$OUT" --sha "$S12" --remote "$REMOTE" $RESYNC
# 手元の古い版も 2 つまで
ls -1dt -- "$OUT/$ENV_NAME"/*/ 2>/dev/null | tail -n +3 | xargs -r rm -rf --
log "送った（${S12}）"
