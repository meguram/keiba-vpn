#!/usr/bin/env bash
# 公開リポジトリに置かない『当て込まれた係数・重み・集計表』（config/fitted_artifacts.txt）の配布用の束ね／展開。
#   bash scripts/ops/fitted_artifacts.sh list                 対象ファイルを 1 行 1 パスで表示（ディレクトリは配下のファイル）
#   bash scripts/ops/fitted_artifacts.sh pack [出力ディレクトリ]   既定 data/local/modeling/fitted_artifacts_stage に、リポジトリ相対のまま集める
#     → 続けて: python -m src.ops.model_publish publish fitted_artifacts <出力ディレクトリ>   （ホストで。GCS の models/fitted_artifacts/ へ）
#   bash scripts/ops/fitted_artifacts.sh unpack <取得済みディレクトリ>   リポジトリのルートへ重ねて置く（git 管理のファイルは上書きしない）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
LIST="config/fitted_artifacts.txt"

expand() {
  while IFS= read -r l; do
    l="${l%%#*}"; l="$(echo "$l" | xargs)"; [ -z "$l" ] && continue
    if [[ "$l" == */ ]]; then [ -d "$l" ] && find "$l" -type f ! -name .gitignore | sort; else [ -e "$l" ] && echo "$l"; fi
  done < "$LIST"
}

case "${1:-}" in
  list) expand ;;
  pack)
    OUT="${2:-data/local/modeling/fitted_artifacts_stage}"
    rm -rf "$OUT"; mkdir -p "$OUT"
    n=0; while IFS= read -r p; do mkdir -p "$OUT/$(dirname "$p")"; cp -p "$p" "$OUT/$p"; n=$((n+1)); done < <(expand)
    echo "束ねました: $n ファイル → $OUT"
    echo "次: python -m src.ops.model_publish publish fitted_artifacts $OUT" ;;
  unpack)
    SRC="${2:?取得済みディレクトリを指定}"
    n=0; skip=0
    while IFS= read -r -d '' f; do
      rel="${f#"$SRC"/}"
      if git ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then skip=$((skip+1)); continue; fi
      mkdir -p "$(dirname "$rel")"; cp -p "$f" "$rel"; n=$((n+1))
    done < <(find "$SRC" -type f ! -name manifest.json -print0)
    echo "展開しました: $n ファイル（git 管理のため上書きしなかった: $skip）" ;;
  *) echo "使い方: bash scripts/ops/fitted_artifacts.sh list|pack [出力]|unpack <取得済みディレクトリ>" >&2; exit 2 ;;
esac
