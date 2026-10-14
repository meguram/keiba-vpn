#!/bin/bash
# frontend/node_modules を安全に再構築するためのラッパー。
#
# 背景: このリポジトリ (/home/jovyan/work/keiba-vpn) は WSL2 + Docker Desktop 環境下で
# Windows ホストのドライブに 9p プロトコル経由でマウントされている（`mount` で確認可能、
# drvfs/9p, msize=65536）。このマウント上でカレントディレクトリ(cwd)を置いたまま
# `npm install` を実行すると、npm 内部の非同期 I/O のタイミングのずれにより、
# 展開中の一部ファイルが「ちょうど 16384 バイト」で静かに切り詰められることがある
# （node_modules を事前にネイティブ領域へシンボリックリンクしていても、npm 自体が
# シンボリックリンクを実ディレクトリへ置き換えてしまうことがあるため、
# node_modules だけ symlink するのでは不十分）。
#
# 対策: package.json / package-lock.json をコンテナのネイティブファイルシステム
# (overlay, 9p ではない場所) にコピーし、そこで npm install を実行してから、
# frontend/node_modules をそのネイティブディレクトリへのシンボリックリンクに
# 張り替える。これにより npm のプロセス cwd は常にネイティブ領域になる。
#
# 使い方:
#   bash scripts/server/frontend_npm_install.sh
#
# frontend/package.json や package-lock.json を更新した後は、必ずこのスクリプト
# 経由で node_modules を再構築すること（素の `npm install` を frontend/ 直下で
# 直接実行しない）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND_DIR="${REPO_ROOT}/frontend"
NATIVE_BUILD_DIR="/opt/keiba_frontend_build"

echo "[frontend_npm_install] ネイティブ領域: ${NATIVE_BUILD_DIR}"
mkdir -p "${NATIVE_BUILD_DIR}"
cp "${FRONTEND_DIR}/package.json" "${NATIVE_BUILD_DIR}/package.json"
if [ -f "${FRONTEND_DIR}/package-lock.json" ]; then
  cp "${FRONTEND_DIR}/package-lock.json" "${NATIVE_BUILD_DIR}/package-lock.json"
fi

echo "[frontend_npm_install] npm install をネイティブ領域で実行します..."
(cd "${NATIVE_BUILD_DIR}" && npm install)

echo "[frontend_npm_install] 破損ファイル(16384バイトちょうど)の有無を検証します..."
SUSPECT_COUNT=$(find "${NATIVE_BUILD_DIR}/node_modules" -type f -size 16384c 2>/dev/null | wc -l)
if [ "${SUSPECT_COUNT}" -gt 0 ]; then
  echo "[frontend_npm_install][警告] 16384バイトちょうどのファイルが ${SUSPECT_COUNT} 件見つかりました。"
  echo "  すべてが破損とは限りませんが、next の commander バンドルなど主要ファイルを手動確認してください。"
fi

echo "[frontend_npm_install] commander バンドルの読み込みテスト..."
node -e "require('${NATIVE_BUILD_DIR}/node_modules/next/dist/compiled/commander/index.js'); console.log('  OK: next/commander 読み込み成功')"

echo "[frontend_npm_install] frontend/node_modules をシンボリックリンクに張り替えます..."
rm -rf "${FRONTEND_DIR}/node_modules"
ln -s "${NATIVE_BUILD_DIR}/node_modules" "${FRONTEND_DIR}/node_modules"

echo "[frontend_npm_install] 完了: $(readlink -f "${FRONTEND_DIR}/node_modules")"
