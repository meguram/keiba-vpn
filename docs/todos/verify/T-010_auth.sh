#!/usr/bin/env bash
# T-010: FastAPI 旧ルート（/api/...）の書き込み系が開発者ログイン必須になったこと／署名鍵の既定値が公開環境で使えないこと
#
# 実サーバーは起動しない。Starlette の TestClient を `with` なしで使うため、アプリの lifespan（スクレイピング
# キューワーカーや定期スレッド）は動かず、AuthMiddleware とルートだけが通る。副作用のあるルートは実行しない:
# ログイン済みの確認には入力検証で弾かれる POST /api/scrape-queue/add（{} → 400）だけを使う。
source "$(dirname "$0")/_lib.sh"
TITLE="T-010 FastAPI の認可と署名鍵"

echo "$TITLE"
check "pytest: 認可判定と署名鍵（tests/api/test_auth_legacy_write_paths.py）" \
  python3 -m pytest tests/api/test_auth_legacy_write_paths.py -q -p no:cacheprovider

check "KEIBA_ENV=prod で DEV_SECRET_KEY 未設定 → 署名鍵の取得が RuntimeError" \
  bash -c 'env -u DEV_SECRET_KEY -u APP_ENV KEIBA_ENV=prod python3 -c "from src.api import auth; auth._get_secret_key()" 2>/dev/null; [ $? -ne 0 ]'
check "KEIBA_ENV=prod でも DEV_SECRET_KEY を設定すれば取得できる" \
  env -u APP_ENV KEIBA_ENV=prod DEV_SECRET_KEY=verify-only-secret-0123456789abcdef \
  python3 -c "from src.api import auth; assert auth._get_secret_key().startswith('verify-only')"

py_report <<'PY'
import hashlib, hmac, os, time

os.environ["DEV_SECRET_KEY"] = "verify-secret-0123456789abcdef"
os.environ["KEIBA_ENV"] = "dev"
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.auth import COOKIE_NAME, _make_token

client = TestClient(app, raise_server_exceptions=False)   # `with` を使わない = lifespan を起動しない


def status(method, path, cookie=None):
    headers = {"Cookie": f"{COOKIE_NAME}={cookie}"} if cookie else {}
    return client.request(method, path, json={}, headers=headers).status_code


def expect(desc, want, got):
    print(f"  [{'PASS' if got == want else 'FAIL'}] {desc} → {got}" + ("" if got == want else f"（期待 {want}）"))


def expect_not(desc, bad, got):
    print(f"  [{'PASS' if got != bad else 'FAIL'}] {desc} → {got}" + ("" if got != bad else f"（{bad} であってはならない）"))


print("-- 未ログイン: 書き込み系は 401")
for m, p in [("POST", "/api/scrape-queue/add"), ("POST", "/api/scrape-queue/kick"), ("POST", "/api/train"),
             ("POST", "/api/odds/train"), ("POST", "/api/betting/optimize")]:
    expect(f"{m} {p}", 401, status(m, p))
print("-- 未ログイン: 読み取りは従来どおり（監視ポータルが Cookie なしで呼ぶ）")
expect("GET /api/health", 200, status("GET", "/api/health"))
expect_not("GET /api/scrape-jobs（401 でないこと）", 401, status("GET", "/api/scrape-jobs"))
print("-- 署名鍵")
valid = _make_token()
payload = f"dev:{int(time.time())}"
forged = payload + ":" + hmac.new(b"keiba-dev-default-secret-2026", payload.encode(), hashlib.sha256).hexdigest()
expect("正しい鍵のログイン済み Cookie: POST /api/scrape-queue/add {} は認可を通り入力検証で 400", 400,
       status("POST", "/api/scrape-queue/add", valid))
expect("既定の公開鍵で偽造した Cookie は 401", 401, status("POST", "/api/scrape-queue/add", forged))
PY
finish
