#!/usr/bin/env bash
# T-032: /betting が入力したレースIDで最適化 API を呼び、応答を画面の表示形に変換できること
# RACE_ID=<GCS に予測キャッシュがあるレースID> を付けると最適化 API の正常系も検証する（GCS 読み取りのみ）。
# サーバーは起動せず Flask の test_client で /api/v1 を直接呼ぶ。画面の目視確認は README の手順（ブラウザ）で行う。
source "$(dirname "$0")/_lib.sh"
TITLE="T-032 /betting のレースID入力と最適化 API の応答形式"

echo "$TITLE"
( cd "$ROOT/frontend" && [ -d node_modules ] ) || { fail "frontend/node_modules が無い（cd frontend && npm ci）"; finish; exit 1; }
check "型検査: frontend tsc --noEmit" bash -c 'cd frontend && npx --no-install tsc --noEmit -p .'
check "lint: frontend/app/betting/page.tsx" bash -c 'cd frontend && npx --no-install next lint --file app/betting/page.tsx'

PASSWORD="$(env_get DEV_PASSWORD)"
# 署名鍵は検証用の一時値にする（.env が KEIBA_ENV=stg/prod で DEV_SECRET_KEY 未設定でも、この検証は実行できるように）
export DEV_SECRET_KEY="verify-secret-$RANDOM-$RANDOM-0123456789"
DEV_PASSWORD_VALUE="$PASSWORD" py_report "${RACE_ID:-}" <<'PY'
import json, os, sys

race_id = sys.argv[1] if len(sys.argv) > 1 else ""
password = os.environ.get("DEV_PASSWORD_VALUE", "")
if password:
    os.environ["DEV_PASSWORD"] = password
from src.api.flask_app import create_app

client = create_app().test_client()


def ok(desc, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {desc}" + (f" {extra}" if extra else ""))


r = client.post("/api/v1/betting/optimize", json={})
ok("未ログインの最適化は 401", r.status_code == 401, f"→ {r.status_code}")

if not password:
    print("  [SKIP] DEV_PASSWORD が .env に無いためログイン後の検証を省略")
    sys.exit(0)
r = client.post("/api/v1/auth/login", json={"password": password})
ok("開発者ログイン", r.status_code == 200, f"→ {r.status_code}")
if r.status_code != 200:
    sys.exit(1)

r = client.post("/api/v1/betting/optimize", json={})
ok("race_id 無し → 400（画面側の 12 桁検証が先に弾く想定）", r.status_code == 400, f"→ {r.status_code}")
r = client.post("/api/v1/betting/optimize", json={"race_id": "000000000000"})
ok("存在しないレース → 404（画面はエラー文言を表示）", r.status_code == 404, f"→ {r.status_code} {r.get_data(as_text=True)[:80]}")

if not race_id:
    print("  [SKIP] 正常系（RACE_ID 未指定。例: RACE_ID=202606010101 bash docs/todos/verify/T-032_betting.sh）")
    sys.exit(0)
r = client.post("/api/v1/betting/optimize", json={"race_id": race_id, "bankroll": 100000})
if r.status_code != 200:
    ok(f"RACE_ID={race_id} の最適化が 200", False, f"→ {r.status_code} {r.get_data(as_text=True)[:200]}")
    sys.exit(1)
d = json.loads(r.get_data(as_text=True))
cands = d.get("candidates") or []
miss = {"total_bet", "expected_return", "candidates"} - set(d)
bad = [c for c in cands if "bet_type" not in c or "bet_amount" not in c]
print(f"         total_bet={d.get('total_bet')} expected_return={d.get('expected_return')} roi_pct={d.get('roi_pct')} 候補数={len(cands)}")
for c in cands[:5]:
    print(f"         {c.get('bet_type')} {c.get('pair_label') or c.get('horse_names')} 賭け金={c.get('bet_amount')} ev={c.get('ev')} kelly={c.get('kelly_fraction')}")
ok("応答が画面変換（toDisplay）の前提形式: total_bet / expected_return / candidates[].bet_type・bet_amount", not miss and not bad,
   f"欠落キー={sorted(miss)} 不正候補={len(bad)}")
PY
finish
