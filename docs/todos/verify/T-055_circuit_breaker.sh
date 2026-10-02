#!/usr/bin/env bash
# T-055: Redis / PostgreSQL の障害時に待たされないこと（サーキットブレーカー＋タイムアウト）
# 障害は「接続は受けるが何も返さない TCP サーバー」をローカルに立てて再現する（外部ネットワーク非依存）。
# 任意: 健全な Redis があれば REDIS_URL（既定 redis://localhost:6379/0）で復旧確認も行う。
source "$(dirname "$0")/_lib.sh"
TITLE="T-055 サーキットブレーカーとタイムアウト"

echo "$TITLE"
check "pytest: ブレーカー本体・Redis キャッシュ・DB オプション（redis が入っていればタイムアウト設定テストも実行）" \
  python3 -m pytest tests/utils/test_circuit_breaker.py tests/api/test_redis_cache_breaker.py tests/db/test_engine_options.py -q -p no:cacheprovider -rs

py_check "Redis 無応答: 失敗は上限時間内、ブレーカー開放後は即時にキャッシュミス" <<'PY'
import os, socket, sys, threading, time

try:
    import redis  # noqa: F401
except ImportError:
    print("         redis パッケージが無い"); sys.exit(3)

# 接続は受けるが応答しないサーバー
srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(16)
port = srv.getsockname()[1]
held = []
threading.Thread(target=lambda: [held.append(srv.accept()[0]) for _ in iter(int, 1)], daemon=True).start()

healthy_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
os.environ["REDIS_URL"] = f"redis://127.0.0.1:{port}/0"
os.environ["REDIS_BREAKER_RECOVERY_SEC"] = "2"
from src.api.cache import redis_cache
from src.api.cache.redis_cache import PredictionCache

c = PredictionCache()
slow = []
for _ in range(3):
    t = time.perf_counter()
    try:
        c.get_prediction("r", "v")
    except Exception:
        pass
    slow.append(time.perf_counter() - t)
print("         失敗 3 回の所要(秒): " + ", ".join(f"{x:.2f}" for x in slow) + "（上限 0.5 秒 + 余裕）")
assert all(x < 1.5 for x in slow), "タイムアウト設定が効いていない"

fast = []
for _ in range(20):
    t = time.perf_counter()
    assert c.get_prediction("r", "v") is None
    fast.append(time.perf_counter() - t)
worst = max(fast) * 1000
print(f"         ブレーカー開放後 20 回の最大 {worst:.2f} ms")
assert worst < 20, "開放後も Redis に触れている"

time.sleep(2.2)                      # recovery_timeout 経過 → 半開
c._client = None
os.environ["REDIS_URL"] = healthy_url
try:
    c.set_prediction("verify", "v", {"ok": True}); got = c.get_prediction("verify", "v")
    print(f"         復旧後の往復: {got}（REDIS_URL={healthy_url.split('@')[-1]}）")
    assert got == {"ok": True}
    assert redis_cache._REDIS_BREAKER.state == "closed"
except Exception as e:
    import traceback
    print("         復旧確認に失敗（健全な Redis が無い場合も含む）:", type(e).__name__, e)
    print("         " + traceback.format_exc(limit=-3).replace("\n", "\n         "))
PY

py_check "PostgreSQL 無応答: 接続タイムアウト（DB_CONNECT_TIMEOUT_SEC=2）内に失敗する" <<'PY'
import os, socket, sys, threading, time
os.environ["DB_CONNECT_TIMEOUT_SEC"] = "2"
try:
    import psycopg  # noqa: F401
except ImportError:
    print("         psycopg が無い"); sys.exit(3)
srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(16)
port = srv.getsockname()[1]
held = []
threading.Thread(target=lambda: [held.append(srv.accept()[0]) for _ in iter(int, 1)], daemon=True).start()
from src.db.session import init_engine
engine = init_engine(f"postgresql+psycopg://u:p@127.0.0.1:{port}/db")
t = time.perf_counter()
try:
    engine.connect()
    print("         接続できてしまった（想定外）"); sys.exit(1)
except Exception as e:
    dt = time.perf_counter() - t
    print(f"         失敗まで {dt:.2f} 秒（{type(e).__name__}）")
    sys.exit(0 if dt < 6 else 1)
PY
finish
