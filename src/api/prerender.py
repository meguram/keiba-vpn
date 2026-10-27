"""画面に出す応答（会員区分で項目を落とす前のもの）を、プロセス・ワーカーをまたいでディスクに事前生成して使い回す。

- 有効化は ``KEIBA_PREWARM=1``（置き場 + 先に作る係 ``src.api.prewarm``）か ``KEIBA_PRERENDER=1``（置き場だけ。Cloud Run のように
  CPU が少なく、先に作る係を動かしたくない所で、要求されたときに作った応答を ``KEIBA_PRERENDER_MAX_AGE_SEC`` 秒使い回す）。
  どちらも未設定（既定）なら何も読み書きせず、従来どおりその場で計算する。同じキーの同時要求は 1 本にまとめる（先の 1 本が作る間、他は待って結果を読む）。
- 保存先: 一時領域の ``keiba-prerender/<kind>/<key>.json``（``KEIBA_PRERENDER_DIR`` で変更。ホストの data 配下に依存せず、Cloud Run でも動く）。1 行目にメタ、2 行目以降が応答本体。
- 無効になる条件（読まれなくなる）:
  1. 期限切れ（``max_age``）
  2. 保存以外の世代（配備・設定変更・手動クリア。``cache_epoch.hard_current``）が進んだ
  3. 応答が依存するレース/開催日のデータが、応答を作り始めたあとに保存された（``mark_dirty`` の目印。レース単位）
- データを保存する側（``HybridStorage.save`` と、予測などのファイル出力）が目印を付け、``src.api.prewarm`` がそのレースの画面だけを作り直す。
- 匿名/会員/開発者の出し分けは、ここから取り出した後に各 API が従来どおり行う（ここには全項目の版だけを置く）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from src.utils import cache_epoch

DEFAULT_MAX_AGE_SEC = 300.0
_PRUNE_EVERY_SEC = 60.0
_KEY_RE = re.compile(r"^(?:\d{12}|\d{8})$")      # race_id（12 桁）または開催日（8 桁）


def enabled() -> bool:
    return os.environ.get("KEIBA_PREWARM", "").strip() == "1" or os.environ.get("KEIBA_PRERENDER", "").strip() == "1"


def warmer_enabled() -> bool:
    """先に作る係（周回・変更への反応）を動かすか。``KEIBA_PRERENDER=1`` だけのときは動かさない。"""
    return os.environ.get("KEIBA_PREWARM", "").strip() == "1"


def default_max_age() -> float:
    try:
        return float(os.environ.get("KEIBA_PRERENDER_MAX_AGE_SEC") or DEFAULT_MAX_AGE_SEC)
    except ValueError:
        return DEFAULT_MAX_AGE_SEC


PAST_RACE_MAX_AGE_SEC = 6 * 3600.0


def _past_race_max_age() -> float:
    try:
        return float(os.environ.get("KEIBA_PRERENDER_PAST_MAX_AGE_SEC") or PAST_RACE_MAX_AGE_SEC)
    except ValueError:
        return PAST_RACE_MAX_AGE_SEC


def race_max_age(body: bytes, *, today: str | None = None) -> float | None:
    """レースの応答の保持時間。終わった開催日（JST の今日より前）で結果まで揃っていれば長く使い回し、それ以外は既定（None）。

    Cloud Run には保存の目印（mark_dirty）が届かないので、短い max_age が唯一の更新手段になっている。結果が確定した日の応答は
    ほぼ変わらないので、毎回作り直さない（race-values の過去日と同じ 6 時間。配備・手動クリアで世代が進めば読まれなくなる）。
    日付は レース詳細の ``date``、予測の ``entries_status.race_date`` から読む。"""
    try:
        d = json.loads(body)
    except (TypeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    es = d.get("entries_status") if isinstance(d.get("entries_status"), dict) else {}
    digits = re.sub(r"\D", "", str(d.get("date") or es.get("race_date") or ""))[:8]
    if len(digits) != 8:
        return None
    if today is None:
        from datetime import timedelta, timezone

        today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
    if digits >= today:
        return None
    if "race_result" in d and not d.get("race_result"):   # レース詳細で結果がまだ取れていない日は、取れるまで短いまま
        return None
    return _past_race_max_age()


def _resolve_max_age(max_age: Any, body: bytes) -> float | None:
    """``max_age`` は秒数か、応答本体から秒数（None なら既定）を求める関数。"""
    if not callable(max_age):
        return max_age
    try:
        return max_age(body)
    except Exception:
        return None


def _max_bytes() -> int:
    try:
        return int(float(os.environ.get("KEIBA_PRERENDER_MAX_MB") or 300) * 1024 * 1024)
    except ValueError:
        return 300 * 1024 * 1024


def base_dir() -> Path:
    return Path(os.environ.get("KEIBA_PRERENDER_DIR") or Path(tempfile.gettempdir()) / "keiba-prerender")


def state_dir() -> Path:
    """事前生成の作る側のロック・状況ファイルの置き場（応答と同じ領域）。"""
    return base_dir() / "_state"


def _dir() -> Path:
    return base_dir()


def _mkdirs(d: Path) -> None:
    """複数のプロセス（API・Flask・取得役。実行ユーザーが違うことがある）が書けるよう、作った置き場は誰でも書ける権限にする。"""
    todo = []
    p = d
    while not p.exists() and p != p.parent and (p == d or p.parent == base_dir() or p == base_dir()):
        todo.append(p)
        p = p.parent
    d.mkdir(parents=True, exist_ok=True)
    for q in todo:
        try:
            os.chmod(q, 0o777)
        except OSError:
            pass


def _dirty_dir() -> Path:
    return _dir() / "_dirty"


def _path(kind: str, key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.=-]", "_", key)
    if len(safe) > 120 or safe != key:
        safe = safe[:100] + "-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return _dir() / kind / f"{safe}.json"


# ───────── 変更の目印（データを保存した側が付ける） ─────────

def mark_dirty(keys: Iterable[str]) -> int:
    """race_id / 開催日の目印を付ける（その時刻より前に作った応答は読まれなくなる）。付けた数を返す。"""
    if not enabled():
        return 0
    n = 0
    d = _dirty_dir()
    try:
        _mkdirs(d)
        for k in keys:
            if not _KEY_RE.match(str(k)):
                continue
            p = d / str(k)
            tmp = d / f".{k}.{os.getpid()}.tmp"      # 別ユーザーが作った目印も置き換えられるよう、更新ではなく作り直して差し替える
            tmp.write_bytes(b"")
            t = time.time_ns()
            os.utime(tmp, ns=(t, t))
            try:
                os.chmod(tmp, 0o666)
            except OSError:
                pass
            os.replace(tmp, p)
            n += 1
    except OSError:
        pass
    return n


def mark_from_save(category: str, key: str) -> None:
    """``HybridStorage.save`` から呼ぶ。race_id / 開催日がキーの保存だけ、今年と昨年のものに目印を付ける（過去分の補填で目印が溢れないように）。"""
    if not enabled() or not _KEY_RE.match(key or ""):
        return
    if int(key[:4]) < datetime.now().year - 1:
        return
    mark_dirty([key])


def dirty_between(since_ns: int, upto_ns: int) -> list[str]:
    """(since_ns, upto_ns] に目印が付いたキー。事前生成側が『どのレースが変わったか』を知るのに使う。"""
    out: list[tuple[int, str]] = []
    try:
        with os.scandir(_dirty_dir()) as it:
            for e in it:
                try:
                    m = e.stat().st_mtime_ns
                except OSError:
                    continue
                if since_ns < m <= upto_ns:
                    out.append((m, e.name))
    except OSError:
        return []
    return [k for _, k in sorted(out)]


def prune_dirty(older_than_sec: float = 3 * 86400) -> int:
    n = 0
    cut = time.time() - older_than_sec
    try:
        with os.scandir(_dirty_dir()) as it:
            for e in it:
                try:
                    if e.stat().st_mtime < cut:
                        os.unlink(e.path)
                        n += 1
                except OSError:
                    pass
    except OSError:
        pass
    return n


def _dirty_after(key: str, since_ns: int) -> bool:
    try:
        return (_dirty_dir() / key).stat().st_mtime_ns > since_ns
    except OSError:
        return False


# ───────── 置き場の読み書き ─────────

def _read(kind: str, key: str) -> tuple[dict[str, Any], bytes] | None:
    try:
        raw = _path(kind, key).read_bytes()
        head, _, body = raw.partition(b"\n")
        return json.loads(head), body
    except (OSError, ValueError):
        return None


def _valid(meta: dict[str, Any], now: float) -> bool:
    if meta.get("hard") != cache_epoch.hard_current() or (now - float(meta.get("at") or 0)) >= float(meta.get("max_age") or 0):
        return False
    since = int(meta.get("since") or 0)
    return not any(_dirty_after(d, since) for d in meta.get("deps") or [])


def get(kind: str, key: str) -> bytes | None:
    """有効な事前生成の応答本体。無ければ None。"""
    if not enabled():
        return None
    hit = _read(kind, key)
    if hit and _valid(hit[0], time.time()):
        return hit[1]
    return None


def age(kind: str, key: str) -> float | None:
    """有効な事前生成の経過秒。無効・無しは None（事前生成側が作り直すかどうかの判定用）。"""
    hit = _read(kind, key)
    now = time.time()
    if hit and _valid(hit[0], now):
        return now - float(hit[0]["at"])
    return None


_last_prune = 0.0


def prune_size(max_bytes: int | None = None) -> int:
    """置き場の合計が上限を超えていたら、古い応答から消す（メモリ上の /tmp を使う Cloud Run で膨らまないように）。消した数を返す。"""
    cap = max_bytes if max_bytes is not None else _max_bytes()
    files: list[tuple[float, int, str]] = []
    total = 0
    try:
        for kind in os.scandir(_dir()):
            if not kind.is_dir() or kind.name.startswith("_"):
                continue
            for e in os.scandir(kind.path):
                try:
                    st = e.stat()
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, e.path))
                total += st.st_size
    except OSError:
        return 0
    n = 0
    if total <= cap:
        return 0
    for _, size, path in sorted(files):
        try:
            os.unlink(path)
            total -= size
            n += 1
        except OSError:
            pass
        if total <= cap * 0.8:
            break
    return n


def _maybe_prune() -> None:
    global _last_prune
    now = time.time()
    if now - _last_prune < _PRUNE_EVERY_SEC:
        return
    _last_prune = now
    try:
        prune_size()
    except Exception:
        pass


def put(kind: str, key: str, body: bytes, *, since_ns: int, hard: int, deps: Iterable[str] = (), max_age: float | None = None) -> None:
    """``since_ns``/``hard`` は計算を始める前の時刻・世代（計算中に deps のデータが保存されたら、この応答は読まれない）。失敗しても何もしない。"""
    if not enabled():
        return
    max_age = max_age or default_max_age()
    _maybe_prune()
    p = _path(kind, key)
    try:
        _mkdirs(p.parent)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        meta = json.dumps({"hard": hard, "since": since_ns, "at": time.time(), "max_age": max_age, "deps": sorted(set(deps))}, ensure_ascii=False).encode("utf-8")
        tmp.write_bytes(meta + b"\n" + body)
        try:
            os.chmod(tmp, 0o666)
        except OSError:
            pass
        os.replace(tmp, p)
    except OSError:
        pass


def _resolve_deps(deps: Any, body: bytes) -> list[str]:
    try:
        return list(deps(body)) if callable(deps) else list(deps or [])
    except Exception:
        return []


_flights: dict[str, Any] = {}


async def cached_response(kind: str, key: str, compute: Callable[[], Awaitable[Any]], *, deps: Any = None, max_age: Any = None, refresh: bool = False) -> Any:
    """``compute`` は JSONResponse を返す。事前生成があればそれを返し、無ければ計算して 200 のものを書き込む（他ワーカーも使える）。
    同じキーの同時要求は 1 本だけ計算し、他は待って結果を読む（人気のレースに一斉に来ても、1 回しか計算しない）。
    ``deps`` は応答が依存する race_id/開催日（または本体から求める関数）。省略時は ``[key]``。"""
    if not enabled():
        return await compute()
    import asyncio

    from starlette.responses import Response

    async def run() -> Any:
        since, hard = time.time_ns(), cache_epoch.hard_current()
        resp = await compute()
        body = getattr(resp, "body", None)
        if body is not None and getattr(resp, "status_code", 200) == 200:
            put(kind, key, bytes(body), since_ns=since, hard=hard, deps=_resolve_deps(deps if deps is not None else [key], bytes(body)),
                max_age=_resolve_max_age(max_age, bytes(body)))
        return resp

    if refresh:
        return await run()
    body = get(kind, key)
    if body is not None:
        return Response(content=body, media_type="application/json")
    token = f"{kind}/{key}"
    lock = _flights.get(token)
    if lock is None:
        lock = _flights[token] = asyncio.Lock()
    try:
        async with lock:
            body = get(kind, key)                         # 先の 1 本が作り終えていれば、それを読む
            if body is not None:
                return Response(content=body, media_type="application/json")
            return await run()
    finally:
        if not lock.locked() and _flights.get(token) is lock:
            _flights.pop(token, None)


_sync_flights: dict[str, threading.Lock] = {}
_flight_guard = threading.Lock()
_refreshing: set[str] = set()
_refreshing_lock = threading.Lock()


def get_stale(kind: str, key: str, max_stale_sec: float) -> bytes | None:
    """期限切れ・データ更新の目印があっても、``max_stale_sec`` 以内に作られた版（世代が同じものに限る）。無ければ None。"""
    if not enabled():
        return None
    hit = _read(kind, key)
    if hit and hit[0].get("hard") == cache_epoch.hard_current() and (time.time() - float(hit[0].get("at") or 0)) < max_stale_sec:
        return hit[1]
    return None


def _store_json(kind: str, key: str, compute: Callable[[], dict[str, Any]], deps: Any, max_age: float | None) -> dict[str, Any]:
    since, hard = time.time_ns(), cache_epoch.hard_current()
    out = compute()
    try:
        raw = json.dumps(out, ensure_ascii=False).encode("utf-8")
        put(kind, key, raw, since_ns=since, hard=hard, deps=_resolve_deps(deps if deps is not None else [key], raw), max_age=_resolve_max_age(max_age, raw))
    except (TypeError, ValueError):
        pass
    return out


def _refresh_in_background(kind: str, key: str, compute: Callable[[], dict[str, Any]], deps: Any, max_age: float | None) -> None:
    token = f"{kind}/{key}"
    with _refreshing_lock:
        if token in _refreshing:
            return
        _refreshing.add(token)

    def _run() -> None:
        try:
            _store_json(kind, key, compute, deps, max_age)
        except Exception:
            pass
        finally:
            with _refreshing_lock:
                _refreshing.discard(token)

    threading.Thread(target=_run, name=f"prerender-refresh-{kind}", daemon=True).start()


def cached_json(
    kind: str, key: str, compute: Callable[[], dict[str, Any]], *, deps: Any = None, max_age: float | None = None, refresh: bool = False,
    stale_ok_sec: float = 0, recompute: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """同期版（Flask 側）。dict を JSON にして置く。
    ``stale_ok_sec`` を指定すると、無効・期限切れでもその秒数以内の版があれば先にそれを返し、裏で ``recompute``（省略時 ``compute``）で作り直す
    （表示の鮮度より待ち時間を優先する画面向け。作り直しはキーごとに 1 本だけ）。"""
    if not enabled():
        return compute()
    if not refresh:
        body = get(kind, key)
        if body is None and stale_ok_sec > 0:
            body = get_stale(kind, key, stale_ok_sec)
            if body is not None:
                _refresh_in_background(kind, key, recompute or compute, deps, max_age)
        if body is not None:
            try:
                return json.loads(body)
            except ValueError:
                pass
    token = f"{kind}/{key}"
    with _flight_guard:
        lock = _sync_flights.setdefault(token, threading.Lock())
    with lock:
        if not refresh:
            body = get(kind, key)                         # 先の 1 本が作り終えていれば、それを読む
            if body is not None:
                try:
                    return json.loads(body)
                except ValueError:
                    pass
        return _store_json(kind, key, compute, deps, max_age)


def race_ids_deps(date: str) -> Callable[[bytes], list[str]]:
    """開催日単位の応答（レース一覧・期待値表示）の deps: その日付と、本体に出てくる全 race_id。"""
    def _f(body: bytes) -> list[str]:
        doc = json.loads(body)
        return [date] + [str(r["race_id"]) for r in (doc.get("races") or []) if isinstance(r, dict) and r.get("race_id")]
    return _f
