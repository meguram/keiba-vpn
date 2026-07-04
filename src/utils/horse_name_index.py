"""
馬名オートコンプリート用 JSON（**常にローカルファイルのみ**。GCS は参照しない）。

高頻度の ``/api/horse-names/search`` は ``load_horse_name_index`` 経由で
``data/knowledge/horse_name_index.json``（または旧 ``data/local/knowledge/``）だけを読み、
リクエストごとに GCS へは行かない。

- 正規の保存先は ``data/knowledge/horse_name_index.json``。
- メモリキャッシュは mtime で無効化。
- ``horse_result`` 保存時に upsert でローカル JSON を更新。
- GCS からの一括取り込みは **オフライン** のみ:
  ``python -m src.scripts.data.build_horse_index --from-gcs``
  （``rebuild_horse_name_index_from_gcs_horse_result``）。
- 起動時 ``ensure_horse_name_index`` は **ローカルディスク上の horse_result キャッシュ／ミラーのみ**を走査する。
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_index_lock = threading.RLock()
_mem_mtime: float = -2.0
_mem_data: dict[str, Any] | None = None


def horse_name_index_candidate_paths(base_dir: str | Path) -> tuple[Path, ...]:
    from src.config.data_paths import HORSE_NAME_INDEX_JSON

    base = Path(base_dir)
    return (
        HORSE_NAME_INDEX_JSON,
        base / "data" / "calculated_data" / "knowledge" / "horse_name_index.json",
        base / "data" / "page_reference" / "knowledge" / "horse_name_index.json",
        base / "data" / "knowledge" / "horse_name_index.json",
        base / "data" / "local" / "knowledge" / "horse_name_index.json",
    )


def canonical_horse_name_index_path(base_dir: str | Path) -> Path:
    """正本: ``data/calculated_data/knowledge/horse_name_index.json``。"""
    from src.config.data_paths import calculated_data_root

    return calculated_data_root() / "knowledge" / "horse_name_index.json"


def _legacy_mirror_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "data" / "knowledge" / "horse_name_index.json"


def _best_existing_path(paths: tuple[Path, ...]) -> tuple[Path | None, float]:
    best: Path | None = None
    best_mtime = -1.0
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime >= best_mtime:
            best_mtime = st.st_mtime
            best = p
    return best, best_mtime


def load_horse_name_index(base_dir: str | Path) -> dict[str, Any]:
    """
    馬名インデックスを返す。ファイルが無い場合は空リスト（未キャッシュで次回も再探索）。
    ロード済みで mtime が変わっていなければメモリキャッシュを返す。
    """
    global _mem_mtime, _mem_data
    paths = horse_name_index_candidate_paths(base_dir)
    picked, mtime = _best_existing_path(paths)
    if picked is None:
        logger.warning(
            "馬名インデックスが見つかりません（%s または %s）",
            paths[0],
            paths[1],
        )
        return {"version": "1.0", "total_horses": 0, "horses": []}

    with _index_lock:
        if _mem_data is not None and mtime == _mem_mtime:
            return _mem_data
        try:
            with open(picked, "r", encoding="utf-8") as f:
                _mem_data = json.load(f)
            _mem_mtime = mtime
            nh = len((_mem_data or {}).get("horses") or [])
            logger.info("馬名インデックスをロードしました: %d 頭 (%s)", nh, picked)
            return _mem_data  # type: ignore[return-value]
        except Exception as e:
            logger.error("馬名インデックスのロードに失敗: %s", e)
            return {"version": "1.0", "total_horses": 0, "horses": []}


def invalidate_horse_name_index_cache() -> None:
    global _mem_mtime, _mem_data
    with _index_lock:
        _mem_mtime = -2.0
        _mem_data = None


def _atomic_json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_horse_name_index_bundle_unlocked(base_dir: str | Path, out: dict[str, Any]) -> Path:
    """calculated_data 正本へ書き、旧 ``data/knowledge/`` にミラーする（ロック済み想定）。"""
    canonical = canonical_horse_name_index_path(base_dir)
    _atomic_json_write(canonical, out)
    legacy = _legacy_mirror_path(base_dir)
    try:
        if legacy.resolve() != canonical.resolve():
            _atomic_json_write(legacy, out)
    except OSError:
        if legacy != canonical:
            _atomic_json_write(legacy, out)
    invalidate_horse_name_index_cache()
    return canonical


def _write_horse_name_index_bundle(base_dir: str | Path, out: dict[str, Any]) -> Path:
    canonical = canonical_horse_name_index_path(base_dir)
    lock_path = canonical.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            return _write_horse_name_index_bundle_unlocked(base_dir, out)
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def upsert_horse_name_index_entry(
    base_dir: str | Path,
    horse_id: str,
    horse_name: str,
    name_en: str = "",
) -> None:
    """``horse_result`` 保存後に 1 頭分をマージして正規パスへ書き出す。"""
    hid = str(horse_id or "").strip()
    hn = str(horse_name or "").strip()
    if not hid or not hn:
        return
    ne = str(name_en or "").strip()
    canonical = canonical_horse_name_index_path(base_dir)
    lock_path = canonical.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            data: dict[str, Any]
            if canonical.is_file():
                try:
                    with open(canonical, "r", encoding="utf-8") as rf:
                        data = json.load(rf)
                except Exception:
                    data = {"version": "1.0", "horses": []}
            else:
                data = {"version": "1.0", "horses": []}
                alt = horse_name_index_candidate_paths(base_dir)[1]
                if alt.is_file():
                    try:
                        with open(alt, "r", encoding="utf-8") as rf:
                            data = json.load(rf)
                    except Exception:
                        pass

            horses: list[dict[str, Any]] = list(data.get("horses") or [])
            by_id: dict[str, int] = {
                str(h.get("horse_id")): i
                for i, h in enumerate(horses)
                if h.get("horse_id")
            }
            year = time.strftime("%Y")
            rec: dict[str, Any] = {
                "horse_id": hid,
                "horse_name": hn,
                "name_en": ne,
                "year": year,
            }
            if hid in by_id:
                horses[by_id[hid]] = rec
            else:
                horses.append(rec)
            data["horses"] = horses
            data["total_horses"] = len(horses)
            data["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _write_horse_name_index_bundle_unlocked(base_dir, data)
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def rebuild_horse_name_index_from_horse_result_cache(base_dir: str | Path) -> int:
    """
    ``data/cache/horse_result`` および ``data/local/mirror/horse_result`` を走査してフル再構築。
    HybridStorage の馬キャッシュレイアウト（prefix ディレクトリ）に対応。
    """
    base = Path(base_dir)
    scan_roots = [
        base / "data" / "cache" / "horse_result",
        base / "data" / "local" / "mirror" / "horse_result",
    ]
    merged: dict[str, dict[str, Any]] = {}
    for root in scan_roots:
        if not root.is_dir():
            continue
        for fp in root.rglob("*.json"):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            hid = str(raw.get("horse_id") or "").strip()
            hn = str(raw.get("horse_name") or "").strip()
            if not hid or not hn:
                continue
            ne = str(raw.get("name_en") or "").strip()
            ydir = fp.parent.name if fp.parent != root else ""
            merged[hid] = {
                "horse_id": hid,
                "horse_name": hn,
                "name_en": ne,
                "year": str(raw.get("year") or ydir or ""),
            }

    horses = sorted(merged.values(), key=lambda h: h["horse_id"])
    out: dict[str, Any] = {
        "version": "1.0",
        "total_horses": len(horses),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "horses": horses,
    }
    path = _write_horse_name_index_bundle(base_dir, out)
    logger.info("馬名インデックス（ローカルキャッシュ）: %d 頭 → %s", len(horses), path)
    return len(horses)


def _horse_result_index_entry(
    raw: dict[str, Any] | None, hid: str, pref: str,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    hid_s = str(raw.get("horse_id") or hid or "").strip()
    hn = str(raw.get("horse_name") or "").strip()
    if not hid_s or not hn:
        return None
    return {
        "horse_id": hid_s,
        "horse_name": hn,
        "name_en": str(raw.get("name_en") or "").strip(),
        "year": pref if pref.isdigit() else (hid_s[:4] if len(hid_s) >= 4 else ""),
    }


def rebuild_horse_name_index_from_gcs_horse_result(base_dir: str | Path) -> int:
    """
    GCS の ``horse_result`` を全走査してローカル JSON を構築する。

    **API サーバのランタイムでは呼ばないこと**（コスト・レイテンシ大）。
    初回構築・バッチ更新は ``python -m src.scripts.data.build_horse_index --from-gcs`` から実行する。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.scraper.storage import HybridStorage

    st = HybridStorage(str(base_dir))
    if not st.gcs_enabled or not getattr(st, "_bucket_name", ""):
        logger.warning("GCS 未設定のため GCS 由来の馬名インデックス再構築をスキップします")
        return 0

    prefixes = st.list_years("horse_result")
    if not prefixes:
        prefixes = [str(y) for y in range(2009, 2027)]

    merged: dict[str, dict[str, Any]] = {}
    n_load = 0
    workers = min(12, max(2, int(os.environ.get("HORSE_NAME_INDEX_GCS_WORKERS", "6"))))

    def _load_one(args: tuple[str, str]) -> dict[str, Any] | None:
        pref, hid = args
        raw = None
        for attempt in range(3):
            try:
                raw = st.load("horse_result", hid)
                break
            except Exception:
                if attempt < 2:
                    time.sleep(0.4 * (attempt + 1))
        return _horse_result_index_entry(raw, hid, pref)

    for pref in prefixes:
        keys = st.list_keys("horse_result", pref)
        if not keys:
            continue
        logger.info("馬名インデックス: prefix=%s keys=%d", pref, len(keys))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_load_one, (pref, hid)): hid for hid in keys}
            for fut in as_completed(futures):
                rec = fut.result()
                if rec:
                    merged[rec["horse_id"]] = rec
                n_load += 1
                if n_load % 2000 == 0:
                    logger.info("馬名インデックス GCS 走査: %d 件処理", n_load)

    horses = sorted(merged.values(), key=lambda h: h["horse_id"])
    out: dict[str, Any] = {
        "version": "1.0",
        "total_horses": len(horses),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "horses": horses,
    }
    path = _write_horse_name_index_bundle(base_dir, out)
    logger.info(
        "馬名インデックス GCS 再構築完了: %d 頭（%d 件ロード）→ %s",
        len(horses),
        n_load,
        path,
    )
    return len(horses)


def rebuild_horse_name_index_from_gcs_race_shutuba(base_dir: str | Path) -> int:
    """GCS ``race_shutuba`` の出馬表から馬名を収集（horse_result より軽い補完用）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.scraper.storage import HybridStorage

    st = HybridStorage(str(base_dir))
    if not st.gcs_enabled:
        return 0

    merged: dict[str, dict[str, Any]] = {}
    years = st.list_years("race_shutuba") or [str(y) for y in range(2015, 2027)]
    workers = min(8, max(2, int(os.environ.get("HORSE_NAME_INDEX_SHUTUBA_WORKERS", "6"))))

    def _load_race(rid: str) -> list[dict[str, Any]]:
        try:
            card = st.load("race_shutuba", rid)
        except Exception:
            return []
        if not isinstance(card, dict):
            return []
        out: list[dict[str, Any]] = []
        for e in card.get("entries") or []:
            hid = str(e.get("horse_id") or "").strip()
            hn = str(e.get("horse_name") or e.get("name") or "").strip()
            if hid and hn:
                out.append({
                    "horse_id": hid,
                    "horse_name": hn,
                    "name_en": "",
                    "year": hid[:4] if len(hid) >= 4 else "",
                })
        return out

    for yr in years:
        rids = st.list_keys("race_shutuba", yr)
        if not rids:
            continue
        logger.info("馬名インデックス shutuba: year=%s races=%d", yr, len(rids))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_load_race, rid) for rid in rids]
            for fut in as_completed(futures):
                for rec in fut.result():
                    merged[rec["horse_id"]] = rec

    if not merged:
        return 0

    horses = sorted(merged.values(), key=lambda h: h["horse_id"])
    out: dict[str, Any] = {
        "version": "1.0",
        "total_horses": len(horses),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "race_shutuba_gcs",
        "horses": horses,
    }
    _write_horse_name_index_bundle(base_dir, out)
    return len(horses)


def run_weekly_horse_name_index_update(base_dir: str | Path) -> dict[str, Any]:
    """
    金曜夕方バッチ用: GCS から馬名リストを再構築する。

    既定は ``horse_result`` 全件。``HORSE_NAME_INDEX_SOURCE=shutuba`` で出馬表のみ。
    ``task_raceday_eve``（金曜 18:00）および cron から呼ぶ。
    """
    t0 = time.time()
    # 金曜 18:00 cron 向け: 出馬表走査は horse_result 全件より高速（約3〜5分）
    source = os.environ.get("HORSE_NAME_INDEX_SOURCE", "shutuba").strip().lower()
    try:
        if source == "shutuba":
            n = rebuild_horse_name_index_from_gcs_race_shutuba(base_dir)
        else:
            n = rebuild_horse_name_index_from_gcs_horse_result(base_dir)
    except Exception as exc:
        logger.error("馬名インデックス週次更新失敗: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
            "elapsed_sec": round(time.time() - t0, 1),
        }
    if n < 1:
        return {
            "status": "skipped",
            "reason": "gcs_empty_or_disabled",
            "total_horses": n,
            "path": str(canonical_horse_name_index_path(base_dir)),
            "elapsed_sec": round(time.time() - t0, 1),
        }
    return {
        "status": "ok",
        "total_horses": n,
        "path": str(canonical_horse_name_index_path(base_dir)),
        "elapsed_sec": round(time.time() - t0, 1),
    }


_last_ensure_result: dict[str, Any] = {}


def get_last_horse_name_index_ensure() -> dict[str, Any]:
    """直近の ``ensure_horse_name_index`` の結果（デバッグ・メタ API 用）。"""
    return dict(_last_ensure_result)


def horse_name_index_meta(base_dir: str | Path) -> dict[str, Any]:
    """参照用メタ（全頭リストは返さない）。データ実体は常にローカル JSON。"""
    base = Path(base_dir)
    paths = horse_name_index_candidate_paths(base)
    picked, mtime = _best_existing_path(paths)
    data = load_horse_name_index(base)
    nh = len(data.get("horses") or [])
    return {
        "storage": "local_json",
        "canonical_path": str(canonical_horse_name_index_path(base)),
        "resolved_path": str(picked) if picked else None,
        "resolved_mtime": mtime if picked and mtime >= 0 else None,
        "total_horses": nh,
        "generated_at": data.get("generated_at"),
        "version": data.get("version"),
        "last_ensure": get_last_horse_name_index_ensure(),
    }


def ensure_horse_name_index(
    base_dir: str | Path,
    *,
    min_horses: int | None = None,
) -> dict[str, Any]:
    """
    馬名インデックスが無い／薄いとき、**ローカル上の** ``horse_result`` だけで再構築する。

    走査対象: ``data/cache/horse_result`` および ``data/local/mirror/horse_result``。

    ローカルに馬 JSON が無く閾値を満たせない場合は ``insufficient_local_only`` を返す。
    そのときのフルリスト初期化はオフラインで
    ``python -m src.scripts.data.build_horse_index --from-gcs`` を実行し、
    生成された ``data/knowledge/horse_name_index.json`` をデプロイする。

    ``HORSE_NAME_INDEX_MIN_HORSES`` で閾値を上書き（デフォルト 1）。
    """
    global _last_ensure_result
    base = Path(base_dir)
    canonical = canonical_horse_name_index_path(base)
    if min_horses is None:
        try:
            min_horses = max(1, int(os.environ.get("HORSE_NAME_INDEX_MIN_HORSES", "1")))
        except ValueError:
            min_horses = 1

    data = load_horse_name_index(base)
    n0 = len(data.get("horses") or [])
    if n0 >= min_horses:
        _last_ensure_result = {
            "status": "skipped",
            "reason": "already_sufficient",
            "total_horses": n0,
            "canonical_path": str(canonical),
            "min_horses": min_horses,
            "storage": "local_json",
        }
        return _last_ensure_result

    n_cache = rebuild_horse_name_index_from_horse_result_cache(base)
    invalidate_horse_name_index_cache()
    data = load_horse_name_index(base)
    n = len(data.get("horses") or [])

    if n >= min_horses:
        _last_ensure_result = {
            "status": "rebuilt_local_cache",
            "total_horses": n,
            "canonical_path": str(canonical),
            "scanned_local_merge": n_cache,
            "min_horses": min_horses,
            "storage": "local_json",
        }
        return _last_ensure_result

    _last_ensure_result = {
        "status": "insufficient_local_only",
        "total_horses": n,
        "canonical_path": str(canonical),
        "scanned_local_merge": n_cache,
        "min_horses": min_horses,
        "storage": "local_json",
        "hint": "オフライン: python -m src.scripts.data.build_horse_index --from-gcs でローカル JSON を生成し配置してください。",
    }
    return _last_ensure_result
