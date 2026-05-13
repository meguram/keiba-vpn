"""
ランタイム負荷設定（data/queue/queue_load_settings.json）

.env の既定より優先。サーバ再起動なしで同時ジョブ数・スタッガー等を下げ、
遠隔側（netkeiba 等）および当サーバの負荷を抑えられる。

未設定（null またはキーなし）の項目は従来どおり環境変数を用いる。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))

QUEUE_DIR = Path(__file__).resolve().parent.parent / "data" / "queue"
QUEUE_LOAD_SETTINGS = QUEUE_DIR / "queue_load_settings.json"

# 検証用境界
_P_MIN, _P_MAX = 1, 32
_STAG_MAX = 30.0
_ETA_MIN, _ETA_MAX = 0.5, 300.0
_INFLIGHT_MAX = 64


def _default_parallel_env() -> int:
    try:
        n = int(os.environ.get("SCRAPE_QUEUE_PARALLEL", "6"))
    except (TypeError, ValueError):
        n = 6
    return max(_P_MIN, min(_P_MAX, n))


def _default_stagger_env() -> float:
    try:
        v = float(os.environ.get("SCRAPE_QUEUE_STAGGER_SEC", "0.25"))
    except (TypeError, ValueError):
        v = 0.25
    return max(0.0, v)


def _default_eta_env() -> float:
    try:
        v = float(os.environ.get("SCRAPE_QUEUE_ETA_SEC_PER_JOB", "10"))
    except (TypeError, ValueError):
        v = 10.0
    return max(0.5, v)


def _read_file() -> dict[str, Any]:
    try:
        with open(QUEUE_LOAD_SETTINGS, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_effective_parallel() -> int:
    d = _read_file()
    if "parallel" in d and d["parallel"] is not None:
        try:
            n = int(d["parallel"])
        except (TypeError, ValueError):
            n = _default_parallel_env()
        return max(_P_MIN, min(_P_MAX, n))
    return _default_parallel_env()


def get_effective_stagger_sec() -> float:
    d = _read_file()
    if "stagger_sec" in d and d["stagger_sec"] is not None:
        try:
            v = float(d["stagger_sec"])
        except (TypeError, ValueError):
            v = _default_stagger_env()
        return max(0.0, min(_STAG_MAX, v))
    return _default_stagger_env()


def get_effective_eta_sec_per_job() -> float:
    d = _read_file()
    if "eta_sec_per_job" in d and d["eta_sec_per_job"] is not None:
        try:
            v = float(d["eta_sec_per_job"])
        except (TypeError, ValueError):
            v = _default_eta_env()
        return max(_ETA_MIN, min(_ETA_MAX, v))
    return _default_eta_env()


def netkeiba_inflight_file_override() -> int | None:
    """
    ファイルに netkeiba 同時 in-flight が入っている場合は 0=無制限, 1..N。
    キーなし or null: None（.env 既定のロジックに任せる）。
    """
    d = _read_file()
    if "netkeiba_max_inflight" not in d or d.get("netkeiba_max_inflight") is None:
        return None
    try:
        n = int(d["netkeiba_max_inflight"])
    except (TypeError, ValueError):
        return None
    if n < 0:
        return None
    if n == 0:
        return 0
    return max(1, min(_INFLIGHT_MAX, n))


def env_snapshot() -> dict[str, Any]:
    return {
        "SCRAPE_QUEUE_PARALLEL": _default_parallel_env(),
        "SCRAPE_QUEUE_STAGGER_SEC": _default_stagger_env(),
        "SCRAPE_QUEUE_ETA_SEC_PER_JOB": _default_eta_env(),
        "NETKEIBA_MAX_CONCURRENT_REQUESTS": (
            os.environ.get("NETKEIBA_MAX_CONCURRENT_REQUESTS", "").strip() or None
        ),
    }


def get_status_summary() -> dict[str, Any]:
    """API / 画面用: 実効値 + 上書きファイル + env 基準（表示専用）。"""
    raw = _read_file()
    fcopy = {k: raw.get(k) for k in (
        "parallel", "stagger_sec", "eta_sec_per_job", "netkeiba_max_inflight",
    )}
    fcopy["updated_at"] = raw.get("updated_at")
    fcopy["note"] = raw.get("note", "") or ""
    nfi = netkeiba_inflight_file_override()
    eff_inflight: str | int
    if nfi is None:
        eff_inflight = "env_or_auto"
    elif nfi == 0:
        eff_inflight = 0
    else:
        eff_inflight = nfi
    return {
        "overrides_file": str(QUEUE_LOAD_SETTINGS),
        "overrides": fcopy,
        "effective": {
            "parallel": get_effective_parallel(),
            "stagger_sec": get_effective_stagger_sec(),
            "eta_sec_per_job": get_effective_eta_sec_per_job(),
            "netkeiba_max_inflight": eff_inflight,
        },
        "env_baseline": env_snapshot(),
    }


def validate_and_save(
    body: dict[str, Any],
) -> dict[str, Any] | dict[str, str]:
    """
    部分更新を保存。各キーに null / 空文字列を送ると上書きを外して .env 基準に戻す。
    数値4つとメモのいずれも無ければファイルを削除する。
    """
    if not isinstance(body, dict):
        return {"error": "body は JSON オブジェクトで送ってください"}
    m: dict[str, Any] = {**_read_file()}

    for key in ("parallel", "stagger_sec", "eta_sec_per_job", "netkeiba_max_inflight"):
        if key not in body:
            continue
        v = body[key]
        if v is None or (isinstance(v, str) and (not v.strip() if v else True)):
            m.pop(key, None)
            continue
        if key == "parallel":
            p = int(v)
            if p < _P_MIN or p > _P_MAX:
                return {"error": f"parallel は {_P_MIN}〜{_P_MAX} です"}
            m[key] = p
        elif key == "stagger_sec":
            s = float(v)
            if s < 0 or s > _STAG_MAX:
                return {"error": f"stagger_sec は 0〜{_STAG_MAX:g} です"}
            m[key] = s
        elif key == "eta_sec_per_job":
            e = float(v)
            if e < _ETA_MIN or e > _ETA_MAX:
                return {"error": f"eta_sec_per_job は {_ETA_MIN:g}〜{_ETA_MAX:g} です"}
            m[key] = e
        else:
            ni = int(v)
            if ni < 0 or ni > _INFLIGHT_MAX:
                return {"error": f"netkeiba_max_inflight は 0(無制限) または 1..{_INFLIGHT_MAX} です"}
            m[key] = ni

    if "note" in body:
        nv = body.get("note")
        if nv is None or (isinstance(nv, str) and not str(nv).strip()):
            m.pop("note", None)
        else:
            m["note"] = str(nv)[:2000]

    m.pop("updated_at", None)
    has_stored = any(
        m.get(x) is not None
        for x in ("parallel", "stagger_sec", "eta_sec_per_job", "netkeiba_max_inflight")
    ) or bool(m.get("note", "").strip())

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not has_stored:
            if QUEUE_LOAD_SETTINGS.exists():
                QUEUE_LOAD_SETTINGS.unlink()
        else:
            m["updated_at"] = datetime.now(tz=JST).isoformat()
            tmp = QUEUE_LOAD_SETTINGS.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=2)
            tmp.replace(QUEUE_LOAD_SETTINGS)
    except OSError as e:
        return {"error": str(e)}

    return get_status_summary()


def clear_overrides_file() -> dict[str, Any]:
    try:
        if QUEUE_LOAD_SETTINGS.exists():
            QUEUE_LOAD_SETTINGS.unlink()
    except OSError as e:
        return {"error": str(e), **get_status_summary()}
    return get_status_summary()
