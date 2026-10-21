"""監視ポータルのモデル性能ページ用: 事前計算したキャッシュ JSON を読んで、画面向けにまとめる（重い計算はしない）。

キャッシュは ``python -m src.scripts.data.model_perf_cache build`` が data/local/meta/model_perf/ に作る。
無い・壊れているモデルは ``status="not_computed"`` にして返し、画面は『未計算』と出す（例外で落とさない）。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(os.environ.get("KEIBA_REPO_ROOT") or Path(__file__).resolve().parents[2])


def perf_dir() -> Path:
    return Path(os.environ.get("KEIBA_MODEL_PERF_DIR") or REPO_ROOT / "data" / "local" / "meta" / "model_perf")


def _read(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _age_hours(path: Path) -> float | None:
    try:
        return round((time.time() - path.stat().st_mtime) / 3600, 1)
    except OSError:
        return None


def load_all() -> dict[str, Any]:
    """画面用の全データ。{built_at, age_hours, models: [...], mlflow: {...}, status}。"""
    d = perf_dir()
    idx = _read(d / "index.json")
    if not isinstance(idx, dict):
        return {"status": "not_computed", "built_at": None, "age_hours": None, "models": [], "mlflow": {},
                "hint": "python -m src.scripts.data.model_perf_cache build（重いので /tmp/heavy_run.sh model-perf … 経由で）"}
    models = []
    for m in idx.get("models") or []:
        key = str(m.get("key") or "")
        body = _read(d / f"{key}.json") if key.replace("_", "").isalnum() else None
        if isinstance(body, dict):
            models.append(_sanitize(body))
        else:
            models.append({"key": key, "label": m.get("label") or key, "kind": m.get("kind"), "status": "not_computed"})
    ml = _read(d / "mlflow.json")
    return {"status": "ok", "built_at": idx.get("built_at"), "age_hours": _age_hours(d / "index.json"), "models": models,
            "mlflow": (ml or {}).get("models") or {}, "mlflow_built_at": (ml or {}).get("built_at")}


def _sanitize(body: dict) -> dict:
    """画面に出さないもの（オッズ生データ・個人情報・秘密）は元々キャッシュに入れない。念のためパスのうちホームの絶対パスを落とす。"""
    out = dict(body)
    out["status"] = "ok"
    meta = dict(out.get("meta") or {})
    if isinstance(meta.get("bundle"), str):
        meta["bundle"] = meta["bundle"].replace(str(REPO_ROOT), "")
    out["meta"] = meta
    return out


def summary_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """/services の小さな表用: 版ごとの 1 行（テスト 2025 の勝率 LogLoss・Top1 的中率、2026 本番の勝率 LogLoss）。"""
    rows = []
    for m in data.get("models") or []:
        t = m.get("test2025") or {}
        win = ((t.get("metrics") or {}).get("win") or {})
        rows.append({"key": m.get("key"), "label": m.get("label"), "status": m.get("status"),
                     "test_win_logloss": win.get("logloss"), "test_top1_hit": (t.get("top1") or {}).get("hit"),
                     "prod_win_logloss": (m.get("prod_total") or {}).get("win_logloss"), "prod_races": (m.get("prod_total") or {}).get("n_races")})
    return rows
