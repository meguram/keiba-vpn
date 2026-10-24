"""挙動フラグの既定を config/behavior_flags.json から環境変数へ入れる（明示済みの環境変数は上書きしない）。

stg / prod / Cloud Run / Jobs で同じ既定を共有し、フラグが .env ごとに別管理になるのを防ぐ。表が空なら何もしない（従来どおり）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


def apply_behavior_flags(root: Path | None = None, environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    try:
        from src.ops.release_manifest import ROOT, load_flag_table

        table = load_flag_table(root or ROOT)
    except Exception:  # noqa: BLE001  表が読めなくても起動を止めない
        return {}
    name = (env.get("KEIBA_ENV") or env.get("APP_ENV") or "").strip().lower()
    name = {"staging": "stg", "production": "prod"}.get(name, name)
    merged = {k: str(v) for k, v in (table.get("defaults") or {}).items()}
    merged.update({k: str(v) for k, v in ((table.get("env_overrides") or {}).get(name) or {}).items()})
    applied: dict[str, str] = {}
    for k, v in merged.items():
        if not k.startswith("KEIBA_") or env.get(k) not in (None, ""):
            continue
        env[k] = v
        applied[k] = v
    return applied
