"""dev 環境用のローカルデータストア（GCS の代わり）。

``KEIBA_ENV=dev`` の ``HybridStorage`` は GCS を使わず、このストアを読み書きする。
配置は GCS のミラーと同じ構造にして、stg 以降のローカルミラー読みでも流用できるようにする。

  race  系: ``{root}/{category}/{年4桁}/{race_id}.json``
  horse 系: ``{root}/{category}/{horse_id先頭4桁}/{horse_id}.json``
  other 系: ``{root}/others/{category}/{key}.json``

既定の root は ``data/dev_mock``（``KEIBA_DEV_MOCK_DIR`` で変更可）。中身は
``python -m src.scripts.data.make_dev_mock``（``make dev-mock``）で生成する。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def dev_mock_root(base_dir: str | Path = ".") -> Path:
    override = os.environ.get("KEIBA_DEV_MOCK_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(base_dir) / "data" / "dev_mock"


class DevStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, category: str, key: str, id_type: str) -> Path:
        if id_type == "other":
            return self.root / "others" / category / f"{key}.json"
        if id_type == "horse":
            shard = key[:4] if len(key) >= 4 else "_"
        else:
            shard = key[:4] if len(key) >= 4 else "unknown"
        return self.root / category / shard / f"{key}.json"

    def read(self, category: str, key: str, id_type: str) -> dict[str, Any] | None:
        p = self.path(category, key, id_type)
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def write(self, category: str, key: str, id_type: str, data: dict[str, Any]) -> Path:
        p = self.path(category, key, id_type)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        tmp.replace(p)
        return p

    def exists(self, category: str, key: str, id_type: str) -> bool:
        return self.path(category, key, id_type).is_file()

    def _category_dir(self, category: str, id_type: str) -> Path:
        return self.root / "others" / category if id_type == "other" else self.root / category

    def list_keys(self, category: str, id_type: str, year: str | None = None) -> list[str]:
        d = self._category_dir(category, id_type)
        if not d.is_dir():
            return []
        if id_type == "other":
            return sorted(p.stem for p in d.glob("*.json"))
        shards = [d / year] if year else [s for s in d.iterdir() if s.is_dir()]
        return sorted(p.stem for s in shards if s.is_dir() for p in s.glob("*.json"))

    def mtimes(self, category: str, id_type: str, keys: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in keys:
            try:
                out[k] = self.path(category, k, id_type).stat().st_mtime
            except OSError:
                continue
        return out
