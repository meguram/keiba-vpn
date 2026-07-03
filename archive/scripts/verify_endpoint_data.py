#!/usr/bin/env python3
"""エンドポイントごとにローカルデータ有無と API 応答を検証する（モンキーテスト）。"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

# 認証オフ。GCS は .env があれば有効（成長曲線など horse_result 参照に必要）
os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("GCS_BUCKET", "")
os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")

sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from src.api.app import app  # noqa: E402
from src.config import data_paths as dp  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

FAKE_RACE = "202501010101"
FAKE_HORSE = "2020100001"
SAMPLE_DATE = "20230618"  # race_lists に存在する日付


@dataclass
class Check:
    name: str
    data_paths: list[str]
    method: str
    url: str
    expect_ok: bool = True  # 200 かつ JSON が理想
    min_json_keys: tuple[str, ...] = ()


def _exists(rel: str) -> bool:
    p = ROOT / rel
    return p.exists() and (p.is_file() or any(p.iterdir()))


def _data_status(paths: list[str]) -> str:
    if not paths:
        return "—"
    ok = [_exists(p) for p in paths]
    if all(ok):
        return "✅"
    if any(ok):
        return "⚠️"
    return "❌"


def _run(c: Check) -> dict:
    r = client.request(c.method, c.url)
    body = {}
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text[:200]}
    ok = r.status_code == 200 and isinstance(body, dict)
    if ok and c.min_json_keys:
        ok = all(k in body for k in c.min_json_keys)
    return {
        "status": r.status_code,
        "ok": ok if c.expect_ok else r.status_code not in (500, 502, 503),
        "keys": list(body.keys())[:8] if isinstance(body, dict) else [],
        "error": body.get("error") if isinstance(body, dict) else None,
    }


CHECKS: list[Check] = [
    Check("health", [], "GET", "/api/health", min_json_keys=("status",)),
    Check("auth", [], "GET", "/api/auth/status", min_json_keys=("is_developer",)),
    Check("predictions", ["data/calculated_data/predictions/predictions.json"], "GET", "/api/predictions"),
    Check("gcs-stats", [], "GET", "/api/gcs-stats", expect_ok=False),
    Check("scrape-dates", ["data/calculated_data/race_lists"], "GET", "/api/scrape-dates", min_json_keys=("dates",)),
    Check("race-list", [f"data/calculated_data/race_lists/{SAMPLE_DATE}.json"], "GET", f"/api/race-list/{SAMPLE_DATE}", min_json_keys=("races",)),
    Check("course-profiles", ["data/calculated_data/knowledge/course_profiles.json"], "GET", "/api/course-profiles"),
    Check("cushion-data", ["data/calculated_data/cushion/cushion_values.json"], "GET", "/api/cushion/data", min_json_keys=("records",)),
    Check("cushion-stats", ["data/calculated_data/cushion/cushion_values.json"], "GET", "/api/cushion/stats", min_json_keys=("total",)),
    Check("myostatin", ["data/calculated_data/knowledge/myostatin_genes.json"], "GET", "/api/myostatin?q=ディープ"),
    Check("bloodline-turf", ["data/calculated_data/bloodline/by_surface/turf/sire_clusters.csv"], "GET", "/api/bloodline/data/clusters?surface=turf", min_json_keys=("rows",)),
    Check("bloodline-surfaces", ["data/calculated_data/bloodline"], "GET", "/api/bloodline/surfaces"),
    Check("bloodline-cluster-meta", ["data/calculated_data/note_aptitude_race"], "GET", "/api/bloodline-cluster/meta"),
    Check("bloodline-cluster-clusters", ["data/calculated_data/note_aptitude_race"], "GET", "/api/bloodline-cluster/clusters"),
    Check("track-speed-meta", ["data/calculated_data/track_speed/meta.json"], "GET", "/api/track-speed/meta"),
    Check("track-speed-dates", ["data/calculated_data/track_speed"], "GET", "/api/track-speed/dates"),
    Check("pedigree-note-aptitude", ["data/calculated_data/knowledge/sire_aptitude_note.json"], "GET", "/api/pedigree/note-aptitude"),
    Check("pedigree-map", ["data/calculated_data/pedigree_map"], "GET", "/api/pedigree-map", expect_ok=False),
    Check("stallion-sire-tree", ["data/calculated_data/pedigree_race_index"], "GET", "/api/stallion-sire-tree", expect_ok=False),
    Check("horse-names-search", ["data/calculated_data/knowledge/horse_name_index.json"], "GET", "/api/horse-names/search?q=ト"),
    Check(
        "growth-curve",
        ["data/calculated_data/knowledge/horse_name_index.json"],
        "GET",
        "/api/growth-curve/2020101543?fetch_speed_index=false",
        min_json_keys=("horse_name", "races"),
    ),
    Check("growth-curve-status", ["data/calculated_data/growth_curve"], "GET", "/api/growth-curve/status"),
    Check("race-api", [], "GET", f"/api/race/{FAKE_RACE}", expect_ok=False),
    Check("race-predictions", [], "GET", f"/api/race/{FAKE_RACE}/predictions", expect_ok=False),
    Check("admin-system-stats", [], "GET", "/api/admin/system-stats", expect_ok=False),
]


def main() -> int:
    print("=== calculated_data ルート ===")
    print(dp.calculated_data_root())
    print()
    print(f"{'endpoint':<28} {'data':<4} {'HTTP':<5} {'result':<6} 備考")
    print("-" * 72)

    fails = 0
    for c in CHECKS:
        data_st = _data_status(c.data_paths)
        res = _run(c)
        mark = "OK" if res["ok"] else "NG"
        if not res["ok"]:
            fails += 1
        note = res.get("error") or ",".join(res.get("keys") or [])[:40]
        print(f"{c.name:<28} {data_st:<4} {res['status']:<5} {mark:<6} {note}")

    print()
    print(f"完了: {len(CHECKS) - fails}/{len(CHECKS)} 合格")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
