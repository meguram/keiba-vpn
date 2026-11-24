"""追い切り表の重ね合わせ（build_training_laps_merged）: 馬 × レースの日の組が手元にあれば手元の行だけ、無ければ固定の表の行を残す。"""

import json

import numpy as np
import pandas as pd

from src.scripts.data import build_training_laps_merged as M


def _entry(race_info, date, t, course="美坂"):
    return {"race_info": race_info, "date": date, "course": course, "time_raw": " ".join(map(str, t)), "lap_times": t,
            "leg_color": "馬也", "rank": "B", "evaluation": "", "position": "", "rider": "助手", "track_condition": "良"}


def test_local_laps_skips_broken_files(tmp_path):
    root = tmp_path / "horse_training" / "2024"
    root.mkdir(parents=True)
    rec = {"horse_id": "2024100001", "entries": [_entry("2026/11/22\xa0\xa0東京1R 2歳未勝利", "2026-11-18", [53.0, 38.5, 25.0, 12.4])]}
    (root / "2024100001.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    (root / "broken.json").write_text("{", encoding="utf-8")
    (root / "list.json").write_text("[1, 2]", encoding="utf-8")         # 形の違う記録（rec.get で落ちる）
    local, bad = M.local_laps(tmp_path / "horse_training")
    assert bad == 2 and len(local) == 1


def test_merge_replaces_whole_group_and_keeps_fresh_only_groups():
    cols = ["horse_id", "train_date", "race_date", "course", "kind", "t4", "t1", "rider"]
    fresh = pd.DataFrame([
        ["h1", "2026-03-04", "2026-03-08", "美坂", "slope", 53.0, 12.4, "古い版"],
        ["h1", "2026-03-04", "2026-03-08", "美プール", "other", np.nan, np.nan, "古い版"],   # 時計の無い種別
        ["h1", "2026-03-04", "2026-03-08", "美プール", "other", np.nan, np.nan, "古い版"],   # 固定の表の中の重複（別の組なら残す）
        ["h2", "2025-01-08", "2025-01-12", "栗坂", "slope", 52.0, 12.1, "古い版"],
        ["h2", "2025-01-08", "2025-01-12", "栗坂", "slope", 52.0, 12.1, "古い版"],
    ], columns=cols)
    local = pd.DataFrame([
        ["h1", "2026-03-04", "2026-03-08", "美坂", "slope", 53.1, 12.4, "新しい版"],          # 時計が少し違う
        ["h1", "2026-02-26", "2026-03-08", "美W", "wood", 67.0, 11.9, "新しい版"],
    ], columns=cols)
    m = M.merge(fresh, local)
    assert list(m.columns) == cols
    g1 = m[m["horse_id"] == "h1"]
    assert len(g1) == 2 and set(g1["rider"]) == {"新しい版"}            # 組ごと手元の行に置き換え（二重に残らない）
    assert len(m[m["horse_id"] == "h2"]) == 2                           # 手元に無い組は固定の表のまま（重複にも手を付けない）
