from src.scripts.maintenance.repair_race_result_header import needs_repair, patch_header


def test_needs_repair_only_when_course_missing():
    assert needs_repair({"surface": "", "distance": 0})
    assert not needs_repair({"surface": "芝", "distance": 1200})
    assert not needs_repair({"surface": "障", "distance": 0})
    assert not needs_repair(None)


def test_patch_changes_only_course_fields():
    old = {"race_id": "r", "surface": "", "direction": "", "distance": 0, "weather": "曇", "entries": [1, 2], "_meta": {"scraped_at": 1}}
    out = patch_header(old, {"surface": "芝", "direction": "右", "distance": 1200, "weather": "雨"})
    assert (out["surface"], out["direction"], out["distance"]) == ("芝", "右", 1200)
    assert out["weather"] == "曇" and out["entries"] == [1, 2] and out["_meta"]["scraped_at"] == 1
    assert "header_repaired_at" in out["_meta"]
    assert old["surface"] == ""   # 元は変えない


def test_unreadable_new_returns_none():
    assert patch_header({"surface": ""}, {"surface": "", "distance": 0}) is None
