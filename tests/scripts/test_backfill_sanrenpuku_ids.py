"""backfill_sanrenpuku_odds の --race-ids-file（指定のレースだけ補う）。"""
import pytest

from src.scripts.data.backfill_sanrenpuku_odds import race_ids_from_file


def test_race_ids_from_file_skips_comments_and_dups(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("# 説明\n202601010202\n\n 202605050503 \n202601010202\n", encoding="utf-8")
    assert race_ids_from_file(f) == ["202601010202", "202605050503"]


def test_race_ids_from_file_rejects_bad_line(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("202601010202\n2026-01-01\n", encoding="utf-8")
    with pytest.raises(ValueError):
        race_ids_from_file(f)


def test_committed_list_is_valid():
    from pathlib import Path

    ids = race_ids_from_file(Path(__file__).resolve().parents[2] / "docs/operations/retrain_trials/sanrenpuku_fill_2026_ids.txt")
    assert len(ids) == 117 and all(r.startswith("2026") for r in ids)


def test_empty_or_bom_file(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("# だけ\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        race_ids_from_file(f)
    f.write_text("\ufeff202601010202\n", encoding="utf-8")
    assert race_ids_from_file(f) == ["202601010202"]


def test_file_does_not_list_years(tmp_path, monkeypatch):
    from src.scripts.data import backfill_sanrenpuku_odds as m

    def boom(y):
        raise AssertionError("年の一覧を作った")

    monkeypatch.setattr(m, "race_ids_for_year", boom)
    f = tmp_path / "ids.txt"
    f.write_text("202601010202\n", encoding="utf-8")
    assert m.target_ids(f, None) == ["202601010202"]
    monkeypatch.setattr(m, "race_ids_for_year", lambda y: [f"{y}01010101"])
    assert m.target_ids(None, None) == [f"{y}01010101" for y in m.DEFAULT_YEARS]
