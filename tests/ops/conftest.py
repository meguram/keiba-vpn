import pytest


@pytest.fixture(autouse=True)
def _isolated_access_hold(tmp_path, monkeypatch):
    """制限の保留ファイルをテストごとの一時ファイルに向ける（実際の data/local を汚さない）。"""
    monkeypatch.setenv("KEIBA_FETCH_HOLD_FILE", str(tmp_path / "fetch_access_hold.json"))
