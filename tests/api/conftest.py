"""tests/api の共通の準備。"""
import sys

import pytest


@pytest.fixture(autouse=True)
def _fresh_confirmed_marks(monkeypatch):
    """結果が出た印（src/api/app.py の _result_levels）を試験ごとに空にする。
    ほかの試験が result-status・レース詳細を通って付けた印で、HTTP キャッシュの見出しの期待が試験の順番で変わらないように。"""
    app_mod = sys.modules.get("src.api.app")
    if app_mod is not None:
        monkeypatch.setattr(app_mod, "_result_levels", {})
    yield
