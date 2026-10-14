"""CI スモークチェック(smoke_endpoints)の部品のテスト。サーバは使わない。"""

from __future__ import annotations

from src.scripts.ci import smoke_endpoints as S


def test_check_records_failure_instead_of_raising():
    def boom() -> str:
        raise AssertionError("空です")

    r = S.check("x", boom)
    assert not r.ok and r.detail == "空です"
    assert S.check("y", lambda: "ok").ok


def test_unreachable_server_gives_exit_code_2(capsys):
    assert S.main(["--base", "http://127.0.0.1:1", "--wait", "0"]) == 2
    assert "::error" in capsys.readouterr().out


def test_fixture_manifest_is_readable():
    date, races = S.fixture_info()
    assert len(date) == 8 and races > 0
