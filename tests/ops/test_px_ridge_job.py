"""px-ridge-window（src/ops/jobs/px_ridge.py）: 次の半年の 30 日前から、無ければ作る。"""
from datetime import date

from src.ops.jobs import px_ridge as PR


def test_next_window():
    assert PR.next_window(date(2026, 10, 26)) == ("2027-01-01", 67)
    assert PR.next_window(date(2026, 6, 30)) == ("2026-07-01", 1)
    assert PR.next_window(date(2026, 7, 1)) == ("2027-01-01", 184)


def test_skip_when_early_or_already_exists_and_error_without_svd(tmp_path):
    assert PR.run(date(2026, 10, 26), tmp_path)["status"] == "skipped"                 # 67 日先 → まだ早い
    d = date(2026, 12, 10)                                                              # 22 日前
    assert PR.run(d, tmp_path)["status"] == "error"                                     # svd.npz が無い
    (tmp_path / "svd.npz").write_bytes(b"x")
    (tmp_path / "ridge_2027-01-01.npz").write_bytes(b"x")
    assert PR.run(d, tmp_path)["reason"].endswith("既にある")


def test_builds_when_due_and_checks_output(tmp_path):
    (tmp_path / "svd.npz").write_bytes(b"x")
    calls = []
    assert PR.run(date(2026, 12, 10), tmp_path, runner=lambda t0: calls.append(t0))["status"] == "error"        # 作ったはずが無い
    assert calls == ["2027-01-01"]
    res = PR.run(date(2026, 12, 10), tmp_path, runner=lambda t0: (tmp_path / f"ridge_{t0}.npz").write_bytes(b"r"))
    assert res == {"status": "ok", "created": "ridge_2027-01-01.npz", "t0": "2027-01-01"}
