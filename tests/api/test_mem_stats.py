"""メモリの内訳（src/api/mem_stats.py）と /ops/memory。外部接続なし。"""

import pytest


def test_cgroup_memory_reads_v2_files(tmp_path):
    from src.api import mem_stats as M

    (tmp_path / "memory.current").write_text("1073741824\n")
    (tmp_path / "memory.max").write_text("2147483648\n")
    (tmp_path / "memory.stat").write_text("anon 600\nfile 400\nshmem 300\nkernel 10\nother 1\n")
    m = M.cgroup_memory(tmp_path)
    assert m["current"] == 1 << 30 and m["max"] == 2 << 30 and m["ratio"] == 0.5
    assert m["shmem"] == 300 and "other" not in m
    (tmp_path / "memory.max").write_text("max\n")                 # 上限なし
    assert M.cgroup_memory(tmp_path)["ratio"] is None
    assert M.cgroup_memory(tmp_path / "none") == {"current": None, "max": None, "ratio": None}


def test_dir_bytes_counts_and_truncates(tmp_path):
    from src.api import mem_stats as M

    for i in range(5):
        (tmp_path / f"f{i}").write_bytes(b"x" * 10)
    assert M.dir_bytes(tmp_path) == {"bytes": 50, "files": 5}
    assert M.dir_bytes(tmp_path, max_files=2)["truncated"] is True
    assert M.dir_bytes(tmp_path / "none") == {"bytes": 0, "files": 0}


def test_approx_bytes_without_building_strings():
    from src.api import mem_stats as M

    assert M.approx_bytes([], 10) == 0
    one = M.obj_bytes({"a": "x" * 1000})[0]
    assert one > 1000 and M.approx_bytes([{"a": "x" * 1000}], 3) == 3 * one
    big = {"rows": [{"i": i} for i in range(10_000)]}
    n, cut = M.obj_bytes(big, max_nodes=100)
    assert cut and n > 0                                          # 大きいものは打ち切る


def test_budget_limits_total_files(tmp_path):
    from src.api import mem_stats as M

    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        for i in range(5):
            (tmp_path / d / f"f{i}").write_bytes(b"x")
    b = M.Budget(max_files=7)
    assert M.dir_bytes(tmp_path / "a", budget=b)["files"] == 5
    r = M.dir_bytes(tmp_path / "b", budget=b)
    assert r["files"] == 2 and r["truncated"] is True


def test_mirror_stats_counts_each_folder_once(tmp_path, monkeypatch):
    from src.api import mem_stats as M
    from src.ops import gcs_mirror

    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f.json").write_bytes(b"x" * 10)
    monkeypatch.setattr(gcs_mirror, "PUSH_TREES", ("x::*.json", "x::sub/*.json", "x/sub"))
    m = M.mirror_stats(tmp_path)
    assert m["_total_bytes"] == 10 and list(m) == ["x", "_total_bytes"]


def test_ops_memory_route_needs_proxy_key(monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_CLOUDRUN_LAZY", "1")
    monkeypatch.delenv("KEIBA_SERVING_STATE_PULL", raising=False)
    monkeypatch.setenv("KEIBA_PROXY_KEY_REQUIRED", "1")
    monkeypatch.setenv("KEIBA_PROXY_KEY", "k-test")
    from src.api import cloudrun_entry

    c = TestClient(cloudrun_entry.build_app())
    assert c.get("/ops/memory").status_code == 401
    r = c.get("/ops/memory", headers={"X-Keiba-Proxy-Key": "k-test"})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    d = r.json()
    assert {"cgroup", "rss", "storage", "tmp", "mirror", "uptime_s"} <= set(d)
    assert "instances" in d["storage"] and "prerender" in d["tmp"]


if __name__ == "__main__":
    pytest.main([__file__])
