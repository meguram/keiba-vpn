"""メモリの内訳（src/api/mem_stats.py）と /ops/memory。外部接続なし。"""

import pytest


def test_cgroup_memory_reads_v2_files(tmp_path):
    from src.api import mem_stats as M

    (tmp_path / "memory.current").write_text("1073741824\n")
    (tmp_path / "memory.max").write_text("2147483648\n")
    (tmp_path / "memory.stat").write_text("anon 600\nfile 400\nshmem 300\nkernel 10\nother 1\n")
    none = tmp_path / "no-meminfo"
    m = M.cgroup_memory(tmp_path, none)
    assert m["current"] == 1 << 30 and m["max"] == 2 << 30 and m["source"] == "cgroup2"
    assert m["used"] == (1 << 30) - 100 and m["ratio"] == round(((1 << 30) - 100) / (2 << 30), 4)   # 回収できる file 100（400-300）は除く
    assert m["shmem"] == 300 and "other" not in m
    (tmp_path / "memory.max").write_text("max\n")                 # 上限なし
    assert M.cgroup_memory(tmp_path, none)["ratio"] is None
    assert M.cgroup_memory(tmp_path / "none", none) == {"current": None, "max": None, "ratio": None, "source": None, "used": None}


def test_cgroup_v1_and_meminfo_fallbacks(tmp_path):
    from src.api import mem_stats as M

    v1 = tmp_path / "v1"
    (v1 / "memory").mkdir(parents=True)
    (v1 / "memory" / "memory.usage_in_bytes").write_text("536870912\n")
    (v1 / "memory" / "memory.limit_in_bytes").write_text("2147483648\n")
    m = M.cgroup_memory(v1, tmp_path / "none")
    assert m["source"] == "cgroup1" and m["ratio"] == 0.25
    st = v1 / "memory" / "memory.stat"
    st.write_text("cache 300\nrss 200\nshmem 50\ntotal_active_file 134217728\ntotal_inactive_file 134217728\n")
    m = M.cgroup_memory(v1, tmp_path / "none")
    assert m["used"] == 536870912 - 268435456 and m["ratio"] == 0.125 and m["file_lru"] == 268435456   # ファイルの LRU は除く
    assert m["anon"] == 200 and m["shmem"] == 50 and m["used_basis"] == "usage-file_lru"
    st.write_text("active_file 100\ninactive_file 268435356\n")                  # total_ の無い形
    assert M.cgroup_memory(v1, tmp_path / "none")["used"] == 536870912 - 268435456
    st.write_text("active_file 600000000\ninactive_file 1\n")                    # 引きすぎても負にならない
    assert M.cgroup_memory(v1, tmp_path / "none")["used"] == 0
    st.write_text("garbage\ninactive_file x\n")                                   # 読めない形: 多く見積もる側（usage のまま）と印
    m = M.cgroup_memory(v1, tmp_path / "none")
    assert m["used"] == 536870912 and m["used_basis"] == "usage"
    mi0 = tmp_path / "meminfo0"
    mi0.write_text("MemTotal:        2097152 kB\nMemAvailable:     209715 kB\n")  # VM 全体では 90% 使っている
    st.write_text("active_file 0\ninactive_file 268435456\n")
    m = M.cgroup_memory(v1, mi0)                                                # cgroup の前提が外れても点検が止まらない
    assert m["ratio"] == m["meminfo_ratio"] == 0.9 and m["ratio_basis"] == "meminfo"
    st.unlink()
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal:        2097152 kB\nMemFree:          100000 kB\nMemAvailable:     524288 kB\nShmem:            300000 kB\n")
    m = M.cgroup_memory(tmp_path / "empty", mi)                  # Cloud Run（gen2）: cgroup のファイルが無い
    assert m["source"] == "meminfo" and m["max"] == 2097152 * 1024 and m["ratio"] == 0.75 and m["shmem"] == 300000 * 1024
    (v1 / "memory" / "memory.limit_in_bytes").write_text(str(1 << 62))     # v1 の上限なし → meminfo へ
    assert M.cgroup_memory(v1, mi)["source"] == "meminfo"


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
