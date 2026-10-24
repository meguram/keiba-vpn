"""ホスト依存の機械検査（オフライン）。"""

import json
from pathlib import Path

from src.scripts.ops import host_dependency_scan as H


def _mk(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_scan_detects_each_kind_only_in_operational_paths(tmp_path):
    _mk(tmp_path, "src/api/a.py", 'P="/home/jovyan/x"\nU="http://127.0.0.1:23220/api"\nimport psycopg2\nd="data/local/meta/x"\n')
    _mk(tmp_path, "src/ops/b.py", 'H="keiba-remote"\nA="https://api-internal.megu-keiba.com"\n')
    _mk(tmp_path, "src/scripts/data/tp99_eval.py", 'OUT="/home/jovyan/work/tp99_out"\n')  # 研究用は対象外
    _mk(tmp_path, "tests/test_x.py", 'x="/home/jovyan"\n')
    kinds = {(f["file"], f["kind"]) for f in H.scan(tmp_path)}
    assert ("src/api/a.py", "host_path") in kinds and ("src/api/a.py", "host_port") in kinds
    assert ("src/api/a.py", "postgres") in kinds and ("src/api/a.py", "local_data") in kinds
    assert ("src/ops/b.py", "host_alias") in kinds and ("src/ops/b.py", "api_internal") in kinds
    assert not any(f.startswith(("src/scripts/data", "tests")) for f, _ in kinds)


def test_compare_flags_only_increases(tmp_path):
    _mk(tmp_path, "src/api/a.py", 'a="/home/jovyan"\nb="/home/jovyan/2"\n')
    cur = H.counts(H.scan(tmp_path))
    assert cur == {"src/api/a.py": {"host_path": 2}}
    assert H.compare(cur, {"src/api/a.py": {"host_path": 2}}) == []          # 同じ件数は許容
    assert H.compare(cur, {"src/api/a.py": {"host_path": 3}}) == []          # 減るのは許容
    assert H.compare(cur, {"src/api/a.py": {"host_path": 1}})                # 増えたら問題
    assert H.compare(cur, {})                                                # 新しいファイルに出たら問題


def test_main_update_baseline_and_check(tmp_path, capsys):
    _mk(tmp_path, "src/api/a.py", 'a="/home/jovyan"\n')
    base = tmp_path / "base.json"
    assert H.main(["--root", str(tmp_path), "--baseline", str(base), "--update-baseline"]) == 0
    assert json.loads(base.read_text())["counts"] == {"src/api/a.py": {"host_path": 1}}
    assert H.main(["--root", str(tmp_path), "--baseline", str(base)]) == 0
    _mk(tmp_path, "src/api/b.py", 'x="keiba-remote"\n')
    assert H.main(["--root", str(tmp_path), "--baseline", str(base)]) == 1
    assert "増え" in capsys.readouterr().err


def test_repo_has_no_new_host_dependencies():
    """運用経路のコードに、ベースラインを超えるホスト固有の参照を新しく足していない。"""
    problems = H.compare(H.counts(H.scan(H.ROOT, tracked_only=True)), H.load_baseline())  # 作業中の未コミットは見ない
    assert problems == [], "\n".join(problems)
