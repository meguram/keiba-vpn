from pathlib import Path

from src.scripts.ops import autonomy_check as ac


def _tree(root: Path):
    (root / "src" / "jobs").mkdir(parents=True)
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "jobs" / "__init__.py").write_text("")
    (root / "src" / "jobs" / "run.py").write_text("import os\nimport foo\nfrom src.jobs import helper\n")
    (root / "src" / "jobs" / "helper.py").write_text("def f():\n    import bar\n    return bar\ntry:\n    import baz\nexcept ImportError:\n    baz = None\n")


def test_closure_separates_always_loaded_from_conditional_imports(tmp_path):
    _tree(tmp_path)
    ext, _ = ac.closure("src.jobs.run", tmp_path)
    assert ext == {"foo": True, "bar": False, "baz": False}          # 標準ライブラリ(os)は含まない


def test_check_deps_reports_hard_and_soft_gaps_only_for_gcp_jobs(tmp_path):
    _tree(tmp_path)
    mapping = {"foo": ["foo"], "bar": ["bar-dist"], "baz": ["baz"]}
    jobs = [{"id": "a", "destination": "gcp", "ready": True, "steps": [{"argv": ["-m", "src.jobs.run"]}]},
            {"id": "b", "destination": "vps", "ready": True, "steps": [{"argv": ["-m", "src.jobs.run"]}]}]
    rows = ac.check_deps(jobs, {"baz"}, tmp_path, mapping)
    assert [r["id"] for r in rows] == ["a"]
    assert rows[0]["missing_hard"] == ["foo"] and rows[0]["missing_soft"] == ["bar-dist"]
    assert ac.check_deps(jobs, {"foo", "bar-dist", "baz"}, tmp_path, mapping)[0]["missing_hard"] == []


def test_requirement_names_skip_comments_options_and_excluded(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("# c\n-r other.txt\npandas>=2.0  # x\nScikit_Learn[extra]>=1\nmlflow>=2\nplaywright\n")
    assert ac.requirement_names(f, ac.JOBS_EXCLUDE) == {"pandas", "scikit-learn"}


def test_expand_with_dependencies_follows_installed_packages():
    out = ac.expand_with_dependencies({"requests"})
    assert {"requests", "urllib3"} <= out


def test_cron_coverage_finds_jobs_missing_from_the_map():
    cron = "# KEIBA\n30 22 * * * cd x && bash scripts/cron/cron_run.sh raceday-runner -- bash y\n0 3 * * * bash scripts/cron/cron_run.sh new-job -- z\n# 0 1 * * * cron_run.sh commented -- q\n"
    names = ac.cron_names(cron)
    assert names == {"raceday-runner", "new-job"}
    jobs = [{"id": "raceday-runner"}, {"id": "other", "origin": {"cron_job_id": "x"}}]
    assert ac.check_coverage(names, jobs) == ["new-job"]
