"""定期実行ジョブの一覧のページ（docs/html/operations/final/scheduled_jobs.html）と、その生成器の検査。

ページは config/scheduled_jobs.json（＋config/gcp_job_map.json）から作る。手で直したり、JSON を直して作り直していなかったりすると (a) で落ちる。
直し方: python -m src.scripts.docs.build_scheduled_jobs_html
"""

from __future__ import annotations

import html
import json
import re

import pytest

from src.scripts.docs import build_scheduled_jobs_html as gen

PAGE = gen.DEFAULT_OUT


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(gen.JOBS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built() -> str:
    return gen.build()


def test_page_matches_generator(built: str) -> None:
    """(a) 置いてある HTML が生成結果と一致する（手で直すとずれる）。"""
    assert PAGE.exists(), f"{PAGE} が無い"
    assert PAGE.read_text(encoding="utf-8") == built, (
        "docs/html/operations/final/scheduled_jobs.html が生成結果と一致しない。"
        "python -m src.scripts.docs.build_scheduled_jobs_html で作り直す（手で直さない）"
    )


def test_build_is_deterministic(built: str) -> None:
    assert gen.build() == built


def test_check_flag(tmp_path, built: str) -> None:
    """--check は一致なら 0、違う・無いなら 1。比べるだけで書かない。"""
    ok = tmp_path / "ok.html"
    ok.write_text(built, encoding="utf-8")
    assert gen.main(["--check", "--out", str(ok)]) == 0
    bad = tmp_path / "bad.html"
    bad.write_text("garbage", encoding="utf-8")
    assert gen.main(["--check", "--out", str(bad)]) == 1
    assert bad.read_text(encoding="utf-8") == "garbage"
    missing = tmp_path / "missing.html"
    assert gen.main(["--check", "--out", str(missing)]) == 1
    assert not missing.exists()


def test_all_jobs_and_names_appear(built: str, data: dict) -> None:
    """(b) JSON の全ジョブの id と、タイマー名・Scheduler 名などがページに出る。"""
    for j in data["jobs"]:
        assert html.escape(j["id"]) in built, j["id"]
        assert f'id="job-{j["id"]}"' in built, f"{j['id']} の詳細（details）が無い"
        for key in ("timer", "service", "scheduler", "run_job", "group", "cron_tag", "job_id"):
            name = (j.get("ref") or {}).get(key)
            if name:
                assert html.escape(name) in built, f"{j['id']}: ref.{key}={name}"
    for r in data.get("retired") or []:
        assert html.escape(r["id"]) in built


def test_audit_items_appear(built: str, data: dict) -> None:
    """(c) 点検（audit）の全項目が出る（指摘は改行を <br> にするので行ごとに見る）。"""
    audit_html = built[built.index('id="audit"'):]
    for a in data.get("audit") or []:
        assert f"<td>{html.escape(str(a['no']))}</td>" in audit_html
        for key in ("finding", "advice", "basis"):
            for line in str(a.get(key) or "").split("\n"):
                assert html.escape(line) in audit_html, f"点検 #{a['no']} の {key}"
        if a.get("status"):
            assert html.escape(a["status"]) in audit_html
    for key in ("audit_note", "audit_ok"):
        if data.get(key):
            assert html.escape(data[key]) in built


def test_no_external_resources(built: str) -> None:
    """(d) 外部の CDN・スクリプト・スタイルを読まない（取得先の URL の例は文字として出すだけ）。"""
    assert "<script" not in built.lower()
    assert not re.search(r"<link[^>]+href=[\"']?(https?:)?//", built, re.I)
    assert not re.search(r"@import|url\(\s*[\"']?(https?:)?//", built, re.I)
    assert not re.search(r"<(img|iframe)[^>]+src=[\"']?(https?:)?//", built, re.I)
    assert not re.search(r"<a[^>]+href=[\"']?https?://", built, re.I), "URL の例はリンクにしない（叩いていない形の例）"


@pytest.mark.parametrize("sid", ["summary", "gcp-live", "vps-live", "future", "detail", "audit"])
def test_section_ids(built: str, sid: str) -> None:
    """(e) 見出しの id（ほかのページからのリンク先）がある。"""
    assert f'<h2 id="{sid}">' in built


def test_tables_are_scrollable(built: str) -> None:
    """スマホで横にはみ出さないよう、表はすべて横スクロールの div で包む。"""
    assert built.count("<table") == built.count('<div class="scroll"><table>')


def test_no_current_time_in_output(built: str, data: dict) -> None:
    """日付は JSON の updated だけ（生成した時刻を入れると --check が毎回ずれる）。"""
    assert html.escape(str(data["updated"])) in built
    assert not re.search(r"\d{1,2}:\d{2}:\d{2}", built.split("<style>")[0])


@pytest.mark.parametrize(
    "expr, words",
    [
        ("30 7 * * *", "毎日 07:30"),
        ("0 10 * * 1", "月 10:00"),
        ("10,40 19 * * *", "毎日 19:10・19:40"),
        ("5 * * * *", "毎時 05 分"),
        ("*/15 * * * *", "15 分おき"),
        ("* 8-16 * * 0,1,6", "土日月 08:00〜16:59 毎分"),
        ("*/20 12-21 * * 0,1,2,6", "土日月火 12:00〜21:40 20 分おき"),
        ("0 0 1 * *", "0 0 1 * *"),  # 読めない形はそのまま
    ],
)
def test_cron_text(expr: str, words: str) -> None:
    assert gen.cron_text(expr) == words
