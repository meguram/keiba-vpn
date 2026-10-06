"""データ格納状況ダッシュボード（サーバ不要の静的 HTML）のテスト。

ペイロード生成は常に検証する。ブラウザでの動作（タブ・絞り込み・自動更新）は Playwright + Chromium があるときだけ検証する。
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.data_health import dashboard, store
from src.data_health.runner import run_health
from src.data_health.spec import RACE_CATEGORIES
from tests.data_health.test_data_health import FakeStorage, _dt, _ids, _write_race_list


@pytest.fixture()
def base(tmp_path, monkeypatch):
    """stg 要件で評価した結果（3 日目が丸ごと欠損・1 件だけ結果が欠損）を保存した保存ルート。"""
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path / "pr"))
    monkeypatch.setenv("KEIBA_CALCULATED_DATA_DIR", str(tmp_path / "calc"))
    day1, day2 = _ids([1]), _ids([2])
    _write_race_list(tmp_path, "20250201", day1)
    _write_race_list(tmp_path, "20250202", day2)
    listing = {c.name: {"2025": day1 | day2} for c in RACE_CATEGORIES}
    listing["race_result"] = {"2025": day1 | (day2 - {"202505010205"})}
    out = tmp_path / "dh"
    rep = run_health(env="stg", actual_env="stg", storage=FakeStorage(listing), now=_dt(2026, 10, 6), since=date(2025, 1, 1), until=date(2025, 12, 31),
                     root=tmp_path, infra=False, horses=False)
    store.save(rep, out)
    return out


def _data(base) -> dict:
    s = (base / "dashboard_data.js").read_text(encoding="utf-8")
    assert s.startswith("window.__DATA_HEALTH__=") and s.rstrip().endswith(";")
    return json.loads(s[len("window.__DATA_HEALTH__="):].rstrip().rstrip(";"))


def test_save_writes_static_dashboard_and_data(base):
    assert (base / "dashboard.html").is_file() and (base / "run_status.js").is_file()
    html = (base / "dashboard.html").read_text(encoding="utf-8")
    assert "dashboard_data.js" in html and "run_status.js" in html          # サーバ不要: 同じ場所のファイルを読み直すだけ
    data = _data(base)
    assert [e["key"] for e in data["envs"]] == ["stg"]
    env = data["envs"][0]
    assert env["label"].startswith("stg")
    assert env["races"]["rows"], "race_keys.csv の行が入る（レース単位の掘り下げ用）"
    codes = {c for r in env["races"]["rows"] for c in r[env["races"]["fixed"]:]}
    assert codes - {""} <= set(dashboard.STATUS_CODES), codes                       # 状態は 1 文字コードに圧縮される
    assert "m" in codes and "h" in codes


def test_payload_orders_envs_and_marks_dry_runs(base):
    for key in ("dev", "stg@dev"):                                           # 別環境の結果ディレクトリを足す
        d = base / key
        d.mkdir()
        rep = json.loads((base / "stg" / "latest.json").read_text(encoding="utf-8"))
        rep["env"] = key.split("@")[0]
        (d / "latest.json").write_text(json.dumps(rep), encoding="utf-8")
    keys = [e["key"] for e in dashboard.build_payload(base)["envs"]]
    assert keys == ["dev", "stg", "stg@dev"]
    assert "dry-run" in dashboard.env_label("stg@dev") or "@" in dashboard.env_label("stg@dev")


def test_payload_ignores_dirs_without_result(base):
    (base / "scratch").mkdir()
    (base / "broken").mkdir()
    (base / "broken" / "latest.json").write_text("{not json", encoding="utf-8")
    assert [e["key"] for e in dashboard.build_payload(base)["envs"]] == ["stg"]


def test_restriction_banner_data_is_carried(base):
    (base / "stg" / "access_restriction.json").write_text(json.dumps({"detected_at": "2026-10-06T10:00:00", "status": 403, "url": "https://x"}),
                                                          encoding="utf-8")
    store.rebuild_index(base)
    assert _data(base)["envs"][0]["restriction"]["status"] == 403


def test_run_status_written_and_finished(tmp_path):
    dashboard.write_run_status(tmp_path, "stg", "健全性の検証", 10, 100, "race_result")
    s = (tmp_path / "run_status.js").read_text(encoding="utf-8")
    assert "健全性の検証" in s and '"running"' in s
    dashboard.finish_run_status(tmp_path, "stg", "完了", "終了コード 0")
    st = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert st["stg"]["state"] == "idle" and st["stg"]["phase"] == "完了"


def test_import_report_dir_copies_side_files(base, tmp_path):
    """学習PCの結果ディレクトリごと取り込むと、レース一覧（race_keys.csv）も開発PCのダッシュボードで見られる。"""
    other = tmp_path / "other"
    store.import_report(base / "stg", other)
    assert (other / "stg" / "race_keys.csv").is_file()
    assert _data(other)["envs"][0]["races"]["rows"]


# ── ブラウザ（Playwright があるときだけ）──────────────────────────────────────────

def _browser():
    sync = pytest.importorskip("playwright.sync_api")
    p = sync.sync_playwright().start()
    try:
        b = p.chromium.launch()
    except Exception as e:  # noqa: BLE001  ブラウザ本体が無い環境
        p.stop()
        pytest.skip(f"Chromium を起動できません: {e}")
    return p, b


def test_browser_tabs_filter_and_auto_refresh(base):
    p, b = _browser()
    try:
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto((base / "dashboard.html").as_uri() + "?refresh=500")        # file:// で開く（サーバなし）。更新間隔だけ短くする
        pg.wait_for_selector("table.hm")
        assert pg.locator(".tab").count() == 1 and "stg" in pg.locator(".tab.on").inner_text()
        assert "2026-10-07" not in pg.locator("#meta").inner_text()

        # チェックが再実行されて新しい結果が書かれた → 画面を触らなくても反映される
        rep = json.loads((base / "stg" / "latest.json").read_text(encoding="utf-8"))
        rep["generated_at"] = "2026-10-07T09:00:00+09:00"
        (base / "stg" / "latest.json").write_text(json.dumps(rep), encoding="utf-8")
        store.rebuild_index(base)
        pg.wait_for_function("document.querySelector('#meta').innerText.indexOf('2026-10-07') >= 0", timeout=15_000)

        # 実行中の進捗は run_status.js の更新で出る
        dashboard.write_run_status(base, "stg", "健全性の検証", 42, 100, "race_result")
        pg.wait_for_function("document.querySelector('#runstatus').innerText.indexOf('健全性の検証') >= 0", timeout=15_000)
        assert "42 / 100" in pg.locator("#runstatus").inner_text()
        dashboard.finish_run_status(base, "stg")
        pg.wait_for_function("document.querySelector('#runstatus').innerText === ''", timeout=15_000)

        # ヒートマップのセルをクリックすると、その（カテゴリ×期間）の一覧に絞り込まれる
        cell = pg.locator("td.c[data-cat='race_result']").first
        cell.click()
        assert pg.evaluate("window.__dh.S.cat") == "race_result"
        assert not errs, errs
    finally:
        b.close()
        p.stop()
