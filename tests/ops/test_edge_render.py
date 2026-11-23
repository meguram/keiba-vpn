"""edge_render（VPS のフォルダに置く API の応答を、ホストが手元のデータだけで作る）の守りのテスト。API の app は読み込まない。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from src.ops import edge_render as er

ROOT = Path(__file__).resolve().parents[2]


def _deploy_env_vars() -> dict[str, str]:
    """scripts/gcp/deploy_cloud_run_api.sh の ENV_VARS のうち、値が固定のもの。"""
    text = (ROOT / "scripts" / "gcp" / "deploy_cloud_run_api.sh").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for m in re.finditer(r'^ENV_VARS\+?="([^"]*)"', text, re.M):
        for kv in m.group(1).split(","):
            if "=" in kv and "$" not in kv:
                k, v = kv.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def test_render_env_matches_cloud_run():
    """応答に効く環境変数が、Cloud Run の配備の値と食い違わない。"""
    cfg = er.load_config()
    dep = _deploy_env_vars()
    env = er.render_env(cfg, "prod")
    flags = [k for k in dep if k.startswith("KEIBA_") and k not in {
        "KEIBA_PROXY_KEY_REQUIRED", "KEIBA_SERVING_STATE_PULL", "KEIBA_MAINTENANCE_WINDOW", "KEIBA_PRERENDER",
        "KEIBA_PRERENDER_MAX_AGE_SEC", "KEIBA_WARMUP_ON_START"}]
    assert flags, "ENV_VARS が読めない"
    for k in flags:
        assert env.get(k) == dep[k], f"{k}: edge_render={env.get(k)!r} Cloud Run={dep[k]!r}"
    # 応答に効くフラグを Cloud Run に足したら、ここに無いと落ちる
    assert "KEIBA_STAGE2_START" in env and env["KEIBA_STAGE2_START"] == ""
    assert er.render_env(cfg, "stg")["KEIBA_STAGE2_START"]


def test_stg_stage2_start_matches_deploy():
    text = (ROOT / "scripts" / "ci" / "deploy_gcp.sh").read_text(encoding="utf-8")
    m = re.search(r"STG_STAGE2_START:-(\d{8})", text)
    assert m and er.render_env(er.load_config(), "stg")["KEIBA_STAGE2_START"] == m.group(1)


def test_clean_environ_drops_host_env(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "real-bucket")
    monkeypatch.setenv("KEIBA_DEV_AUTO_LOGIN", "1")
    monkeypatch.setenv("KEIBA_SOMETHING", "x")
    saved = dict(__import__("os").environ)
    try:
        er.apply_clean_environ(er.load_config(), "prod", "s" * 64)
        import os

        assert os.environ["GCS_BUCKET"] == "edge-render-no-gcs"          # 無いバケット。GCS に行けば守りに当たる
        assert os.environ["KEIBA_DEV_AUTO_LOGIN"] == "0"
        assert os.environ["KEIBA_SKIP_DOTENV"] == "1"
        assert os.environ["KEIBA_CLOUDRUN_LAZY"] == "1"
        assert "KEIBA_SOMETHING" not in os.environ
        assert "KEIBA_STAGE2_START" not in os.environ          # 空の値は未設定
        assert os.environ["KEIBA_ENV"] == "prod"
    finally:
        import os

        os.environ.clear()
        os.environ.update(saved)


def test_eligible_dates_respects_mirror_since_and_grace():
    cfg = {"mirror_since": "20261123", "entries_lead_days": 5, "window_days": 35, "settled_grace_days": 1}
    today = datetime(2026, 11, 30, 10, tzinfo=er.JST)
    assert er.eligible_dates(cfg, today) == ["20261128", "20261129"]
    assert er.eligible_dates(cfg, datetime(2026, 11, 27, tzinfo=er.JST)) == []
    late = er.eligible_dates(cfg, datetime(2027, 3, 1, tzinfo=er.JST))
    assert late[0] == "20270125" and late[-1] == "20270228"     # 35 日の窓


def _mirror(tmp: Path, cats: list[str]) -> Path:
    (tmp / "data" / "local" / "meta").mkdir(parents=True)
    (tmp / "data" / "local" / "meta" / "local_mirror_config.json").write_text(json.dumps({"enabled": True, "categories": cats}))
    return tmp


def _put(base: Path, cat: str, key: str, data: dict) -> None:
    p = base / "data" / "local" / "mirror" / cat / key[:4] / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def test_mirror_store_flags(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_EDGE_MIRROR_DIR", raising=False)
    base = _mirror(tmp_path, ["race_shutuba", "horse_result"])
    _put(base, "race_shutuba", "202611280101", {"a": 1})
    fb_root = base / "data" / "local" / "horse_pedigree_5gen" / "2019"
    fb_root.mkdir(parents=True)
    (fb_root / "2019100001.json").write_text('{"ped": 1}')
    store = er.make_mirror_store(base, er.mirrored_categories(base), {"horse_pedigree_5gen": "data/local/horse_pedigree_5gen"})
    er.FLAGS.clear()
    assert store.read("race_shutuba", "202611280101", "race") == {"a": 1}
    assert store.read("race_shutuba", "202611280102", "race") is None          # レースの欠けは「無い」（対象の日を絞ってある）
    assert er.FLAGS.reasons == []
    assert store.read("horse_pedigree_5gen", "2019100001", "horse") == {"ped": 1}
    assert er.FLAGS.reasons == []
    assert store.read("horse_result", "2019100001", "horse") is None             # 馬の欠けは「分からない」
    assert er.FLAGS.reasons == ["写しに無い: horse_result"]
    er.FLAGS.clear()
    assert store.read("race_odds", "202611280101", "race") is None               # 写しの対象外
    assert store.exists("race_odds", "202611280101", "race") is False
    assert store.list_keys("race_odds", "race", "2026") == []
    assert all(r == "写しの対象外: race_odds" for r in er.FLAGS.reasons) and er.FLAGS.reasons
    with pytest.raises(er.EdgeGuardError):
        store.write("race_shutuba", "202611280101", "race", {})
    # 有無・更新時刻も、別の置き場を見て、馬のキーが無ければ印を立てる（「先に有無を聞いてから読む」処理が黙って空にならない）
    er.FLAGS.clear()
    assert store.exists("horse_pedigree_5gen", "2019100001", "horse") is True
    assert set(store.mtimes("horse_pedigree_5gen", "horse", ["2019100001"])) == {"2019100001"}
    assert er.FLAGS.reasons == []
    assert store.exists("horse_result", "2019100009", "horse") is False
    assert store.mtimes("horse_result", "horse", ["2019100009"]) == {}
    assert er.FLAGS.reasons == ["写しに無い: horse_result", "写しに無い: horse_result"]
    er.FLAGS.clear()
    assert store.list_keys("horse_result", "horse") == []                       # 馬の一覧は写しでは欠ける
    assert er.FLAGS.reasons
    er.FLAGS.clear()
    assert store.list_keys("race_shutuba", "race", "2026") == ["202611280101"]
    assert er.FLAGS.reasons == []


def test_dirty_keys_spread_to_race_list(tmp_path, monkeypatch):
    rl = tmp_path / "race_lists"
    rl.mkdir()
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path))
    rl.joinpath("20261128.json").write_text(json.dumps({"races": [{"race_id": "202605050101"}, {"race_id": "202605050102"}]}))
    rl.joinpath("20261129.json").write_text(json.dumps({"races": [{"race_id": "202605050201"}]}))
    targets = [("anon", "/api/race-list/20261128"), ("anon", "/api/race-list/20261129"),
               ("anon", "/api/race/202605050101"), ("anon", "/api/race/202605050101/predictions"), ("anon", "/api/race/202605050201")]
    bad = er.dirty_keys(tmp_path, targets, [("anon", "/api/race/202605050101/predictions")])
    assert bad == {"202605050101", "20261128"}                                   # 同じレースのほかの対象と、そのレースの日の一覧も捨てる
    man: dict[str, str] = {}
    for _, p in targets:
        er.write_if_changed(tmp_path / "out", er.out_rel("anon", p), b"x", man)
    fps = dict.fromkeys(man, "f")
    all_rel = {er.out_rel(t, p) for t, p in targets}
    n = er.prune_outputs(tmp_path / "out", man, fps, {er.out_rel(t, p) for t, p in targets if er.target_key(p) not in bad}, only=all_rel)
    assert n == 3 and sorted(man) == ["anon/api/race-list/20261129.json", "anon/api/race/202605050201.json"]
    assert sorted(fps) == sorted(man)


def test_targets_skip_incomplete_dates(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_EDGE_MIRROR_DIR", raising=False)
    base = _mirror(tmp_path, ["race_shutuba", "race_result"])
    rl = base / "data" / "page_reference" / "race_lists"
    rl.mkdir(parents=True)
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(base / "data" / "page_reference"))
    rl.joinpath("20261128.json").write_text(json.dumps({"races": [{"race_id": "202605050101"}, {"race_id": "202605050102"}]}))
    rl.joinpath("20261129.json").write_text(json.dumps({"races": [{"race_id": "202605050201"}]}))
    for rid in ("202605050101", "202605050201"):
        _put(base, "race_shutuba", rid, {})
        _put(base, "race_result", rid, {})
    _put(base, "race_shutuba", "202605050102", {})                               # 結果がまだ（確定前）
    cfg = er.load_config()
    t = er.targets_for(cfg, base, ["20261128", "20261129"], [], ["anon"])
    assert ("anon", "/api/race-list/20261129") in t
    assert ("anon", "/api/race-list/20261128") not in t                          # 1 レースが欠けた日は一覧を作らない
    assert ("anon", "/api/race/202605050101") in t and ("anon", "/api/race/202605050101/predictions") in t
    assert not any("202605050102" in p for _, p in t)
    assert er.targets_for(cfg, base, [], ["202605050102"], ["anon"]) == []          # 名指しでも写しに揃っていなければ作らない


def test_write_if_changed(tmp_path):
    man: dict[str, str] = {}
    assert er.write_if_changed(tmp_path, "anon/api/race/1.json", b'{"a":1}', man) is True
    p = tmp_path / "anon/api/race/1.json"
    m1 = p.stat().st_mtime_ns
    assert (tmp_path / "anon/api/race/1.json.gz").is_file()
    assert er.write_if_changed(tmp_path, "anon/api/race/1.json", b'{"a":1}', man) is False
    assert p.stat().st_mtime_ns == m1                                            # 同じなら書き換えない（rsync が送らない）
    assert er.write_if_changed(tmp_path, "anon/api/race/1.json", b'{"a":2}', man) is True
    import gzip

    assert gzip.decompress((tmp_path / "anon/api/race/1.json.gz").read_bytes()) == b'{"a":2}'


def test_out_rel():
    assert er.out_rel("anon", "/api/race/202611280101/predictions") == "anon/api/race/202611280101/predictions.json"


def test_plan_push_uses_local_records_only():
    man = {"anon/a.json": "1", "anon/b.json": "2", "anon/c.json": "3"}
    sent = {"anon/a.json": "1", "anon/b.json": "old", "anon/z.json": "9"}
    send, delete = er.plan_push(man, sent)
    assert send == ["anon/b.json", "anon/c.json"]
    assert delete == ["anon/z.json"]


def test_remote_script_keeps_ctl_and_one_old_version():
    s = er.remote_script("stg", "0123456789ab")
    assert "/srv/keiba-edge/stg/0123456789ab" in s
    assert "mv -T .current.tmp current" in s
    assert "grep -E '^[0-9a-f]{12}$'" in s                       # _ctl・.incoming は消さない
    assert "tail -n +2" in s                                      # 今の版のほかに 1 つ残す


def test_fingerprint_changes_with_mirror(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_EDGE_MIRROR_DIR", raising=False)
    base = _mirror(tmp_path, ["race_shutuba", "race_result"])
    _put(base, "race_shutuba", "202605050101", {"a": 1})
    f1 = er.race_fingerprint(base, "202605050101", ["race_shutuba", "race_result"])
    _put(base, "race_result", "202605050101", {"r": 1})            # 結果が入った
    f2 = er.race_fingerprint(base, "202605050101", ["race_shutuba", "race_result"])
    assert f1 != f2
    assert f2 == er.race_fingerprint(base, "202605050101", ["race_shutuba", "race_result"])


def test_prune_outputs(tmp_path):
    man: dict[str, str] = {}
    er.write_if_changed(tmp_path, "anon/api/race/1.json", b"1", man)
    er.write_if_changed(tmp_path, "anon/api/race/2.json", b"2", man)
    fps = {"anon/api/race/1.json": "x", "anon/api/race/2.json": "y"}
    assert er.prune_outputs(tmp_path, man, fps, {"anon/api/race/2.json"}) == 1
    assert not (tmp_path / "anon/api/race/1.json").exists() and not (tmp_path / "anon/api/race/1.json.gz").exists()
    assert list(man) == ["anon/api/race/2.json"] and list(fps) == ["anon/api/race/2.json"]


def test_remote_apply_replaces_deletes_and_resyncs(tmp_path):
    """VPS 側の置き換え（_REMOTE_APPLY）: 届いたものを置き、消す一覧を消し、送り直しでは手元の記録に無いものを消す。"""
    import subprocess
    import sys

    root, inc = tmp_path / "v", tmp_path / "inc"
    (root / "anon/api/race").mkdir(parents=True)
    (root / "_meta").mkdir()
    for n in ("old", "gone", "stray"):
        (root / f"anon/api/race/{n}.json").write_text(n)
        (root / f"anon/api/race/{n}.json.gz").write_text(n)
    (root / "_meta/heartbeat.json").write_text("{}")
    (inc / "anon/api/race").mkdir(parents=True)
    (inc / "anon/api/race/old.json").write_text("new")
    (inc / "_delete.json").write_text(json.dumps(["anon/api/race/gone.json"]))
    subprocess.run([sys.executable, "-c", er._REMOTE_APPLY, str(root), str(inc)], check=True)
    assert (root / "anon/api/race/old.json").read_text() == "new"
    assert not (root / "anon/api/race/gone.json").exists() and not (root / "anon/api/race/gone.json.gz").exists()
    assert (root / "anon/api/race/stray.json").exists()              # 送り直しでなければ残す
    assert not inc.exists()
    (inc / "anon").mkdir(parents=True)
    (inc / "_delete.json").write_text("[]")
    (inc / "_keep.json").write_text(json.dumps(["anon/api/race/old.json"]))
    subprocess.run([sys.executable, "-c", er._REMOTE_APPLY, str(root), str(inc)], check=True)
    assert not (root / "anon/api/race/stray.json").exists() and not (root / "anon/api/race/stray.json.gz").exists()
    assert (root / "anon/api/race/old.json").exists() and (root / "anon/api/race/old.json.gz").exists()
    assert (root / "_meta/heartbeat.json").exists()                  # _meta は消さない


def test_prune_mirror_only_old_race_categories(tmp_path):
    import os
    import time as _t
    from argparse import Namespace

    old = _t.time() - 50 * 86400
    for cat, key in (("race_shutuba", "202605050101"), ("horse_result", "2019100001")):
        _put(tmp_path, cat, key, {})
        os.utime(tmp_path / "data/local/mirror" / cat / key[:4] / f"{key}.json", (old, old))
    _put(tmp_path, "race_result", "202611280101", {})                  # 新しい
    assert er.cmd_prune(Namespace(base=str(tmp_path), mirror_days=40)) == 0
    m = tmp_path / "data/local/mirror"
    assert not (m / "race_shutuba/2026/202605050101.json").exists()
    assert (m / "horse_result/2019/2019100001.json").exists()          # 馬のカテゴリは消さない
    assert (m / "race_result/2026/202611280101.json").exists()


# ---- E5（確定前）


def test_pending_dates():
    cfg = {"mirror_since": "20261123", "entries_lead_days": 5, "settled_grace_days": 1, "upcoming_days": 7}
    assert er.pending_dates(cfg, datetime(2026, 11, 26, 10, tzinfo=er.JST)) == ["20261128", "20261129", "20261130", "20261201", "20261202", "20261203"]
    d = er.pending_dates(cfg, datetime(2026, 12, 5, 10, tzinfo=er.JST))
    assert d[0] == "20261204" and d[-1] == "20261212"                 # 前日（結果待ち）から 7 日先まで


def test_fingerprint_follows_inference_entries_events_and_time(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_EDGE_MIRROR_DIR", raising=False)
    base = _mirror(tmp_path, ["race_shutuba", "race_result"])
    rid, date = "202605050101", "20261128"
    _put(base, "race_shutuba", rid, {})
    now = datetime(2026, 11, 28, 10, 0, tzinfo=er.JST)
    fp = lambda n=now: er.race_fingerprint(base, rid, ["race_shutuba", "race_result"], date, now=n, bucket_sec=600)
    f0 = fp()
    inf = base / "data/local/inference"
    (inf / "predictions_v6" / date).mkdir(parents=True)
    (inf / "predictions_v6" / date / f"{rid}.parquet").write_bytes(b"x")
    f1 = fp()
    assert f1 != f0                                                    # 予測の出し直し
    (base / "data/local/meta/entries_state").mkdir(parents=True)
    (base / "data/local/meta/entries_state" / f"{rid}.json").write_text("{}")
    f2 = fp()
    assert f2 != f1                                                    # 出馬表の段階
    (inf / "events").mkdir()
    (inf / "events" / f"{date}.jsonl").write_text('{"race_id": "202605050102"}\n')
    assert fp() == f2                                                  # 別のレースの経過の行は効かない
    (inf / "events" / f"{date}.jsonl").write_text('{"race_id": "202605050102"}\n{"race_id": "202605050101"}\n')
    f3 = fp()
    assert f3 != f2                                                    # このレースの行
    assert fp(datetime(2026, 11, 28, 10, 0, 1, tzinfo=er.JST)) == f3   # 同じ区切りの中
    assert fp(datetime(2026, 11, 28, 10, 10, tzinfo=er.JST)) != f3     # 確定前は 10 分ごとに作り直す（区切りはレースごとにずれる）
    _put(base, "race_result", rid, {})
    f4 = fp()
    assert fp(datetime(2026, 11, 28, 16, 0, tzinfo=er.JST)) == f4      # 確定済みは時刻で変わらない
    early = datetime(2026, 11, 26, 10, tzinfo=er.JST)
    g = er.race_fingerprint(base, "202605050999", ["race_shutuba"], date, now=early, bucket_sec=600)
    assert er.race_fingerprint(base, "202605050999", ["race_shutuba"], date, now=datetime(2026, 11, 26, 10, 50, tzinfo=er.JST), bucket_sec=600) == g
    assert er.race_fingerprint(base, "202605050999", ["race_shutuba"], date, now=datetime(2026, 11, 26, 11, 1, tzinfo=er.JST), bucket_sec=600) != g   # 先の日は 1 時間ごと


def test_content_md5_ignores_as_of():
    a = b'{"pipeline_status":{"as_of":"2026-11-28T10:00:00+09:00","stages":[1]}}'
    b = b'{"pipeline_status":{"as_of":"2026-11-28T10:09:00+09:00","stages":[1]}}'
    c = b'{"pipeline_status":{"as_of":"2026-11-28T10:09:00+09:00","stages":[2]}}'
    assert er.content_md5(a) == er.content_md5(b) != er.content_md5(c)


def test_default_targets_mix_settled_and_pending(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_EDGE_MIRROR_DIR", raising=False)
    base = _mirror(tmp_path, ["race_shutuba", "race_result"])
    rl = base / "data/page_reference/race_lists"
    rl.mkdir(parents=True)
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(base / "data/page_reference"))
    rl.joinpath("20261128.json").write_text(json.dumps({"races": [{"race_id": "202605050101"}]}))   # 確定済み
    rl.joinpath("20261205.json").write_text(json.dumps({"races": [{"race_id": "202605050201"}]}))   # 確定前（出馬表だけ）
    _put(base, "race_shutuba", "202605050101", {})
    _put(base, "race_result", "202605050101", {})
    _put(base, "race_shutuba", "202605050201", {})
    cfg = {**er.load_config(), "mirror_since": "20261123", "entries_lead_days": 5, "upcoming_days": 7}
    t, race_date = er.default_targets(cfg, base, ["anon"], datetime(2026, 12, 3, 10, tzinfo=er.JST))
    assert ("anon", "/api/race/202605050101") in t                       # 確定済みは anon
    assert ("pre", "/api/race/202605050201/predictions") in t             # 確定前は pre（VPS が heartbeat の古さで止められる）
    assert ("anon", "/api/race-list/20261128") in t and ("pre", "/api/race-list/20261205") in t
    assert er.out_rel("pre", "/api/race/202605050201") == "pre/api/race/202605050201.json"
    assert race_date["202605050201"] == "20261205"
    t0, _ = er.default_targets({**cfg, "upcoming_days": 0}, base, ["anon"], datetime(2026, 12, 3, 10, tzinfo=er.JST))
    assert not any("202605050201" in p for _, p in t0)                 # upcoming_days=0 は確定済みだけ（E3 の形）


def test_cmd_render_skips_remembered_failures_without_loading_api(tmp_path, monkeypatch):
    """捨てた対象は、指紋が変わるまで作り直さない（入力の欠けが続くレースで毎分 API を読み込まない）。負荷が高い回は 3 で返す。"""
    from argparse import Namespace

    monkeypatch.delenv("KEIBA_EDGE_MIRROR_DIR", raising=False)
    base = _mirror(tmp_path, ["race_shutuba", "race_result"])
    t = [("anon", "/api/race/202605050101")]
    monkeypatch.setattr(er, "default_targets", lambda cfg, b, tiers, now: (t, {"202605050101": "20261128"}))
    monkeypatch.setattr(er, "tree_sha", lambda root: "0123456789ab")
    monkeypatch.setattr(er, "load_config", lambda: {**json.loads(er.CONFIG_PATH.read_text(encoding="utf-8")), "max_load": 0})
    fp = er.target_fingerprints(er.load_config(), base, "stg", "0123456789ab", t, {"202605050101": "20261128"})
    meta = tmp_path / "out" / "stg" / "0123456789ab" / "_meta"
    meta.mkdir(parents=True)
    (meta / "skipped.json").write_text(json.dumps(fp))
    called = []
    monkeypatch.setattr(er, "apply_clean_environ", lambda *a, **k: called.append(1))
    args = Namespace(base=str(base), tier=None, diagnose=False, race=None, date=None, env="stg", out=str(tmp_path / "out"), full=False)
    assert er.cmd_render(args) == 0
    assert called == []                                                   # API を読み込まない
    hb = json.loads((meta / "heartbeat.json").read_text())
    assert hb["rendered"] == 0
    monkeypatch.setattr(er, "load_config", lambda: {**json.loads(er.CONFIG_PATH.read_text(encoding="utf-8")), "max_load": 0.0001})
    (meta / "skipped.json").write_text("{}")
    monkeypatch.setattr(er.os, "getloadavg", lambda: (99.0, 0, 0))
    assert er.cmd_render(args) == er.DEFERRED                             # 混んでいる回は見送り
