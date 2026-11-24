"""smartrc の系統の台帳の置き場（2026-11-24）: 既定は data/local、手元に無い・壊れたときだけ git の種の版（SEED）を読む。"""

import json

from src.scraper import smartrc_sire_lines as ssl
from tests.scraper.test_smartrc_sire_lines import env  # noqa: F401  （同じ手元の表の fixture を使う）


def _seed(tmp_path, monkeypatch, doc):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ssl, "SEED", seed)
    return seed


def test_default_out_is_local_and_falls_back_to_seed(env, monkeypatch, tmp_path):  # noqa: F811
    seed = _seed(tmp_path, monkeypatch, {"sires": {"S_NEW": {"small_name": "ロベルト系", "from": "fetch"}},
                                         "tried": {"S_X": "2026-11-01"}, "complete_at": "2026-11-01T00:00:00+09:00"})
    local_root = tmp_path / "local"
    local_root.mkdir()
    local = local_root / "meta" / "out.json"
    monkeypatch.setattr(ssl, "LOCAL", local_root)
    monkeypatch.setattr(ssl, "_DEFAULT_OUT", local)
    monkeypatch.setattr(ssl, "OUT", local)
    assert ssl.labels() == {"S_NEW": "ロベルト系"}            # 手元が無い → 種の版
    before = seed.read_text(encoding="utf-8")
    ssl.build()
    assert local.exists() and seed.read_text(encoding="utf-8") == before   # 種の版は書き換えない
    doc = ssl.load()
    assert doc["sires"]["S_NEW"]["from"] == "fetch" and "S_DEEP" in doc["sires"]   # 種の版の取得済みの行を引き継ぐ
    assert doc["tried"] == {"S_X": "2026-11-01"}
    local.write_text("{壊れた", encoding="utf-8")            # 手元が壊れた → 種の版
    assert ssl.labels() == {"S_NEW": "ロベルト系"}


def test_other_path_no_fallback_and_no_local_dir_created(env, monkeypatch, tmp_path):  # noqa: F811
    _seed(tmp_path, monkeypatch, {"sires": {"S_NEW": {"small_name": "ロベルト系"}}})
    assert ssl.load(tmp_path / "other.json") == {"sires": {}}   # 既定でない場所（環境変数など）は種の版に落ちない
    missing_root = tmp_path / "no_local"
    out = missing_root / "meta" / "out.json"
    monkeypatch.setattr(ssl, "LOCAL", missing_root)
    monkeypatch.setattr(ssl, "_DEFAULT_OUT", out)
    assert ssl._save({"sires": {}}, out) is False          # data/local が無い場所に本物のフォルダを作らない
    assert ssl.fetch(10, out) == {"fetched": 0, "not_writable": 1}   # 書けない場所では smartrc に取りに行かない
    assert not missing_root.exists()


def test_broken_local_is_rewritten_even_if_same_as_seed(monkeypatch, tmp_path):
    doc = {"sires": {"S_NEW": {"small_name": "ロベルト系"}}}
    _seed(tmp_path, monkeypatch, doc)
    local_root = tmp_path / "local"
    local = local_root / "meta" / "out.json"
    local.parent.mkdir(parents=True)
    local.write_text("{壊れた", encoding="utf-8")
    monkeypatch.setattr(ssl, "LOCAL", local_root)
    monkeypatch.setattr(ssl, "_DEFAULT_OUT", local)
    assert ssl._save(dict(doc), local) is True              # 種の版と同じ中身でも、壊れた手元の台帳は書き直す
    assert json.loads(local.read_text(encoding="utf-8")) == doc
