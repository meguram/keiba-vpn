"""モデル性能ページ: 事前計算（model_perf_cache）と画面用の読み出し（src/monitor/model_perf.py・API）のオフラインテスト。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.monitor import app as monitor_app
from src.monitor import model_perf
from src.scripts.data import model_perf_cache as C


def _preds(n_races: int = 40, seed: int = 0, prefix: str = "2025060101") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        k = 8
        score = rng.normal(size=k)
        pw = np.exp(score) / np.exp(score).sum()
        order = np.argsort(-(score + rng.normal(scale=1.0, size=k)))
        fin = np.empty(k, dtype=int)
        fin[order] = np.arange(1, k + 1)
        for i in range(k):
            rows.append({"race_id": f"{prefix}{r:02d}", "horse_id": f"h{r}_{i}", "finish_position": int(fin[i]), "p_win": float(pw[i]),
                         "p_top2": float(min(0.99, pw[i] * 2)), "p_top3": float(min(0.99, pw[i] * 3))})
    return pd.DataFrame(rows)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    tp, inf, rank, out = tmp_path / "tp", tmp_path / "inf", tmp_path / "rank", tmp_path / "out"
    (tp / "final_eval_v4").mkdir(parents=True)
    _preds().to_parquet(tp / "final_eval_v4" / "test2025_predictions.parquet")
    (tp / "final_eval_v4" / "report.json").write_text(json.dumps({"logloss": {"win": 0.2}}))
    # 2026 の予測と実着順（1 週間ぶん・少数レース）
    p26 = _preds(6, seed=1, prefix="2026010401")
    day = inf / "predictions_backfill" / "bundle_v4_2025" / "20260104"
    day.mkdir(parents=True)
    for rid, g in p26.groupby("race_id"):
        g[["horse_id", "p_win", "p_top2", "p_top3"]].to_parquet(day / f"{rid}.parquet")
    (rank / "2026").mkdir(parents=True)
    p26[["race_id", "horse_id", "finish_position"]].rename(columns={"finish_position": "rank"}).to_parquet(rank / "2026" / "rank.parquet")
    monkeypatch.setattr(C, "TP", tp)
    monkeypatch.setattr(C, "INF", inf)
    monkeypatch.setattr(C, "RANK_TBL", rank)
    monkeypatch.setattr(C, "OUT_DIR", out)
    monkeypatch.setenv("KEIBA_MODEL_PERF_DIR", str(out))
    return out


def test_metrics_are_sane():
    d = _preds()
    m = C.head_metrics(d, with_reliability=True)
    assert 0 < m["win"]["logloss"] < 1 and 0.4 < m["win"]["auc"] <= 1
    assert len(m["win"]["reliability"]) == 10 and m["win"]["ece"] is not None
    t1 = C.top1_stats(d)
    assert t1["n_races"] == 40 and 0 <= t1["hit"] <= 1 and t1["ci"][0] <= t1["hit"] <= t1["ci"][1]


def test_group_of_uses_prefix_rules():
    rules, default = C._group_rules()
    assert C.group_of("o_win_rate", rules, default) == "馬主"
    assert C.group_of("tl_a_mean", rules, default) == "調教・追い切り"
    assert C.group_of("ped_sire_x", rules, default) == "血統"
    assert C.group_of("jockey_prior_win_rate", rules, default) == "騎手・調教師"
    assert C.group_of("distance", rules, default) == "レース条件・当日の枠/斤量"
    assert C.group_of("zzz_unknown", rules, default) == default


def test_build_and_read_back_with_missing_pieces(tree):
    idx = C.build_all(["ens_v4", "ens_v2"], importance=False, prod=True)
    assert [m["key"] for m in idx["models"]] == ["ens_v4", "ens_v2", "shadow"]
    d = model_perf.load_all()
    assert d["status"] == "ok"
    by = {m["key"]: m for m in d["models"]}
    v4 = by["ens_v4"]
    assert v4["test2025"]["n_races"] == 40 and v4["prod_series"][0]["low_sample"] is True      # 6 レース = 少数
    assert v4["prod_series"][0]["week"].startswith("2026-W") and v4["prod_total"]["n_races"] == 6
    assert by["ens_v2"]["test2025"] is None and by["ens_v2"]["prod_series"] == []              # 出力が無い版は空（落ちない）
    assert by["shadow"]["prod_series"] == []
    rows = model_perf.summary_rows(d)
    assert rows[0]["key"] == "ens_v4" and rows[0]["test_win_logloss"] is not None


def test_importance_from_tiny_lightgbm_bundle(tree, tmp_path):
    lgb = pytest.importorskip("lightgbm")
    b = tmp_path / "tp" / "final_v4" / "bundle_v4_2025"
    cols = ["o_a", "tl_a", "distance", "zzz_x"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=cols)
    y = (X["o_a"] + 0.2 * rng.normal(size=300) > 0).astype(int)
    for head in ("win", "top3"):
        (b / "lgb_extras_ped" / "seed0").mkdir(parents=True, exist_ok=True)
        lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 4}, lgb.Dataset(X, y), num_boost_round=10).save_model(str(b / "lgb_extras_ped" / "seed0" / f"lgb_{head}.txt"))
    (b / "bundle.json").write_text(json.dumps({"members": [{"name": "lgb_extras_ped", "recipe": "lgb_binary", "feature_cols": cols}, {"name": "x", "recipe": "mtl_nn"}]}))
    imp = C.build_importance("final_v4/bundle_v4_2025")
    h = imp["heads"]["win"]
    assert h["top_features"][0]["name"] == "o_a" and h["by_group"][0]["group"] == "馬主"
    assert abs(sum(g["share"] for g in h["by_group"]) - 1) < 1e-3 and imp["members"][0]["name"] == "lgb_extras_ped"


def test_reader_survives_missing_and_corrupt_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_MODEL_PERF_DIR", str(tmp_path / "none"))
    d = model_perf.load_all()
    assert d["status"] == "not_computed" and d["models"] == [] and d["hint"]
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.json").write_text(json.dumps({"built_at": "x", "models": [{"key": "ens_v9", "label": "壊れ"}, {"key": "../etc", "label": "不正"}]}))
    (out / "ens_v9.json").write_text("{broken")
    monkeypatch.setenv("KEIBA_MODEL_PERF_DIR", str(out))
    d = model_perf.load_all()
    assert [m["status"] for m in d["models"]] == ["not_computed", "not_computed"]


@pytest.fixture
def client(tree):
    C.build_all(["ens_v4"], importance=False, prod=True)
    monitor_app.app.config["TESTING"] = True
    return monitor_app.app.test_client()


def test_pages_and_api_require_login(client):
    for url in ("/models", "/api/internal/model-perf"):
        r = client.get(url)
        assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_api_and_pages_render(client):
    with client.session_transaction() as s:
        s["monitor_auth"] = True
    d = client.get("/api/internal/model-perf").get_json()
    assert d["status"] == "ok" and {m["key"] for m in d["models"]} == {"ens_v4", "shadow"}
    s1 = client.get("/api/internal/model-perf?summary=1").get_json()
    assert "rows" in s1 and "models" not in s1
    html = client.get("/models").get_data(as_text=True)
    assert "モデルの予測性能" in html and "chart-prod" in html and "テストセット②" in html
    assert 'href="/models"' in html                                    # ナビ
    assert "model-perf" in client.get("/services").get_data(as_text=True)


def test_api_without_cache_says_not_computed(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_MODEL_PERF_DIR", str(tmp_path / "none"))
    monitor_app.app.config["TESTING"] = True
    c = monitor_app.app.test_client()
    with c.session_transaction() as s:
        s["monitor_auth"] = True
    assert c.get("/api/internal/model-perf").get_json()["status"] == "not_computed"
    assert c.get("/models").status_code == 200
