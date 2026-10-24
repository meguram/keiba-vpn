"""serving_v5_px の部品: 半年の窓、状態の作成、静的な馬の埋め込みが学習側と一致、新しい馬（血統 JSON）の追加。"""

import json

import numpy as np
import pandas as pd

from src.pipeline.features import factory_ped as fp
from src.pipeline.features import serving_v5_px as sv
from src.pipeline.features import tp38_pedigree as P


def test_window_start_and_end_follow_six_month_steps():
    assert sv.window_start(pd.Timestamp("2025-03-02")) == "2025-01-01" and sv.window_start(pd.Timestamp("2025-06-30")) == "2025-01-01"
    assert sv.window_start(pd.Timestamp("2025-07-01")) == "2025-07-01" and sv.window_start(pd.Timestamp("2025-12-31")) == "2025-07-01"
    assert sv.window_end("2025-01-01") == "2025-07-01" and sv.window_end("2025-07-01") == "2026-01-01"


def _static(tmp_path, n_h=300, n_a=120, seed=0):
    rng = np.random.default_rng(seed)
    horses = [f"h{i:04d}" for i in range(n_h)]
    rows = []
    for i, h in enumerate(horses):
        for gen in range(1, 6):
            for pos in range(2 ** gen):
                rows.append((h, gen, pos, f"a{rng.integers(0, n_a)}"))
    A = pd.DataFrame(rows, columns=["horse_id", "gen", "pos", "anc_id"])
    hi = pd.Index(horses)
    anc, uniq = P.prep_ancestors(A, hi)
    d = tmp_path / "static"
    d.mkdir()
    anc.to_parquet(d / "anc.parquet", index=False)
    pd.Series(hi).to_frame("horse_id").to_parquet(d / "horses.parquet", index=False)
    pd.Series(uniq).to_frame("anc_id").to_parquet(d / "ancestors.parquet", index=False)
    X, _ = P.horse_vectors(anc, len(hi), len(uniq))
    E, _cl = P.embed_and_cluster(X, ks=(4,))
    np.save(d / "emb.npy", E)
    return d, anc, hi, uniq, rows


def test_static_horse_embedding_matches_training_and_new_horse_is_added(tmp_path, monkeypatch):
    d, anc, hi, uniq, _rows = _static(tmp_path)
    info = sv.build_state(d, tmp_path / "state")
    assert info["emb_max_abs_err_vs_tp38"] < 1e-3
    st = sv.PXState(tmp_path / "state")
    ext = st.extend(["h0003", "h0010"])
    assert len(ext.horse_index) == len(hi)                          # 学習時の馬だけなら追加なし
    E = st.embed(ext, np.array([3, 10, -1]))
    ref = np.load(d / "emb.npy")[:, : sv.EMB_DIM]
    assert np.allclose(E.iloc[:2].to_numpy(), ref[[3, 10]], atol=1e-3) and E.iloc[2].isna().all()
    # 新しい馬: 血統 JSON から祖先を足す（未知の祖先はコードを末尾に足す）
    ped = tmp_path / "ped" / "2026"
    ped.mkdir(parents=True)
    ancestors = [{"generation": g, "position": p, "horse_id": f"a{(g * 7 + p) % 50}" if g < 5 else "zz_new"} for g in range(1, 6) for p in range(2 ** g)]
    (ped / "2026100001.json").write_text(json.dumps({"ancestors": ancestors}), encoding="utf-8")
    monkeypatch.setattr(fp, "PED_DIRS", [tmp_path / "ped"])
    ext2 = st.extend(["2026100001"])
    assert len(ext2.horse_index) == len(hi) + 1 and len(ext2.uniq) == len(uniq) + 1 and ext2.anc["h"].max() == len(hi)
    E2 = st.embed(ext2, np.array([len(hi)]))
    assert E2.notna().all(axis=None) and np.isfinite(E2.to_numpy()).all()
