"""v5 の構成選択の定義の整合（重い評価はせず、構成・対の比較・群の合成だけを確かめる）。"""

import pandas as pd

from src.scripts.data import v5_common, v5_select_eval as sel


def test_pairs_refer_to_known_configs_and_chain_is_cumulative():
    for new, old in sel.PAIRS:
        assert new in sel.CONFIGS and old in sel.CONFIGS
    assert sel.CONFIGS["a_V4"]["base"] == "old" and sel.CONFIGS["b_tlfresh"]["base"] == "fresh"
    # 累積: d, e は c の群を含み、f は d と e の群を両方含む
    c = set(sel.CONFIGS["c_CY"]["groups"])
    assert c <= set(sel.CONFIGS["d_CY_BEST"]["groups"]) and c <= set(sel.CONFIGS["e_CY_RLC"]["groups"])
    assert set(sel.CONFIGS["d_CY_BEST"]["groups"]) | set(sel.CONFIGS["e_CY_RLC"]["groups"]) == set(sel.CONFIGS["f_CY_BEST_RLC"]["groups"])


def test_merge_keys_drops_duplicate_keys_and_keeps_row_order():
    df = pd.DataFrame({"race_id": ["r1", "r1", "r2"], "horse_id": ["a", "b", "a"]}, index=[10, 11, 12])
    T = pd.DataFrame({"race_id": ["r1", "r1", "r1", "r2"], "horse_id": ["a", "b", "b", "a"], "x": [1.0, 2.0, 9.0, 3.0]})
    m = sel._merge_keys(df, T)
    assert list(m.index) == [10, 11, 12] and m["x"].tolist() == [1.0, 2.0, 3.0]


def test_fresh_tlx_replaces_only_existing_tl_columns(monkeypatch, tmp_path):
    new = pd.DataFrame({"race_id": ["r1", "r1"], "horse_id": ["a", "b"], "best_t1z_28": [0.5, 0.7], "extra_only_in_new": [1, 2]})
    p = tmp_path / "tp35"
    p.mkdir()
    new.to_parquet(p / "cand_TL_A_v2.parquet", index=False)
    monkeypatch.setattr(v5_common, "V", tmp_path)
    df = pd.DataFrame({"race_id": ["r1", "r1"], "horse_id": ["a", "b"]}, index=[5, 6])
    old = pd.DataFrame({"tl_best_t1z_28": [9.0, 9.0], "tl_only_in_old": [1.0, 2.0]}, index=[5, 6])
    out = v5_common.fresh_tlx(df, old)
    assert out["tl_best_t1z_28"].tolist() == [0.5, 0.7]
    assert out["tl_only_in_old"].tolist() == [1.0, 2.0]          # 再取得版に無い列は触らない
    assert "tl_extra_only_in_new" not in out.columns           # 列セット（旧 TLX の名前）は変えない
