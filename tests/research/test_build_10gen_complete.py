from src.research.pedigree import build_horse_pedigree_10gen as b


def _doc(prefix):
    anc = [{"generation": g, "position": p, "name": f"{prefix}{g}-{p}", "horse_id": f"{prefix}_{g}_{p}"}
           for g in range(1, 6) for p in range(2 ** g)]
    return {"ancestors": anc, "sex": "牡", "sire": "s", "dam": "d", "dam_sire": "ds"}


def test_full_table_has_2046_unique_paths_and_both_anchor_sexes():
    idx = {"H": _doc("h")}
    for a in idx["H"]["ancestors"]:
        if a["generation"] == 5:
            idx[a["horse_id"]] = _doc(a["horse_id"])
    rec = b.build_10gen_record("H", idx)
    assert rec["n_ancestors"] == b.EXPECTED_POSITIONS == 2046 and rec["complete"]
    assert len({a["path_fm"] for a in rec["ancestors"]}) == 2046
    assert {a.get("source") for a in rec["ancestors"]} >= {"merged_gen5_sire", "merged_gen5_dam"}
    assert all(v == 1.0 for v in rec["fill"].values())


def test_missing_anchor_leaves_positions_empty_and_incomplete():
    idx = {"H": _doc("h")}
    rec = b.build_10gen_record("H", idx)          # アンカーの 5 世代表が無い
    assert rec["n_ancestors"] == 62 and not rec["complete"] and rec["fill"]["6"] == 0.0


def test_path_fm_matches_side():
    assert b.path_fm(1, 1) == "M" and b.path_fm(2, 2) == "MF" and b.path_fm(5, 0) == "FFFFF"


def test_maternal_complete_when_only_maternal_anchors_exist():
    idx = {"H": _doc("h")}
    for a in idx["H"]["ancestors"]:
        if a["generation"] == 5 and b.is_maternal(5, a["position"]):     # 母方 16 アンカーだけ
            idx[a["horse_id"]] = _doc(a["horse_id"])
    rec = b.build_10gen_record("H", idx)
    assert rec["maternal_complete"] and rec["n_maternal"] == b.EXPECTED_MATERNAL == 1023
    assert not rec["complete"] and rec["paternal_fill"]["6"] == 0.0 and rec["maternal_fill"]["10"] == 1.0
