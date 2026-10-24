import pandas as pd

from src.scripts.maintenance.restore_mojibake_fields import build_maps, restore_doc


def _maps():
    flat = pd.DataFrame({"race_id": ["r1", "r1"], "race_name": ["2歳新馬", "2歳新馬"], "horse_id": ["h1", "h2"], "horse_name": ["ディープ", "ブライアン"],
                         "jockey_id": ["j1", "j2"], "jockey_name": ["武豊", "川田"], "trainer_id": ["t1", "t2"], "trainer_name": ["矢作", "友道"],
                         "sex_age": ["牡2", "牝2"]})
    return build_maps(flat)


def test_names_restored_from_ids_and_free_text_blanked_with_record():
    doc = {"race_id": "r1", "race_name": "2��", "entries": [
        {"horse_id": "h1", "horse_name": "��", "jockey_id": "j1", "jockey_name": "��", "sex_age": "��2", "comment": "��絖", "evaluation": "C"},
        {"horse_id": "h2", "horse_name": "ブライアン", "trainer_id": "t2", "trainer_name": "�", "comment": "良い"}], "_meta": {"x": 1}}
    out, info = restore_doc(doc, _maps())
    e0, e1 = out["entries"]
    assert out["race_name"] == "2歳新馬"
    assert (e0["horse_name"], e0["jockey_name"], e0["sex_age"]) == ("ディープ", "武豊", "牡2")
    assert e1["trainer_name"] == "友道" and e1["comment"] == "良い"
    assert e0["comment"] == "" and e0["evaluation"] == "C"
    assert info["blanked"] == {"comment": 1}
    assert out["_meta"]["mojibake_blanked"] == {"comment": 1} and out["_meta"]["x"] == 1
    assert "�" not in str(out["entries"]) and doc["entries"][0]["horse_name"] == "��"   # 元は変えない


def test_unknown_ids_are_blanked_not_kept_garbled():
    out, info = restore_doc({"race_id": "rX", "entries": [{"horse_id": "zz", "horse_name": "��"}]}, _maps())
    assert out["entries"][0]["horse_name"] == "" and info["blanked"] == {"horse_name": 1}
