"""ルールの会議の頁（src/scripts/docs/build_rules_council.py）が、結果の JSON の鍵で作れて、結論の文の数字が結果と食い違わないこと。"""

import json
from pathlib import Path

from src.scripts.docs import build_rules_council as B

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return json.loads((ROOT / "docs/operations/retrain_trials" / name).read_text(encoding="utf-8"))


def test_pages_build_from_results():
    texts = _load("rules_council_pages.json")
    t60, rc2 = _load("rules_council_t60.json"), _load("rules_council_rc2.json")
    P = json.loads(B.PREREG2.read_text(encoding="utf-8"))
    bd = _load("rules_council_rc3_bound.json")
    for h in (B.build_rc1(t60, texts["rc1"]), B.build_rc2(rc2, P, texts["rc2"]), B.build_rc3(bd, texts["rc3"]), B.build_summary(texts["summary"])):
        assert h.startswith("<!doctype html>") and h.endswith("</html>")


def test_conclusion_numbers_match_results():
    texts = _load("rules_council_pages.json")
    s = _load("rules_council_rc2.json")["search"]
    P = json.loads(B.PREREG2.read_text(encoding="utf-8"))
    c = " ".join(texts["rc2"]["conclusion"])
    assert f"{s['n_up1']} 水準" in c and s["n_candidates"] == 0 and "候補 0" in c
    assert B.pc(s["reference"]["baseline_all"]["roi"]) in c                     # 全部買う 80.3%
    plant = P["stage_B"]["result"]["arms"]["1000"]["plant_cand"]
    assert B.pc(plant["1.10"], 0) in c and B.pc(plant["1.05"], 1) in c         # 検出力 95%・67.5%
    t = _load("rules_council_t60.json")["versions"]["主: 前半でずれ → 後半で比べる"]["< 10"]
    c1 = " ".join(texts["rc1"]["conclusion"])
    assert B.pc(t["入れ替わりの割合（q）"]) in c1 and B.pc(t["再現どうしの入れ替わり（床）"]) in c1


def test_rc3_bound_matches_text():
    texts = _load("rules_council_pages.json")
    bd = _load("rules_council_rc3_bound.json")
    mults = [v for k in ("inner", "front") for kk, o in bd[k].items() if isinstance(o, dict) for m, v in o.items() if "mult" in m]
    assert abs(max(mults) - bd["max_mult"]) < 1e-12 and bd["max_mult"] < 1.27   # 要る倍率（一桁の 損益分岐 ÷ 帯の比）に届かない、という結論の前提
    assert f"{bd['max_mult']:.2f}" in " ".join(texts["rc3"]["conclusion"])
