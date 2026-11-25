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
    for h in (B.build_rc1(t60, texts["rc1"]), B.build_rc2(rc2, P, texts["rc2"]), B.build_rc3(bd, texts["rc3"]),
              B.build_rc4(_load("rules_council_rc4_bound.json"), texts["rc4"]), B.build_summary(texts["summary"])):
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


def test_rc4_text_matches_bound():
    texts = _load("rules_council_pages.json")
    bd = _load("rules_council_rc4_bound.json")
    v = [r[b][f"{t}_lift_over_Q"] for r in bd["cols"].values() for b in ("一桁", "穴馬") if b in r for t in ("top20", "top10")]
    c = " ".join(texts["rc4"]["conclusion"])
    assert f"{min(v):.2f}〜{max(v):.2f}" in c and max(v) < 1.27


def test_results_table_numbers_come_from_json():
    """成績表（R0-07）の数字は結果の JSON の写し。注意の文が先頭にあり、子ページへのリンクが全行にある。"""
    from src.scripts.docs import build_rules_council_table as BT
    D = BT.load()
    sec = BT.easy_section(D)
    assert BT.HEAD_NOTE in sec and sec.index(BT.HEAD_NOTE) < sec.index("| やり方")
    rows = D["T"]["rows"]
    assert BT.pc(rows["RC-2 和集合"]["roi"]) in sec and BT.pc(rows["RC-5 H1"]["roi"]) in sec
    rc6 = next(r for r in BT.results_rows(D) if r["id"] == "RC-6")                 # RC-6 は 2-1 の写し（数え直した均一の値ではない）
    assert rc6["roi"].startswith(BT.pc(rows["RC-6 c13（2-1 の写し）"]["roi"])) and rc6["hit"] == "未集計"
    m13 = D["conf"]["c13_flat"]["confirm"]
    assert f"{BT.pc(m13['roi'])}（{BT.ci(m13['roi_ci'])}）" in sec
    for r in BT.ROWS:
        assert f"(easy/{r['page']})" in sec
    pages = BT.child_pages(D)
    assert set(pages) == {r["page"] for r in BT.ROWS}
    assert "<h2>やり方ごとの成績表</h2>" in BT.results_html()
