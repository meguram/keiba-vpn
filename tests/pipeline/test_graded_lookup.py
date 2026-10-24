"""出馬表のレース名から格（G1/G2/G3/L）を引く照合（回数・括弧・略称・スポンサー名の違い）。"""

from src.pipeline.features.horse_history.inference import graded_lookup, graded_name_key

G = {"産経賞オールカマー": "G2", "リゲルステークス": "L", "中山記念": "G2", "ジャパンＣ": "G1"}


def test_name_key_strips_edition_and_parenthesis():
    assert graded_name_key("第96回中山記念") == "中山記念"
    assert graded_name_key("リゲルステークス(L)") == "リゲルステークス"


def test_lookup_handles_abbreviation_and_sponsor_prefix():
    assert graded_lookup("オールカマー", G) == "G2"          # 正式名の末尾が一致
    assert graded_lookup("リゲルS", G) == "L"                # ステークス → S
    assert graded_lookup("第96回中山記念", G) == "G2"
    assert graded_lookup("3歳未勝利", G) == "OP"             # 不明は OP のまま
    assert graded_lookup("", G) == "OP"
