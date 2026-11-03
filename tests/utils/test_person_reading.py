"""騎手・調教師の読みのマスタと 50 音順キー（src/utils/person_reading.py）。"""

from src.utils import person_reading as pr
from src.utils.person_reading import jockey_reading, kana_sort_key, sort_by_reading, trainer_reading


def test_jockey_reading_from_alias_id_and_marks():
    assert jockey_reading("横山武") == "よこやま たけし"
    assert jockey_reading("横山武史") == "よこやま たけし"
    assert jockey_reading("01170") == "よこやま たけし"
    assert jockey_reading("▲水沼元輝") == "みずぬま げんき"      # 見習いの印は外して引く
    assert jockey_reading("Ｍ．デム") == "でむーろ みるこ"        # 全角の略称
    assert jockey_reading("Ｃデムーロ") == "でむーろ くりすちゃん"
    assert jockey_reading("木幡巧") == "こわた たくや"            # 「こはた」ではない
    assert jockey_reading("岩田望") == "いわた みらい"


def test_jockey_reading_ambiguous_surname_and_unknown():
    assert jockey_reading("角田") == "つのだ"                      # 角田大和・角田大河: 姓の読みだけ
    assert jockey_reading("存在しない騎手") is None
    assert jockey_reading("") is None


def test_trainer_reading():
    assert trainer_reading("国枝栄") == "くにえだ さかえ"          # netkeiba のフリガナを姓・名に分けたもの
    assert trainer_reading("矢作芳人") == "やはぎ よしと"
    assert trainer_reading("01075") == "やはぎ よしと"
    assert trainer_reading("中内田充正") == "なかうちだ みつまさ"


def test_kana_sort_key_order():
    rs = ["おのでら ゆうた", "おの ふうま", "こう ひであき", "ごとう", "こうの", "でむーろ みるこ", "でむーろ くりすちゃん", None, "あ"]
    got = sorted(rs, key=kana_sort_key)
    assert got[0] == "あ" and got[-1] is None
    assert got.index("おの ふうま") < got.index("おのでら ゆうた")      # 姓が先
    assert got.index("でむーろ くりすちゃん") < got.index("でむーろ みるこ")
    assert kana_sort_key("カワダ") == kana_sort_key("かわだ")             # カタカナ = ひらがな
    assert kana_sort_key("かわた") < kana_sort_key("かわだ") < kana_sort_key("かわち")  # 清音・濁音は同じ位置、清音が先


def test_sort_by_reading():
    assert sort_by_reading(["横山武", "ルメール", "川田", "謎の人", "岩田望"], "jockey") == ["岩田望", "川田", "横山武", "ルメール", "謎の人"]


def test_master_complete_and_unique():
    for kind in ("jockey", "trainer"):
        by_id, _, persons = pr._master(kind)
        assert len(by_id) == len(persons) > 300
        for p in persons:
            r = p["reading"]
            assert r and all("ぁ" <= c <= "ゖ" or c in "ー ゔ" for c in r), (kind, p)
            assert p["confidence"] in {"high", "medium", "low"}
