"""layer_a_dataset._schema_any: 先頭の年にテーブルが無い（2016〜2018 年の race_index など）とき、ある年のスキーマを使う。"""

from src.pipeline.models.layer_a_dataset import _schema_any


class _Store:
    def __init__(self, schemas):
        self.schemas = schemas

    def source_schema(self, source, year):
        return self.schemas.get((source, str(year)), {})


def test_uses_first_year_that_has_the_table():
    st = _Store({("race_index", "2019"): {"race_id": "str", "speed_max": "f8"}, ("race_index", "2020"): {"race_id": "str"}})
    assert _schema_any(st, "race_index", ["2016", "2017", "2019", "2020"]) == {"race_id": "str", "speed_max": "f8"}


def test_empty_when_no_year_has_the_table():
    assert _schema_any(_Store({}), "race_index", ["2016", "2017"]) == {}
