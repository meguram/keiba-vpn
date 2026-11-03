"""めぐコラム『夏は牝馬』の年齢別の節の差し替え（src/research/columns/summer_mare_age.py）。"""

from __future__ import annotations

import json

from src.research.columns import summer_mare_age as sm


def test_section_is_inserted_then_replaced_without_touching_the_rest():
    text = ('{\n  "slug": "summer-mare",\n  "body": [\n    { "type": "text", "text": "前" },\n'
            f'    {{ "type": "heading", "text": "{sm.BEFORE}" }},\n    {{ "type": "text", "text": "後" }}\n  ]\n}}\n')
    new = [{"type": "heading", "text": sm.HEADING}, {"type": "text", "text": "1 回目"},
           {"type": "table", "columns": ["a"], "rows": [["x"]]}]
    once = sm.splice(text, new)
    body = json.loads(once)["body"]
    assert [b.get("text") for b in body] == ["前", sm.HEADING, "1 回目", None, sm.BEFORE, "後"]
    assert '{ "type": "text", "text": "前" }' in once                    # 他の部分の書式はそのまま
    twice = sm.splice(once, [{"type": "heading", "text": sm.HEADING}, {"type": "text", "text": "2 回目"}])
    assert [b.get("text") for b in json.loads(twice)["body"]] == ["前", sm.HEADING, "2 回目", sm.BEFORE, "後"]
