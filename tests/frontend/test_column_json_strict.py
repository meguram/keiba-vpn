"""めぐコラムの JSON は TypeScript が import するので、厳密な JSON（NaN/Infinity なし）でなければビルドが落ちる。"""
import json
from pathlib import Path

import pytest

COLUMNS = sorted((Path(__file__).resolve().parents[2] / "frontend/content/columns").glob("*.json"))


def _reject(token: str):
    raise ValueError(f"非標準の JSON 値 {token}（欠測は null にする）")


@pytest.mark.parametrize("path", COLUMNS, ids=lambda p: p.name)
def test_column_json_is_strict(path):
    json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
