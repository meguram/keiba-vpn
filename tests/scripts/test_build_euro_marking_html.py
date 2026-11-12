"""欧州瞬発力のマーキング用 HTML（build_euro_marking_html）: 欧州型の種牡馬が全て出て、エスケープされ、保存の形が決まっている。"""

import re

import pandas as pd

from src.scripts.docs import build_euro_marking_html as B


def _t():
    return pd.DataFrame({
        "sire_id": ["S1", "S2", "S3"], "name": ["Galileo (愛)", "ハービンジャー", '危険"<b>&名'],
        "small_name": ["サドラーズウェルズ系", "ダンチヒ系", "ダンチヒ系"], "type_name": ["欧州型ノーザンダンサー系"] * 3, "n": [20, 368, 5]})


def test_every_sire_has_a_row_with_three_buttons():
    h = B.build(_t(), {"S1": ["フランケル産駒"]})
    rows = re.findall(r'<tr data-id="(S\d)"', h)
    assert sorted(rows) == ["S1", "S2", "S3"]
    assert h.count('data-m="o"') == 3 and h.count('data-m="x"') == 3 and h.count('data-m=""') == 3
    assert "フランケル産駒" in h


def test_html_is_escaped_and_ids_are_attribute_safe():
    t = _t().assign(sire_id=["S1", "S2", 'S"3\'<x>'])
    h = B.build(t, {})
    assert "<b>&名" not in h and "&lt;b&gt;&amp;名" in h
    assert 'data-id="S&quot;3&#x27;&lt;x&gt;"' in h and 'data-name="危険&quot;&lt;b&gt;&amp;名"' in h   # 属性の中も、引用符・山かっこをエスケープする


def test_export_shape_and_storage_key_are_fixed():
    h = B.build(_t(), {})
    assert "euro_kick_marks_v1" in h and "version:1" in h and "euro_kick_marks.json" in h


def test_no_measured_values_are_shown():
    h = B.build(_t(), {})
    body = h.split("<tbody>")[1].split("</tbody>")[0]          # 表の中身（種牡馬の行）には、素点・順位などの測定値を出さない
    for w in ("素点", "hold", "contrast", "順位", "縮小"):
        assert w not in body
    assert len(re.findall(r"<td", body)) == 3 * 6              # 種牡馬・小系統・国別タイプ・産駒数・代表産駒・判定の 6 列だけ


def test_import_export_logic_guards_data_loss():
    """JS の動作は実行できないので、データを失わないための仕組み（読み込みの検証・置き換え・書き出し・保存先の確認）が入っていることを固定する。"""
    h = B.build(_t(), {})
    assert "Object.create(null)" in h                      # toString などの ID が混ざらない
    assert "d.version!==1" in h and "isMark(m)" in h       # 形式と値の検証
    assert "marks=next" in h and "inp.value=''" in h       # 検証後に一括で置き換え／同じファイルを選び直せる
    assert "(この表に無い種牡馬)" in h                       # 表に無い種牡馬のマークも書き出しで落とさない
    assert 'id="warn"' in h and "自動保存が使えません" in h    # 保存先が使えないときの常時警告
