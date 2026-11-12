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


def test_nd_euro_sires_are_default_o_and_others_are_not():
    t = _t().assign(type_name=["欧州型ノーザンダンサー系", "欧州型ミスプロ系", "欧州型ノーザンダンサー系"])
    h = B.build(t, {})
    rows = re.findall(r'<tr data-id="(S\d)"[^>]*data-def="([^"]*)"', h)
    assert dict(rows) == {"S1": "o", "S2": "", "S3": "o"}
    assert 'id="defo"' in h and "euro_kick_defaults_v1" in h        # 初期値の一回適用の目印と、○に戻すボタン


# ── ページの JS を実際に動かして、マークを失わないための挙動を確かめる（quickjs が無い環境では飛ばす） ─────────────

_STUB = r"""
var __ls=SEED, __h={}, __conf=true, __fail=false, __blob=null;
function mkcls(){var s={};return {add:function(c){s[c]=1},remove:function(c){delete s[c]},toggle:function(c,on){if(on)s[c]=1;else delete s[c]},contains:function(c){return !!s[c]}}}
function mkrow(id,name,small,n,def,nm){var r={dataset:{id:id,name:name,small:small,n:n,def:def},classList:mkcls(),querySelector:function(){return {textContent:nm}},closest:function(){return r}};return r}
var ROWS=ROWSJS;var els={};
function el(id){return els[id]=els[id]||{id:id,value:'',textContent:'',classList:mkcls(),files:[],addEventListener:function(ev,fn){__h[id+':'+ev]=fn}}}
var document={querySelectorAll:function(){return ROWS},querySelector:function(){return {appendChild:function(){}}},getElementById:function(id){return el(id)},body:{appendChild:function(){}},createElement:function(){return {click:function(){},remove:function(){}}}};
var localStorage={getItem:function(k){return (k in __ls)?__ls[k]:null},setItem:function(k,v){if(__fail&&k==='euro_kick_marks_v1') throw new Error('quota'); __ls[k]=String(v)},removeItem:function(k){delete __ls[k]}};
var navigator={};var URL={createObjectURL:function(){return 'x'},revokeObjectURL:function(){}};function Blob(p){__blob=p[0]}
var confirm=function(){return __conf};function FileReader(){}function setTimeout(){}
"""


def _page(seed=None, fail=False):
    qjs = __import__("pytest").importorskip("quickjs")
    import json
    t = pd.DataFrame({"sire_id": ["A1", "A2", "B1"], "name": ["nd1", "nd2", "mp1"], "small_name": ["x", "x", "y"],
                      "type_name": ["欧州型ノーザンダンサー系", "欧州型ノーザンダンサー系", "欧州型ミスプロ系"], "n": [3, 2, 1]})
    h = B.build(t, {})
    js = re.search(r"<script>([\s\S]*)</script>", h).group(1)
    rows = re.findall(r'<tr data-id="([^"]+)" data-name="([^"]*)" data-small="([^"]*)" data-n="(\d+)" data-def="([^"]*)">.*?<td class="nm">(.*?)</td>', h)
    rj = "[" + ",".join(f"mkrow({json.dumps(a)},{json.dumps(b)},{json.dumps(c)},{int(d)},{json.dumps(e)},{json.dumps(f)})" for a, b, c, d, e, f in rows) + "]"
    c = qjs.Context()
    c.eval(_STUB.replace("SEED", json.dumps(seed or {})).replace("ROWSJS", rj))
    if fail:
        c.eval("__fail=true")
    c.eval(js)
    return c


def _marks(c):
    import json
    v = c.eval("__ls['euro_kick_marks_v1']")
    return json.loads(v) if v else {}


def _ls(c):
    import json
    return json.loads(c.eval("JSON.stringify(__ls)"))


def _click(c, i, m):
    c.eval(f"(function(){{var r=ROWS[{i}];var b={{dataset:{{m:'{m}'}},closest:function(){{return r}}}};__h['t:click']({{target:{{closest:function(){{return b}}}}}})}})()")


def test_js_defaults_apply_once_and_do_not_overwrite_or_return():
    c = _page()
    assert _marks(c) == {"A1": "o", "A2": "o"}                       # 初回: 欧州型ノーザンダンサー系だけ○。ミスプロ系は未判定
    _click(c, 0, "x")                                                  # 利用者が ×に直す
    assert _marks(_page(_ls(c))) == {"A1": "x", "A2": "o"}           # 開き直しても戻らない
    c2 = _page(_ls(c))
    c2.eval("__h['reset:click']()")
    assert _marks(_page(_ls(c2))) == {}                                # クリア後も戻らない
    c2.eval("__h['defo:click']()")                                     # ボタンで ○ に戻す
    assert _marks(c2) == {"A1": "o", "A2": "o"}


def test_js_existing_marks_are_never_overwritten_by_defaults():
    import json
    c = _page({"euro_kick_marks_v1": json.dumps({"B1": "x"})})        # すでに作業済みのブラウザ（初期値のフラグが無い）
    assert _marks(c) == {"B1": "x"}                                    # ND 欧州型を勝手に○にしない
    assert _ls(c)["euro_kick_defaults_v1"] == "1"


def test_js_corrupt_storage_is_not_overwritten_at_startup():
    c = _page({"euro_kick_marks_v1": "{壊れた"})
    assert c.eval("__ls['euro_kick_marks_v1']") == "{壊れた"            # 開いただけでは上書きしない
    assert "euro_kick_defaults_v1" not in _ls(c)


def test_js_default_flag_is_not_set_when_saving_fails():
    c = _page(fail=True)                                               # マークの保存に失敗する環境
    assert "euro_kick_defaults_v1" not in _ls(c)                       # フラグだけ立って、次に○が消えることがない


def test_js_export_distinguishes_default_from_user_marks():
    import json
    c = _page()
    _click(c, 1, "o")                                                  # A2 は利用者が○を付け直した（触った）
    c.eval("__h['save:click']()")
    d = json.loads(c.eval("__blob"))
    assert d["marks"]["A1"]["source"] == "default" and d["marks"]["A2"]["source"] == "user" and "default_rule" in d
