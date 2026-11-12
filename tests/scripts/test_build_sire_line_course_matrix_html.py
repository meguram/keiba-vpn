"""父の系統 × 会場・コースの成績マトリクス（build_sire_line_course_matrix_html）: 集計・並び・馬場状態の行・埋め込みデータ・画面の計算。"""

import json
import re

import pandas as pd
import pytest

from src.scripts.docs import build_sire_line_course_matrix_html as M


def _runs():
    rows = []

    def add(line, venue, sf, dist, cond, poss):
        rows.extend({"venue": venue, "surface": sf, "distance": dist, "line": line, "pos": p, "cond": cond} for p in poss)
    add("A系", "東京", "芝", 1600, "良", [1, 1, 2, 3, 5])
    add("A系", "東京", "芝", 1600, "稍重", [4, 6, 7, 8, 9])
    add("B系", "阪神", "ダート", 1200, "重・不良", [1, 5, 6, 7])
    return pd.DataFrame(rows)


META = {"A系": {"sires": 4, "o": 3, "large": "甲系"}, "B系": {"sires": 2, "o": 0, "large": "乙系"}}


def _model(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(M, "MIN_LINE_RUNS", 1)                 # 少ない出走数の系統も列に出す
    runs = _runs()
    return M.model(M.aggregate(runs), META, M.aggregate_cond(runs))


def test_aggregate_counts_places():
    c = M.aggregate(_runs())
    a = c[(c.venue == "東京") & (c.line == "A系")].iloc[0]
    assert (a.c1, a.c2, a.c3, a.c4) == (2, 1, 1, 6)
    b = c[c.line == "B系"].iloc[0]
    assert (b.c1, b.c2, b.c3, b.c4) == (1, 0, 0, 3)
    k = M.aggregate_cond(_runs())
    t = k[(k.surface == "芝") & (k.cond == "良") & (k.line == "A系")].iloc[0]
    assert (t.c1, t.c2, t.c3, t.c4) == (2, 1, 1, 1)               # 良の 5 走: 1,1,2,3,5


def test_model_has_cond_venue_and_course_rows_with_correct_counts(monkeypatch):
    d = _model(monkeypatch)
    ids = {r["id"]: r for r in d["rows"]}
    assert "cond:芝:良" in ids and "cond:芝:稍重" in ids and "cond:ダート:重・不良" in ids
    assert {"venue:東京:all", "venue:東京:芝", "venue:阪神:ダート", "course:東京:芝:1600", "course:阪神:ダート:1200"} <= set(ids)
    cols = [c["name"] for c in d["cols"]]
    assert ids["course:東京:芝:1600"]["c"][cols.index("A系")] == [2, 1, 1, 6]
    assert ids["venue:東京:all"]["t"] == [2, 1, 1, 6]
    assert ids["cond:ダート:重・不良"]["c"][cols.index("B系")] == [1, 0, 0, 3]
    assert "course:東京:ダート:1600" not in ids                    # 無い組み合わせの行は作らない


def test_columns_are_grouped_by_large_line_and_ordered_by_runs(monkeypatch):
    d = _model(monkeypatch)
    assert [(c["large"], c["name"]) for c in d["cols"]] == [("甲系", "A系"), ("乙系", "B系")]      # 大系統の出走数が多い順
    assert d["cols"][0]["o"] == 3 and d["cols"][0]["sires"] == 4


def test_lines_with_few_runs_are_left_out_of_the_columns():
    d = M.model(M.aggregate(_runs()), META, None)                  # 既定の最少出走数（300）に満たない系統は列にしない
    assert d["cols"] == []


def test_html_embeds_valid_data_and_is_escaped(monkeypatch):
    monkeypatch.setattr(M, "MIN_LINE_RUNS", 1)
    runs = _runs().assign(line=lambda r: r.line.replace({"A系": "</script><b>A"}))
    meta = {"</script><b>A": {"sires": 1, "o": 1, "large": "甲系"}, "B系": META["B系"]}
    h = M.build(M.aggregate(runs), meta, M.aggregate_cond(runs))
    body = re.search(r'<script type="application/json" id="data">([\s\S]*?)</script>', h).group(1)
    assert "<" not in body                                           # < を含めない（</script> や <!-- で HTML の解析が切れない）
    d = json.loads(body)                                             # JSON としては同じ値に戻る
    assert any(c["name"] == "</script><b>A" for c in d["cols"])
    assert "<table" in h and 'id="metric"' in h and "勝率" in h and "複勝率" in h and "競走中止・取消・除外は数えない" in h


def _pure():
    qjs = pytest.importorskip("quickjs")
    c = qjs.Context()
    c.eval(M.JS_PURE)
    return c


def test_js_pure_rates_sort_and_colors():
    c = _pure()
    assert c.eval("PURE.rate([2,1,1,6],'w')") == pytest.approx(0.2) and c.eval("PURE.rate([2,1,1,6],'p')") == pytest.approx(0.4)
    assert json.loads(c.eval("JSON.stringify(PURE.sortIdx([0.2,NaN,0.5,0.1],1))")) == [2, 0, 3, 1]      # 降順。NaN は末尾
    assert json.loads(c.eval("JSON.stringify(PURE.sortIdx([0.2,NaN,0.5,0.1],-1))")) == [3, 0, 2, 1]     # 昇順。NaN は末尾
    assert c.eval("PURE.level([1,0,0,2],0.3,'p',30)") is None                                            # 最少出走数に満たない
    big = c.eval("PURE.level([300,300,300,1100],0.30,'p',30)")                                           # 大標本で基準 +10pt 以上 → 最大
    assert big == pytest.approx(1.0)
    small = c.eval("PURE.level([3,1,1,25],0.30,'p',30)")                                                 # 30 走で複勝率 16.7%（基準 30% より低い）→ 負
    assert -1 <= small < 0
    lo, mid, hi = (json.loads(c.eval(f"JSON.stringify(PURE.color({s}))")) for s in (-1, 0, 1))
    assert lo["lum"] < mid["lum"] < hi["lum"] and hi["hi"] and not mid["hi"]                            # 明るいほど好成績、暗いほど不振（中立で途切れない）
    half = json.loads(c.eval("JSON.stringify(PURE.color(0.5))"))
    assert mid["lum"] < half["lum"] < hi["lum"]


def test_js_pure_orders_columns_within_groups():
    c = _pure()
    c.eval("var cols=[{large:'A'},{large:'A'},{large:'B'},{large:'B'}];var rc=[[1,0,0,9],[5,3,2,0],[0,0,0,10],[3,3,3,1]];")
    g = json.loads(c.eval("JSON.stringify(PURE.orderCols(cols,rc,'p',5,1))"))
    assert g == [{"large": "A", "idx": [1, 0]}, {"large": "B", "idx": [3, 2]}]                           # 大系統の枠を保ち、枠の中を降順
    g2 = json.loads(c.eval("JSON.stringify(PURE.orderCols(cols,rc,'p',5,-1))"))
    assert g2 == [{"large": "B", "idx": [2, 3]}, {"large": "A", "idx": [0, 1]}]                          # 昇順では大系統も入れ替わる（B: 10走で複勝 0% が最も低い）
    assert json.loads(c.eval("JSON.stringify(PURE.orderCols(cols,null,'p',5,1))")) == [{"large": "A", "idx": [0, 1]}, {"large": "B", "idx": [2, 3]}]


def test_script_is_valid_javascript():
    esprima = pytest.importorskip("esprima")
    h = M.build(M.aggregate(_runs()), META, M.aggregate_cond(_runs()))
    for s in re.findall(r"<script>([\s\S]*?)</script>", h):
        esprima.parseScript(s)


def test_load_runs_filters(tmp_path, monkeypatch):
    """集計の条件: 着順が付かない走（-1）・障害・父の系統が空欄の走を外し、馬場状態の重・不良をまとめる。"""
    flat = tmp_path / "flat" / "2024"
    flat.mkdir(parents=True)
    pd.DataFrame({
        "venue": ["東京"] * 7, "surface": ["芝", "芝", "芝", "ダート", "芝", "芝", "芝"], "distance": [1600] * 7,
        "race_name": ["A", "A", "障害3歳以上未勝利", "A", "A", "A", "A"], "finish_position": [1, -1, 1, 2, 3, 4, 5],
        "horse_id": ["h1", "h2", "h3", "h4", "h5", "h6", "h7"], "track_condition": ["良", "良", "良", "重", "不良", "稍重", "良"]}).to_parquet(flat / "race_result_flat.parquet")
    pd.DataFrame({"horse_id": ["h1", "h2", "h3", "h4", "h5", "h6", "h7"], "sire_id": ["s1", "s1", "s1", "s1", "s1", "s1", "s9"]}).to_parquet(tmp_path / "map.parquet")
    pd.DataFrame({"sire_id": ["s1", "s9"], "small_name": ["A系", None], "large_name": ["甲系", None], "status": ["ok", "blank"]}).to_csv(tmp_path / "c.csv", index=False)
    monkeypatch.setattr(M, "FLAT", tmp_path / "flat")
    monkeypatch.setattr(M, "YEARS", range(2024, 2025))
    monkeypatch.setattr(M, "SIRE_MAP", tmp_path / "map.parquet")
    monkeypatch.setattr(M, "COLORS", tmp_path / "c.csv")
    monkeypatch.setattr(M, "MARKS", tmp_path / "none.json")
    runs, meta = M.load_runs()
    assert sorted(runs.pos) == [1, 2, 3, 4]                          # -1（中止・取消・除外）・障害・系統が空欄の馬（h7）は入らない
    assert dict(zip(runs.pos, runs.cond)) == {1: "良", 2: "重・不良", 3: "重・不良", 4: "稍重"}
    assert meta == {"A系": {"sires": 1, "o": 0, "large": "甲系"}}


def _ui(monkeypatch=None, onlyo=False):
    """簡易 DOM で画面の JS を動かす。クリック・絞り込みを模擬して、描画された HTML を返す関数群を返す。"""
    qjs = pytest.importorskip("quickjs")
    runs = pd.DataFrame(
        [{"venue": v, "surface": sf, "distance": d, "line": ln, "pos": p, "cond": "良"}
         for v, sf, d in (("東京", "芝", 1600), ("東京", "ダート", 1400), ("阪神", "芝", 1600))
         for ln, ps in (("A系", [1, 2, 3, 5]), ("B系", [6, 7, 8, 9]), ("C系", [1, 1, 4, 5]))
         for p in ps])
    meta = {"A系": {"sires": 2, "o": 2, "large": "甲系"}, "B系": {"sires": 2, "o": 0, "large": "甲系"}, "C系": {"sires": 2, "o": 0, "large": "乙系"}}
    old = M.MIN_LINE_RUNS
    M.MIN_LINE_RUNS = 1
    try:
        h = M.build(M.aggregate(runs), meta, M.aggregate_cond(runs))
    finally:
        M.MIN_LINE_RUNS = old
    sc = re.findall(r"<script[^>]*>([\s\S]*?)</script>", h)
    c = qjs.Context()
    c.eval("""var __h={}, __els={};
function El(id){return {id:id,value:'',checked:true,textContent:'',innerHTML:'',style:{setProperty:function(){}},addEventListener:function(ev,fn){__h[id+':'+ev]=fn},querySelector:function(){return {offsetHeight:30}}}}
var document={getElementById:function(id){return __els[id]=__els[id]||El(id)}};
document.getElementById('data').textContent=%s;""" % json.dumps(sc[0]))
    c.eval(sc[1])
    c.eval(sc[2])

    def rows():
        return re.findall(r'<th class="sticky rl" data-row="([^"]+)">', c.eval("document.getElementById('m').innerHTML"))

    def cols():
        return re.findall(r'<th class="line[^"]*" data-key="\d+"><div class="nm">([^<▼▲ ]*)', c.eval("document.getElementById('m').innerHTML"))

    def click(kind, val):
        pick = {"key": "th[data-key]", "row": "th[data-row]", "tg": ".tg"}[kind]
        field = {"key": "key", "row": "row", "tg": "v"}[kind]
        c.eval("(function(){var e={target:{closest:function(s){ return s===%s?{dataset:{%s:%s}}:null; }}};__h['m:click'](e)})()" % (json.dumps(pick), field, json.dumps(val)))

    def setf(i, **kw):
        sets = ";".join(f"x.{k}={json.dumps(v)}" for k, v in kw.items())
        ev = "input" if i == "minn" else "change"                      # 最少出走数は input イベント、他は change
        c.eval(f"(function(){{var x=document.getElementById('{i}');{sets};__h['{i}:{ev}'].call(x)}})()")
    return c, rows, cols, click, setf


def test_ui_layout_sort_and_state_consistency():
    c, rows, cols, click, setf = _ui()
    info = lambda: c.eval("document.getElementById('sortinfo').textContent")
    setf("minn", value="1")                                                                                  # 少ないデータでも並べ替えに使う
    assert rows() == ["cond:芝:良", "cond:ダート:良", "venue:東京:all", "venue:阪神:all"]                       # 馬場状態の行 → 会場（閉じている）
    click("tg", "東京")
    assert [r for r in rows() if r.startswith("course")] == ["course:東京:ダート:1400", "course:東京:芝:1600"]   # 開くとコース行（ダート → 芝の順で、距離順）
    assert cols() == ["A系", "B系", "C系"]                                                                  # 甲系（A・B）→ 乙系（C）
    click("row", "venue:東京:all")                                                                          # 東京の行の複勝率（A 75%・B 0%・C 50%）で列を降順に並べ替え
    assert cols() == ["C系", "A系", "B系"]                                                                  # 大系統どうし（乙系 50% > 甲系 37.5%）も、枠の中（A > B）も降順
    click("row", "venue:東京:all")
    assert cols() == ["B系", "A系", "C系"]                                                                  # 昇順（甲系 37.5% < 乙系 50%、枠の中は B < A）
    click("row", "venue:東京:all")
    assert cols() == ["A系", "B系", "C系"] and info() == ""                                                  # 解除で元の並びに戻る
    click("row", "venue:東京:all")
    setf("sf", value="芝")                                                                                  # 芝のみ: 基準の行は「東京 芝全体」へ付け替わり、ダートの行は消える
    assert rows() == ["cond:芝:良", "venue:東京:芝", "course:東京:芝:1600", "venue:阪神:芝"]
    assert "列の並べ替え: 東京 芝全体" in info()                                                               # 並べ替えの状態が常に表示される
    click("key", 0)                                                                                          # 列 A 系で行を並べ替え → 降順（会場の行だけが並ぶ）
    assert "行の並べ替え: A系 ▼降順" in info()
    setf("onlyo", checked=True)                                                                              # ○ の小系統（A 系だけ）。基準の列 A 系は見えるので維持
    assert cols() == ["A系"] and "行の並べ替え: A系" in info()


def test_ui_clears_row_sort_when_its_column_is_hidden_and_group_sum_uses_visible_columns():
    c, rows, cols, click, setf = _ui()
    click("key", 1)                                                                                          # 列 B 系で行を並べ替え
    assert "行の並べ替え: B系" in c.eval("document.getElementById('sortinfo').textContent")
    setf("onlyo", checked=True)                                                                              # B 系は ○ でない → 隠れるので、基準を解除する
    assert "行の並べ替え" not in c.eval("document.getElementById('sortinfo').textContent")
    click("key", "g:甲系")                                                                                   # 大系統の値は、見えている列（A 系だけ）の合計
    assert "行の並べ替え: 甲系" in c.eval("document.getElementById('sortinfo').textContent")
