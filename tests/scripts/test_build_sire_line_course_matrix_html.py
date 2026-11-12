"""父の系統・系統配合 × クラス・舞台の成績マトリクス（build_sire_line_course_matrix_html）: 集計・行の定義・柵位置・コースのまとめ・母方の型・画面の計算。"""

import json
import re

import pandas as pd
import pytest

from src.scripts.docs import build_sire_line_course_matrix_html as M

META = {"A系": {"sires": 4, "o": 3, "large": "甲系"}, "B系": {"sires": 2, "o": 0, "large": "乙系"}}
LEDGER = {"sA": "日本型サンデー系", "sB": "米国型ミスプロ系"}


def _runs():
    """A 系（甲系）の馬 h1・h2 と B 系（乙系）の馬 h3。東京 芝1600 と 阪神 ダート1200。"""
    rows = []

    def add(rid, date, venue, sf, dist, day, kai, horse, line, large, pos, cond, cls):
        rows.append({"race_id": rid, "date": date, "venue": venue, "surface": sf, "distance": dist, "day": day, "kai": kai, "horse_id": horse,
                     "line": line, "father_large": large, "pos": pos, "cond": cond, "cls": cls})
    for i, (p, cls) in enumerate(((1, "2勝"), (2, "2勝"), (5, "未勝利"), (7, "3勝"))):
        add(f"2019050101{i:02d}", "2019-04-06", "東京", "芝", 1600, 1, 1, "h1", "A系", "甲系", p, "良", cls)
    for i, p in enumerate((1, 3, 4)):
        add(f"2019050105{i:02d}", "2019-04-13", "東京", "芝", 1600, 3, 1, "h2", "A系", "甲系", p, "稍重", "2勝")
    for i, p in enumerate((1, 5, 6, 7)):
        add(f"2019090201{i:02d}", "2019-06-01", "阪神", "ダート", 1200, 7, 2, "h3", "B系", "乙系", p, "重・不良", "1勝")
    return pd.DataFrame(rows)


def _model(monkeypatch=None, rail=None, mat=None):
    if monkeypatch is not None:
        monkeypatch.setattr(M, "MIN_LINE_RUNS", 1)
        monkeypatch.setattr(M, "MIN_PAIR_RUNS", 1)
    return M.model(_runs(), META, LEDGER, rail, mat)


def _row(d, rid):
    return next(r for r in d["rows"] if r["id"] == rid)


def test_class_rows_count_places_and_sum_rows(monkeypatch):
    d = _model(monkeypatch)
    cols = [c["name"] for c in d["tables"][0]["cols"]]
    a = cols.index("A系")
    assert _row(d, "class:芝:2勝")["k"][0]["c"][a] == [2, 1, 1, 1]                     # h1 の 2勝（1着・2着）＋ h2 の 2勝（1着・3着・4着）
    assert _row(d, "class:芝:未勝利")["k"][0]["c"][a] == [0, 0, 0, 1]
    assert _row(d, "class:芝:3勝以上")["k"][0]["c"][a] == [0, 0, 0, 1]                   # 3勝（7着）だけ
    assert _row(d, "class:芝:3勝以上")["sum"] is True and _row(d, "class:芝:2勝").get("sum") is None
    assert _row(d, "class:ダート:1勝")["k"][0]["c"][cols.index("B系")] == [1, 0, 0, 3]
    assert "class:ダート:G1" not in {r["id"] for r in d["rows"]}                        # 走が無い行は作らない
    assert _row(d, "class:芝:2勝")["k"][0]["t"] == [2, 1, 1, 1]


def test_cond_day_and_course_rows(monkeypatch):
    d = _model(monkeypatch)
    assert _row(d, "cond:芝:良")["k"][0]["t"] == [1, 1, 0, 2] and _row(d, "cond:芝:稍重")["k"][0]["t"] == [1, 0, 1, 1]
    assert _row(d, "day:開催1-2日目")["k"][0]["t"] == [1, 1, 0, 2] and _row(d, "day:開催3-4日目")["k"][0]["t"] == [1, 0, 1, 1]   # 芝だけ。ダートの 7 日目は入らない
    ids = {r["id"] for r in d["rows"]}
    assert {"venue:東京:all", "venue:東京:芝", "venue:阪神:ダート", "course:東京:芝:芝1600m", "course:阪神:ダート:ダート1200m"} <= ids
    assert "venue:東京:ダート" not in ids                                                # 無い組み合わせの行は作らない


def test_two_tables_and_mix_totals_use_horses_with_known_mother(monkeypatch):
    mat = {"h1": ["日本型サンデー系", "米国型ミスプロ系"], "h3": ["米国型ミスプロ系"]}      # h2 は母方の型が分からない
    d = _model(monkeypatch, mat=mat)
    t0, t1 = d["tables"]
    assert [c["name"] for c in t0["cols"]] == ["A系", "B系"] and [c["large"] for c in t0["cols"]] == ["甲系", "乙系"]
    names = [(c["large"], c["name"]) for c in t1["cols"]]
    assert ("甲系", "日本型サンデー系") in names and ("甲系", "米国型ミスプロ系") in names and ("乙系", "米国型ミスプロ系") in names
    r = _row(d, "class:芝:2勝")["k"][1]
    assert r["c"][names.index(("甲系", "日本型サンデー系"))] == [1, 1, 0, 0]               # h1 の 2勝（1着・2着）。2 つの型の列に重ねて数える
    assert r["c"][names.index(("甲系", "米国型ミスプロ系"))] == [1, 1, 0, 0]
    assert r["t"] == [1, 1, 0, 0]                                                       # 行の合計は展開前（h2 は母方不明なので入らない）
    assert d["tables"][1]["note"].startswith("母方")


def test_lines_and_pairs_with_few_runs_are_left_out():
    d = M.model(_runs(), META, LEDGER, None, {"h1": ["日本型サンデー系"]})              # 既定の最少出走数に満たない
    assert d["tables"][0]["cols"] == []
    assert len(d["tables"]) == 2 and d["tables"][1]["cols"] == []


def test_merged_courses_groups_short_distances_within_the_same_venue_and_surface(monkeypatch):
    monkeypatch.setattr(M, "MIN_COURSE_RACES", 100)
    rows = []
    for venue, sf, dist, n in (("東京", "芝", 1600, 150), ("東京", "芝", 2400, 50), ("東京", "芝", 2500, 30), ("東京", "芝", 3400, 10),
                              ("札幌", "芝", 1000, 5), ("札幌", "芝", 1200, 120), ("札幌", "芝", 2600, 20), ("東京", "ダート", 1600, 130)):
        rows.extend({"race_id": f"{venue}{sf}{dist}-{i}", "venue": venue, "surface": sf, "distance": dist} for i in range(n))
    c = M.merged_courses(pd.DataFrame(rows))
    assert c[("東京", "芝", 1600)][0] == "芝1600m"
    assert {c[("東京", "芝", x)][0] for x in (2400, 2500, 3400)} == {"芝2400m以上"}               # 少ない長距離は「以上」でまとめる（会場・芝/ダートは揃える）
    assert c[("札幌", "芝", 1000)][0] == "芝1200m（1000mを含む）" == c[("札幌", "芝", 1200)][0]    # 少ない 1000m は隣の 1200m とまとめるが、主力の 1200m の名前を範囲にしない
    assert c[("札幌", "芝", 2600)][0] == "芝2600m"                                             # 端数は独立の行（直前に足して範囲を広げない）。1 距離だけならその距離
    assert c[("東京", "ダート", 1600)][0] == "ダート1600m"                                      # 芝とダートはまとめない


def test_rail_categories_blocks_and_day_positions():
    days = pd.DataFrame({"date": [f"2025-01-{d:02d}" for d in (4, 5, 11, 12, 13, 18, 19)], "venue": "中山", "kai": 1, "day": [1, 2, 3, 4, 5, 6, 7]})
    rail = pd.DataFrame({"date": days["date"], "venue": "中山", "letter": ["B", "B", "C", "C", "C", "C", "C"]})
    r = M.rail_categories(days, rail).set_index("date")["rail_cat"].to_dict()
    assert r["2025-01-04"] == "開幕 1-2日目" and r["2025-01-05"] == "開幕 1-2日目"                    # 最初の柵（B）が開幕
    assert r["2025-01-11"] == "柵替わり 1-2日目" and r["2025-01-12"] == "柵替わり 1-2日目"             # C に替わった初日・2 日目は、開幕週に近い
    assert r["2025-01-13"] == "柵替わり 3-4日目" and r["2025-01-18"] == "柵替わり 3-4日目"
    assert r["2025-01-19"] == "柵替わり 5日目〜"
    assert set(M.rail_categories(days, rail.iloc[0:0]).columns) == {"date", "venue", "letter", "rail_cat"}   # コース区分が無ければ空


def test_rail_rows_only_for_turf_days_with_letters(monkeypatch):
    rail = pd.DataFrame({"date": ["2019-04-06", "2019-04-13"], "venue": "東京", "letter": ["A", "B"]})
    d = _model(monkeypatch, rail=rail)
    assert _row(d, "rail:開幕 1-2日目")["k"][0]["t"] == [1, 1, 0, 2]                                 # 4/6（A）が開幕 1 日目
    assert _row(d, "letter:B")["k"][0]["t"] == [1, 0, 1, 1]
    assert "letter:C" not in {r["id"] for r in d["rows"]}
    d2 = _model(monkeypatch)
    assert not [r for r in d2["rows"] if r["id"].startswith(("rail:", "letter:"))]                  # 柵位置が無ければ行を作らない


def test_mother_types_reads_positions_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "PED5", tmp_path / "ped")
    monkeypatch.setattr(M, "COLORS", tmp_path / "c.csv")
    (tmp_path / "c.csv").write_text("x", encoding="utf-8")
    d = tmp_path / "ped" / "2020"
    d.mkdir(parents=True)
    anc = [{"generation": 1, "position": 0, "horse_id": "sA"},      # 父（母方ではない）
           {"generation": 2, "position": 2, "horse_id": "sB"},      # 母父
           {"generation": 3, "position": 6, "horse_id": "sA"},      # 母母父
           {"generation": 3, "position": 4, "horse_id": "sA"}]      # 母父の父（母母父ではない）
    (d / "2020100001.json").write_text(json.dumps({"ancestors": anc}), encoding="utf-8")
    (d / "2020100002.json").write_text(json.dumps({"ancestors": [{"generation": 1, "position": 0, "horse_id": "sA"}]}), encoding="utf-8")
    cache = tmp_path / "cache.json"
    out = M.mother_types(["2020100001", "2020100002", "2020100003"], LEDGER, cache)
    assert out == {"2020100001": ["日本型サンデー系", "米国型ミスプロ系"]}                          # 母父・母母父だけ。母方が分からない馬・ファイルの無い馬は入れない
    (d / "2020100001.json").write_text(json.dumps({"ancestors": []}), encoding="utf-8")
    assert M.mother_types(["2020100001"], LEDGER, cache) == {"2020100001": ["日本型サンデー系", "米国型ミスプロ系"]}   # 2 回目はキャッシュから


def test_html_embeds_valid_data_and_is_escaped(monkeypatch):
    monkeypatch.setattr(M, "MIN_LINE_RUNS", 1)
    runs = _runs().assign(line=lambda r: r.line.replace({"A系": "</script><b>A"}))
    meta = {"</script><b>A": {"sires": 1, "o": 1, "large": "甲系"}, "B系": META["B系"]}
    h = M.build(runs, meta, LEDGER, None, {"h1": ["日本型サンデー系"]})
    body = re.search(r'<script type="application/json" id="data">([\s\S]*?)</script>', h).group(1)
    assert "<" not in body                                                                   # </script> や <!-- で HTML の解析が切れない
    d = json.loads(body)
    assert any(c["name"] == "</script><b>A" for c in d["tables"][0]["cols"])
    assert "<table" in h and 'data-tab="1"' in h and "クラス別" in h and "競走中止・取消・除外は数えない" in h


def _pure():
    qjs = pytest.importorskip("quickjs")
    c = qjs.Context()
    c.eval(M.JS_PURE)
    return c


def test_js_pure_never_returns_nan_for_small_samples_and_orders_columns():
    c = _pure()
    assert c.eval("PURE.val([0,0,0,0],0.3,'p')") == pytest.approx(0.3)                                      # 出走 0 でも基準そのもの（NaN にしない）
    v = c.eval("PURE.val([2,0,0,0],0.3,'p')")                                                              # 2 走 2 勝（複勝率 100%）→ 縮めた率（基準より上、100% 未満）
    assert 0.3 < v < 1.0 and c.eval("PURE.val([1,0,0,2],0.3,'p')") == pytest.approx((1 + 30 * 0.3) / (3 + 30))
    assert c.eval("PURE.level([2,0,0,0],0.3,'p')") > 0 and c.eval("PURE.level([0,0,0,0],0.3,'p')") is None
    assert c.eval("PURE.level([2,0,0,0],NaN,'p')") is None                                                  # 行に全くデータが無いときだけ色を付けない
    assert json.loads(c.eval("JSON.stringify(PURE.sortIdx([0.2,NaN,0.5,0.1],1))")) == [2, 0, 3, 1]
    c.eval("var cols=[{large:'A'},{large:'A'},{large:'B'},{large:'B'}];var rc=[[1,0,0,9],[5,3,2,0],[0,0,0,10],[3,3,3,1]];")
    g = json.loads(c.eval("JSON.stringify(PURE.orderCols(cols,rc,0.4,'p',1))"))
    assert g == [{"large": "A", "idx": [1, 0]}, {"large": "B", "idx": [3, 2]}]
    g2 = json.loads(c.eval("JSON.stringify(PURE.orderCols(cols,rc,0.4,'p',-1))"))
    assert g2 == [{"large": "B", "idx": [2, 3]}, {"large": "A", "idx": [0, 1]}]
    assert json.loads(c.eval("JSON.stringify(PURE.orderCols(cols,null,NaN,'p',1))")) == [{"large": "A", "idx": [0, 1]}, {"large": "B", "idx": [2, 3]}]


def test_js_pure_colors_are_continuous_and_bright_means_good():
    c = _pure()
    lo, mid, hi = (json.loads(c.eval(f"JSON.stringify(PURE.color({s}))")) for s in (-1, 0, 1))
    assert lo["lum"] < mid["lum"] < hi["lum"] and hi["hi"] and not mid["hi"]


def test_script_is_valid_javascript():
    esprima = pytest.importorskip("esprima")
    h = M.build(_runs(), META, LEDGER, None, {"h1": ["日本型サンデー系"]})
    for s in re.findall(r"<script>([\s\S]*?)</script>", h):
        esprima.parseScript(s)


def _ui():
    qjs = pytest.importorskip("quickjs")
    old = (M.MIN_LINE_RUNS, M.MIN_PAIR_RUNS)
    M.MIN_LINE_RUNS = M.MIN_PAIR_RUNS = 1
    try:
        runs = pd.DataFrame(
            [{"race_id": f"{v}{sf}{i}{ln}{k}", "date": "2019-04-06", "venue": v, "surface": sf, "distance": d, "day": 1, "kai": 1, "horse_id": f"h{ln}", "line": ln,
              "father_large": {"A系": "甲系", "B系": "甲系", "C系": "乙系"}[ln], "pos": p, "cond": "良", "cls": "2勝"}
             for v, sf, d in (("東京", "芝", 1600), ("東京", "ダート", 1400), ("阪神", "芝", 1600))
             for i, (ln, ps) in enumerate((("A系", [1, 2, 3, 5]), ("B系", [6, 7, 8, 9]), ("C系", [1, 1, 4, 5])))
             for k, p in enumerate(ps)])
        meta = {"A系": {"sires": 2, "o": 2, "large": "甲系"}, "B系": {"sires": 2, "o": 0, "large": "甲系"}, "C系": {"sires": 2, "o": 0, "large": "乙系"}}
        mat = {"hA系": ["日本型サンデー系"], "hB系": ["日本型サンデー系"], "hC系": ["米国型ミスプロ系"]}
        h = M.build(runs, meta, LEDGER, None, mat)
    finally:
        M.MIN_LINE_RUNS, M.MIN_PAIR_RUNS = old
    sc = re.findall(r"<script[^>]*>([\s\S]*?)</script>", h)
    c = qjs.Context()
    c.eval("""var __h={}, __els={};
function El(id){return {id:id,value:'',checked:true,textContent:'',innerHTML:'',dataset:{},classList:{toggle:function(){}},style:{setProperty:function(){}},addEventListener:function(ev,fn){__h[id+':'+ev]=fn},querySelector:function(){return {offsetHeight:30}}}}
var __tabs=[El('t0'),El('t1')];__tabs[0].dataset.tab='0';__tabs[1].dataset.tab='1';
var document={getElementById:function(id){return __els[id]=__els[id]||El(id)},querySelectorAll:function(s){return s==='.tab'?__tabs:[]}};
document.getElementById('data').textContent=%s;""" % json.dumps(sc[0]))
    c.eval(sc[1])
    c.eval(sc[2])
    html = lambda: c.eval("document.getElementById('m').innerHTML")

    def rows():
        return re.findall(r'<th class="sticky rl" data-row="([^"]+)">', html())

    def cols():
        return re.findall(r'<th class="line[^"]*" data-key="\d+"><div class="nm">([^<▼▲ ]*)', html())

    def click(kind, val):
        pick = {"key": "th[data-key]", "row": "th[data-row]", "tg": ".tg"}[kind]
        field = {"key": "key", "row": "row", "tg": "v"}[kind]
        c.eval("(function(){var e={target:{closest:function(s){ return s===%s?{dataset:{%s:%s}}:null; }}};__h['m:click'](e)})()" % (json.dumps(pick), field, json.dumps(val)))

    def setf(i, **kw):
        sets = ";".join(f"x.{k}={json.dumps(v)}" for k, v in kw.items())
        ev = "input" if i == "minn" else "change"                      # 最少出走数は input イベント、他は change
        c.eval(f"(function(){{var x=document.getElementById('{i}');{sets};__h['{i}:{ev}'].call(x)}})()")
    return c, rows, cols, click, setf, html


def test_ui_tabs_class_rows_sorting_and_state_consistency():
    c, rows, cols, click, setf, html = _ui()
    info = lambda: c.eval("document.getElementById('sortinfo').textContent")
    assert rows()[:2] == ["class:芝:2勝", "class:芝:3勝以上"] or rows()[0] == "class:芝:2勝"               # クラス別（芝）が先頭
    assert "class:ダート:2勝" in rows() and rows().index("class:芝:3勝以上") > rows().index("class:芝:2勝") if "class:芝:3勝以上" in rows() else True
    assert cols() == ["A系", "B系", "C系"]
    c.eval("__tabs[1].__h=null;__h['t1:click']()")                                                          # 系統配合のタブへ
    assert cols() == ["日本型サンデー系", "米国型ミスプロ系"] and "全体" in html()
    c.eval("__h['t0:click']()")
    assert cols() == ["A系", "B系", "C系"]                                                                  # 戻る
    setf("minn", value="1")
    click("tg", "東京")
    assert [r for r in rows() if r.startswith("course:東京")] == ["course:東京:芝:芝1600m", "course:東京:ダート:ダート1400m"]       # 芝が先、ダートが後
    setf("flat", checked=False)                                                                             # 大系統の枠を保つ並べ方（既定は枠をこえる）で確かめる
    click("row", "venue:東京:all")                                                                          # 東京の行の複勝率（A 75%・B 0%・C 50%）で列を並べ替え → 降順
    assert cols() == ["C系", "A系", "B系"] and "列の並べ替え（複勝率・大系統の枠の中）: 東京 全コース ▼降順" in info()
    setf("sf", value="芝")                                                                                  # 芝のみ: 基準の行は「東京 芝全体」に付け替わる
    assert "列の並べ替え（複勝率・大系統の枠の中）: 東京 芝全体" in info() and not [r for r in rows() if r.startswith("class:ダート")]
    click("key", 0)
    assert "行の並べ替え（複勝率）: A系 ▼降順" in info()
    setf("onlyo", checked=True)
    assert cols() == ["A系"]


def test_rail_categories_do_not_mix_years_and_skip_blocks_with_unknown_days():
    # 2017・2018 の「1回中山」を別の開催として扱う（年をまたいで日目を並べない）
    days = pd.DataFrame({"date": ["2017-01-05", "2017-01-06", "2018-01-06", "2018-01-07"], "venue": "中山", "kai": 1, "day": [1, 2, 1, 2]})
    rail = pd.DataFrame({"date": days["date"], "venue": "中山", "letter": ["A", "A", "B", "B"]})
    r = M.rail_categories(days, rail).set_index("date")["rail_cat"].to_dict()
    assert r == {"2017-01-05": "開幕 1-2日目", "2017-01-06": "開幕 1-2日目", "2018-01-06": "開幕 1-2日目", "2018-01-07": "開幕 1-2日目"}
    # A(1日目) → 不明(2〜4日目) → B(5日目): 5 日目が柵替わりの初日とは言えない。特定できないので付けない
    days2 = pd.DataFrame({"date": [f"2019-02-0{d}" for d in (1, 2, 3, 4, 5, 6, 7)], "venue": "東京", "kai": 1, "day": [1, 2, 3, 4, 5, 6, 7]})
    rail2 = pd.DataFrame({"date": ["2019-02-01", "2019-02-05", "2019-02-06", "2019-02-07"], "venue": "東京", "letter": ["A", "B", "B", "C"]})
    r2 = M.rail_categories(days2, rail2).set_index("date")["rail_cat"].to_dict()
    assert r2["2019-02-01"] == "開幕 1-2日目" and r2["2019-02-02"] is None and r2["2019-02-05"] is None and r2["2019-02-06"] is None   # 不明な日をまたぐ間は付けない
    assert r2["2019-02-07"] == "柵替わり 1-2日目"                                                      # 前日（B）が分かり、C に替わった日は特定できる


def test_mother_types_cache_does_not_store_horses_without_pedigree_files(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "PED5", tmp_path / "ped")
    monkeypatch.setattr(M, "COLORS", tmp_path / "c.csv")
    (tmp_path / "c.csv").write_text("x", encoding="utf-8")
    (tmp_path / "ped" / "2020").mkdir(parents=True)
    cache = tmp_path / "cache.json"
    assert M.mother_types(["2020100009"], LEDGER, cache) == {}                                     # ファイルがまだ無い
    (tmp_path / "ped" / "2020" / "2020100009.json").write_text(json.dumps({"ancestors": [{"generation": 2, "position": 2, "horse_id": "sA"}]}), encoding="utf-8")
    assert M.mother_types(["2020100009"], LEDGER, cache) == {"2020100009": ["日本型サンデー系"]}      # 後で取得されたら付く（空をキャッシュしない）
    monkeypatch.setattr(M, "MAT_PATHS", [(2, 2), (3, 6)])                                         # 対象の位置が変わったら作り直す
    (tmp_path / "ped" / "2020" / "2020100009.json").write_text(json.dumps({"ancestors": [{"generation": 2, "position": 2, "horse_id": "sB"}]}), encoding="utf-8")
    assert M.mother_types(["2020100009"], LEDGER, cache) == {"2020100009": ["米国型ミスプロ系"]}


def test_mix_table_has_group_runs_without_overlap(monkeypatch):
    mat = {"h1": ["日本型サンデー系", "米国型ミスプロ系"], "h2": ["日本型サンデー系"], "h3": ["米国型ミスプロ系"]}
    d = _model(monkeypatch, mat=mat)
    t1 = d["tables"][1]
    assert t1["groups"] == {"甲系": 7, "乙系": 4}                                                   # 父の大系統の実数（重ねて数えない）
    assert sum(c["runs"] for c in t1["cols"] if c["large"] == "甲系") == 11                          # 列の和は、2 つの型に当たる馬を重ねるので大きくなる
    r = _row(d, "class:芝:2勝")["k"][1]
    assert r["g"] == {"甲系": [2, 1, 1, 1]} and r["t"] == [2, 1, 1, 1]


def test_ui_sort_metric_is_independent_from_color_metric():
    c, rows, cols, click, setf, html = _ui()
    info = lambda: c.eval("document.getElementById('sortinfo').textContent")
    setf("minn", value="1")
    click("key", 0)
    assert "行の並べ替え（複勝率）" in info()                                    # 既定は色の基準（複勝率）と同じ
    bgs = lambda: sorted(re.findall(r'style="background:(rgb\([^)]*\))', html()))           # セルの色（色の基準だけで決まる）
    before = bgs()
    setf("sortmetric", value="w")                                                # 並べ替えだけ勝率にする（色の基準は複勝率のまま）
    assert "行の並べ替え（勝率）" in info() and bgs() == before and before
    setf("metric", value="w")                                                    # 色の基準を勝率にしても、並べ替えの基準は選んだまま
    setf("sortmetric", value="p")
    assert "行の並べ替え（複勝率）" in info()
    setf("sortmetric", value="")                                                 # 空＝色の基準と同じ（いまは勝率）
    assert "行の並べ替え（勝率）" in info()


def test_js_pure_flat_order_crosses_groups_and_merges_adjacent_runs():
    c = _pure()
    c.eval("var cols=[{large:'A'},{large:'A'},{large:'B'},{large:'B'}];var rc=[[1,0,0,9],[5,3,2,0],[0,0,0,10],[3,3,3,1]];")
    f = json.loads(c.eval("JSON.stringify(PURE.orderCols(cols,rc,0.4,'p',1,null,true))"))
    assert f == [{"large": "A", "idx": [1]}, {"large": "B", "idx": [3]}, {"large": "A", "idx": [0]}, {"large": "B", "idx": [2]}]   # A,B,A,B と交互に並ぶ（枠をこえる）。隣り合う同じ大系統だけまとめる
    f2 = json.loads(c.eval("JSON.stringify(PURE.orderCols(cols,rc,0.4,'p',1,null,false))"))
    assert f2 == [{"large": "A", "idx": [1, 0]}, {"large": "B", "idx": [3, 2]}]                                                   # 枠を保つ並べ方は従来どおり


def test_ui_flat_column_sort_crosses_groups_by_default_and_can_keep_groups():
    c, rows, cols, click, setf, html = _ui()
    setf("minn", value="1")
    click("row", "venue:東京:all")                                                                          # 東京の行の複勝率（A 75%・C 50%・B 0%）。既定は大系統をまたぐ
    assert cols() == ["A系", "C系", "B系"]                                                                  # 甲系（A）→乙系（C）→甲系（B）と交互。枠に縛られない
    assert html().count('class="grp"') == 3                                                                  # 1 階層目は、隣り合う同じ大系統ごと（甲・乙・甲）
    setf("flat", checked=False)                                                                              # 枠を保つ
    assert cols() == ["C系", "A系", "B系"] and html().count('class="grp"') == 2


def test_load_rail_uses_only_netkeiba_daily_values(tmp_path, monkeypatch):
    cpos = tmp_path / "course_position"
    cpos.mkdir()
    (cpos / "course_position_2026.json").write_text(json.dumps({"version": 1, "dates": {
        "2026-10-03": [{"venue": "京都", "kai": 4, "day": 1, "turf_course": "A", "turf_condition": "良", "dirt_condition": "良"}],
        "2026-10-31": [{"venue": "京都", "kai": 4, "day": 10, "turf_course": "B", "turf_condition": "良", "dirt_condition": "良"},
                       {"venue": "東京", "kai": 5, "day": 1, "turf_course": None, "turf_condition": "良", "dirt_condition": "良"}]}}), encoding="utf-8")
    monkeypatch.setattr(M, "CPOS", cpos)
    r = M.load_rail()
    assert {(x.date, x.venue): x.letter for x in r.itertuples()} == {("2026-10-03", "京都"): "A", ("2026-10-31", "京都"): "B"}   # 日ごとに替わる値。コース区分が無い会場は入れない
    assert not hasattr(M, "CUSHION")                                                           # cushion の course_position（開催の最後の値が全日に入る）は使わない


def test_ui_split_headers_sort_only_their_columns_and_say_what_they_cover():
    c, rows, cols, click, setf, html = _ui()
    info = lambda: c.eval("document.getElementById('sortinfo').textContent")
    setf("minn", value="1")
    click("row", "venue:東京:all")                                                                          # 大系統をまたぐ: 甲(A) → 乙(C) → 甲(B) と割れる
    keys = re.findall(r'<th class="grp" colspan="\d+" data-key="([^"]+)">', html())
    assert keys == ["c:0", "g:乙系", "c:1"]                                                                 # 割れた見出しは列ごとのキー。割れていない大系統は従来のキー
    assert html().count("この枠") == 2 and "全体" in html()                                                   # 割れた見出しは「この枠」の走数
    click("key", "c:1")                                                                                     # 見出し（B 系だけ）で行を並べ替え → 矢印はその見出しだけ
    assert "見出しの枠の列（B系）" in info()
    head = html().split("</thead>")[0]
    assert head.count(" ▼") == 1 and 'data-key="c:1"><div' not in head or "▼" in head                    # 見出しの矢印は、押した見出し 1 つだけ（行名の矢印は表の本体）
    assert "大系統をまたぐ" in info()
