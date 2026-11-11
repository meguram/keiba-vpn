import re

import numpy as np
import pandas as pd

from src.scripts.data import trainer_pattern_table as T


def _frame():
    """調教師 t1 のパターン A: 日付ごとに 1 頭。着順 1,2,3,5 / 払戻 単勝 4.0 のとき複勝 1.5, 複勝 1.2, 複勝 1.1。2 日目は 2 頭が同日。"""
    rows = [
        ("r1", "2021-01-01", 1, 4.0, 1.5), ("r2", "2021-01-08", 2, 0.0, 1.2),
        ("r3", "2021-01-15", 3, 0.0, 1.1), ("r4", "2021-01-22", 5, 0.0, 0.0),
        ("r5", "2021-01-22", 1, 3.0, 1.3),                                    # r4 と同日（同日分は互いの累計に入らない）
    ]
    d = pd.DataFrame(rows, columns=["race_id", "race_date", "finish_position", "win_ret", "place_ret"])
    d["horse_id"], d["horse_number"] = [f"h{i}" for i in range(len(d))], 1
    d["trainer_id"], d["trainer_name"], d["pat"], d["year"] = "t1", "調教師1", "栗東坂路|A2", 2021
    x = d.iloc[[0]].copy()   # 別パターン・別調教師は混ざらない
    x["race_id"], x["horse_id"], x["pat"], x["trainer_id"], x["race_date"], x["finish_position"] = "r9", "hz", "美浦坂路|X", "t2", "2021-01-15", 1
    y = d.iloc[[0]].copy()   # パターンなし（調教の記録なし）
    y["race_id"], y["horse_id"], y["pat"] = "r8", "hn", np.nan
    return pd.concat([d, x, y], ignore_index=True)


def test_summary_counts_rates_and_rois():
    S = T.summary(_frame(), 2021, 2021).set_index(["調教師ID", "調教パターン"])
    r = S.loc[("t1", "栗東坂路|A2")]
    assert r["出走数"] == 5 and r["着順内訳"] == "2-1-1-1" and r["4着以下"] == 1
    assert abs(r["勝率"] - 2 / 5) < 1e-9 and abs(r["複勝率"] - 4 / 5) < 1e-9
    assert abs(r["単勝回収率"] - 7.0 / 5) < 1e-9 and abs(r["複勝回収率"] - (1.5 + 1.2 + 1.1 + 0 + 1.3) / 5) < 1e-9
    assert len(S) == 2   # パターンなしの行は入らない


def test_summary_excludes_unreadable_returns_from_both_numerator_and_denominator():
    d = _frame()
    d.loc[d["race_id"] == "r4", ["win_ret", "place_ret"]] = np.nan
    r = T.summary(d, 2021, 2021).set_index(["調教師ID", "調教パターン"]).loc[("t1", "栗東坂路|A2")]
    assert r["出走数"] == 4 and r["着順内訳"] == "2-1-1-0"


def test_asof_uses_only_strictly_earlier_dates():
    A = T.asof_table(_frame()).set_index("race_id")
    assert A.loc["r1", "trpat_n"] == 0 and np.isnan(A.loc["r1", "trpat_win_rate"])
    assert A.loc["r2", "trpat_n"] == 1 and A.loc["r2", "trpat_c1"] == 1 and abs(A.loc["r2", "trpat_win_roi"] - 4.0) < 1e-9
    assert A.loc["r3", "trpat_n"] == 2 and abs(A.loc["r3", "trpat_top3_rate"] - 1.0) < 1e-9
    for rid in ("r4", "r5"):   # 同日の 2 頭はどちらも前の 3 日分だけ
        assert A.loc[rid, "trpat_n"] == 3 and A.loc[rid, "trpat_c4p"] == 0
        assert abs(A.loc[rid, "trpat_place_roi"] - (1.5 + 1.2 + 1.1) / 3) < 1e-9
    assert A.loc["r9", "trpat_n"] == 0   # 別調教師・別パターンは混ざらない
    assert "r8" not in A.index


def test_asof_final_total_matches_summary():
    d = _frame()
    A, S = T.asof_table(d), T.summary(d, 2021, 2021)
    last = A[A["trainer_id"] == "t1"].sort_values("race_date").iloc[-1]
    # 最後の日の累計 + その日の 2 頭 = 期間集計
    assert last["trpat_n"] + 2 == S.set_index("調教師ID").loc["t1", "出走数"]


def test_asof_order_independent_multi_pattern_and_unknown_trainer():
    d = _frame()
    z = d[d["race_id"] == "r2"].copy()   # 同じ調教師 t1 の別パターン（B）。A の累計に混ざらない
    z["race_id"], z["horse_id"], z["pat"], z["race_date"] = "r10", "hq", "栗東坂路|B1", "2021-01-20"
    u = d[d["race_id"] == "r1"].copy()   # 調教師不明（"nan" の文字列）は累計しない
    u["race_id"], u["horse_id"], u["trainer_id"], u["race_date"] = "r11", "hu", "nan", "2021-01-02"
    full = pd.concat([d, z, u], ignore_index=True).sample(frac=1, random_state=0)   # 入力は日付順でなくてよい
    A = T.asof_table(full).set_index("race_id")
    assert "r11" not in A.index
    assert A.loc["r10", "trpat_n"] == 0 and A.loc["r4", "trpat_n"] == 3 and A.loc["r4", "trpat_c4p"] == 0
    S = T.summary(full, 2021, 2021)
    assert "nan" not in set(S["調教師ID"]) and S.set_index(["調教師ID", "調教パターン"]).loc[("t1", "栗東坂路|B1"), "出走数"] == 1


def test_summary_trainer_name_is_the_latest_one():
    d = _frame()
    d.loc[d["race_id"] == "r1", "trainer_name"] = "旧名"
    assert T.summary(d, 2021, 2021).set_index("調教師ID").loc["t1", "調教師"] == "調教師1"


def test_html_block_groups_trainer_place_pattern_with_totals():
    d = _frame()
    z = d[d["race_id"] == "r2"].copy()   # t1 の同じ場所の別パターン
    z["race_id"], z["horse_id"], z["pat"], z["race_date"] = "r10", "hq", "栗東坂路|B1", "2021-01-20"
    S = T.summary(pd.concat([d, z], ignore_index=True), 2021, 2021)
    h = T.html_block(S)
    assert h.startswith(T.START) and h.endswith(T.END) and "|" not in h.split("<table")[1].split("</table>")[0]
    assert h.count("<tbody") == 2 and h.count('<tr class="p') == 2 and h.count('<tr class="q') == 3   # t1(栗東坂路:A2,B1) と t2(美浦坂路:X)
    t1 = h.split("<tbody")[1] if "t1" in h.split("<tbody")[1].split(">")[0] else h.split("<tbody")[2]
    assert '<td class=id>t1</td><td class=l>調教師1</td><td class="num">6</td>' in t1   # 5 + 1 頭の合計
    assert f">{7.0 / 6 * 100:.1f}%</td>" in t1   # 単勝回収率 (4+3)/6 を % 表示（調教師の行と栗東坂路の行の両方）
    assert t1.count(f"{(1.5 + 1.2 + 1.1 + 1.3 + 1.2) / 6 * 100:.1f}%") >= 2   # 複勝回収率
    assert "<td class=l>A2<" in t1 and "<td class=l>B1<" in t1 and "<td class=l>栗東坂路</td>" in t1


def test_inject_replaces_between_markers(tmp_path):
    f = tmp_path / "c.html"
    f.write_text(f"<h3>x</h3>{T.START}古い{T.END}<p>後</p>", encoding="utf-8")
    T.inject(f, f"{T.START}新{T.END}")
    assert f.read_text(encoding="utf-8") == f"<h3>x</h3>{T.START}新{T.END}<p>後</p>"


def test_html_block_escapes_names():
    d = _frame()
    d["trainer_name"] = 'a<b>"&'
    h = T.html_block(T.summary(d, 2021, 2021))
    assert "<b>" not in h and 'a&lt;b&gt;&quot;&amp;' in h


def _light(h, label, cls):
    r = [r for r in h.split("<tr ") if f"<td class=l>{label}</td>" in r][0]
    return [int(v) for v in re.findall(rf'class="num {cls}[^"]*" style="--l:(\d+)"', r)], r


def test_html_block_color_is_brighter_for_better_roi_and_absent_at_or_below_100():
    def row(n, w, p):
        return {"調教師ID": "t", "調教師": "x", "調教パターン": f"栗東坂路|P{n}", "出走数": n, "1着": 1, "2着": 0, "3着": 0, "4着以下": n - 1, "単勝回収率": w, "複勝回収率": p}
    h = T.html_block(pd.DataFrame([row(T.HI_N, 0.9, 1.0), row(T.HI_N + 1, 1.1, 1.3), row(T.HI_N + 2, 1.7, 0.5), row(T.HI_N - 1, 1.7, 1.7)]))
    assert 'class="num gr' not in _light(h, f"P{T.HI_N}", "gr")[1]   # 1.00 以下（ちょうど 1.00 を含む）は色なし
    mid, _ = _light(h, f"P{T.HI_N + 1}", "gr")
    top, rtop = _light(h, f"P{T.HI_N + 2}", "gr")
    assert len(mid) == 2 and len(top) == 1 and mid[0] < top[0]   # 1.1 < 1.7 の順に明るい。1.7 の行の複勝 0.5 は色なし
    assert 'class="num gr' not in _light(h, f"P{T.HI_N - 1}", "gr")[1]   # 30 頭未満は色なし
    h2 = T.html_block(pd.DataFrame([row(T.HI_N, 1.0004, 1.0004)]))
    assert 'class="num gr' not in h2 and ">100.0%<" in h2   # 100.0% と表示されるものも色なし


def test_html_block_rates_brighter_the_further_above_the_overall_level():
    def row(pat, n, c1, c3):
        return {"調教師ID": "t", "調教師": "x", "調教パターン": f"栗東坂路|{pat}", "出走数": n, "1着": c1, "2着": c3 - c1, "3着": 0, "4着以下": n - c3, "単勝回収率": 0.5, "複勝回収率": 0.5}
    h = T.html_block(pd.DataFrame([row("L", 3000, 150, 600), row("M", 100, 20, 40), row("H", 100, 40, 60), row("S", 20, 10, 15)]))
    L, _ = _light(h, "L", "bl")
    M, rm = _light(h, "M", "bl")
    H, rh = _light(h, "H", "bl")
    assert L == [] and 0 < max(M) < max(H) and "bl hot" in rm and "bl hot" in rh   # 水準以下の L は色なし
    assert _light(h, "S", "bl")[0] == []


def test_html_block_sorts_every_level_in_place():
    h = T.html_block(pd.DataFrame([{"調教師ID": "t", "調教師": "x", "調教パターン": "栗東坂路|A", "出走数": 5, "1着": 0, "2着": 0, "3着": 0, "4着以下": 5, "単勝回収率": 0.0, "複勝回収率": 0.0}]))
    assert 'data-v="' not in h.split("</script>")[1] and "ns" not in h.split("<table")[1].split("</thead>")[0]   # 全列が並べ替え対象（ヒストグラムの列だけは 50 倍以上の割合で並べる）


def test_html_block_accents_rows_where_rate_and_roi_both_stand_out():
    def row(pat, n, c1, c3, w, p):
        return {"調教師ID": "t", "調教師": "x", "調教パターン": f"栗東坂路|{pat}", "出走数": n, "1着": c1, "2着": c3 - c1, "3着": 0, "4着以下": n - c3, "単勝回収率": w, "複勝回収率": p}
    S = pd.DataFrame([row("L", 3000, 150, 600, 0.5, 0.5), row("WIN", 100, 20, 30, 1.3, 0.8), row("PLC", 100, 8, 60, 0.8, 1.2), row("ROI", 100, 3, 15, 1.5, 1.5),
                      row("LOWROI", 100, 20, 60, 0.9, 0.9), row("SMALL", 20, 10, 15, 2.0, 2.0)])
    h = T.html_block(S)
    def tr(label):
        return [r for r in h.split("<tr ") if f"<td class=l>{label}</td>" in r][0].split(">")[0]
    assert "ac" in tr("WIN").split('"')[1].split() and "勝率と単勝回収率" in tr("WIN")
    assert "ac" in tr("PLC").split('"')[1].split() and "複勝率と複勝回収率" in tr("PLC")
    for lab in ("L", "ROI", "LOWROI", "SMALL"):   # 回収率だけ（率が水準以下）・率だけ（回収率 100% 以下）・30 頭未満は対象外
        assert "ac" not in tr(lab).split('"')[1].split()
    assert not any("ac" in r.split(">")[0].split('"')[1].split() for r in h.split("<tr ")[1:] if r.startswith(('class="t', 'class="p')))   # 調教師・場所の行にはアクセントを付けない


def test_summary_counts_unique_horses_at_pattern_place_and_trainer_level():
    d = _frame()
    d.loc[d["race_id"].isin(["r2", "r3"]), "horse_id"] = "hA"   # 同じ馬が 2 回、パターン A2 に出走
    z = d[d["race_id"] == "r2"].copy()                            # 同じ馬 hA が同じ場所の別パターンにも出走
    z["race_id"], z["pat"], z["race_date"] = "r10", "栗東坂路|B1", "2021-02-01"
    S = T.summary(pd.concat([d, z], ignore_index=True), 2021, 2021).set_index(["調教師ID", "調教パターン"])
    a, b = S.loc[("t1", "栗東坂路|A2")], S.loc[("t1", "栗東坂路|B1")]
    assert a["出走数"] == 5 and a["頭数"] == 4 and b["頭数"] == 1
    assert a["頭数_場所"] == 4 and a["頭数_調教師"] == 4   # 場所・調教師の段は、パターンをまたいで同じ馬を 1 頭と数える（4 + 1 ではない）


def test_html_block_shows_unique_horses_and_flags_repeated_runners():
    def row(pat, n, u):
        return {"調教師ID": "t", "調教師": "x", "調教パターン": f"栗東坂路|{pat}", "出走数": n, "頭数": u, "頭数_場所": u, "頭数_調教師": u, "1着": 0, "2着": 0, "3着": 0, "4着以下": n, "単勝回収率": 0.0, "複勝回収率": 0.0}
    h = T.html_block(pd.DataFrame([row("ONE", 10, 10), row("REP", 30, 10)]))
    r = [r for r in h.split("<tr ") if "<td class=l>REP</td>" in r][0]
    assert '<td class="num">10</td><td class="num dup">3.00</td>' in r
    assert 'class="num dup"' not in [r for r in h.split("<tr ") if "<td class=l>ONE</td>" in r][0]
    assert "<th>頭数</th>" in h and "<th>出走/頭</th>" in h
    assert "-</td>" in T.html_block(pd.DataFrame([{k: v for k, v in row("Z", 5, 5).items() if not k.startswith("頭数")}]))   # 頭数の列が無い古い集計でも落ちない


def _odds_frame():
    """t1 / 栗東坂路|A2 の 5 走: オッズ 1.5(1着 払戻1.5/複勝1.1), 8(2着, 0/1.5), 49.9(3着, 0/3.0), 50(1着 払戻50.0/複勝10.0), 120(5着)。オッズ不明の 1 走は含めない。"""
    d = _frame().iloc[:5].copy()
    d["odds"] = [1.5, 8.0, 49.9, 50.0, 120.0]
    d["finish_position"], d["win_ret"], d["place_ret"] = [1, 2, 3, 1, 5], [1.5, 0.0, 0.0, 50.0, 0.0], [1.1, 1.5, 3.0, 10.0, 0.0]
    return d


def test_summary_excludes_odds_50_and_above_from_every_number_including_distribution():
    r = T.summary(_odds_frame(), 2021, 2021, max_odds=50.0).set_index(["調教師ID", "調教パターン"]).loc[("t1", "栗東坂路|A2")]
    # 残る 3 走: オッズ 1.5(1着 1.5/1.1), 8(2着 0/1.5), 49.9(3着 0/3.0)。ちょうど 50 倍は除く
    assert r["出走数"] == 3 and r["頭数"] == 3 and r["着順内訳"] == "1-1-1-0" and r["4着以下"] == 0
    assert abs(r["単勝回収率"] - 1.5 / 3) < 1e-9 and abs(r["複勝回収率"] - 5.6 / 3) < 1e-9 and abs(r["勝率"] - 1 / 3) < 1e-9
    assert r["オッズ分布"] == "1-0-0-1-0-1"   # 分布も残った 3 走だけ（1.5 / 8 / 49.9）。50 倍以上の区間は作らない


def test_summary_without_max_odds_keeps_everything_and_without_odds_has_no_distribution():
    r = T.summary(_odds_frame(), 2021, 2021).set_index(["調教師ID", "調教パターン"]).loc[("t1", "栗東坂路|A2")]
    assert r["出走数"] == 5 and abs(r["単勝回収率"] - 51.5 / 5) < 1e-9
    assert r["オッズ分布"] == "1-0-0-1-0-1-1-1"   # 除かないときは 8 区間（1.5 / 8 / 49.9 / 50 / 120）
    assert "オッズ分布" not in T.summary(_frame(), 2021, 2021).columns


def test_summary_drops_a_group_whose_runs_are_all_50_or_more():
    d = _odds_frame()
    d["odds"] = 80.0
    assert T.summary(d, 2021, 2021, max_odds=50.0).empty


def test_html_block_shows_distribution_and_no_split_columns():
    h = T.html_block(T.summary(_odds_frame(), 2021, 2021, max_odds=50.0))
    q = [r for r in h.split("<tr ") if "<td class=l>A2<" in r][0]
    assert 'data-b="1,0,0,1,0,1"' in q and 'data-v="0.3333"' in q   # 20 倍以上は 1/3
    assert "<th>オッズ分布</th>" in h and "50倍" not in h.split("<table")[1].split("</thead>")[0]
    assert h.count('data-b="1,0,0,1,0,1"') == 3   # 調教師・場所・パターンの 3 行


def test_html_block_has_accent_only_button_and_hides_non_accent_rows_by_class():
    h = T.html_block(T.summary(_odds_frame(), 2021, 2021, max_odds=50.0))
    assert 'id="tpta"' in h and "tr.na" in h and "tbody.na" in h


def test_html_block_explains_slope_lap_patterns_only_on_slope_rows():
    def row(pat):
        return {"調教師ID": "t", "調教師": "x", "調教パターン": pat, "出走数": 5, "1着": 0, "2着": 0, "3着": 0, "4着以下": 5, "単勝回収率": 0.0, "複勝回収率": 0.0}
    h = T.html_block(pd.DataFrame([row("栗東坂路|A3"), row("栗東坂路|X"), row("栗東ウッド|A3")]))
    assert "<td class=l>A3<span class=ds>（11秒台以下で加速）</span></td>" in h and "<td class=l>X<span class=ds>（終い2Fとも13秒以上）</span></td>" in h
    assert h.count("11秒台以下で加速") == 1   # 坂路以外の行には付けない
