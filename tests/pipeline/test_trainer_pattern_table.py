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
    assert '<td class=id>t1</td><td class=l>調教師1</td><td class="num" data-v="6.000000">6</td>' in t1   # 5 + 1 頭の合計
    assert f'data-v="{7.0 / 6:.6f}"' in t1 and f'<td class="num"' in t1 and f">{7.0 / 6:.3f}</td>" in t1 in t1   # 単勝回収率 (4+3)/6（調教師の行と栗東坂路の行の両方）
    assert t1.count(f"{(1.5 + 1.2 + 1.1 + 1.3 + 1.2) / 6:.3f}") >= 2   # 複勝回収率
    assert "<td class=l>A2</td>" in t1 and "<td class=l>B1</td>" in t1 and "<td class=l>栗東坂路</td>" in t1


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


def test_html_block_marks_roi_over_100_only_with_enough_runs():
    def row(n, w, p):
        return {"調教師ID": "t", "調教師": "x", "調教パターン": f"栗東坂路|P{n}", "出走数": n, "1着": 1, "2着": 0, "3着": 0, "4着以下": n - 1, "単勝回収率": w, "複勝回収率": p}
    h = T.html_block(pd.DataFrame([row(T.HI_N, 1.2, 0.9), row(T.HI_N - 1, 1.2, 1.5), row(T.HI_N, 1.0, 1.0)]))
    assert h.count('class="num hi"') == 5   # 30 頭の P30 の単勝 1 + 合計（89 頭: 単勝 1.133・複勝 1.129）が調教師の行と場所の行で 2 × 2。29 頭の行と、ちょうど 1.0 は対象外
    assert '<td class="num hi">1.200</td>' in h and '<td class="num hi">1.500</td>' not in h
