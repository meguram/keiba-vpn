"""馬の戦績表（/horse/result/）を列名で読む: スタート指数・追走指数・上がり指数・厩舎コメント・備考と、通過・上り・勝ち馬。

実ヘッダーは33列（会員表示）。ヘッダー名にはツールチップの文言が後ろに付く。サンプルは合成の最小例。
"""

from __future__ import annotations

from src.scraper.parsers import HorseParser

H33 = ["日付", "開催", "天気", "R", "レース名", "映像", "頭数", "枠番", "馬番", "オッズ", "人気", "着順", "騎手", "斤量", "距離", "水分量", "馬場", "馬場指数",
       "タイム", "着差", "ﾀｲﾑ指数タイム指数(通常)タイム指数マスター", "ﾀｲﾑ指数Mタイム指数(通常)タイム指数マスター", "ｽﾀｰﾄ指数", "追走指数", "上がり指数",
       "通過", "ペース", "上り", "馬体重", "厩舎ｺﾒﾝﾄ", "備考", "勝ち馬(2着馬)", "賞金"]


def _row33(**over):
    v = {"日付": "2026/10/12", "開催": "5東京1", "天気": "晴", "R": "7", "レース名": "テスト", "映像": "", "頭数": "16", "枠番": "5", "馬番": "9", "オッズ": "12.3",
         "人気": "6", "着順": "13", "騎手": "騎手A", "斤量": "56", "距離": "芝2200", "水分量": "", "馬場": "良", "馬場指数": "10", "タイム": "2:14.3", "着差": "2.5",
         "ﾀｲﾑ指数タイム指数(通常)タイム指数マスター": "-", "ﾀｲﾑ指数Mタイム指数(通常)タイム指数マスター": "73", "ｽﾀｰﾄ指数": "62", "追走指数": "88", "上がり指数": "91",
         "通過": "15-15-14-12", "ペース": "36.5-35.0", "上り": "35.4", "馬体重": "508(+2)", "厩舎ｺﾒﾝﾄ": "", "備考": "出遅", "勝ち馬(2着馬)": "勝ち馬X", "賞金": ""}
    v.update(over)
    return "<tr>" + "".join(f"<td>{v[h]}</td>" for h in H33) + "</tr>"


def _html(rows, headers=H33):
    return ('<html><body><table class="db_h_race_results"><tr>' + "".join(f"<th>{h}</th>" for h in headers) + "</tr>" + "".join(rows) + "</table></body></html>")


def _hist(html):
    return HorseParser().parse("<html></html>", horse_id="x", result_html=html)["race_history"]


def test_named_columns_33():
    h = _hist(_html([_row33(), _row33(**{"日付": "2026/06/14", "備考": "", "ｽﾀｰﾄ指数": "**", "厩舎ｺﾒﾝﾄ": "好調"})]))
    a, b = h
    assert (a["start_index"], a["chase_index"], a["closing_index"]) == (62, 88, 91)
    assert a["remarks"] == "出遅" and a["stable_comment"] == ""
    assert a["passing_order"] == "15-15-14-12" and a["last_3f"] == 35.4 and a["winner"] == "勝ち馬X"
    assert a["time_index"] == 73 and a["weight"] == 508 and a["weight_change"] == 2
    assert b["start_index"] is None and b["remarks"] == "" and b["stable_comment"] == "好調"


def test_old_layout_keeps_fixed_positions_and_adds_no_new_keys():
    # 指数・備考の列が無い短い表（旧レイアウト）。固定位置で読み、新しいキーは出さない
    hs = ["日付", "開催", "天気", "R", "レース名", "映像", "頭数", "枠番", "馬番", "オッズ", "人気", "着順", "騎手", "斤量", "距離", "水分量", "馬場", "指数", "タイム", "着差",
          "ペース", "通過", "ペース", "上り", "馬体重", "x", "y", "勝ち馬", "賞金"]
    cells = {i: "" for i in range(len(hs))}
    cells.update({0: "2024/01/06", 1: "1中山1", 3: "11", 4: "R", 6: "16", 7: "1", 8: "2", 11: "1", 14: "ダ1800", 16: "良", 18: "1:51.0", 19: "0.1", 21: "3-3-2-1", 23: "36.1", 24: "480(0)", 27: "W"})
    row = "<tr>" + "".join(f"<td>{cells[i]}</td>" for i in range(len(hs))) + "</tr>"
    (r,) = _hist(_html([row], hs))
    assert r["passing_order"] == "3-3-2-1" and r["last_3f"] == 36.1 and r["winner"] == "W"
    assert not any(k in r for k in ("start_index", "chase_index", "closing_index", "remarks", "stable_comment"))


def test_remarks_record_and_audit():
    from src.scraper import horse_race_remarks as hr

    hist = _hist(_html([_row33(**{"race_id": "x"})]))
    hist[0]["race_id"] = "202605040507"
    rec = hr.to_record("2022106199", hist, "2026-10-17T00:00:00+09:00")
    (row,) = rec["race_history"]
    assert row["race_id"] == "202605040507" and row["date"] == "2026-10-12" and row["remarks"] == "出遅" and row["start_index"] == 62
    assert hr.to_record("x", [{"race_id": "", "start_index": 1}], "t")["race_history"] == []        # race_id の無い行は捨てる
    assert hr.audit_html(_html([_row33()])) == "ok"
    assert hr.audit_html(_html([_row33()], [h for h in H33 if h not in ("備考",)])) == "unreadable"   # 必須列が無い → 保存しない
    assert hr.audit_html("<html><body>no table</body></html>") == "empty"
