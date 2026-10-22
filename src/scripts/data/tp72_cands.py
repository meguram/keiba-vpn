"""tp-72 第 1 段: 『買う要素』の候補（型）を事前に列挙して固定する。結果を見る前に実行し、tp72_candidates.json をコミットする。

python -m src.scripts.data.tp72_cands            # tp72_candidates.json を書く（件数を表示）

各候補 = (family, name, expr)。expr は tp72_data.py が作る表（mining.parquet）の列に対する pandas の式。
買う側（正の残差・妙味）だけを片側で検定する。結果を見てから型を増やさない（増やす場合は別の探索として分けて記録する）。
候補の定義に使う閾値（±10kg、3 番手以内、90 日など）は慣習値で、データを見て決めたものではない。
クッション値・含水率の三分位は 2020〜2021 年から固定済み（tp72_data.py）。
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("tp72_candidates.json")

SEX = {"牡": "male", "牝": "female", "セ": "gelding"}
SEASONS = ["spring", "summer", "autumn", "winter"]
STYLES = ["front", "mid", "closer"]
DG = {"sprint": "(dist_band=='d_sprint')", "mile": "(dist_band=='d_mile_s')|(dist_band=='d_mile')", "mid_long": "(dist_band=='d_mid')|(dist_band=='d_long')"}
REST = {"rest90": "lay_days>=90", "rest180": "lay_days>=180"}
WT_DW = {"dw_up": "weight_change>=10", "dw_down": "weight_change<=-10", "dw_stable": "abs(weight_change)<=4"}
WT_DEV = {"dev_up": "wt_dev>=8", "dev_down": "wt_dev<=-8", "dev_stable": "abs(wt_dev)<=4"}
EV = {"ev_good": "ev_score>=3", "ev_up": "ev_delta>=1", "ev_down": "ev_delta<=-1"}


def _and(*xs: str) -> str:
    return " & ".join(f"({x})" for x in xs)


def build() -> list[dict]:
    c: list[dict] = []

    def add(fam: str, name: str, expr: str) -> None:
        c.append({"family": fam, "name": name, "expr": expr})

    # F1 性別 × 季節 / F2 性別 × 月
    for sx, en in SEX.items():
        for se in SEASONS:
            add("F1_sex_season", f"{en}_{se}", _and(f"sex=='{sx}'", f"season=='{se}'"))
        for mo in range(1, 13):
            add("F2_sex_month", f"{en}_m{mo:02d}", _and(f"sex=='{sx}'", f"month=={mo}"))
    # F3 夏の牝馬の細分
    base = ["sex=='牝'", "is_summer==1"]
    for ag, ae in (("a3", "age_grp=='a3'"), ("a4p", "age_grp=='a4p'")):
        add("F3_female_summer", f"female_summer_{ag}", _and(*base, ae))
    for sf, se in (("turf", "is_turf==1"), ("dirt", "is_turf==0")):
        add("F3_female_summer", f"female_summer_{sf}", _and(*base, se))
    for lc, le in (("local", "is_local==1"), ("central", "is_local==0")):
        add("F3_female_summer", f"female_summer_{lc}", _and(*base, le))
    for ag, ae in (("a3", "age_grp=='a3'"), ("a4p", "age_grp=='a4p'")):
        for sf, se in (("turf", "is_turf==1"), ("dirt", "is_turf==0")):
            for lc, le in (("local", "is_local==1"), ("central", "is_local==0")):
                add("F3_female_summer", f"female_summer_{ag}_{sf}_{lc}", _and(*base, ae, se, le))
    # F4 年齢 × 季節 × 休み / 上がり馬
    add("F4_age_season", "a3_summer_prev2_top3", _and("age==3", "is_summer==1", "prev_fin<=3", "prev2_fin<=3"))
    add("F4_age_season", "a3_summer_prev_win", _and("age==3", "is_summer==1", "prev_fin==1"))
    add("F4_age_season", "a4p_summer_rest90", _and("age>=4", "is_summer==1", "lay_days>=90"))
    add("F4_age_season", "a4p_summer_rest180", _and("age>=4", "is_summer==1", "lay_days>=180"))
    add("F4_age_season", "a3_summer_rest90", _and("age==3", "is_summer==1", "lay_days>=90"))
    add("F4_age_season", "female_a4p_summer_rest90", _and("sex=='牝'", "age>=4", "is_summer==1", "lay_days>=90"))
    # F5 種牡馬 × 季節 / クッション値（as-of の同条件超過複勝率）
    for th in (0.05, 0.10):
        add("F5_sire_cond", f"sire_summer_aff{th}", _and(f"sire_summer_aff>={th}", "is_summer==1"))
        add("F5_sire_cond", f"sire_winter_aff{th}", _and(f"sire_winter_aff>={th}", "season=='winter'"))
        add("F5_sire_cond", f"sire_hard_aff{th}", _and(f"sire_hard_aff>={th}", "cush_bin==2"))
        add("F5_sire_cond", f"sire_soft_aff{th}", _and(f"sire_soft_aff>={th}", "cush_bin==0"))
    # F6 馬ごとの同季節の実績
    for th in (0.25, 0.40):
        add("F6_horse_season", f"seas_aff{th}", f"seas_aff>={th}")
        for se in ("summer", "winter"):
            add("F6_horse_season", f"seas_aff{th}_{se}", _and(f"seas_aff>={th}", f"season=='{se}'"))
    # F7 寒い時期の馬体重 / 全季節の馬体重
    for k, e in WT_DW.items():
        if k != "dw_stable":
            add("F7_weight", f"winter_{k}", _and("season=='winter'", e))
            add("F7_weight", f"all_{k}", e)
    for k, e in WT_DEV.items():
        if k != "dev_stable":
            add("F7_weight", f"winter_{k}", _and("season=='winter'", e))
            add("F7_weight", f"all_{k}", e)
    # F9 騎手・調教師のローカル場実績
    for th in (0.05, 0.10):
        add("F9_local", f"t_local_aff{th}", _and(f"t_local_aff>={th}", "is_local==1"))
        add("F9_local", f"j_local_aff{th}", _and(f"j_local_aff>={th}", "is_local==1"))
        add("F9_local", f"t_local_aff{th}_summer", _and(f"t_local_aff>={th}", "is_local==1", "is_summer==1"))
        add("F9_local", f"j_local_aff{th}_summer", _and(f"j_local_aff>={th}", "is_local==1", "is_summer==1"))
    add("F9_local", "tj_local_aff0.05", _and("t_local_aff>=0.05", "j_local_aff>=0.05", "is_local==1"))
    # F10 開催の節目
    for nm, e in (("day1", "day_no==1"), ("first_week", "day_no<=2"), ("last_week", "day_no>=day_max-1")):
        add("F10_meeting", nm, e)
        add("F10_meeting", f"{nm}_turf", _and(e, "is_turf==1"))
        add("F10_meeting", f"{nm}_dirt", _and(e, "is_turf==0"))
    for wk in ("first_week", "last_week"):
        e = "day_no<=2" if wk == "first_week" else "day_no>=day_max-1"
        add("F10_meeting", f"{wk}_turf_inner", _and(e, "is_turf==1", "inner==1"))
        add("F10_meeting", f"{wk}_turf_outer", _and(e, "is_turf==1", "outer==1"))
        add("F10_meeting", f"{wk}_turf_front", _and(e, "is_turf==1", "style_grp=='front'"))
        add("F10_meeting", f"{wk}_turf_closer", _and(e, "is_turf==1", "style_grp=='closer'"))
    # F11 G1 開催の前後
    for nm, e in (("g1day_notg1", _and("is_g1_day==1", "~grade.astype('str').str.contains('G1|GI|Ｇ１')")), ("before_g1", "d_before_g1==1"), ("after_g1", "d_after_g1==1")):
        add("F11_g1", nm, e)
        add("F11_g1", f"{nm}_turf", _and(e, "is_turf==1"))
        add("F11_g1", f"{nm}_dirt", _and(e, "is_turf==0"))
    # F12 クッション値・含水率の深掘り
    draws = {"inner": "inner==1", "mid": "(inner==0)&(outer==0)", "outer": "outer==1"}
    for b in (0, 1, 2):
        for st in STYLES:
            for dn, de in draws.items():
                add("F12_cushion", f"turf_cush{b}_{st}_{dn}", _and(f"cush_bin=={b}", f"style_grp=='{st}'", de))
    for b in (0, 1, 2):
        for st in STYLES:
            for dn, de in draws.items():
                add("F12_cushion", f"dirt_moist{b}_{st}_{dn}", _and(f"dmoist_bin=={b}", f"style_grp=='{st}'", de))
    for b in (0, 1, 2):
        for st in STYLES:
            add("F12_cushion", f"turf_moist{b}_{st}", _and(f"tmoist_bin=={b}", f"style_grp=='{st}'"))
    for nm, e in (("cush_harder", "cush_delta_day>=1.0"), ("cush_softer", "cush_delta_day<=-1.0")):
        for st in ("front", "closer"):
            add("F12_cushion", f"{nm}_{st}", _and(e, "is_turf==1", f"style_grp=='{st}'"))
    for th in (0.15, 0.25):
        add("F12_cushion", f"pref_hard{th}_today_hard", _and(f"pref_hard>={th}", "cush_bin==2"))
        add("F12_cushion", f"pref_soft{th}_today_soft", _and(f"pref_hard<=-{th}", "cush_bin==0"))
    add("F12_cushion", "today_harder_than_last_by2", _and("is_turf==1", "cushion_value-prev_cush>=2"))
    add("F12_cushion", "today_softer_than_last_by2", _and("is_turf==1", "cushion_value-prev_cush<=-2"))
    # F13 展開（隊列）
    pace = {"lone_front": "(front==1)&(n_front==1)", "front_crowd_closer": "(closer==1)&(n_front>=3)", "front_crowd_mid": "(style_grp=='mid')&(n_front>=3)",
            "lone_front_bigfield": "(front==1)&(n_front<=1)&(fs>=12)", "closer_crowd_front": "(front==1)&(n_closer>=5)"}
    for nm, e in pace.items():
        add("F13_pace", nm, e)
        add("F13_pace", f"{nm}_turf", _and(e, "is_turf==1"))
        add("F13_pace", f"{nm}_dirt", _and(e, "is_turf==0"))
    # F14 前走の不利・人気と着順の乖離
    unl = {"prev_agari_top2_fin6p": "(prev_agari_rank<=2)&(prev_fin>=6)", "prev_agari1_fin4p": "(prev_agari_rank==1)&(prev_fin>=4)",
           "prev_late_agari3": "(prev_posL_frac>=0.7)&(prev_agari_rank<=3)", "prev_pop3_fin10p": "(prev_pop<=3)&(prev_fin>=10)",
           "prev_pop3_fin7p": "(prev_pop<=3)&(prev_fin>=7)", "prev_pop10p_top3": "(prev_pop>=10)&(prev_fin<=3)"}
    for nm, e in unl.items():
        add("F14_prev", nm, e)
        add("F14_prev", f"{nm}_turf", _and(e, "is_turf==1"))
        add("F14_prev", f"{nm}_dirt", _and(e, "is_turf==0"))
    # F15 初経験・距離・斤量・頭数・間隔
    add("F15_misc", "first_turf", _and("first_surface==1", "is_turf==1"))
    add("F15_misc", "first_dirt", _and("first_surface==1", "is_turf==0"))
    for nm, e in (("dist_up200", "dist_chg>=200"), ("dist_up400", "dist_chg>=400"), ("dist_dn200", "dist_chg<=-200"), ("dist_dn400", "dist_chg<=-400"),
                  ("jw_dn2", "jw_chg<=-2"), ("jw_up2", "jw_chg>=2"), ("jw_dn3", "jw_chg<=-3"), ("jw_up3", "jw_chg>=3"),
                  ("fs_le9", "fs<=9"), ("fs_ge16", "fs>=16"), ("fs_ge14_inner", "(fs>=14)&(inner==1)"), ("fs_ge14_outer", "(fs>=14)&(outer==1)"),
                  ("lay_le13", "(lay_days<=13)&(run_no>0)"), ("lay_14_27", "(lay_days>=14)&(lay_days<=27)&(run_no>0)"), ("lay_28_62", "(lay_days>=28)&(lay_days<=62)"),
                  ("lay_63_119", "(lay_days>=63)&(lay_days<=119)"), ("lay_120_365", "(lay_days>=120)&(lay_days<=365)"), ("lay_gt365", "lay_days>365"), ("debut", "run_no==0")):
        add("F15_misc", nm, e)
    # F16 逃げはさみ（隣の馬番の馬の前走位置取り）
    for d in ("d1", "d2", "d3"):
        nl = f"{d}_self_nl==1"
        for k in ("pin", "p1", "edge", "in1"):
            add("F16_pincer", f"{d}_{k}", _and(f"{d}_{k}==1", nl))
        add("F16_pincer", f"{d}_any", _and(f"{d}_cnt>=1", nl))
    for d in ("d1", "d2"):
        nl = f"{d}_self_nl==1"
        for k in ("pin", "p1"):
            b = _and(f"{d}_{k}==1", nl)
            for st in STYLES:
                add("F16_pincer", f"{d}_{k}_{st}", _and(b, f"style_grp=='{st}'"))
            add("F16_pincer", f"{d}_{k}_turf", _and(b, "is_turf==1"))
            add("F16_pincer", f"{d}_{k}_dirt", _and(b, "is_turf==0"))
            for gn, ge in DG.items():
                add("F16_pincer", f"{d}_{k}_{gn}", _and(b, ge))
            add("F16_pincer", f"{d}_{k}_bigfield", _and(b, "fs>=14"))
            for cb in (0, 1, 2):
                add("F16_pincer", f"{d}_{k}_cush{cb}", _and(b, f"cush_bin=={cb}"))
    # F17 季節 × 馬場 × 脚質
    for se in SEASONS:
        for sf, ss in (("turf", "is_turf==1"), ("dirt", "is_turf==0")):
            for st in ("front", "closer"):
                add("F17_season_style", f"{se}_{sf}_{st}", _and(f"season=='{se}'", ss, f"style_grp=='{st}'"))
    # F18 体重・追い切り・間隔の同時発生
    for rk, re_ in REST.items():
        for wk, we in WT_DW.items():
            add("F18_combo", f"{rk}_{wk}", _and(re_, we))
        for wk, we in WT_DEV.items():
            add("F18_combo", f"{rk}_{wk}", _and(re_, we))
        for ek, ee in EV.items():
            add("F18_combo", f"{rk}_{ek}", _and(re_, ee))
    for wk, we in {**WT_DW, **WT_DEV}.items():
        for ek in ("ev_good", "ev_up"):
            add("F18_combo", f"{wk}_{ek}", _and(we, EV[ek]))
    for rk, re_ in REST.items():
        for wk, we in {**WT_DW, **WT_DEV}.items():
            for ek in ("ev_good", "ev_up"):
                if rk == "rest90":
                    add("F18_combo", f"{rk}_{wk}_{ek}", _and(re_, we, EV[ek]))
    spring_summer3 = _and("age==3", "month>=3", "month<=8")
    for wk, we in (("dw_up6", "weight_change>=6"), ("dev_up6", "wt_dev>=6")):
        add("F18_combo", f"a3_spsm_{wk}", _and(spring_summer3, we))
        for ek in ("ev_good", "ev_up"):
            add("F18_combo", f"a3_spsm_{wk}_{ek}", _and(spring_summer3, we, EV[ek]))
    add("F18_combo", "comment_pos", "comment_pos==1")
    add("F18_combo", "comment_pos_rest90", _and("comment_pos==1", "lay_days>=90"))
    add("F18_combo", "comment_pos_ev_up", _and("comment_pos==1", "ev_delta>=1"))
    add("F18_combo", "comment_pos_dw_stable", _and("comment_pos==1", "abs(weight_change)<=4"))
    add("F18_combo", "debut_ev_good", _and("run_no==0", "ev_score>=3"))
    add("F18_combo", "debut_comment_pos", _and("run_no==0", "comment_pos==1"))
    seen = set()
    for x in c:
        assert x["name"] not in seen, x["name"]
        seen.add(x["name"])
    return c


if __name__ == "__main__":
    cs = build()
    OUT.write_text(json.dumps({"n_candidates": len(cs), "tests_per_candidate": 2, "n_tests": 2 * len(cs), "candidates": cs}, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter

    print(len(cs), "candidates;", 2 * len(cs), "tests")
    for k, v in sorted(Counter(x["family"] for x in cs).items()):
        print(f"  {k}: {v}")
