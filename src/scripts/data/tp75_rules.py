"""tp-75: ルールベースのペルソナ『期待値追求家』『複勝率追求家』。過去の分析で測った条件ごとの統計を台帳（ルール集合）に束ね、
馬ごとに『買い条件・消し条件への合致』で順位を付ける。モデルの学習はしない（集計と単純な規則）。

python -m src.scripts.data.tp75_rules prereg            # ルール集合（型）を数えて tp75_prereg.json に書く（評価の前に実行しコミットする）
python -m src.scripts.data.tp75_rules eval              # 台帳の検定・採用・クラスタ・ペルソナの評価（<TP75_OUT>/ledger.json, results.json）
python -m src.scripts.data.tp75_rules reeval FROZEN.json [--years 2026]   # 凍結規則を新しいデータで再評価し、失効を挙げる

市場情報（単勝・複勝オッズ、人気、市場確率、想定オッズ、期待値、投票比率）は、条件にも順位付けにも採用の判断にも使わない（`market_free()` で検査し、
ルールごとに印を持たせる）。確定オッズ・払戻は、回収率の測定（評価）にだけ使う。
基準は現行モデルの予測に対する残差（実績 − 予測）。モデルは v6e（LOYO 2022〜2024・test 2025）。2026 年の予測は v4（bundle_v4_2025）で、記述のみ。
発見期間は 2022〜2024（残差の基準になる OOF 予測がある期間）。2025 は選んだ規則集合の評価 1 回、2026 は記述統計のみ。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("TP75_OUT", "/home/jovyan/work/tp75_out"))
SRC72 = Path(os.environ.get("TP72_OUT", "/home/jovyan/work/tp72_out"))
MODELING = Path(os.environ.get("TP75_MODELING", "/home/jovyan/work/keiba-vpn/data/local/modeling"))
HERE = Path(__file__).parent
PREREG = HERE / "tp75_prereg.json"

DISC = (2022, 2023, 2024)
CONF = 2025
DESC = 2026
ZA, ZB = 1.959964, 1.644854           # 両側 95%（A）／両側 90%（B）の限界
MIN_RUNS, MIN_RACES, MIN_YEAR_RUNS = 300, 200, 100
FDR = 0.10
SHRINK_N0 = 300.0                      # 効果の縮約（n/(n+n0)）
OVERLAP_JAC, OVERLAP_CONT = 0.5, 0.9

MARKET_TOKENS = re.compile(r"popularity|prev_pop|odds|ret_win|ret_place|ret_|m_win|m_top3|q_win|place_eff|\bev_win\b|\bpop\b|ninki|tansho|fukusho")


def market_free(expr: str) -> bool:
    """条件の式が市場由来の列（オッズ・人気・市場確率・払戻）を含まないか。"""
    return MARKET_TOKENS.search(expr) is None


def phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ====================================================================================== ルール集合（型）
def _and(*xs: str) -> str:
    return " & ".join(f"({x})" for x in xs)


DG = {"sprint": "(dist_band=='d_sprint')", "mile": "((dist_band=='d_mile_s')|(dist_band=='d_mile'))", "mid_long": "((dist_band=='d_mid')|(dist_band=='d_long'))"}
SF = {"turf": "(is_turf==1)", "dirt": "(is_turf==0)"}

HEADLINE = {
    "geld_n1": "geld_n==1", "geld_n2": "geld_n==2", "first_foal": "dam_first_foal==1", "dam_age_ge15": "dam_age_at_foaling>=15",
    "jsw_up": "jchg_cls=='up'", "jsw_down": "jchg_cls=='down'", "blk_first": "blk_state=='first'", "lay_ge180": "lay_days>=180",
    "lay_90_179": _and("lay_days>=90", "lay_days<180"), "wt_up10": "weight_change>=10", "wt_down10": "weight_change<=-10",
    "female_summer": _and("sex=='牝'", "is_summer==1"), "style_front": "front==1", "style_closer": "closer==1",
    "first_surface": "first_surface==1", "dist_ext400": "dist_chg>=400",
}


def library() -> list[dict]:
    """ルールの型を、結果を見る前に固定して返す。各要素 = {id, family, name, expr, source, from_tp72}。"""
    rules: list[dict] = []

    def add(fam: str, name: str, expr: str, src: str) -> None:
        rules.append({"id": f"{fam}/{name}", "family": fam, "name": name, "expr": expr, "source": src, "from_tp72": False})

    # T72: 事前登録の 881 型のうち、市場由来の条件（前走の人気: F14 の 9 型）を除く
    spec = json.loads((HERE / "tp72_candidates.json").read_text(encoding="utf-8"))
    excluded: list[dict] = []
    for c in spec["candidates"]:
        if market_free(c["expr"]):
            rules.append({"id": f"T72/{c['name']}", "family": "T72_" + c["family"], "name": c["name"], "expr": c["expr"], "source": "edge_factor_mining.html（tp-72 " + c["family"] + "）", "from_tp72": True})
        else:
            excluded.append({"id": f"T72/{c['name']}", "expr": c["expr"], "reason": "前走の人気（市場由来）を条件に含む"})

    # G 去勢明け N 戦目（geld_after_races / tp-32 GELD）
    src = "geld_after_races.html"
    add("G", "n1", "geld_n==1", src); add("G", "n2", "geld_n==2", src); add("G", "n3", "geld_n==3", src)
    add("G", "n4_5", _and("geld_n>=4", "geld_n<=5"), src); add("G", "n6p", "geld_n>=6", src); add("G", "any", "geld_n>=1", src)
    for n in (1, 2):
        for sk, se in SF.items():
            add("G", f"n{n}_{sk}", _and(f"geld_n=={n}", se), src)
        for dk, de in DG.items():
            add("G", f"n{n}_{dk}", _and(f"geld_n=={n}", de), src)
    # D 母の出産時年齢・初仔・種牡馬の供用年（dam_age_firstfoal）
    src = "dam_age_firstfoal.html"
    for k, e in {"age_le6": "dam_age_at_foaling<=6", "age_7_11": _and("dam_age_at_foaling>=7", "dam_age_at_foaling<=11"), "age_12_14": _and("dam_age_at_foaling>=12", "dam_age_at_foaling<=14"),
                 "age_15_17": _and("dam_age_at_foaling>=15", "dam_age_at_foaling<=17"), "age_ge18": "dam_age_at_foaling>=18", "age_ge15": "dam_age_at_foaling>=15",
                 "first_foal": "dam_first_foal==1", "first_foal_certain": _and("dam_first_foal==1", "dam_first_foal_conf=='確実'"),
                 "not_first_certain": _and("dam_first_foal==0", "dam_first_foal_conf=='確実'"), "first_foal_age_le6": _and("dam_first_foal==1", "dam_age_at_foaling<=6")}.items():
        add("D", k, e, src)
    for ak, ae in {"a2": "age==2", "a3": "age==3", "a4p": "age>=4"}.items():
        add("D", f"age_ge15_{ak}", _and("dam_age_at_foaling>=15", ae), src)
    for ak, ae in {"a2": "age==2", "a3": "age==3"}.items():
        add("D", f"first_foal_{ak}", _and("dam_first_foal==1", ae), src)
    for g in ("1", "2", "3", "4+", "既存"):
        add("D", f"sire_year_{g}", f"sire_year_group=='{g}'", src)
    add("D", "sire_year_1_a2", _and("sire_year_group=='1'", "age==2"), src)
    # J 乗り替わりの型（jockey_switch_ev）
    src = "jockey_switch_ev.html"
    for k in ("up", "down", "top_to_top", "non_to_non", "none"):
        add("J", k, f"jchg_cls=='{k}'", src)
    for k in ("up", "down"):
        add("J", f"{k}_prev_top3", _and(f"jchg_cls=='{k}'", "prev_fin<=3"), src)
        add("J", f"{k}_prev_4p", _and(f"jchg_cls=='{k}'", "prev_fin>=4"), src)
        for dk, de in DG.items():
            add("J", f"{k}_{dk}", _and(f"jchg_cls=='{k}'", de), src)
        for sk, se in SF.items():
            add("J", f"{k}_{sk}", _and(f"jchg_cls=='{k}'", se), src)
    # R 休養・間隔（rest_and_interval）
    src = "rest_and_interval.html"
    bands = {"lt14": "lay_days<14", "14_27": _and("lay_days>=14", "lay_days<28"), "28_55": _and("lay_days>=28", "lay_days<56"), "56_89": _and("lay_days>=56", "lay_days<90"),
             "90_179": _and("lay_days>=90", "lay_days<180"), "180_364": _and("lay_days>=180", "lay_days<365"), "ge365": "lay_days>=365"}
    for k, e in bands.items():
        add("R", k, e, src)
        add("R", f"{k}_prev_top3", _and(e, "prev_fin<=3"), src)
        add("R", f"{k}_prev_6p", _and(e, "prev_fin>=6"), src)
    for k, e in (("up10", "weight_change>=10"), ("down10", "weight_change<=-10")):
        add("R", f"ge90_wt_{k}", _and("lay_days>=90", e), src)
        add("R", f"ge180_wt_{k}", _and("lay_days>=180", e), src)
    # E 馬具・替わり（equipment_and_change / tp-32）
    src = "equipment_and_change.html"
    for k in ("first", "again", "off", "cont"):
        add("E", f"blk_{k}", f"blk_state=='{k}'", src)
    for k, e in {"ext200": "dist_chg>=200", "ext400": "dist_chg>=400", "short200": "dist_chg<=-200", "short400": "dist_chg<=-400"}.items():
        add("E", f"dist_{k}", e, src)
    add("E", "first_surface", "first_surface==1", src)
    add("E", "turf_to_dirt", _and("prev_surface=='芝'", "is_turf==0"), src)
    add("E", "dirt_to_turf", _and("prev_surface=='ダート'", "is_turf==1"), src)
    add("E", "jw_up2", "jw_chg>=2", src); add("E", "jw_down2", "jw_chg<=-2", src); add("E", "chg_trainer", "chg_trainer==1", src)
    for sk, se in SF.items():
        add("E", f"blk_first_{sk}", _and("blk_state=='first'", se), src)
    # C コース（course_specific_effects）: 競馬場×面、距離帯×面、クラス×面
    src = "course_specific_effects.html"
    for v in range(1, 11):
        for sk, se in SF.items():
            add("C", f"venue{v:02d}_{sk}", _and(f"venue_code=='{v:02d}'", se), src)
    for db in ("d_sprint", "d_mile_s", "d_mile", "d_mid", "d_long"):
        for sk, se in SF.items():
            add("C", f"{db}_{sk}", _and(f"dist_band=='{db}'", se), src)
    for cl in ("c0_newcomer", "c1_maiden", "c2_1win", "c3_2win", "c4_3win", "op", "g3", "g2"):
        for sk, se in SF.items():
            add("C", f"{cl}_{sk}", _and(f"cls=='{cl}'", se), src)
    # B9 脚質・位置取り・前走の上がり（position_pace_course / pace 系の近い条件）
    src = "position_pace_course.html"
    for st in ("nige", "senko", "sashi", "oikomi"):
        for sk, se in SF.items():
            add("P", f"{st}_{sk}", _and(f"style4=='{st}'", se), src)
        for fk, fe in {"fs_le11": "fs<=11", "fs_12_15": _and("fs>=12", "fs<=15"), "fs_ge16": "fs>=16"}.items():
            add("P", f"{st}_{fk}", _and(f"style4=='{st}'", fe), src)
    add("P", "agari1_prev6p", _and("prev_agari_rank==1", "prev_fin>=6"), src)
    add("P", "agari_le2_prev6p", _and("prev_agari_rank<=2", "prev_fin>=6"), src)
    add("P", "agari1_prev_top3", _and("prev_agari_rank==1", "prev_fin<=3"), src)
    # H 見出しの条件 × 面（2 条件の組合せ）
    for hk, he in HEADLINE.items():
        for sk, se in SF.items():
            add("H", f"{hk}_{sk}", _and(he, se), "上記各章の見出し条件 × 面")

    for r in rules:
        r["market_free"] = market_free(r["expr"])
    library.excluded = excluded  # type: ignore[attr-defined]
    return rules


def prereg() -> dict:
    rules = library()
    fam: dict[str, int] = {}
    for r in rules:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    n_new = sum(1 for r in rules if not r["from_tp72"])
    spec = {
        "version": 1,
        "design": {
            "purpose": "ルールベースのペルソナ（期待値追求家・複勝率追求家）。分析で測った条件を台帳にし、買い条件・消し条件への合致で馬を順位付けする。モデルの学習はしない。",
            "market_free": "条件・順位付け・採用の判断に、単勝・複勝オッズ、人気、市場確率、想定オッズ、期待値、投票比率を使わない。確定払戻は回収率の測定（評価）にだけ使う。",
            "base_model": "v6e（LOYO OOF 2022〜2024、test 2025）。2026 は v4（bundle_v4_2025）の予測で記述のみ。",
            "discovery_years": list(DISC), "confirm_year": CONF, "descriptive_year": DESC,
            "note_discovery": "発見期間は 2022〜2024。残差の基準になる OOF 予測が 2022 年からしか無いため（2020〜21 は予測が無い）。",
            "testable": {"min_runs": MIN_RUNS, "min_races": MIN_RACES, "min_year_runs_for_replication": MIN_YEAR_RUNS},
            "race_level_rules_excluded": "出走馬の 90% 以上が同じレースの全頭に当たる条件（レース単位）は馬を順位付けできないので採用しない",
            "personas": {
                "ev_seeker": {"meaning": "過去統計から見て、モデルが勝ちを過小評価している条件に該当する馬を選ぶ。回収率はその結果を測る指標（採用の基準ではない）。", "buy": "勝ちの残差（実績−予測）の下側信頼限界>0", "kill": "勝ちの残差の上側信頼限界<0"},
                "place_seeker": {"meaning": "3着以内に入りやすい条件（複勝率がモデルより高い）に該当する馬を選ぶ。", "buy": "3着以内の残差の下側信頼限界>0", "kill": "3着以内の残差の上側信頼限界<0"},
            },
            "tiers": {"A": "両側 95% の信頼区間が 0 を越えて正（負）", "B": "両側 90%（暫定）"},
            "replication": "発見期間の 3 年のうち、n≥100 の年が 2 年以上あり、それらの年で残差の符号が同じ（過半が採用の向き）",
            "robust": "BH（FDR 10%）。族 = ev_buy / ev_kill / place_buy / place_kill、それぞれ検定可能なルール全体",
            "clusters": "発見期間の該当行の Jaccard≥0.5 または包含率≥0.9 のルールは同じクラスタ（代表 = z が最大）。1 頭が数えるのはクラスタごとに 1 回",
            "scores": {"a": "買いクラスタの合致数 − 消しクラスタの合致数", "b": "縮約した効果（pt, n/(n+300)）の合計（買い−消し）", "c": "最強の買い根拠 − 最強の消し根拠（縮約した効果）"},
            "kill_handling": ["exclude（消しに当たる馬は選ばない）", "penalize（スコアから減点）"],
            "tests": {"statistics_per_rule": 2, "directions": 2, "tests_per_rule": 4, "note": "e_win と e_top3 の 2 統計 × 買い・消しの 2 方向"},
        },
        "counts": {"n_rules": len(rules), "n_rules_tp72": len(rules) - n_new, "n_rules_new": n_new, "n_tests": 4 * len(rules), "by_family": fam},
        "excluded_market": library.excluded,  # type: ignore[attr-defined]
        "rules": [{k: r[k] for k in ("id", "family", "name", "expr", "source", "market_free")} for r in rules],
    }
    PREREG.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    return spec


# ====================================================================================== 表の作成
def geld_n(d: pd.DataFrame) -> pd.Series:
    """去勢明け N 戦目（牡→セ の最初の出走を 1）。最初からセの馬・去勢前の履歴が表に無い馬は欠損。"""
    s = d.sort_values(["horse_id", "date", "race_id"])
    h = s["horse_id"]
    male = (s["sex"] == "牡").astype(int)
    male_before = male.groupby(h).cummax().groupby(h).shift(1).fillna(0).astype(int)
    gel = (s["sex"] == "セ") & (male_before == 1)
    started = gel.astype(int).groupby(h).cummax()
    on = started == 1
    out = pd.Series(np.nan, index=s.index)
    out[on] = on.astype(int).groupby(h).cumsum()[on]       # 開始以降の走を 1 から数える（セ→牡 の戻りは無い）
    return out.reindex(d.index)


def load_table(models: Path | None = None) -> pd.DataFrame:
    """共通の表（tp72 の mining.parquet）に、去勢明け・母の年齢・初仔・供用年と、v6e の予測を結合する。"""
    models = models or MODELING
    d = pd.read_parquet(SRC72 / "mining.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d["geld_n"] = geld_n(d)
    fl = pd.read_parquet(models / "shared_cache/foal_lineage/foal_lineage.parquet", columns=["horse_id", "dam_age_at_foaling", "dam_first_foal", "dam_first_foal_conf", "sire_year_group"])
    d = d.merge(fl.drop_duplicates("horse_id"), on="horse_id", how="left")
    d = d[(d["fs"] >= 8) & d["finish_position"].notna() & (d["win_odds"] > 0)].copy()      # win_odds は払戻の測定のための行の絞り込み（条件には使わない）
    # v6e の予測（2022〜24: LOYO OOF、2025: bundle_v6_2024 の test 予測）。2026 は v4（bundle_v4_2025）
    oof = pd.read_parquet(models / "training_pipeline_v4/tp27_v6e/ens/v6.parquet", columns=["race_id", "horse_id", "p_win", "p_top3"])
    t25 = pd.read_parquet(models / "training_pipeline/final_eval_v6e/test2025_predictions.parquet", columns=["race_id", "horse_id", "p_win", "p_top3"])
    pv6 = pd.concat([oof, t25], ignore_index=True).rename(columns={"p_win": "pw6", "p_top3": "pt6"})
    d = d.rename(columns={"p_win": "pw4", "p_top3": "pt4"}).merge(pv6, on=["race_id", "horse_id"], how="left")
    d["p_win"] = np.where(d["year"] == DESC, d["pw4"], d["pw6"])
    d["p_top3"] = np.where(d["year"] == DESC, d["pt4"], d["pt6"])
    d = d[d["p_win"].notna() & d["p_top3"].notna() & (d["year"] >= 2022)].copy()
    d = d.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    d["y1"] = d["win"].astype(float)
    d["y3"] = d["top3"].astype(float)
    d["r1"] = d["ret_win"].astype(float) / 100.0          # 回収率の測定のみ
    d["r3"] = d["ret_place"].astype(float) / 100.0
    d["e_m1"] = d["y1"] - d["p_win"]
    d["e_m3"] = d["y3"] - d["p_top3"]
    d["e_v1"] = d["y1"] - d["pw4"]                         # 参考: v4 に対する残差
    d["e_v3"] = d["y3"] - d["pt4"]
    d["rc"] = pd.factorize(d["race_id"])[0]
    return d


# ====================================================================================== 検定
class Ctx:
    def __init__(self, d: pd.DataFrame) -> None:
        self.d = d
        self.rc = d["rc"].to_numpy()
        self.nrace = int(self.rc.max()) + 1
        self.year = d["year"].to_numpy()
        self.field = np.bincount(self.rc, minlength=self.nrace)
        self.cols = {c: d[c].to_numpy(dtype=float) for c in ("e_m1", "e_m3", "e_v1", "e_v3", "r1", "r3", "y1", "y3")}
        self.periods = {"disc": np.isin(self.year, DISC), "y2022": self.year == 2022, "y2023": self.year == 2023, "y2024": self.year == 2024, "y2025": self.year == CONF, "y2026": self.year == DESC}


def cstat(ctx: Ctx, mask: np.ndarray, col: str) -> dict:
    """レース単位のクラスタ頑健な平均・標準誤差・z（H0: 平均 0）。"""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "races": 0, "mean": None, "se": None, "z": None}
    rc = ctx.rc[mask]
    v = ctx.cols[col][mask]
    R = np.bincount(rc, weights=v, minlength=ctx.nrace)
    C = np.bincount(rc, minlength=ctx.nrace)
    mean = float(R.sum() / n)
    se = float(math.sqrt(float(((R - mean * C) ** 2).sum())) / n)
    return {"n": n, "races": int((C > 0).sum()), "mean": mean, "se": se, "z": (mean / se) if se > 0 else None}


def boot_mean(ctx: Ctx, mask: np.ndarray, col: str, reps: int = 1000, seed: int = 0) -> list[float | None]:
    """レース単位ブートストラップ（平均, 2.5%, 97.5%）。"""
    if mask.sum() == 0:
        return [None, None, None]
    rc = ctx.rc[mask]
    v = ctx.cols[col][mask]
    s = np.bincount(rc, weights=v, minlength=ctx.nrace)
    c = np.bincount(rc, minlength=ctx.nrace).astype(float)
    keep = c > 0
    s, c = s[keep], c[keep]
    rng = np.random.default_rng(seed)
    n = len(s)
    vals = np.empty(reps)
    for k in range(reps):
        i = rng.integers(0, n, n)
        vals[k] = s[i].sum() / c[i].sum()
    return [float(s.sum() / c.sum()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def rule_mask(d: pd.DataFrame, expr: str) -> np.ndarray:
    return d.eval(expr, engine="python").fillna(False).to_numpy().astype(bool)


def evaluate_rules(d: pd.DataFrame, rules: list[dict]) -> list[dict]:
    ctx = Ctx(d)
    rows = []
    for r in rules:
        try:
            mask = rule_mask(d, r["expr"])
            err = None
        except Exception as exc:  # 列が無い等。検定不能として数に含める
            mask = np.zeros(len(d), bool)
            err = str(exc)[:120]
        dm = mask & ctx.periods["disc"]
        full = 0.0
        if dm.sum():
            nsel = np.bincount(ctx.rc[dm], minlength=ctx.nrace)
            full = float((nsel[ctx.rc[dm]] == ctx.field[ctx.rc[dm]]).mean())
        st: dict = {}
        for tag, sel in ctx.periods.items():
            mm = mask & sel
            blk = {"e_m1": cstat(ctx, mm, "e_m1"), "e_m3": cstat(ctx, mm, "e_m3")}
            if tag in ("disc", "y2025", "y2026"):
                blk["roi_win"] = float(ctx.cols["r1"][mm].mean()) if mm.any() else None
                blk["roi_place"] = float(ctx.cols["r3"][mm].mean()) if mm.any() else None
                blk["hit_top3"] = float(ctx.cols["y3"][mm].mean()) if mm.any() else None
                blk["hit_win"] = float(ctx.cols["y1"][mm].mean()) if mm.any() else None
            st[tag] = blk
        rows.append({**r, "error": err, "race_level": bool(full >= 0.9), "full_race_frac": full, "stats": st})
    return rows


def _testable(r: dict) -> bool:
    s = r["stats"]["disc"]
    return r["error"] is None and not r["race_level"] and s["e_m1"]["n"] >= MIN_RUNS and s["e_m1"]["races"] >= MIN_RACES


def _years_ok(r: dict, col: str, sign: int) -> bool:
    """発見期間の年別の符号（n≥100 の年が 2 年以上、そのうち過半が採用の向き）。"""
    ok = []
    for y in ("y2022", "y2023", "y2024"):
        b = r["stats"][y][col]
        if b["n"] >= MIN_YEAR_RUNS and b["mean"] is not None:
            ok.append(sign * b["mean"] > 0)
    return len(ok) >= 2 and sum(ok) * 2 > len(ok)


FAMILIES = {            # 族 → (勝ち/複勝の列, 向き)
    "ev_buy": ("e_m1", +1), "ev_kill": ("e_m1", -1),
    "place_buy": ("e_m3", +1), "place_kill": ("e_m3", -1),
}


def adopt(rows: list[dict]) -> dict:
    """採用の区分（A/B）、再現、BH の頑健性を付ける。"""
    for r in rows:
        r["adopt"] = {}
    summary: dict = {}
    for fam, (col, sign) in FAMILIES.items():
        idx = [i for i, r in enumerate(rows) if _testable(r)]
        z = np.array([sign * (rows[i]["stats"]["disc"][col]["z"] or 0.0) for i in idx])
        p = np.array([1 - phi(v) for v in z])
        q = bh(p) if len(idx) else np.array([])
        n_a = n_b = n_rob = 0
        for k, i in enumerate(idx):
            r = rows[i]
            tier = "A" if z[k] > ZA else "B" if z[k] > ZB else None
            rep = _years_ok(r, col, sign)
            robust = bool(q[k] < FDR)
            r["adopt"][fam] = {"z": float(z[k]), "p": float(p[k]), "q": float(q[k]), "tier": tier, "replicated": rep, "robust_bh": robust,
                               "adopted": bool(tier is not None and rep), "effect": rows[i]["stats"]["disc"][col]["mean"]}
            if tier and rep:
                n_a += tier == "A"; n_b += tier == "B"; n_rob += robust
        summary[fam] = {"testable": len(idx), "adopted_A": int(n_a), "adopted_B": int(n_b), "adopted_robust_bh": int(n_rob),
                        "z_gt_1.645_before_replication": int((z > ZB).sum()), "expected_false_pos_at_1.645": float(0.05 * len(idx))}
    return summary


def cluster_set(d: pd.DataFrame, rows: list[dict], fam: str) -> list[dict]:
    """採用されたルールを、発見期間の該当行の重なりでクラスタにまとめる。代表 = z 最大。"""
    ctx_disc = np.isin(d["year"].to_numpy(), DISC)
    ad = [r for r in rows if r["adopt"].get(fam, {}).get("adopted")]
    ad.sort(key=lambda r: -r["adopt"][fam]["z"])
    clusters: list[dict] = []
    for r in ad:
        m = rule_mask(d, r["expr"]) & ctx_disc
        placed = False
        for c in clusters:
            inter = int((m & c["rep_mask"]).sum())
            if inter == 0:
                continue
            a, b = int(m.sum()), int(c["rep_mask"].sum())
            if inter / (a + b - inter) >= OVERLAP_JAC or inter / min(a, b) >= OVERLAP_CONT:
                c["members"].append(r["id"]); c["union"] |= rule_mask(d, r["expr"]); placed = True
                break
        if not placed:
            clusters.append({"rep": r["id"], "rep_row": r, "rep_mask": m, "members": [r["id"]], "union": rule_mask(d, r["expr"])})
    return clusters


def cluster_weight(c: dict, fam: str) -> float:
    r = c["rep_row"]
    col, sign = FAMILIES[fam]
    b = r["stats"]["disc"][col]
    eff = sign * b["mean"] * b["n"] / (b["n"] + SHRINK_N0)
    return float(max(eff, 0.0))


def score_persona(d: pd.DataFrame, buy: list[dict], kill: list[dict], fam_buy: str, fam_kill: str) -> dict[str, np.ndarray]:
    nb = np.zeros(len(d)); nk = np.zeros(len(d))
    wb = np.zeros(len(d)); wk = np.zeros(len(d))
    mb = np.zeros(len(d)); mk = np.zeros(len(d))
    kill_hit = np.zeros(len(d), bool)
    for c in buy:
        w = cluster_weight(c, fam_buy); u = c["union"]
        nb += u; wb += w * u; mb = np.maximum(mb, w * u)
    for c in kill:
        w = cluster_weight(c, fam_kill); u = c["union"]
        nk += u; wk += w * u; mk = np.maximum(mk, w * u); kill_hit |= u
    return {"a": nb - nk, "b": wb - wk, "c": mb - mk, "kill": kill_hit, "n_buy": nb, "n_kill": nk}


# ====================================================================================== ペルソナの評価
def _pick_groups(d: pd.DataFrame, score: np.ndarray, mask_year: np.ndarray) -> np.ndarray:
    """レースごとに、スコア最大（>0）の馬の集合（同点は全員）。返り値: bool 配列。"""
    s = pd.Series(np.where(mask_year, score, -np.inf), index=d.index)
    mx = s.groupby(d["rc"]).transform("max")
    return ((s == mx) & (mx > 0)).to_numpy()


def eval_picks(ctx: Ctx, pick: np.ndarray, label: str) -> dict:
    """選んだ馬の成績（レース単位の集計・ブートストラップ区間）。各馬 1 単位を賭けた場合の回収率。"""
    d = ctx.d
    out = {"label": label, "n_picks": int(pick.sum()), "n_races": int(len(np.unique(ctx.rc[pick]))) if pick.any() else 0}
    if not pick.any():
        return out
    for tag, col in (("win_rate", "y1"), ("top3_rate", "y3"), ("roi_win", "r1"), ("roi_place", "r3")):
        out[tag] = boot_mean(ctx, pick, col)
    pm = d["p_win"].to_numpy()[pick]; pt = d["p_top3"].to_numpy()[pick]
    out["model_p_win_mean"] = float(pm.mean()); out["model_p_top3_mean"] = float(pt.mean())
    out["resid_win"] = cstat(ctx, pick, "e_m1"); out["resid_top3"] = cstat(ctx, pick, "e_m3")
    return out


def baseline_picks(d: pd.DataFrame, col: str, mask_year: np.ndarray, races: np.ndarray | None = None) -> np.ndarray:
    s = pd.Series(np.where(mask_year, d[col].to_numpy(), -np.inf), index=d.index)
    mx = s.groupby(d["rc"]).transform("max")
    pick = ((s == mx) & np.isfinite(mx)).to_numpy()
    if races is not None:
        pick &= np.isin(d["rc"].to_numpy(), races)
    return pick


def run_eval(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_table()
    rules = library()
    rows = evaluate_rules(d, rules)
    asummary = adopt(rows)
    ctx = Ctx(d)
    clusters = {fam: cluster_set(d, rows, fam) for fam in FAMILIES}
    personas = {"ev_seeker": ("ev_buy", "ev_kill"), "place_seeker": ("place_buy", "place_kill")}
    scores = {p: score_persona(d, clusters[b], clusters[k], b, k) for p, (b, k) in personas.items()}
    periods = {"disc": ctx.periods["disc"], "y2025": ctx.periods["y2025"], "y2026": ctx.periods["y2026"]}
    res: dict = {"adopt_summary": asummary, "clusters": {}, "personas": {}}
    for fam, cl in clusters.items():
        res["clusters"][fam] = [{"rep": c["rep"], "n_members": len(c["members"]), "members": c["members"], "weight": cluster_weight(c, fam),
                                 "expr": c["rep_row"]["expr"], "source": c["rep_row"]["source"]} for c in cl]
    for p, (fb, fk) in personas.items():
        sc = scores[p]
        pr: dict = {}
        for dn in ("a", "b", "c"):
            for pn, my in periods.items():
                pick = _pick_groups(d, sc[dn], my)
                pr[f"def_{dn}/{pn}/top_group"] = eval_picks(ctx, pick, f"{p} def_{dn} {pn}")
                # 同じレースでのモデル本命（勝率最大）との比較
                races = np.unique(ctx.rc[pick]) if pick.any() else np.array([], int)
                bp = baseline_picks(d, "p_win", my, races)
                pr[f"def_{dn}/{pn}/model_top1_same_races"] = eval_picks(ctx, bp, f"model top1 on same races {pn}")
        # 消し条件の扱い: モデル本命（勝率最大）から、消しに当たる馬を除く
        for pn, my in periods.items():
            base = baseline_picks(d, "p_win", my)
            kill = sc["kill"]
            allow = my & ~kill
            filt = baseline_picks(d, "p_win", allow)
            changed_races = np.unique(ctx.rc[base & kill]) if (base & kill).any() else np.array([], int)
            pr[f"kill_filter/{pn}/model_top1_all"] = eval_picks(ctx, base, "model top1 all")
            pr[f"kill_filter/{pn}/model_top1_excluding_kill"] = eval_picks(ctx, filt, "model top1 excluding kill")
            pr[f"kill_filter/{pn}/changed_races_before"] = eval_picks(ctx, base & np.isin(ctx.rc, changed_races), "model top1 where top1 is a kill (before)")
            pr[f"kill_filter/{pn}/changed_races_after"] = eval_picks(ctx, filt & np.isin(ctx.rc, changed_races), "replacement pick (after)")
            # 複勝確率の上位 3 頭のうち消しに当たる馬 / 当たらない馬
            rk3 = d.groupby("rc")["p_top3"].rank(ascending=False, method="first").to_numpy() <= 3
            pr[f"kill_filter/{pn}/top3_by_ptop3_kill"] = eval_picks(ctx, rk3 & kill & my, "p_top3 上位3頭のうち消しに当たる")
            pr[f"kill_filter/{pn}/top3_by_ptop3_other"] = eval_picks(ctx, rk3 & ~kill & my, "p_top3 上位3頭のうち消しに当たらない")
        # スコアの帯別の残差
        bins = {"neg": sc["b"] < 0, "zero": sc["b"] == 0}
        pos = sc["b"] > 0
        if pos.any():
            thr = float(np.median(sc["b"][pos]))
            bins["pos_low"] = pos & (sc["b"] <= thr); bins["pos_high"] = pos & (sc["b"] > thr)
        for pn, my in periods.items():
            pr[f"bins/{pn}"] = {k: {"n": int((m & my).sum()), "resid_win": cstat(ctx, m & my, "e_m1"), "resid_top3": cstat(ctx, m & my, "e_m3"),
                                    "roi_win": boot_mean(ctx, m & my, "r1"), "roi_place": boot_mean(ctx, m & my, "r3"), "top3_rate": boot_mean(ctx, m & my, "y3")} for k, m in bins.items()}
        res["personas"][p] = pr
    # 2 つのペルソナの一致・食い違い（スコア b の符号）
    cross: dict = {}
    sev, spl = scores["ev_seeker"]["b"], scores["place_seeker"]["b"]
    quads = {"both_pos": (sev > 0) & (spl > 0), "ev_only": (sev > 0) & (spl <= 0), "place_only": (sev <= 0) & (spl > 0), "neither": (sev <= 0) & (spl <= 0)}
    rk3 = d.groupby("rc")["p_top3"].rank(ascending=False, method="first").to_numpy() <= 3
    for pn, my in periods.items():
        cross[pn] = {k: {"n": int((m & my).sum()), "top3_rate": boot_mean(ctx, m & my, "y3"), "win_rate": boot_mean(ctx, m & my, "y1"), "roi_win": boot_mean(ctx, m & my, "r1"),
                         "roi_place": boot_mean(ctx, m & my, "r3"), "resid_win": cstat(ctx, m & my, "e_m1"), "resid_top3": cstat(ctx, m & my, "e_m3")} for k, m in quads.items()}
        cross[pn]["model_top3_both_pos"] = {"n": int((rk3 & quads["both_pos"] & my).sum()), "top3_rate": boot_mean(ctx, rk3 & quads["both_pos"] & my, "y3"), "roi_place": boot_mean(ctx, rk3 & quads["both_pos"] & my, "r3")}
        cross[pn]["model_top3_not_both"] = {"n": int((rk3 & ~quads["both_pos"] & my).sum()), "top3_rate": boot_mean(ctx, rk3 & ~quads["both_pos"] & my, "y3"), "roi_place": boot_mean(ctx, rk3 & ~quads["both_pos"] & my, "r3")}
    res["cross"] = cross
    res["table"] = {"rows": int(len(d)), "races": int(ctx.nrace), "by_year": {int(k): int(v) for k, v in d["year"].value_counts().sort_index().items()}}
    # 台帳（人が見直せる形）。参考として v4 に対する残差と、回収率（測定）を併記
    ledger = []
    for r in rows:
        s = r["stats"]
        rec = {"id": r["id"], "family": r["family"], "name": r["name"], "expr": r["expr"], "source": r["source"], "market_free": r["market_free"],
               "race_level": r["race_level"], "error": r["error"],
               "n_disc": s["disc"]["e_m1"]["n"], "races_disc": s["disc"]["e_m1"]["races"],
               "win_resid_pt": None if s["disc"]["e_m1"]["mean"] is None else 100 * s["disc"]["e_m1"]["mean"], "win_resid_z": s["disc"]["e_m1"]["z"],
               "top3_resid_pt": None if s["disc"]["e_m3"]["mean"] is None else 100 * s["disc"]["e_m3"]["mean"], "top3_resid_z": s["disc"]["e_m3"]["z"],
               "roi_win_disc": s["disc"].get("roi_win"), "roi_place_disc": s["disc"].get("roi_place"),
               "z_2025_win": s["y2025"]["e_m1"]["z"], "z_2025_top3": s["y2025"]["e_m3"]["z"], "n_2025": s["y2025"]["e_m1"]["n"],
               "z_2026_win": s["y2026"]["e_m1"]["z"], "z_2026_top3": s["y2026"]["e_m3"]["z"], "n_2026": s["y2026"]["e_m1"]["n"],
               "adopt": r["adopt"]}
        ledger.append(rec)
    (out_dir / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in rec.items() if k != "adopt"} | {f"{f}_tier": rec["adopt"].get(f, {}).get("tier") for f in FAMILIES}
                  | {f"{f}_adopted": rec["adopt"].get(f, {}).get("adopted") for f in FAMILIES} | {f"{f}_q": rec["adopt"].get(f, {}).get("q") for f in FAMILIES}
                  for rec in ledger]).to_csv(out_dir / "ledger.csv", index=False)
    return res


def reeval(frozen: Path, years: list[int]) -> dict:
    """凍結規則を新しいデータ（指定年）で再評価し、失効（区間が 0 を跨ぐ・符号が反転）を挙げる。"""
    spec = json.loads(Path(frozen).read_text(encoding="utf-8"))
    d = load_table()
    ctx = Ctx(d)
    sel = np.isin(ctx.year, years)
    out = []
    for r in spec["rules"]:
        col, sign = ("e_m1" if r["outcome"] == "win" else "e_m3"), (+1 if r["direction"] == "buy" else -1)
        m = rule_mask(d, r["expr"]) & sel
        b = cstat(ctx, m, col)
        z = None if b["z"] is None else sign * b["z"]
        lapsed = bool(z is None or z < ZB)
        out.append({"id": r["id"], "n": b["n"], "z": z, "lapsed": lapsed, "reason": "データ不足" if z is None else ("符号の反転" if z < 0 else ("区間が 0 を跨ぐ" if lapsed else ""))})
    return {"frozen": str(frozen), "years": years, "rules": out, "n_lapsed": sum(o["lapsed"] for o in out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg", "eval", "reeval"])
    ap.add_argument("frozen", nargs="?")
    ap.add_argument("--years", nargs="*", type=int, default=[2026])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.cmd == "prereg":
        s = prereg()
        print(json.dumps(s["counts"], ensure_ascii=False, indent=1))
        print("excluded_market:", len(s["excluded_market"]))
    elif a.cmd == "eval":
        r = run_eval(Path(a.out))
        print(json.dumps(r["adopt_summary"], ensure_ascii=False, indent=1))
    else:
        print(json.dumps(reeval(Path(a.frozen), a.years), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
