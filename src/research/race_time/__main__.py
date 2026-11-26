"""基準の走破時計の関数の CLI（費用 0 円・手元の確定結果の表だけ。nice -n 19 で流す）。

  python -m src.research.race_time fit [--tables DIR]        # 向き・λ を決め、2016〜24 で当てはめて config/race_time_model/v1.json と確かめの JSON
  python -m src.research.race_time bootstrap [--reps 200]     # 主なコースの曲線の 95% 区間（開催日で束ねたブートストラップ）

決まり（数字を見る前に固定。2026-11-26）:
- 当てはめ 2016〜2022、確かめ 2023〜2024。採用の版は 2016〜2024 で当てはめて固定し、2025 年以降に当てる。
- 第 1 版（fit）は頭数の向き・λ を確かめの期間の MAD で選んでいた（漏れ。不採用の版なので直さず残す）。
  第 2 版（compare2）は学習の中（2016〜20 で当てはめ 2021〜22）で選ぶ。λ の格子は {1, 5, 20, 100, 500}（下の端の 1 で決まり、追いかけない）。
- 今の track_speed（engine の 2 着の時計 − 基準の平均）と、次の 3 つで比べ、2 つ以上で良い方を採る:
  残差の MAD（それぞれの目的の値で）・同じ日 × 場 × 芝ダの中の級内相関（ICC。高いほど同じ日の値がそろう）・土日の開催日レベルの相関。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.race_time.features import race_table
from src.pipeline.race_time.model import RaceTimeModel

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("KEIBA_RACE_TIME_DATA") or REPO / "data")      # 作業ツリーから流すときは本体の data/ を指す
TABLES = DATA / "page_reference/tables"
TRACK_SPEED = DATA / "page_reference/track_speed"
MODEL_PATH = REPO / "config/race_time_model/v1.json"
REPORT = REPO / "docs/operations/race_time/fit_report.json"
CACHE = REPO / "data/local/research/race_time/races.parquet"
VERSION = "race_time_v1"
LAMS = (20.0, 100.0, 500.0)


def _mad(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    return float(np.median(np.abs(r - np.median(r))))


def load(tables: Path, rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    df = race_table(range(2016, date.today().year + 1), tables)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def icc(df: pd.DataFrame, col: str) -> float:
    """日 × 場 × 芝ダ（3 レース以上）の中の級内相関 ICC(1)。"""
    d = df.dropna(subset=[col])
    g = d.groupby(["date", "venue", "surf"])[col]
    d = d[g.transform("size") >= 3]
    g = d.groupby(["date", "venue", "surf"])[col]
    k, n, N = g.size(), g.ngroups, len(d)
    if n < 2:
        return float("nan")
    grand = d[col].mean()
    msb = float(((g.mean() - grand) ** 2 * k).sum() / (n - 1))
    msw = float(((d[col] - g.transform("mean")) ** 2).sum() / (N - n))
    k0 = (N - (k ** 2).sum() / N) / (n - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


def next_day_corr(df: pd.DataFrame, col: str) -> tuple[float, int]:
    """土日（連続する 2 日）の同じ場・芝ダの開催日レベル（レースの値の中央値）の相関。"""
    day = df.dropna(subset=[col]).groupby(["date", "venue", "surf"])[col].median().reset_index()
    day["d"] = pd.to_datetime(day["date"])
    nxt = day.copy()
    nxt["d"] = nxt["d"] - pd.Timedelta(days=1)
    m = day.merge(nxt, on=["d", "venue", "surf"], suffixes=("", "_next"))
    if len(m) < 10:
        return float("nan"), int(len(m))
    return float(np.corrcoef(m[col], m[f"{col}_next"])[0, 1]), int(len(m))


def engine_levels(years: list[int]) -> pd.DataFrame:
    rows = []
    for y in years:
        p = TRACK_SPEED / f"races_{y}.parquet"
        if p.exists():
            e = pd.read_parquet(p, columns=["race_id", "time_2nd_adj", "baseline_mean"])
            rows.append(e)
    if not rows:
        return pd.DataFrame(columns=["race_id", "engine_level"])
    e = pd.concat(rows)
    e["engine_level"] = e["time_2nd_adj"] - e["baseline_mean"]
    return e[["race_id", "engine_level"]].drop_duplicates("race_id")


def cmd_fit(a: argparse.Namespace) -> None:
    t0 = time.time()
    df = load(Path(a.tables), a.rebuild)
    tr, te = df[df.date < "20230101"], df[(df.date >= "20230101") & (df.date < "20250101")]
    trials = []
    for fs in (+1, -1):
        m = RaceTimeModel(lam=100.0, field_sign=fs).fit(tr)
        trials.append({"field_sign": fs, "lam": 100.0, "test_mad": _mad(te.y_top5 - m.predict(te))})
    field_sign = min(trials, key=lambda x: x["test_mad"])["field_sign"]
    for lam in LAMS:
        if lam == 100.0:
            continue
        m = RaceTimeModel(lam=lam, field_sign=field_sign).fit(tr)
        trials.append({"field_sign": field_sign, "lam": lam, "test_mad": _mad(te.y_top5 - m.predict(te))})
    lam = min((t for t in trials if t["field_sign"] == field_sign), key=lambda x: x["test_mad"])["lam"]
    m_val = RaceTimeModel(lam=lam, field_sign=field_sign).fit(tr)
    free = RaceTimeModel(lam=lam, field_sign=field_sign, monotone=False).fit(tr)
    te = te.copy()
    te["level"] = te.y_top5 - m_val.predict(te)
    te["level_free"] = te.y_top5 - free.predict(te)
    m_win = RaceTimeModel(target="y_win", lam=lam, field_sign=field_sign).fit(tr)
    te["level_win"] = te.y_win - m_win.predict(te)
    e = engine_levels([2023, 2024])
    both = te.merge(e, on="race_id", how="inner")
    cmp = {
        "n_races_common": int(len(both)),
        "new": {"mad": _mad(both["level"]), "icc": icc(both, "level"), "next_day": next_day_corr(both, "level")},
        "engine": {"mad": _mad(both["engine_level"]), "icc": icc(both, "engine_level"), "next_day": next_day_corr(both, "engine_level")},
    }
    wins = int(cmp["new"]["mad"] < cmp["engine"]["mad"]) + int(cmp["new"]["icc"] > cmp["engine"]["icc"]) + int(cmp["new"]["next_day"][0] > cmp["engine"]["next_day"][0])
    status = "採用" if wins >= 2 else "研究の版（未採用。事前の規則で track_speed に 3 つ中 2 つ以上で勝てなかった）"
    final = RaceTimeModel(lam=lam, field_sign=field_sign).fit(df[df.date < "20250101"])
    viol = final.monotone_violations()
    final.meta.update({"version": VERSION, "built": date.today().isoformat(), "train": "2016-01-01〜2024-12-31", "status": status,
                       "field_sign": field_sign, "lam": lam, "target": "1〜5 着の走破時計の中央値（秒）"})
    final.save(MODEL_PATH)
    win_final = RaceTimeModel(target="y_win", lam=lam, field_sign=field_sign).fit(df[df.date < "20250101"])
    win_final.meta.update({"version": VERSION + "_win", "built": date.today().isoformat(), "train": "2016-01-01〜2024-12-31"})
    win_final.save(MODEL_PATH.with_name("v1_win.json"))
    report = {
        "version": VERSION, "built": date.today().isoformat(), "n_races": int(len(df)),
        "rule": "頭数の向き・λ は確かめ（2023〜24）の残差の MAD で選ぶ。track_speed と 3 つの物差しで比べ 2 つ以上で良い方を採る",
        "trials": trials, "chosen": {"field_sign": field_sign, "lam": lam},
        "test": {"mad": _mad(te["level"]), "mad_free": _mad(te["level_free"]), "mad_win": _mad(te["level_win"]),
                 "corr_top5_win": float(np.corrcoef(te["level"], te["level_win"])[0, 1]), "n": int(len(te))},
        "compare_track_speed": cmp, "new_wins": wins, "adopt_new": wins >= 2, "status": status,
        "caveat": "track_speed の基準表は 2020〜2025 で作られ（確かめの 2023〜24 が学習の中）、群のキーに馬場状態（cond_pool）と内外（layout）を含む",
        "monotone_violations_total": int(sum(sum(r.values()) for r in viol.values())), "monotone_violations": viol,
        "final_meta": final.meta, "elapsed_sec": round(time.time() - t0, 1),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("chosen", "test", "compare_track_speed", "new_wins", "adopt_new", "monotone_violations_total", "elapsed_sec")},
                     ensure_ascii=False, indent=1, default=float))


def loo_mad(df: pd.DataFrame, col: str) -> tuple[float, int]:
    """一つ抜きの残差の MAD: レースの値 −（同じ日・場・芝ダのほかのレースの値の中央値）。ほかのレースが 2 つ以上ある行だけ。"""
    d = df.dropna(subset=[col])
    out = []
    for _, g in d.groupby(["date", "venue", "surf"]):
        if len(g) < 3:
            continue
        v = g[col].to_numpy(float)
        for i in range(len(v)):
            out.append(v[i] - np.median(np.delete(v, i)))
    return _mad(np.asarray(out)), len(out)


def engine_rebuilt(years_fit: list[int], years_eval: list[int], cond: bool, work: Path) -> pd.DataFrame:
    """今の track_speed（src/research/race/track_speed_engine.py）の作りを、当てはめの窓を替えて作り直し、確かめの年に当てる。
    基準表は work の中だけに書く（本物の page_reference/knowledge は上書きしない）。cond=False は馬場状態の群を 1 つにまとめる。"""
    import src.research.race.track_speed_engine as E

    saved = (E.BASELINES_PATH, E.PACE_BASELINES_PATH, E.COND_POOLS, E.COND_CANDIDATES, E.PROJECT_ROOT)
    try:
        tag = "cond" if cond else "nocond"
        # engine は内外の表（course_layout_map.json）と改修の表（track_renovations.json）を PROJECT_ROOT/data/local/knowledge から読むが、
        # 実物は data/page_reference/knowledge にある（2026-11-26 の critic の指摘）。読めないと内外を分けず改修前も混ぜた弱い基準になるので、
        # 一時の root の data/local/knowledge を実物へのリンクにして、本物と同じ作りにする
        kroot = work / f"root_{tag}"
        (kroot / "data/local").mkdir(parents=True, exist_ok=True)
        if not (kroot / "data/local/knowledge").exists():
            (kroot / "data/local/knowledge").symlink_to(DATA / "page_reference/knowledge")
        E.PROJECT_ROOT = kroot
        E.BASELINES_PATH = work / f"bl_{tag}.parquet"
        E.PACE_BASELINES_PATH = work / f"pace_{tag}.parquet"
        if not cond:
            E.COND_POOLS = {"all": frozenset({"良", "稍重", "重", "不良"})}
            E.COND_CANDIDATES = {k: ("all",) for k in ("良", "稍重", "重", "不良")}
        eng = E.TrackSpeedEngine(base_dir=DATA.parent)
        eng.build_baselines(years=[str(y) for y in years_fit])
        races = eng.load_races_from_parquet([str(y) for y in years_eval])
        rows = []
        for _, r in races.iterrows():
            sc = eng.score_race(r)
            if sc and sc.get("baseline_mean") is not None and sc.get("time_2nd_adj") is not None:
                rows.append({"race_id": str(r["race_id"]), f"eng_{tag}": float(sc["time_2nd_adj"]) - float(sc["baseline_mean"])})
        return pd.DataFrame(rows)
    finally:
        E.BASELINES_PATH, E.PACE_BASELINES_PATH, E.COND_POOLS, E.COND_CANDIDATES, E.PROJECT_ROOT = saved


def cmd_compare(a: argparse.Namespace) -> None:
    """直した比べ（統括役の判断 2026-11-26。数字を見る前に固定）:
    物差し = 一つ抜きの残差の MAD（小さいほど良い）・ICC（大きいほど良い）・土日の相関（大きいほど良い）。2 つ以上で採用。
    今の track_speed は同じ窓（2016〜22）で作り直し、馬場状態の群なし（主）・あり（参考）。確かめは 2023〜24。"""
    import tempfile

    df = load(Path(a.tables))
    tr, te = df[df.date < "20230101"], df[(df.date >= "20230101") & (df.date < "20250101")].copy()
    rep = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    ch = rep.get("chosen") or {"lam": 100.0, "field_sign": 1}
    m = RaceTimeModel(lam=ch["lam"], field_sign=ch["field_sign"]).fit(tr)
    te["new"] = te.y_top5 - m.predict(te)
    with tempfile.TemporaryDirectory(dir=os.environ.get("CLAUDE_JOB_DIR")) as w:
        e0 = engine_rebuilt(list(range(2016, 2023)), [2023, 2024], cond=False, work=Path(w))
        e1 = engine_rebuilt(list(range(2016, 2023)), [2023, 2024], cond=True, work=Path(w))
    both = te.merge(e0, on="race_id").merge(e1, on="race_id")
    res = {}
    for col in ("new", "eng_nocond", "eng_cond"):
        lm, n = loo_mad(both, col)
        nd, npairs = next_day_corr(both, col)
        res[col] = {"loo_mad": lm, "n_loo": n, "icc": icc(both, col), "next_day": nd, "n_pairs": npairs}
    new, eng = res["new"], res["eng_nocond"]
    wins = int(new["loo_mad"] < eng["loo_mad"]) + int(new["icc"] > eng["icc"]) + int(new["next_day"] > eng["next_day"])
    out = {"rule": "一つ抜きの残差の MAD・ICC・土日の相関で、track_speed（同じ窓 2016〜22・馬場状態の群なし）に 2 つ以上で良ければ採用",
           "why_changed": "残差の MAD は、測りたい馬場の分と日の分が残る新しい関数に不利で、馬場状態で条件を付けた基準ほど有利になる（統括役の判断 2026-11-26）",
           "n_races_common": int(len(both)), "metrics": res, "new_wins": wins, "adopt_new": wins >= 2}
    p = REPORT.with_name("compare_report.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


LAMS_V2 = (1.0, 5.0, 20.0, 100.0, 500.0)


def cmd_compare2(a: argparse.Namespace) -> None:
    """第 2 版（ユーザーの判断「単調の関数を細かくして比べ直す」2026-11-26）。作りは数字を見る前に固定して commit した:
    - 形: spec=v2（クラスの節点を順位の段ごと・前半 600m は分位 10〜90% の 9 点・頭数は 7〜17 の奇数・年齢の段をコースごとに。すべて単調・共通の形へ縮める）
    - λ と頭数の向き: 学習の中だけで選ぶ（2016〜2020 で当てはめ、2021〜2022 の残差の MAD）。λ の格子 {1, 5, 20, 100, 500}
    - 比べ: 2016〜2022 で当てはめ、2023〜2024 で track_speed（同じ窓・馬場状態の群なし）と、一つ抜きの残差の MAD・ICC・土日の相関。2 つ以上で採用
    - 参考: 目的の値を 2 着の時計にそろえた版（track_speed と同じ）も 1 行で
    """
    import tempfile

    df = load(Path(a.tables), a.rebuild)
    if "y_2nd" not in df.columns:
        sys.exit("races.parquet に y_2nd が無い（--rebuild で作り直す）")
    inner_tr, inner_va = df[df.date < "20210101"], df[(df.date >= "20210101") & (df.date < "20230101")]
    trials = []
    for fs in (+1, -1):
        for lam in LAMS_V2:
            m = RaceTimeModel(lam=lam, field_sign=fs, spec="v2").fit(inner_tr)
            trials.append({"field_sign": fs, "lam": lam, "inner_mad": _mad(inner_va.y_top5 - m.predict(inner_va))})
    best = min(trials, key=lambda x: x["inner_mad"])
    tr, te = df[df.date < "20230101"], df[(df.date >= "20230101") & (df.date < "20250101")].copy()
    m = RaceTimeModel(lam=best["lam"], field_sign=best["field_sign"], spec="v2").fit(tr)
    te["new_v2"] = te.y_top5 - m.predict(te)
    m2 = RaceTimeModel(target="y_2nd", lam=best["lam"], field_sign=best["field_sign"], spec="v2").fit(tr)
    te["new_v2_2nd"] = te.y_2nd - m2.predict(te)
    with tempfile.TemporaryDirectory(dir=os.environ.get("CLAUDE_JOB_DIR")) as w:
        e0 = engine_rebuilt(list(range(2016, 2023)), [2023, 2024], cond=False, work=Path(w))
        e25 = engine_rebuilt(list(range(2016, 2023)), [2025], cond=False, work=Path(w))
    t25 = df[(df.date >= "20250101") & (df.date < "20260101")].copy()
    t25["new_v2"] = t25.y_top5 - m.predict(t25)
    b25 = t25.merge(e25, on="race_id")
    ref2025 = {}
    for col in ("new_v2", "eng_nocond"):
        lm, n = loo_mad(b25, col)
        nd, npairs = next_day_corr(b25, col)
        ref2025[col] = {"loo_mad": lm, "icc": icc(b25, col), "next_day": nd, "n": int(len(b25)), "n_pairs": npairs}
    both = te.merge(e0, on="race_id")
    res = {}
    for col in ("new_v2", "new_v2_2nd", "eng_nocond"):
        lm, n = loo_mad(both, col)
        nd, npairs = next_day_corr(both, col)
        res[col] = {"loo_mad": lm, "n_loo": n, "icc": icc(both, col), "next_day": nd, "n_pairs": npairs}
    new, eng = res["new_v2"], res["eng_nocond"]
    wins = int(new["loo_mad"] < eng["loo_mad"]) + int(new["icc"] > eng["icc"]) + int(new["next_day"] > eng["next_day"])
    adopt = wins >= 2
    status = "採用" if adopt else "研究の版（未採用。第 2 版も track_speed（群なし）に 3 つ中 2 つ以上で勝てなかった）"
    viol = m.monotone_violations()
    final = RaceTimeModel(lam=best["lam"], field_sign=best["field_sign"], spec="v2").fit(df[df.date < "20250101"])
    final.meta.update({"version": "race_time_v2", "built": date.today().isoformat(), "train": "2016-01-01〜2024-12-31", "status": status,
                       "field_sign": best["field_sign"], "lam": best["lam"], "target": "1〜5 着の走破時計の中央値（秒）"})
    final.save(MODEL_PATH.with_name("v2.json"))
    out = {"spec": "v2", "rule": cmd_compare2.__doc__, "trials_inner": trials, "chosen": best, "n_races_common": int(len(both)),
           "metrics": res, "new_wins": wins, "adopt_new": adopt, "status": status,
           "reference_2025_not_for_decision": ref2025,
           "opponent_fix": "engine に内外の表・改修の表を読ませた（data/page_reference/knowledge）。先の比べ（2026-11-26 夕方）は読めておらず弱い基準だった",
           "monotone_violations_total": int(sum(sum(r.values()) for r in viol.values())),
           "final_monotone_violations_total": int(sum(sum(r.values()) for r in final.monotone_violations().values()))}
    p = REPORT.with_name("compare_v2_report.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("chosen", "metrics", "new_wins", "adopt_new", "reference_2025_not_for_decision", "monotone_violations_total")},
                     ensure_ascii=False, indent=1, default=float))


LEVEL_DIR = Path(os.environ.get("KEIBA_RACE_LEVEL_OUT") or REPO / "data/page_reference/race_level")   # 書き先（既定は作業場所の data/）


def member_next_run_check(levels: pd.DataFrame, tables: Path, years: list[int]) -> dict:
    """メンバーレベルの確かめ（計画で固定。2023〜24）: 3 分位ごとに、出走馬が次走で同じクラスに出たときの 3 着内率と着順の偏差（着順 ÷ 頭数）。
    高い組（速い側）が、低い組より 3 着内率が高く着順の偏差が小さいこと。結果の後に作る値なので予測には使わない（後ろ向きの確かめ）。"""
    from src.pipeline.race_time.features import CLASS_ORD

    cols = ["race_id", "date", "horse_id", "finish_position", "field_size", "grade", "surface"]
    h = pd.concat([pd.read_parquet(tables / str(y) / "race_result_flat.parquet", columns=cols) for y in years if (tables / str(y) / "race_result_flat.parquet").exists()])
    h = h[h["surface"].isin(["芝", "ダート"])].copy()                     # 障害は除く（平地の次走だけを見る）
    h = h[h["finish_position"] != 0]                                    # 0 は出走していない（取消・除外）。中止（-1）は次走に残し 3 着内でないに数える
    h["date"] = h["date"].astype(str).str.replace("-", "")
    h["cls"] = h["grade"].map(CLASS_ORD)                                # 同じクラスはモデルと同じ段（新馬＝未勝利・OP＝L）
    h = h.sort_values(["horse_id", "date"])
    g = h.groupby("horse_id")
    h["next_race"], h["next_pos"], h["next_field"], h["next_cls"] = g["race_id"].shift(-1), g["finish_position"].shift(-1), g["field_size"].shift(-1), g["cls"].shift(-1)
    h = h[(h["finish_position"] > 0) & h["next_race"].notna() & (h["next_cls"] == h["cls"])]
    h["next_dev"] = np.where(h["next_pos"] > 0, h["next_pos"] / h["next_field"], 1.0)   # 中止は最下位の扱い
    lv = levels[(levels["date"] >= "20230101") & (levels["date"] < "20250101") & (levels["n_other_races"] >= 2)].dropna(subset=["member_level_per1k"])
    lv = lv.assign(tert=pd.qcut(lv["member_level_per1k"], 3, labels=["高い（速い）", "中", "低い（遅い）"]))
    m = h.merge(lv[["race_id", "tert"]], on="race_id")
    g = m.groupby("tert", observed=True).apply(lambda x: pd.Series({"n": len(x), "top3": float(((x["next_pos"] > 0) & (x["next_pos"] <= 3)).mean()),
                                                                     "pos_dev": float(x["next_dev"].mean())}))
    hi, lo = g.loc["高い（速い）"], g.loc["低い（遅い）"]
    corr = float(lv[["member_level_per1k", "member_level_win_per1k"]].dropna().corr().iloc[0, 1]) if "member_level_win_per1k" in lv else float("nan")
    ok = bool(hi["top3"] > lo["top3"] and hi["pos_dev"] < lo["pos_dev"] and corr >= 0.7)
    return {"by_tertile": g.round(4).reset_index().to_dict("records"), "corr_top5_win": round(corr, 4), "pass": ok,
            "rule": "高い組の次走（同じクラス）の 3 着内率 > 低い組 かつ 着順の偏差 < 低い組、かつ 2 つの作りの相関 ≥ 0.7"}


def cmd_levels(a: argparse.Namespace) -> None:
    """v2 でレースレベル・開催日レベル・メンバーレベルを作り、page_reference/race_level/races_<年>・days_<年>.parquet に書く（手で流す）。"""
    from src.pipeline.race_time.levels import compute, shrink_k

    df = load(Path(a.tables), True)                                       # 毎回作り直す（新しいレースを黙って落とさない）
    model = RaceTimeModel.load(MODEL_PATH.with_name("v2.json"))
    wp = MODEL_PATH.with_name("v2_win.json")
    if wp.exists():
        win = RaceTimeModel.load(wp)
    else:
        win = RaceTimeModel(target="y_win", lam=model.lam, field_sign=model.field_sign, spec="v2").fit(df[df.date < "20250101"])
        win.meta.update({"version": "race_time_v2_win", "built": date.today().isoformat(), "train": "2016-01-01〜2024-12-31"})
        win.save(wp)
    tr = df[df.date < "20250101"].dropna(subset=["front600"])
    k_raw = shrink_k((tr["y_top5"] - model.predict(tr)) / (tr["distance"] / 1000.0), tr[["date", "venue", "surf"]], clip=False)
    k = float(np.clip(k_raw, 2.0, 10.0))
    races, days = compute(df, model, k, win_model=win)
    # メンバーレベルの確かめは学習の外で: 2016〜22 で当てはめた版で 2023〜24 を作り、次走で確かめる（表示の可否はこれで決める）
    tr22 = df[df.date < "20230101"]
    m22 = RaceTimeModel(lam=model.lam, field_sign=model.field_sign, spec="v2").fit(tr22)
    w22 = RaceTimeModel(target="y_win", lam=model.lam, field_sign=model.field_sign, spec="v2").fit(tr22)
    r22, _ = compute(df[df.date < "20250101"], m22, k, win_model=w22)
    check = member_next_run_check(r22, Path(a.tables), [2023, 2024, 2025])
    LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    for y in sorted({d[:4] for d in races["date"]}):
        races[races["date"].str.startswith(y)].to_parquet(LEVEL_DIR / f"races_{y}.parquet", index=False)
        days[days["date"].str.startswith(y)].to_parquet(LEVEL_DIR / f"days_{y}.parquet", index=False)
    meta = {"version": model.meta.get("version"), "built": date.today().isoformat(), "model_built": model.meta.get("built"),
            "shrink_k": round(k, 3), "shrink_k_raw": round(float(k_raw), 3), "shrink_k_note": "生の推定が下限 2 を下回ったので 2 に切った",
            "member_check": check, "member_check_fit": "2016〜2022（学習の外の 2023〜24 で確かめる）", "member_visible": check["pass"], "n_races": int(len(races))}
    (LEVEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    REPORT.with_name("levels_report.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1, default=float))


def cmd_bootstrap(a: argparse.Namespace) -> None:
    """採用の版（2016〜24）について、主なコース（件数の上位 8）の曲線の 95% 区間。開催日を単位に取り直す。"""
    df = load(Path(a.tables))
    d = df[df.date < "20250101"].dropna(subset=["front600"])
    base = RaceTimeModel.load(Path(a.model))
    top = d["course"].value_counts().head(8).index.tolist()
    grids = {t: np.linspace(term.lo, term.hi, 41) for t, term in base.terms.items()}
    days = d["date"].unique()
    rng = np.random.default_rng(20261126)
    curves: dict[str, dict[str, list]] = {c: {t: [] for t in grids} for c in [*top, "_common"]}
    age = []
    init = {k: np.asarray(v, float) for k, v in base.params.items()}
    viol_total = 0
    conv = 0
    for i in range(a.reps):
        pick = rng.choice(days, size=len(days), replace=True)
        cnt = pd.Series(pick).value_counts()
        s = d[d["date"].isin(cnt.index)]
        s = s.loc[s.index.repeat(s["date"].map(cnt).to_numpy())]
        m = RaceTimeModel(lam=base.lam, field_sign=base.field_sign, spec=base.spec)
        m.terms, m.courses, m.course_band, m.pace_ref, m.pace_sd = base.terms, base.courses, base.course_band, base.pace_ref, base.pace_sd
        m.fit(s, maxiter=400, init=init)
        conv += int(m.meta.get("converged", False))
        for c in curves:
            for t, xs in grids.items():
                curves[c][t].append(m.term_curve(None if c == "_common" else c, t, xs).tolist())
        age.append(m.params["age"])
        viol_total += sum(sum(r.values()) for r in m.monotone_violations(n_grid=60).values())
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{a.reps}", flush=True)
    out = {"reps": a.reps, "model": base.meta.get("version"), "courses": top, "grids": {t: xs.tolist() for t, xs in grids.items()}, "curves": {}}
    for c, tt in curves.items():
        out["curves"][c] = {}
        for t, arr in tt.items():
            A = np.asarray(arr)
            out["curves"][c][t] = {"mid": base.term_curve(None if c == "_common" else c, t, grids[t]).round(4).tolist(),
                                   "lo": np.quantile(A, 0.025, axis=0).round(4).tolist(), "hi": np.quantile(A, 0.975, axis=0).round(4).tolist()}
    A = np.asarray(age)
    out["age"] = {"mid": base.params["age"], "lo": np.quantile(A, 0.025, axis=0).round(4).tolist(), "hi": np.quantile(A, 0.975, axis=0).round(4).tolist()}
    out["monotone_violations_all_reps"] = int(viol_total)
    out["converged_reps"] = int(conv)
    out["note"] = "年齢の段はコースごと（B_age）だが、ここの区間は共通の中心（age）の分だけ"
    p = REPORT.with_name(f"bootstrap_{base.spec}.json")
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("wrote", p)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.research.race_time")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--tables", default=str(TABLES))
    f.add_argument("--rebuild", action="store_true")
    b = sub.add_parser("bootstrap")
    b.add_argument("--tables", default=str(TABLES))
    b.add_argument("--reps", type=int, default=200)
    b.add_argument("--model", default=str(MODEL_PATH.with_name("v2.json")))
    c = sub.add_parser("compare")
    c.add_argument("--tables", default=str(TABLES))
    c2 = sub.add_parser("compare2")
    c2.add_argument("--tables", default=str(TABLES))
    c2.add_argument("--rebuild", action="store_true")
    lv = sub.add_parser("levels")
    lv.add_argument("--tables", default=str(TABLES))
    lv.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "bootstrap": cmd_bootstrap, "compare": cmd_compare, "compare2": cmd_compare2, "levels": cmd_levels}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
