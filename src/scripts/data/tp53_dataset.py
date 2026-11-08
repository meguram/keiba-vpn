"""tp-53: 想定位置取り（position_forecast）・追走難度/想定上がり（tracking_difficulty）と実際の結果を 1 表にまとめる。

python -m src.scripts.data.tp53_dataset --years 2020 2021 2022 2023 2024 2025   # ホストで実行
出力: data/local/modeling/tp53/horse.parquet（1 行 = 1 頭）, race.parquet（1 行 = 1 レース）

- 実測: race_result_flat の通過順位・上がり 3F、レースの前半 3F / 後半 3F（pace JSON）
- 追走難度側の予測: data/calculated_data/tracking_difficulty/{race_id}.json（事前計算。pre_race_only=True、対象日より前の走だけ）
- 位置取り側の予測: race_shutuba_past_flat の過去走から position_forecast をその場で再計算（保存物ではなく同じ関数）
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
TD_DIR = ROOT / "data/calculated_data/tracking_difficulty"
OUT = ROOT / "data/local/modeling/tp53"

_INT = re.compile(r"\d+")


def _passing(s) -> list[int]:
    return [int(x) for x in _INT.findall(str(s or ""))]


def _going_group(tc: str) -> str:
    return "良" if str(tc) == "良" else "道悪"


def load_actual(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = pd.read_parquet(TABLES / str(year) / "race_result_flat.parquet",
                        columns=["race_id", "date", "venue", "surface", "distance", "track_condition", "field_size", "grade",
                                 "horse_number", "horse_id", "finish_position", "passing_order", "last_3f", "pace", "odds"])
    r = r[r["field_size"].fillna(0) >= 8].copy()
    ps = r["passing_order"].map(_passing)
    r["act_pass_first"] = ps.map(lambda x: x[0] if x else np.nan)
    r["act_pass_last"] = ps.map(lambda x: x[-1] if x else np.nan)
    fs = r["field_size"].astype(float)
    r["act_early_norm"] = (r["act_pass_first"] - 1) / (fs - 1)
    r["act_late_norm"] = (r["act_pass_last"] - 1) / (fs - 1)
    r["act_l3f"] = pd.to_numeric(r["last_3f"], errors="coerce")
    r.loc[(r["act_l3f"] < 30) | (r["act_l3f"] > 50), "act_l3f"] = np.nan
    pj = r.drop_duplicates("race_id")[["race_id", "pace"]].copy()

    def _pace(s):
        try:
            d = json.loads(s) if isinstance(s, str) else (s or {})
        except Exception:
            d = {}
        return d.get("t3f") or d.get("first_half_3f"), d.get("l3f") or d.get("second_half_3f")

    pj[["act_t3f", "act_race_l3f"]] = pj["pace"].map(_pace).apply(pd.Series)
    race = r.drop_duplicates("race_id")[["race_id", "date", "venue", "surface", "distance", "track_condition", "field_size", "grade"]].merge(
        pj[["race_id", "act_t3f", "act_race_l3f"]], on="race_id")
    med = r.groupby("race_id")["act_l3f"].median().rename("act_l3f_med")
    race = race.merge(med, on="race_id", how="left")
    r = r.merge(med, on="race_id", how="left")
    r["act_l3f_dev"] = r["act_l3f"] - r["act_l3f_med"]
    r["year"] = year
    race["year"] = year
    race["going_group"] = race["track_condition"].map(_going_group)
    return r.drop(columns=["pace", "passing_order", "last_3f"]), race


def load_td(race_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    hrows, rrows = [], []
    for rid in race_ids:
        p = TD_DIR / f"{rid}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pp = d.get("pace_prediction") or {}
        dist = pp.get("distribution") or {}
        lap = pp.get("lap_times") or {}
        rrows.append({
            "race_id": rid, "td_pace_type": pp.get("pace_type"), "td_pace_index": pp.get("pace_index"),
            "td_front": pp.get("front_runner_count"), "td_stalker": pp.get("stalker_count"), "td_pressure": pp.get("early_pressure"),
            "td_first3f": lap.get("first_3f"), "td_l3f_base": pp.get("last_3f_baseline_sec"),
            "td_n_nige": dist.get("逃げ"), "td_n_senko": dist.get("先行"), "td_n_sashi": dist.get("差し"), "td_n_oikomi": dist.get("追込"),
        })
        ent = {int(e.get("horse_number")): e for e in (d.get("entries") or []) if e.get("horse_number") is not None}
        for f in d.get("position_flow") or []:
            hn = f.get("horse_number")
            if hn is None:
                continue
            pos = f.get("positions") or {}
            e3 = f.get("expected_last_3f") or {}
            stm = f.get("stamina") or {}
            en = ent.get(int(hn), {})
            prof = en.get("profile") or {}
            td = en.get("tracking_difficulty") or {}
            hrows.append({
                "race_id": rid, "horse_number": int(hn), "td_style": f.get("style"), "td_early": (pos.get("early") or {}).get("position_norm"),
                "td_mid": (pos.get("mid") or {}).get("position_norm"), "td_late": (pos.get("late") or {}).get("position_norm"),
                "td_flow": f.get("flow_pattern"), "td_stamina": stm.get("stamina_index"), "td_ease": (f.get("allocation") or {}).get("ease_pct", td.get("ease_pct")),
                "td_l3f": e3.get("seconds"), "td_l3f_delta": e3.get("delta_sec"), "td_l3f_rank": e3.get("rank"),
                "td_l3f_ability": e3.get("ability_sec"), "td_l3f_n": e3.get("n_samples"), "td_n_races": prof.get("n_races"),
                "td_typical": prof.get("typical_position"), "td_conf": f.get("confidence"),
            })
    # 旧版の事前計算が無い年（2016〜2019 年）は、結合できるよう列だけの空の表を返す（その年の td_* は欠けたまま）
    h = pd.DataFrame(hrows) if hrows else pd.DataFrame(columns=["race_id", "horse_number", "td_l3f"])
    r = pd.DataFrame(rrows) if rrows else pd.DataFrame(columns=["race_id"])
    return h, r


def _past_from_results(year: int) -> pd.DataFrame:
    """馬柱（race_shutuba_past_flat）が無い年（2016〜2018）の代替。結果の表（2016 年以降）から、各馬のそのレースより前の走を
    馬柱の 1 走と同じ形（date・raw に「頭数」と通過順）にして past_races に入れる。position_forecast はそのまま使う。

    2020・2024 年で馬柱から作った値と比べ、pf_early の相関 0.987〜0.988・脚質の一致 97%（差は馬柱に載る地方競馬の走）。
    初角位置の誤差は馬柱 0.2193 / 代替 0.2187 でほぼ同じ（2026-11-06）。"""
    import bisect
    from collections import defaultdict

    cols = ["race_id", "date", "horse_id", "field_size", "passing_order"]
    hist: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for y in range(2016, year + 1):
        f = TABLES / str(y) / "race_result_flat.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f, columns=cols).dropna(subset=["horse_id", "passing_order"])
        for hid, d, fs, po in zip(r["horse_id"].astype(str), r["date"].astype(str).str.replace("-", ""), r["field_size"], r["passing_order"]):
            hist[hid].append((d, f"{d[:4]}.{d[4:6]}.{d[6:]} {int(fs) if pd.notna(fs) else 0}頭 {po}"))
    for v in hist.values():
        v.sort()
    e = pd.read_parquet(TABLES / str(year) / "race_result_flat.parquet", columns=["race_id", "date", "venue", "surface", "distance", "horse_number", "horse_id"])
    out = []
    for rid, d, hid in zip(e["race_id"], e["date"].astype(str).str.replace("-", ""), e["horse_id"].astype(str)):
        v = hist.get(hid, [])
        i = bisect.bisect_left(v, (d, ""))
        out.append(json.dumps([{"date": dd, "raw": raw} for dd, raw in v[max(0, i - 8):i]], ensure_ascii=False))
    return e.assign(past_races=out)


def load_pf(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.pipeline.inference.position_forecast import build_position_forecast

    src = TABLES / str(year) / "race_shutuba_past_flat.parquet"
    if src.exists():
        p = pd.read_parquet(src, columns=["race_id", "date", "venue", "surface", "distance", "horse_number", "horse_id", "past_races"])
    else:                                              # 2016〜2018 年は馬柱が無いので、結果の表の過去走から作る
        p = _past_from_results(year)
    hrows, rrows = [], []
    for rid, g in p.groupby("race_id", sort=False):
        r0 = g.iloc[0]
        ents, pd_entries = [], []
        for _, r in g.iterrows():
            if pd.isna(r["horse_number"]):
                continue
            hn = int(r["horse_number"])
            try:
                pr = json.loads(r["past_races"]) if isinstance(r["past_races"], str) else []
            except Exception:
                pr = []
            ents.append({"horse_number": hn, "horse_id": r["horse_id"]})
            pd_entries.append({"horse_id": r["horse_id"], "horse_number": hn, "past_races": pr})
        if len(ents) < 8:
            continue
        by_no, pace, gate = build_position_forecast({"entries": pd_entries}, ents, str(r0["date"]).replace("-", ""),
                                                   {"venue": r0["venue"], "surface": r0["surface"], "distance": r0["distance"]})
        rrows.append({"race_id": rid, "pf_level": pace["level"], "pf_front": pace["front_runner_count"], "pf_stalker": pace["stalker_count"],
                      "pf_pressure": pace["early_pressure"], "pf_known": pace["known_horses"], "pf_gate_applied": gate.get("applied")})
        for hn, f in by_no.items():
            if not f:
                continue
            hrows.append({"race_id": rid, "horse_number": hn, "pf_early": f.get("early_pos_norm"), "pf_rank": f.get("early_pos_est"),
                          "pf_style": f.get("style_code"), "pf_style_raw": f.get("style_raw_code") or f.get("style_code"), "pf_runs": f.get("runs_used")})
    return pd.DataFrame(hrows), pd.DataFrame(rrows)


def apply_learned_pos(year: int, pfh: pd.DataFrame, pfr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TP53_POS_PRED（pos_model の oof の parquet）があれば、位置取り予想を学習モデルの予測に差し替える（2026-11-07、§2.6 の区分 A）。
    予測が無い馬は従来の集計の値のまま。元の値は pf_early_agg に残す。レース単位の量（逃げ・先行の頭数・圧力・判定）は、
    position_forecast と同じ閾値（style_of）・同じ規則で数え直す。"""
    import os

    path = os.environ.get("TP53_POS_PRED")
    if not path or pfh.empty:
        return pfh, pfr
    from src.pipeline.inference.position_forecast import style_of

    pp = pd.read_parquet(path, columns=["race_id", "horse_number", "year", "pos_pred"])
    pp = pp[pp["year"] == year].drop(columns="year")
    pp["race_id"], pp["horse_number"] = pp["race_id"].astype(str), pp["horse_number"].astype(int)
    h = pfh.copy()
    h["race_id"] = h["race_id"].astype(str)
    h = h.merge(pp, on=["race_id", "horse_number"], how="left")
    h["pf_early_agg"] = h["pf_early"]
    h["pf_early"] = h["pos_pred"].where(h["pos_pred"].notna(), h["pf_early"])
    h["pf_rank"] = h.groupby("race_id")["pf_early"].rank(method="first")
    h["pf_style"] = h["pf_early"].map(lambda v: style_of(v) if pd.notna(v) else None)
    h["pf_style_raw"] = h["pf_style"]
    g = h.groupby("race_id")
    agg = pd.DataFrame({"front": g["pf_style"].apply(lambda x: (x == "front").sum()), "stalker": g["pf_style"].apply(lambda x: (x == "stalker").sum()),
                        "known": g["pf_early"].count()}).reset_index()
    r = pfr.copy()
    r["race_id"] = r["race_id"].astype(str)
    r = r.merge(agg, on="race_id", how="left")
    fs = r["race_id"].map(h.groupby("race_id").size()).astype(float)
    r["pf_front"], r["pf_stalker"], r["pf_known"] = r["front"].fillna(0).astype(int), r["stalker"].fillna(0).astype(int), r["known"].fillna(0).astype(int)
    r["pf_pressure"] = ((r["pf_front"] + r["pf_stalker"]) / fs).round(3)

    def level(row) -> str:
        if row["pf_known"] < max(4, row["_fs"] * 0.5):
            return "unknown"
        if row["pf_front"] >= 3 or (row["pf_front"] >= 2 and row["pf_pressure"] >= 0.4):
            return "high"
        if row["pf_front"] == 0 or (row["pf_front"] == 1 and row["pf_pressure"] < 0.35):
            return "slow"
        return "mid"

    r["_fs"] = fs
    r["pf_level"] = r.apply(level, axis=1)
    return h.drop(columns=["pos_pred"]), r.drop(columns=["front", "stalker", "known", "_fs"])


def build(years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    H, R = [], []
    for y in years:
        act, race = load_actual(y)
        tdh, tdr = load_td(race["race_id"].unique())
        pfh, pfr = load_pf(y)
        pfh, pfr = apply_learned_pos(y, pfh, pfr)
        h = act.merge(tdh, on=["race_id", "horse_number"], how="left").merge(pfh, on=["race_id", "horse_number"], how="left")
        r = race.merge(tdr, on="race_id", how="left").merge(pfr, on="race_id", how="left")
        print(y, "horses", len(h), "races", len(r), "td", int(h["td_l3f"].notna().sum()), "pf", int(h["pf_early"].notna().sum()), flush=True)
        H.append(h)
        R.append(r)
    return pd.concat(H, ignore_index=True), pd.concat(R, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h, r = build(a.years)
    h.to_parquet(OUT / "horse.parquet", index=False)
    r.to_parquet(OUT / "race.parquet", index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
