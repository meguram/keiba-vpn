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
    return pd.DataFrame(hrows), pd.DataFrame(rrows)


def load_pf(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.pipeline.inference.position_forecast import build_position_forecast

    p = pd.read_parquet(TABLES / str(year) / "race_shutuba_past_flat.parquet",
                        columns=["race_id", "date", "venue", "surface", "distance", "horse_number", "horse_id", "past_races"])
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


def build(years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    H, R = [], []
    for y in years:
        act, race = load_actual(y)
        tdh, tdr = load_td(race["race_id"].unique())
        pfh, pfr = load_pf(y)
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
