"""追走難度 v2 の最終モデル（試行 6、2016〜2025 年で学習した final_tracking_t6）で、1 日分の出走馬の drain / demand / ease_pct を作る。

学習側（tp54_dataset・tp53_dataset・fit_final_tracking_t6）と同じ式を共有して使う。馬の過去の上がり・位置取りは馬柱（直近 5 走）では足りない
（過去の終いの落ち hy3 は直近 8 走ほど要る）ので、結果表（race_result_flat 2016 年〜）から作る。ホスト・バックフィル用（Cloud Run の起動時には使わない）。

入力
  entrants: 出走馬の表（race_id・date・horse_id・horse_number・venue・surface・distance・track_condition・field_size。1 日分）
  hist_a / hist_b: ``history_a`` / ``history_b``（結果表から。全期間を 1 回作って日ごとに使い回す）
  pos: ``pos_serving.pos_pred_for_card`` の出力（race_id・horse_id・pos_pred・pp_*）
  race_cols: race_id → ラップ予想の列（serving_v6.lap_race_columns と lapd_race_columns の和）
出力は ``day_features`` → ``predict`` → ``make_docs``（tracking_v2_service と同じ保存形式。source=backfill）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.inference import pace_coherence as pcoh
from src.pipeline.inference.position_forecast import style_of
from src.pipeline.models import tracking_difficulty_v2 as td2
from src.scripts.data import tp54_model as tm
from src.scripts.data.pos_model import PP_COLS
from src.scripts.data.tp53_dataset import actuals_from_results
from src.scripts.data.tp54_dataset import run_features

ENV_DIR = "KEIBA_TRACKING_T6_DIR"
VERSION = "tv2-t6-1"
MIN_FIELD = 8
JST = timezone(timedelta(hours=9))
ACT_COLS = ["act_early_norm", "act_late_norm", "act_l3f", "act_l3f_dev", "finish_position"]
HIST_A_COLS = ["race_id", "horse_id", "date", "venue", "surface", "distance", "track_condition", "field_size", "horse_number", *ACT_COLS]
RESULT_COLS = ["race_id", "date", "venue", "surface", "distance", "track_condition", "field_size", "grade", "horse_number", "horse_id", "finish_position",
               "passing_order", "last_3f", "pace", "odds"]
LP_BASE = ("lp_p_slow", "lp_p_mid", "lp_p_high", "lp_exp_f3_dev", "lp_exp_l3_dev", "lp_exp_lap0_dev")
LD_BASE = ("ld_f3", "ld_l3", "ld_q")
INPUT_KEYS = ("pf_early", "pf_rank", "z_hat", *PP_COLS, "w_adj", "w_early", "w_n")


def model_dir() -> Path | None:
    v = os.environ.get(ENV_DIR)
    return Path(v) if v else None


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.replace(r"[./]", "-", regex=True), errors="coerce")


class T6Model:
    def __init__(self, meta: dict[str, Any], drain: list[Any], demand: Any, pe: dict, cfg: dict, pace_v2: dict):
        self.meta, self.drain, self.demand, self.pe, self.cfg, self.pace_v2 = meta, drain, demand, pe, cfg, pace_v2

    @classmethod
    def load(cls, d: str | Path | None = None) -> "T6Model":
        import lightgbm as lgb

        d = Path(d) if d else model_dir()
        if d is None:
            raise FileNotFoundError(f"{ENV_DIR} が未設定")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        cfg, pace_v2 = pcoh.load_config(), pcoh.load_pace_v2()
        if not cfg or not pace_v2:
            raise FileNotFoundError("config/pace_coherence.json または pace_coherence_v2.json が読めない")
        return cls(meta, [lgb.Booster(model_file=str(d / f"drain_s{s}.txt")) for s in meta["seeds"]], lgb.Booster(model_file=str(d / "demand.txt")),
                   json.loads((d / "position_effect.json").read_text(encoding="utf-8")), cfg, pace_v2)


# ───────────────────────── 履歴（結果表から。学習と同じ母集団）

def history_a(results: pd.DataFrame) -> pd.DataFrame:
    """追走難度の履歴: 8 頭立て以上のレースの実測（初角・終角の位置・上がり 3F・その偏差）。results は race_result_flat の RESULT_COLS。"""
    act, _ = actuals_from_results(results[RESULT_COLS], 0)
    act["race_id"], act["horse_id"] = act["race_id"].astype(str), act["horse_id"].astype(str)
    act["date"] = _date(act["date"])
    return act[HIST_A_COLS]


def history_b(results: pd.DataFrame) -> pd.DataFrame:
    """ラップ予想側の馬の履歴: 芝・ダートの全走の初角の正規化位置（頭数の制限なし。tp55 の h_early_w の母集団）。"""
    r = results[results["surface"].isin(["芝", "ダート"])]
    early = [lf.first_corner_norm(p, f) for p, f in zip(r["passing_order"], r["field_size"])]
    return pd.DataFrame({"race_id": r["race_id"].astype(str).to_numpy(), "horse_id": r["horse_id"].astype(str).to_numpy(), "date": _date(r["date"]).to_numpy(), "early": early})


# ───────────────────────── 1 日分の特徴量

def pf_columns(e: pd.DataFrame) -> pd.DataFrame:
    """位置取り予想（pf_early）から、レース内の順位・逃げ先行の頭数・圧力・既知頭数。tp53_dataset.apply_learned_pos と同じ規則（同じ閾値 style_of）。"""
    g = e.groupby("race_id")
    style = e["pf_early"].map(lambda v: style_of(v) if pd.notna(v) else None)
    front = (style == "front").groupby(e["race_id"]).transform("sum")
    stalker = (style == "stalker").groupby(e["race_id"]).transform("sum")
    n = g["pf_early"].transform("size").astype(float)
    return pd.DataFrame({"pf_rank": g["pf_early"].rank(method="first"), "pf_front": front.astype(int), "pf_stalker": stalker.astype(int),
                         "pf_known": g["pf_early"].transform("count").astype(int), "pf_pressure": ((front + stalker) / n).round(3)}, index=e.index)


def _prep(entrants: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    e = entrants.copy()
    e["race_id"], e["horse_id"] = e["race_id"].astype(str), e["horse_id"].astype(str)
    e = e.drop_duplicates(["race_id", "horse_id"]).dropna(subset=["horse_number"]).reset_index(drop=True)
    e["date"] = pd.to_datetime(e["date"])
    if e["date"].nunique() != 1:
        raise ValueError("entrants は 1 日分にする（過去走の切り方が日付単位のため）")
    size = e.groupby("race_id")["horse_id"].transform("size")
    e["field_size"] = e["field_size"].fillna(size) if "field_size" in e.columns else size
    p = pos.copy()
    p["race_id"], p["horse_id"] = p["race_id"].astype(str), p["horse_id"].astype(str)
    e = e.merge(p[["race_id", "horse_id", "pos_pred", *PP_COLS]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    e["pf_early"] = e["pos_pred"].astype(float)
    e = pd.concat([e, pf_columns(e)], axis=1)
    return e


def _zhat(model: T6Model, e: pd.DataFrame) -> pd.Series:
    first = e.drop_duplicates("race_id").set_index("race_id")
    z = {rid: pcoh.predict_pace_z_v2(model.pace_v2, r["pf_front"], r["pf_stalker"], r["pf_pressure"], r["pf_known"], r["field_size"], r["surface"], r["distance"],
                                     r["track_condition"]) for rid, r in first.iterrows()}
    return e["race_id"].map(z).astype(float)


def day_features(model: T6Model, entrants: pd.DataFrame, hist_a: pd.DataFrame, hist_b: pd.DataFrame, pos: pd.DataFrame,
                 race_cols: dict[str, dict[str, float]], z_override: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """戻り値: (ent=入力の写し（pf_*・z_hat・pp_*・w_*）, XC=drain の入力（meta の列順）, XD=demand の入力)。行は entrants の並び（馬番 NaN・同じ馬の重複行を除いたもの。index は 0..n-1 に振り直す）。
    z_override（race_id → 想定ペース）は学習側との突き合わせ用。通常は渡さず、pace_coherence_v2 で作る。"""
    e = _prep(entrants, pos)
    e["z_hat"] = e["race_id"].map(z_override).astype(float) if z_override else _zhat(model, e)
    day = e["date"].iloc[0]
    ids = set(e["horse_id"])
    hist = hist_a[hist_a["horse_id"].isin(ids) & (hist_a["date"] < day)]
    base = e.assign(**{c: np.nan for c in ACT_COLS})
    h = pd.concat([hist.assign(_ent=False), base.assign(_ent=True, _pos=np.arange(len(base)))], ignore_index=True, sort=False)
    h = h.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
    h, _ = run_features(h, model.cfg, model.pe)
    _, y = tm.make_target(h, np.asarray(model.meta["baseline_coef"], float))
    hi = td2.past_drop_inputs(h, y, h["pf_early"].astype(float))
    m = h["_ent"].astype(bool)
    ent_h = h[m].sort_values("_pos")
    hi = hi.loc[ent_h.index]
    X = td2.make_features(ent_h)
    out_index = ent_h.index
    # ラップ予想（レース単位）と、馬の脚質との掛け合わせ
    rc = pd.DataFrame.from_dict(race_cols, orient="index") if race_cols else pd.DataFrame()
    cols = {c: ent_h["race_id"].map(rc[c]).astype(float) if c in rc.columns else pd.Series(np.nan, index=ent_h.index) for c in (*LP_BASE, *LD_BASE)}
    hb = hist_b[hist_b["horse_id"].isin(ids)]
    hf = lf.asof_horse_features(ent_h[["race_id", "horse_id", "date"]].reset_index(drop=True), hb)
    ew = pd.Series(hf["h_early_w"].to_numpy(), index=ent_h.index)
    L = pd.DataFrame(cols).drop(columns=list(LD_BASE))
    L["lp_front_x_high"] = (1 - ew) * L["lp_p_high"]
    L["lp_front_x_slow"] = (1 - ew) * L["lp_p_slow"]
    L["lp_closer_x_slow"] = ew * L["lp_p_slow"]
    L["lp_pos_x_f3"] = (1 - ew) * L["lp_exp_f3_dev"]
    D = pd.DataFrame({c: cols[c] for c in LD_BASE})
    D = pd.concat([D, td2.ld_cross(D["ld_f3"], D["ld_q"], ent_h["pf_early"].astype(float))], axis=1)
    P = ent_h[PP_COLS]
    XC = pd.concat([X, L, D, P, hi], axis=1)[model.meta["drain_features"]]
    XD = X[model.meta["demand_features"]]
    ent = ent_h[["race_id", "date", "horse_id", "horse_number", "venue", "surface", "distance", "track_condition", "field_size", "pos_pred", "pf_early", "pf_rank", "pf_front",
                 "pf_stalker", "pf_pressure", "pf_known", "z_hat", *PP_COLS, "w_adj", "w_early", "w_n"]].copy()
    for f in (ent, XC, XD):
        f.index = out_index.map(dict(zip(ent_h.index, ent_h["_pos"].astype(int))))
    return ent.sort_index(), XC.sort_index(), XD.sort_index()


def predict(model: T6Model, XC: pd.DataFrame, XD: pd.DataFrame) -> pd.DataFrame:
    """drain_sec（大きい＝終いが落ちる）・demand_sec・capacity_sec（= drain − demand）・ease_pct（100 に近いほど楽）。"""
    drain = np.mean([b.predict(XC) for b in model.drain], axis=0)
    demand = model.demand.predict(XD)
    out = pd.DataFrame({"drain_sec": drain, "demand_sec": demand, "capacity_sec": drain - demand}, index=XC.index)
    out["ease_pct"] = td2.to_ease_pct(drain, model.meta["quantiles"])
    return out


# ───────────────────────── 保存用の文書（tracking_v2_service の形）

def _num(v: Any, nd: int | None = None) -> float | None:
    if v is None or pd.isna(v):
        return None
    return round(float(v), nd) if nd is not None else float(v)


def make_docs(model: T6Model, ent: pd.DataFrame, pred: pd.DataFrame, *, source: str = "backfill") -> list[dict]:
    now = datetime.now(JST).isoformat(timespec="seconds")
    docs = []
    for rid, g in ent.groupby("race_id", sort=False):
        if g["field_size"].iloc[0] < MIN_FIELD:
            continue
        horses: dict[str, Any] = {}
        for i, r in g.iterrows():
            p = pred.loc[i]
            if pd.isna(r["pf_early"]) or not np.isfinite(p["drain_sec"]):
                continue
            pct = round(float(p["ease_pct"]), 1)
            horses[str(int(r["horse_number"]))] = {
                "horse_id": r["horse_id"],
                "tracking_v2": {"ease_pct": pct, "label": td2.ease_label(pct), "drain_sec": round(float(p["drain_sec"]), 3),
                                "demand_sec": round(float(p["demand_sec"]), 3), "capacity_sec": round(float(p["capacity_sec"]), 3)},
                "position_early": _num(r["pf_early"]),
                "inputs": {k: _num(r[k]) for k in INPUT_KEYS if k in r.index} | {"pp_pos": _num(r["pos_pred"])},
            }
        if not horses:
            continue
        f = g.iloc[0]
        docs.append({
            "version": VERSION, "engine": "t6", "race_id": rid, "date": str(f["date"])[:10] if "date" in f.index else None, "venue": f["venue"], "surface": f["surface"],
            "distance": _num(f["distance"]), "track_condition": f["track_condition"], "n_horses": int(len(g)), "n_with_v2": len(horses),
            "horse_numbers": sorted(int(x) for x in g["horse_number"]), "computed_at": now, "source": source,
            "model": {"model_version": model.meta.get("version"), "trained_years": model.meta.get("trained_years")},
            "pace_forecast": {"front_runner_count": int(f["pf_front"]), "stalker_count": int(f["pf_stalker"]), "early_pressure": _num(f["pf_pressure"]),
                              "known_horses": int(f["pf_known"]), "field_size": int(f["field_size"])},
            "tracking_v2_meta": {"version": model.meta.get("version"), "pace_z": round(float(f["z_hat"]), 2) if pd.notna(f["z_hat"]) else None,
                                 "definition": "想定位置取りを取るのに必要な通過ラップが、その馬にとって楽か大変か（終いの落ちの予測。50=標準、60以上=楽、30未満=大変）"},
            "horses": horses,
        })
    return docs
