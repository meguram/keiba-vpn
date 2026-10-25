"""v6 の推論用: めぐ指数/netkeiba 指数の履歴 16 列（RB、tp-65）と、前段 S3B の入力の対応。

RB: 学習側（tp65_feat.asof_features）は『各走について、その日より前の直近 5 走（指数がある走）』から ewm/max/median/trimmed/huber と
  下限つき（勝ち馬−40 点）の ewm/加重 Huber、走数を作る。推論では全出走行の表を持たず、馬ごとの履歴（hist: 1 走 1 行、クリップ後の値 x と
  下限つきの値 xf）と定数（params: 2020〜2021 年の分位点・尺度）だけで同じ値を出す。学習側の式は tp65_var の関数をそのまま使う（二重実装にしない）。
  過去日で学習側と完全一致することは tests/pipeline/test_serving_v6.py（合成データ）と、ホストの verify（build_rb_state --verify）で確認する。
前段 S3B: 学習側の入力列（base + tl_ + wc_ + EASE2_* + lp_* + PF 6 列）を StageModels に渡す。EASE2_* / lp_* / PF 列は本番の保存値（追走難度 v2・ラップ予想・
  位置取り予想）を同じ名前に写す（STAGE_INPUT_MAP）。学習行は leave-one-year-out の予測、推論行は本番の予測なので、過去日で分布を確認する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEY = ["race_id", "horse_id"]
SERIES = {"rb_mg": "megu_index_v3a", "rb_nk": "nk"}
CLIP_YEARS = (2020, 2021)
FLOOR = 40.0
LOOKBACK = 5
VARIANTS = ("ewm5", "max5", "median5", "trimmed5", "huber5", "fl_ewm", "fl_whuber")
COLUMNS = [f"{p}_{k}" for p in SERIES for k in (*VARIANTS, "n")]

# 前段 S3B の追加入力（学習側の列名 ← 本番の保存値）。値の写しは serving 側で実装し、過去日で学習側と比べる（未実装の項目は None）。
STAGE_INPUT_MAP = {
    "EASE2_*(6)": "tracking_difficulty v2 の馬別出力（drain/demand/full/capacity/pct/vs_prev）。学習側は tp54 の LOYO 予測",
    "lp_*(12)": "ラップ予想（lap_forecast_service）の p_slow/mid/high・期待ラップ。学習側は tp55 の LOYO 予測",
    "pf_early, pf_rank, z_hat, td_ease, td_l3f_ability, td_stamina": "位置取り予想・追走難度の馬別値（position_forecast / tracking_difficulty_service）",
}


def rb_params(runs: pd.DataFrame) -> dict[str, dict[str, float]]:
    """2020〜2021 年の走だけから、系列ごとのクリップ下限・上限と尺度を決める（将来の値で変わらない）。runs: date と各指数列。"""
    r = runs.copy()
    r["year"] = pd.to_datetime(r["date"]).dt.year
    out = {}
    for pref, col in SERIES.items():
        ref = r.loc[r["year"].isin(CLIP_YEARS) & r[col].notna(), col]
        lo, hi = ref.quantile([0.005, 0.995])
        out[pref] = {"lo": float(lo), "hi": float(hi), "scale": float(1.4826 * np.median(np.abs(ref - ref.median())))}
    return out


def build_hist(runs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """runs: race_id, horse_id, date, finish_position と各指数列（全出走行）。戻り値: 1 走 1 行の履歴（指数がある走のみ、列 <pref>_x / <pref>_xf）。"""
    r = runs[KEY + ["date", "finish_position"] + list(SERIES.values())].copy()
    r["date"] = pd.to_datetime(r["date"])
    r["fp"] = pd.to_numeric(r["finish_position"], errors="coerce")
    out = r[KEY + ["date"]].copy()
    for pref, col in SERIES.items():
        p = params[pref]
        x = r[col].clip(p["lo"], p["hi"])
        wx = x.where((r["fp"] == 1) & r[col].notna()).groupby(r["race_id"]).transform("max")
        out[f"{pref}_x"] = x
        out[f"{pref}_xf"] = np.where(wx.notna(), np.maximum(x, wx - FLOOR), x)
        out.loc[r[col].isna(), [f"{pref}_x", f"{pref}_xf"]] = np.nan
    return out


def rb_features(hist: pd.DataFrame, targets: pd.DataFrame, params: dict) -> pd.DataFrame:
    """targets: race_id, horse_id, date。各行について、その日より前（厳密）の直近 5 走から RB の 16 列を返す。履歴が無い馬の値は欠損（走数 0）。"""
    from src.scripts.data.tp65_var import ewm_decay, feature_variants, weighted_huber

    t = targets[KEY + ["date"]].copy()
    t["date"] = pd.to_datetime(t["date"])
    h = hist.sort_values(["horse_id", "date", "race_id"], kind="stable")
    n = len(t)
    res: dict[str, np.ndarray] = {}
    for pref in SERIES:
        v = h[h[f"{pref}_x"].notna()]
        by_h: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        hid_v = v["horse_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, hid_v[1:] != hid_v[:-1]]) if len(v) else np.array([], dtype=int)
        ends = np.r_[starts[1:], len(v)] if len(v) else starts
        dts, xs, xfs = v["date"].to_numpy("datetime64[ns]"), v[f"{pref}_x"].to_numpy(float), v[f"{pref}_xf"].to_numpy(float)
        for a, b in zip(starts, ends):
            by_h[hid_v[a]] = (dts[a:b], xs[a:b], xfs[a:b])
        M = np.full((n, LOOKBACK), np.nan)
        Mf = np.full((n, LOOKBACK), np.nan)
        for i, (hid, d) in enumerate(zip(t["horse_id"].to_numpy(), t["date"].to_numpy("datetime64[ns]"))):
            g = by_h.get(hid)
            if g is None:
                continue
            k_end = int(np.searchsorted(g[0], d, side="left"))          # 日付が厳密に前の走は [0, k_end)
            if k_end:
                # 同じ日に 2 走ある馬（記録の重複）は、学習側の merge_asof が日付の先頭の行を拾うので、窓の終わりをそこにそろえる
                k_end = int(np.searchsorted(g[0], g[0][k_end - 1], side="left")) + 1
            lo = max(0, k_end - LOOKBACK)
            k = k_end - lo
            if k:
                M[i, :k] = g[1][lo:k_end][::-1]
                Mf[i, :k] = g[2][lo:k_end][::-1]
        scale = params[pref]["scale"]
        feats = feature_variants(M, scale)
        feats = {k: feats[k] for k in ("ewm5", "max5", "median5", "trimmed5", "huber5")}
        feats["fl_ewm"] = ewm_decay(Mf, 0.65)
        feats["fl_whuber"] = weighted_huber(Mf, 0.65, scale)
        for k, a in feats.items():
            res[f"{pref}_{k}"] = a
        res[f"{pref}_n"] = np.sum(~np.isnan(M), axis=1).astype(float)
        # 履歴が無い行は学習側では欠損（merge_asof に相手が無い）。走数も欠損になる
        none = res[f"{pref}_n"] == 0
        for k in (*feats, "n"):
            res[f"{pref}_{k}"] = np.where(none, np.nan, res[f"{pref}_{k}"])
    out = pd.DataFrame(res, index=t.index)[COLUMNS].astype("float32")
    return pd.concat([t[KEY], out], axis=1)


def save_state(dirpath: str | Path, hist: pd.DataFrame, params: dict, cut: str | None = None) -> None:
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(d / "hist.parquet", index=False)
    (d / "params.json").write_text(json.dumps({"params": params, "cut": cut, "columns": COLUMNS}, ensure_ascii=False, indent=1), encoding="utf-8")


def load_state(dirpath: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    d = Path(dirpath)
    return pd.read_parquet(d / "hist.parquet"), json.loads((d / "params.json").read_text(encoding="utf-8"))["params"]


# ───────────────────────── 前段 S3B の追加入力（追走難度 v2 / ラップ予想 / 位置取り予想）を本番の保存値から作る
STATE_ENV = "KEIBA_V6_RB_STATE"
EASE2_MAP = {"EASE2_drain": "drain_sec", "EASE2_demand": "demand_sec", "EASE2_capacity": "capacity_sec", "EASE2_pct": "ease_pct"}
PF_MAP = {"pf_early": "pf_early", "pf_rank": "pf_rank", "z_hat": "z_hat"}
LAP_RACE = ("lp_p_slow", "lp_p_mid", "lp_p_high", "lp_exp_f3_dev", "lp_exp_l3_dev", "lp_exp_lap0_dev")
# 学習側にあって、本番の保存値からはまだ写せない列（追走難度 v2 の内部量 EASE2_full / EASE2_vs_prev、馬の追走難度 td_*、
# ラップ予想の馬別の交互作用 lp_*_x_*（馬の h_early_w が必要）。写しが実装されるまで shadow は起動時に止まる（黙って欠損にしない）。
UNMAPPED = ("EASE2_full", "EASE2_vs_prev", "td_ease", "td_l3f_ability", "td_stamina", "lp_front_x_high", "lp_front_x_slow", "lp_closer_x_slow", "lp_pos_x_f3")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def lap_race_columns(j: dict) -> dict[str, float]:
    """ラップ予想の保存値（tp55-1）→ lp_* のレース単位の列。*_dev は期待ラップ − par（学習側は混合の mu_dev の和）。"""
    by = {s["name"]: float(s["prob"]) for s in j["scenarios"]}
    exp, par = np.asarray(j["expected_laps"], float), np.asarray(j["par"], float)
    n = len(exp)
    if len(par) != n:  # 保存した期待ラップと par の本数が食い違う予想は差が取れない（保存なしと同じ欠損にする）
        return {"lp_p_slow": by.get("スロー", np.nan), "lp_p_mid": by.get("ミドル", np.nan), "lp_p_high": by.get("ハイ", np.nan),
                "lp_exp_f3_dev": np.nan, "lp_exp_l3_dev": np.nan, "lp_exp_lap0_dev": np.nan}
    dev = exp - par
    return {"lp_p_slow": by.get("スロー", np.nan), "lp_p_mid": by.get("ミドル", np.nan), "lp_p_high": by.get("ハイ", np.nan),
            "lp_exp_f3_dev": float(dev[:3].sum()), "lp_exp_l3_dev": float(dev[max(n - 3, 0):n].sum()), "lp_exp_lap0_dev": float(dev[0])}


def stage_forecast_columns(base: pd.DataFrame, names: list[str], inference_dir: str | Path) -> pd.DataFrame:
    """前段 S3B が要求する EASE2_* / lp_* / pf 列を、本番の保存値（tracking_v2/<日>/<race_id>.json・lap_forecast/<日>/<race_id>.json）から作る。
    写せない列（UNMAPPED）を要求されたら KeyError。保存が無いレースの行は欠損（学習でも約 17% は欠損）。"""
    bad = [n for n in names if n in UNMAPPED]
    if bad:
        raise KeyError(f"本番の保存値から写せない前段の入力列: {bad}（serving_v6.UNMAPPED。写しの実装と過去日の一致確認が要る）")
    root = Path(inference_dir)
    out = pd.DataFrame(index=base.index, columns=names, dtype="float64")
    date = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")
    cache: dict[tuple[str, str], tuple[dict | None, dict | None]] = {}
    for idx, (rid, hid, d) in enumerate(zip(base["race_id"].astype(str), base["horse_id"].astype(str), date)):
        if (rid, d) not in cache:
            cache[(rid, d)] = (_load_json(root / "tracking_v2" / d / f"{rid}.json"), _load_json(root / "lap_forecast" / d / f"{rid}.json"))
        tj, lj = cache[(rid, d)]
        row = {}
        if tj:
            hs = tj.get("horses") or {}
            hs = hs.values() if isinstance(hs, dict) else hs
            h = next((x for x in hs if str(x.get("horse_id")) == hid), {})
            v2, inp = h.get("tracking_v2") or {}, h.get("inputs") or {}
            row.update({k: v2.get(src) for k, src in EASE2_MAP.items()})
            row.update({k: inp.get(src) for k, src in PF_MAP.items()})
        if lj:
            row.update(lap_race_columns(lj))
        for n in names:
            v = row.get(n)
            out.iat[idx, out.columns.get_loc(n)] = np.nan if v is None else v
    return out.astype("float32")
