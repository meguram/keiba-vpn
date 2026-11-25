"""RC-3 の候補の列: その日の馬場の偏り（同じ場・同じ日・同じ芝ダで、T-60 より前に結果が出たレースの 3 着内の枠・位置取り）。

- T-60 に分かる結果: 発走が「今回の発走 − 75 分」以前のレース（本番は発走 +15 分に速報 race_result_on_time を取る。
  src/scraper/auto_scrape.py の RESULT_OFFSET_MINUTES・src/pipeline/inference/v3_serving.py の on_time_done）。
- 前のレースの 3 着内の馬の 枠（1〜4 を内）・最後の角の位置（passing_order の最後の数字 ≤ 3 を前）から、その日の偏りの水準を作る。
  今回のレースの着順は使わない（作るのは前のレースの結果だけ）。
- 出力: <COURSE_OUT>/rules_council/daybias_race.parquet（レースごと: 前のレースの数・3 着内の内枠と前の数・その期待の数）。費用 0 円（手元の表だけ）。
  R0 の資料の水準の大きさの表は、この表と frame の枠・先行率から作った（担当の一時の計算。RC-3 は R0 で閉じたので道具に入れていない）。
使い方: python -m src.research.rules_council.daybias（表）・daybias bound（RC-3 の上限の計算）。nice -n 19
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.rules_council import rc_frame as F

LAG_MIN = 75
TABLES = F.CR.REPO / "data/page_reference/tables"


def last_corner(s) -> float:
    try:
        x = [int(v) for v in str(s).split("-") if v.strip().isdigit()]
        return float(x[-1]) if x else np.nan
    except ValueError:
        return np.nan


def build() -> pd.DataFrame:
    parts = []
    for y in range(2016, F.YEAR_END + 1):
        r = pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet",
                            columns=["race_id", "date", "venue", "surface", "start_time", "finish_position", "bracket_number", "passing_order"])
        parts.append(r)
    r = pd.concat(parts, ignore_index=True)
    r["race_id"] = r["race_id"].astype(str)
    r["surf"] = r["surface"].astype(str).str[0]
    r = r[r["surf"].isin(["芝", "ダ"])]
    t = pd.to_datetime(r["start_time"].astype(str), format="%H:%M", errors="coerce")
    r["min"] = t.dt.hour * 60 + t.dt.minute
    r["fin"] = pd.to_numeric(r["finish_position"], errors="coerce")
    r["inner"] = (pd.to_numeric(r["bracket_number"], errors="coerce") <= 4).astype(float)
    r["front"] = (r["passing_order"].map(last_corner) <= 3).astype(float)
    top = r[r["fin"] <= 3]
    race = r.drop_duplicates("race_id")[["race_id", "date", "venue", "surf", "min"]]
    tr = top.groupby("race_id").agg(n3=("fin", "size"), in3=("inner", "sum"), fr3=("front", "sum")).reset_index()
    # 期待の数（偏りが無いときの 3 着内の内枠・前の数）: 3 × 内枠の頭数の割合、3 × 3 ÷ 頭数（前は最後の角の 3 番手まで）
    allr = r.groupby("race_id").agg(field=("fin", "size"), in_share=("inner", "mean")).reset_index()
    allr["in3_exp"] = 3 * allr["in_share"]
    allr["fr3_exp"] = 3 * np.minimum(3, allr["field"]) / allr["field"]
    race = race.merge(tr, on="race_id", how="left").fillna({"n3": 0, "in3": 0, "fr3": 0}).merge(allr[["race_id", "in3_exp", "fr3_exp"]], on="race_id", how="left")
    out = []
    for _, g in race.groupby(["date", "venue", "surf"]):
        g = g.sort_values("min")
        m = g["min"].to_numpy()
        n3, i3, f3 = g["n3"].to_numpy(), g["in3"].to_numpy(), g["fr3"].to_numpy()
        ie, fe = g["in3_exp"].to_numpy(), g["fr3_exp"].to_numpy()
        for k in range(len(g)):
            ok = m <= m[k] - LAG_MIN
            out.append((g["race_id"].iloc[k], int(ok.sum()), float(n3[ok].sum()), float(i3[ok].sum()), float(f3[ok].sum()),
                        float(ie[ok].sum()), float(fe[ok].sum())))
    D = pd.DataFrame(out, columns=["race_id", "prior_races", "prior_top3", "prior_inner3", "prior_front3", "prior_inner3_exp", "prior_front3_exp"])
    return D


def cmd_bound() -> dict:
    """RC-3 の条件 3（上限の計算）。市場と組まない（φ = 0 の上限）。偏りの信頼度・日の中の持続・直前 2 レースの偏りの後の倍率（3 着内と 1 着）。
    2016〜24 の全部の着順を、偏りの持続の記述にだけ使う（市場比・回収率は出さない）。結果は docs/operations/retrain_trials/rules_council_rc3_bound.json。
    - 走った馬だけ（着順 ≥ 1。中止・除外・取消を除く）で、頭数・内枠の数・3 着内を数える。
    - 倍率の基準は「ふだん」: 内外は 場 × 芝ダ の常の内枠の上積み（e の平均）を足した期待、前後は 場 × 芝ダ × 頭数の群 の平均。
    - 有利な側と逆の側の両方を、3 着内と 1 着で出す（逆の側は、内外は外枠、前後は前以外）。"""
    import json
    parts = [pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet",
                             columns=["race_id", "date", "venue", "surface", "start_time", "finish_position", "bracket_number", "passing_order"])
             for y in range(2016, F.YEAR_END + 1)]
    r = pd.concat(parts, ignore_index=True)
    r["surf"] = r["surface"].astype(str).str[0]
    r["fin"] = pd.to_numeric(r["finish_position"], errors="coerce")
    r = r[r["surf"].isin(["芝", "ダ"]) & (r["fin"] >= 1)].copy()
    r["inner"] = (pd.to_numeric(r["bracket_number"], errors="coerce") <= 4).astype(float)
    r["front"] = (r["passing_order"].map(last_corner) <= 3).astype(float)
    r["top"] = (r["fin"] <= 3).astype(float)
    r["w"] = (r["fin"] == 1).astype(float)
    g = r.groupby("race_id")
    R = pd.DataFrame({"date": g["date"].first(), "venue": g["venue"].first(), "surf": g["surf"].first(), "st": g["start_time"].first(),
                      "N": g["fin"].size(), "K": g["inner"].sum(),
                      "in3": g.apply(lambda x: (x["inner"] * x["top"]).sum()), "fr3": g.apply(lambda x: (x["front"] * x["top"]).sum()),
                      "n3": g["top"].sum(), "inw": g.apply(lambda x: (x["inner"] * x["w"]).sum()), "frw": g.apply(lambda x: (x["front"] * x["w"]).sum()),
                      "nw": g["w"].sum()})
    R = R[(R["n3"] == 3) & (R["nw"] == 1)].copy()                  # 3 着内が 3 頭・1 着が 1 頭のレースだけ（同着を除く）
    R["e_in"] = R["in3"] - 3 * R["K"] / R["N"]
    R["v_in"] = 3 * (R["K"] / R["N"]) * (1 - R["K"] / R["N"]) * (R["N"] - 3) / (R["N"] - 1)
    R["in_base"] = R.groupby(["venue", "surf"])["e_in"].transform("mean")           # 場 × 芝ダ の常の内枠の上積み
    R["inw_exp"] = R["K"] / R["N"] + R.groupby(["venue", "surf"])["inw"].transform("mean") - (R["K"] / R["N"]).groupby([R["venue"], R["surf"]]).transform("mean")
    R["ng"] = np.searchsorted([8, 13], R["N"].to_numpy(), side="right")
    R["fr_base"] = R.groupby(["venue", "surf", "ng"])["fr3"].transform("mean")
    R["frw_base"] = R.groupby(["venue", "surf", "ng"])["frw"].transform("mean")
    R["e_fr"] = R["fr3"] - R["fr_base"]
    R["e_in_d"] = R["e_in"] - R["in_base"]
    R = R.sort_values(["date", "venue", "surf", "st"])
    R["k"] = R.groupby(["date", "venue", "surf"]).cumcount()
    R["n_day"] = R.groupby(["date", "venue", "surf"])["k"].transform("size")
    out = {"rho_race_inner": float(1 - R["v_in"].mean() / R["e_in_d"].var()), "var_obs_inner": float(R["e_in_d"].var()),
           "var_null_inner": float(R["v_in"].mean()), "races_used": int(len(R)),
           "note_rho": "観測の分散は 場 × 芝ダ の常の上積みを引いた後。奇数・偶数の相関は Spearman–Brown で直していない（日の半分どうしの相関）"}
    D = R[R["n_day"] >= 4].copy()
    for col, side in (("e_in_d", "inner"), ("e_fr", "front")):
        a = D[D["k"] % 2 == 0].groupby(["date", "venue", "surf"])[col].sum()
        b = D[D["k"] % 2 == 1].groupby(["date", "venue", "surf"])[col].sum()
        j = pd.concat([a, b], axis=1).dropna()
        D["prev2"] = D.groupby(["date", "venue", "surf"])[col].transform(lambda s_: s_.shift(1).rolling(2).sum())
        y = D.dropna(subset=["prev2"])
        o = {"split_half_corr": float(j.corr().iloc[0, 1]), "days": int(len(j)), "corr_prev2_next": float(np.corrcoef(y["prev2"], y[col])[0, 1]), "races": int(len(y))}
        for thr in (2, 3):
            hi, lo = y[y["prev2"] >= thr], y[y["prev2"] <= -thr]
            if side == "inner":
                ex_t = 3 * hi.K / hi.N + hi.in_base
                fav3, fav1 = float(hi.in3.sum() / ex_t.sum()), float(hi.inw.sum() / hi.inw_exp.sum())
                ex_o = 3 * (1 - lo.K / lo.N) - lo.in_base
                oth3, oth1 = float((3 - lo.in3).sum() / ex_o.sum()), float((1 - lo.inw).sum() / (1 - lo.inw_exp).sum())
            else:
                fav3, fav1 = float(hi.fr3.sum() / hi.fr_base.sum()), float(hi.frw.sum() / hi.frw_base.sum())
                oth3, oth1 = float((3 - lo.fr3).sum() / (3 - lo.fr_base).sum()), float((1 - lo.frw).sum() / (1 - lo.frw_base).sum())
            o[f"ge_{thr}"] = {"n": int(len(hi)), "top3_mult": fav3, "win_mult": fav1}
            o[f"le_-{thr}"] = {"n": int(len(lo)), "top3_mult_other_side": oth3, "win_mult_other_side": oth1}
        out[side] = o
    mults = [v for k in ("inner", "front") for kk in out[k] if kk.startswith(("ge_", "le_")) for m_, v in out[k][kk].items() if "mult" in m_]
    out["max_mult"] = float(max(mults))
    out["note"] = ("直前の 2 レース（75 分の遅れを入れない）で作った楽観の上限。市場は使わない（φ = 0）。側の全体の平均で、部分集合（例 内枠 × 先行型）の上限ではない。"
                   "前後の基準は 場 × 芝ダ × 頭数の群 の全期間の平均（先の期間を含む。楽観側）")
    p_ = F.CR.REPO / "docs/operations/retrain_trials/rules_council_rc3_bound.json"
    p_.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main() -> None:
    import sys
    if sys.argv[1:] == ["bound"]:
        cmd_bound()
        return
    D = build()
    D.to_parquet(F.OUTW / "daybias_race.parquet", index=False)
    print("races", len(D), "prior>=2", float((D["prior_races"] >= 2).mean()), "prior_top3>=6", float((D["prior_top3"] >= 6).mean()))


if __name__ == "__main__":
    main()
