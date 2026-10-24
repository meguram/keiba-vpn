"""tp-70 市場補正（KEIBA_PROB_CALIBRATION=1）を入れたときの下流の再当てはめ（二重補正を避ける）。

python -m src.scripts.data.tp70_pc ev   --preds tp70_cal_preds.parquet --out config/ev_calibration_pc.json   # 補正後の確率から作った EV の単調回帰（単勝・複勝）
python -m src.scripts.data.tp70_pc star --preds tp70_cal_preds.parquet --out DIR                             # ★の EV しきい値の再設計（補正後の確率）
python -m src.scripts.data.tp70_pc kesi --preds tp70_cal_preds.parquet --out DIR                             # 消印のしきい値の再設計（補正後の確率）

preds は tp70_fit compare の出力（前年までで当てはめた補正後の確率 cal_C3_win / cal_C6_top2 / cal_C6_top3 を含む）。
勝率は C3、連対・複勝は C6 を使う（config/prob_calibration.json の default_method と同じ）。2026 年は最終確認だけで、しきい値・折れ線の当てはめには使わない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from src.scripts.data.calibrate_ev import fit_isotonic_knots
from src.utils.ev_calibration import apply_spec, load_config

WIN, T2, T3 = "cal_C3_win", "cal_C6_top2", "cal_C6_top3"


def cp_upper(h: int, n: int, alpha: float = 0.05) -> float:
    return 1.0 if n <= 0 else (float(beta.ppf(1 - alpha / 2, h + 1, n - h)) if n > h else 1.0)


def load(preds: str) -> pd.DataFrame:
    d = pd.read_parquet(preds)
    d = d[d["year"] >= 2023].copy()
    d["po"] = d["place_odds_min"] + 0.3 * (d["place_odds_max"] - d["place_odds_min"])
    d["ev_win_pc"] = d[WIN] * d["win_odds"]
    d["ev_place_pc"] = d[T3] * d["po"]
    d["real_win"] = d["ret_win"] / 100.0
    d["real_place"] = d["ret_place"] / 100.0
    d["pop_rank"] = d.groupby("race_id")["win_odds"].rank(method="first")
    return d


def gap_stats(ev_cal: np.ndarray, real: np.ndarray, bins: int = 10) -> dict:
    q = pd.qcut(pd.Series(ev_cal), bins, duplicates="drop", labels=False)
    g = pd.DataFrame({"e": ev_cal, "r": real, "q": q.to_numpy()}).groupby("q").agg(e=("e", "mean"), r=("r", "mean"), n=("e", "size"))
    return {"gap": float(real.mean() - ev_cal.mean()), "ece": float((g["n"] * (g["e"] - g["r"]).abs()).sum() / g["n"].sum())}


def ev_cmd(a) -> None:
    d = load(a.preds)
    cfg = json.loads(json.dumps(load_config()))
    out = json.loads(json.dumps(cfg))
    out["version"] = 3
    out["label"] = "期待回収率（市場補正後の確率・較正後）"
    out["note"] = ("市場補正した確率（KEIBA_PROB_CALIBRATION=1）から作った EV に当てはめ直した版（tp-70）。補正した確率のレースだけが使う（active_config）。"
                   "win・place は補正後の確率の EV で再当てはめ。top2 と推奨ポートフォリオの cand_* は従来のまま（連対 EV の実現が無い／ポートフォリオの再実行が必要。calibrate_ev build_portfolio を補正後の確率で再実行して差し替える）。")
    out["fit"] = {**(cfg.get("fit") or {}), "pc_horse": "2023-2025（前年までで当てはめた補正後の確率。勝率=C3・複勝=C6）。単勝は 2026 を最終確認のみに使用"}
    res = {}
    # 単勝: 2023〜2025 年で当てはめ、2026 年で確認。leave-one-year-out も出す
    tr = d[d["year"].between(2023, 2025)]
    spec = fit_isotonic_knots(tr["ev_win_pc"].to_numpy(), tr["real_win"].to_numpy())
    res["win"] = {"fit": gap_stats(apply_spec(spec, tr["ev_win_pc"].to_numpy()), tr["real_win"].to_numpy())}
    te = d[d["year"] == 2026]
    res["win"]["check_2026"] = gap_stats(apply_spec(spec, te["ev_win_pc"].to_numpy()), te["real_win"].to_numpy())
    res["win"]["check_2026_uncalibrated"] = gap_stats(te["ev_win_pc"].to_numpy(), te["real_win"].to_numpy())
    loyo = []
    for y in (2023, 2024, 2025):
        s = fit_isotonic_knots(d[d["year"].between(2023, 2025) & (d["year"] != y)]["ev_win_pc"].to_numpy(), d[d["year"].between(2023, 2025) & (d["year"] != y)]["real_win"].to_numpy())
        t = d[d["year"] == y]
        loyo.append({"year": y, **gap_stats(apply_spec(s, t["ev_win_pc"].to_numpy()), t["real_win"].to_numpy())})
    res["win"]["loyo"] = loyo
    out["kinds"]["win"] = {**spec, "validated": True, "n": int(len(tr)), "gap_2026": round(res["win"]["check_2026"]["gap"], 4), "ece_2026": round(res["win"]["check_2026"]["ece"], 4)}
    # 複勝: 払戻のある 2023〜2025 年のみ。leave-one-year-out で確認
    pl = d[d["year"].between(2023, 2025) & d["ev_place_pc"].notna() & d["real_place"].notna()]
    spec_p = fit_isotonic_knots(pl["ev_place_pc"].to_numpy(), pl["real_place"].to_numpy())
    res["place"] = {"fit": gap_stats(apply_spec(spec_p, pl["ev_place_pc"].to_numpy()), pl["real_place"].to_numpy()), "uncalibrated": gap_stats(pl["ev_place_pc"].to_numpy(), pl["real_place"].to_numpy())}
    loyo = []
    for y in (2023, 2024, 2025):
        s = fit_isotonic_knots(pl[pl["year"] != y]["ev_place_pc"].to_numpy(), pl[pl["year"] != y]["real_place"].to_numpy())
        t = pl[pl["year"] == y]
        loyo.append({"year": y, **gap_stats(apply_spec(s, t["ev_place_pc"].to_numpy()), t["real_place"].to_numpy())})
    res["place"]["loyo"] = loyo
    out["kinds"]["place"] = {**spec_p, "validated": True, "n": int(len(pl)), "note": "2026 年は複勝の払戻が無く確認できない"}
    out["marks"] = json.loads(Path(a.marks).read_text()) if a.marks and Path(a.marks).exists() else {}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])


def star_cmd(a) -> None:
    """★の EV しきい値（単勝 EV の下限 w、複勝 EV の下限 w+0.08）の再設計。補正後の確率で ◎○▲ を付けたあとの残りの馬から選ぶ（従来と同じ手順）。"""
    from src.utils import race_probabilities as RP

    d = load(a.preds)
    rows = []
    sel = d[d["race_id"].isin(set(d["race_id"].unique()))]
    base_w, base_p = RP._STAR_EV_WIN_MIN, RP._STAR_EV_PLACE_MIN
    grid = [0.80, 0.85, 0.90, 0.95, 1.00]
    stats = {w: {} for w in grid}
    n_races = {"2023-25": d[d["year"] <= 2025]["race_id"].nunique(), "2026": d[d["year"] == 2026]["race_id"].nunique()}
    for rid, g in sel.groupby("race_id"):
        g = g.sort_values("horse_number")
        per = "2026" if int(g["year"].iloc[0]) == 2026 else "2023-25"
        for w in grid:
            RP._STAR_EV_WIN_MIN, RP._STAR_EV_PLACE_MIN = w, w + 0.08
            es = []
            for r in g.itertuples(index=False):
                po = RP.place_effective_odds(r.place_odds_min, r.place_odds_max) if pd.notna(r.place_odds_min) else None
                pw, p2, p3 = float(getattr(r, WIN)), float(getattr(r, T2)), float(getattr(r, T3))
                es.append({"horse_number": int(r.horse_number), "win_prob": pw, "top2_prob": p2, "top3_prob": p3, "win_odds": float(r.win_odds), "ev_win": pw * float(r.win_odds),
                           "ev_place": (p3 * po) if po else 0.0, "_h1": int(r.h1), "_h3": int(r.h3), "_ret": float(r.ret_win) / 100.0})
            RP.assign_mece_marks(es)
            for e in es:
                if e["mark_type"] == "star":
                    s = stats[w].setdefault(per, [0, 0, 0, 0.0])
                    s[0] += 1
                    s[1] += e["_h1"]
                    s[2] += e["_h3"]
                    s[3] += e["_ret"]
    RP._STAR_EV_WIN_MIN, RP._STAR_EV_PLACE_MIN = base_w, base_p
    for w in grid:
        for per, s in stats[w].items():
            rows.append({"star_ev_win_min": w, "star_ev_place_min": round(w + 0.08, 2), "period": per, "stars": s[0], "per_race": round(s[0] / n_races[per], 3), "win_rate": round(s[1] / s[0], 4), "top3_rate": round(s[2] / s[0], 4), "win_roi": round(s[3] / s[0], 4)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tp70_star.json").write_text(json.dumps({"rows": rows, "races": n_races}, ensure_ascii=False, indent=1))
    print(pd.DataFrame(rows).to_string(index=False))


def kesi_cmd(a) -> None:
    """消印（10 人気以下）の補正後の複勝確率のしきい値。期間別に Clopper-Pearson の上側限界を見る。"""
    d = load(a.preds)
    d["n"] = d.groupby("race_id")["horse_id"].transform("size")
    x = d[(d["pop_rank"] >= 10) & (d["n"] >= 8)]
    per_of = {2023: "設計2023", 2024: "確認2024", 2025: "参考2025", 2026: "最終2026"}
    races = d.groupby("year")["race_id"].nunique()
    rows = []
    for th in (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10):
        for y, nm in per_of.items():
            g = x[(x["year"] == y) & (x[T3] <= th)]
            h, n = int(g["h3"].sum()), len(g)
            rows.append({"p3_max_pc": th, "period": nm, "n": n, "per_race": round(n / races[y], 3), "rate": round(h / n, 4) if n else None, "cp_upper": round(cp_upper(h, n), 4)})
    # 従来（補正なし・生の確率 8%）との頭数比較
    raw = []
    for y, nm in per_of.items():
        g = x[(x["year"] == y) & (x["p_top3"] <= 0.08)]
        raw.append({"period": nm, "n": len(g), "per_race": round(len(g) / races[y], 3), "rate": round(float(g["h3"].mean()), 4), "cp_upper": round(cp_upper(int(g["h3"].sum()), len(g)), 4)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tp70_kesi.json").write_text(json.dumps({"calibrated": rows, "raw_p3_le_0.08": raw}, ensure_ascii=False, indent=1))
    print(pd.DataFrame(rows).pivot_table(index="p3_max_pc", columns="period", values=["n", "rate", "cp_upper"]).round(4).to_string())
    print(pd.DataFrame(raw).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("ev", ev_cmd), ("star", star_cmd), ("kesi", kesi_cmd)):
        p = sp.add_parser(name)
        p.add_argument("--preds", required=True)
        p.add_argument("--out", required=True)
        if name == "ev":
            p.add_argument("--marks", default=None, help="★のしきい値などの JSON（star の結果から作る）")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
