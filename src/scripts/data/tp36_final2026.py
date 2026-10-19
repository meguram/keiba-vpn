"""tp-36 (5) 2026 年（未使用の hold-out）での最終評価。単勝のみ（2026 年は race_result_flat のオッズ・着順しか無い）。

python -m src.scripts.data.tp36_final2026 --out data/local/modeling/betting/tp36/final2026.json
方策は 2023〜2025 年の walk-forward 結果だけから事前に決めて固定し、2026 年を 1 回だけ評価する（しきい値の再調整はしない）。
 A 混合(市場込み) EV>=1.2 均等買い / B 混合 EV>=1.1 均等買い / C 混合 Kelly(1/4, 上限10%, EV>=1.0) / D 市場なし(モデルのみ) EV>=1.5 均等買い
混合の重み (a, b) は 2022〜2025 年で当てはめる（2026 年は使わない）。予測は pred2026（bundle_v4_2025、その日より前の履歴だけ）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data.tp36_edge import blend, boot, fit_benter
from src.scripts.data.tp36_edge2 import kelly_growth
from src.scripts.data.tp36_single import PAY_CAP, drop_broken, process_race, summarize as summarize_single

ROOT = Path(__file__).resolve().parents[3]
TP = ROOT / "data/local/modeling/betting/tp36"


def build_frame2026(pred_dir: Path, result_path: Path) -> pd.DataFrame:
    pr = pd.concat([pd.read_parquet(p) for p in sorted(pred_dir.glob("*.parquet"))], ignore_index=True)
    pr["race_id"], pr["horse_id"] = pr["race_id"].astype(str), pr["horse_id"].astype(str)
    res = pd.read_parquet(result_path, columns=["race_id", "date", "horse_id", "horse_number", "odds", "finish_position"])
    res["race_id"], res["horse_id"] = res["race_id"].astype(str), res["horse_id"].astype(str)
    d = res.merge(pr[["race_id", "horse_id", "p_win", "p_top2", "p_top3"]], on=["race_id", "horse_id"], how="inner")
    d = d.rename(columns={"odds": "win_odds"})
    d["win_odds"] = pd.to_numeric(d["win_odds"], errors="coerce")
    d["fin"] = pd.to_numeric(d["finish_position"], errors="coerce")
    bad = set(d.loc[~(d["win_odds"] >= 1.0), "race_id"])
    inv = (1 / d["win_odds"]).groupby(d["race_id"]).sum()
    bad |= set(inv[~inv.between(1.0, 1.6)].index)         # 単勝オッズが壊れているレース（Σ1/odds が 1.0〜1.6 の外。追走指数などが入っている）
    d = d[~d["race_id"].isin(bad)].copy()
    s = d.groupby("race_id")["p_win"].transform("sum")
    d = d[(s > 0.97) & (s < 1.03)].copy()
    d["win"] = (d["fin"] == 1).astype(float)
    d = d[d.groupby("race_id")["win"].transform("sum") >= 1].copy()
    d["ret_win"] = np.where(d["win"] == 1, d["win_odds"] * 100.0, 0.0)
    d["q_win"] = (1 / d["win_odds"]) / (1 / d["win_odds"]).groupby(d["race_id"]).transform("sum")
    d["year"] = 2026
    d["date"] = d["date"].astype(str)
    d["horse_number"] = pd.to_numeric(d["horse_number"], errors="coerce")
    return d.drop(columns=["fin", "finish_position"]).reset_index(drop=True)


JP_KEYS = {"単勝": "tansho", "複勝": "fukusho", "馬連": "umaren", "ワイド": "wide", "馬単": "umatan", "三連複": "sanrenpuku", "三連単": "sanrentan"}


ARITY = {"tansho": 1, "fukusho": 1, "umaren": 2, "wide": 2, "umatan": 2, "sanrenpuku": 3, "sanrentan": 3}


def parse_payoff_json(raw: str, fix_rows: list[dict] | None = None) -> dict:
    """race_result_flat.payoff（JSON 文字列）→ {bet_type: {組合せ(タプル): 払戻}}。組合せは 'a - b' / 'a → b' / 'a - b - c'。
    複数的中が区切りなしに連結されて壊れている券種（例: ワイド numbers='7 - 97 - 129 - 12'）は、補正行（scraper.payoff_repair）があれば置き換え、
    無ければ out['_broken'] に入れて払戻は載せない。"""
    out: dict = {"_broken": set()}
    try:
        js = json.loads(raw)
    except (TypeError, ValueError):
        return out
    if fix_rows:
        from src.scraper.payoff_repair import apply_fix

        js = apply_fix(js, fix_rows)
    for jp, bt in JP_KEYS.items():
        v = js.get(jp)
        if v is None:
            continue
        for e in (v if isinstance(v, list) else [v]):
            try:
                combo = tuple(int(x) for x in str(e["numbers"]).replace("→", "-").replace(" ", "").split("-"))
                pay = float(str(e["payout"]).replace(",", ""))
            except (KeyError, ValueError):
                out["_broken"].add(bt)
                continue
            if len(combo) != ARITY[bt] or max(combo) > 18 or pay >= PAY_CAP.get(bt, float("inf")):
                out["_broken"].add(bt)
                continue
            out.setdefault(bt, {})[combo] = pay
    return out


def single_2026(new: pd.DataFrame, result_path: Path, alpha_market: tuple[float, float], out_raw: Path, out_json: Path) -> None:
    pj = pd.read_parquet(result_path, columns=["race_id", "payoff"]).drop_duplicates("race_id")
    from src.scraper.payoff_repair import load_fix

    fix = load_fix(2026)
    hits = {str(r.race_id): parse_payoff_json(r.payoff, fix.get(str(r.race_id))) for r in pj.itertuples(index=False)}
    recs, meta = [], []
    for rid, g in new.groupby("race_id", sort=False):
        h = hits.get(str(rid), {})
        broken = h.get("_broken", set())
        rec = process_race(g, {k: v for k, v in h.items() if k != "_broken"}, None, alpha_market)
        recs.append(drop_broken(rec, broken))
        meta.append((rid, str(g["date"].iloc[0]), 2026))
    out_raw.write_bytes(__import__("pickle").dumps({"recs": recs, "meta": meta, "alpha_market": alpha_market}))
    summarize_single(argparse.Namespace(raw=str(out_raw), out=str(out_json)))


def flat(d: pd.DataFrame, mask: pd.Series, n_boot: int = 2000) -> dict:
    sub = d[mask]
    if len(sub) == 0:
        return {"n_bets": 0}
    agg = pd.DataFrame({"race_id": sub["race_id"], "stake": 100.0, "ret": sub["ret_win"]}).groupby("race_id").sum().reset_index()
    roi, lo, hi, n, hit = boot(agg, n_boot=n_boot)
    return {"roi": roi, "lo": lo, "hi": hi, "n_bets": n, "bet_races": int(len(agg)), "race_hit": hit,
            "bet_hit": float((sub["ret_win"] > 0).mean()), "races_total": int(d["race_id"].nunique()),
            "stake_per_race": float(100.0 * n / d["race_id"].nunique())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TP / "final2026.json"))
    ap.add_argument("--frame", default=str(TP / "edge_frame.parquet"))
    ap.add_argument("--pred-dir", default=str(TP / "pred2026"))
    ap.add_argument("--single-raw", default=str(TP / "single_raw.pkl"))
    ap.add_argument("--result", default=str(ROOT / "data/page_reference/tables/2026/race_result_flat.parquet"))
    a = ap.parse_args()
    old = pd.read_parquet(a.frame)
    old = old[old["year"] >= 2022]
    new = build_frame2026(Path(a.pred_dir), Path(a.result))
    ab = fit_benter(old)
    new["p_blend"] = blend(new, *ab)
    ev_b, ev_m = new["p_blend"] * new["win_odds"], new["p_win"] * new["win_odds"]
    out: dict = {"benter_ab": ab, "races": int(new["race_id"].nunique()), "rows": int(len(new)), "days": int(new["date"].nunique()),
                 "date_range": [new["date"].min(), new["date"].max()]}
    s = new[new["win"] == 1]
    out["logloss_win"] = {"model": float(-np.log(s["p_win"].clip(lower=1e-9)).mean()), "market": float(-np.log(s["q_win"].clip(lower=1e-9)).mean()),
                          "blend": float(-np.log(s["p_blend"].clip(lower=1e-9)).mean())}
    pol = {
        "A_mixed_ev1.2_flat": flat(new, ev_b >= 1.2),
        "B_mixed_ev1.1_flat": flat(new, ev_b >= 1.1),
        "D_nomix_model_ev1.5_flat": flat(new, ev_m >= 1.5),
        "ref_mixed_ev1.0_flat": flat(new, ev_b >= 1.0),
        "ref_mixed_ev0.9_flat": flat(new, ev_b >= 0.9),
        "ref_favorite_flat": flat(new, new.groupby("race_id")["win_odds"].transform("min") == new["win_odds"]),
        "ref_all_horses_flat": flat(new, new["win_odds"] > 0),
    }
    out["policies"] = pol
    out["C_mixed_kelly_f0.25_cap0.1_ev1.0"] = kelly_growth(new, "p_blend", 0.25, 0.10, 1.0, years=(2026,))
    old_ev = blend(old.assign(), *ab)
    out["in_sample_reference_2022_2025_mixed_ev1.2"] = flat(old.assign(p_blend=old_ev), old_ev * old["win_odds"] >= 1.2)
    alpha_m = __import__("pickle").loads(Path(a.single_raw).read_bytes())["alpha_market"]
    single_2026(new, Path(a.result), tuple(alpha_m), Path(a.out).with_name("single2026_raw.pkl"), Path(a.out).with_name("single2026.json"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.concat([old, new[old.columns.intersection(new.columns)]], ignore_index=True).to_parquet(Path(a.out).with_name("frame_all.parquet"), index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
