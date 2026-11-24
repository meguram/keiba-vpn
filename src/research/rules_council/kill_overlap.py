"""(b) 既存の買い目への消しの段 0（着順なし）: めぐの候補 c13（平場・T-60・EV ≥ 1.3・50 倍以下）の買い目と、RC-2 の消しの 7 水準の重なり。
- 買い目は src/research/megu_cycles/cycles.py の strat_ev（再現 5 本のうち 1 本でも買う行）。2017〜24（2016 は予測が無い）。
- 探す側 70% の一桁（再現 < 10 倍）の買い目の中での重なりの E と割合。着順・払戻は読まない。
- 出力: docs/operations/retrain_trials/rules_council_kill_overlap.json。費用 0 円（手元の表だけ）。
使い方: python -m src.research.rules_council.kill_overlap（nice -n 19）
"""

import json
import numpy as np, pandas as pd
from src.research.megu_cycles import cycles as C, mc_data as D, mc_eval as ME
from src.research.rules_council import rc2 as RC, engine as E


def main() -> None:
    sp = json.load(open(C.CAND / "c13_flat.json")); C.RICH_FILE = D.OUT / sp["probability"]["file"]
    d = C.load_all(); d = d[ME.seg_mask(d, sp["segment"])].reset_index(drop=True)
    ab = tuple(sp["combo_ab_t60"]); st = C.strat_ev(lambda x, k: C.combo_p(x, k, *ab, col="p_rich"), sp["t"], sp["omax"])
    d["stake"] = np.vstack([st(d, k) for k in range(D.N_PRE)]).mean(0)
    b = d[d.stake > 0][["race_id", "horse_id", "stake", "year"]].astype({"race_id": str, "horse_id": str})
    # 全走の表の帯・探す側
    full = pd.read_parquet(E.F.OUTW / "full.parquet", columns=["race_id", "horse_id", "kaisai", "q"])
    full["race_id"], full["horse_id"] = full["race_id"].astype(str), full["horse_id"].astype(str)
    full["pre"] = np.load(E.F.OUTW / "full_pre.npy")[0]
    full["srch"] = E.search70(full["kaisai"].to_numpy())
    x = b.merge(full, on=["race_id", "horse_id"], how="left")
    R = RC.RC2(0, mode="k70")
    s = json.loads(open(E.F.OUTW / "rc2_search.json", encoding="utf-8").read())
    nm = {R.P.level_name(j): j for j in range(R.P.n_levels)}
    sel = np.zeros(R.P.n_levels, bool)
    for n in [r["level"] for r in s["down1"]]:
        sel[nm[n]] = True
    hit = np.where(R.flat >= 0, sel[np.maximum(R.flat, 0)], False).any(1)
    kill = set(zip(R.P.d["race_id"].astype(str)[hit], R.P.d["horse_id"].astype(str)[hit]))
    x["kill"] = [(a, c) in kill for a, c in zip(x["race_id"], x["horse_id"])]
    fav = x["srch"] & (x["pre"] < 10)
    out = {"megu_c13_bets_2017_24": int(len(x)), "bets_search_side": int(x["srch"].sum()), "bets_search_fav": int(fav.sum()),
           "E_search_fav": float(x.loc[fav, "q"].sum()), "overlap_runs": int((fav & x["kill"]).sum()), "overlap_E": float(x.loc[fav & x["kill"], "q"].sum()),
           "overlap_share_of_fav_bets": float((fav & x["kill"]).sum() / max(fav.sum(), 1)), "overlap_share_of_all_search_bets": float((fav & x["kill"]).sum() / max(x["srch"].sum(), 1))}
    w = x["stake"].astype(float)
    out["weighted"] = {"fav_bets": float(w[fav].sum()), "overlap": float(w[fav & x["kill"]].sum()),
                       "overlap_share_of_fav_bets": float(w[fav & x["kill"]].sum() / max(w[fav].sum(), 1e-9)),
                       "overlap_E": float((w * x["q"])[fav & x["kill"]].sum())}
    share = out["overlap_share_of_fav_bets"]
    out["note"] = (f"線（1 号の判断）: 重なりの E ≥ 200 かつ 一桁の買い目に占める割合 ≤ 50%。割合は {share:.1%}（行の和集合）・"
                   f"{out['weighted']['overlap_share_of_fav_bets']:.1%}（5 本の買う割合の重み）で、{'外れ' if share > 0.5 else '通り'}。"
                   "買い目の数は、再現 5 本のうち 1 本でも買う行の数")
    (E.F.CR.REPO / "docs/operations/retrain_trials/rules_council_kill_overlap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
