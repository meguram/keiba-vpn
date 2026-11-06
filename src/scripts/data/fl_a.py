"""最終層 A: 精度派（基準）。較正した確率の上位を買う。対照 = 同じレース・同じ点数で最も人気の券。

python -m src.scripts.data.fl_a
"""

from __future__ import annotations

import json

import numpy as np

from src.scripts.data import fl_core as C


def main() -> None:
    D = C.load()
    print("races", D.R, "horses", len(D.H), flush=True)
    pop_score = np.where(D.mask, -D.pop, -np.inf)
    s_raw = np.where(D.mask, D.lp, -np.inf)
    res: dict = {"prob": {}, "coverage": {}, "theta": {}}

    # 確率の質: 市場・親モデル生・A0（親モデルの較正）・A1（Benter 混合）
    ev = C.races_of(D, C.OOF_YEARS)
    mk = np.where(D.mask, D.q, 0.0)
    res["prob"]["market"] = C.prob_metrics(D, mk)
    res["prob"]["parent_raw"] = C.prob_metrics(D, np.where(D.mask, D.p_win, 0.0))
    sA0, th0 = C.loyo_scores(D, lambda: C.CLogit(["lp"]))
    sA1, th1 = C.loyo_scores(D, lambda: C.CLogit(["lp", "lq"]))
    res["theta"] = {"A0": th0, "A1": th1}

    def soft(s):
        e = np.exp(np.where(np.isfinite(s), s - np.nanmax(np.where(np.isfinite(s), s, -1e18), 1, keepdims=True), -np.inf))
        return e / e.sum(1, keepdims=True)

    pA0, pA1 = np.nan_to_num(soft(np.where(np.isnan(sA0), -np.inf, sA0))), np.nan_to_num(soft(np.where(np.isnan(sA1), -np.inf, sA1)))
    res["prob"]["A0_calibrated_parent"] = C.prob_metrics(D, pA0)
    res["prob"]["A1_benter_mix"] = C.prob_metrics(D, pA1)
    for k, v in res["prob"].items():
        print(f"{k:24s} win_ll {v['win_ll']:.4f} brier {v['brier']:.4f} | 10倍以上 n={v['long10_n']} ll {v['long10_ll']:.4f} Σp {v['long10_sum_p']:.1f} 実勝ち {v['long10_wins']:.0f}", flush=True)

    # 3 着内の取りこぼし（レース単位の一致）
    sc_sets = {"popularity": pop_score, "parent": s_raw, "A1_mix": np.where(np.isnan(sA1), -np.inf, sA1)}
    for K in (3, 4, 5, 6):
        for nm, sc in sc_sets.items():
            res["coverage"][f"{nm}_K{K}"] = C.coverage(D, sc, K)
        print(f"K={K} 上位{K}頭に3着内の3頭が全部入る割合: " + " ".join(f"{nm} {res['coverage'][f'{nm}_K{K}']['all3']:.3f}/{res['coverage'][f'{nm}_K{K}']['ge2']:.3f}" for nm in sc_sets), flush=True)

    # 方策（モデル別 × 券種）
    l3 = np.where(D.mask, np.log(np.clip(D.p_top3, 1e-9, None)), -np.inf)
    models = {"A0": np.where(np.isnan(sA0), -np.inf, sA0), "A1": np.where(np.isnan(sA1), -np.inf, sA1)}
    out = []
    for mn, sc in models.items():
        for bt, k in (("tansho", 1), ("tansho", 3), ("fukusho", 1), ("fukusho", 3)):
            sc_use = sc if bt == "tansho" else (l3 if mn == "A0" else sc)
            tk = C.gen_single(D, sc_use, bt, k, ridx=ev)
            ct = C.gen_single(D, pop_score, bt, k, ridx=ev)
            o = C.summarize(D, f"A/{mn}/{bt}_top{k}", tk, ct, group="A", params={"model": mn, "bt": bt, "k": k})
            out.append(o); print(C.fmt(o), flush=True)
        for bt, K, kind in (("sanrenpuku", 4, "box"), ("sanrenpuku", 5, "box"), ("sanrenpuku", 6, "box"), ("sanrenpuku", 5, "nagashi"), ("sanrenpuku", 6, "nagashi"),
                            ("umaren", 4, "box"), ("wide", 4, "box"), ("wide", 5, "nagashi"), ("umaren", 5, "nagashi")):
            tk = C.gen_bundle(D, sc, bt, K, kind, ridx=ev)
            ct = C.gen_bundle(D, pop_score, bt, K, kind, ridx=ev)
            o = C.summarize(D, f"A/{mn}/{bt}_{kind}{K}", tk, ct, group="A", params={"model": mn, "bt": bt, "K": K, "kind": kind})
            out.append(o); print(C.fmt(o), flush=True)
    res["policies"] = out
    (C.OUT / "stage_A.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    np.save(C.OUT / "score_A0.npy", np.where(np.isfinite(sA0), sA0, np.nan))
    np.save(C.OUT / "score_A1.npy", np.where(np.isfinite(sA1), sA1, np.nan))
    C.save_trials("A")


if __name__ == "__main__":
    main()
