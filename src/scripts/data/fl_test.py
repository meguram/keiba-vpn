"""最終層 テスト 2025 の 1 回きり評価（設定は fl_a〜fl_g で 2022–24 のみを使って凍結済み）。

- 基礎モデルは 2022–24 で学習し、2025 を採点する。条件セルの補正比も 2022–24 だけから作る。
- top30 のしきい値と、舞台別の選択は 2022–24 の結果（stage_F.json の final）を使う。
- 親のテスト保護ファイル TEST2025_EVALUATED.json とは別に、TEST2025_FL_EVALUATED.json で再実行を拒む。
python -m src.scripts.data.fl_test
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np

from src.scripts.data import fl_b  # noqa: F401  (FEATS["gapL"] の登録)
from src.scripts.data import fl_core as C
from src.scripts.data import fl_e, fl_f
from src.scripts.data.fl_b import Resid, soft
from src.scripts.data.fl_d import NAMES, Alloc

GUARD = C.OUT / "TEST2025_FL_EVALUATED.json"
EXTRA = ["B3_ev1.2", "E1_ev1.2"]


def extra_policies(D, S, ridx):
    pop = np.where(D.mask, -D.pop, -np.inf)
    cand = D.mask & (D.win_odds >= C.LONG_ODDS)
    out = {}
    for nm, key in (("B3_ev1.2", "B3"), ("E1_ev1.2", "E1")):
        p = soft(S[key])
        sel_c = cand & (p * D.win_odds >= 1.2)
        sc = np.where(sel_c, p * D.win_odds, -np.inf)
        out[nm] = (C.gen_single(D, sc, "tansho", 1, cand=sel_c, ridx=ridx), C.gen_single(D, pop, "tansho", 1, sel=sel_c.any(1), ridx=ridx))
    return out


def main() -> None:
    if GUARD.exists():
        print("すでにテスト 2025 を評価済み。再実行しない:", GUARD.read_text()[:200])
        sys.exit(1)
    D = C.load()
    tr = C.races_of(D, C.OOF_YEARS)
    te = C.races_of(D, [C.TEST_YEAR])
    print(f"学習 {len(tr)} レース / テスト {len(te)} レース", flush=True)
    frozen = json.loads((C.OUT / "stage_F.json").read_text(encoding="utf-8"))
    thr = float(frozen["thr_top30"])

    coarse, fine = fl_e.keys(D)
    lrc = fl_e.nested_lrc(D, C.OOF_YEARS, C.TEST_YEAR, coarse, fine)
    C.FEATS["lrc"] = lambda D_, lrc=lrc: lrc

    def full(mdl):
        s = np.full((D.R, D.N), -np.inf)
        s[te] = mdl.score(D, te)
        return s

    models = {
        "A0": C.CLogit(["lp"]).fit(D, tr),
        "A1": C.CLogit(["lp", "lq"]).fit(D, tr),
        "B0": Resid().fit(D, tr),
        "B3": C.CLogit(["gapL"], offset=lambda D_, idx: D_.lq[idx]).fit(D, tr),
        "D1": Alloc(NAMES, 0.001, 0.0, "win").fit(D, tr),
        "E1": C.CLogit(["lp", "lq", "lrc"]).fit(D, tr),
    }
    S = {k: full(m) for k, m in models.items()}
    print("θ:", {k: getattr(m, "theta", None).round(3).tolist() for k, m in models.items()}, flush=True)

    # 確率の質（2025）
    prob = {}
    mk = np.where(D.mask, D.q, 0.0)
    prob["market"] = C.prob_metrics(D, mk, years=(C.TEST_YEAR,))
    prob["parent_raw"] = C.prob_metrics(D, np.where(D.mask, D.p_win, 0.0), years=(C.TEST_YEAR,))
    for k in ("A0", "A1", "B3", "E1"):
        prob[k] = C.prob_metrics(D, soft(S[k]), years=(C.TEST_YEAR,))
    for k, v in prob.items():
        print(f"  {k:12s} win_ll {v['win_ll']:.4f} brier {v['brier']:.4f} | 10倍以上 ll {v['long10_ll']:.4f} Σp {v['long10_sum_p']:.1f}/実 {v['long10_wins']:.0f}", flush=True)

    pol, _ = fl_f.make_policies(D, S, te, thr_top30=thr)
    pol.update(extra_policies(D, S, te))
    res: dict = {"prob": prob, "single": {}, "compose": {}, "thr_top30": thr, "when": datetime.now().isoformat()}
    print("単独方策（テスト 2025、対照 = 同じレース・同じ予算で最も人気の券）", flush=True)
    for nm, (tk, ct) in pol.items():
        o = C.summarize(D, f"T/{nm}", tk, ct, years=(C.TEST_YEAR,), group="T", params={"policy": nm})
        res["single"][nm] = o
        if o.get("empty"):
            print(f"  {nm}: 買いなし"); continue
        print("  " + C.fmt(o).replace("年別差 2025:", "") + f" | レース的中率 {o['race_hit_rate']:.3f}", flush=True)

    GUARD.write_text(json.dumps({"evaluated_at": res["when"], "note": "最終層(fl_*)のテスト 2025 評価。最初の結果を見た時点で書く。再実行しない。親の TEST2025_EVALUATED.json とは別。"}, ensure_ascii=False), encoding="utf-8")

    # 人気薄の的中（B0 の 1 頭買い）
    cand = D.mask & (D.win_odds >= C.LONG_ODDS)
    from src.scripts.data.fl_b import pick_stats
    res["pick"] = {k: pick_stats(D, S[k], cand, te) for k in ("B0", "B3", "E1")}
    for k, v in res["pick"].items():
        print(f"  人気薄の選抜 {k}: 的中 {v['wins']}/市場期待 {v['market_expected']:.1f} (≥10倍 {v['wins_ge10']}/{v['market_expected_ge10']:.1f})", flush=True)

    # 一様な人気薄買い（市場中立の対照）
    uni = C.gen_uniform(D, "tansho", cand, 1.0, ridx=te)
    for nm in ("B0_long1", "B3_ev1.0", "E1_ev1.0"):
        o = C.summarize(D, "tmp", pol[nm][0], uni, years=(C.TEST_YEAR,), log=False, detail=False)
        if not o.get("empty"):
            res["single"][nm]["vs_uniform_long"] = {"ctrl_roi": o["ctrl_roi"], "diff": o["diff"], "diff_ci": o["diff_ci"]}
            print(f"  {nm} 対 一様人気薄買い: 対照 {o['ctrl_roi']:.3f} 差 {o['diff']:+.3f}[{o['diff_ci'][0]:+.3f},{o['diff_ci'][1]:+.3f}]", flush=True)

    # 凍結した舞台別合成
    B, M, MC = fl_f.multiples(D, {k: v for k, v in pol.items() if k in fl_f.POLICIES})
    segs = fl_f.segments(D)
    print("合成（2022–24 で選んだ舞台別方策をそのまま 2025 に適用）", flush=True)
    for sn, seg in segs.items():
        fin = frozen["final"][sn]
        u, m, mc = fl_f.apply(seg, te, fin["choice"], B, M, MC)
        ug, mg, mcg = fl_f.apply(seg, te, {}, B, M, MC, default=fin["best_global"])
        st, sg = fl_f.stats(u, m, mc), fl_f.stats(ug, mg, mcg)
        res["compose"][sn] = {"composed": st, "best_single": sg, "best_global": fin["best_global"]}
        if st.get("empty"):
            continue
        print(f"  {sn:10s} 合成 買い {st['n_bought']:5d} 回収 {st['roi']:.3f}[{st['roi_ci'][0]:.3f},{st['roi_ci'][1]:.3f}] 対照 {st['ctrl_roi']:.3f} 差 {st['diff']:+.3f}[{st['diff_ci'][0]:+.3f},{st['diff_ci'][1]:+.3f}] 最大除き {st['roi_ex_top1']:.3f}/{st['roi_ex_top3']:.3f}"
              f" || 最良単独({fin['best_global']}) 回収 {sg['roi']:.3f} 差 {sg['diff']:+.3f}[{sg['diff_ci'][0]:+.3f},{sg['diff_ci'][1]:+.3f}]", flush=True)

    # 層別（月・芝ダ・会場）
    month = np.array([int(str(d)[5:7]) for d in D.date])
    strata = {"月": month, "会場": np.array([str(x) for x in D.venue]), "芝ダ": np.array([str(x) for x in D.surface])}
    res["strata"] = {}
    for nm in fl_f.POLICIES + EXTRA:
        if nm in fl_f.POLICIES:
            bb, mm, mmc = B[nm], M[nm], MC[nm]
        else:
            tk, ct = pol[nm]
            bb, mm, mmc = (x[nm] for x in fl_f.multiples(D, {nm: pol[nm]}))
        res["strata"][nm] = {}
        for sn, sv in strata.items():
            rows = []
            for c in np.unique(sv[te]):
                k = te[(sv[te] == c) & bb[te]]
                if len(k) >= 20:
                    rows.append((str(c), len(k), float(mm[k].mean()), float((mm[k] - mmc[k]).mean())))
            res["strata"][nm][sn] = {"rows": rows, "pos": sum(1 for r in rows if r[3] > 0), "n": len(rows)}
        s = res["strata"][nm]
        print(f"  層別 {nm:16s} 差>0: 月 {s['月']['pos']}/{s['月']['n']} 会場 {s['会場']['pos']}/{s['会場']['n']} 芝ダ " + " ".join(f"{r[0]}:{r[3]:+.2f}(n{r[1]})" for r in s["芝ダ"]["rows"]), flush=True)

    (C.OUT / "stage_T.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    C.save_trials("T")
    print("保存: stage_T.json, trials_T.json", flush=True)


if __name__ == "__main__":
    main()
