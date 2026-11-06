"""最終層 G: 全試行への Reality Check と、月・会場・芝ダ別の層別（2022–24、テスト 2025 は使わない）。

python -m src.scripts.data.fl_g
"""

from __future__ import annotations

import json

import numpy as np

from src.scripts.data import fl_core as C
from src.scripts.data.fl_f import POLICIES
from src.scripts.data.tp36_rc import reality_check


def main() -> None:
    D = C.load()
    ev = C.races_of(D, C.OOF_YEARS)
    names, diff, prof = [], [], []
    for st in "ABCDE":
        z = np.load(C.OUT / f"series_{st}.npz")
        for k in z.files:
            p, cd = z[k][0].astype(float), z[k][1].astype(float)
            if p.std() < 1e-12:
                continue
            names.append(f"{st}:{k}"); prof.append(p); diff.append(p - cd)
    P, Dm = np.stack(prof, 1), np.stack(diff, 1)
    # 賭け金の規模が違う方策を同列に比べるため、レースごとの利益差は標準偏差で割った t 値（reality_check 内）で比較する
    out: dict = {"n_settings": len(names)}
    rc_d = reality_check(Dm, n_boot=2000, seed=1)
    rc_p = reality_check(P, n_boot=2000, seed=2)
    out["rc_diff_vs_control"] = {"n": rc_d["n_strategies"], "t_max": rc_d["t_max"], "best": names[rc_d["best"]], "p_value": rc_d["p_value"]}
    out["rc_profit_gt0"] = {"n": rc_p["n_strategies"], "t_max": rc_p["t_max"], "best": names[rc_p["best"]], "p_value": rc_p["p_value"]}
    for tag, rc, M_ in (("対照との差", rc_d, Dm), ("利益>0(回収率>100%)", rc_p, P)):
        order = np.argsort(-np.array(rc["t"]))[:8]
        print(f"[{tag}] 試行数 {rc['n_strategies']}  最大t {rc['t_max']:.2f}  Reality Check p = {rc['p_value']:.4f}  最良: {names[rc['best']]}", flush=True)
        for j in order:
            print(f"    t={rc['t'][j]:+.2f} {names[j]}", flush=True)
        out["top_" + ("diff" if M_ is Dm else "profit")] = [(names[j], float(rc["t"][j])) for j in order]
    # 『回収率>100%』の p が小さくても対照との差の p が大きい、などの確認用に族ごとの p
    fam = {}
    for st in "ABCDE":
        cols = [i for i, n in enumerate(names) if n.startswith(st + ":")]
        r1 = reality_check(Dm[:, cols], n_boot=1000, seed=3)
        r2 = reality_check(P[:, cols], n_boot=1000, seed=4)
        fam[st] = {"n": len(cols), "p_diff": r1["p_value"], "best_diff": names[cols[r1["best"]]], "p_profit": r2["p_value"], "best_profit": names[cols[r2["best"]]]}
        print(f"  族{st}: 試行 {len(cols)} 差のp {r1['p_value']:.4f}({names[cols[r1['best']]]}) 利益>0のp {r2['p_value']:.4f}({names[cols[r2['best']]]})", flush=True)
    out["family"] = fam

    # 層別
    z = np.load(C.OUT / "mult_F.npz")
    b, m, mc = ({p: z[f"{k}_{p}"] for p in POLICIES} for k in ("b", "m", "mc"))
    month = np.array([int(str(d)[5:7]) for d in D.date])
    strata = {"月": month, "会場": np.array([str(x) for x in D.venue]), "芝ダ": np.array([str(x) for x in D.surface])}
    out["strata"] = {}
    for p in POLICIES:
        out["strata"][p] = {}
        for sn, sv in strata.items():
            rows = []
            for c in np.unique(sv[ev]):
                k = ev[(sv[ev] == c) & b[p][ev]]
                if len(k) < 30:
                    continue
                rows.append((str(c), len(k), float(m[p][k].mean()), float((m[p][k] - mc[p][k]).mean())))
            pos = sum(1 for r in rows if r[3] > 0)
            out["strata"][p][sn] = {"rows": rows, "pos": pos, "n": len(rows)}
        s = out["strata"][p]
        print(f"  層別 {p:16s} 差>0 の層: 月 {s['月']['pos']}/{s['月']['n']} 会場 {s['会場']['pos']}/{s['会場']['n']} 芝ダ " + " ".join(f"{r[0]}:{r[3]:+.2f}(n{r[1]})" for r in s["芝ダ"]["rows"]), flush=True)
    (C.OUT / "stage_G.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
