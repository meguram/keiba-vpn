"""tp-70 較正後の確率で印（◎○▲★△消）を付け直したときの変化。

python -m src.scripts.data.tp70_marks run --preds tp70_cal_preds.parquet --out DIR

同じレースで『生の確率』『較正後（C3）の確率』のそれぞれから assign_mece_marks / assign_kesi_mark を実行し、印ごとの件数・勝率・3着内率と、
印が変わる割合を比べる。単勝 EV は p_win×確定の単勝オッズ（2026 年は複勝オッズ無しなので複勝 EV は 0）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.race_probabilities import assign_kesi_mark, assign_mece_marks, place_effective_odds

SETS = {"raw": ("p_win", "p_top2", "p_top3"), "C3": ("cal_C3_win", "cal_C3_top2", "cal_C3_top3"), "C6": ("cal_C6_win", "cal_C6_top2", "cal_C6_top3")}
SYM = {"honmei": "◎", "pair": "○", "show_val": "▲", "star": "★", "anchor": "△", "kesi": "消", "none": "-"}


def entries_for(g: pd.DataFrame, cols: tuple[str, str, str]) -> list[dict]:
    es = []
    for r in g.itertuples(index=False):
        po = place_effective_odds(getattr(r, "place_odds_min", None), getattr(r, "place_odds_max", None)) if pd.notna(getattr(r, "place_odds_min", np.nan)) else None
        pw, p2, p3 = (float(getattr(r, c)) for c in cols)
        es.append({"horse_number": int(r.horse_number), "win_prob": pw, "top2_prob": p2, "top3_prob": p3, "win_odds": float(r.win_odds),
                   "ev_win": pw * float(r.win_odds), "ev_place": (p3 * po) if po else 0.0, "_h1": int(r.h1), "_h3": int(r.h3), "_pop": float(r.pop)})
    return es


def run(args) -> None:
    d = pd.read_parquet(args.preds)
    d = d[d["year"] >= 2023]
    res = {}
    for per, ys in (("2023-25", [2023, 2024, 2025]), ("2026", [2026])):
        x = d[d["year"].isin(ys)]
        stat = {k: defaultdict(lambda: [0, 0, 0]) for k in SETS}
        changed = Counter()
        total = 0
        for rid, g in x.groupby("race_id"):
            g = g.sort_values("horse_number")
            marks = {}
            for k, cols in SETS.items():
                es = entries_for(g, cols)
                assign_mece_marks(es)
                assign_kesi_mark(es)
                marks[k] = [e["mark_type"] for e in es]
                for e in es:
                    s = stat[k][e["mark_type"]]
                    s[0] += 1
                    s[1] += e["_h1"]
                    s[2] += e["_h3"]
            total += len(g)
            for k in ("C3", "C6"):
                changed[k] += sum(a != b for a, b in zip(marks["raw"], marks[k]))
        res[per] = {"horses": total, "changed_share": {k: changed[k] / total for k in ("C3", "C6")},
                    "marks": {k: {SYM[m]: {"n": v[0], "win": v[1] / v[0], "top3": v[2] / v[0]} for m, v in sorted(stat[k].items()) if v[0]} for k in SETS}}
        for k in SETS:
            print(per, k, {SYM[m]: (v[0], round(v[1] / v[0], 3), round(v[2] / v[0], 3)) for m, v in sorted(stat[k].items()) if v[0]})
        print(per, "changed", {k: round(changed[k] / total, 4) for k in ("C3", "C6")})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tp70_marks.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--preds", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
