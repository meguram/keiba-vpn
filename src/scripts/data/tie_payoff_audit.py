"""同着（ダッドヒート）と、全券種の払戻の組合せの過不足を洗い出す（読み取り専用）。

python -m src.scripts.data.tie_payoff_audit [--years 2019-2026] [--out data/local/meta/tie_payoff]

- 着順（race_result_flat.finish_position）の同着レースを年別・種類別に数える（1着/2着/3着/4着以下の同着）。
- 払戻（betting/payoff.parquet に payoff_fix を当てたもの）の組合せを、winning_combos（src/pipeline/betting/winning_combos.py）の期待集合と
  レース別・券種別に突き合わせ、欠け（missing）・余り（extra）を JSON に出す。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.pipeline.betting import winning_combos as W
from src.scripts.data import audit_payoff_odds as A

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/meta/tie_payoff"


def tie_kind(groups: list[list[int]]) -> list[str]:
    """同着のあるレースの種類（'1着同着' など）。着順は同着グループの先頭の着順（1 始まりの順位）で数える。"""
    kinds, rank = [], 1
    for g in groups:
        if len(g) > 1:
            kinds.append(f"{rank}着同着" if rank <= 3 else "4着以下同着")
        rank += len(g)
    return kinds


def audit_year(y: int) -> dict:
    D = A.load_year(y)
    rr, wp = D["rr"], D["wp"]
    pay = D["pay"]  # 連結払戻は保存本体へ焼き込み済み（bake_payoff_fix）なので、補正テーブルは当てない
    runners = wp[wp.win_odds > 0].groupby("race_id").size()
    rrp = rr[rr.finish_position > 0]
    brk = {rid: dict(zip(g.horse_number, g.bracket_number)) for rid, g in rr.groupby("race_id")}
    fsize = rr.groupby("race_id").field_size.max().to_dict()
    obs: dict[tuple[str, str], set] = defaultdict(set)
    bad_parse = 0
    for rid, bt, combo in zip(pay.race_id.to_numpy(), pay.bet_type.to_numpy(), pay.combo.to_numpy()):
        c = W.parse_combo(bt, combo)
        if c is None:
            bad_parse += 1
            continue
        obs[(rid, bt)].add(c)
    flag = pay.groupby(["race_id", "bet_type"]).agg(amb=("ambiguous", "max"), unp=("parsed", lambda s: (~s).any())).to_dict("index")
    ties: dict[str, list[str]] = {}
    summary = Counter()
    per_race: dict[str, dict] = {}
    for rid, g in rrp.groupby("race_id"):
        gg = g.sort_values("finish_position")
        groups = [list(x.horse_number) for _, x in gg.groupby("finish_position", sort=True)]
        kinds = tie_kind(groups)
        if kinds:
            ties[rid] = kinds
            for k in kinds:
                summary[f"tie_{k}"] += 1
        n = int(runners.get(rid, 0)) or len(brk.get(rid, {}))
        nf = int(fsize.get(rid, n) or n)
        for bt in W.BET_TYPES:
            exp = W.winning_combos(groups, bt, n_runners=n, brackets=brk.get(rid))
            o = obs.get((rid, bt), set())
            if bt == "fukusho" and o and o != exp:
                if o == W.winning_combos(groups, bt, n_runners=nf):
                    continue  # 登録頭数で発売が決まるケース
            if bt == "wakuren" and not o and not exp:
                continue
            if bt == "wakuren" and not o and n <= 8:
                continue
            d = W.diff_expected(groups, bt, o, n_runners=n, brackets=brk.get(rid)) if (o or exp) else {"missing": [], "extra": []}
            if d["missing"] or d["extra"]:
                f = flag.get((rid, bt), {})
                cause = "ambiguous" if f.get("amb") else ("unparsed" if f.get("unp") else ("whole_bet_missing" if not o else ("rows_missing" if d["missing"] and not d["extra"] else ("rows_extra" if d["extra"] and not d["missing"] else "rows_differ"))))
                per_race.setdefault(rid, {"kinds": kinds, "bets": {}})["bets"][bt] = {"cause": cause, "missing": [list(c) for c in d["missing"]], "extra": [list(c) for c in d["extra"]], "observed": len(o)}
                summary[f"mismatch_{bt}_{cause}"] += 1
    return {"year": y, "races_ranked": int(rrp.race_id.nunique()), "tie_races": len(ties), "tie_by_kind": {k: v for k, v in summary.items() if k.startswith("tie_")},
            "unparseable_rows": bad_parse, "mismatch_counts": {k: v for k, v in summary.items() if k.startswith("mismatch_")},
            "mismatch_races": per_race, "tie_races_list": ties}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2019-2026")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    lo, _, hi = a.years.partition("-")
    years = list(range(int(lo), int(hi or lo) + 1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for y in years:
        if not (A.TABLES / str(y) / "race_result_flat.parquet").exists():
            print(f"{y}: テーブルなし", flush=True)
            continue
        r = audit_year(y)
        rep[str(y)] = r
        print(f"{y}: 着順あり {r['races_ranked']}R / 同着 {r['tie_races']}R {r['tie_by_kind']} / 不整合 {len(r['mismatch_races'])}R {r['mismatch_counts']}", flush=True)
    (out / "tie_payoff_audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    allm = sorted({rid for r in rep.values() for rid in r["mismatch_races"]})
    (out / "mismatch_races.txt").write_text("\n".join(allm) + "\n", encoding="utf-8")
    print(f"不整合レース {len(allm)} 件 → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
