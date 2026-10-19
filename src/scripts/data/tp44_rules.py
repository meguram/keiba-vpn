"""tp-44 (B) 公開されている購入ルールの再現: 「函館の3番人気の単勝」（Zenn 記事は発見期間 105.2% → 検証期間 56.9% と報告）を 2020〜2025 年で見る。

python -m src.scripts.data.tp44_rules --out data/local/modeling/betting/tp44/rules.json
単勝オッズ・払戻は tp36_census と同じ入力（曖昧なレースは除く）。回収率は 1 点 100 円、区間はレース単位ブートストラップ。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.scripts.data.tp36_census import boot_roi, census_horse_bets, load

ROOT = Path(__file__).resolve().parents[3]
HAKODATE = "02"


def rule_table(d: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    sub = d[mask]
    agg = sub.groupby(["race_id", "year"], sort=False).agg(stake=("stake", "sum"), ret=("ret", "sum")).reset_index()
    roi, lo, hi, n = boot_roi(agg)
    row = {"rule": name, "roi": roi, "lo": lo, "hi": hi, "n_bets": n}
    for y in range(2020, 2026):
        a = agg[agg["year"] == y]
        row[str(y)] = float(a["ret"].sum() / a["stake"].sum()) if a["stake"].sum() > 0 else None
        row[f"n{y}"] = int(round(a["stake"].sum() / 100))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/local/modeling/betting/tp44/rules.json")
    args = ap.parse_args()
    w, _, pay = load()
    d = census_horse_bets(w, pay, "tansho")
    venue = d["race_id"].str[4:6]
    rows = [
        rule_table(d, "函館の3番人気の単勝", (venue == HAKODATE) & (d["popularity"] == 3)),
        rule_table(d, "全場の3番人気の単勝", d["popularity"] == 3),
        rule_table(d, "函館の単勝（全人気）", venue == HAKODATE),
        rule_table(d, "17番人気以下の単勝", d["popularity"] >= 17),
        rule_table(d, "1番人気の単勝", d["popularity"] == 1),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(r["rule"], f"ROI={r['roi']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] n={r['n_bets']}", {y: (round(r[y], 3) if r[y] else None) for y in map(str, range(2020, 2026))})


if __name__ == "__main__":
    main()
