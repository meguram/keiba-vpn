"""tp-69: V5（1,142 列）への増分（LightGBM）の結果を集計する。ホストの <TP69_OUT>/lgb_v3bA・lgb_v3bX の result.json を読む。

python -m src.scripts.data.tp69_report --in <result.json を置いたディレクトリ>   # lgb_v3bA/result.json, lgb_v3bX/result.json
組 A = v3b（下限＋低信頼の重み＋加重 Huber。excusable の除外なし）の 3 列、組 X = A に excusable の除外を足した 7 列。
採用基準: 勝率の 95% 区間の上限 < 0、効果 ≥ 0.0002、連対・複勝の悪化 ≤ 0.0002（feature_bakeoff.compare_oof）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    a = ap.parse_args()
    rows = {}
    for name in ("v3bA", "v3bX"):
        p = Path(a.inp) / f"lgb_{name}" / "result.json"
        if not p.exists():
            print(f"{name}: result.json がありません")
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r["compare"]
        rows[name] = {"added": r["added"], "Δwin": c["Δwin"], "win_hi": c["win_hi"], "Δtop2": c["Δtop2"], "Δtop3": c["Δtop3"], "採用": c["採用"],
                      "per_year": {y: {k: v[k] for k in ("Δwin", "Δtop2", "Δtop3")} for y, v in r["per_year"].items()}}
        print(f"{name}: Δwin={c['Δwin']:.5f}（上限 {c['win_hi']:.5f}） Δtop2={c['Δtop2']:.5f} Δtop3={c['Δtop3']:.5f} 採用={c['採用']}")
        for y, v in rows[name]["per_year"].items():
            print(f"   {y}: Δwin={v['Δwin']:.5f} Δtop2={v['Δtop2']:.5f} Δtop3={v['Δtop3']:.5f}")
    if "v3bA" in rows and "v3bX" in rows:
        print(f"X − A（excusable の除外を足した差）: Δwin={rows['v3bX']['Δwin'] - rows['v3bA']['Δwin']:+.5f}")
    Path(a.inp, "tp69_v5.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
