"""ニックスの系統 × 系統（父父 × 母父の父）の 2 期の相関に、帰無（同じ父父・同じ世代の中で母父の父の札を入れ替える 20 回）と
Fisher の z の 95% 区間を足す（記述だけ・未確認）。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.nicks_line_null
"""
from __future__ import annotations

import json

import numpy as np

from src.research.pedigree_lab import data, nicks


def fisher(r: float, n: int) -> list[float]:
    z = np.arctanh(r)
    se = 1 / np.sqrt(max(n - 3, 1))
    return [round(float(np.tanh(z - 1.96 * se)), 3), round(float(np.tanh(z + 1.96 * se)), 3)]


def main() -> None:
    H = nicks.horse_table()
    for m in ("winup", "maxcls", "prize"):
        H[f"{m}_adj"] = nicks.dam_adjust(H, m)
    res = {}
    for m in ("winup", "maxcls", "prize"):
        for ver, col in (("add", m), ("dam_adj", f"{m}_adj")):
            r = nicks.persistence(H, col, "ss", "bms_sire")
            if "pearson" not in r:
                res[f"line_{m}_{ver}"] = r
                continue
            r["fisher_ci"] = fisher(r["pearson"], r["pairs"])
            r["null"] = nicks.null_persistence(H, col, "ss", "bms_sire")
            res[f"line_{m}_{ver}"] = r
            print(f"line_{m}_{ver}", r, flush=True)
    path = data.OUT / "nicks_line_null.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
