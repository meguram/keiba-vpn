"""tp-59 (1): V5 の LightGBM メンバー（lgb_extras_ped / lgb_extras / lgb_softmax）での騎手・調教師の列の寄与（gain の割合と順位）。

python -m src.scripts.data.tp59_importance --bundle data/local/modeling/training_pipeline/final_v5/bundle_v5_2024 --out <TP59_OUT>/importance.json
関係者の列 = 名前に jockey / trainer を含む列（騎手・調教師の通算成績・件数）。斤量（jockey_weight）は馬の負担重量なので別枠で数える。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np

PAT_RIDER = re.compile(r"(^|_)(jockey|trainer)_(?!weight)")


def group_of(name: str) -> str | None:
    if name == "jockey_weight":
        return "jockey_weight(斤量)"
    if name.startswith("jockey_"):
        return "jockey"
    if name.startswith("trainer_"):
        return "trainer"
    return None


def importance_for(model_dir: Path, targets: tuple[str, ...]) -> dict:
    out = {}
    for tg in targets:
        f = model_dir / f"lgb_{tg}.txt"
        if not f.exists():
            continue
        b = lgb.Booster(model_file=str(f))
        names = b.feature_name()
        gain = b.feature_importance("gain").astype(float)
        tot = gain.sum()
        order = np.argsort(-gain)
        rank = {names[i]: int(r + 1) for r, i in enumerate(order)}
        share = {names[i]: float(gain[i] / tot) for i in range(len(names))}
        grp: dict[str, float] = {}
        cols: dict[str, dict] = {}
        for n in names:
            g = group_of(n)
            if g:
                grp[g] = grp.get(g, 0.0) + share[n]
                cols[n] = {"gain_share": share[n], "rank": rank[n]}
        out[tg] = {"n_features": len(names), "group_share": grp, "columns": cols}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.bundle)
    res: dict = {}
    for member in ("lgb_extras_ped", "lgb_extras", "lgb_softmax"):
        seeds = sorted((root / member).glob("seed*"))
        per = [importance_for(s, ("win", "top2", "top3")) for s in seeds]
        if not per:
            continue
        agg: dict = {}
        for tg in per[0]:
            groups = {}
            for p in per:
                for g, v in p[tg]["group_share"].items():
                    groups.setdefault(g, []).append(v)
            cols = {}
            for p in per:
                for c, v in p[tg]["columns"].items():
                    cols.setdefault(c, {"share": [], "rank": []})
                    cols[c]["share"].append(v["gain_share"])
                    cols[c]["rank"].append(v["rank"])
            agg[tg] = {"n_features": per[0][tg]["n_features"], "seeds": len(per), "group_share_mean": {g: float(np.mean(v)) for g, v in groups.items()},
                       "columns": {c: {"gain_share": float(np.mean(v["share"])), "rank_median": float(np.median(v["rank"]))} for c, v in cols.items()}}
        res[member] = agg
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
