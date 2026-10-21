"""ドリフト監視で PSI を見る特徴量（上位 N 列）を、v5 バンドルの木モデル(勝率)の gain 重要度から決めて config/drift_top_features.json に保存する。

  python -m src.scripts.data.drift_top_features --bundle data/local/modeling/training_pipeline/final_v5/bundle_v5_2025 --top 100

重要度は学習済みモデルだけから計算し、評価データ(2025/2026 の結果)は使わない。各木モデルの gain を合計 1 に正規化して平均する（seed・メンバー等重み）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def lgb_gain(path: Path) -> dict[str, float]:
    import lightgbm as lgb

    b = lgb.Booster(model_file=str(path))
    g = b.feature_importance(importance_type="gain")
    tot = float(sum(g)) or 1.0
    return {n: float(v) / tot for n, v in zip(b.feature_name(), g)}


def xgb_gain(path: Path) -> dict[str, float]:
    import xgboost as xgb

    m = xgb.XGBClassifier(enable_categorical=True)
    m.load_model(str(path))
    sc = m.get_booster().get_score(importance_type="gain")
    tot = float(sum(sc.values())) or 1.0
    return {k: float(v) / tot for k, v in sc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--out", default="config/drift_top_features.json")
    a = ap.parse_args()
    d = Path(a.bundle)
    per_member: dict[str, dict[str, float]] = {}
    for mem in sorted(p for p in d.iterdir() if p.is_dir()):
        accs: list[dict[str, float]] = []
        for seed in sorted(mem.glob("seed*")):
            if (seed / "lgb_win.txt").exists():
                accs.append(lgb_gain(seed / "lgb_win.txt"))
            elif (seed / "xgb_win.json").exists():
                accs.append(xgb_gain(seed / "xgb_win.json"))
        if accs:
            keys = set().union(*accs)
            per_member[mem.name] = {k: sum(x.get(k, 0.0) for x in accs) / len(accs) for k in keys}
    agg: dict[str, float] = defaultdict(float)
    for imp in per_member.values():
        for k, v in imp.items():
            agg[k] += v / len(per_member)
    order = sorted(agg, key=lambda k: -agg[k])[: a.top]
    out = {"source": str(d.name), "members": sorted(per_member), "method": "木モデル(勝率)の gain を正規化して等重み平均", "features": order,
           "importance": {k: round(agg[k], 6) for k in order}}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"members": out["members"], "top10": order[:10]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
