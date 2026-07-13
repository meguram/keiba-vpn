"""
想定めぐ指数 (megu_final) と実測の乖離を評価する CLI。

例:
  DATABASE_URL=... python -m src.scripts.data.eval_megu_predict --year 2025 --sample 500
"""

from __future__ import annotations

import argparse
import random
import statistics as stats

from sqlalchemy import create_engine, text

from src.api.megu_predict_race import build_race_megu_predictions
from src.db.session import get_session, init_engine


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 5:
        return None
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rankx = {rx[i]: i + 1 for i in range(n)}
    ranky = {ry[i]: i + 1 for i in range(n)}
    d = [rankx[i] - ranky[i] for i in range(n)]
    return 1 - 6 * sum(x * x for x in d) / (n * (n * n - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="想定 vs 実測めぐ指数の乖離評価")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--sample", type=int, default=500, help="評価レース数（0=全件）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    init_engine()
    prefix = f"{args.year:04d}"
    engine = create_engine(__import__("os").environ["DATABASE_URL"])

    with engine.connect() as conn:
        race_ids = [
            r[0]
            for r in conn.execute(
                text("""
                    SELECT DISTINCT race_id FROM megu_index
                    WHERE model_version = 'v1' AND computation_status = 'valid'
                      AND race_id LIKE :pfx
                    ORDER BY race_id
                """),
                {"pfx": f"{prefix}%"},
            ).all()
        ]

    if args.sample and len(race_ids) > args.sample:
        rng = random.Random(args.seed)
        race_ids = rng.sample(race_ids, args.sample)

    rows: list[dict] = []
    race_groups: dict[str, list[tuple[float, float]]] = {}

    with get_session() as session:
        for rid in race_ids:
            payload = build_race_megu_predictions(session, rid)
            if not payload:
                continue
            grp: list[tuple[float, float]] = []
            for h in payload["horses"]:
                pred, actual = h.get("megu_final"), h.get("actual_megu")
                if pred is None or actual is None:
                    continue
                rows.append({"pred": pred, "actual": actual, "abs": abs(actual - pred)})
                grp.append((pred, actual))
            if len(grp) >= 5:
                race_groups[rid] = grp

    if not rows:
        print("評価対象なし")
        return

    abs_d = [r["abs"] for r in rows]
    delta = [r["actual"] - r["pred"] for r in rows]
    spearman_race = [
        s for g in race_groups.values() if (s := _spearman([a for a, _ in g], [b for _, b in g])) is not None
    ]

    def pct(p: float) -> float:
        xs = sorted(abs_d)
        k = int(len(xs) * p / 100)
        return xs[min(k, len(xs) - 1)]

    print(f"=== megu predict eval ({prefix}, races={len(race_groups)}, pairs={len(rows)}) ===")
    print(f"MAE:           {stats.mean(abs_d):.2f}")
    print(f"RMSE:          {(stats.mean([d * d for d in delta])) ** 0.5:.2f}")
    print(f"Mean delta:    {stats.mean(delta):+.2f} (actual - pred)")
    print(f"Median |delta|: {stats.median(abs_d):.2f}")
    print(f"P90 |delta|:   {pct(90):.2f}")
    print(f">10pt:         {sum(1 for x in abs_d if x > 10)} ({100 * sum(1 for x in abs_d if x > 10) / len(abs_d):.1f}%)")
    print(f"Within-race Spearman (pred vs actual megu): mean={stats.mean(spearman_race):.3f}")


if __name__ == "__main__":
    main()
