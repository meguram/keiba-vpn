"""推奨馬券の『プラン』（券種集合×点数上限×M×k×τ）ごとの過去成績を、同じ候補表で一括評価する。

python -m src.scripts.data.portfolio_plans_eval           # config/target_bets.json の plans を評価し plans_report.json に保存
2022〜2024 年は設定の選択に使った期間、2025 年は検証。回収率の区間はレース単位のブートストラップ。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.scripts.data import portfolio_target_eval as E

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "target_bets.json"


def plan_tuple(p: dict) -> tuple:
    name = f"P_{p['id']}"
    E.TYPESETS[name] = set(p["bt"])
    return (name, p.get("max_bets"), 5.0, float(p["k"]), float(p["tau"]))


def main() -> int:
    plans = [p for p in json.loads(CONFIG.read_text(encoding="utf-8"))["plans"] if p["kind"] == "greedy"]
    races = E.load_races(E.YEARS)
    by_year = {y: [r for r in races if r["year"] == y] for y in E.YEARS}
    pooled = [r for y in E.SEL_YEARS for r in by_year[y]]
    out = {}
    for p in plans:
        c = plan_tuple(p)
        out[p["id"]] = {"label": p["label"], "config": list(c),
                        "pooled_2022_2024": E.summarize(E.run_design(pooled, c), c[2]),
                        "2025": E.summarize(E.run_design(by_year[2025], c), c[2])}
        s, v = out[p["id"]]["pooled_2022_2024"], out[p["id"]]["2025"]
        print(f"{p['id']:<12} 22-24 roi={s['roi']} [{s['roi_lo']},{s['roi_hi']}] hit={s['hit_rate']} | 2025 roi={v['roi']} [{v['roi_lo']},{v['roi_hi']}] hit={v['hit_rate']} skip={v['skip_rate']}", flush=True)
    (E.OUT / "plans_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
