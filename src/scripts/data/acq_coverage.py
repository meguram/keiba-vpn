"""新規取得の候補データの『既存データでの被覆』を数える（読み取りのみ・取得なし。ホストで実行）。

python -m src.scripts.data.acq_coverage --out data/local/research/acquisition/coverage.json

- horse_lap（馬ごとの 200m 区間ラップ・位置）: 年ごとのレース数に対する保存済み件数
- partner_notes（併せ馬）: 追い切り 1 本ごとの併せ馬メモの有無（horse_training、馬サンプル）
- race_history の非 JRA（地方・海外）走: 出走馬サンプルでの割合
- owner_id / breeder_id（horse_result）: 保存済みの割合
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("KEIBA_ROOT") or Path(__file__).resolve().parents[3])
JRA_VENUE = re.compile(r"^\d+(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)\d+$")


def is_jra_race_id(race_id: str) -> bool:
    """JRA の race_id は 場コード（5〜10 桁目の先頭 2 桁）が 01〜10。"""
    r = str(race_id or "")
    return len(r) == 12 and r[4:6] in {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"}


def horse_lap_coverage(storage, years: list[int]) -> dict:
    import polars as pl

    out = {}
    for y in years:
        f = ROOT / f"data/page_reference/tables/{y}/race_result_flat.parquet"
        if not f.exists():
            continue
        races = set(pl.read_parquet(f, columns=["race_id"])["race_id"].unique().to_list())
        blobs = storage.batch_list_blobs("horse_lap", str(y))
        have = {k for k in blobs if k in races}
        out[str(y)] = {"races": len(races), "horse_lap": len(have), "rate": round(len(have) / max(len(races), 1), 4)}
    return out


def partner_coverage(sample: int, seed: int) -> dict:
    base = ROOT / "data/local/horse_training"
    files = sorted(base.rglob("*.json"))
    rnd = random.Random(seed)
    files = rnd.sample(files, min(sample, len(files)))
    by_year: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    horses_with = 0
    notes = Counter()
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        got = False
        for s in d.get("entries") or []:
            y = str(s.get("date", ""))[:4]
            by_year[y][0] += 1
            pn = s.get("partner_notes")
            if pn:
                by_year[y][1] += 1
                got = True
                if len(notes) < 300:
                    notes[re.sub(r"\d+", "#", pn)[:20]] += 1
        horses_with += got
    return {
        "horses": len(files),
        "horses_with_partner": horses_with,
        "by_year": {y: {"sessions": a, "with_partner": b, "rate": round(b / max(a, 1), 4)} for y, (a, b) in sorted(by_year.items())},
    }


def history_coverage(storage, horse_ids: list[str]) -> dict:
    n = nj = nn = 0
    horses_nonjra = horses_before_jra = 0
    owner_id = breeder_id = loaded = 0
    # JRA 出走（2020 年以降）ごとの、直前走・過去走に非 JRA が含まれる割合と成績
    rl = {k: [0, 0, 0] for k in ("all", "prev_nonjra", "any_prior_nonjra", "no_prior_nonjra")}  # [runs, wins, top3]
    for h in horse_ids:
        d = storage.load("horse_result", h)
        if not d:
            continue
        loaded += 1
        owner_id += bool(d.get("owner_id"))
        breeder_id += bool(d.get("breeder_id"))
        rows = sorted(d.get("race_history") or [], key=lambda r: str(r.get("date", "")))
        jra = [is_jra_race_id(r.get("race_id")) for r in rows]
        n += len(rows)
        nj += sum(jra)
        nn += len(rows) - sum(jra)
        for i, r in enumerate(rows):
            if not jra[i] or str(r.get("date", "")) < "2020/01/01" or i == 0:
                continue
            fin = r.get("finish_position")
            if not isinstance(fin, int):
                continue
            keys = ["all", "prev_nonjra" if not jra[i - 1] else None, "any_prior_nonjra" if not all(jra[:i]) else "no_prior_nonjra"]
            for k in keys:
                if k:
                    rl[k][0] += 1; rl[k][1] += fin == 1; rl[k][2] += fin <= 3
        if len(rows) - sum(jra):
            horses_nonjra += 1
        if jra and any(not j for j in jra[: jra.index(True)]):
            horses_before_jra += 1
    run_level = {k: {"runs": a, "win": round(b / max(a, 1), 4), "top3": round(c / max(a, 1), 4)} for k, (a, b, c) in rl.items()}
    return {
        "run_level": run_level,
        "horses": loaded, "runs": n, "jra_runs": nj, "non_jra_runs": nn,
        "horses_with_non_jra": horses_nonjra, "horses_non_jra_before_first_jra": horses_before_jra,
        "owner_id_rate": round(owner_id / max(loaded, 1), 4), "breeder_id_rate": round(breeder_id / max(loaded, 1), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/local/research/acquisition/coverage.json")
    ap.add_argument("--horses", type=int, default=600)
    ap.add_argument("--partner-sample", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    import polars as pl
    from src.scraper.storage import HybridStorage

    st = HybridStorage(base_dir=str(ROOT))
    ids = pl.read_parquet(ROOT / "data/local/features/horse_tbl/shutuba_horse.parquet", columns=["horse_id"])["horse_id"].to_list()
    rnd = random.Random(a.seed)
    res = {
        "horse_lap": horse_lap_coverage(st, list(range(2020, 2027))),
        "partner_notes": partner_coverage(a.partner_sample, a.seed),
        "history": history_coverage(st, rnd.sample(ids, min(a.horses, len(ids)))),
    }
    out = Path(a.out) if os.path.isabs(a.out) else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
