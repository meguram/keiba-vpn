"""tp-48: 3連単の実オッズ（race_trifecta_odds）を取り出して表にし、確定払戻との整合を確かめる。

python -m src.scripts.data.tp48_trifecta collect --years 2024 2025 2026     # ストレージ（stg: GCS）から全組合せを parquet へ
python -m src.scripts.data.tp48_trifecta qa --years 2024 2025 2026          # 払戻との一致・Σ(1/odds)・組合せ数・除外リスト

出力: data/local/modeling/betting/tp48/trifecta_<年>.parquet（race_id, a, b, c, odds, pop）と qa.json / exclude.json。
オッズは確定後に取得した値で、購入時点の値ではない。
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("TP48_OUT", str(ROOT / "data/local/modeling/betting/tp48")))
TABLES = ROOT / "data/page_reference/tables"
RATE = 0.725


def race_ids(year: int) -> list[str]:
    f = TABLES / str(year) / "race_result_flat.parquet"
    d = pd.read_parquet(f, columns=["race_id"])
    return sorted({str(r) for r in d["race_id"].dropna().unique()})


def doc_to_frame(doc: dict) -> pd.DataFrame | None:
    t = doc.get("trifecta") if isinstance(doc, dict) else None
    if not t:
        return None
    arr = np.asarray(t, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None
    return pd.DataFrame({
        "race_id": str(doc.get("race_id") or ""),
        "a": arr[:, 0].astype(np.int8), "b": arr[:, 1].astype(np.int8), "c": arr[:, 2].astype(np.int8),
        "odds": arr[:, 3].astype(np.float32), "pop": (arr[:, 4] if arr.shape[1] > 4 else np.zeros(len(arr))).astype(np.int16),
    })


def collect(a) -> None:
    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    OUT.mkdir(parents=True, exist_ok=True)
    for year in a.years:
        ids = race_ids(year)
        print(year, "races", len(ids), flush=True)

        def load(rid: str):
            try:
                d = st.load("race_trifecta_odds", rid)
            except Exception as e:      # noqa: BLE001
                return rid, None, str(e)
            return rid, d, ""

        frames, missing = [], []
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for k, (rid, d, err) in enumerate(ex.map(load, ids)):
                f = doc_to_frame(d) if d else None
                if f is None:
                    missing.append(rid)
                else:
                    f["race_id"] = rid
                    frames.append(f.assign(n_head=int(d.get("n_head") or 0), expected=int(d.get("expected_combos") or 0)))
                if k % 500 == 0:
                    print(year, k, len(frames), len(missing), flush=True)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        df.to_parquet(OUT / f"trifecta_{year}.parquet", index=False)
        (OUT / f"missing_{year}.json").write_text(json.dumps(missing), encoding="utf-8")
        print(year, "saved", len(df), "rows /", df["race_id"].nunique() if len(df) else 0, "races; missing", len(missing), flush=True)


def qa(a) -> None:
    from src.scripts.data.tp36_final2026 import parse_payoff_json
    from src.scraper.payoff_repair import load_fix

    res: dict = {}
    exclude: dict = {}
    for year in a.years:
        f = OUT / f"trifecta_{year}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        g = d.groupby("race_id")
        stat = pd.DataFrame({"n_rows": g.size(), "inv_sum": g["odds"].apply(lambda s: float((1.0 / s.clip(lower=1.0)).sum())),
                             "n_head": g["n_head"].first(), "expected": g["expected"].first()})
        stat["combo_ok"] = stat["n_rows"] >= 0.97 * stat["expected"]        # 応答に一部の組合せ（ごく人気薄）が欠ける日がある。97% 未満は除外
        stat["miss_rate"] = 1.0 - stat["n_rows"] / stat["expected"].clip(lower=1)
        stat["inv_ok"] = stat["inv_sum"].between(1.37, 1.39)
        # 払戻との一致: 1 着-2 着-3 着の的中組合せのオッズ×100 と払戻
        if year >= 2026:
            pj = pd.read_parquet(TABLES / str(year) / "race_result_flat.parquet", columns=["race_id", "payoff"]).drop_duplicates("race_id")
            fix = load_fix(year)
            hit = {}
            for r in pj.itertuples(index=False):
                h = parse_payoff_json(r.payoff, fix.get(str(r.race_id)))
                hit[str(r.race_id)] = h.get("sanrentan", {})
        else:
            pay = pd.read_parquet(ROOT / "data/local/modeling/betting/payoff.parquet")
            pay = pay[(pay["bet_type"] == "sanrentan") & (pay["race_id"].str.startswith(str(year)[:4]) | True)]
            hit = {}
            for r in pay.itertuples(index=False):
                try:
                    c = tuple(int(x) for x in str(r.combo).replace(">", "-").replace("→", "-").replace(" ", "").split("-"))
                except ValueError:
                    continue
                hit.setdefault(str(r.race_id), {})[c] = float(r.payout)
        odds_map = {rid: dict(zip(zip(x["a"], x["b"], x["c"]), x["odds"])) for rid, x in d.groupby("race_id")}
        match, total, rel = 0, 0, []
        bad_pay = []
        for rid, h in hit.items():
            om = odds_map.get(rid)
            if not om or not h:
                continue
            for c, pay_ in h.items():
                o = om.get(tuple(int(v) for v in c))
                if o is None:
                    bad_pay.append(rid)
                    continue
                total += 1
                r = pay_ / (o * 100.0)
                rel.append(r)
                if abs(r - 1.0) < 0.002:
                    match += 1
                else:
                    bad_pay.append(rid)
        rel = np.array(rel) if rel else np.array([np.nan])
        res[str(year)] = {
            "races": int(len(stat)), "rows": int(len(d)), "combo_ok_rate": float(stat["combo_ok"].mean()), "any_missing_rate": float((stat["n_rows"] < stat["expected"]).mean()), "miss_rate_mean": float(stat["miss_rate"].mean()), "inv_sum_median": float(stat["inv_sum"].median()),
            "inv_sum_p5_p95": [float(stat["inv_sum"].quantile(0.05)), float(stat["inv_sum"].quantile(0.95))], "inv_ok_rate": float(stat["inv_ok"].mean()),
            "hit_checked": int(total), "hit_exact_rate": float(match / max(total, 1)), "hit_ratio_median": float(np.nanmedian(rel)),
            "hit_ratio_p1_p99": [float(np.nanquantile(rel, 0.01)), float(np.nanquantile(rel, 0.99))]}
        ex = set(stat.index[~stat["combo_ok"] | ~stat["inv_ok"]]) | set(bad_pay)
        missing = json.loads((OUT / f"missing_{year}.json").read_text(encoding="utf-8")) if (OUT / f"missing_{year}.json").exists() else []
        exclude[str(year)] = {"missing": missing, "inconsistent": sorted(ex)}
    (OUT / "qa.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "exclude.json").write_text(json.dumps(exclude, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "qa"])
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    {"collect": collect, "qa": qa}[a.cmd](a)


if __name__ == "__main__":
    main()
