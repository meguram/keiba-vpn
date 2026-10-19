"""フラット表を年×列で監査し、特定の年だけ分布が変わった列（パーサの列ずれなど）を見つける。

2025 年の race_result_flat.odds が単勝オッズではなかった問題の再発検知と、他の列の健全性確認が目的。
  python -m src.scripts.data.audit_column_drift_by_year [--tables race_result_flat race_odds_flat ...] [--out DIR]
出力: <out>/column_drift.json と column_drift.md。ずれた疑いのある (表, 列, 年) を一覧にする。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path("data/page_reference/tables")
DEFAULT_TABLES = ["race_result_flat", "race_odds_flat", "race_shutuba_flat", "race_result_lap_flat", "race_index_flat"]
# 乖離の判定: 年の中央値が他年の中央値の分布（中央値の中央値）から、他年の四分位幅の何倍離れたか
SHIFT_Z = 1.5
# 平均 ±（他の年の spread）に対する分位点の比で見る。0〜1 しか取らない列などは分位点で拾う
MIN_ROWS = 2000


def _years(tbl: str) -> list[int]:
    return sorted(int(p.name) for p in BASE.iterdir() if p.is_dir() and p.name.isdigit() and (p / f"{tbl}.parquet").exists())


def _profile(df: pl.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    n = df.height
    for c, dt in df.schema.items():
        s = df[c]
        if dt in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            v = s.cast(pl.Float64).to_numpy()
            ok = np.isfinite(v)
            p = {"n": n, "null": float(1 - ok.mean()) if n else 0.0}
            if ok.sum() >= 10:
                q = np.quantile(v[ok], [0.01, 0.25, 0.5, 0.75, 0.99])
                p.update({"p01": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p99": float(q[4]),
                          "zero": float((v[ok] == 0).mean()), "frac_int": float((np.abs(v[ok] - np.round(v[ok])) < 1e-9).mean())})
            out[c] = p
        elif dt == pl.String:
            e = s.fill_null("")
            out[c] = {"n": n, "null": float((e == "").mean()) if n else 0.0, "n_unique": int(s.n_unique())}
    return out


def _flag(prof_by_year: dict[int, dict]) -> list[dict]:
    flags: list[dict] = []
    years = sorted(prof_by_year)
    cols = set().union(*[set(p) for p in prof_by_year.values()])
    for c in sorted(cols):
        for y in years:
            cur = prof_by_year[y].get(c)
            others = [prof_by_year[o].get(c) for o in years if o != y and prof_by_year[o].get(c)]
            if not cur or len(others) < 2:
                continue
            # 欠損率の急変
            nm = float(np.median([o["null"] for o in others]))
            if abs(cur["null"] - nm) > 0.15:
                flags.append({"col": c, "year": y, "kind": "null_rate", "this": round(cur["null"], 3), "others": round(nm, 3)})
            if "p50" not in cur:
                continue
            os_ = [o for o in others if "p50" in o]
            if len(os_) < 2:
                continue
            med = float(np.median([o["p50"] for o in os_]))
            iqr = float(np.median([o["p75"] - o["p25"] for o in os_]))
            spread = max(iqr, abs(med) * 0.02, 1e-9)
            z = (cur["p50"] - med) / spread
            if abs(z) > SHIFT_Z:
                flags.append({"col": c, "year": y, "kind": "median_shift", "this": round(cur["p50"], 3), "others": round(med, 3), "z": round(z, 2)})
            # 整数らしさ（小数のはずが整数、またはその逆）の変化
            fi = float(np.median([o["frac_int"] for o in os_]))
            if abs(cur["frac_int"] - fi) > 0.4:
                flags.append({"col": c, "year": y, "kind": "int_share", "this": round(cur["frac_int"], 3), "others": round(fi, 3)})
    return flags


def _cross_checks(years: list[int]) -> list[dict]:
    """表をまたいだ整合: 結果の odds ↔ race_odds.win_odds、popularity、レース内 Σ(1/odds)。"""
    rows: list[dict] = []
    for y in years:
        rr_p, ro_p = BASE / str(y) / "race_result_flat.parquet", BASE / str(y) / "race_odds_flat.parquet"
        if not (rr_p.exists() and ro_p.exists()):
            continue
        rr = pl.read_parquet(rr_p, columns=["race_id", "horse_number", "odds", "popularity"]).with_columns(pl.col("odds").cast(pl.Float64, strict=False))
        ro = pl.read_parquet(ro_p, columns=["race_id", "horse_number", "win_odds", "popularity"]).rename({"popularity": "pop_odds"})
        j = rr.join(ro, on=["race_id", "horse_number"], how="inner")
        if j.height == 0:
            continue
        j = j.with_columns(((pl.col("odds") - pl.col("win_odds")).abs() <= 0.05 + 0.01 * pl.col("win_odds")).alias("odds_ok"),
                           (pl.col("popularity") == pl.col("pop_odds")).alias("pop_ok"))
        s = rr.group_by("race_id").agg((1 / pl.col("odds").clip(lower_bound=1.0)).sum().alias("s"))
        rows.append({"year": y, "n_join": j.height, "odds_match": round(float(j["odds_ok"].mean()), 4),
                     "pop_match": round(float(j["pop_ok"].mean()), 4),
                     "races": s.height, "share_races_sum_inv_odds_lt_0_9": round(float((s["s"] < 0.9).mean()), 4)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="*", default=DEFAULT_TABLES)
    ap.add_argument("--out", default="data/local/meta/odds_fix")
    a = ap.parse_args()
    report: dict = {"flags": {}, "cross": []}
    for tbl in a.tables:
        yrs = _years(tbl)
        prof: dict[int, dict] = {}
        for y in yrs:
            df = pl.read_parquet(BASE / str(y) / f"{tbl}.parquet")
            if df.height >= MIN_ROWS:
                prof[y] = _profile(df)
        report["flags"][tbl] = _flag(prof) if len(prof) >= 3 else []
    report["cross"] = _cross_checks(_years("race_result_flat"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "column_drift.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 年×列の分布ずれ監査", ""]
    for tbl, fl in report["flags"].items():
        lines.append(f"## {tbl}（疑い {len(fl)} 件）")
        for f in fl:
            lines.append(f"- {f['col']} / {f['year']}: {f['kind']} 今年={f['this']} 他年={f['others']}" + (f" z={f['z']}" if "z" in f else ""))
        lines.append("")
    lines.append("## 表間の整合（結果 odds ↔ race_odds.win_odds）")
    for r in report["cross"]:
        lines.append(f"- {r['year']}: odds一致 {r['odds_match']:.1%} / 人気一致 {r['pop_match']:.1%} / Σ(1/odds)<0.9 のレース {r['share_races_sum_inv_odds_lt_0_9']:.1%}")
    (out / "column_drift.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
