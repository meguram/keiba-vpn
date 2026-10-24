"""V4 の各特徴量群の『厳密一致』先読み監査（端数リーク含む）。

python -m src.scripts.data.audit_exact_leak --out /tmp/leak_audit_v4.json [--groups self8,conn,...] [--years 2021 2022 2023]

方法: あるレース R の結果（着順・時計・着差・通過順・上がり）を、同じレースの馬の間でずらした入力と元の入力で特徴量を作り、
  (a) R の日より前の行 と (b) R 自身の行 の特徴量が bit 単位で同一か（許容誤差なし）を調べる。
(b) が崩れる列は『自分の結果に依存する』、(a) が崩れる列は『未来の結果に依存する』。
「累積和 − 自分」は端数に自分の値の痕跡が残り、木モデルが拾う（tp-34 で発見）。通常の許容誤差つき検査では見えないので、ここでは許容しない。
同日の前のレースの結果を使う列（track_bias の当日傾向、同日成績の群）は (b) では崩れない（R 自身の結果は使わない）設計であること。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
V4 = ROOT / "data/local/modeling/training_pipeline_v4"
KEY = ["race_id", "horse_id"]
RESULT_COLS = ["finish_position", "finish_time", "time_sec", "margin", "passing_order", "last_3f", "corner_passing"]


def perturb(df: pd.DataFrame, rid: str, cols: list[str] | None = None, race_col: str = "race_id") -> pd.DataFrame:
    """レース rid の結果の列を、同じレースの馬の間で 1 つずらす（全馬の結果が変わる）。"""
    out = df.copy()
    cols = [c for c in (cols or RESULT_COLS) if c in out.columns]
    idx = np.flatnonzero((out[race_col].astype(str) == str(rid)).to_numpy())
    if len(idx) < 2:
        raise ValueError(f"レース {rid} の行が足りない")
    for c in cols:
        vals = out[c].to_numpy(copy=True)
        out[c] = pd.Series(vals, index=out.index).astype(object)
        out.iloc[idx, out.columns.get_loc(c)] = list(np.roll(vals[idx], 1))
        try:
            out[c] = out[c].astype(df[c].dtype)
        except Exception:
            pass
    return out


def _eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        a, b = a.astype("float64"), b.astype("float64")
        return (a == b) | (np.isnan(a) & np.isnan(b))
    return a == b


def compare(base: pd.DataFrame, pert: pd.DataFrame, keys: list[str], dates: pd.Series, d0: pd.Timestamp, rid: str, tol_rel: float = 0.0) -> dict:
    """base / pert は keys + 特徴量列。dates は keys に対応する日付（keys を持つ DataFrame ではなく、base と同じ index の Series）。"""
    b = base.copy()
    b["_d"] = pd.to_datetime(dates.to_numpy())
    m = b.merge(pert, on=keys, how="inner", suffixes=("", "__p"))
    feats = [c for c in base.columns if c not in keys]
    before = (m["_d"] < d0).to_numpy()
    own = (m["race_id"].astype(str) == str(rid)).to_numpy() if "race_id" in m.columns else np.zeros(len(m), bool)
    viol_before, viol_own, tiny, tiny_b, scale = {}, {}, {}, {}, {}
    tol_before, tol_own = {}, {}
    for c in feats:
        e = _eq(m[c].to_numpy(), m[c + "__p"].to_numpy())
        nb, no = int((~e[before]).sum()), int((~e[own]).sum())
        if nb:
            viol_before[c] = nb
            x, y = m[c].to_numpy(dtype="float64", na_value=np.nan)[before], m[c + "__p"].to_numpy(dtype="float64", na_value=np.nan)[before]
            tiny_b[c] = float(np.nanmax(np.abs(x - y))) if np.isfinite(x - y).any() else float("nan")
            scale[c] = float(np.nanstd(x)) if np.isfinite(x).any() else float("nan")
        if tol_rel > 0:
            xa, ya = m[c].to_numpy(dtype="float64", na_value=np.nan), m[c + "__p"].to_numpy(dtype="float64", na_value=np.nan)
            sd = max(float(np.nanstd(xa)) if np.isfinite(xa).any() else 0.0, 1e-9)
            big = np.abs(xa - ya) > tol_rel * sd
            big |= np.isnan(xa) != np.isnan(ya)
            if big[before].any():
                tol_before[c] = int(big[before].sum())
            if big[own].any():
                tol_own[c] = int(big[own].sum())
        if no:
            viol_own[c] = no
            x, y = m[c].to_numpy(dtype="float64", na_value=np.nan)[own], m[c + "__p"].to_numpy(dtype="float64", na_value=np.nan)[own]
            dd = np.nanmax(np.abs(x - y)) if np.isfinite(x - y).any() else np.nan
            tiny[c] = float(dd)
    return {"rows_before": int(before.sum()), "rows_own": int(own.sum()), "n_features": len(feats),
            "violations_before": viol_before, "violations_own": viol_own, "own_max_abs_diff": tiny,
            "before_max_abs_diff": tiny_b, "col_std": scale,
            "tolerance_rel": tol_rel, "violations_before_beyond_tol": tol_before, "violations_own_beyond_tol": tol_own}


def pick_race(runs: pd.DataFrame, which: int, date_col: str = "date", min_year: int = 0) -> tuple[str, pd.Timestamp]:
    """JRA の芝の本場レースを 2 通り選ぶ（which=0: 2022 年中、1: 2023 年中）。頭数が 12 以上。"""
    d = runs.copy()
    d["_d"] = pd.to_datetime(d[date_col])
    c = d.groupby("race_id").agg(n=("horse_id", "size"), dd=("_d", "first"))
    c = c[(c["n"] >= 12) & (c["dd"].dt.month.isin([6, 7, 9])) & (c["dd"].dt.year >= min_year)].sort_values("dd")
    years = sorted(c["dd"].dt.year.unique())
    y = years[min(which, len(years) - 1)]
    cy = c[c["dd"].dt.year == y]
    rid = cy.index[len(cy) // 2]
    return str(rid), pd.Timestamp(cy.loc[rid, "dd"])


# ───────────────────────── 群ごとの監査
def audit_self8(years, which):
    from src.pipeline.features import factory_self as fs

    runs = fs.load_runs(TABLES, years)
    rid, d0 = pick_race(runs, which)
    names = ["change", "experience", "lags_margin", "pair", "rest", "seq_stats", "timedev"]

    def build(r):
        g = fs.build_groups(r, only=names)
        return pd.concat([v.drop(columns=KEY) for v in g.values()], axis=1).assign(**{k: next(iter(g.values()))[k].to_numpy() for k in KEY})

    base = build(runs)
    pert = build(perturb(runs, rid))
    dates = base.merge(runs[KEY + ["date"]], on=KEY, how="left")["date"]
    return rid, d0, compare(base, pert, KEY, dates, d0, rid)


def audit_conn(years, which):
    from src.pipeline.features import factory_conn as fc

    h0 = fc.load_history(TABLES, years)
    rid, d0 = pick_race(h0, which)

    def build(h):
        hp = fc.prepare(h)
        groups = fc.build_all(hp)
        cols = {}
        for gname, d in groups.items():
            for c, arr in d.items():
                cols[f"{gname}:{c}"] = np.asarray(arr)
        F = pd.DataFrame(cols)
        F["race_id"], F["horse_id"], F["_date"] = hp["race_id"].to_numpy(), hp["horse_id"].to_numpy(), hp["date"].to_numpy()
        return F

    base, pert = build(h0), build(perturb(h0, rid))
    dates = base["_date"]
    return rid, d0, compare(base.drop(columns=["_date"]), pert.drop(columns=["_date"]), KEY, dates, d0, rid)


def audit_layer_a_priors(years, which):
    from src.pipeline.features import historical_entity_stats as hes

    import polars as pl

    parts = []
    for y in years:
        f = TABLES / str(y) / "race_result_flat.parquet"
        sch = pl.read_parquet_schema(f)
        parts.append(pl.read_parquet(f, columns=[c for c in ["race_id", "horse_number", "horse_id", "jockey_id", "trainer_id", "date", "finish_position", "venue_code"] if c in sch]))
    rr = pl.concat(parts, how="diagonal_relaxed").to_pandas()
    rr["race_id"], rr["horse_id"] = rr["race_id"].astype(str), rr["horse_id"].astype(str)
    rid, d0 = pick_race(rr, which)

    def build(r):
        t = hes.build_history_feature_table(r)
        return t.merge(r[["race_id", "horse_number", "horse_id"]].drop_duplicates(["race_id", "horse_number"]), on=["race_id", "horse_number"], how="left").drop(columns=["horse_number"])

    base, pert = build(rr), build(perturb(rr, rid, ["finish_position"]))
    dates = base.merge(rr[KEY + ["date"]].drop_duplicates(KEY), on=KEY, how="left")["date"]
    return rid, d0, compare(base, pert, KEY, dates, d0, rid)


def _v4_frame(years):
    from src.pipeline.models import ablation as ab

    df = ab.load_table(V4 / "features/train_table.parquet")
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    df = df[pd.to_datetime(df["date"]).dt.year.isin(years)].reset_index(drop=True)
    return df


def audit_field(years, which):
    from src.pipeline.features import factory_field as ff

    df = _v4_frame(years)
    rid, d0 = pick_race(df, which)
    pace = ff.load_pace_table()
    names = ["mem_strength", "mem_demo", "mem_conn", "course_draw", "calendar_env", "race_type"]

    def build(x):
        parts = []
        for g in names:
            f = ff.build_group(g, x, **({"pace_table": pace} if g == "race_type" else {}))
            parts.append(f.drop(columns=[c for c in KEY if c in f.columns]).add_prefix(f"{g}:"))
        F = pd.concat(parts, axis=1)
        F["race_id"], F["horse_id"] = x["race_id"].to_numpy(), x["horse_id"].to_numpy()
        return F

    base, pert = build(df), build(perturb(df, rid, ["finish_position"]))
    return rid, d0, compare(base, pert, KEY, df["date"], d0, rid)


def audit_ped_sire_cond(years, which):
    from src.pipeline.features import factory_ped as fp

    df = _v4_frame(years)
    rid, d0 = pick_race(df, which)
    A, H = fp.build_ped_tables(df["horse_id"].unique())

    def build(x):
        f = fp.ped_sire_cond(x, A, H)
        f["race_id"], f["horse_id"] = x["race_id"].to_numpy(), x["horse_id"].to_numpy()
        return f

    base, pert = build(df), build(perturb(df, rid, ["finish_position"]))
    return rid, d0, compare(base, pert, KEY, df["date"], d0, rid)


def audit_pedigree_context(years, which):
    from src.pipeline.features import pedigree_context as pc

    df = _v4_frame(years)
    labels = pd.read_parquet(ROOT / "data/local/modeling/training_pipeline/oof/stage_labels.parquet")
    labels["race_id"], labels["horse_id"] = labels["race_id"].astype(str), labels["horse_id"].astype(str)
    rid, d0 = pick_race(df, which)
    for c in ("sire", "dam_sire"):
        if c not in df.columns:
            df[c] = ""

    def build(x, lab):
        f = pc.build(x.reset_index(drop=True), lab)
        f["race_id"], f["horse_id"] = x["race_id"].to_numpy(), x["horse_id"].to_numpy()
        return f

    lab2 = perturb(labels, rid, ["pos_early", "l3f"])
    base, pert = build(df, labels), build(perturb(df, rid, ["finish_position"]), lab2)
    return rid, d0, compare(base, pert, KEY, df["date"], d0, rid)


def audit_horse_history(years, which):
    import polars as pl

    from src.pipeline.features.horse_history.builder import build_from_runs
    from src.pipeline.features.horse_history.common import load_runs
    from src.pipeline.features.horse_history.serving import EMPTY_RM

    runs = load_runs(TABLES, years)
    pdf = runs.to_pandas()
    rid, d0 = pick_race(pdf, which, min_year=2022)       # 全体フィット(FIT_END 以前)の窓の外のレースで調べる
    pdf2 = perturb(pdf, rid)

    def build(p):
        q = pl.from_pandas(p, include_index=False)
        if q.schema["date"] != pl.Date:
            q = q.with_columns(pl.col("date").cast(pl.Date))
        F, _P = build_from_runs(q, EMPTY_RM)
        f = F.to_pandas()
        return f

    base, pert = build(pdf), build(pdf2)
    base["race_id"], base["horse_id"] = base["race_id"].astype(str), base["horse_id"].astype(str)
    pert["race_id"], pert["horse_id"] = pert["race_id"].astype(str), pert["horse_id"].astype(str)
    dates = base[KEY].merge(pdf[KEY + ["date"]].drop_duplicates(KEY), on=KEY, how="left")["date"]
    fam = json.loads((V4 / "features/feature_families.json").read_text(encoding="utf-8"))
    fam_cols = sorted({c for v in fam.values() for c in v})
    feats = [c for c in fam_cols if c in base.columns and c in pert.columns and base[c].dtype.kind in "fiub"]
    return rid, d0, compare(base[KEY + feats], pert[KEY + feats], KEY, dates, d0, rid, tol_rel=1e-6)


AUDITS = {
    "self8": audit_self8, "conn": audit_conn, "layer_a_priors": audit_layer_a_priors, "field": audit_field,
    "ped_sire_cond": audit_ped_sire_cond, "pedigree_context": audit_pedigree_context, "horse_history": audit_horse_history,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default=",".join(AUDITS))
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023])
    ap.add_argument("--scenarios", type=int, default=2)
    ap.add_argument("--stored", default="", help="self|conn|field|all: 保存済みの学習用特徴量と現行コードでの再構築を比べる")
    a = ap.parse_args()
    if a.stored:
        Path(a.out).write_text(json.dumps(stored_check(a.stored), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return
    res: dict = {}
    for g in a.groups.split(","):
        res[g] = []
        for which in range(a.scenarios):
            t0 = time.time()
            try:
                rid, d0, r = AUDITS[g](a.years, which)
                r.update({"race_id": rid, "date": str(d0.date()), "sec": round(time.time() - t0, 1)})
                nb, no = len(r["violations_before"]), len(r["violations_own"])
                extra = f" / 許容(相対1e-6)超: 未来 {len(r['violations_before_beyond_tol'])} 列・自分 {len(r['violations_own_beyond_tol'])} 列" if r.get("tolerance_rel") else ""
                print(f"[{g}] race={rid} {d0.date()}: 未来依存 {nb} 列 / 自分依存 {no} 列 / 全 {r['n_features']} 列{extra} ({r['sec']}s)", flush=True)
                res[g].append(r)
            except Exception as exc:  # noqa: BLE001
                import traceback

                print(f"[{g}] 失敗: {exc}", flush=True)
                res[g].append({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]})
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")



# ───────────────────────── 保存済みの学習用特徴量 vs 現行コードでの再構築（修正前の形で作った分に端数が残っていないか）
def _stored_vs_rebuilt(stored: pd.DataFrame, rebuilt: pd.DataFrame, label: str) -> dict:
    s = stored.drop_duplicates(KEY)
    r = rebuilt.drop_duplicates(KEY)
    m = s.merge(r, on=KEY, how="inner", suffixes=("", "__r"))
    cols = [c for c in s.columns if c not in KEY and c + "__r" in m.columns]
    bad, worst = {}, {}
    for c in cols:
        x = m[c].to_numpy(dtype="float32", na_value=np.nan)
        y = m[c + "__r"].to_numpy(dtype="float32", na_value=np.nan)
        e = (x == y) | (np.isnan(x) & np.isnan(y))
        if not e.all():
            bad[c] = int((~e).sum())
            d = np.abs(x.astype("float64") - y.astype("float64"))
            worst[c] = float(np.nanmax(d[~e])) if np.isfinite(d[~e]).any() else float("nan")
    return {"label": label, "rows": int(len(m)), "n_cols": len(cols), "n_cols_differ": len(bad), "differ": bad, "max_abs_diff": worst}


def stored_check(which: str) -> dict:
    out = {}
    if which in ("self", "all"):
        from src.pipeline.features import factory_self as fs

        runs = fs.load_runs(TABLES, list(range(2020, 2026)))
        d = ROOT / "data/local/modeling/training_pipeline/factory/self"
        at = pd.read_parquet(d / "horse_attrs.parquet").set_index("horse_id") if (d / "horse_attrs.parquet").exists() else None
        bday = at["birthday"] if at is not None else None
        names = ["change", "experience", "lags_margin", "pair", "rest", "seq_stats", "timedev", "attrs"]
        for n in names:
            p = d / f"{n}.parquet"
            if not p.exists():
                continue
            g = fs.build_groups(runs, birthday=bday, attrs=at, only=[n]).get(n)
            if g is None:
                continue
            st = pd.read_parquet(p)
            st["race_id"], st["horse_id"] = st["race_id"].astype(str), st["horse_id"].astype(str)
            out[f"self:{n}"] = _stored_vs_rebuilt(st, g, f"self:{n}")
            print(out[f"self:{n}"]["label"], out[f"self:{n}"]["n_cols"], "列中", out[f"self:{n}"]["n_cols_differ"], "列が相違", flush=True)
    if which in ("conn", "all"):
        from src.pipeline.features import factory_conn as fc

        h = fc.prepare(fc.load_history())
        groups = fc.build_all(h)
        d = ROOT / "data/local/modeling/training_pipeline/factory/conn/groups"
        raw = _v4_frame(list(range(2020, 2026)))[KEY]
        for gname in ["ent_o", "win_o", "streak_o", "ent_ob"]:
            p = d / f"{gname}.parquet"
            if gname not in groups or not p.exists():
                continue
            F = pd.DataFrame({c: np.asarray(a, dtype="float32") for c, a in groups[gname].items()})
            F["race_id"], F["horse_id"] = h["race_id"].to_numpy(), h["horse_id"].to_numpy()
            st = pd.read_parquet(p)
            if "race_id" not in st.columns:
                from src.pipeline.models import ablation as ab

                kk = ab.load_table(ROOT / "data/local/modeling/training_pipeline_fixed/features/train_table.parquet")[KEY].astype(str)
                st = pd.concat([kk.reset_index(drop=True), st.reset_index(drop=True)], axis=1)
            out[f"conn:{gname}"] = _stored_vs_rebuilt(st, F, f"conn:{gname}")
            print(out[f"conn:{gname}"]["label"], out[f"conn:{gname}"]["n_cols"], "列中", out[f"conn:{gname}"]["n_cols_differ"], "列が相違", flush=True)
    if which in ("field", "all"):
        from src.pipeline.features import factory_field as ff

        df = _v4_frame(list(range(2020, 2026)))
        pace = ff.load_pace_table()
        for fdir in (V4 / "factory_field", ROOT / "data/local/modeling/training_pipeline/factory/field"):
            if not fdir.exists():
                continue
            for g in ["mem_strength", "mem_demo", "mem_conn", "course_draw", "calendar_env", "race_type"]:
                p = fdir / f"{g}.parquet"
                if not p.exists():
                    continue
                f = ff.build_group(g, df, **({"pace_table": pace} if g == "race_type" else {}))
                F = pd.concat([df[KEY].reset_index(drop=True), f.reset_index(drop=True)], axis=1)
                st = pd.read_parquet(p)
                st["race_id"], st["horse_id"] = st["race_id"].astype(str), st["horse_id"].astype(str)
                key = f"field[{fdir.parent.name}/{fdir.name}]:{g}"
                out[key] = _stored_vs_rebuilt(st, F, key)
                print(key, out[key]["n_cols"], "列中", out[key]["n_cols_differ"], "列が相違", flush=True)
    return out


if __name__ == "__main__":
    main()
