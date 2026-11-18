"""好走距離レンジ（短）（長）の定義と境界（相対値のしきい値 τ）をデータで決める（読むだけ・JSON と馬ごとの最新値を書く）。

入力（ホストのローカルだけ。GCS は読まない）:
  data/page_reference/tables/<年>/race_result_flat.parquet（2016〜2026）
候補: τ = 0.10 / 0.15 / 0.20 / 0.25 × 芝ダ別・合算（8 通り）。走ごとに、その走の日より前の走だけから as-of のレンジを作る。
評価: 前走までの走数 3 以上の走で、対象レースの距離が「レンジの中／短い側の外／長い側の外」の群別に、勝率・複勝率と、
  人気から期待される率（axis_score_curves.expected_tables。期待値の学習は 2016〜2023）との差（人気補正後）を出し、「中 − 外」を比べる。
  区間はレース単位のブートストラップ（2,000 回）。
選び方（事前に固定）: 開発 2022〜2024 年で、複勝率差・勝率差の 95% 区間の下限がともに 0 を超える候補のうち、複勝率差（人気補正後・中 − 外）が最大のもの。
  該当が無ければ複勝率差が最大のもの。選んだ後に、その候補だけ 2025 年（test）を 1 回見る。2026 年は参考（選択に使わない）。
使い方（ホストで）: cd <repo> && nice -n 19 /opt/conda/bin/python -I /tmp/good_distance_eval.py OUT.json [--latest-out OUT.parquet]
  good_distance.py は /tmp に置くか、リポジトリの src/research/race/ に有ればそれを使う。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/tmp")

try:
    from src.research.race import good_distance as gd  # noqa: E402
except ImportError:
    import good_distance as gd  # noqa: E402  /tmp に置いた版（リポジトリに無いホストで実行するとき）
from src.scripts.data import axis_score_curves as asc  # noqa: E402

TABLE_DIR = REPO / "data/page_reference/tables"
YEARS = range(2016, 2027)
COLS = ["race_id", "date", "surface", "distance", "race_class", "sex_age", "finish_position", "horse_id", "popularity", "field_size"]
TAUS = (0.10, 0.15, 0.20, 0.25)
DEV = (2022, 2023, 2024)
TEST = (2025,)
REF = (2026,)
MIN_PRIOR = 3          # 評価・表示の最低走数（前走までの走数）
BOOT = 2000
QS = (0.05, 0.25, 0.5, 0.75, 0.95)


def load_flat() -> pd.DataFrame:
    parts = [pd.read_parquet(TABLE_DIR / str(y) / "race_result_flat.parquet", columns=COLS) for y in YEARS if (TABLE_DIR / str(y) / "race_result_flat.parquet").exists()]
    d = pd.concat(parts, ignore_index=True)
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    for c in ("popularity", "field_size", "finish_position", "distance"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    n = d.groupby("race_id")["race_id"].transform("size")
    d["field_size"] = d["field_size"].where(d["field_size"] > 0, n)
    d = d.drop_duplicates(["race_id", "horse_id"]).reset_index(drop=True)
    d["year"] = d["race_id"].str[:4].astype(int)
    return d


def qdict(x: pd.Series | np.ndarray) -> dict:
    x = pd.Series(x).dropna()
    if x.empty:
        return {"n": 0}
    return {"n": int(len(x)), **{f"p{int(q * 100):02d}": round(float(x.quantile(q)), 4) for q in QS}, "mean": round(float(x.mean()), 4)}


def place_hits(finish: np.ndarray, field_size: np.ndarray) -> np.ndarray:
    """複勝の的中（8 頭以上は 3 着以内、5〜7 頭は 2 着以内）。"""
    return (np.asarray(finish) <= asc.place_cut(np.asarray(field_size).astype(int))).astype(float)


def group_of(dist: np.ndarray, short: np.ndarray, long_: np.ndarray) -> np.ndarray:
    """対象レースの距離がレンジの 中=0・短い側の外=1・長い側の外=2。"""
    return np.where(dist < short, 1, np.where(dist > long_, 2, 0))


def group_stats(race_idx: np.ndarray, grp: np.ndarray, stat: np.ndarray, n_races: int) -> np.ndarray:
    """[レース, 群(0 中・1 短い側の外・2 長い側の外), 統計] の合計。統計は列 n, w, p, ew, ep。"""
    out = np.zeros((n_races, 3, stat.shape[1]))
    np.add.at(out, (race_idx, grp), stat)
    return out


def contrast(rs: np.ndarray) -> dict:
    """rs: [レース, 3, 5]。中 − 外（短い側と長い側を合わせる）の 人気補正後の 勝率差・複勝率差。"""
    t = rs.sum(axis=0)
    inn, out = t[0], t[1] + t[2]

    def diff(a: np.ndarray, k: int) -> float:
        return (a[k] - a[k + 2]) / a[0] if a[0] > 0 else float("nan")   # k=1: 勝率（w−ew）、k=2: 複勝率（p−ep）

    def d2(k: int) -> float:
        return diff(inn, k) - diff(out, k)

    return {"n_in": int(inn[0]), "n_out_short": int(t[1][0]), "n_out_long": int(t[2][0]),
            "win_in": inn[1] / inn[0], "win_out": out[1] / out[0], "plc_in": inn[2] / inn[0], "plc_out": out[2] / out[0],
            "dwin_in": diff(inn, 1), "dwin_out": diff(out, 1), "dplc_in": diff(inn, 2), "dplc_out": diff(out, 2),
            "d_win": d2(1), "d_plc": d2(2)}


def boot_ci(rs: np.ndarray, rng: np.random.Generator) -> dict:
    """レース単位で復元抽出（2,000 回）した 中 − 外 の 勝率差・複勝率差の 95% 区間。"""
    R = rs.shape[0]
    flat = rs.reshape(R, -1)
    dw, dp = [], []
    for _ in range(BOOT // 200):
        cnt = rng.multinomial(R, np.full(R, 1.0 / R), size=200)
        t = (cnt @ flat).reshape(200, 3, -1)
        inn, out = t[:, 0], t[:, 1] + t[:, 2]
        for k, acc in ((1, dw), (2, dp)):
            acc.append((inn[:, k] - inn[:, k + 2]) / inn[:, 0] - (out[:, k] - out[:, k + 2]) / out[:, 0])
    dw, dp = np.concatenate(dw), np.concatenate(dp)
    return {"win_lo": float(np.nanpercentile(dw, 2.5)), "win_hi": float(np.nanpercentile(dw, 97.5)),
            "plc_lo": float(np.nanpercentile(dp, 2.5)), "plc_hi": float(np.nanpercentile(dp, 97.5))}


def evaluate(ev: pd.DataFrame, short: np.ndarray, long_: np.ndarray, years, rng: np.random.Generator, common: np.ndarray | None = None) -> dict:
    """common: 全候補で共通の評価対象（None なら各候補でレンジが出る走）。"""
    m = ev["year"].isin(years).to_numpy() & ~np.isnan(short)
    if common is not None:
        m &= common
    if not m.any():
        return {"n": 0}
    d = ev["dist100"].to_numpy()[m]
    grp = group_of(d, short[m], long_[m])
    rid, uniq = pd.factorize(ev["race_id"].to_numpy()[m])
    stat = np.column_stack([np.ones(m.sum()), ev["y_w"].to_numpy()[m], ev["y_p"].to_numpy()[m], ev["e_w"].to_numpy()[m], ev["e_p"].to_numpy()[m]])
    rs = group_stats(rid, grp, stat, len(uniq))
    res = contrast(rs)
    res.update(boot_ci(rs, rng))
    res["n_races"] = int(len(uniq))
    return {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for k, v in res.items()}


def points_summary(d: pd.DataFrame) -> dict:
    v = d[d["points"].notna()]
    return {"runs": int(len(v)), "points_share": {str(int(k)): round(float(x), 4) for k, x in v["points"].value_counts(normalize=True).sort_index().items()},
            "kind_share": {k: round(float(x), 4) for k, x in v["kind"].value_counts(normalize=True).items()},
            "kind_blank_races": int(d.loc[d["race_class"].fillna("").eq(""), "race_id"].nunique())}


def distribution(d: pd.DataFrame) -> dict:
    """全馬（その時点で走った全走）の点数・相対値・幅の分布。境界 τ を決める根拠。"""
    out: dict = {}
    for by in (False, True):
        key = "by_surface" if by else "merged"
        p = gd.profile_table(d, by)
        h5 = p[p["n_runs"] >= 5]
        per_horse_max = h5.groupby(["horse_id", "surface"])["rel"].max()
        n_dist = h5.groupby(["horse_id", "surface"]).size()
        blk = {"horse_units": int(p.drop_duplicates(["horse_id", "surface"]).shape[0]), "units_5plus": int(h5.drop_duplicates(["horse_id", "surface"]).shape[0]),
               "rel_all_cells_5plus": qdict(h5["rel"]), "rel_max_per_unit_5plus": qdict(per_horse_max), "n_distances_5plus": qdict(n_dist), "tau": {}}
        for tau in TAUS:
            lr = gd.latest_range(d, tau, by)
            lr = lr[lr["n_runs"] >= 5]
            w = lr["kd_long"] - lr["kd_short"]
            blk["tau"][f"{tau:.2f}"] = {"width": qdict(w), "single_share": round(float((w == 0).mean()), 4), "wide_800_share": round(float((w >= 800).mean()), 4)}
        out[key] = blk
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("--latest-out", default=None)
    a = ap.parse_args()
    out_json = Path(a.out_json)
    latest_out = Path(a.latest_out) if a.latest_out else out_json.with_suffix(".latest.parquet")

    d = load_flat()
    d = gd.run_points(d)
    d = d[d["surface"].isin(gd.GROUND)].copy()
    d["dist100"] = gd.round_distance(d["distance"])
    pts_sum = points_summary(d)
    print("runs", len(d), pts_sum, flush=True)

    Tw, Tp = asc.expected_tables(d[d["year"].isin(list(asc.EXP_YEARS))])
    ev_ok = (d["finish_position"] > 0) & (d["field_size"] >= asc.N_MIN_FIELD) & (d["popularity"] >= 1)
    nf = d["field_size"].fillna(0).to_numpy().astype(int)
    nb = np.clip(nf, asc.N_MIN_FIELD, 18)
    pp = np.minimum(d["popularity"].fillna(1).to_numpy().astype(int), nb)
    d["y_w"] = (d["finish_position"] == 1).astype(float)
    d["y_p"] = place_hits(d["finish_position"].to_numpy(), nf)
    d["e_w"] = Tw[nb, pp]
    d["e_p"] = Tp[nb, pp]

    dist_summary = distribution(d)
    print("distribution done", flush=True)

    runs = d[["horse_id", "date", "surface", "distance", "points"]]
    cands: dict[str, pd.DataFrame] = {}
    for by in (False, True):
        res = gd.as_of_ranges(runs, TAUS, by, MIN_PRIOR)
        for tau in TAUS:
            cands[f"tau{tau:.2f}_{'surf' if by else 'all'}"] = res[tau]
        print("as-of done", "by_surface" if by else "merged", flush=True)

    sel_mask = ev_ok.fillna(False).to_numpy(bool) & d["year"].isin(DEV + TEST + REF).to_numpy(bool)
    ev = d[sel_mask][["race_id", "year", "dist100", "y_w", "y_p", "e_w", "e_p"]].reset_index(drop=True)
    sel_idx = d.index[sel_mask]

    shorts = {k: r["kd_short"].reindex(sel_idx).to_numpy() for k, r in cands.items()}
    common = np.all([~np.isnan(v) for v in shorts.values()], axis=0)     # 全候補でレンジが出る走（合算・芝ダ別の両方で前走 3 以上）にそろえる
    print("common rows", int(common.sum()), "of", len(common), flush=True)

    def run_dev(common_mask):
        rows = {}
        for name, r in cands.items():
            rng = np.random.default_rng(20261118)
            short, long_ = shorts[name], r["kd_long"].reindex(sel_idx).to_numpy()
            rows[name] = evaluate(ev, short, long_, DEV, rng, common_mask)
            e = rows[name]
            print(name, "n_in", e["n_in"], "short", e["n_out_short"], "long", e["n_out_long"], "dplc", e["d_plc"], (e["plc_lo"], e["plc_hi"]), "dwin", e["d_win"], (e["win_lo"], e["win_hi"]), flush=True)
        return rows

    dev_unaligned = run_dev(None)        # 参考: 候補ごとに評価に入る走が違う旧い比べ方
    dev_rows = run_dev(common)           # 選ぶのはこちら
    qual = {k: v for k, v in dev_rows.items() if v["plc_lo"] > 0 and v["win_lo"] > 0}
    pool = qual or dev_rows
    chosen = max(pool, key=lambda k: pool[k]["d_plc"])
    unaligned_chosen = max(dev_unaligned, key=lambda k: dev_unaligned[k]["d_plc"])
    r = cands[chosen]
    short, long_ = shorts[chosen], r["kd_long"].reindex(sel_idx).to_numpy()
    by_year = {str(y): evaluate(ev, short, long_, (y,), np.random.default_rng(20261118), common) for y in DEV}
    test = evaluate(ev, short, long_, TEST, np.random.default_rng(20261118), common)
    ref = evaluate(ev, short, long_, REF, np.random.default_rng(20261118), common)
    print("chosen", chosen, "unaligned_chosen", unaligned_chosen, "qualified", list(qual), "test2025", test, "ref2026", ref, flush=True)

    tau_c = float(chosen.split("_")[0][3:])
    by_c = chosen.endswith("surf")
    latest = gd.latest_range(runs, tau_c, by_c)
    latest.loc[latest["n_runs"] < MIN_PRIOR, ["kd_short", "kd_long"]] = np.nan
    latest_out.parent.mkdir(parents=True, exist_ok=True)
    latest.to_parquet(latest_out, index=False)

    known = latest[latest["n_runs"] >= MIN_PRIOR]
    asof_c = r.reindex(d.index[d["year"].isin(DEV + TEST + REF)])
    latest_summary = {"units": int(len(latest)), "unknown_share_units": round(float((latest["n_runs"] < MIN_PRIOR).mean()), 4),
                      "kd_short": qdict(known["kd_short"]), "kd_long": qdict(known["kd_long"]), "width": qdict(known["kd_long"] - known["kd_short"]),
                      "n_runs": qdict(latest["n_runs"]), "single_share_known": round(float((known["kd_long"] == known["kd_short"]).mean()), 4),
                      "unknown_share_asof_runs_2022_2026": round(float(asof_c["kd_short"].isna().mean()), 4)}
    for k, v in (("kd_short_dist", known["kd_short"]), ("kd_long_dist", known["kd_long"])):
        latest_summary[k] = {str(int(x)): int(c) for x, c in v.value_counts().sort_index().items()}

    meta = {"built": date.today().isoformat(), "years": [YEARS.start, YEARS.stop - 1], "dev": list(DEV), "test": list(TEST), "ref": list(REF),
            "taus": list(TAUS), "min_prior": MIN_PRIOR, "bootstrap": BOOT, "exp_years": [asc.EXP_YEARS.start, asc.EXP_YEARS.stop - 1],
            "selection_rule": "dev 2022-2024 で 複勝率差・勝率差の 95% 区間の下限がともに 0 超の候補のうち 複勝率差（人気補正後・中−外）が最大。該当なしなら複勝率差が最大",
            "expected_note": "期待値表は 2016〜2023 年で学習。開発年の 2022・2023 年を含む（依頼どおり）。2025・2026 年は含まない",
            "eval_rows": "全候補で共通（合算・芝ダ別の両方でレンジが出る走）", "common_rows": int(common.sum()), "unaligned_chosen": unaligned_chosen,
            "latest_file": str(latest_out), "latest_unit": "芝ダ別なら 馬×馬場、合算なら 馬" }
    out_json.write_text(json.dumps({"meta": meta, "points": pts_sum, "distribution": dist_summary, "dev": dev_rows, "dev_unaligned": dev_unaligned, "qualified": list(qual), "chosen": chosen,
                                    "chosen_by_year": by_year, "test2025": test, "ref2026": ref, "latest": latest_summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out_json, latest_out)


if __name__ == "__main__":
    main()
