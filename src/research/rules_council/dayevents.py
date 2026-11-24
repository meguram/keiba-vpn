"""RC-5 の段 0（着順を使わない量の計算）: 当日の出来事の数と、その後の走の数（年あたり）。

- 馬場状態の発表の替わり: 同じ日・同じ場・同じ芝ダで、前のレースと馬場状態が違うレース（替わった後のレースを数える）。
- 当日の乗り替わり: 出馬表（race_shutuba_flat）の騎手と、結果（race_result_flat）の騎手が違う走。
- 取消・除外: 出馬表にいて、結果で着順が付かない（取消・除外）馬がいるレース。逃げ候補（前走までの逃げ率 ≥ 0.3）が抜けたレースも数える。
- 母集団の走: 再現の T-60 < 10 倍・≤ 50 倍（full.parquet と full_pre.npy の種 1000）。着順・払戻は読まない（取消の判定に着順の欠けだけを使う）。
- 出力: docs/operations/retrain_trials/rules_council_rc5_stage0.json。費用 0 円（手元の表だけ）。
使い方: python -m src.research.rules_council.dayevents（段 0）・dayevents stage0b（段 0b）。nice -n 19
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.rules_council import rc_frame as F

TABLES = F.CR.REPO / "data/page_reference/tables"


def main() -> None:
    import sys
    if sys.argv[1:] == ["stage0b"]:
        cmd_stage0b()
        return
    res, shu = [], []
    for y in range(2016, F.YEAR_END + 1):
        r = pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet",
                            columns=["race_id", "date", "venue", "surface", "start_time", "track_condition", "horse_number", "horse_id", "jockey_id", "finish_position"])
        res.append(r)
        p = TABLES / str(y) / "race_shutuba_flat.parquet"
        if p.exists():
            s = pd.read_parquet(p, columns=["race_id", "horse_number", "horse_id", "jockey_id"])
            shu.append(s)
    r = pd.concat(res, ignore_index=True)
    r["race_id"] = r["race_id"].astype(str)
    r["surf"] = r["surface"].astype(str).str[0]
    r = r[r["surf"].isin(["芝", "ダ"])].copy()
    fin = pd.to_numeric(r["finish_position"], errors="coerce")
    full = pd.read_parquet(F.OUTW / "full.parquet", columns=["race_id", "horse_id", "year"])
    full["race_id"], full["horse_id"] = full["race_id"].astype(str), full["horse_id"].astype(str)
    pre = np.load(F.OUTW / "full_pre.npy")[0]
    full["fav"], full["le50"] = pre < 10, pre <= 50
    out = {"years": [2016, F.YEAR_END]}

    def per_year(race_ids: set, horses: pd.DataFrame | None = None) -> dict:
        x = full[full["race_id"].isin(race_ids)] if horses is None else full.merge(horses, on=["race_id", "horse_id"])
        n_y = x.groupby("year").size()
        return {"races": int(len(race_ids)), "fav_runs_per_year": float(x["fav"].sum() / 9), "le50_runs_per_year": float(x["le50"].sum() / 9),
                "fav_runs_2024": int(x[x["year"] == 2024]["fav"].sum()), "runs_by_year": {int(k): int(v) for k, v in n_y.items()}}

    # 1. 馬場状態の発表の替わり
    race = r.drop_duplicates("race_id")[["race_id", "date", "venue", "surf", "start_time", "track_condition"]].sort_values(["date", "venue", "surf", "start_time"])
    race["prev_tc"] = race.groupby(["date", "venue", "surf"])["track_condition"].shift(1)
    race["changed_once"] = race.groupby(["date", "venue", "surf"])["track_condition"].transform(lambda s: (s != s.iloc[0]).cumsum() > 0)
    after = set(race.loc[race["changed_once"], "race_id"])
    out["going_change_after"] = per_year(after)
    out["going_change_after"]["note"] = "同じ日・場・芝ダで、最初のレースと馬場状態が替わった後のレース（発表の時刻は表に無い。T-60 に分かるかは要確かめ）"
    # 2. 当日の乗り替わり
    if shu:
        s = pd.concat(shu, ignore_index=True)
        s["race_id"] = s["race_id"].astype(str)
        m = r.merge(s, on=["race_id", "horse_number"], how="inner", suffixes=("", "_shu"))
        ch = m[(m["jockey_id"].astype(str) != m["jockey_id_shu"].astype(str)) & m["jockey_id_shu"].notna() & m["jockey_id"].notna()]
        out["jockey_change"] = per_year(set(ch["race_id"]), ch[["race_id", "horse_id"]].astype(str).drop_duplicates())
        out["jockey_change"]["matched_runs"] = int(len(m))
        out["jockey_change"]["changed_runs"] = int(len(ch))
        out["jockey_change"]["note"] = "出馬表（取得の時刻は前日〜当日で揃っていない見込み）と結果の騎手が違う走。出馬表の年: " + ",".join(sorted({str(x)[:4] for x in s["race_id"].unique()}))
    # 3. 取消・除外
    scr = r[fin.isna() | (fin <= 0)]
    out["scratch_races"] = per_year(set(scr["race_id"]))
    out["scratch_races"]["scratched_runs"] = int(len(scr))
    out["scratch_races"]["note"] = "結果の表で着順が付かない馬（取消・除外・中止を含む。中止は発走後なので T-60 には分からない。区別は要確かめ）がいるレース"
    p = F.CR.REPO / "docs/operations/retrain_trials/rules_council_rc5_stage0.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "runs_by_year"} if isinstance(v, dict) else v for k, v in out.items()}, ensure_ascii=False, indent=1))


GOING_RANK = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}


def cmd_stage0b() -> dict:
    """RC-5 の段 0b（着順を使わない）。1 号の決定の H1 と、取消の区別の監査。
    H1: 今回の発走の 75 分前までに発走した同じ日・場・芝ダの直前のレースの馬場状態が、その日の最初のレースより悪い × 道悪の実績がある馬
    （前の日までの稍重以上での 3 着内 ≥ 1）。取消の区別: finish_status（2016〜19 年）で出走取消・競走除外・中止を分ける。"""
    rr = []
    for y in range(2016, F.YEAR_END + 1):
        rr.append(pd.read_parquet(TABLES / str(y) / "race_result_race.parquet", columns=["race_id", "date", "venue", "surface", "start_time", "track_condition"]))
    race = pd.concat(rr, ignore_index=True)
    race["race_id"] = race["race_id"].astype(str)
    race["surf"] = race["surface"].astype(str).str[0]
    race = race[race["surf"].isin(["芝", "ダ"])].copy()
    race["g"] = race["track_condition"].astype(str).map(lambda v: GOING_RANK.get(v, GOING_RANK.get(v[:1], np.nan)))
    t = pd.to_datetime(race["start_time"].astype(str), format="%H:%M", errors="coerce")
    race["min"] = t.dt.hour * 60 + t.dt.minute
    race = race.sort_values(["date", "venue", "surf", "min"])
    worse = {}
    for _, g in race.groupby(["date", "venue", "surf"]):
        m, gg, ids = g["min"].to_numpy(), g["g"].to_numpy(), g["race_id"].to_numpy()
        for k in range(len(g)):
            ok = np.flatnonzero(m <= m[k] - 75)
            worse[ids[k]] = bool(len(ok) >= 1 and gg[ok[-1]] > gg[0]) if len(ok) else False
    # 道悪の実績（1 号の決定の定義: as-of で稍重以上の 3 着内が 1 回以上）。前のレースの着順から、日をずらした累計で作る（今回のレースは入れない）
    hist = []
    for y in range(2016, F.YEAR_END + 1):
        h = pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "date", "horse_id", "finish_position", "track_condition", "surface"])
        hist.append(h[h["surface"].astype(str).str[0].isin(["芝", "ダ"])])     # 障害は実績に数えない
    h = pd.concat(hist, ignore_index=True)
    h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
    fp = pd.to_numeric(h["finish_position"], errors="coerce")
    gr = h["track_condition"].astype(str).map(lambda v: GOING_RANK.get(v, GOING_RANK.get(v[:1], np.nan)))
    h["wet3"] = ((fp >= 1) & (fp <= 3) & (gr >= 1)).astype(float)
    h = h.sort_values(["horse_id", "date"])
    day = h.groupby(["horse_id", "date"])["wet3"].sum().groupby(level=0).cumsum() - h.groupby(["horse_id", "date"])["wet3"].sum()
    h = h.merge(day.rename("wet3_prev").reset_index(), on=["horse_id", "date"], how="left")
    fr = h[["race_id", "horse_id", "wet3_prev"]].drop_duplicates(["race_id", "horse_id"])
    full = pd.read_parquet(F.OUTW / "full.parquet", columns=["race_id", "horse_id", "year"])
    full["race_id"], full["horse_id"] = full["race_id"].astype(str), full["horse_id"].astype(str)
    full["pre"] = np.load(F.OUTW / "full_pre.npy")[0]
    x = full.merge(fr, on=["race_id", "horse_id"], how="left")
    x["worse"] = x["race_id"].map(worse).fillna(False).astype(bool)
    h1 = x["worse"] & (x["wet3_prev"] >= 1)
    out = {"H1": {"races_worse": int(sum(worse.values())), "fav_runs_per_year": float((h1 & (x["pre"] < 10)).sum() / 9),
                  "fav_runs_2024": int((h1 & (x["pre"] < 10) & (x["year"] == 2024)).sum()),
                  "le50_runs_per_year": float((h1 & (x["pre"] <= 50)).sum() / 9),
                  "note": "道悪の実績は、前の日までの稍重・重・不良での 3 着内の数 ≥ 1（芝ダの race_result_flat から作った as-of。表は 2016 年からなので 2016〜17 年は低め。前の版は frame の bad_top3_prev＝重・不良だけで、定義と違った）"}}
    # 取消の区別の監査: 2016〜19 年は finish_status（scratched・excluded・dnf）がある。2020 年以降は列が無いか空。芝ダだけ
    aud, sc = {}, []
    for y in range(2016, F.YEAR_END + 1):
        f = TABLES / str(y) / "race_result_flat.parquet"
        import pyarrow.parquet as pq
        names = pq.ParquetFile(f).schema_arrow.names
        cols = ["race_id", "surface", "finish_position", "passing_order", "odds"] + (["finish_status"] if "finish_status" in names else [])
        r = pd.read_parquet(f, columns=cols)
        r = r[r["surface"].astype(str).str[0].isin(["芝", "ダ"]) & (r["finish_position"].astype(str) == "-1")]
        st = r["finish_status"].astype(str) if "finish_status" in r else pd.Series(["None"] * len(r), index=r.index)
        aud[y] = {"no_finish": int(len(r)), **{k: int(v) for k, v in st.value_counts().items()}}
        sc.append(r.loc[st == "scratched", ["race_id"]].assign(year=y))
    out["scratch_audit"] = aud
    scr = pd.concat(sc, ignore_index=True)
    yrs = {y for y, v in aud.items() if v.get("scratched", 0) >= 10}
    races = set(scr.loc[scr["year"].isin(yrs), "race_id"].astype(str))
    fav_in = full[full["race_id"].isin(races) & (full["pre"] < 10)]
    n_years = len({y for y, v in aud.items() if v.get("scratched", 0) >= 10})   # 2024 年の 1 頭のような年は数えない
    out["scratch_races_known"] = {"races": int(len(races)), "years_with_status": n_years,
                                  "fav_runs_per_year": float(len(fav_in) / max(n_years, 1)),
                                  "note": "出走取消が分かるのは finish_status のある年だけ。一桁の走は、逃げ候補・先行型の条件を掛ける前の上限"}
    p_ = F.CR.REPO / "docs/operations/retrain_trials/rules_council_rc5_stage0b.json"
    p_.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    main()
