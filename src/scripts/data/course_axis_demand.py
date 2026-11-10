"""良馬場のコース別に、走法軸（地図の 6 軸）が好走にどのくらい効くかを測り直す（読むだけ・JSON を書く）。

背景（2026-11-10 ユーザー指摘）: 走法マップの寄与率は、どのコースでも Hペース耐性が 7〜9 割を占める。コース形状と結びつかない。
  - 現行の寄与率 = 「得点 − 頭数だけの期待」を 6 軸へ回帰したときの、軸ごとの R² の落ち（コース内で合計 100% に規格化）。
    期待が人気を見ない（running_map_course.POP_TABLE=None）ので、実力の代理になる軸が全コースで勝つ疑いがある。
  - ここでは (A) 現行の再現、(B) 人気込みの期待（実力を市場で除く）、(C) B から「芝/ダート全体の平均」を引いたコース固有の差、
    (D) クラス別（未勝利・新馬 / 1勝 / 2・3勝 / OP以上）を出す。
使い方（ホストで）: cd <repo> && nice -n 19 /opt/conda/bin/python -I src/scripts/data/course_axis_demand.py OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()   # リポジトリの直下で実行する（出走記録の表の相対パスもここから）
sys.path.insert(0, str(REPO))

from src.research.race import pop_expect  # noqa: E402
from src.research.race import running_map as rm  # noqa: E402
from src.research.race import running_map_aptitude as apt  # noqa: E402
from src.research.race import running_map_course as rmc  # noqa: E402

KEYS = list(rmc.MAP_AXES)
CLASS_BANDS = [("未勝利・新馬", ("未勝利", "新馬")), ("1勝", ("1勝",)), ("2・3勝", ("2勝", "3勝")), ("OP以上", ("OP", "L", "G3", "G2", "G1"))]
STRAIGHT = {"短い直線(札幌・函館・福島・小倉・中山)": ("札幌", "函館", "福島", "小倉", "中山"),
            "中間(京都・阪神・新潟)": ("京都", "阪神", "新潟"), "長い直線(東京・中京)": ("東京", "中京")}
MIN_CELL = 250     # 回帰を出す最低の走数


def band_of_grade(g: object) -> str | None:
    for name, members in CLASS_BANDS:
        if g in members:
            return name
    return None


LEVEL_BANDS = ["Lv1(低)", "Lv2", "Lv3", "Lv4(高)"]
MIN_LEVEL_HORSES = 5     # レースの水準を出すのに必要な、タイム指数のある馬の数


def race_level(years: list[int]) -> pd.Series:
    """レースの水準 = 出走馬の「出走前のタイム指数の平均（speed_avg）」の平均。0 は欠け。有効な馬が MIN_LEVEL_HORSES 頭未満または出走の半数未満のレースは NaN。
    出走前に分かる値なので、結果は使っていない（新馬戦は過去の指数が無く NaN）。"""
    parts = []
    for y in years:
        p = Path(pop_expect.TABLE_DIR) / str(y) / "race_index_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["race_id", "horse_id", "speed_avg"]))
    d = pd.concat(parts, ignore_index=True)
    d["race_id"] = d["race_id"].astype(str)
    d["v"] = pd.to_numeric(d["speed_avg"], errors="coerce").where(lambda x: x != 0)
    g = d.groupby("race_id")["v"].agg(["mean", "count", "size"])
    ok = (g["count"] >= MIN_LEVEL_HORSES) & (g["count"] >= 0.5 * g["size"])
    return g["mean"].where(ok)


def load_table() -> tuple[pd.DataFrame, np.ndarray]:
    """良馬場・2024 年以降・6 軸がそろう走の表と、その 6 軸の z（標準偏差 1 にそろえる）。"""
    d = rm._data()
    metrics, ref = d["metrics"], d["ref"]
    table = rm.rp.profile_table(metrics, d["sires"], cols=[col for _, col, *_ in rm.AXES])
    q = {k: np.array(v) for k, v in ref["quantiles"].items()}
    Z = np.column_stack([rm._to_normal(table[col].to_numpy(dtype=float), q[key]) for key, col, *_ in rm.AXES]) / np.array(ref["sd"]) * rm.SIGNS
    ids = ["race_id", "horse_id", "date"]
    m = metrics[ids + ["venue", "surface", "distance", "track_condition", "finish_position"]].drop_duplicates(ids)
    m = m.assign(n_field=m.groupby("race_id")["race_id"].transform("size"))
    tab = table[ids].merge(m, on=ids, how="left")
    years = sorted(tab["date"].dt.year.unique())
    pf = pop_expect.popularity_frame(years)[["race_id", "horse_id", "popularity"]]
    tab = tab.merge(pf, on=["race_id", "horse_id"], how="left")
    parts = []
    for y in years:
        p = Path(pop_expect.TABLE_DIR) / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["race_id", "grade"]).drop_duplicates("race_id"))
    gr = pd.concat(parts, ignore_index=True).drop_duplicates("race_id")
    gr["race_id"] = gr["race_id"].astype(str)
    tab["race_id"] = tab["race_id"].astype(str)
    tab = tab.merge(gr, on="race_id", how="left")
    cols = [rm.AXIS_KEYS.index(k) for k in KEYS]
    Zk = Z[:, cols]
    # 人気込みの期待は、2016〜2023 年の走で作った表（評価期間の 2024 年以降を含めない）
    ptab = pop_expect.build_table(pop_expect.popularity_frame(range(2016, 2024)))
    ok = ((tab["date"] >= pd.Timestamp(rmc.SINCE)) & (tab["n_field"] >= rmc.MIN_FIELD) & (tab["track_condition"] == "良")
          & tab[["venue", "surface", "distance", "finish_position"]].notna().all(axis=1).to_numpy() & ~np.isnan(Zk).any(axis=1)).to_numpy()
    t = tab[ok].reset_index(drop=True)
    Zk = Zk[ok]
    t["y"] = rmc.score_of(t["finish_position"].to_numpy())
    t["e_n"] = rmc.expected_score(t["n_field"].to_numpy())
    t["e_pop"] = pop_expect.expect(ptab, t["n_field"].to_numpy(), t["popularity"].to_numpy(), t["e_n"].to_numpy())
    t["ckey"] = [apt.course_key(v, s, dd) for v, s, dd in zip(t["venue"], t["surface"], t["distance"])]
    t["gkey"] = [rmc.group_key(s, dd) for s, dd in zip(t["surface"], t["distance"])]
    t["cls"] = [band_of_grade(g) for g in t["grade"]]
    lv = race_level([int(y) for y in years])
    t["level"] = t["race_id"].map(lv)
    one = t.drop_duplicates("race_id")["level"].dropna()
    cuts = np.quantile(one, [0.25, 0.5, 0.75])
    t["lvl"] = [None if pd.isna(v) else LEVEL_BANDS[int(np.searchsorted(cuts, v, side="right"))] for v in t["level"]]
    t.attrs["level_cuts"] = [round(float(c), 1) for c in cuts]
    return t, Zk


def fit(Z: np.ndarray, r: np.ndarray, race: np.ndarray) -> dict:
    """r を Z へ回帰。標準化した係数・レース単位でまとめた標準誤差・R² の落ちによる寄与率。"""
    sd = Z.std(0)
    Zs = (Z - Z.mean(0)) / np.where(sd > 0, sd, 1)
    drop, beta, full = rmc._importance(Zs, r)
    X = np.column_stack([np.ones(len(Zs)), Zs])
    b = np.linalg.lstsq(X, r, rcond=None)[0]
    u = r - X @ b
    codes = pd.factorize(race)[0]
    S = np.zeros((X.shape[1], codes.max() + 1))
    for j in range(X.shape[1]):
        S[j] = np.bincount(codes, weights=X[:, j] * u)
    bread = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(bread @ (S @ S.T) @ bread))[1:]
    share = drop / drop.sum() if drop.sum() > 0 else np.zeros(len(drop))
    return {"n": int(len(r)), "races": int(len(set(codes))), "r2": round(float(full), 4),
            "beta": [round(float(v), 4) for v in b[1:]], "se": [round(float(v), 4) for v in se],
            "share": [round(float(v), 3) for v in share], "drop": [round(float(v), 5) for v in drop]}


def dev_fit(Z: np.ndarray, r: np.ndarray, race: np.ndarray, base: np.ndarray) -> dict:  # base: 先頭が切片、残りが軸ごとの係数（行 = 走）
    """コース固有の差。r から「同じ芝ダ・同じクラスの平均の係数 × 軸」を引いた残りを、軸へ回帰する（係数 = 平均からのずれ）。
    Z は芝ダ全体の標準偏差で割った値。クラス構成の違い（地方は未勝利が多い、など）はここで除かれる。se はレース単位でまとめた値。"""
    adj = r - base[:, 0] - (Z * base[:, 1:]).sum(1)
    X = np.column_stack([np.ones(len(Z)), Z])
    b = np.linalg.lstsq(X, adj, rcond=None)[0]
    u = adj - X @ b
    codes = pd.factorize(race)[0]
    S = np.zeros((X.shape[1], codes.max() + 1))
    for j in range(X.shape[1]):
        S[j] = np.bincount(codes, weights=X[:, j] * u)
    bread = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(bread @ (S @ S.T) @ bread))[1:]
    return {"dbeta": [round(float(v), 4) for v in b[1:]], "dse": [round(float(v), 4) for v in se]}


def class_base(Zs: np.ndarray, r: np.ndarray, surf_mask: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """芝ダ × クラスごとの回帰（切片と軸の係数）を、各走へ割り当てる（行 = 走、列 = 切片 + 軸）。クラス不明の走は芝ダ全体。"""
    base = np.zeros((len(Zs), Zs.shape[1] + 1))
    base[surf_mask] = np.linalg.lstsq(np.column_stack([np.ones(surf_mask.sum()), Zs[surf_mask]]), r[surf_mask], rcond=None)[0]
    for cname, _ in CLASS_BANDS:
        m = surf_mask & (cls == cname)
        if m.sum() >= MIN_CELL:
            base[m] = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), Zs[m]]), r[m], rcond=None)[0]
    return base


def run(t: pd.DataFrame, Z: np.ndarray) -> dict:
    out: dict = {"axes": KEYS, "courses": {}, "groups": {}, "surface": {}, "class_groups": {}, "class_surface": {}, "axis_vs_ability": {}}
    r_a = (t["y"] - t["e_n"]).to_numpy()
    r_b = (t["y"] - t["e_pop"]).to_numpy()
    race = t["race_id"].to_numpy()
    cls = t["cls"].to_numpy()
    Zs = np.zeros_like(Z)
    bases: dict[str, np.ndarray] = {}
    for surf in ("芝", "ダート"):
        ms = (t["surface"] == surf).to_numpy()
        Zs[ms] = (Z[ms] - Z[ms].mean(0)) / Z[ms].std(0)
        bases[surf] = ms
    base_a = class_base(Zs, r_a, bases["芝"], cls) + class_base(Zs, r_a, bases["ダート"], cls)
    base_b = class_base(Zs, r_b, bases["芝"], cls) + class_base(Zs, r_b, bases["ダート"], cls)

    def both(mask: np.ndarray) -> dict | None:
        if mask.sum() < MIN_CELL:
            return None
        return {"A": fit(Z[mask], r_a[mask], race[mask]), "B": fit(Z[mask], r_b[mask], race[mask])}

    out["level_surface"], out["level_groups"] = {}, {}
    for surf in ("芝", "ダート"):
        ms = (t["surface"] == surf).to_numpy()
        out["surface"][surf] = both(ms)
        for cname, _ in CLASS_BANDS:
            out["class_surface"][f"{surf}|{cname}"] = both(ms & (t["cls"] == cname).to_numpy())
        for lname in LEVEL_BANDS:
            out["level_surface"][f"{surf}|{lname}"] = both(ms & (t["lvl"] == lname).to_numpy())
    def dev(mask: np.ndarray) -> dict:
        return {"A": dev_fit(Zs[mask], r_a[mask], race[mask], base_a[mask]), "B": dev_fit(Zs[mask], r_b[mask], race[mask], base_b[mask])}

    for gk in sorted(t["gkey"].unique()):
        mg = (t["gkey"] == gk).to_numpy()
        out["groups"][gk] = both(mg)
        if out["groups"][gk]:
            out["groups"][gk]["dev"] = dev(mg)
        for cname, _ in CLASS_BANDS:
            out["class_groups"][f"{gk}|{cname}"] = both(mg & (t["cls"] == cname).to_numpy())
        for lname in LEVEL_BANDS:
            out["level_groups"][f"{gk}|{lname}"] = both(mg & (t["lvl"] == lname).to_numpy())
    for ck in sorted(t["ckey"].unique()):
        res = both((t["ckey"] == ck).to_numpy())
        if res:
            g = t.loc[t["ckey"] == ck, "gkey"].iloc[0]
            res["gkey"] = g
            res["dev"] = dev((t["ckey"] == ck).to_numpy())
            res["class_n"] = {c: int(((t["ckey"] == ck) & (t["cls"] == c)).sum()) for c, _ in CLASS_BANDS}
            out["courses"][ck] = res
    out["venue_dev"], out["straight_dev"] = {}, {}
    for surf in ("芝", "ダート"):
        ms = (t["surface"] == surf).to_numpy()
        for v in sorted(t["venue"].unique()):
            m = ms & (t["venue"] == v).to_numpy()
            if m.sum() >= MIN_CELL:
                out["venue_dev"][f"{v}|{surf}"] = {"n": int(m.sum()), "races": int(t.loc[m, "race_id"].nunique()), **dev(m)}
        for name, vs in STRAIGHT.items():
            m = ms & t["venue"].isin(vs).to_numpy()
            if m.sum() >= MIN_CELL:
                out["straight_dev"][f"{surf}|{name}"] = {"n": int(m.sum()), "races": int(t.loc[m, "race_id"].nunique()), **dev(m)}
    out["level_by_class"] = {c: {l: int(((t["cls"] == c) & (t["lvl"] == l)).sum()) for l in LEVEL_BANDS} for c, _ in CLASS_BANDS}
    # 軸が「実力の代理」になっていないか: 軸の z と、人気込みの期待得点・頭数だけの期待得点との相関（芝・ダート別）
    for surf in ("芝", "ダート"):
        ms = (t["surface"] == surf).to_numpy()
        ab = (t.loc[ms, "e_pop"] - t.loc[ms, "e_n"]).to_numpy()
        out["axis_vs_ability"][surf] = [round(float(np.corrcoef(Z[ms][:, j], ab)[0, 1]), 3) for j in range(len(KEYS))]
    return out


if __name__ == "__main__":
    t, Z = load_table()
    res = run(t, Z)
    res["meta"] = {"since": rmc.SINCE, "n_runs": int(len(t)), "n_races": int(t["race_id"].nunique()),
                   "class_bands": [c for c, _ in CLASS_BANDS], "min_cell": MIN_CELL,
                   "class_counts": {k: int(v) for k, v in t["cls"].value_counts(dropna=False).items() if k},
                   "level_cuts": t.attrs.get("level_cuts"), "level_counts": {k: int(v) for k, v in t["lvl"].value_counts().items()},
                   "level_missing_runs": int(t["lvl"].isna().sum()), "popularity_missing_runs": int(t["popularity"].isna().sum())}
    Path(sys.argv[1]).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("ok", res["meta"])
