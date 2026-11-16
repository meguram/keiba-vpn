"""重賞のルールベース予想家: パターン（切り口 × 水準）の [1-2-3-着外] を、層ごとに as-of で数える。

層（docs/ideas/graded_rule_based_handicapper.md 第 3 節）:
- main（主）: 同じレース名 × 同じコースの過去の回（OP 以上の行。格付けが変わった年も同じレース）
- sub（副）: 同じコースの OP 以上のレースから、主のレース名の行を除いたもの。件数が足りないときに使う 3 勝クラス以上の版（sub4）も数える
- glob（全体）: すべての重賞
数えるのは「予想するレースの日より前」に終わったレースだけ（同じ日のレースは入れない）。検証の凍結版（件数を 2024 年末で止める）も作る。

各パターンの値: n（走数）・w1・w2・w3（1・2・3 着の数）・e3（期待の 3 着内 = Σ 3/頭数）・e1（期待の勝ち = Σ 1/頭数）・
ed（そのパターンの馬がいたレースの数）・edhit（そのパターンの馬が 3 着内に来たレースの数）。
回収率の物差し（評価にだけ使い、切り口・点には使わない）: np（払戻が解けたレースの走数）・rt・rf（単勝・複勝を 100 円ずつ買ったときの
払戻の倍率の和）・rtc・rfc（単勝 50 倍・複勝 10 倍で打ち切った版）。回収率 = rt / np。払戻の列が無い表（テストなど）では 0。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.graded.catalog import UNKNOWN

STATS = ["n", "w1", "w2", "w3", "e3", "e1", "ed", "edhit", "np", "rt", "rf", "rtc", "rfc"]
RET = ["np", "rt", "rf", "rtc", "rfc"]
LAYERS = ("main", "sub", "sub4", "glob")
GRADES = ("G1", "G2", "G3")
FREEZE_END = pd.Timestamp("2024-12-31")


def base_frame(d: pd.DataFrame) -> pd.DataFrame:
    """集計に使う行（3 勝クラス以上の平地）と、結果の列。"""
    p = d[(pd.to_numeric(d["cls"], errors="coerce") >= 4)].copy()
    fin = pd.to_numeric(p["fin"], errors="coerce")
    field = pd.to_numeric(p["field"], errors="coerce")
    b = pd.DataFrame({
        "row": p.index.to_numpy(), "race_id": p["race_id"].astype(str).to_numpy(), "dt": p["dt"].to_numpy(),
        "rkey": p["rkey"].astype(str).to_numpy(), "course": p["course"].astype(str).to_numpy(),
        "graded": p["grade"].isin(GRADES).to_numpy(), "op": (pd.to_numeric(p["cls"], errors="coerce") >= 5).to_numpy(),
        "w1": (fin == 1).astype(float).to_numpy(), "w2": (fin == 2).astype(float).to_numpy(), "w3": (fin == 3).astype(float).to_numpy(),
        "e3": (3.0 / field).clip(upper=1).to_numpy(), "e1": (1.0 / field).to_numpy(),
    })
    b["top3"] = b["w1"] + b["w2"] + b["w3"]
    if "ret_tan" in p.columns:
        ok = p["pay_ok"].fillna(False).astype(bool).to_numpy()
        b["np"] = ok.astype(float)
        for c, src in (("rt", "ret_tan"), ("rf", "ret_fuku"), ("rtc", "ret_tan_c"), ("rfc", "ret_fuku_c")):
            b[c] = np.where(ok, pd.to_numeric(p[src], errors="coerce").fillna(0.0).to_numpy(), 0.0)
    else:
        for c in RET:
            b[c] = np.nan
    return b


def _asof(b: pd.DataFrame, gid: np.ndarray, mask: np.ndarray, targets: pd.DataFrame, freeze: bool) -> pd.DataFrame:
    """gid（グループの番号）ごとに、各日より前の累計を作り、targets（row, gid, dt）に付ける。"""
    x = b.loc[mask, ["race_id", "dt", "w1", "w2", "w3", "e3", "e1", "top3"] + RET].copy()
    x["gid"] = gid[mask]
    r = x.groupby(["gid", "race_id"], sort=False).agg(dt=("dt", "first"), n=("w1", "size"), w1=("w1", "sum"), w2=("w2", "sum"),
                                                     w3=("w3", "sum"), e3=("e3", "sum"), e1=("e1", "sum"), hit=("top3", "max"),
                                                     **{c: (c, "sum") for c in RET}).reset_index()
    r["ed"] = 1.0
    r = r.rename(columns={"hit": "edhit"})
    if freeze:
        tot = r[r["dt"] <= FREEZE_END].groupby("gid")[STATS].sum()
        return targets[["row", "gid"]].join(tot, on="gid").drop(columns="gid").fillna(0.0)
    dd = r.groupby(["gid", "dt"], sort=True)[STATS].sum().reset_index()
    cs = dd.groupby("gid", sort=False)[STATS].cumsum() - dd[STATS]   # その日より前の累計
    dd[STATS] = cs
    t = targets[["row", "gid", "dt"]].merge(dd, on=["gid", "dt"], how="left")
    # 対象の行（重賞）は、どの層でも数える側の行に入っているので、同じ (gid, 日) が必ずある
    assert not t["n"].isna().any(), "対象の行が数える側に無い"
    return t.drop(columns=["gid", "dt"]).fillna(0.0)


def tally_type(b: pd.DataFrame, lv: pd.Series, tmask: np.ndarray, freeze: bool = False) -> pd.DataFrame:
    """1 つの型について、対象の行（tmask）の水準ごとの as-of の件数を、4 つの層で返す。列は <層>_<統計>。"""
    lev = pd.Series(lv.reindex(b["row"]).to_numpy(), index=b.index).fillna(UNKNOWN).astype(str)
    lcode = pd.factorize(lev)[0]
    rk = pd.factorize(b["rkey"])[0]
    co = pd.factorize(b["course"])[0]
    op = b["op"].to_numpy()
    graded = b["graded"].to_numpy()
    allm = np.ones(len(b), dtype=bool)
    gids = {
        "main": pd.factorize(pd.Series(lcode).astype(str) + "|" + pd.Series(rk).astype(str) + "|" + pd.Series(co).astype(str))[0],
        "course": pd.factorize(pd.Series(lcode).astype(str) + "|" + pd.Series(co).astype(str))[0],
        "glob": lcode,
    }
    tg = b.loc[tmask, ["row", "dt"]].copy()
    res = {}
    for name, gk, m in (("main", "main", op), ("course_op", "course", op), ("course_4", "course", allm), ("main4", "main", allm), ("glob", "glob", graded)):
        t = tg.copy()
        t["gid"] = gids[gk][tmask]
        res[name] = _asof(b, gids[gk], m, t, freeze).set_index("row")
    out = pd.DataFrame(index=tg["row"].to_numpy())
    for s in STATS:
        out[f"main_{s}"] = res["main"][s]
        out[f"sub_{s}"] = res["course_op"][s] - res["main"][s]
        out[f"sub4_{s}"] = res["course_4"][s] - res["main4"][s]
        out[f"glob_{s}"] = res["glob"][s]
    out["level"] = lev[tmask].to_numpy()
    return out


def tally_all(d: pd.DataFrame, L: pd.DataFrame, target_mask: pd.Series, type_ids: list[str], freeze: bool = False) -> pd.DataFrame:
    """全型の件数の長い表（row, type, level, <層>_<統計>）。型「ALL」は水準を 1 つにした母集団全体（主の回の数などに使う）。"""
    b = base_frame(d)
    tmask = target_mask.reindex(b["row"]).fillna(False).to_numpy()
    parts = []
    L = L.copy()
    L["ALL"] = "全体"
    for i, tid in enumerate(["ALL"] + list(type_ids)):
        t = tally_type(b, L[tid], tmask, freeze)
        t["type"] = tid
        parts.append(t.rename_axis("row").reset_index())
        if i % 20 == 0:
            print("tally", tid, i, flush=True)
    out = pd.concat(parts, ignore_index=True)
    for c in out.columns:
        if c not in ("row", "type", "level"):
            out[c] = out[c].astype("float32")
    return out
