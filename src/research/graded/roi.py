"""重賞のルールベース予想家: 回収率で裏付けたパターンだけで印を付ける版（2 回目の評価。計画書 第 17 節）。

考え方:
- パターン = 型 × 水準 × 向き（好 = そのレースの as-of の点 x が正、凡 = 負）。x を作った層（主・副）も記録する。
- 2016〜2024 年の中で、発見（2018〜2021）と確認（2022〜2024）の重賞で、そのパターンに当てはまった馬を全頭 100 円ずつ
  買ったときの単勝・複勝の回収率を実際の払戻で数え、同じ期間の重賞の全頭の回収率（約 0.75〜0.8）と比べる。
  区間はレース単位のブートストラップ（レースごとにポアソン(1) の重みを付ける近似、N_BOOT 回）。
- 好のパターンは「回収率が全体を上回る」、凡のパターンは「下回る」ことを、両方の期間で満たすものだけを残す。
  あわせて、成績の向き（3 着内の数が期待 Σ3/頭数 を上回る／下回る）も両方の期間で満たすことを条件にする。
- 基準は 1 つに絞らず、GRID（件数が多い版 192）と SMALL_GRID（件数が少なく過学習のおそれがある版 108）の全組み合わせ（300 版）を並べる。主に見る版（PRIMARY）は結果を見る前に決めた。
- 印: 版ごとに、残ったパターンの x だけを足し（ほかは 0）、等しい重みで点にする。縮める強さ m・主の重み λ は第 1 版の
  prereg.json のまま（第 1 版と同じ x）。束ね（|相関| ≥ 0.7）は残ったパターンの x で、発見期間で作り直す。
回収率（払戻）は評価の物差しにだけ使い、切り口・点の x には入れない。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.research.graded import catalog as CAT
from src.research.graded import model as M

BETS = {"単勝": ("ret_tan", "ret_tan_c"), "複勝": ("ret_fuku", "ret_fuku_c")}
GRID = {
    "bet": ["複勝", "単勝", "両方"],
    "crit": ["区間の下限", "点＋0.10"],
    "cap": ["頭打ちでも同じ向き", "頭打ちを見ない"],
    "nmin": [20, 50],
    "layer": ["主＋副", "主だけ"],
    "base": ["採用した型", "全部の型"],
    # 同点の並べ方。回収率で残るのが凡のパターンだけの版では、多くの馬の点が 0 で並ぶ。
    # 「成績版の点」= 同点を第 1 版（成績だけで選んだ版）の点で並べる（成績で並べ、回収率で裏付けた凡の馬を下げる形）。
    # 「馬番」= 回収率で残ったパターンだけで並べ、同点は主の好成績パターンの数 → 馬番（純粋な版）
    "tie": ["成績版の点", "馬番"],
}
# 件数が少なく過学習のおそれがある版（ユーザーの追記「母数 10 件未満などでも構わない」）。件数の下限 3・5・8 では区間の下限は
# ほぼ通らないので、点で判定する。「縮めた点」= (払戻の和 + m × 全体の回収率) / (走数 + m)。縮めても全体との大小は変わらないので、
# どれも全体 ±0.10 と比べる（m = 0 が「点＋0.10」）。頭打ちは「頭打ちでも同じ向き」、元の型は両方、同点の並べ方は両方
SMALL_GRID = {
    "bet": ["複勝", "単勝", "両方"],
    "crit": ["点＋0.10", "縮めた点＋0.10（m=30）", "縮めた点＋0.10（m=5）"],
    "cap": ["頭打ちでも同じ向き"],
    "nmin": [3, 5, 8],
    "layer": ["主＋副", "主だけ"],
    "base": ["採用した型"],
    "tie": ["成績版の点", "馬番"],
}
CRIT_M = {"点＋0.10": 0.0, "縮めた点＋0.10（m=30）": 30.0, "縮めた点＋0.10（m=5）": 5.0}
FAMILY_BIG, FAMILY_SMALL = "件数が多い版", "件数が少なく過学習のおそれがある版"
# 主に見る版（結果を見る前に決めた。複勝は単勝より払戻のばらつきが小さく、区間が意味を持ちやすいため）
PRIMARY = {"bet": "複勝", "crit": "区間の下限", "cap": "頭打ちでも同じ向き", "nmin": 20, "layer": "主＋副", "base": "採用した型", "tie": "成績版の点"}
ALPHA = 0.10                     # 「点＋0.10」: 好は全体 + 0.10 以上、凡は全体 − 0.10 以下
Q = {"disc": 0.05, "conf": 0.10}  # 区間の下限（上限）の片側の確率。発見は片側 95%、確認は片側 90%（成績の z の基準と同じ）
N_BOOT = 2000
LAYERS = {"主＋副": ("主", "副"), "主だけ": ("主",)}
QS = (0.025, 0.05, 0.10, 0.90, 0.95, 0.975)
RET_COLS = ["ret_tan", "ret_fuku", "ret_tan_c", "ret_fuku_c"]


def version_name(v: dict) -> str:
    head = "回収率版（件数少）" if v.get("family") == FAMILY_SMALL else "回収率版"
    return f"{head}[{v['bet']}・{v['crit']}・{v['cap']}・{v['nmin']}走以上・{v['layer']}・{v['base']}・同点は{v['tie']}]"


def all_versions() -> list[dict]:
    out = []
    for fam, grid in ((FAMILY_BIG, GRID), (FAMILY_SMALL, SMALL_GRID)):
        keys = list(grid)
        out += [dict(zip(keys, c)) | {"family": fam} for c in itertools.product(*(grid[k] for k in keys))]
    return out


def instances(S: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """shrink の出力 S（馬 × 型）に払戻を付け、x ≠ 0 で払戻の解けたレースの行だけにする。dir = 好／凡。"""
    s = S[S["x"] != 0][["row", "race_id", "year", "type", "level", "x", "layer", "y3", "field"]]
    s = s.merge(rows[["row"] + RET_COLS + ["pay_ok"]], on="row", how="left")
    s = s[s["pay_ok"].fillna(False).astype(bool)].copy()
    s["dir"] = np.where(s["x"] > 0, "好", "凡")
    s["e3"] = (3.0 / s["field"]).clip(upper=1.0)
    return s


def boot_weights(race_ids, n_boot: int = N_BOOT, seed: int = 0) -> tuple[dict, np.ndarray]:
    """レースごとのポアソン(1) の重み（レース × 回数）。レース単位のブートストラップの近似。"""
    rid = sorted(set(race_ids))
    W = np.random.default_rng(seed).poisson(1.0, size=(len(rid), n_boot)).astype(np.float32)
    return {r: i for i, r in enumerate(rid)}, W


def unit_stats(inst: pd.DataFrame, ridx: dict, W: np.ndarray) -> pd.DataFrame:
    """パターン（型 × 水準 × 向き）ごとの走数・レース数・3 着内・期待・回収率（点と区間の分位点）。"""
    agg = {"n": ("row", "size"), "y3": ("y3", "sum"), "e3": ("e3", "sum"), **{c: (c, "sum") for c in RET_COLS}}
    pr = inst.groupby(["type", "level", "dir", "race_id"], sort=False).agg(**agg).reset_index()
    pr["ri"] = pr["race_id"].map(ridx)
    out = []
    for key, g in pr.groupby(["type", "level", "dir"], sort=False):
        Wi = W[g["ri"].to_numpy()]                       # レース × 回数
        nb = Wi.T @ g["n"].to_numpy(np.float32)
        rec = {"type": key[0], "level": key[1], "dir": key[2], "n": float(g["n"].sum()), "races": int(len(g)),
               "y3": float(g["y3"].sum()), "e3": float(g["e3"].sum())}
        for c in RET_COLS:
            v = g[c].to_numpy(np.float32)
            rec[f"roi_{c}"] = float(v.sum() / g["n"].sum())
            if c in ("ret_tan", "ret_fuku"):
                with np.errstate(divide="ignore", invalid="ignore"):
                    b = (Wi.T @ v) / nb
                b = b[np.isfinite(b)]
                for q in QS:
                    rec[f"q{int(q * 1000):03d}_{c}"] = float(np.quantile(b, q)) if len(b) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def overall_roi(rows: pd.DataFrame) -> dict:
    """同じ期間の重賞の全頭（払戻の解けたレース）に 100 円ずつ買った回収率。"""
    r = rows[rows["pay_ok"].fillna(False).astype(bool)]
    return {c: float(r[c].sum() / len(r)) for c in RET_COLS} | {"runners": int(len(r)), "races": int(r["race_id"].nunique())}


def _bet_ok(u: pd.Series, period: str, bet: str, v: dict, rho: dict) -> bool:
    col, capc = BETS[bet]
    q = Q[period]
    good = u["dir"] == "好"
    if v["crit"] == "区間の下限":
        ok = (u[f"q{int(q * 1000):03d}_{col}"] > rho[col]) if good else (u[f"q{int((1 - q) * 1000):03d}_{col}"] < rho[col])
    else:
        m = CRIT_M[v["crit"]]
        est = (u[f"roi_{col}"] * u["n"] + m * rho[col]) / (u["n"] + m)
        ok = (est >= rho[col] + ALPHA) if good else (est <= rho[col] - ALPHA)
    if v["cap"] == "頭打ちでも同じ向き":
        ok = ok and ((u[f"roi_{capc}"] > rho[capc]) if good else (u[f"roi_{capc}"] < rho[capc]))
    return bool(ok)


def passes(ud: pd.Series, uc: pd.Series, v: dict, rho_d: dict, rho_c: dict) -> bool:
    """1 つのパターンが版 v の基準を、発見・確認の両方で満たすか。"""
    bets = ["単勝", "複勝"] if v["bet"] == "両方" else [v["bet"]]
    for per, u, rho in (("disc", ud, rho_d), ("conf", uc, rho_c)):
        if u is None or u["n"] < v["nmin"]:
            return False
        if not ((u["y3"] > u["e3"]) if u["dir"] == "好" else (u["y3"] < u["e3"])):     # 成績の向きも同じ（期待ちょうどは通さない）
            return False
        if not all(_bet_ok(u, per, b, v, rho) for b in bets):
            return False
    return True


def base_types(pre: dict, base: str) -> list[str]:
    """採用した型（第 1 版で成績の z が発見・確認とも通った前日の型）か、前日の全部の型（採否の対象の型）。"""
    out = []
    for t, info in pre["types"].items():
        if t not in CAT.BY_ID or CAT.BY_ID[t].day_of:     # 今の目録に無い型は使わない
            continue
        if base == "全部の型" or info["status"] == "採用":
            out.append(t)
    return out


def select_versions(S: pd.DataFrame, rows: pd.DataFrame, pre: dict, disc, conf) -> tuple[list[dict], pd.DataFrame, dict]:
    """全版のパターンの採否と、使う型（束ねた代表）を決める。S は 2018〜2024 年の重賞の shrink の出力（第 1 版の m・λ）。"""
    inst_all = instances(S, rows)
    yrs = {"disc": disc, "conf": conf}
    rr = rows[rows["race_id"].isin(set(S["race_id"]))]
    rho = {p: overall_roi(rr[(rr["year"] >= a) & (rr["year"] <= b)]) for p, (a, b) in yrs.items()}
    ridx, W = boot_weights(inst_all["race_id"])
    units = []
    for lname, lset in LAYERS.items():
        for p, (a, b) in yrs.items():
            ii = inst_all[inst_all["layer"].isin(lset) & (inst_all["year"] >= a) & (inst_all["year"] <= b)]
            u = unit_stats(ii, ridx, W)
            u["layer_set"], u["period"] = lname, p
            units.append(u)
            print("units", lname, p, len(u), flush=True)
    U = pd.concat(units, ignore_index=True)
    key = ["type", "level", "dir"]
    zd = {t: (info.get("z_disc") if info.get("z_disc") is not None else -np.inf) for t, info in pre["types"].items()}
    Sd = S[(S["year"] >= disc[0]) & (S["year"] <= disc[1])]
    out = []
    for v in all_versions():
        types = set(base_types(pre, v["base"]))
        Ud = U[(U["layer_set"] == v["layer"]) & (U["period"] == "disc") & U["type"].isin(types)].set_index(key)
        Uc = U[(U["layer_set"] == v["layer"]) & (U["period"] == "conf")].set_index(key)
        pats = []
        for k, ud in Ud.iterrows():
            uc = Uc.loc[k] if k in Uc.index else None
            ud = ud.copy()
            ud["dir"] = k[2]
            if uc is not None:
                uc = uc.copy()
                uc["dir"] = k[2]
            if passes(ud, uc, v, rho["disc"], rho["conf"]):
                pats.append(list(k))
        rec = {"name": version_name(v), "params": v, "family": v["family"], "primary": v == PRIMARY | {"family": FAMILY_BIG}, "patterns": pats,
               "n_good": sum(1 for p in pats if p[2] == "好"), "n_bad": sum(1 for p in pats if p[2] == "凡")}
        ptypes = sorted({p[0] for p in pats}, key=lambda t: -zd.get(t, -np.inf))
        if ptypes:
            f = filtered(Sd[Sd["type"].isin(ptypes)], pats, LAYERS[v["layer"]])
            rep = M.cluster_types(f, ptypes)
            rec["rep"] = rep
            rec["used"] = sorted(set(rep.values()))
        else:
            rec["rep"], rec["used"] = {}, []
        out.append(rec)
    return out, U, rho


def filtered(S: pd.DataFrame, patterns: list, layer_set) -> pd.DataFrame:
    """残ったパターン（型 × 水準 × 向き、その層の組）に当たる行だけ x を残し、ほかの行の x を 0 にする（行は消さない）。"""
    keep = {tuple(p) for p in patterns}
    d = np.where(S["x"] > 0, "好", np.where(S["x"] < 0, "凡", ""))
    k = pd.Series(list(zip(S["type"], S["level"].astype(str), d)), index=S.index)
    m = k.isin(keep) & S["layer"].isin(layer_set)
    out = S.copy()
    out["x"] = np.where(m, out["x"], 0.0)
    return out


def marks_for(S: pd.DataFrame, rec: dict, ref: pd.Series | None = None) -> pd.DataFrame | None:
    """版 rec の印。使う型が無い版と、残ったパターンが S の行に 1 つも当たらない（全馬の点が 0 で、印が馬番の順になる）版は
    None（評価では「該当なし」とする）。ref（行 → 成績版の点）は「同点は成績版の点」の版で、点の次の並べ替えに使う。"""
    if not rec["used"]:
        return None
    s = filtered(S[S["type"].isin(rec["used"])], rec["patterns"], LAYERS[rec["params"]["layer"]])
    if not (s["x"] != 0).any():
        return None
    sc = M.score(s, rec["used"])
    if rec["params"].get("tie") != "成績版の点":
        return M.marks(sc)
    if ref is None:
        raise ValueError("同点を成績版の点で並べる版には ref が要る")
    sc["ref_score"] = sc["row"].map(ref).fillna(-np.inf)
    sc = sc.sort_values(["race_id", "score", "ref_score", "n_good_main", "hn"], ascending=[True, False, False, False, True]).copy()
    sc["rank"] = sc.groupby("race_id").cumcount() + 1
    sc["mark"] = sc["rank"].map(lambda k: M.MARKS[k - 1] if k <= len(M.MARKS) else "")
    return sc


def tie_share(mk: pd.DataFrame) -> dict:
    """◎ と ○ の点が同じで、◎ が同点の並べ方（版により 成績版の点、または 主の好成績パターンの数 → 馬番）で決まったレースの割合と、
    ◎ の点が 0 のレースの割合。"""
    t = mk.sort_values(["race_id", "rank"]).groupby("race_id")["score"].apply(lambda s: (s.iloc[0], s.iloc[1] if len(s) > 1 else np.nan))
    a = np.array([x[0] for x in t])
    b = np.array([x[1] for x in t])
    return {"tie_top": float(np.mean(np.isclose(a, b))), "zero_top": float(np.mean(np.isclose(a, 0.0)))}
