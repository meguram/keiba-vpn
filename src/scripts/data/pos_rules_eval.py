"""過去位置取りの段 2: 山が 2 つある馬の「どちらの山か」(A) と、走数が少ない馬の補い (B) を、発走前に分かる事柄の規則で探す。

python -m src.scripts.data.pos_rules_eval            # → /tmp/pos_rules/report.json

着順・オッズ・払戻は使わない。的は次走の初角の相対位置（通過順位÷頭数）だけ。段 1（pos_late_start_eval）と同じ読み込み・分け方。
比べる相手は段 1 で採った M0（past_position.horse_position の加重平均）。平地だけ（障害は除く）。2016〜2024 年。

事前に固定した決まり（数を見る前。2026-11-27）:
(A) 山が 2 つの馬（履歴 3 走以上で position_estimators.kde_peaks が 2 つ返す走）。前の山 pf・後ろの山 pb。
  規則（+1 = 前の山・-1 = 後ろの山・0 = 何も言わない。向きは先に決める）:
    A1 枠: 馬番 ≤ 頭数/3 → +1、馬番 > 頭数×2/3 → -1
    A2 距離: 前走より 200m 以上短い → -1（追走が速くなる）、200m 以上長い → +1
    A3 先行馬の数: 同じレースで M0 ≤ 0.25 の他の馬が 4 頭以上 → -1、1 頭以下 → +1
    A4 騎手の先行の率: 今回の騎手の過去の走のうち初角 0.25 以内の割合（as-of・100 走以上）が 0.35 以上 → +1、0.20 以下 → -1
    A5 前走の位置: 前走の位置が pf と pb の中点より前 → +1、後ろ → -1
    A6 休み明け: 前走から 90 日以上 → -1
    A7 頭数: 10 頭以下 → +1、16 頭以上 → -1
  使わない事柄（理由）: クラスの上下（軽い表に段の対応が無い）・初角までの距離（コースの形の表が軽い表に無い）・騎手の替わりだけ（向きが決まらない）。
  予想の形: 規則が言うとき H = (M0 + 選んだ山)/2、言わないとき M0。参考に P = 選んだ山（言わないとき M0）も出す。
  探す側で、山が 2 つの全行の H の平均絶対誤差が M0 より小さく、開催単位のブートストラップ（B=2000・seed 7332）の p を Holm（7 本）で直して 5% 未満の規則を通す。
  通った規則の符号の和の向きで山を選ぶ合成 A* を作り、確かめる側を 1 回だけ開ける（各規則と A*）。
(B) 走数が 0〜3 走の馬（初出走を含む）。as-of の集団の平均位置（その走の日より前の走だけ・30 走以上）:
    B1 父の産駒・B2 母父の産駒・B3 今回の騎手・B4 今回の調教師。B5 = B1〜B4 のある物の平均（無ければ全体の as-of 平均）。
  B6 = (n·M0 + k·B5)/(n + k)（n は走数。n=0 なら B5）。k は探す側で {0.5, 1, 2, 4, 8} から平均絶対誤差の最小で選ぶ。
  比べる相手 B0 = M0（n=0 は全体の as-of 平均）。確かめる側で B6 と B0 の差をブートストラップ（seed 7333）で出す。
  母の産駒は、軽い表に母の ID が無いので使わない。
(C) 確かめる側でも M0 より良かった規則だけを、画面の説明の 1 文の型にする。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.race_analysis import position_estimators as PE
from src.research.race_analysis.past_position import WINDOW, horse_position
from src.scripts.data import pos_late_start_eval as S1

ROOT = S1.ROOT
SIRE = Path(os.environ.get("POS_EVAL_SIRE") or ROOT / "data/page_reference/pedigree_race_index/horse_sire_map.parquet")
OUT = Path(os.environ.get("POS_RULES_OUT") or "/tmp/pos_rules")
RULES = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
KS = [0.5, 1, 2, 4, 8]
MIN_GROUP = 30
MIN_JOCKEY = 100


def load() -> pd.DataFrame:
    import glob
    cols = ["race_id", "date", "surface", "distance", "horse_id", "horse_number", "jockey_id", "trainer_id",
            "passing_order", "field_size", "finish_position"]
    parts = [pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(S1.TABLES / "*/race_result_flat.parquet"))) if int(Path(f).parent.name) <= 2024]
    df = pd.concat(parts, ignore_index=True)
    for c in ("race_id", "horse_id", "jockey_id", "trainer_id"):
        df[c] = df[c].astype(str)
    df["date"] = df["date"].astype(str).str[:10]
    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    df = df[fp.notna() & (fp > 0) & ~df["surface"].astype(str).str.startswith("障")].drop(columns=["finish_position"])
    fs = pd.to_numeric(df["field_size"], errors="coerce")
    first = pd.to_numeric(df["passing_order"].astype("string").str.extract(r"^(\d+)")[0], errors="coerce")
    df = df.assign(fs=fs, r=first / fs, dist=pd.to_numeric(df["distance"], errors="coerce"),
                   num=pd.to_numeric(df["horse_number"], errors="coerce"))[first.between(1, fs)].copy()
    sm = pd.read_parquet(SIRE, columns=["horse_id", "sire_id", "dam_sire_id"]).astype(str).drop_duplicates("horse_id")
    df = df.merge(sm, on="horse_id", how="left")
    return df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)


def asof_mean(df: pd.DataFrame, key: str, val: str, min_n: int) -> pd.Series:
    """その走の日より前の、同じ key の走の val の平均（min_n 走未満は NaN）。"""
    g = df.groupby([key, "date"], sort=True)[val].agg(["sum", "count"]).reset_index()
    g[["cs", "cn"]] = g.groupby(key)[["sum", "count"]].cumsum().to_numpy() - g[["sum", "count"]].to_numpy()
    m = df[[key, "date"]].merge(g[[key, "date", "cs", "cn"]], on=[key, "date"], how="left")
    cs, cn = m["cs"].astype(float), m["cn"].astype(float)
    out = (cs / cn.where(cn > 0)).where(cn >= min_n)
    return pd.Series(out.to_numpy(), index=df.index)


def overall_asof(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("date")["r"].agg(["sum", "count"]).sort_index()
    cs, cn = g["sum"].cumsum() - g["sum"], g["count"].cumsum() - g["count"]
    v = cs.astype(float) / cn.astype(float).where(cn > 0)
    return df["date"].map(v).astype(float)


def rows(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("horse_id", sort=False):
        r, rid, dt, dist = g["r"].tolist(), g["race_id"].tolist(), g["date"].tolist(), g["dist"].tolist()
        for t in range(len(g)):
            first = r[max(0, t - WINDOW):t][::-1]
            rec = {"idx": g.index[t], "race_id": rid[t], "y": r[t], "n": len(first),
                   "M0": horse_position(first).value if first else np.nan, "pf": np.nan, "pb": np.nan, "pf_main": False,
                   "prev_r": first[0] if first else np.nan,
                   "prev_dist": dist[t - 1] if t else np.nan,
                   "gap_days": (pd.Timestamp(dt[t]) - pd.Timestamp(dt[t - 1])).days if t else np.nan}
            if len(first) >= PE.MIN_RUNS:
                pk = PE.kde_peaks(first)
                if len(pk) > 1:
                    rec["pf"], rec["pb"] = sorted(p for p, _ in pk)
                    rec["pf_main"] = pk[0][0] < pk[1][0]          # 密度の高い主の山が前の山か（比べ相手「いつも主の山」）
            out.append(rec)
    R = pd.DataFrame(out).set_index("idx")
    return df.join(R.drop(columns=["race_id"]))


def signals(D: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=D.index)
    s["A1"] = np.where(D["num"] <= D["fs"] / 3, 1, np.where(D["num"] > D["fs"] * 2 / 3, -1, 0))
    dd = D["dist"] - D["prev_dist"]
    s["A2"] = np.where(dd <= -200, -1, np.where(dd >= 200, 1, 0))
    front = (D["M0"] <= 0.25).astype(float)
    others = front.groupby(D["race_id"]).transform("sum") - front
    s["A3"] = np.where(others >= 4, -1, np.where(others <= 1, 1, 0))
    jr = D["jockey_front"]
    s["A4"] = np.where(jr >= 0.35, 1, np.where(jr <= 0.20, -1, 0))
    mid = (D["pf"] + D["pb"]) / 2
    s["A5"] = np.where(D["prev_r"] < mid, 1, np.where(D["prev_r"] > mid, -1, 0))
    s["A6"] = np.where(D["gap_days"] >= 90, -1, 0)
    s["A7"] = np.where(D["fs"] <= 10, 1, np.where(D["fs"] >= 16, -1, 0))
    return s


def pred(D: pd.DataFrame, sig: pd.Series, form: str) -> pd.Series:
    peak = np.where(sig > 0, D["pf"], D["pb"])
    p = (D["M0"] + peak) / 2 if form == "H" else pd.Series(peak, index=D.index)
    return pd.Series(np.where(sig != 0, p, D["M0"]), index=D.index)


def boot_p(y, a, b, unit, seed: int) -> float:
    diff = pd.Series(np.abs(a - y) - np.abs(b - y))
    g = diff.groupby(np.asarray(unit)).agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(S1.BOOT_B, len(s)))
    m = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))


def holm(ps: dict[str, float]) -> dict[str, float]:
    order, run, out = sorted(ps, key=ps.get), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, ps[k] * (len(order) - i)))
        out[k] = round(run, 4)
    return out


def rule_table(D: pd.DataFrame, sig: pd.DataFrame, names: list[str], seed: int) -> dict:
    tab, ps = {}, {}
    y, unit = D["y"].to_numpy(), D["race_id"].str[:8].to_numpy()
    for k in names:
        s = sig[k]
        on = s != 0
        closer_front = (D["y"] - D["pf"]).abs() < (D["y"] - D["pb"]).abs()
        hit = np.where(s[on] > 0, closer_front[on], ~closer_front[on])
        main_front = D["pf_main"][on]
        base_hit = np.where(main_front, closer_front[on], ~closer_front[on])
        h = pred(D, s, "H").to_numpy()
        tab[k] = {"fires": int(on.sum()), "hit": round(float(hit.mean()), 4) if on.any() else None,
                  "hit_main_peak": round(float(base_hit.mean()), 4) if on.any() else None,
                  "mae_H": round(float(np.abs(h - y).mean()), 5), "mae_P": round(float(np.abs(pred(D, s, "P").to_numpy() - y).mean()), 5),
                  "mae_M0": round(float(np.abs(D["M0"].to_numpy() - y).mean()), 5)}
        ps[k] = boot_p(y, h, D["M0"].to_numpy(), unit, seed)
        tab[k]["p"] = round(ps[k], 4)
    return {"rules": tab, "holm": holm(ps)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    df["front"] = (df["r"] <= 0.25).astype(float)
    df["jockey_front"] = asof_mean(df, "jockey_id", "front", MIN_JOCKEY)
    for key, col in (("sire_id", "B1"), ("dam_sire_id", "B2"), ("jockey_id", "B3"), ("trainer_id", "B4")):
        df[col] = asof_mean(df, key, "r", MIN_GROUP)
    df["glob"] = overall_asof(df)
    D = rows(df)
    D["side"] = D["race_id"].map(S1.side_of)
    rep: dict = {"rows": int(len(D))}

    # (A)
    A = D[D["pf"].notna()].copy()
    sig = signals(A)
    a_s, a_v = A["side"] == 0, A["side"] == 1
    rep["A_rows"] = {"search": int(a_s.sum()), "verify": int(a_v.sum())}
    srch = rule_table(A[a_s], sig[a_s], RULES, 7332)
    rep["A_search"] = srch
    passed = [k for k in RULES if srch["holm"][k] < 0.05 and srch["rules"][k]["mae_H"] < srch["rules"][k]["mae_M0"]]
    rep["A_passed"] = passed
    if passed:
        sig["Astar"] = np.sign(sig[passed].sum(axis=1))
        rep["A_verify"] = rule_table(A[a_v], sig[a_v], passed + ["Astar"], 7334)

    # (B)
    B = D[D["n"] <= 3].copy()
    B["B5"] = B[["B1", "B2", "B3", "B4"]].mean(axis=1).fillna(B["glob"])
    B["B0"] = B["M0"].where(B["n"] > 0, B["glob"])
    B = B[B["B0"].notna() & B["B5"].notna()]
    bs, bv = B[B["side"] == 0], B[B["side"] == 1]

    def b6(X: pd.DataFrame, k: float) -> pd.Series:
        return ((X["n"] * X["M0"].fillna(0) + k * X["B5"]) / (X["n"] + k)).where(X["n"] > 0, X["B5"])

    mae = lambda X, c: round(float((X[c] - X["y"]).abs().mean()), 5)  # noqa: E731
    k_mae = {k: round(float((b6(bs, k) - bs["y"]).abs().mean()), 5) for k in KS}
    kbest = min(KS, key=lambda k: k_mae[k])
    rep["B_search_k"] = {"mae_by_k": k_mae, "k": kbest}
    bv = bv.assign(B6=b6(bv, kbest))
    bs = bs.assign(B6=b6(bs, kbest))
    for name, X in (("B_search", bs), ("B_verify", bv)):
        sec = {}
        for n in (0, 1, 2, 3):
            Xn = X[X["n"] == n]
            sec[f"n{n}"] = {c: mae(Xn, c) for c in ("B0", "B1", "B2", "B3", "B4", "B5", "B6") if Xn[c].notna().any()} | {"rows": int(len(Xn))}
        sec["all"] = {c: mae(X, c) for c in ("B0", "B5", "B6")} | {"rows": int(len(X))}
        rep[name] = sec
    rep["B_verify_p_B6_vs_B0"] = round(boot_p(bv["y"].to_numpy(), bv["B6"].to_numpy(), bv["B0"].to_numpy(), bv["race_id"].str[:8].to_numpy(), 7333), 4)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
