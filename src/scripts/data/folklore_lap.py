"""ラップ・地形の型（参考: ユリス【UMASTカルテ】の note『ラップで驚くほど競馬が見える【基礎編】』『競馬場の「地形」で見えなかったものが見える』の無料部分）。

記事の無料部分で読める考え方だけを使い、閾値は自分で決めた（記事の段階分けの閾値は有料部分で読めない）:
- レースのラップはレースラップ（先頭の通過タイム）。上り 3F を『後半』、それ以外を『前半』とし、前半を 3F に換算して
  前後半差 = 前半（3F 換算）− 上り 3F（＋なら後半が速い）。
- ラップの型（記事の 3 つ + 1）:
    瞬発 = スローからの瞬発力勝負（前後半差が上位 1/3 で、上り 3F の前の 3F より上り 3F が速い = 終盤に加速）
    消耗 = ハイペースの勝負根性比べ（前後半差が下位 1/3 = 前半が速く、減速していく）
    持続 = 緩みのない持続ラップ（上の 2 つ以外で、2F 目以降のラップのばらつきが下位 1/3）
    平均 = それ以外
  1/3 の境目は、面 × 距離（コース）ごとに 2016〜2022 年のレースで決める（距離・コースの違いを除く）。
- 馬の得意な型: その馬の過去の走（その日より前だけ）の、型ごとの勝ち馬とのタイム差の平均。
- 今回の想定の型: そのコース（競馬場・面・距離）で、その日より前に最も多かった型。
- 地形: 最後の直線の長さ（400m 以上を長い）・ゴール前の坂の有無・回り（右/左）・コーナー（小回りかどうか）。
  馬の過去の走を、今回と同じ側 / 違う側に分けたタイム差の平均の差で、得意な側を決める。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TYPES = ("瞬発", "消耗", "持続", "平均")


def race_laps(years) -> pd.DataFrame:
    rows = []
    for y in years:
        r = pd.read_parquet(ROOT / f"data/page_reference/tables/{y}/race_result_race.parquet", columns=["race_id", "distance", "lap_times"])
        for rid, dist, lt in zip(r["race_id"], r["distance"], r["lap_times"]):
            try:
                laps = json.loads(lt) if isinstance(lt, str) else list(lt)
                laps = [float(x) for x in laps]
                dist = int(dist)
            except Exception:  # noqa: BLE001
                continue
            if dist % 200 != 0 and len(laps) > 1:
                laps = laps[1:]  # 200m に満たない最初の区間は外す
            if len(laps) < 6:
                continue
            last3 = sum(laps[-3:])
            front = laps[:-3]
            front3 = float(np.mean(front)) * 3
            acc = float(np.mean(laps[-6:-3]) - np.mean(laps[-3:]))  # ＋なら上り 3F で加速
            cv = float(np.std(laps[1:]))
            rows.append((str(rid), front3 - last3, acc, cv))
    return pd.DataFrame(rows, columns=["race_id", "lp_diff", "lp_acc", "lp_cv"])


def _asof_cum(d: pd.DataFrame, keys: list[str], val: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """keys ごとに、その日より前の mask の行の件数と val の合計（同じ日は含めない）。d は horse・dt の順に並んでいる前提はしない。"""
    g = d[keys + ["dt"]].copy()
    g["_n"] = mask.astype(float)
    g["_s"] = np.where(mask, val, 0.0)
    day = g.groupby(keys + ["dt"], sort=True)[["_n", "_s"]].sum()
    cum = day.groupby(level=list(range(len(keys)))).cumsum() - day
    idx = pd.MultiIndex.from_frame(g[keys + ["dt"]])
    c = cum.reindex(idx)
    return c["_n"].to_numpy(), c["_s"].to_numpy()


def add_laps(d: pd.DataFrame) -> pd.DataFrame:
    rl = race_laps(sorted(d["year"].unique().tolist()))
    d = d.drop(columns=[c for c in d.columns if c.startswith(("lp_", "lt_", "tr_")) and c not in ("tr_pop_rank",)], errors="ignore")
    d["race_id"] = d["race_id"].astype(str)
    d = d.merge(rl, on="race_id", how="left")
    ck = d["surf"].astype(str) + d["dist"].astype(str)
    race = d.drop_duplicates("race_id")[["race_id", "year", "lp_diff", "lp_acc", "lp_cv"]].copy()
    race["ck"] = ck[d.drop_duplicates("race_id").index].to_numpy()
    fit = race[race["year"] <= 2022]
    q = fit.groupby("ck").agg(d1=("lp_diff", lambda s: s.quantile(1 / 3)), d2=("lp_diff", lambda s: s.quantile(2 / 3)),
                              c1=("lp_cv", lambda s: s.quantile(1 / 3)), n=("lp_diff", "size"))
    q = q[q["n"] >= 30]
    race = race.join(q, on="ck")
    t = np.where(race["lp_diff"] >= race["d2"], np.where(race["lp_acc"] > 0, "瞬発", "平均"),
                 np.where(race["lp_diff"] <= race["d1"], "消耗", np.where(race["lp_cv"] <= race["c1"], "持続", "平均")))
    t = np.where(race["d1"].isna() | race["lp_diff"].isna(), None, t)
    race["lp_type"] = t
    d = d.merge(race[["race_id", "lp_type"]], on="race_id", how="left")
    td = d["tdiff"].clip(upper=5.0).to_numpy(float)
    ok_td = ~np.isnan(td)
    good = ((d["fin"] <= 3) | (d["tdiff"] <= 0.5)).to_numpy()
    # 馬の型ごとの過去（その日より前）
    for T in TYPES:
        m = (d["lp_type"] == T).to_numpy() & ok_td
        n, s = _asof_cum(d, ["horse_id"], td, m)
        ng, _ = _asof_cum(d, ["horse_id"], td, m & good)
        d[f"lt_n_{T}"], d[f"lt_td_{T}"], d[f"lt_good_{T}"] = n, np.where(n > 0, s / np.maximum(n, 1), np.nan), ng
    # 得意な型 = 2 走以上ある型のうち平均タイム差が最小（2 つ以上の型に 2 走以上あるときだけ）
    tdm = np.vstack([np.where(d[f"lt_n_{T}"] >= 2, d[f"lt_td_{T}"], np.nan) for T in TYPES]).T
    nvalid = (~np.isnan(tdm)).sum(1)
    best = np.where(nvalid >= 2, np.nanargmin(np.where(np.isnan(tdm), 99, tdm), axis=1), -1)
    d["lt_best"] = np.where(best >= 0, np.array(TYPES, dtype=object)[np.clip(best, 0, 3)], None)
    # 今回の想定の型 = そのコースでその日より前に最も多かった型
    cs = d.drop_duplicates("race_id")[["race_id", "course", "dt", "lp_type"]].copy()
    for T in TYPES:
        n, _ = _asof_cum(cs, ["course"], np.zeros(len(cs)), (cs["lp_type"] == T).to_numpy())
        cs[f"c_{T}"] = n
    cc = cs[[f"c_{T}" for T in TYPES]].to_numpy()
    tot = cc.sum(1)
    cs["lt_exp"] = np.where(tot >= 10, np.array(TYPES, dtype=object)[cc.argmax(1)], None)
    cs["lt_exp_share"] = np.where(tot > 0, cc.max(1) / np.maximum(tot, 1), np.nan)
    d = d.merge(cs[["race_id", "lt_exp", "lt_exp_share"]], on="race_id", how="left")
    # 前走の型
    d = d.sort_values(["horse_id", "dt", "race_id"], kind="stable").reset_index(drop=True)
    d["lt_prev"] = d.groupby("horse_id", sort=False)["lp_type"].shift(1)
    # 地形: 今回と同じ側 / 違う側のタイム差の平均
    side = {"tr_str": d["straight"] >= 400, "tr_slope": d["slope"].astype(bool), "tr_dir": d["direction"] == "右", "tr_small": d["shape"] == "小回り"}
    valid_dir = d["direction"].isin(["右", "左"]).to_numpy()
    for k, sd in side.items():
        sd = sd.to_numpy(bool)
        v = ok_td & (valid_dir if k == "tr_dir" else True)
        n1, s1 = _asof_cum(d, ["horse_id"], td, v & sd)
        n0, s0 = _asof_cum(d, ["horse_id"], td, v & ~sd)
        same_n = np.where(sd, n1, n0)
        same_s = np.where(sd, s1, s0)
        oth_n = np.where(sd, n0, n1)
        oth_s = np.where(sd, s0, s1)
        okk = (same_n >= 2) & (oth_n >= 2) & (valid_dir if k == "tr_dir" else True)
        d[k + "_ok"] = okk
        d[k + "_gain"] = np.where(okk, oth_s / np.maximum(oth_n, 1) - same_s / np.maximum(same_n, 1), np.nan)  # ＋なら今回の側が得意
    gains = np.vstack([d[k + "_gain"].to_numpy() for k in side]).T
    d["tr_n_ok"] = (~np.isnan(gains)).sum(1)
    d["tr_n_good"] = (gains >= 0.2).sum(1)
    d["tr_n_bad"] = (gains <= -0.2).sum(1)
    return d


def lap_hyps(H) -> list:
    """ラップ・地形の型（R・T 分類）。H は folklore_verify.H。"""
    T = TYPES
    L = []
    a = L.append
    a(H("R01", 1, lambda d: d["lt_exp"].notna() & (sum(d[f"lt_n_{t}"].fillna(0) for t in T) >= 3),
        lambda d: np.select([d["lt_exp"] == t for t in T], [d[f"lt_good_{t}"] for t in T], 0) >= 1,
        "今回の想定の型（そのコースで最も多い型）で 3 着内か 0.5 秒差以内の経験がある（過去 3 走以上の馬）"))
    a(H("R02", 1, lambda d: d["lt_exp"].notna() & d["lt_best"].notna(), lambda d: d["lt_best"] == d["lt_exp"],
        "得意な型（型ごとのタイム差の平均が最小）が今回の想定の型と一致"))
    a(H("R03", 1, lambda d: d["lt_prev"].notna() & (d["p_fin"] >= 4) & (d["p_tdiff"] <= 1.0),
        lambda d: (d["lt_prev"] == "消耗") & (d["p_pos4r"] <= 0.3),
        "前走 4 着以下・1.0 秒差以内の馬で、前走が消耗戦（前半が速い）なのに 4 角で前 3 割（展開の不利）"))
    a(H("R04", 1, lambda d: d["lt_prev"].notna() & (d["p_fin"] >= 4) & (d["p_tdiff"] <= 1.0),
        lambda d: (d["lt_prev"] == "瞬発") & (d["p_pos4r"] >= 0.6),
        "前走 4 着以下・1.0 秒差以内の馬で、前走が瞬発戦（スローからの加速）なのに 4 角で後ろ 4 割（展開の不利）"))
    a(H("R05", -1, lambda d: d["lt_prev"].notna() & (d["p_fin"] <= 3), lambda d: (d["lt_prev"] == "瞬発") & (d["p_pos4r"] <= 0.3),
        "前走 3 着内の馬で、前走が瞬発戦で 4 角で前 3 割（展開の恩恵）→ 過剰人気"))
    a(H("R06", -1, lambda d: d["lt_prev"].notna() & (d["p_fin"] <= 3), lambda d: (d["lt_prev"] == "消耗") & (d["p_pos4r"] >= 0.6),
        "前走 3 着内の馬で、前走が消耗戦で 4 角で後ろ 4 割（展開の恩恵）→ 過剰人気"))
    a(H("R07", 1, lambda d: d["lt_prev"].notna() & d["lt_exp"].notna() & (d["p_fin"] <= 3), lambda d: d["lt_prev"] == d["lt_exp"],
        "前走 3 着内の馬で、前走の型が今回の想定の型と同じ（得意な型を再現できるレース）"))
    for k, nm in (("tr_str", "最後の直線の長さ（400m 以上か）"), ("tr_slope", "ゴール前の坂の有無"), ("tr_dir", "回り（右/左）"), ("tr_small", "小回りかどうか")):
        a(H({"tr_str": "T01", "tr_slope": "T02", "tr_dir": "T03", "tr_small": "T04"}[k], 1, lambda d, k=k: d[k + "_ok"].astype(bool),
            lambda d, k=k: d[k + "_gain"] >= 0.2, f"地形: {nm}が今回と同じ側の過去の走のタイム差が、違う側より 0.2 秒以上小さい（各 2 走以上）"))
    a(H("T05", 1, lambda d: d["tr_n_ok"] >= 2, lambda d: (d["tr_n_good"] >= 2) & (d["tr_n_bad"] == 0), "地形: 判定できる地形のうち 2 つ以上が得意で、苦手が無い"))
    a(H("T06", -1, lambda d: d["tr_n_ok"] >= 2, lambda d: (d["tr_n_bad"] >= 2) & (d["tr_n_good"] == 0), "地形: 判定できる地形のうち 2 つ以上が苦手で、得意が無い"))
    return L
