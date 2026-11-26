"""R0-17 決定 3: 複勝の実効オッズの係数 w を、着順を使わずに人気帯ごとに出す（A）。

本番の place_effective_odds は「下限 + 0.3 ×（上限 − 下限）」（src/utils/race_probabilities.py の PLACE_RANGE_WEIGHT）。
ここでは、確定の単勝オッズと確定の複勝の幅（race_odds_flat の place_odds_min・max）だけから、
「その馬が 3 着内に来たときの複勝の払戻の期待値」を作り、w = (期待値 − 下限) ÷ (上限 − 下限) を帯ごとに出す。

作り:
1. 1 円あたりの払戻 = 1 + (R − (s_i + s_j + s_k)) ÷ (K × s_i)（JRA の配分の式）。一緒に来た馬の和 T = s_j + s_k に対して直線なので、
   w = (H_i − E[T]) ÷ (H_i − L_i)（H・L は T の最大・最小）。ほかの馬の割合 s_j は、上限の式から近い値を出す（race_w の説明）。
   R は複勝の払い戻し率 0.8、K は払い戻す頭数（8 頭以上で 3、5〜7 頭で 2）。下限は、ほかの K−1 頭が最も売れた馬のときの値。
   上限の式 s_i = (R − 下位の他の K−1 頭の和) ÷ (K × (上限 − 1) + 1) を、和が 1 になるように繰り返して解く（下限は人気馬の 110 円に張り付くことがあるため）。
2. 一緒に来る馬の組の確率は、確定の単勝の q（1/オッズをレース内で 1 に）から作る。Harville と、Lo・Bacon-Shone の割り引いた型
   （2 着は q^0.81、3 着は q^0.65 で並べ直す）の 2 通り。
3. その馬が 3 着内に来た条件の下での T の期待値を、組の確率で重み付けて出す。\n下限が 110 円に張り付いた馬（表示の下限 ≤ 1.1）は、式の幅と表示の幅が合わないので、w の集計から外して件数だけ出す。\n上限が 1.1 に張り付いた大本命は、割合の近似が頭打ちになるので、その馬のいるレースは「大本命あり」として別に集計する。
着順・払戻は読まない。年は 2016〜24。費用 0 円。土日の 07:30〜17:00 は止める。
使い方: python -m src.research.rules_council.place_w（nice -n 19）。出力 docs/operations/retrain_trials/rules_council_r17_place_w.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.rules_council import engine as E
from src.research.rules_council import rc_frame as F

REPO = F.CR.REPO
TABLES = REPO / "data/page_reference/tables"
OUT = REPO / "docs/operations/retrain_trials/rules_council_r17_place_w.json"
BANDS = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1e9]
BAND_NAMES = ["1〜3 倍", "3〜5 倍", "5〜10 倍", "10〜20 倍", "20〜50 倍", "50 倍以上"]
LBS = (1.0, 0.81, 0.65)


def ordered_probs(q: np.ndarray, K: int, lam: tuple[float, ...]) -> np.ndarray:
    """順序つきの上位 K 頭の確率（K=2 は n×n、K=3 は n×n×n）。各着は q^lam で並べ直す（lam=(1,1,1) で Harville）。"""
    n = len(q)
    w1, w2, w3 = q ** lam[0], q ** lam[1], q ** lam[2]
    p1 = w1 / w1.sum()
    W2 = w2.sum()
    p2 = w2[None, :] / (W2 - w2[:, None])
    np.fill_diagonal(p2, 0.0)
    P2 = p1[:, None] * p2
    if K == 2:
        return P2
    W3 = w3.sum()
    den = W3 - w3[:, None, None] - w3[None, :, None]
    P3 = P2[:, :, None] * (w3[None, None, :] / den)
    idx = np.arange(n)
    P3[idx, :, idx] = 0.0
    P3[:, idx, idx] = 0.0
    P3[idx, idx, :] = 0.0
    return P3


def race_w(win_odds: np.ndarray, pmin: np.ndarray, pmax: np.ndarray, K: int) -> list[dict]:
    """払戻は、一緒に来た K−1 頭の票の割合の和 T に対して直線（1 + (R − s_i − T) ÷ (K s_i)）。
    下限は T が最大の H_i、上限は T が最小の L_i のときなので、w = (H_i − E[T | i が 3 着内]) ÷ (H_i − L_i) となり、R と s_i の大きさに依らない。
    ほかの馬の票の割合は、上限の式から s_j ∝ 1 ÷ (K × (上限 − 1) + 1) で近い値を出す（下位の 2 頭の和は小さいので無視できる）。"""
    q = (1.0 / win_odds) / np.sum(1.0 / win_odds)
    sh = 1.0 / (K * (pmax - 1.0) + 1.0)                                  # 上限が 1.1 に張り付いた馬は頭打ち（大本命ありの印で分ける）
    sh = sh / sh.sum()
    n = len(q)
    H = np.empty(n)
    L = np.empty(n)
    for i in range(n):
        ot = np.sort(np.delete(sh, i))
        L[i] = ot[: K - 1].sum()
        H[i] = ot[::-1][: K - 1].sum()
    res = []
    for name, lam in (("harville", (1.0, 1.0, 1.0)), ("lbs", LBS)):
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.nan_to_num(ordered_probs(q, K, lam))
        num = np.zeros(n)
        den = np.zeros(n)
        if K == 3:
            T2 = sh[:, None] + sh[None, :]                              # ほかの 2 頭の和（位置 1・2）
            for ax in range(3):
                Pa = np.moveaxis(P, ax, 0)                               # 0 軸がその馬。残りの 2 軸がほかの 2 頭
                den += Pa.sum(axis=(1, 2))
                num += (Pa * T2[None, :, :]).sum(axis=(1, 2))
        else:
            for ax in range(2):
                Pa = np.moveaxis(P, ax, 0)
                den += Pa.sum(axis=1)
                num += (Pa * sh[None, :]).sum(axis=1)
        ET = num / np.maximum(den, 1e-12)
        w = np.where(H > L, (H - ET) / np.where(H > L, H - L, 1), np.nan)
        ev = pmin + w * (pmax - pmin)
        res.append({"model": name, "w": w, "pay_given_place": ev, "p_place": den})
    return res


def main() -> None:
    E.check_stop()
    rows = []
    for y in range(2016, F.YEAR_END + 1):
        d = pd.read_parquet(TABLES / str(y) / "race_odds_flat.parquet", columns=["race_id", "surface", "field_size", "horse_number", "win_odds", "place_odds_min", "place_odds_max"])
        d = d[d["surface"].astype(str).str[0].isin(["芝", "ダ"])]
        for c in ("win_odds", "place_odds_min", "place_odds_max"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        for rid, g in d.groupby("race_id"):
            g = g[(g["win_odds"] > 0) & (g["place_odds_min"] >= 1.0) & (g["place_odds_max"] >= g["place_odds_min"])]
            n = len(g)
            fsv = pd.to_numeric(g["field_size"], errors="coerce").max()
            if pd.isna(fsv):
                continue
            fs = int(fsv)
            if n < 5 or n != fs:                                          # 出走頭数とオッズのある頭数が合わないレースは外す
                continue
            K = 3 if fs >= 8 else 2
            fav = bool((g["place_odds_max"] <= 1.1).any())
            wo, pmn, pmx = g["win_odds"].to_numpy(float), g["place_odds_min"].to_numpy(float), g["place_odds_max"].to_numpy(float)
            for r in race_w(wo, pmn, pmx, K):
                for i in range(n):
                    rows.append((y, str(rid), r["model"], wo[i], pmn[i], pmx[i], r["w"][i], r["pay_given_place"][i], r["p_place"][i], fav))
        print(y, len(rows), flush=True)
        E.check_stop()
    x = pd.DataFrame(rows, columns=["year", "race_id", "model", "win_odds", "pmin", "pmax", "w", "pay", "p_place", "fav_race"])
    x["stuck"] = x["pmin"] <= 1.1 + 1e-9
    x["band"] = pd.cut(x["win_odds"], BANDS, labels=BAND_NAMES, right=False)
    out = {"note": "着順なし。確定の単勝と複勝の幅から、3 着内に来たときの払戻の期待値（式の値）と w を出した。本番の w は 0.3",
           "years": "2016〜2024", "rows_per_model": int((x["model"] == "harville").sum()), "bands": {}}
    for (m, fv), g in x.groupby(["model", "fav_race"]):
        b = {}
        for band, h in g.groupby("band", observed=True):
            ok = ~h["stuck"]
            ww = h.loc[ok, "w"].dropna()
            if not len(ww):
                b[str(band)] = {"n": int(len(h)), "stuck_min_1_1": int((~ok).sum())}
                continue
            b[str(band)] = {"n": int(len(h)), "stuck_min_1_1": int((~ok).sum()), "w_mean": float(ww.mean()), "w_median": float(ww.median()),
                            "w_q25": float(ww.quantile(0.25)), "w_q75": float(ww.quantile(0.75)),
                            "pay_over_min_median": float((h.loc[ok, "pay"] / h.loc[ok, "pmin"]).median())}
        out["bands"][f"{m}・{'大本命あり' if fv else '大本命なし'}"] = b
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["bands"], ensure_ascii=False, indent=1))


def cmd_check() -> dict:
    """B（1 号の決定: A の commit の後の照らし合わせ）。探す側 70%（2016〜24）で、3 着内に来た馬の確定の複勝の払戻が、
    確定の幅のどこに落ちたか（w = (払戻 − 下限) ÷ (上限 − 下限)）の中央値と分位を、人気帯 × 大本命あり／なしで出す。
    出すのは中央値・分位・件数だけ。的中の率・回収率は出さない。確かめの側 30%・2025・26 は読まない。"""
    E.check_stop()
    pay = pd.read_parquet(REPO / "data/local/modeling/betting/payoff.parquet", columns=["race_id", "bet_type", "combo", "payout"])
    pay["race_id"] = pay["race_id"].astype(str)
    pay = pay[(pay["bet_type"] == "fukusho") & (pay["race_id"].str[:4].astype(int) <= F.YEAR_END)].copy()
    pay = pay[E.search70(pay["race_id"].str[:8].to_numpy())]
    pay["horse_number"] = pd.to_numeric(pay["combo"].astype(str), errors="coerce")
    pay = pay.groupby(["race_id", "horse_number"], as_index=False)["payout"].sum()        # 同着は足す
    rows = []
    for y in range(2016, F.YEAR_END + 1):
        d = pd.read_parquet(TABLES / str(y) / "race_odds_flat.parquet", columns=["race_id", "surface", "field_size", "horse_number", "win_odds", "place_odds_min", "place_odds_max"])
        d = d[d["surface"].astype(str).str[0].isin(["芝", "ダ"])].copy()
        d["race_id"] = d["race_id"].astype(str)
        d = d[E.search70(d["race_id"].str[:8].to_numpy())]
        for c in ("win_odds", "place_odds_min", "place_odds_max", "horse_number"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["fav_race"] = d.groupby("race_id")["place_odds_max"].transform(lambda s_: bool((s_ <= 1.1).any()))
        rows.append(d)
    d = pd.concat(rows, ignore_index=True).merge(pay, on=["race_id", "horse_number"], how="inner")
    d = d[(d["place_odds_min"] > 1.1 + 1e-9) & (d["place_odds_max"] > d["place_odds_min"])]
    d["w_real"] = (d["payout"] / 100.0 - d["place_odds_min"]) / (d["place_odds_max"] - d["place_odds_min"])
    d["band"] = pd.cut(d["win_odds"], BANDS, labels=BAND_NAMES, right=False)
    out = {"note": "探す側 70%（払戻の表がある 2020〜24）の 3 着内の馬だけ。下限が 1.1 に張り付いた馬は外した。出すのは中央値・分位・件数だけ（的中の率・回収率は出さない）",
           "bands": {}}
    for fv, g in d.groupby("fav_race"):
        b = {}
        for band, h in g.groupby("band", observed=True):
            ww = h["w_real"]
            b[str(band)] = {"n": int(len(h)), "w_mean": float(ww.mean()), "w_median": float(ww.median()), "w_q25": float(ww.quantile(0.25)), "w_q75": float(ww.quantile(0.75))}
        out["bands"]["大本命あり" if fv else "大本命なし"] = b
    p_ = REPO / "docs/operations/retrain_trials/rules_council_r17_place_w_check.json"
    p_.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["bands"], ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    import sys
    cmd_check() if sys.argv[1:] == ["check"] else main()
