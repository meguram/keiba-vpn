"""議題 RC-1（本物の T-60 で再現の帯を確かめる）。着順・払戻は読まない（2026 年の確定オッズ・確定の人気・本物の T-60 のオッズだけ）。

- 本物の T-60 は data/local/research/persona1/pre_odds.parquet（phase t60）。2026-10-18〜。
- 再現は megu_cycles の simulate_pre と同じ作り（レースの全頭で、2026 のレースのずれを確定の人気の順位ごとに丸ごと借り、
  1/オッズの和を確定に合わせる）。比べるのは、本物の T-60 の記録がある馬だけ（q は全頭の確定オッズで作る）。
- 自分と比べないように、開催日を日付の順に前半と後半に分ける。ずれは前半だけから作り、後半で比べる（主）。
  開催日を交互に分けた版（副）・全部から作ったずれで全部を見る版（参考。今の道具の作り）・11/21 以降（参考）も並べる。
- 見るもの（上端 x の『一桁』< x ごと）: 入れ替わりの割合（q）・向きの偏り（外→内 ÷ 内→外、q）・母集団の q の和の比（再現 ÷ 本物）・
  再現 2 本どうしの入れ替わり（再現の乱数の床）。
- 線: 元の線（1 号の決定 RC-0: 入れ替わり ≤ 10%・偏り 0.67〜1.5）と、替えた線（1 号の判断 RC-1: 入れ替わり ≤ 床 + 2 ポイント・
  偏り 0.67〜1.5・q の和の比 0.95〜1.05）の両方を出す。判定は替えた線。
- 台帳: family rules_council、purpose eval_only、reads はオッズだけ（KEIBA_SESSION=rules-council）。
使い方: KEIBA_SESSION=rules-council python -m src.research.rules_council.t60_check [--seeds 5]
出力: docs/operations/retrain_trials/rules_council_t60.json。費用 0 円（手元の表だけ）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PRE = ROOT / "data/local/research/persona1/pre_odds.parquet"
OUT = ROOT / "docs/operations/retrain_trials/rules_council_t60.json"
CUTS = (10.0, 9.0, 8.0, 7.0, 6.5)     # 一桁の上端の候補（1 号の決定 RC-0 段 A）
OLD = {"swap_max": 0.10, "bias": (0.67, 1.5)}
NEW = {"floor_plus": 0.02, "bias": (0.67, 1.5), "qratio": (0.95, 1.05)}


def load() -> pd.DataFrame:
    """本物の T-60 の記録があるレースの全頭（確定オッズ・人気）に、T-60 のオッズ（無い馬は NaN）を付ける。"""
    from src.config.data_paths import TABLES_DIR
    pre = pd.read_parquet(PRE)
    pre["race_id"] = pre["race_id"].astype(str)
    pre = pre[(pre["phase"] == "t60") & (pd.to_numeric(pre["win_odds"], errors="coerce") > 0)][["race_id", "horse_number", "win_odds"]]
    pre["horse_number"] = pd.to_numeric(pre["horse_number"], errors="coerce")
    pre = pre.drop_duplicates(["race_id", "horse_number"])
    fl = pd.read_parquet(Path(TABLES_DIR) / "2026" / "race_result_flat.parquet", columns=["race_id", "horse_number", "odds", "popularity", "date"])
    fl["race_id"] = fl["race_id"].astype(str)
    fl = fl[fl["race_id"].isin(set(pre["race_id"]))]
    for c in ("horse_number", "odds", "popularity"):
        fl[c] = pd.to_numeric(fl[c], errors="coerce")
    fl = fl[fl["odds"] > 0]
    m = fl.merge(pre, on=["race_id", "horse_number"], how="left")
    m = m.sort_values(["race_id", "popularity"]).reset_index(drop=True)
    m["date"] = pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d")
    inv = 1.0 / m["odds"].to_numpy(float)
    m["q"] = inv / pd.Series(inv).groupby(m["race_id"].to_numpy()).transform("sum").to_numpy()
    m["d"] = np.log(m["odds"].to_numpy(float) / m["win_odds"].to_numpy(float))
    return m


def pool_of(m: pd.DataFrame) -> list[np.ndarray]:
    """ずれ（確定の人気の順に並べた log(確定 ÷ T-60)）。load_drift と同じく、T-60 と人気のある馬が 5 頭以上のレース。"""
    x = m[m["win_odds"].notna() & (m["popularity"] > 0)]
    return [g.sort_values("popularity")["d"].to_numpy() for _, g in x.groupby("race_id") if len(g) >= 5]


def reproduce(m: pd.DataFrame, pool: list[np.ndarray], seed: int) -> np.ndarray:
    """simulate_pre と同じ作り（人気の欠けは借りたレースの最後の順位）。"""
    rng = np.random.default_rng(1000 + seed)
    out = np.empty(len(m))
    rid = m["race_id"].to_numpy()
    odds = m["odds"].to_numpy(float)
    pop = np.nan_to_num(m["popularity"].to_numpy(float), nan=99.0)
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(m)]
    for s, e in zip(starts, ends):
        src = pool[int(rng.integers(len(pool)))]
        rk = np.clip(np.where(pop[s:e] < 99, pop[s:e], len(src)).astype(int) - 1, 0, len(src) - 1)
        o = odds[s:e] * np.exp(-src[rk])
        out[s:e] = o * (np.sum(1.0 / o) / np.sum(1.0 / odds[s:e]))
    return out


def compare(m: pd.DataFrame, rep: np.ndarray, x: float) -> dict:
    """上端 x の『一桁』（< x）について、再現と本物の T-60 の食い違い（本物の T-60 の記録がある馬だけ）。"""
    has = m["win_odds"].notna().to_numpy()
    real = m["win_odds"].to_numpy(float)[has]
    fin = m["odds"].to_numpy(float)[has]
    q = m["q"].to_numpy(float)[has]
    r = rep[has]
    a, b = real < x, r < x
    io, oi = b & ~a, a & ~b
    return {"本物の母集団の走": int(a.sum()), "本物の母集団の q の和": float(q[a].sum()), "再現の母集団の q の和": float(q[b].sum()),
            "入れ替わりの割合（q）": float((q[io].sum() + q[oi].sum()) / max(q[a].sum(), 1e-9)),
            "向きの偏り（外→内 ÷ 内→外、q）": float(q[io].sum() / max(q[oi].sum(), 1e-9)),
            "q の和の比（再現 ÷ 本物）": float(q[b].sum() / max(q[a].sum(), 1e-9)),
            "外→内の走": int(io.sum()), "内→外の走": int(oi.sum()),
            "外→内の log(確定 ÷ 本物 T-60) の中央値": float(np.median(np.log(fin[io] / real[io]))) if io.any() else None,
            "内→外の log(確定 ÷ 本物 T-60) の中央値": float(np.median(np.log(fin[oi] / real[oi]))) if oi.any() else None,
            "走ごとの一致率": float((a == b).mean())}


def floor(m: pd.DataFrame, pool: list[np.ndarray], x: float, seeds: int) -> float:
    """再現 2 本どうし（種 s と s+100）の入れ替わり（q）の中央値。比べる馬は本物の T-60 の記録がある馬（同じ集合）。"""
    has = m["win_odds"].notna().to_numpy()
    q = m["q"].to_numpy(float)[has]
    v = []
    for s in range(seeds):
        r1, r2 = reproduce(m, pool, s)[has], reproduce(m, pool, s + 100)[has]
        a, b = r1 < x, r2 < x
        v.append((q[a & ~b].sum() + q[b & ~a].sum()) / max(q[a].sum(), 1e-9))
    return float(np.median(v))


def run_version(target: pd.DataFrame, pool: list[np.ndarray], seeds: int) -> dict:
    out = {}
    for x in CUTS:
        rs = [compare(target, reproduce(target, pool, s), x) for s in range(seeds)]
        agg = {}
        for k in rs[0]:
            v = [r[k] for r in rs if r[k] is not None]
            agg[k] = float(np.median(v)) if v else None
        agg["入れ替わりの割合の種ごと"] = [round(r["入れ替わりの割合（q）"], 4) for r in rs]
        agg["向きの偏りの種ごと"] = [round(r["向きの偏り（外→内 ÷ 内→外、q）"], 3) for r in rs]
        fl = floor(target, pool, x, seeds)
        sw, bi, qr = agg["入れ替わりの割合（q）"], agg["向きの偏り（外→内 ÷ 内→外、q）"], agg["q の和の比（再現 ÷ 本物）"]
        agg["再現どうしの入れ替わり（床）"] = fl
        agg["元の線を満たす"] = bool(sw <= OLD["swap_max"] and OLD["bias"][0] <= bi <= OLD["bias"][1])
        agg["替えた線を満たす"] = bool(sw <= fl + NEW["floor_plus"] and NEW["bias"][0] <= bi <= NEW["bias"][1] and NEW["qratio"][0] <= qr <= NEW["qratio"][1])
        out[f"< {x:g}"] = agg
    return out


def final_band(target: pd.DataFrame) -> dict:
    """参考: 確定オッズで母集団を決めた場合と本物の T-60 の食い違い（< 10）。"""
    return compare(target, target["odds"].to_numpy(float), 10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    from src.ops import data_ledger as DL
    from src.research.rules_council import engine as E
    E.check_stop()
    pre_days = pd.read_parquet(PRE, columns=["race_id", "phase", "snapshot_at"])
    period = (str(pre_days["snapshot_at"].min())[:10], str(pre_days["snapshot_at"].max())[:10])
    with DL.guard(tool="rules_council.t60_check", family="rules_council", version="t60_check_v2", period=period, purpose="eval_only",
                  reads=("確定オッズ", "確定の人気", "T-60 のオッズ"), reason="RC-1（議題 4）: 再現の帯と本物の T-60 の帯の食い違い。着順・払戻は読まない",
                  session="rules-council") as tok:
        m0 = load()
        days = np.array(sorted(m0["date"].unique()))
        half = days[len(days) // 2]
        alt_a = set(days[0::2])
        first, second = m0[m0["date"] < half].reset_index(drop=True), m0[m0["date"] >= half].reset_index(drop=True)
        ev, od = m0[m0["date"].isin(alt_a)].reset_index(drop=True), m0[~m0["date"].isin(alt_a)].reset_index(drop=True)
        fwd = m0[m0["date"] >= "2026-11-21"].reset_index(drop=True)
        has = m0["win_odds"].notna()
        out = {"period": [days[0], days[-1]], "days": [str(x) for x in days], "split_day": str(half), "races": int(m0["race_id"].nunique()),
               "runners_with_t60": int(has.sum()), "runners_all": int(len(m0)), "races_first": int(first["race_id"].nunique()),
               "races_second": int(second["race_id"].nunique()), "races_after_1121": int(fwd["race_id"].nunique()),
               "pre_sha256": hashlib.sha256(PRE.read_bytes()).hexdigest()[:16], "seeds": a.seeds,
               "lines_old": {"swap_max": OLD["swap_max"], "bias": list(OLD["bias"])},
               "lines_new": {"floor_plus": NEW["floor_plus"], "bias": list(NEW["bias"]), "qratio": list(NEW["qratio"])},
               "cuts": list(CUTS), "versions": {}}
        V = out["versions"]
        V["主: 前半でずれ → 後半で比べる"] = run_version(second, pool_of(first), a.seeds)
        V["副: 開催日を交互に（奇数日でずれ → 偶数日で比べる）"] = run_version(od, pool_of(ev), a.seeds)
        V["参考: 全部でずれ → 全部で比べる（今の道具の作り。自分と比べる）"] = run_version(m0, pool_of(m0), a.seeds)
        if len(fwd):
            V["参考: 11/21 以降（前半のずれで）"] = run_version(fwd, pool_of(first), a.seeds)
        out["参考: 確定オッズで母集団を決めた場合（後半、< 10）"] = final_band(second)
        mv = V["主: 前半でずれ → 後半で比べる"]
        ok_old = [x for x in CUTS if mv[f"< {x:g}"]["元の線を満たす"]]
        ok_new = [x for x in CUTS if mv[f"< {x:g}"]["替えた線を満たす"]]
        out["元の線で決まった上端"] = max(ok_old) if ok_old else None
        out["替えた線で決まった上端"] = max(ok_new) if ok_new else None
        out["判定（替えた線。1 号の判断 RC-1）"] = ("一桁 < 10 のまま" if out["替えた線で決まった上端"] == 10.0 else
                                       f"母集団を再現 < {out['替えた線で決まった上端']:g} にする" if ok_new else "6.5 倍でも届かない: 議題 1 は流さず 1 号へ返す")
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        tok.finish(result_path=OUT)
    print(json.dumps({k: v for k, v in out.items() if k not in ("days", "versions")}, ensure_ascii=False, indent=1))
    for v, xs in out["versions"].items():
        print(v)
        for c, s in xs.items():
            print("  ", c, "n", s["本物の母集団の走"], "swap %.3f" % s["入れ替わりの割合（q）"], "floor %.3f" % s["再現どうしの入れ替わり（床）"],
                  "bias %.2f" % s["向きの偏り（外→内 ÷ 内→外、q）"], "qratio %.3f" % s["q の和の比（再現 ÷ 本物）"], "new_ok", s["替えた線を満たす"])


if __name__ == "__main__":
    main()
