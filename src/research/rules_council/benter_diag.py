"""RC-6（Benter の 2 段）の着順を使わない診断（1 号の決定 RC-6 の 1-1・1-3・1-4 と、事前登録の λ_be・満期）。

- volume : 1-1 量。本物の T-60・T-15 のあるレースの数と、写しが発走の何分前か（T-45 の印）。2024 年の c13・c21・c23 の買い目の数。
- power  : 1-3 R の検定の検出力の模擬（合成の勝ち。記録用。α は近い値 0.0003・0.0049 で、正式の OBF は OBF の定数）。
- gate1  : 1-3 門 1（対数尤度の差）の検出力（記録用。α は近い値）。
- sstar  : 1-4 s* の区間（2026-10-18〜11-20 の本物の発走前オッズの、市場の対数損失だけ）。
- lambda : 事前登録の λ_be・下限・満期・満期の検出力（config/rules_council/forward_benter_rc6.json に写す）。
- depend : 1-2 再現の作りへの依存（判定のオッズの作りを替えたときの買い目の残り。着順なし）。
- split  : 2-1 探す側 70% の c13 の回収率の分解（探す側の着順を見る。1 回だけ。台帳の guard は無く、回数は decisions.md の通算の欄で数える）。
凍結の候補: config/megu_cycles/candidates/c13_flat.json・c21_maiden_t60.json・c23_flat_t15.json（変えない）。
費用 0 円（手元の表だけ。GCS は読まない）。土日の 07:30〜17:00 は止める。
使い方: KEIBA_SESSION=rules-council python -m src.research.rules_council.benter_diag volume|power|gate1|sstar|lambda|depend|split（nice -n 19）
出力: docs/operations/retrain_trials/rules_council_rc6_<名前>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.rules_council import engine as E

ROOT = Path(__file__).resolve().parents[3]
PRE = ROOT / "data/local/research/persona1/pre_odds.parquet"
CANDS = ("c13_flat", "c21_maiden_t60", "c23_flat_t15")
OUT = ROOT / "docs/operations/retrain_trials"


def load_cand(name: str):
    """凍結の候補の、2017〜24 年の走ごとの表（買う割合 stake 付き）。着順の列は持つが、この道具では使わない。"""
    from src.research.megu_cycles import cycles as C
    from src.research.megu_cycles import mc_data as D
    from src.research.megu_cycles import mc_eval as ME
    sp = json.loads((C.CAND / f"{name}.json").read_text(encoding="utf-8"))
    C.RICH_FILE = D.OUT / sp["probability"]["file"]
    d = C.load_all()
    d = d[ME.seg_mask(d, sp["segment"])].reset_index(drop=True)
    if sp.get("phase") and sp["phase"] != "t60":
        d = C.swap_phase(d, sp["phase"])
    if sp.get("group"):
        d = d[C.group_of(d) == sp["group"]].reset_index(drop=True)
    ab = tuple(sp.get(f"combo_ab_{sp.get('phase', 't60')}") or sp["combo_ab_t60"])
    st = C.strat_ev(lambda x, k: C.combo_p(x, k, *ab, col="p_rich"), sp["t"], sp["omax"])
    d["stake"] = np.vstack([st(d, k) for k in range(D.N_PRE)]).mean(0)
    return d, sp, ab


def cmd_volume() -> dict:
    E.check_stop()
    from src.config.data_paths import TABLES_DIR
    from src.ops import data_ledger as DL
    pre = pd.read_parquet(PRE, columns=["race_id", "phase", "snapshot_at"])
    period = (str(pre["snapshot_at"].min())[:10], str(pre["snapshot_at"].max())[:10])
    out = {}
    with DL.guard(tool="rules_council.benter_diag", family="rules_council", version="rc6_volume", period=period, purpose="eval_only",
                  reads=("T-60・T-15 のオッズの取得の時刻", "発走の時刻"), reason="RC-6 の 1-1: 本物の発走前オッズの量と写しの時刻（着順・払戻は読まない）",
                  session="rules-council") as tok:
        rr = pd.read_parquet(Path(TABLES_DIR) / "2026" / "race_result_race.parquet", columns=["race_id", "date", "start_time"])
        rr["race_id"] = rr["race_id"].astype(str)
        pre["race_id"] = pre["race_id"].astype(str)
        rr = rr[rr["race_id"].isin(set(pre["race_id"]))]          # 写しのあるレースの発走時刻だけ（台帳の期間の中）
        tok.check_dates(pd.to_datetime(rr["date"]).dt.strftime("%Y-%m-%d"))
        snap = pre.drop_duplicates(["race_id", "phase"]).merge(rr, on="race_id", how="left")
        post = pd.to_datetime(snap["date"].astype(str) + " " + snap["start_time"].astype(str), errors="coerce").dt.tz_localize("Asia/Tokyo")
        at = pd.to_datetime(snap["snapshot_at"], errors="coerce")
        snap["min_before"] = (post - at).dt.total_seconds() / 60
        for ph in ("t60", "t15", "t05"):
            s = snap[snap["phase"] == ph]
            mb = s["min_before"].dropna()
            out[ph] = {"races": int(s["race_id"].nunique()), "days": int(s["date"].nunique()),
                       "min_before_q": [float(x) for x in mb.quantile([0.05, 0.25, 0.5, 0.75, 0.95])] if len(mb) else None}
            if ph == "t60":
                out[ph]["t45_like（50 分前より遅い写し）"] = int((mb < 50).sum())
        tok.finish()
    # 2024 年の買い目の率（凍結の候補、再現 5 本の買う割合の和）
    rate = {}
    for c in CANDS:
        d, sp, ab = load_cand(c)
        y = d[d["year"] == 2024]
        rate[c] = {"races_2024": int(y["race_id"].nunique()), "bets_2024": float(y["stake"].sum()),
                   "bets_per_race": float(y["stake"].sum() / max(y["race_id"].nunique(), 1)),
                   "bets_per_month": float(y["stake"].sum() / 12)}
    out["bets_2024"] = rate
    res = OUT / "rules_council_rc6_volume.json"
    res.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def _bet_dist(name: str) -> tuple[pd.DataFrame, dict]:
    """凍結の候補の 2017〜24 年の買い目の、判定の q（再現 T-60 の種 0 のオッズをレースで和 1）・確定オッズ・買う割合。着順は使わない。"""
    d, sp, ab = load_cand(name)
    o0 = d["odds_pre0"].to_numpy(float)
    inv = np.where(o0 > 0, 1.0 / o0, 0.0)
    d["q_dec"] = inv / pd.Series(inv).groupby(d["race_id"].to_numpy()).transform("sum").to_numpy()
    b = d[d["stake"] > 0][["race_id", "year", "q_dec", "odds_final", "odds_pre0", "stake"]].copy()
    return b, sp


def cmd_power(n_sim: int = 4000, seed: int = 20261125) -> dict:
    """1-3 検出力の模擬（合成の勝ち）。勝つ確率 = R × 判定の q（R は 勝ち ÷ Σ 判定の q）。払戻は確定オッズ。
    主の検定（記録用に下げた）: 帰無 R = R_be（回収率 1.00 になる R ＝ 1 ÷ 平均(q × 確定オッズ)）、片側、50% の時点 p ≤ 0.0003・満期 p ≤ 0.0049（近い値。正式の OBF は OBF の定数）。
    止める線: 150 点・300 点で R の推定が線より下なら止める。線は『本当の R = 1.3 で誤って止める確率 ≤ 15%』の最大の値（0.01 刻み）。"""
    from scipy.stats import norm
    E.check_stop()
    rng = np.random.default_rng(seed)
    out = {"n_sim": n_sim, "alpha_final": 0.0049, "alpha_interim": 0.0003, "cands": {}}
    for c in ("c21_maiden_t60", "c23_flat_t15", "c13_flat"):
        b, sp = _bet_dist(c)
        w = (b["stake"] / b["stake"].sum()).to_numpy()
        q, o = b["q_dec"].to_numpy(), b["odds_final"].to_numpy(float)
        r_be = float(1.0 / np.sum(w * q * o))
        per_year = float(b.groupby("year")["stake"].sum().mean())
        res = {"bets_per_year": per_year, "q_mean": float(np.sum(w * q)), "odds_final_mean": float(np.sum(w * o)), "R_breakeven": r_be}

        def draw(R, n):
            i = rng.choice(len(q), size=(n_sim, n), p=w)
            qq, oo = q[i], o[i]
            win = rng.random((n_sim, n)) < np.minimum(R * qq, 0.95)
            return qq, oo, win

        def zstat(qq, win, k):
            Eq = qq[:, :k].sum(1)
            W = win[:, :k].sum(1)
            v = (r_be * qq[:, :k] * (1 - np.minimum(r_be * qq[:, :k], 0.95))).sum(1)
            return (W - r_be * Eq) / np.sqrt(v), W / Eq
        z_i, z_f = norm.isf(0.0003), norm.isf(0.0049)
        grid = [300, 500, 1000, 2000, 4000, 8000, 16000]
        pw = {}
        for R in (1.0, 1.15, 1.3, 1.5):
            row = {}
            for n in grid:
                qq, oo, win = draw(R, n)
                zi, _ = zstat(qq, win, n // 2)
                zf, _ = zstat(qq, win, n)
                row[str(n)] = float(np.mean((zi >= z_i) | (zf >= z_f)))
            pw[str(R)] = row
        res["power_R_test"] = pw
        need = {}
        for R in ("1.3", "1.5"):
            ok = [int(n) for n, v in pw[R].items() if v >= 0.8]
            need[R] = min(ok) if ok else None
        res["points_for_80pct"] = need
        res["years_for_80pct"] = {k: (v / per_year if v else None) for k, v in need.items()}
        # 止める線（150・300 点）
        stops = {}
        for k in (150, 300):
            qq, oo, win = draw(1.3, k)
            _, rh13 = zstat(qq, win, k)
            qq0, oo0, win0 = draw(1.0, k)
            _, rh10 = zstat(qq0, win0, k)
            thr = max([t for t in np.arange(0.5, 1.5, 0.01) if np.mean(rh13 < t) <= 0.15] or [0.5])
            stops[str(k)] = {"line": round(float(thr), 2), "p_stop_if_R1.3": float(np.mean(rh13 < thr)), "p_stop_if_R1.0": float(np.mean(rh10 < thr))}
        res["stop_lines"] = stops
        # 回収率の門（門 2）: 回収率の点推定の SD と、R = 1.3 での回収率
        roi13 = 1.3 / r_be
        sd1 = float(np.sqrt(np.sum(w * (np.minimum(1.3 * q, 0.95) * o ** 2)) - roi13 ** 2))
        n_roi = ((norm.isf(0.0049) + norm.isf(0.2)) * sd1 / max(roi13 - 1, 1e-9)) ** 2 if roi13 > 1 else None
        res["gate2"] = {"roi_if_R1.3": roi13, "sd_per_bet": sd1, "points_for_80pct": n_roi, "years": (n_roi / per_year if n_roi else None)}
        out["cands"][c] = res
        print(c, json.dumps(res, ensure_ascii=False), flush=True)
    (OUT / "rules_council_rc6_power.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def cmd_gate1() -> dict:
    """1-3 の門 1 の検出力（着順なし）: 組んだ確率 p（c13 の a・b）と判定の q の、レースごとの対数尤度の差 log p_勝ち − log q_勝ち。
    真の確率を p とみたとき（Benter が正しい）の 1 レースあたりの期待の差 Σ p log(p/q) と SD、真の確率を q とみたとき（市場が正しい）の値。
    検出力 80%・片側 α 0.005（O'Brien–Fleming の満期 0.0049）に要るレース数。効きを薄めた版（真 = q と p の対数の中間 λ）も出す。"""
    from scipy.stats import norm
    from src.research.megu_cycles import cycles as C
    E.check_stop()
    d, sp, ab = load_cand("c13_flat")
    d = d[d["year"] == 2024].reset_index(drop=True)
    pc = C.combo_p(d, 0, *ab, col="p_rich")
    o0 = d["odds_pre0"].to_numpy(float)
    inv = np.where(o0 > 0, 1.0 / o0, 0.0)
    q = inv / pd.Series(inv).groupby(d["race_id"].to_numpy()).transform("sum").to_numpy()
    rid = d["race_id"].to_numpy()
    lr = np.log(np.maximum(pc, 1e-9)) - np.log(np.maximum(q, 1e-9))
    out = {"races_2024": int(pd.unique(rid).size)}
    z = norm.isf(0.0049) + norm.isf(0.2)
    for lam in (1.0, 0.5, 0.25):
        # 真の確率: log t = λ log p + (1 − λ) log q（レースで和 1）
        lt = lam * np.log(np.maximum(pc, 1e-9)) + (1 - lam) * np.log(np.maximum(q, 1e-9))
        t = np.exp(lt)
        t = t / pd.Series(t).groupby(rid).transform("sum").to_numpy()
        g = pd.DataFrame({"r": rid, "m1": t * lr, "m2": t * lr * lr})
        agg = g.groupby("r")[["m1", "m2"]].sum()
        mu, var = agg["m1"].mean(), (agg["m2"] - agg["m1"] ** 2).mean() + agg["m1"].var()
        n = (z * np.sqrt(var) / mu) ** 2 if mu > 0 else None
        out[f"lambda_{lam}"] = {"mean_ll_diff_per_race": float(mu), "sd_per_race": float(np.sqrt(var)), "races_for_80pct": (float(n) if n else None),
                                "years_at_3200_flat_races": (float(n) / 3200 if n else None)}
    out["note"] = ("λ = 1 は組んだ確率がそのまま真（Benter が正しい）、λ = 0.5・0.25 は効きを半分・4 分の 1 にした真。判定の q は再現 T-60 の種 0。"
                   "本物の T-60 では q が確定に近い分だけ差は小さくなる（楽観）")
    (OUT / "rules_council_rc6_gate1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


S_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
N_REP = 10
OBF = {"alpha": 0.005, "looks": [0.5, 1.0], "z_interim": 3.648, "z_final": 2.580, "p_interim": 0.000132, "p_final": 0.00495,
       "how": "O'Brien–Fleming（等間隔 2 回、相関 √0.5 の 2 変量正規で 片側合計 0.005 を厳密に合わせた値）"}


def _sstar_of(ll_real: np.ndarray, sims: dict) -> float | None:
    t = float(np.mean(ll_real))
    ys = np.array([float(np.mean(sims[s_])) for s_ in S_GRID])
    for i in range(len(S_GRID) - 1):
        if (ys[i] - t) * (ys[i + 1] - t) <= 0 and ys[i + 1] != ys[i]:
            return float(S_GRID[i] + (t - ys[i]) * (S_GRID[i + 1] - S_GRID[i]) / (ys[i + 1] - ys[i]))
    return None


def cmd_sstar(n_boot: int = 1000) -> dict:
    """1-4 s* の区間（megu_cycles.mc_final.cmd_drift の考え方の作り直し。2026-10-18〜11-20、市場の対数損失だけ）。
    cmd_drift との違い（critic の指摘で開示）: 倍率の格子に 0・0.1・0.25 を足した（新馬未勝利で下側が格子の端に貼り付いたため）／
    借りるずれの抽選を 10 回してレースごとに保ち、ブートストラップでは開催日を引き直すたびに、レースごとに 10 回のどれかを引き直す（借りる揺れを区間に入れる）／
    対数損失はレースの勝ち馬（同着は平均）の値の平均／障害を除く。交点が無い回は None として数え、区間はそれを除いて出し、回数を記録する。
    層: 全部（平場の候補の segment flat＝重賞以外 に合わせて重賞も除く）・新馬未勝利。T-60 は写しが 50 分前より遅いレースを外した版も。"""
    E.check_stop()
    from src.config.data_paths import TABLES_DIR
    from src.ops import data_ledger as DL
    out = {"period": ["2026-10-18", "2026-11-20"], "n_boot": n_boot, "n_rep": N_REP, "grid": list(S_GRID), "phases": {}}
    with DL.guard(tool="rules_council.benter_diag:sstar", family="rules_council", version="rc6_sstar_v2", period=("2026-10-18", "2026-11-20"),
                  purpose="eval_only", reads=("着順（勝ち馬だけ）", "発走前オッズ"),
                  reason="RC-6 の 1-4: s* の区間（めぐの drift_check の作り直し。市場の対数損失だけ。合否に使わない）", session="rules-council") as tok:
        pre = pd.read_parquet(PRE)
        pre["race_id"] = pre["race_id"].astype(str)
        fl = pd.read_parquet(Path(TABLES_DIR) / "2026" / "race_result_flat.parquet",
                             columns=["race_id", "date", "horse_number", "odds", "popularity", "finish_position", "start_time"])
        fl["race_id"] = fl["race_id"].astype(str)
        fl["date"] = pd.to_datetime(fl["date"]).dt.strftime("%Y-%m-%d")
        fl = fl[(fl["date"] >= "2026-10-18") & (fl["date"] <= "2026-11-20")].copy()
        for c in ("horse_number", "odds", "popularity", "finish_position"):
            fl[c] = pd.to_numeric(fl[c], errors="coerce")
        fl = fl[fl["odds"] > 0]
        tok.check_dates(fl["date"].astype(str))
        rr = pd.read_parquet(Path(TABLES_DIR) / "2026" / "race_result_race.parquet", columns=["race_id", "grade", "surface"])
        rr["race_id"] = rr["race_id"].astype(str)
        flat_ok = set(rr.loc[rr["surface"].astype(str).str[0].isin(["芝", "ダ"]) & ~rr["grade"].astype(str).isin(["G1", "G2", "G3"]), "race_id"])
        maiden = set(rr.loc[rr["grade"].astype(str).isin(["新馬", "未勝利"]), "race_id"])
        fl = fl[fl["race_id"].isin(flat_ok)]
        for phase in ("t60", "t15", "t05"):
            pp = pre[pre["phase"] == phase][["race_id", "horse_number", "win_odds", "snapshot_at"]].copy()
            pp["horse_number"] = pd.to_numeric(pp["horse_number"], errors="coerce")
            pp["win_odds"] = pd.to_numeric(pp["win_odds"], errors="coerce")
            m = fl.merge(pp, on=["race_id", "horse_number"], how="inner")
            m = m[m["win_odds"] > 0]
            n_all = fl.groupby("race_id").size()
            m = m[m.groupby("race_id")["race_id"].transform("size").to_numpy() == n_all.reindex(m["race_id"]).to_numpy()]
            m = m.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
            rid = m["race_id"].to_numpy()
            post = pd.to_datetime(m["date"] + " " + m["start_time"].astype(str), errors="coerce").dt.tz_localize("Asia/Tokyo")
            m["min_before"] = (post - pd.to_datetime(m["snapshot_at"], errors="coerce")).dt.total_seconds() / 60

            def qn(x):
                inv = 1.0 / x
                return inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
            win = (m["finish_position"].to_numpy() == 1)
            q_real = qn(m["win_odds"].to_numpy(float))
            m["d"] = np.log(m["odds"].to_numpy(float) / m["win_odds"].to_numpy(float))
            races = {r: g.sort_values("popularity")["d"].to_numpy() for r, g in m.groupby("race_id")}
            keys = [r for r in races if win[pd.Series(np.arange(len(m))).groupby(rid).indices[r]].any()]
            idx_of = pd.Series(np.arange(len(m))).groupby(rid).indices
            rng = np.random.default_rng(7)
            ll_real = np.array([float(np.mean(-np.log(np.clip(q_real[idx_of[r]][win[idx_of[r]]], 1e-9, 1)))) for r in keys])
            sim = {s_: np.zeros((len(keys), N_REP)) for s_ in S_GRID}
            for rep in range(N_REP):
                donors = {r: races[keys[int(rng.integers(len(keys)))]] for r in keys}
                for r in keys:
                    while len(keys) > 1 and donors[r] is races[r]:
                        donors[r] = races[keys[int(rng.integers(len(keys)))]]
                for s_ in S_GRID:
                    so = np.full(len(m), np.nan)
                    for r in keys:
                        idx = idx_of[r]
                        src = donors[r]
                        pop = m["popularity"].to_numpy()[idx]
                        rk = np.clip(np.nan_to_num(pop, nan=len(src)).astype(int) - 1, 0, len(src) - 1)
                        so[idx] = m["odds"].to_numpy(float)[idx] * np.exp(-s_ * src[rk])
                    ok = ~np.isnan(so)
                    qs = np.full(len(m), np.nan)
                    inv = 1.0 / so[ok]
                    qs[ok] = inv / pd.Series(inv).groupby(rid[ok]).transform("sum").to_numpy()
                    sim[s_][:, rep] = [float(np.mean(-np.log(np.clip(qs[idx_of[r]][win[idx_of[r]]], 1e-9, 1)))) for r in keys]
            R = pd.DataFrame({"race_id": keys})
            R["date"] = R["race_id"].map(m.groupby("race_id")["date"].first())
            R["late"] = (R["race_id"].map(m.groupby("race_id")["min_before"].median()) < 50).to_numpy()
            R["maiden"] = R["race_id"].isin(maiden).to_numpy()

            def boot(mask):
                ii = np.flatnonzero(mask)
                if len(ii) < 20:
                    return {"races": int(len(ii)), "s_star": None}
                point = _sstar_of(ll_real[ii], {s_: sim[s_][ii].mean(1) for s_ in S_GRID})
                days = R["date"].to_numpy()[ii]
                ud = np.unique(days)
                by_day = {d_: ii[days == d_] for d_ in ud}
                rb = np.random.default_rng(20261125)
                v, none = [], 0
                for _ in range(n_boot):
                    pick = np.concatenate([by_day[d_] for d_ in rb.choice(ud, size=len(ud))])
                    reps = rb.integers(0, N_REP, size=len(pick))
                    st = _sstar_of(ll_real[pick], {s_: sim[s_][pick, reps] for s_ in S_GRID})
                    if st is None:
                        none += 1
                    else:
                        v.append(st)
                q = (lambda a, b: [float(np.quantile(v, a)), float(np.quantile(v, b))]) if v else (lambda a, b: None)
                return {"races": int(len(ii)), "s_star": point, "ci90": q(0.05, 0.95), "ci95": q(0.025, 0.975), "boot_no_crossing": none}
            allm = np.ones(len(R), bool)
            res = {"all": boot(allm), "maiden": boot(R["maiden"].to_numpy()), "other": boot(~R["maiden"].to_numpy())}
            if phase == "t60":
                res["all_without_late_snapshots"] = boot(~R["late"].to_numpy())
            out["phases"][phase] = res
            print(phase, json.dumps(res, ensure_ascii=False), flush=True)
        res_p = OUT / "rules_council_rc6_sstar.json"
        res_p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        tok.finish(result_path=res_p)
    return out


FAMILIES = {"G1-a": ("c13_flat", "平場（重賞以外。新馬未勝利を含む）・T-60・c13 の a・b", "all", 3200),
            "G1-b": ("c23_flat_t15", "平場（重賞以外。新馬未勝利を含む）・T-15・c23 の a・b", "all", 3200),
            "G1-c": ("c21_maiden_t60", "新馬未勝利・T-60・c21 の a・b", "maiden", None)}


def cmd_lambda() -> dict:
    """事前登録の λ_be と満期（着順なし。2024 年の行）。p_λ ∝ q·exp(λ·d)、d = (a·log p_rich + b·log q) − log q。
    q は判定の時点の再現のオッズを倍率 s で動かした値（odds_s = 確定 × (再現 ÷ 確定)^s をレースで和 1）。
    s は 1・s* の点推定・s* の 90% 区間の下側（族の時点の「全部」の層の値。新馬未勝利の層は使わない）。
    λ_be(s): 凍結の買い目（λ = 1 の確率 × odds_s ≥ t・odds_s ≤ 50）で、p_λ が真のときの回収率の見込み Σ p_λ·確定オッズ ÷ 点数 が 1 になる λ。
    下限（事前登録の lambda_be_floor）＝ 族の時点の「全部」の層の s* の点推定での、族の買い目の λ_be（規則を全族で同じにする）。
    検出力は Fisher の SE で計算する（本番の検定は開催日で束ねたサンドイッチの SE なので、ここの検出力は楽観の側）。
    満期: 線 = max(λ_be(s), 下限)。λ̂ の標準誤差 ≈ 1 ÷ √(レース数 × Fisher 情報)。真の λ = 1 で 満期の線（OBF の z_final）を 80% で越えるレース数を s ごとに出し、
    一番多い値（上限 3,200）を満期にする。新馬未勝利は 1 年（2024 年のレース数）に固定し、検出力を書く。"""
    from scipy.optimize import brentq
    from scipy.stats import norm
    E.check_stop()
    ss = json.loads((OUT / "rules_council_rc6_sstar.json").read_text(encoding="utf-8"))["phases"]
    zf = OBF["z_final"]
    zb = norm.isf(0.2)
    out = {"obf": OBF, "families": {}}
    for fam, (cand, label, layer, cap) in FAMILIES.items():
        d, sp, ab = load_cand(cand)
        a, b = ab
        y = d[d["year"] == 2024].reset_index(drop=True)
        rid = y["race_id"].to_numpy()
        ph = sp.get("phase", "t60")
        lay = ss[ph]["all"]          # 1 号の判断: 下限と s の振りは、族の時点の「全部」の層の s* を使う（新馬未勝利の層は交点なし 17% で不安定）
        s_pt = float(lay["s_star"])
        s_low = float(lay["ci90"][0])
        fo = y["odds_final"].to_numpy(float)
        n_races = int(pd.unique(rid).size)
        res = {"label": label, "phase": ph, "layer": layer, "a": a, "b": b, "t": sp["t"], "omax": sp["omax"], "races_2024": n_races,
               "s_star_point": s_pt, "s_star_low90": s_low, "by_s": {}}

        def stats_at(s_):
            o_s = fo * (y["odds_pre0"].to_numpy(float) / fo) ** s_
            inv = 1.0 / o_s
            q = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
            lp = np.log(np.clip(y["p_rich"].to_numpy(float), 1e-6, 1))
            dlt = (a * lp + b * np.log(np.clip(q, 1e-6, 1))) - np.log(np.clip(q, 1e-6, 1))

            def p_lam(lam):
                w = q * np.exp(lam * (dlt - pd.Series(dlt).groupby(rid).transform("max").to_numpy()))
                return w / pd.Series(w).groupby(rid).transform("sum").to_numpy()
            p1 = p_lam(1.0)
            buy = (p1 * o_s >= sp["t"]) & (o_s <= sp["omax"])

            def roi(lam):
                return float((p_lam(lam)[buy] * fo[buy]).sum() / max(int(buy.sum()), 1))
            lam_be = float(brentq(lambda L: roi(L) - 1.0, -2, 3)) if (roi(-2) - 1) * (roi(3) - 1) < 0 else None
            g = pd.DataFrame({"r": rid, "m1": p1 * dlt, "m2": p1 * dlt * dlt})
            agg = g.groupby("r")[["m1", "m2"]].sum()
            info = float((agg["m2"] - agg["m1"] ** 2).mean())
            return {"bets_2024": int(buy.sum()), "roi_if_lambda0": roi(0.0), "roi_if_lambda1": roi(1.0), "lambda_be": lam_be, "fisher_info_per_race": info}
        svals = {"s_one": 1.0, "s_point": round(s_pt, 4), "s_low90": round(s_low, 4)}       # 名前で持つ（点推定が 1 に近くても上書きしない）
        rows = {f"{k} ({v:.3f})": stats_at(v) for k, v in svals.items()}
        floor = rows[f"s_point ({svals['s_point']:.3f})"]["lambda_be"]
        if floor is None:
            raise SystemExit("s* の点推定で λ_be が求まらない（下限を決められない）")
        need = []
        for k, r in rows.items():
            line = max(r["lambda_be"] if r["lambda_be"] is not None else 9, floor)
            r["line"] = line
            r["races_for_80pct"] = float(((zf + zb) / ((1 - line) * np.sqrt(r["fisher_info_per_race"]))) ** 2) if line < 1 else None
            need.append(r["races_for_80pct"] if r["races_for_80pct"] else np.inf)
        mat = n_races if cap is None else int(min(cap, np.ceil(max(need) / 100) * 100))
        for k, r in rows.items():
            se = 1 / np.sqrt(mat * r["fisher_info_per_race"])
            r["power_at_maturity"] = float(norm.cdf((1 - r["line"]) / se - zf)) if r["line"] < 1 else 0.0
        res["by_s"] = rows
        res["lambda_be_floor"] = floor
        res["races_at_maturity"] = mat
        res["maturity_rule"] = "一番多い必要数（上限 3,200）" if cap else "1 年（2024 年のレース数）に固定"
        out["families"][fam] = res
        print(fam, json.dumps(res, ensure_ascii=False), flush=True)
    (OUT / "rules_council_rc6_lambda.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


DEP_S = (0.5, 0.86, 1.04, 1.62)


def _band_pool(phase: str) -> tuple[list[np.ndarray], int]:
    """2026 年の本物の写し（phase）のずれ d = log(確定 ÷ 写し) を、確定オッズの帯（rc_frame.ODDS_EDGES）ごとに集める。"""
    from src.config.data_paths import TABLES_DIR
    from src.research.rules_council import rc_frame as RF
    pre = pd.read_parquet(PRE)
    pre["race_id"] = pre["race_id"].astype(str)
    pre = pre[(pre["phase"] == phase) & (pd.to_numeric(pre["win_odds"], errors="coerce") > 0)][["race_id", "horse_number", "win_odds"]]
    pre["horse_number"] = pd.to_numeric(pre["horse_number"], errors="coerce")
    fl = pd.read_parquet(Path(TABLES_DIR) / "2026" / "race_result_flat.parquet", columns=["race_id", "horse_number", "odds", "popularity"])
    fl["race_id"] = fl["race_id"].astype(str)
    for c in ("horse_number", "odds", "popularity"):
        fl[c] = pd.to_numeric(fl[c], errors="coerce")
    m = pre.merge(fl, on=["race_id", "horse_number"])
    m = m[(m["odds"] > 0) & (m["popularity"] > 0)]                       # load_drift と同じ母集団
    dd = np.log(m["odds"].to_numpy(float) / pd.to_numeric(m["win_odds"]).to_numpy(float))
    b = RF.odds_bin(m["odds"].to_numpy(float))
    nb = len(RF.ODDS_EDGES) - 1
    pool = [dd[b == k] for k in range(nb)]
    for k in range(nb):                                   # 空の帯は、近い帯から借りる
        if not len(pool[k]):
            j = min((j for j in range(nb) if len(pool[j])), key=lambda j: abs(j - k))
            pool[k] = pool[j]
    return pool, int(len(m))


def _sim_band(odds: np.ndarray, rid: np.ndarray, pool: list[np.ndarray], s: float, seed: int) -> np.ndarray:
    """帯ごとのずれから馬ごとに無作為に引き、判定のオッズ = 確定 × exp(−s·d)。レースで 1/オッズの和を確定に合わせる。"""
    from src.research.rules_council import rc_frame as RF
    rng = np.random.default_rng(seed)
    b = RF.odds_bin(odds)
    d = np.empty(len(odds))
    for k in np.unique(b):
        i = np.flatnonzero(b == k)
        d[i] = pool[k][rng.integers(len(pool[k]), size=len(i))]
    o = odds * np.exp(-s * d)
    g = pd.Series(1.0 / odds).groupby(rid).transform("sum").to_numpy() / pd.Series(1.0 / o).groupby(rid).transform("sum").to_numpy()
    return o / g


def _sim_rank(d: pd.DataFrame, pool_races: list, s: float, seed0: int) -> np.ndarray:
    """megu の作り（レースを丸ごと借り、確定の人気の順位のずれを当てる）に s を掛けた版。種を変えると、元の 5 本どうしの揺れ（床）が出る。"""
    rid = d["race_id"].to_numpy()
    odds = d["odds_final"].to_numpy(float)
    pop = np.nan_to_num(pd.to_numeric(d["pop_final"], errors="coerce").to_numpy(float), nan=99.0)
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(d)]
    rng = np.random.default_rng(seed0)
    out = np.empty(len(d))
    for a, e in zip(starts, ends):
        src = pool_races[int(rng.integers(len(pool_races)))]
        rk = np.clip(np.where(pop[a:e] < 99, pop[a:e], len(src)).astype(int) - 1, 0, len(src) - 1)
        o = odds[a:e] * np.exp(-s * src[rk])
        out[a:e] = o * (np.sum(1.0 / o) / np.sum(1.0 / odds[a:e]))
    return out


def cmd_depend(seed: int = 20261126) -> dict:
    """1-2 再現への依存（着順なし。1 号の決定 RC-6 の 1-2）。
    凍結の 3 候補の 2017〜24 年の買い目（再現 5 本の平均の買う割合 stake）が、判定のオッズの作りを替えるとどれだけ残るか（Σ min(元, 新) ÷ Σ 元）。
    - band_s: 帯ごとのずれから馬ごとに無作為に引く版（確定の人気の順位を使わない）。s = 0.5・0.86・1.04・1.62。
    - rank_s: 元の作り（人気の順位）に s を掛けた版。s = 1 で種だけ変えたものが床（元の作りどうしの揺れ）。
    残りが 5 割未満（入れ替わりが 5 割以上）なら『過去の上積みは再現の作りが生んだ値』と記録し、模擬に植える効きを R 1.15 に下げる。"""
    E.check_stop()
    from src.ops import data_ledger as DL
    from src.research.megu_cycles import cycles as C
    from src.research.megu_cycles import mc_data as D
    from src.scripts.data.bet_cycles import load_drift
    pre = pd.read_parquet(PRE, columns=["snapshot_at"])
    period = (str(pre["snapshot_at"].min())[:10], str(pre["snapshot_at"].max())[:10])
    out = {"s_grid": list(DEP_S), "line": "残り < 0.5 なら『再現の作りが生んだ値』", "cands": {}}
    with DL.guard(tool="rules_council.benter_diag", family="rules_council", version="rc6_depend", period=period, purpose="eval_only",
                  reads=("T-60・T-15 のオッズ", "確定オッズ"), reason="RC-6 の 1-2: 再現の作りへの依存（買い目の入れ替わり。着順・払戻の列は load_all が読み込むが使わない）",
                  session="rules-council") as tok:
        pools = {}
        for name in CANDS:
            E.check_stop()
            d, sp, ab = load_cand(name)
            d = d.sort_values(["race_id", "hn"]).reset_index(drop=True)
            phase = sp.get("phase", "t60")
            if phase not in pools:
                pools[phase] = (_band_pool(phase), load_drift(phase)["races"])
            (bpool, n_pool), rpool = pools[phase]
            st = C.strat_ev(lambda x, k: C.combo_p(x, k, *ab, col="p_rich"), sp["t"], sp["omax"])
            base = d["stake"].to_numpy(float)
            odds = d["odds_final"].to_numpy(float)
            rid = d["race_id"].to_numpy()

            def stake_with(make) -> np.ndarray:
                e = d.copy()
                for k in range(D.N_PRE):
                    e[f"odds_pre{k}"] = make(k)
                return np.vstack([st(e, k) for k in range(D.N_PRE)]).mean(0)

            def keep(new: np.ndarray) -> dict:
                m = float(np.minimum(base, new).sum())
                # kept は Σ新 ÷ Σ元 が上限（数が減るだけで下がる）。数の増減に左右されない kept_sym（Σmin ÷ Σmax）と kept_norm（Σmin ÷ min(Σ元, Σ新)）も並べる
                return {"kept": m / float(base.sum()), "kept_sym": m / float(np.maximum(base, new).sum()),
                        "kept_norm": m / float(min(base.sum(), new.sum())), "bets_ratio": float(new.sum() / base.sum())}
            r = {"phase": phase, "rows": int(len(d)), "bets_base": float(base.sum()), "pool_rows_2026": n_pool}
            r["floor_rank_s1_reseed"] = keep(stake_with(lambda k: _sim_rank(d, rpool, 1.0, seed + 100 + k)))
            for s in DEP_S:
                r[f"band_s{s}"] = keep(stake_with(lambda k, s=s: _sim_band(odds, rid, bpool, s, seed + 1000 * round(s * 100) + k)))
                r[f"rank_s{s}"] = keep(stake_with(lambda k, s=s: _sim_rank(d, rpool, s, seed + 200 + 1000 * round(s * 100) + k)))
            worst = min(r[f"band_s{s}"]["kept"] for s in DEP_S)
            r["verdict_literal"] = "残り < 0.5 の s がある（決定の線どおりなら R 1.15 に下げる）" if worst < 0.5 else "どの s でも残り ≥ 0.5"
            r["band_minus_rank"] = {f"s{s}": r[f"band_s{s}"]["kept"] - r[f"rank_s{s}"]["kept"] for s in DEP_S}
            out["cands"][name] = r
            print(name, json.dumps(r, ensure_ascii=False), flush=True)
        path = OUT / "rules_council_rc6_depend.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        tok.finish(result_path=path)
    return out


def cmd_split(n_boot: int = 2000, seed: int = 20261127) -> dict:
    """2-1 分解（1 号の決定 RC-6 の 2-1。探す側 70% の着順を見る 6 回目。1 回だけ。結果で凍結・線・母集団を変えない）。
    c13 の買い目（再現 5 本の平均の stake）の、探す側 70% の開催（2017〜24）の回収率を ROI = 基準線 × 超過の市場比 × 払戻の項 に分ける。
    - 基準線 B = Σ s·ROI_帯 ÷ Σ s（ROI_帯 = 同じ表・探す側 70% の全馬の、確定オッズの帯ごとの単勝回収率）。
    - 超過の市場比 = MR ÷ MR_帯（MR = Σ s·勝ち ÷ Σ s·q確定、MR_帯 = Σ s·q·ρ_帯 ÷ Σ s·q、ρ_帯 = 帯の 勝ち ÷ Σq）。
    - 払戻の項 = (ROI ÷ B) ÷ (MR ÷ MR_帯)。勝った馬が帯の中で高いオッズに偏ったかどうか（運か物差しのずれ）。
    信頼区間は開催で束ねたブートストラップ。確かめの側 30% は開かない。"""
    E.check_stop()
    from src.research.megu_cycles import mc_eval as ME
    from src.research.rules_council import rc_frame as RF
    d, sp, ab = load_cand("c13_flat")
    d = d[d["year"].astype(int) <= 2024].reset_index(drop=True)
    d["kaisai"] = d["race_id"].astype(str).str[:8]
    d = d[E.search70(d["kaisai"].to_numpy())].reset_index(drop=True)
    s = d["stake"].to_numpy(float)
    pay = ME._ret(d, "win")
    win = ME._hit(d, "win")
    q = d["q_final"].to_numpy(float)
    b = RF.odds_bin(d["odds_final"].to_numpy(float))
    ks = d["kaisai"].to_numpy()

    def terms(idx: np.ndarray) -> dict:
        s_, p_, w_, q_, b_ = s[idx], pay[idx], win[idx], q[idx], b[idx]
        nb = len(RF.ODDS_EDGES) - 1
        n_b = np.bincount(b_, minlength=nb).astype(float)
        roi_b = np.divide(np.bincount(b_, p_, nb), n_b, out=np.zeros(nb), where=n_b > 0)
        qb = np.bincount(b_, q_, nb)
        rho_b = np.divide(np.bincount(b_, w_, nb), qb, out=np.ones(nb), where=qb > 0)
        S = s_.sum()
        roi = (s_ * p_).sum() / S
        B = (s_ * roi_b[b_]).sum() / S
        mr = (s_ * w_).sum() / (s_ * q_).sum()
        mr_b = (s_ * q_ * rho_b[b_]).sum() / (s_ * q_).sum()
        return {"roi": roi, "base": B, "excess_mr": mr / mr_b, "payout_term": (roi / B) / (mr / mr_b), "mr": mr, "mr_band": mr_b}
    allidx = np.arange(len(d))
    pt = terms(allidx)
    uk = np.unique(ks)
    pos = {k: np.flatnonzero(ks == k) for k in uk}
    rng = np.random.default_rng(seed)
    bs = []
    for i in range(n_boot):
        if i % 200 == 0:
            E.check_stop()
        idx = np.concatenate([pos[k] for k in rng.choice(uk, size=len(uk))])
        bs.append(terms(idx))
    ci = {k: [float(np.quantile([x[k] for x in bs], 0.05)), float(np.quantile([x[k] for x in bs], 0.95))] for k in pt}
    out = {"cand": "c13_flat", "side": "探す側 70%（2017〜24）", "kaisai": int(len(uk)), "bets": float(s.sum()),
           "point": {k: float(v) for k, v in pt.items()}, "ci90": ci,
           "note": "探す側 70% の着順を見た 6 回目。結果で凍結・線・母集団を変えない。確かめの側 30% は開かない"}
    path = OUT / "rules_council_rc6_split.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["volume", "power", "gate1", "sstar", "lambda", "depend", "split"])
    a = ap.parse_args()
    {"volume": cmd_volume, "power": cmd_power, "gate1": cmd_gate1, "sstar": cmd_sstar, "lambda": cmd_lambda, "depend": cmd_depend, "split": cmd_split}[a.cmd]()


if __name__ == "__main__":
    main()
