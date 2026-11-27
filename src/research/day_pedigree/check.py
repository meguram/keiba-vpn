"""段 2 の確かめ（主の検定 1 本）。着順を使う（台帳 select・開発の行 2016〜24 だけ）。

    KEIBA_SESSION=day-pedigree OMP_NUM_THREADS=4 nice -n 19 python -m src.research.day_pedigree.check select
    （30% の側は rules-council の 1 号の許可の後だけ: ... check confirm --approval <decisions.md の見出し>）

固定（docs/operations/day_pedigree/log.md「段 1 の固定」。数を見る前に commit した）:
- 主の検定: A 族（aptitude-group の vec_v1 の a_wet・a_front・a_pace_h・a_pace_s・a_dirt・a_dist の 6 次元を同時に推定）・確定オッズ・
  2017〜24 年・k ≥ 2・9 頭以上のレース。式は logit P(5 着内) = logit π̃ + α·s（オフセット）。
- 日の効きの入力は、確定（発走の 15 分後とみなす）が「そのレースの発走 − 60 分」より前のレースだけ（rules-council の T-60）。
  レースの中で使える前のレースが 0 本なら s = 0（α に寄与しない）。
- 帰無 (i) 日の入れ替え: 同じ場 × 芝ダ × 季節（3 か月）× 馬場の 3 段（良・稍重・重と不良）× 開催の週の前後（日目 ≤ 4）で別の年の日を、
  日ごとに 1 つ無作為に当てる（200 回）。当てる事後は、その日の同じ数の前のレースまで（足りなければその日の全部）で作る。
- 帰無 (ii) 札の入れ替え: 前のレース（更新の入力）の行だけ、レースの中で馬の特徴の行を入れ替える（判定のレースはそのまま・200 回）。
  入れ替えた U に静的な効きの分 I·μ を足し戻す（「日の効き = 0・静的な効き = μ」のデータにする）。
- select では 30% の側の行を読み込みの直後に落とす。帰無 (i) の候補の日も同じ part の日だけ。
- φ は、作り物（π̃ から m 頭を引く）と実データの Pearson の比（2016 年）の大きい方（1 以上）。Σ π̃ は min(5, 頭数) にそろえる。
- 合否（4 つとも）: α の 95% 区間の下限 > 0、本物 − 帰無 (i) の平均の 95% 区間の下限 > 0、α > 帰無 (ii) の 95% 点、
  2 期（2017〜20・2021〜24）とも α > 0。区間は開催の束ねの頑健な分散と、開催の単位のブートストラップ（1,000 回）。
- 30% を開ける前の検出力（rules-council）: 70% の α を半分に割り引いた大きさで、30% の側の標準誤差（70% の SE × √(70% の情報 / 30% の情報)）
  から Φ(α/2 / SE − 1.96)。0.5 未満なら 30% を開けずに閉じる。
- 5 着内・確定オッズの α は、市場の上の情報の検定。回収率・買い目はこの道具では出さない。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from src.research.day_pedigree import data as dd
from src.research.day_pedigree import posterior as po

A_COLS = ["a_wet", "a_front", "a_pace_h", "a_pace_s", "a_dirt", "a_dist"]
TABLES = "/home/jovyan/work/keiba-vpn/data/page_reference/tables"
DECISIONS = Path(__file__).resolve().parents[3] / "docs/operations/roi_plus_council/decisions.md"
VEC = "/home/jovyan/work/aptitude_group_out/vec_v1.parquet"     # aptitude-group の段 1-a の出力（同じ 6 次元を共有する）
CONF_MIN = 15          # 確定は発走の 15 分後とみなす
TEST_LAG = 60          # 検定は T-60
SHOW_LAG = 0           # 表示は発走の時点
N_NULL, N_BOOT = int(os.environ.get("DAYPED_NNULL", 200)), int(os.environ.get("DAYPED_NBOOT", 1000))   # 試しの小さい回数だけ環境変数で替える（本番の結果は 200・1000）
PERIODS = {"2017-20": (2017, 2020), "2021-24": (2021, 2024)}


def start_times() -> pd.DataFrame:
    fr = []
    for y in range(2016, 2025):
        d = pd.read_parquet(f"{TABLES}/{y}/race_result_flat.parquet", columns=["race_id", "date", "start_time"])
        fr.append(d.drop_duplicates("race_id"))
    d = pd.concat(fr, ignore_index=True)
    d["t_start"] = pd.to_datetime(d["date"].astype(str).str[:10] + " " + d["start_time"].astype(str), errors="coerce")
    return d[["race_id", "t_start"]]


def build(part: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """行（馬）とレースの表。A 族の列・π̃・y を付ける。

    select（part 0）のときは、30% の側（part 1）の行を読み込みの直後に落とす（着順を持ち込まない。critic の 1）。
    30% の側の判定の行の数（着順を使わない）だけを返す。confirm のときは両方を残す（事前は part 0 だけから作る）。
    """
    R = dd.load_rows()
    n30 = int(((R["part"] == 1) & (R["year"] >= 2017) & (R["k"] >= 2)).sum())
    if part == 0:
        R = R[R["part"] == 0].reset_index(drop=True)
    V = pd.read_parquet(VEC, columns=["race_id", "horse_id"] + A_COLS)
    V["horse_id"] = V["horse_id"].astype(str)
    R = R.merge(V.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    R[A_COLS] = R[A_COLS].fillna(0.0)          # vec_v1 の無い馬は 0（尺度を前の年の sd で割った値の中心。レースの中で中心化する）
    R = R.merge(start_times(), on="race_id", how="left")
    assert R["t_start"].notna().all()
    R["y"] = (R["finish_position"] <= dd.TOP).astype(float)
    R["m"] = np.minimum(dd.TOP, R["n_run"]).astype(float)   # Σ π̃ は 5 にそろえる（同着の数は判定のレースの結果なので使わない。critic の 12）
    R["rid"] = pd.factorize(R["race_id"])[0]
    R["lq"] = np.log(R["q"].clip(1e-4, 1))
    fitm = (R["part"] == 0) & (R["year"] == 2016)
    beta = po.fit_eta(R.loc[fitm, "lq"].to_numpy(), R.loc[fitm, "y"].to_numpy())
    eta = beta[0] + beta[1] * R["lq"] + beta[2] * R["lq"] ** 2
    R["pt"] = po.calibrate(eta.to_numpy(), R["rid"].to_numpy(), R["m"].to_numpy())
    Rc = R.drop_duplicates("rid").sort_values("rid")[["rid", "race_id", "day", "surf", "year", "part", "k", "k_last", "rno", "meeting",
                                                       "venue", "date", "track_condition", "t_start"]].reset_index(drop=True)
    Rc["t_conf"] = Rc["t_start"] + pd.Timedelta(minutes=CONF_MIN)
    tc = Rc["track_condition"].fillna("良")
    Rc["going3"] = np.where(tc.eq("良"), 0, np.where(tc.eq("稍重"), 1, 2))
    Rc["season"] = (Rc["date"].dt.month % 12) // 3
    Rc["whalf"] = (Rc["race_id"].str[8:10].astype(int) > 4).astype(int)     # 日目（race_id の 9〜10 桁目）≤ 4 が前半
    R.attrs["eta_beta"] = beta.tolist()
    return R, Rc, n30


def scores(R: pd.DataFrame, dev: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """行ごとの s = dev_rᵀ (x − x̄_w)。dev はレースごとの θ̂ − μ（日の効き）。欠けのレースは 0。"""
    d = dev[R["rid"].to_numpy()]
    s = np.nansum(d * Xc, axis=1)
    return np.where(np.isfinite(d).all(axis=1), s, 0.0)


def mu_by_race(Rc: pd.DataFrame, pri: dict, Fn: int) -> np.ndarray:
    out = np.full((len(Rc), Fn), np.nan)
    for i, (s, y) in enumerate(zip(Rc["surf"], Rc["year"])):
        p = pri.get((s, y))
        if p is not None:
            out[i] = p["mu"]
    return out


def gh(y, pt, s, cl, a: float) -> pd.DataFrame:
    """開催ごとのスコアの和 g と情報 h（α̂ のまわり。ブートストラップの 1 段の推定 α̂ + Σg/Σh に使う。critic の 10）。"""
    p = expit(logit(np.clip(pt, 1e-9, 1 - 1e-9)) + a * s)
    return pd.DataFrame({"g": (y - p) * s, "h": p * (1 - p) * s * s, "cl": cl}).groupby("cl")[["g", "h"]].sum()


def day_paths(Rc: pd.DataFrame, dev: np.ndarray, nprev: np.ndarray, races: np.ndarray) -> dict:
    """日ごとに、前のレースの数 n → 日の効き（θ̂ − μ）の列。n は飛ぶことがある（同じ窓で 2 本が確定）。"""
    P: dict[str, list] = {}
    for i in races:
        if nprev[i] > 0 and np.isfinite(dev[i]).all():
            P.setdefault(Rc.at[i, "day"], []).append((int(nprev[i]), dev[i]))
    return {d: sorted(v, key=lambda t: t[0]) for d, v in P.items()}


def lookup(path: list, n: int):
    """n 以下で最大の n の日の効き（無ければ None。critic の 5）。"""
    best = None
    for k, v in path:
        if k <= n:
            best = v
        else:
            break
    return best


def evaluate(R: pd.DataFrame, Rc: pd.DataFrame, part: int, U, I, Xc, phi, pri, rng) -> dict:
    Fn = U.shape[1]
    mean, var, nprev = po.day_posterior(Rc, U, I, phi, pri, TEST_LAG)
    mu_r = mu_by_race(Rc, pri, Fn)
    dev = mean - mu_r
    rid = R["rid"].to_numpy()
    s = scores(R, dev, Xc)
    sel = ((R["part"] == part) & (R["year"] >= 2017) & (R["k"] >= 2)).to_numpy()
    rsel = ((Rc["part"] == part) & (Rc["year"] >= 2017) & (Rc["k"] >= 2)).to_numpy()
    y, pt, cl = R["y"].to_numpy(), R["pt"].to_numpy(), R["meeting"].to_numpy()
    res = {"rows": int(sel.sum()), "races": int(rsel.sum()), "rows_with_prev": int((sel & (nprev[rid] > 0)).sum()),
           "mean_prev": float(nprev[rsel].mean())}
    res["main"] = po.offset_alpha(y[sel], pt[sel], s[sel], cl[sel])
    a_hat = res["main"]["alpha"]
    res["s_sd"] = float(np.std(s[sel]))
    for nm, (a, b) in PERIODS.items():
        m = sel & (R["year"].between(a, b)).to_numpy()
        res[f"period_{nm}"] = po.offset_alpha(y[m], pt[m], s[m], cl[m])
    if not np.isfinite(a_hat):          # τ² が全部 0 などで s ≡ 0。合否は不合格
        res["pass"] = {"alpha_defined": False}
        res["pass_all"] = False
        return res
    # 帰無 (i) 日の入れ替え（同じ part の日だけ・別の年・同じ層。当てるのは「その日の効き θ̂ − μ」で、年の μ の差は入れない）
    keyc = ["venue", "surf", "season", "going3", "whalf"]
    pr = Rc[(Rc["part"] == part) & (Rc["year"] >= 2017)]
    dayinfo = pr.groupby("day").agg(**{c: (c, "first") for c in keyc}, year=("year", "first")).reset_index()
    pools = {k: g[["day", "year"]].to_numpy() for k, g in dayinfo.groupby(keyc)}
    paths = day_paths(Rc, dev, nprev, pr.index.to_numpy())
    races_sel = np.flatnonzero(rsel)
    nulls_i, gh_i = [], []
    for rep in range(N_NULL):
        swap = {}
        for ds in pools.values():
            for d_, y_ in ds:
                cand = [x for x, yy in ds if yy != y_]
                swap[d_] = cand[rng.integers(len(cand))] if cand else None
        dev_n = np.full_like(dev, np.nan)
        for i in races_sel:
            d2 = swap.get(Rc.at[i, "day"])
            if d2 is None or nprev[i] == 0 or d2 not in paths:
                continue
            v = lookup(paths[d2], int(nprev[i]))
            if v is not None:
                dev_n[i] = v
        s_n = scores(R, dev_n, Xc)
        an = po.offset_alpha(y[sel], pt[sel], s_n[sel], cl[sel])["alpha"]
        nulls_i.append(an)
        gh_i.append((an, gh(y[sel], pt[sel], s_n[sel], cl[sel], an if np.isfinite(an) else 0.0)))
    res["null_i"] = {"mean": float(np.nanmean(nulls_i)), "q95": float(np.nanpercentile(nulls_i, 95)), "n": len(nulls_i),
                     "matched_share": float(np.mean([np.isfinite(v) for v in nulls_i]))}
    # 帰無 (ii) 札の入れ替え（更新の入力だけ）。入れ替えると ΣU ≈ 0 になり事後が μ から 0 へ縮むので、
    # 静的な効きの分 I_p·μ を足し戻し、「日の効き = 0・静的な効き = μ」のデータにそろえる（critic の 3）
    nulls_ii = []
    X = R[A_COLS].to_numpy(float)
    groups = list(R.groupby("rid").indices.values())
    mu_rc = np.nan_to_num(mu_r)
    for rep in range(N_NULL):
        Xp = X.copy()
        for ii in groups:
            Xp[ii] = X[rng.permutation(ii)]
        Up, Ip, _ = po.race_stats(Xp, y, pt, rid)
        Up = Up + np.einsum("rab,rb->ra", Ip, mu_rc)
        mean_p, _, _ = po.day_posterior(Rc, Up, Ip, phi, pri, TEST_LAG)
        s_p = scores(R, mean_p - mu_r, Xc)
        nulls_ii.append(po.offset_alpha(y[sel], pt[sel], s_p[sel], cl[sel])["alpha"])
    res["null_ii"] = {"mean": float(np.nanmean(nulls_ii)), "q95": float(np.nanpercentile(nulls_ii, 95)), "n": len(nulls_ii)}
    # ブートストラップ（開催の単位・α̂ のまわりの 1 段の推定）: α と、本物 − 帰無 (i) の平均（200 回すべて）
    G = gh(y[sel], pt[sel], s[sel], cl[sel], a_hat)
    Gi = [(an, g.reindex(G.index).fillna(0.0)) for an, g in gh_i if np.isfinite(an)]
    Gg, Gh = G["g"].to_numpy(), G["h"].to_numpy()
    Ng = np.stack([g["g"].to_numpy() for _, g in Gi]) if Gi else np.zeros((0, len(G)))
    Nh = np.stack([g["h"].to_numpy() for _, g in Gi]) if Gi else np.zeros((0, len(G)))
    Na = np.array([an for an, _ in Gi])
    a_b, d_b = [], []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(G), len(G)), minlength=len(G))
        a = a_hat + (Gg @ w) / max(Gh @ w, 1e-12)
        an = float(np.mean(Na + (Ng @ w) / np.maximum(Nh @ w, 1e-12))) if len(Na) else np.nan
        a_b.append(a)
        d_b.append(a - an)
    res["boot_alpha_ci"] = np.percentile(a_b, [2.5, 97.5]).tolist()
    res["boot_diff_i_ci"] = np.nanpercentile(d_b, [2.5, 97.5]).tolist()
    m = res["main"]
    lo = m["alpha"] - 1.96 * m["se"]
    res["pass"] = {
        "alpha_lo_gt_0": bool(lo > 0 and res["boot_alpha_ci"][0] > 0),
        "real_minus_null_i_lo_gt_0": bool(res["boot_diff_i_ci"][0] > 0),
        "gt_null_ii_q95": bool(m["alpha"] > res["null_ii"]["q95"]),
        "both_periods_pos": all(res[f"period_{nm}"]["alpha"] > 0 for nm in PERIODS),
    }
    res["pass_all"] = all(res["pass"].values())
    return res


def power_for_confirm(res70: dict, info30: float) -> dict:
    """30% を開ける前の検出力（rules-council: 70% の α を半分に割り引く）。"""
    m = res70["main"]
    se30 = m["se"] * np.sqrt(m["info"] / max(info30, 1e-12))
    a = m["alpha"] / 2.0
    pw = float(norm.cdf(a / se30 - 1.96)) if np.isfinite(se30) else 0.0
    return {"alpha_half": a, "se30": float(se30), "power": pw, "open_30": bool(pw >= 0.5)}


def prepare(R: pd.DataFrame, Rc: pd.DataFrame, rng) -> tuple:
    X = R[A_COLS].to_numpy(float)
    rid = R["rid"].to_numpy()
    U, I, Xc = po.race_stats(X, R["y"].to_numpy(), R["pt"].to_numpy(), rid)
    # φ: 探す側の 2016 年のレースで測る（燃やしの年）
    m16 = ((R["part"] == 0) & (R["year"] == 2016)).to_numpy()
    sub = R[m16]
    sid = pd.factorize(sub["rid"])[0]
    phi_sim = po.phi_check(sub[A_COLS].to_numpy(float), sub["pt"].to_numpy(), sid, sub["m"].to_numpy())
    r16 = np.unique(sub["rid"].to_numpy())
    phi_obs = po.phi_pearson(U[r16], I[r16])
    # 作り物（m を固定した分の縮み）と実データ（モデルのずれの過分散）の大きい方。φ < 1 で日の情報を過大に信じないため（critic の 8）
    # vec_v1 のペースの 2 次元（a_pace_h・a_pace_s）は 2016 年は全部 0（ペースの札は 2017 年以降だけ）。I が 0 の次元は φ の平均から外す
    phi = float(np.nanmax([np.nanmean(phi_sim), np.nanmean(phi_obs), 1.0]))
    if not np.isfinite(phi):
        raise RuntimeError("φ が有限でない")
    pri = po.priors(Rc, U, I, phi, range(2017, 2025))
    return U, I, Xc, phi, {"sim": phi_sim.tolist(), "obs": phi_obs.tolist()}, pri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "confirm"])
    ap.add_argument("--approval", default="", help="confirm のときの rules-council の許可の日付（見出し「day-pedigree の 30% を開ける許可（YYYY-MM-DD）」）")
    a = ap.parse_args()
    if a.mode == "confirm":
        # 30% の側は rules-council の 1 号の許可の後だけ（aptitude-group の段 1 の結論の後）。許可の見出しが decisions.md にあることを確かめる
        # 見出しは 1 号が「day-pedigree の 30% を開ける許可（YYYY-MM-DD）」の形で書く（rules-council 11/27）。この形だけに当てる
        import re
        pat = re.compile(r"^#+ day-pedigree の 30% を開ける許可（(\d{4}-\d{2}-\d{2})）")
        heads = [m.group(1) for ln in DECISIONS.read_text(encoding="utf-8").splitlines() if (m := pat.match(ln))]
        if a.approval not in heads:           # python -O でも消えないよう raise にする（critic の 15）
            raise SystemExit("decisions.md に「day-pedigree の 30% を開ける許可（--approval の日付）」の見出しが無い")
        if "DAYPED_NNULL" in os.environ or "DAYPED_NBOOT" in os.environ:     # 試しの回数で 30% の 1 回を使い切らない（critic の 1）
            raise SystemExit("confirm では試しの回数の環境変数（DAYPED_NNULL・DAYPED_NBOOT）を使わない")
    dd.open_token("select" if a.mode == "select" else "confirm", "check_v1",
                  f"段 2 の主の検定（A 族・T-60・{'探す 70%' if a.mode == 'select' else '確かめ 30%・' + a.approval}）")
    rng = np.random.default_rng(20261128)
    part = 0 if a.mode == "select" else 1
    R, Rc, n30 = build(part)
    U, I, Xc, phi, phi_f, pri = prepare(R, Rc, rng)
    res = {"eta_beta": R.attrs["eta_beta"], "phi": phi, "phi_by_feature": phi_f,
           "tau2": {f"{s}{y}": (None if v is None else v["tau2"].round(5).tolist()) for (s, y), v in pri.items()},
           "mu": {f"{s}{y}": (None if v is None else v["mu"].round(4).tolist()) for (s, y), v in pri.items()}}
    zero = {s: all(pri[(s, y)] is not None and (pri[(s, y)]["tau2"] <= 0).all() for y in range(2017, 2025)) for s in ("T", "D")}
    res["tau2_all_zero"] = zero
    res["eval"] = evaluate(R, Rc, part, U, I, Xc, phi, pri, rng)
    if all(zero.values()):           # τ² が全部 0 の族は不合格と数える（log.md）
        res["eval"]["pass_all"] = False
        res["eval"].setdefault("pass", {})["tau2_not_all_zero"] = False
    if a.mode == "select":
        # 30% の情報は、着順を使わずに見積もる: 70% の行あたりの情報 × 30% の判定の行の数（読み込みの直後に数えた）
        info30 = res["eval"]["main"].get("info", 0.0) * n30 / max(res["eval"]["rows"], 1)
        res["power_for_30"] = power_for_confirm(res["eval"], info30) if info30 > 0 else {"power": 0.0, "open_30": False}
    path = dd.OUT / (f"check_{a.mode}.json" if N_NULL == 200 and N_BOOT == 1000 else f"check_{a.mode}_trial.json")
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("tau2", "mu")}, ensure_ascii=False, indent=1, default=float))
    dd.finish_all(str(path))


if __name__ == "__main__":
    main()
