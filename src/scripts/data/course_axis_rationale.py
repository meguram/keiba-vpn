"""コース別の走法軸の優位を、実力・人気から切り離して測る（読むだけ・JSON を書く）。

背景（2026-11-11 ユーザー依頼）: 「各コースでこの走法軸が優位」と言うなら、人気や実力といった適性と薄い要素を除いたうえで根拠を言語化したい。
人気を結果の補正にも判定にも使わない。実力は『その馬の出走前の自分の成績』（市場を見ない）で代理する。
  - ability = 各馬の、そのレースより前の走の得点（rmc.score_of）の平均を全体平均へ縮小（K=3）し、レース内で中心化した値。
  - raw  : 得点 − 頭数だけの期待（現行の寄与率と同じ）。補正なし。
  - res  : raw の結果と各軸を、それぞれ ability へ回帰した残差にしたもの（芝ダ別）。本命。
  - pop  : 参考。結果を人気込みの期待で補正（人気は診断のみ・判定には使わない）。
  各モードで、芝ダ×クラスの平均を引いたコース固有の差を 2024-25 年（fit）、2024 年、2025 年、2026 年（テストセット②・確認のみ）に分けて推定。
  res_pts: res の結果を、ポイント制（1〜3着=3点・4〜5着=1点・6着以下=0点。ユーザー指定）に替えた頑健性の確認。期待は 11/頭数（5頭立て以上）。
  対照として、レース内で得点を入れ替えた乱数ラベルでも res を 5 回測る。
  また、コース×軸ごとに、軸を芝ダ別の十分位で 1〜10 点にした得点帯ごとの出走数・3着以内の数・ポイントのヒストグラム用の度数と、全出走馬の得点分布と好走馬の得点分布の差（好走馬の平均得点 − 全出走馬の平均得点 = Δ）を出す（raw と res の 2 通り）。
使い方（ホストで）: cd <repo> && nice -n 19 /opt/conda/bin/python -I <このファイル> OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from src.research.race import running_map as rm  # noqa: E402
from src.research.race import running_map_course as rmc  # noqa: E402
from src.scripts.data import course_axis_demand as cad  # noqa: E402

KEYS = cad.KEYS
SHRINK_K = 3.0
MIN_YEAR = 120          # 年別に推定するコース×年の最低走数
N_SHUFFLE = 5
N_ZONE = 10   # 得点帯（芝ダ別の十分位。10 = 上位 1 割、1 = 下位 1 割）


def prior_ability(metrics: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """各走について『それより前の走の得点の平均（縮小）』を返す。そのレースの結果は使わない。レース内で中心化した値 ab_c も付ける。"""
    m = metrics[["race_id", "horse_id", "date", "finish_position"]].drop_duplicates(["race_id", "horse_id"]).copy()
    m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
    fp = pd.to_numeric(m["finish_position"], errors="coerce")
    s = np.full(len(m), np.nan)
    ok = fp.notna().to_numpy()
    s[ok] = rmc.score_of(fp.to_numpy()[ok])
    m["s"] = s
    m = m.sort_values(["horse_id", "date", "race_id"], kind="mergesort").reset_index(drop=True)
    s0 = m["s"].fillna(0.0)
    hit = m["s"].notna().astype(float)
    csum = s0.groupby(m["horse_id"]).cumsum() - s0
    cn = hit.groupby(m["horse_id"]).cumsum() - hit
    mu = float(m["s"].mean())
    m["n_prior"] = cn
    m["ab"] = (csum + SHRINK_K * mu) / (cn + SHRINK_K)
    m["ab_c"] = m["ab"] - m.groupby("race_id")["ab"].transform("mean")
    return m[["race_id", "horse_id", "n_prior", "ab_c"]], mu


def dev2(Z: np.ndarray, r: np.ndarray, race: np.ndarray, base: np.ndarray):
    """コース固有の差（cad.dev_fit と同じ。丸めない）。失敗時は None。"""
    adj = r - base[:, 0] - (Z * base[:, 1:]).sum(1)
    X = np.column_stack([np.ones(len(Z)), Z])
    try:
        b = np.linalg.lstsq(X, adj, rcond=None)[0]
        u = adj - X @ b
        codes = pd.factorize(race)[0]
        S = np.zeros((X.shape[1], codes.max() + 1))
        for j in range(X.shape[1]):
            S[j] = np.bincount(codes, weights=X[:, j] * u)
        bread = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(bread @ (S @ S.T) @ bread))[1:]
    except np.linalg.LinAlgError:
        return None
    return b[1:], se


def base_all(Zs: np.ndarray, r: np.ndarray, surf: np.ndarray, fit: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """芝ダ × クラスの回帰を fit の走だけで当て、同じ芝ダの全ての走へ割り当てる（行 = 走、列 = 切片 + 軸）。"""
    base = np.zeros((len(Zs), Zs.shape[1] + 1))
    X1 = lambda m: np.column_stack([np.ones(int(m.sum())), Zs[m]])
    mf = surf & fit
    base[surf] = np.linalg.lstsq(X1(mf), r[mf], rcond=None)[0]
    for c, _ in cad.CLASS_BANDS:
        mc = surf & (cls == c)
        mfc = mc & fit
        if mfc.sum() >= cad.MIN_CELL:
            base[mc] = np.linalg.lstsq(X1(mfc), r[mfc], rcond=None)[0]
    return base


def course_stats(Zm: np.ndarray, r: np.ndarray, base: np.ndarray, race: np.ndarray, periods: dict, masks: dict) -> dict:
    out: dict = {}
    for ck, mc in masks.items():
        row: dict = {}
        for name, my in periods.items():
            m = mc & my
            if m.sum() < (cad.MIN_CELL if name == "fit" else MIN_YEAR):
                row[name] = None
                continue
            res = dev2(Zm[m], r[m], race[m], base[m])
            row[name] = None if res is None else {"n": int(m.sum()), "races": int(pd.factorize(race[m])[0].max() + 1),
                                                   "b": [round(float(v), 5) for v in res[0]], "se": [round(float(v), 5) for v in res[1]]}
        out[ck] = row
    return out


def points_of(fp: np.ndarray) -> np.ndarray:
    """ポイント制: 1〜3着 3点、4〜5着 1点、6着以下 0点（着順が数値でない走は 0）。"""
    fp = np.nan_to_num(fp, nan=99.0)
    return np.where(fp <= 3, 3.0, np.where(fp <= 5, 1.0, 0.0))


def zone_scores(Z: np.ndarray, surfs: dict) -> np.ndarray:
    """各軸を、芝ダ別の十分位で 1〜10 点にする（10 点 = その軸が上位 1 割）。"""
    S = np.zeros(Z.shape, dtype=int)
    q = np.linspace(0, 1, N_ZONE + 1)[1:-1]
    for ms in surfs.values():
        for j in range(Z.shape[1]):
            S[ms, j] = np.digitize(Z[ms, j], np.quantile(Z[ms, j], q)) + 1
    return S


def delta_ref(S: np.ndarray, y: np.ndarray, cls: np.ndarray, surfs: dict, periods: dict) -> dict:
    """基準: 同じ芝ダ × 同じクラスの全体での、好走馬(重み y)の平均得点と全出走馬の平均得点。キー (期間, 芝ダ, クラス) → (好走馬, 全体)、軸ごとの値。"""
    out = {}
    for pn, my in periods.items():
        for sn, ms in surfs.items():
            for c in np.unique(cls[ms]):
                m = ms & my & (cls == c)
                if m.sum() < 30 or y[m].sum() <= 0:
                    continue
                out[(pn, sn, c)] = ((S[m] * y[m, None]).sum(0) / y[m].sum(), S[m].mean(0))
    return out


def hist_counts(S: np.ndarray, good: np.ndarray, pts: np.ndarray, mc: np.ndarray, race: np.ndarray, periods: dict, cls: np.ndarray, surf: str, ref: dict) -> dict:
    """得点帯ごとの度数（全出走馬・好走馬・ポイント）と、好走馬の平均得点が全出走馬の平均得点よりどれだけ高いか（Δ。レース単位でまとめた標準誤差つき）。ref は、同じ芝ダ・同じクラス構成の全体で期待されるΔ（コース固有でない分）。"""
    out = {}
    codes = pd.factorize(race[mc])[0]
    for j, k in enumerate(KEYS):
        sc = S[mc, j]
        z = sc - 1
        h = {"all": np.bincount(z, minlength=N_ZONE).tolist(),
             "good": np.bincount(z, weights=good[mc], minlength=N_ZONE).astype(int).tolist(),
             "pts": np.bincount(z, weights=pts[mc], minlength=N_ZONE).astype(int).tolist()}
        for kind, y in (("g", good[mc]), ("p", pts[mc])):
            rf = ref[kind]
            for pn, my in periods.items():
                m = my[mc]
                if m.sum() < MIN_YEAR or y[m].sum() <= 0:
                    h[f"d{kind}_{pn}"] = None
                    continue
                s_, y_, c_ = sc[m], y[m], codes[m]
                Y, N = float(y_.sum()), float(m.sum())
                my_, ma_ = float((y_ * s_).sum() / Y), float(s_.mean())
                u = np.bincount(c_, weights=y_ * (s_ - my_) / Y - (s_ - ma_) / N)
                cc = cls[mc][m]
                yc = y_
                cs = [c for c in np.unique(cc) if (pn, surf, c) in rf]
                rj = sum(float(yc[cc == c].sum() / Y) * float(rf[(pn, surf, c)][0][j]) - float((cc == c).mean()) * float(rf[(pn, surf, c)][1][j]) for c in cs)
                h[f"d{kind}_{pn}"] = {"d": round(my_ - ma_, 4), "se": round(float(np.sqrt((u * u).sum())), 4), "base": round(ma_, 3), "ref": round(rj, 4)}
        out[k] = h
    return out


def main() -> None:
    t, Z = cad.load_table()
    d = rm._data()
    ab, mu = prior_ability(d["metrics"])
    t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    t = t.merge(ab, on=["race_id", "horse_id"], how="left")
    cover = float((t["n_prior"].fillna(0) > 0).mean())
    a = t["ab_c"].fillna(0.0).to_numpy()
    yr = t["date"].dt.year.to_numpy()
    fit = yr <= 2025
    periods = {"fit": fit, "y24": yr == 2024, "y25": yr == 2025, "y26": yr >= 2026}
    race = t["race_id"].to_numpy()
    cls = t["cls"].to_numpy()
    y, e_n, e_pop = t["y"].to_numpy(), t["e_n"].to_numpy(), t["e_pop"].to_numpy()
    r_raw, r_pop = y - e_n, y - e_pop
    fpos = pd.to_numeric(t["finish_position"], errors="coerce").to_numpy()
    good = (fpos <= 3).astype(float)
    pts = points_of(fpos)
    ex_pts = 11.0 / t["n_field"].to_numpy()
    r_pts = pts - ex_pts

    Zraw, Zres = np.zeros_like(Z), np.zeros_like(Z)
    r_res = np.zeros_like(r_raw)
    r_res_pts = np.zeros_like(r_raw)
    diag = {"axis_r2_by_ability": {}, "axis_corr_ability": {}, "ability_vs_pop_corr": {}, "ability_vs_outcome_corr": {}}
    for surf in ("芝", "ダート"):
        ms = (t["surface"] == surf).to_numpy()
        mf = ms & fit
        mu_z, sd_z = Z[mf].mean(0), Z[mf].std(0)
        Zraw[ms] = (Z[ms] - mu_z) / sd_z
        va = a[mf].var()
        sl = ((Zraw[mf] - Zraw[mf].mean(0)) * (a[mf] - a[mf].mean())[:, None]).mean(0) / va
        Zr = Zraw[ms] - sl * a[ms][:, None]
        Zres[ms] = Zr / Zr[fit[ms]].std(0)
        slr = ((r_raw[mf] - r_raw[mf].mean()) * (a[mf] - a[mf].mean())).mean() / va
        r_res[ms] = r_raw[ms] - slr * a[ms]
        slp = ((r_pts[mf] - r_pts[mf].mean()) * (a[mf] - a[mf].mean())).mean() / va
        r_res_pts[ms] = r_pts[ms] - slp * a[ms]
        diag["axis_corr_ability"][surf] = [round(float(np.corrcoef(Zraw[mf][:, j], a[mf])[0, 1]), 3) for j in range(len(KEYS))]
        diag["axis_r2_by_ability"][surf] = [round(c * c, 3) for c in diag["axis_corr_ability"][surf]]
        diag["ability_vs_pop_corr"][surf] = round(float(np.corrcoef(a[mf], (e_pop - e_n)[mf])[0, 1]), 3)
        diag["ability_vs_outcome_corr"][surf] = round(float(np.corrcoef(a[mf], r_raw[mf])[0, 1]), 3)

    ckeys = sorted(t["ckey"].unique())
    masks = {ck: (t["ckey"] == ck).to_numpy() for ck in ckeys if (t["ckey"] == ck).sum() >= cad.MIN_CELL}
    surfs = {s: (t["surface"] == s).to_numpy() for s in ("芝", "ダート")}

    def run_mode(Zm: np.ndarray, r: np.ndarray) -> dict:
        base = np.zeros((len(r), Zm.shape[1] + 1))
        for s, ms in surfs.items():
            base[ms] = base_all(Zm, r, ms, fit, cls)[ms]
        return course_stats(Zm, r, base, race, periods, masks)

    modes = {"raw": run_mode(Zraw, r_raw), "pop": run_mode(Zraw, r_pop), "res": run_mode(Zres, r_res), "res_pts": run_mode(Zres, r_res_pts)}
    print("modes done", flush=True)

    rng = np.random.default_rng(20261111)
    codes = pd.factorize(race)[0]
    order = np.argsort(codes, kind="mergesort")
    null = []
    for k in range(N_SHUFFLE):
        ys = y.copy()
        for lo, hi in zip(np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1], np.r_[np.flatnonzero(np.diff(codes[order])) + 1, len(order)]):
            idx = order[lo:hi]
            ys[idx] = y[rng.permutation(idx)]
        rs = ys - e_n
        rsr = np.zeros_like(rs)
        for s, ms in surfs.items():
            mf = ms & fit
            sl = ((rs[mf] - rs[mf].mean()) * (a[mf] - a[mf].mean())).mean() / a[mf].var()
            rsr[ms] = rs[ms] - sl * a[ms]
        null.append(run_mode(Zres, rsr))
        print("null", k, flush=True)

    pz = {"all": np.ones(len(t), dtype=bool), "fit": fit, "y26": yr >= 2026}
    Sraw, Sres = zone_scores(Zraw, surfs), zone_scores(Zres, surfs)
    ref_raw = {"g": delta_ref(Sraw, good, cls, surfs, pz), "p": delta_ref(Sraw, pts, cls, surfs, pz)}
    ref_res = {"g": delta_ref(Sres, good, cls, surfs, pz), "p": delta_ref(Sres, pts, cls, surfs, pz)}
    sf = lambda ck: ck.split("|")[1]
    hist = {ck: hist_counts(Sraw, good, pts, mc, race, pz, cls, sf(ck), ref_raw) for ck, mc in masks.items()}
    histr = {ck: hist_counts(Sres, good, pts, mc, race, pz, cls, sf(ck), ref_res) for ck, mc in masks.items()}
    res = {"axes": KEYS, "modes": modes, "null": null, "hist": hist, "histr": histr, "diag": diag,
           "meta": {"since": rmc.SINCE, "n_runs": int(len(t)), "n_races": int(t["race_id"].nunique()), "n_fit": int(fit.sum()), "n_2026": int((yr >= 2026).sum()),
                    "shrink_k": SHRINK_K, "mean_score": round(mu, 4), "history_coverage": round(cover, 3), "n_courses": len(masks),
                    "min_cell": cad.MIN_CELL, "min_year": MIN_YEAR, "n_shuffle": N_SHUFFLE,
                    "class_counts": {k: int(v) for k, v in t["cls"].value_counts().items() if k}}}
    Path(sys.argv[1]).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("ok", res["meta"], diag)


if __name__ == "__main__":
    main()
