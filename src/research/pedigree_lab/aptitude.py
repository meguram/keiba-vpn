"""ユーザーの追加の依頼: 8 つの観点の適性（同じ型・開発の行・探す側・未確認・買い方に使わない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.aptitude <観点> [--boot 300]

型の名札（数を見る前に固定。log.md「8 観点の作り」）:
  early 早熟: A = 2 歳戦
  pace_h H ペース: A = 前半 3F − 後半 3F が、場 × 芝ダ × 距離の「前の年までのレース」の 3 分位で一番小さい（前が速い）
  pace_s スローの瞬発力: A = 同じ 3 分位で一番大きい（後半が速い）。1・2 の型はレースの後で分かる値なので記述だけ
  wet 道悪: A = 重・不良 ／ ext 延長: A = 前走から +200m 以上 ／ short 短縮: A = −200m 以下
  steep 急坂: A = 中山・阪神・中京 ／ straight 長い直線: A = 芝の東京・新潟・阪神 1600/1800/2400・京都 1800/2200/2400
  yoshiba 洋芝: A = 札幌・函館の芝 ／ dirt ダート向き: A = ダート ／ long 長い距離: A = 2000m 以上
祖先の適性（2 号の R1 の作り。洋芝の失敗の再発を防ぐ）:
  鍵（祖先 × 芝ダ）ごとに、群 A での子孫の −分位の as-of の和から「群 A の件数 × その祖先自身の全条件の素の平均」と
  「群 A の件数 × 全体での同じ差 Δ0」を引き、擬似件数 200 で 0 へ縮める（dirt は芝ダで分けない）。自分の走りは引く。
  父・母父は産駒、牝系 5 代は「母の側の牡の祖先（代 2〜5）」の子孫で作り、血量 2^−代 で重み付けて平均する。
  牝系は、登場率 50% 以上の祖先を外した版と、子孫を半分に割った 2 つの列の相関（信頼度）も出す。
判定: 群 A の行だけで、父・母父・牝系の 3 列（レースの中で中心化・群 A の行の sd で割る）と、父の全体の水準（sire_all_fpct）を制御に、
  lp に足した γ0（市場なし）と γ（市場あり）。開催の束ねの最大 |t| の同時の区間（観点の中の 3 本）。
持続: 父の「群 A の平均 − 全体の平均」を走りの期 2016〜19 と 2020〜24 で別々に作り、両期とも群 A に 100 走以上の父で相関。
  帰無は父の札を生年の中で入れ替え（20 回）。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.explore2 import fit_boot
from src.research.pedigree_lab.explore5 import attach, long_straight, sim_ci

K = 200.0
MAT_POS = {2: [2], 3: [4, 6], 4: [8, 10, 12, 14], 5: list(range(16, 32, 2))}     # 母の側の牡の祖先（代: 位置）
TABLES = "/home/jovyan/work/keiba-vpn/data/page_reference/tables"


def base_frame() -> pd.DataFrame:
    cols = ["race_id", "horse_id", "meeting", "date", "win", "fpct", "odds", "horse_number", "sire", "ds", "surf", "venue",
            "distance", "going", "rage", "dchg", "split", "sire_all_fpct"]
    F = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=cols)
    F = F[F["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)
    F["by"] = F["horse_id"].str[:4]
    return F


def pace_labels(F: pd.DataFrame) -> np.ndarray:
    """前半 3F − 後半 3F（秒）を、場 × 芝ダ × 距離ごとに「前の年まで」のレースの 3 分位で 0・1・2 に分ける。
    前の年までが 30 レース未満の鍵は、その年の中の分位で代える（2016 年はすべてこれ。記録に書く）。"""
    fr = []
    for y in range(2016, 2025):
        fr.append(pd.read_parquet(f"{TABLES}/{y}/race_result_lap_race.parquet", columns=["race_id", "pace"]))
    L = pd.concat(fr, ignore_index=True).drop_duplicates("race_id")
    pj = L["pace"].map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    L["diff"] = pj.map(lambda d: (d.get("first_half_3f") or np.nan) - (d.get("second_half_3f") or np.nan))
    R = F.drop_duplicates("race_id")[["race_id", "date", "venue", "surf", "distance"]].merge(L[["race_id", "diff"]], on="race_id", how="left")
    R = R.dropna(subset=["diff"])
    R["y"] = R["date"].dt.year
    key = ["venue", "surf", "distance"]
    out = []
    for y in sorted(R["y"].unique()):
        cur = R[R["y"] == y].copy()
        prior = R[R["y"] < y]
        q = prior.groupby(key)["diff"].agg(n="size", q1=lambda s: s.quantile(1 / 3), q2=lambda s: s.quantile(2 / 3)).reset_index()
        q = q[q["n"] >= 30]
        own = cur.groupby(key)["diff"].agg(no="size", q1o=lambda s: s.quantile(1 / 3), q2o=lambda s: s.quantile(2 / 3)).reset_index()
        own = own[own["no"] >= 9]                      # その年の中で代えるのは 9 レース以上の鍵だけ（critic の 7）
        cur = cur.merge(q, on=key, how="left").merge(own, on=key, how="left")
        q1 = cur["q1"].fillna(cur["q1o"]).to_numpy()
        q2 = cur["q2"].fillna(cur["q2o"]).to_numpy()
        d = cur["diff"].to_numpy()
        cur["t"] = np.where(np.isnan(q1), np.nan, np.where(d <= q1, 0, np.where(d >= q2, 2, 1)))
        out.append(cur[["race_id", "t"]])
    T = pd.concat(out).set_index("race_id")["t"]
    return F["race_id"].map(T).to_numpy(float)


def group_a(F: pd.DataFrame, what: str) -> tuple[np.ndarray, list[str]]:
    v, d, s = F["venue"].to_numpy(), F["distance"].to_numpy(), F["surf"].to_numpy()
    turf = s == "T"
    if what == "rand":                                   # 偽の型（陰性の対照）: レースの id の hash で 3 分の 1 を選ぶ
        return (pd.util.hash_pandas_object(F["race_id"], index=False).to_numpy() % 3) == 0, ["surf"]
    if what == "early":
        return F["rage"].eq("A2").to_numpy(), ["surf"]
    if what in ("pace_h", "pace_s"):
        t = pace_labels(F)
        return t == (0 if what == "pace_h" else 2), ["surf"]
    if what == "wet":
        return F["going"].eq("W").to_numpy(), ["surf"]
    if what == "ext":
        return F["dchg"].eq("U").to_numpy(), ["surf"]
    if what == "short":
        return F["dchg"].eq("D").to_numpy(), ["surf"]
    if what == "steep":
        return np.isin(v, ["中山", "阪神", "中京"]), ["surf"]
    if what == "straight":
        return turf & np.array([long_straight(a, b) for a, b in zip(v, d)]), ["surf"]
    if what == "yoshiba":
        return turf & np.isin(v, ["札幌", "函館"]), ["surf"]
    if what == "dirt":
        return s == "D", []
    if what == "long":
        return d >= 2000, ["surf"]
    raise ValueError(what)


def base_means(F: pd.DataFrame, A: np.ndarray, keys: list[str], w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """全体（馬の行）での as-of の「群 A の平均」と「全条件の平均」（鍵に芝ダがあれば芝ダごと）。"""
    w = np.ones(len(F)) if w is None else w
    A = A.astype(float)
    g = -F["fpct"].to_numpy()
    E = F.assign(_a=g * A * w, _na=A * w, _t=g * w, _nt=w.astype(float), _k="1")
    b, _ = asof(E, ["_k"] + [k for k in keys if k == "surf"], ["_a", "_na", "_t", "_nt"])
    return b[:, 0] / np.maximum(b[:, 1], 1), b[:, 2] / np.maximum(b[:, 3], 1)


def apt_by(D: pd.DataFrame, A: np.ndarray, keys: list[str], w: np.ndarray | None = None,
           base: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """鍵の「群 A の as-of の平均（全条件の平均へ縮める）− 全条件の平均」− 全体での同じ差。w は集計に入れる行の重み（半分割り用）。
    base は D の行ごとの全体の平均（牝系の長い表では馬の行で作って渡す。祖先の数だけ重ならないように）。"""
    w = np.ones(len(D)) if w is None else w
    A = A.astype(float)
    g = -D["fpct"].to_numpy()
    E = D.assign(_a=g * A * w, _na=A * w, _t=g * w, _nt=w.astype(float))
    s, _ = asof(E, keys, ["_a", "_na", "_t", "_nt"])
    mA0, mT0 = base_means(D, A, keys, w) if base is None else base
    # 群 A のずれ（s_a − n_A・素の全条件の平均）を、全体のずれ Δ0 を引いてから擬似件数 K で 0 へ縮める。
    # 前の作り（mA を縮めた mT へ寄せてから mT を引く）は、祖先の全条件の水準（家系の質）が混ざる（critic 2 回目の 1）。
    rawT = s[:, 2] / np.maximum(s[:, 3], 1)
    return (s[:, 0] - s[:, 1] * rawT - s[:, 1] * (mA0 - mT0)) / (s[:, 1] + K)


def maternal_long(F: pd.DataFrame, drop_common: bool, null_seed: int | None = None, min_gen: int = 2,
                  ped: pd.DataFrame | None = None) -> pd.DataFrame:
    ped = (data.load_ped() if ped is None else ped)[["horse_id", "anc5"]]
    want = {(g, p) for g, ps in MAT_POS.items() for p in ps}
    rows = []
    for h, a in zip(ped["horse_id"], ped["anc5"].fillna("")):
        for t in a.split(";"):
            if t:
                g, p, i = t.split(":")
                if (int(g), int(p)) in want:
                    rows.append((h, int(g), i))
    Anc = pd.DataFrame(rows, columns=["horse_id", "gen", "anc"])
    Anc = Anc[Anc["gen"] >= min_gen]                    # 代で絞るのは、まとめる前（critic 2 回目の 6）
    Anc["bw"] = 0.5 ** Anc["gen"]
    # 牝系に何度も出る祖先は、血量を足し合わせて 1 行にまとめる（インブリードの分を数える。critic の 5）
    Anc = Anc.groupby(["horse_id", "anc"], as_index=False).agg(gen=("gen", "min"), bw=("bw", "sum"))
    if null_seed is not None:
        # 帰無: 祖先の札を、行（馬 × 祖先）の単位で、登場の数の十分位の同じ帯の中で入れ替える。
        # 祖先の名前の付け替え（1 対 1）では子孫の集まりが変わらず帰無にならない（11/26 の 1 回目の誤り）。
        # 帯は祖先の単位の十分位（登場の数の順）で作り、行に配る（行の順位で切ると上の帯が数頭の祖先で埋まる。critic 3 回目の 2）
        cnt = Anc.groupby("anc")["horse_id"].nunique()
        aband = pd.qcut(cnt.rank(method="first"), 10, labels=False)
        band = Anc["anc"].map(aband).to_numpy()
        rng = np.random.default_rng(null_seed)
        v = Anc["anc"].to_numpy().copy()
        for b in np.unique(band):
            ii = np.where(band == b)[0]
            v[ii] = v[ii][rng.permutation(len(ii))]
        Anc["anc"] = v
        Anc = Anc.groupby(["horse_id", "anc"], as_index=False).agg(gen=("gen", "min"), bw=("bw", "sum"))
    if drop_common:
        share = Anc.groupby("anc")["horse_id"].nunique() / ped["horse_id"].nunique()
        Anc = Anc[~Anc["anc"].isin(share[share >= 0.5].index)]
    R = F[["date", "horse_id", "surf", "fpct"]].reset_index().rename(columns={"index": "row"})
    return R.merge(Anc, on="horse_id", how="inner").sort_values(["date", "row"], kind="stable").reset_index(drop=True)


def maternal(F: pd.DataFrame, A: np.ndarray, keys: list[str], L: pd.DataFrame, half: int | None = None) -> np.ndarray:
    """half = 0/1 なら、子孫のうち馬の id の hash の偶奇がそれの馬だけを集計に入れる（信頼度の半分割り）。"""
    row = L["row"].to_numpy()
    if half is None:
        wF = None
    else:
        wF = ((pd.util.hash_pandas_object(F["horse_id"], index=False).to_numpy() % 2) == half).astype(float)
    mA0, mT0 = base_means(F, A, keys, wF)
    d = apt_by(L, A[row], ["anc"] + keys, None if wF is None else wF[row], base=(mA0[row], mT0[row]))
    wt = L["bw"].to_numpy()
    agg = pd.DataFrame({"row": L["row"], "wd": wt * d, "w": wt}).groupby("row")[["wd", "w"]].sum()
    return (agg["wd"] / agg["w"]).reindex(np.arange(len(F))).fillna(0.0).to_numpy()


def persistence(F: pd.DataFrame, A: np.ndarray, n_null: int = 20) -> dict:
    g = -F["fpct"].to_numpy()
    per = np.where(F["date"].dt.year <= 2019, 1, 2)

    def table(sire):
        T = pd.DataFrame({"sire": sire, "per": per, "g": g, "A": A})
        a = T[T["A"]].groupby(["sire", "per"])["g"].agg(["mean", "size"])
        t = T.groupby(["sire", "per"])["g"].mean().rename("all")
        J = a.join(t, how="inner")
        J = J[J["size"] >= 100]
        return (J["mean"] - J["all"]).unstack("per").dropna()
    W = table(F["sire"].to_numpy())
    if len(W) < 10 or W.shape[1] < 2:
        return {"sires": int(len(W))}
    r = float(W[1].corr(W[2]))
    H = F.drop_duplicates("horse_id")[["horse_id", "sire", "by"]].reset_index(drop=True)
    nulls = []
    for k in range(n_null):
        rng = np.random.default_rng(700 + k)
        perm = np.arange(len(H))
        for ii in H.groupby("by").indices.values():
            perm[ii] = rng.permutation(ii)
        fake = dict(zip(H["horse_id"], H["sire"].to_numpy()[perm]))
        Wn = table(F["horse_id"].map(fake).to_numpy())
        if len(Wn) >= 10 and Wn.shape[1] == 2:
            nulls.append(float(Wn[1].corr(Wn[2])))
    if not nulls:
        return {"sires": int(len(W)), "corr": round(r, 3), "null_runs": 0}
    z, se = np.arctanh(r), 1 / np.sqrt(max(len(W) - 3, 1))
    return {"sires": int(len(W)), "corr": round(r, 3), "fisher_ci": [round(float(np.tanh(z - 1.96 * se)), 3), round(float(np.tanh(z + 1.96 * se)), 3)],
            "null_runs": len(nulls), "null_mean": round(float(np.mean(nulls)), 3), "null_q95": round(float(np.percentile(nulls, 95)), 3)}


def zA(S: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    rid = S["race_id"].to_numpy()
    v = raw - pd.Series(raw).groupby(rid).transform("mean").to_numpy()
    sd = v.std()
    return v / (sd if sd > 0 else 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--null-seed", type=int, default=None, help="牝系の祖先の札を登場の数の帯の中で入れ替える帰無")
    a = ap.parse_args()
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"aptitude_{a.what}", f"8 観点の適性: {a.what}（尤度だけ・記述）"))
    F = base_frame()
    A, keys = group_a(F, a.what)
    L = maternal_long(F, drop_common=False, null_seed=a.null_seed)
    Lc = maternal_long(F, drop_common=True, null_seed=a.null_seed)
    cols = {
        "sire": apt_by(F, A, ["sire"] + keys),
        "bms": apt_by(F, A, ["ds"] + keys),
        "mat5": maternal(F, A, keys, L),
        "mat5_nocommon": maternal(F, A, keys, Lc),
        "mat5_h0": maternal(F, A, keys, L, half=0),
        "mat5_h1": maternal(F, A, keys, L, half=1),
    }
    for k, v in cols.items():
        F[f"z_{k}"] = v
    F["_A"] = A
    X = attach(F)
    S = X[X["_A"].astype(bool)].reset_index(drop=True)
    res = {"what": a.what, "rows_A_all": int(A.sum()), "rows_all": int(len(A)), "dev_rows_A": int(len(S)),
           "sd_on_A": {k: round(float(np.std(S[f"z_{k}"])), 4) for k in ("sire", "bms", "mat5")},
           "mat_reliability_half": round(float(np.corrcoef(S["z_mat5_h0"], S["z_mat5_h1"])[0, 1]), 3)}
    rh = res["mat_reliability_half"]
    res["mat_reliability_sb"] = round(2 * rh / (1 + rh), 3) if rh > -1 else None     # Spearman-Brown の補正（critic の 6）
    names = ["sire", "bms", "mat5"]
    Z = np.c_[tuple(zA(S, S[f"z_{n}"].to_numpy()) for n in names)]
    ctrl = zA(S, S["sire_all_fpct"].fillna(0).to_numpy())
    for market in (False, True):
        base = [S["lp"].to_numpy(), S["lq"].to_numpy()] if market else [S["lp"].to_numpy()]
        M = np.c_[tuple(base) + (ctrl, Z)]
        idx = {n: len(base) + 1 + i for i, n in enumerate(names)}
        b, bs = fit_boot(S, M, a.boot)
        c, r = sim_ci(b, bs, idx)
        res["with_market" if market else "no_market"] = {"c_maxT": round(c, 3), **r}
        print(a.what, "market" if market else "no_market", json.dumps(r, ensure_ascii=False), flush=True)
    b, bs = fit_boot(S, np.c_[S["lp"], ctrl, zA(S, S["z_mat5_nocommon"].to_numpy())], max(a.boot // 2, 100))
    res["mat5_nocommon_no_market"] = {"gamma": round(float(b[2]), 4), "ci": np.percentile(bs[:, 2], [2.5, 97.5]).round(4).tolist()}
    res["persistence_sire"] = persistence(F, A)
    path = data.OUT / (f"aptitude_{a.what}.json" if a.null_seed is None else f"aptitude_{a.what}_null{a.null_seed}.json")
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
