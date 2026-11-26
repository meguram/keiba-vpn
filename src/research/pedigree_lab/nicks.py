"""ユーザーの切り口 A: ニックスを 3 つの物差しで作り直す（log.md「ユーザーの新しい切り口 A」。開発の行・探す側・未確認）。

- 馬の単位（1 頭 1 票・完結した成績だけ・2014〜19 年生まれ）:
  winup = 3 歳の 9 月末までに 1 勝した・maxcls = 4 歳の年末までに到達したクラス（勝ち数 0〜4、重賞の勝ちで 5）・
  prize = log(1 + 4 歳の年末までの近似の賞金の合計)（クラス × 着順の目安。手元に賞金の列が無いため）。
- 組み合わせの効き = 組み合わせの平均 − 父と母父の足し算の見込み（winup はロジットの尺度）。頭数 10 で 0 へ縮める。
  母の質を除く版: 各馬の値から「同じ母の、父の違う子」の平均を引いた値で同じことをする（同じ父の子は引かない）。
- 単位: id（父 × 母父）と系統（父父 × 母父の父）。
- 持続: 世代 2014〜16 と 2017〜19 の 2 期で、両期とも 10 頭以上の組み合わせの効きの相関（馬が重ならないので漏れが無い）。
- 帰無: 同じ父・同じ世代の中で母父の札を入れ替える（20 回）。
- 市場: 2021〜24 年の開発の行のうち 2017 年以降の生まれの馬に、第 1 期の効きを当て、Benter（lp・lq）に足した係数。
- よく知られたニックス: ディープインパクト × 母父がストームキャットかその息子。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.nicks
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.explore2 import fit_boot
from src.research.pedigree_lab.explore5 import attach, frame, zcol

FIRST = {"NEW": 700, "MDN": 550, "C1": 780, "C2": 1060, "C3": 1550, "OP": 2200}
GRADE_FIRST = {"L": 2400, "G3": 4000, "G2": 6000, "G1": 13000}
SHARE = {1: 1.0, 2: 0.4, 3: 0.25, 4: 0.15, 5: 0.10}
K = 10.0
DEEP, STORMCAT = "2002100816", "000a001a98"


def horse_table() -> pd.DataFrame:
    r = data.load_dev(reason="ニックスの 3 つの物差し（馬の単位の完結した成績）", version="nicks")
    ped = data.load_ped()
    ped["bms_sire"] = ped["anc5"].fillna("").str.extract(r"(?:^|;)3:4:([^;]+)")[0].fillna("")
    r["by"] = pd.to_numeric(r["horse_id"].str[:4], errors="coerce")
    r = r[(r["by"] >= 2014) & (r["by"] <= 2019)].copy()
    cls = r["grade"].map({"新馬": "NEW", "未勝利": "MDN", "1勝": "C1", "2勝": "C2", "3勝": "C3"}).fillna("OP")
    first = np.where(r["grade"].isin(list(GRADE_FIRST)), r["grade"].map(GRADE_FIRST), cls.map(FIRST))
    r["prize"] = first * r["finish_position"].map(SHARE).fillna(0)
    end3 = pd.to_datetime((r["by"] + 3).astype(int).astype(str) + "-09-30")
    end4 = pd.to_datetime((r["by"] + 4).astype(int).astype(str) + "-12-31")
    a = r[r["date"] <= end3]
    b = r[r["date"] <= end4]
    H = pd.DataFrame({"winup": a.groupby("horse_id")["win"].max()})
    wins = b.groupby("horse_id")["win"].sum()
    gwin = b[b["grade"].isin(["G1", "G2", "G3"]) & (b["win"] == 1)].groupby("horse_id").size()
    H = H.join(pd.DataFrame({"maxcls": np.minimum(wins, 4).astype(float), "prize": np.log1p(b.groupby("horse_id")["prize"].sum())}), how="outer")
    H.loc[H.index.isin(gwin.index), "maxcls"] = 5.0
    H["winup"] = H["winup"].fillna(0).astype(float)
    H = H.reset_index().merge(ped[["horse_id", "sire", "ds", "dam", "ss", "bms_sire"]], on="horse_id", how="left")
    H["by"] = pd.to_numeric(H["horse_id"].str[:4])
    H["per"] = np.where(H["by"] <= 2016, 1, 2)
    ok = (H["sire"].fillna("") != "") & (H["ds"].fillna("") != "") & (H["dam"].fillna("") != "")
    H["ss"] = H["ss"].fillna("").replace("", np.nan)                # 空の系統を 1 つの群にしない（critic の 9）
    H["bms_sire"] = H["bms_sire"].replace("", np.nan)
    return H[ok].reset_index(drop=True)


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def dam_adjust(H: pd.DataFrame, m: str) -> pd.Series:
    """同じ母の、父の違う子の平均を引く（母の質を除く）。同じ母に父の違う子がいない馬は NaN。"""
    # 兄弟は同じ期の中だけで取る（期をまたぐ共通の成分を作らない。critic の 2）。件数は欠けを数えない（critic の 8）
    g = H.groupby(["per", "dam"])[m].agg(["sum", "count"])
    gs = H.groupby(["per", "dam", "sire"])[m].agg(["sum", "count"])
    tot = g.reindex(pd.MultiIndex.from_arrays([H["per"], H["dam"]])).to_numpy()
    same = gs.reindex(pd.MultiIndex.from_arrays([H["per"], H["dam"], H["sire"]])).to_numpy()
    s_other, n_other = tot[:, 0] - same[:, 0], tot[:, 1] - same[:, 1]
    return pd.Series(np.where(n_other > 0, H[m].to_numpy() - s_other / np.maximum(n_other, 1), np.nan), index=H.index)


def effects(H: pd.DataFrame, m: str, a_key: str, b_key: str) -> pd.DataFrame:
    """組み合わせの効き（足し算の見込みからの差・頭数 K で 0 へ縮める）。"""
    D = H.dropna(subset=[m])
    v = D[m].to_numpy()
    use_logit = m == "winup"          # 素の勝ち上がり率だけロジットの見込み。母を除く版は差なので足し算の見込み（eff は確率の差）
    G = v.mean()
    A = D.groupby(a_key)[m].mean()
    B = D.groupby(b_key)[m].mean()
    C = D.groupby([a_key, b_key])[m].agg(["mean", "size"]).reset_index()
    ea, eb = A.reindex(C[a_key]).to_numpy(), B.reindex(C[b_key]).to_numpy()
    if use_logit:
        exp = 1 / (1 + np.exp(-(logit(ea) + logit(eb) - logit(G))))
    else:
        exp = ea + eb - G
    C["eff"] = (C["mean"] - exp) * C["size"] / (C["size"] + K)
    return C


def persistence(H: pd.DataFrame, m: str, a_key: str, b_key: str, min_n: int = 10) -> dict:
    E1 = effects(H[H["per"] == 1], m, a_key, b_key)
    E2 = effects(H[H["per"] == 2], m, a_key, b_key)
    J = E1.merge(E2, on=[a_key, b_key], suffixes=("1", "2"))
    J = J[(J["size1"] >= min_n) & (J["size2"] >= min_n)]
    if len(J) < 10:
        return {"pairs": int(len(J))}
    return {"pairs": int(len(J)), "pearson": round(float(J["eff1"].corr(J["eff2"])), 3),
            "spearman": round(float(J["eff1"].corr(J["eff2"], method="spearman")), 3),
            "sd_eff1": round(float(J["eff1"].std()), 4)}


def null_persistence(H: pd.DataFrame, m: str, a_key: str, b_key: str, n: int = 20) -> dict:
    out = []
    for k in range(n):
        rng = np.random.default_rng(500 + k)
        P = H.copy()
        # 母の単位で入れ替える。母ごとに 1 つの札（母父は母で決まる）を持たせ、母の一番多い父（a_key）の中で母どうしで入れ替える。
        # 同じ母の期をまたぐ子は同じ札のまま（全きょうだいの持続を保つ。critic 1 回目の 4・2 回目の 2）
        U = P.groupby("dam").agg(_a=(a_key, lambda x: x.mode().iat[0] if x.notna().any() else np.nan), _b=(b_key, "first")).reset_index()
        for _, idx in U.groupby("_a").indices.items():
            U.iloc[idx, U.columns.get_loc("_b")] = U["_b"].to_numpy()[idx][rng.permutation(len(idx))]
        P[b_key] = P["dam"].map(U.set_index("dam")["_b"]).to_numpy()
        r = persistence(P, m, a_key, b_key)
        if "pearson" in r:
            out.append(r["pearson"])
    if not out:
        return {"runs": 0}
    return {"runs": len(out), "mean": round(float(np.mean(out)), 3), "sd": round(float(np.std(out)), 3), "q95": round(float(np.percentile(out, 95)), 3)}


def main() -> None:
    H = horse_table()
    res = {"horses": int(len(H)), "per_counts": H["per"].value_counts().to_dict()}
    for m in ("winup", "maxcls", "prize"):
        H[f"{m}_adj"] = dam_adjust(H, m)
    for unit, (ak, bk) in {"id": ("sire", "ds"), "line": ("ss", "bms_sire")}.items():
        for m in ("winup", "maxcls", "prize"):
            for ver, col in (("add", m), ("dam_adj", f"{m}_adj")):
                key = f"{unit}_{m}_{ver}"
                r = persistence(H, col, ak, bk)
                if "pearson" in r and unit == "id":
                    r["null"] = null_persistence(H, col, ak, bk)
                res[key] = r
                print(key, r, flush=True)
    # よく知られたニックス: ディープインパクト × 母父がストームキャットかその息子（第 1 期・2 期の効きと順位）
    for per in (1, 2):
        E = effects(H[H["per"] == per], "maxcls", "sire", "ds")
        E = E[E["size"] >= 10].copy()
        E["pct"] = E["eff"].rank(pct=True)
        sc_bms = set(H.loc[(H["ds"] == STORMCAT) | (H["bms_sire"] == STORMCAT), "ds"])
        D = E[(E["sire"] == DEEP) & E["ds"].isin(sc_bms)]
        res[f"deep_x_stormcat_per{per}"] = D[["ds", "size", "eff", "pct"]].round(3).to_dict("records")
    # 市場: 2021〜24 年の開発の行で 2017 年以降の生まれに、第 1 期の効き（id・maxcls・足し算と母を除く版）を当てる
    F = frame()
    X = attach(F)
    pedk = data.load_ped()[["horse_id", "sire", "ds"]].rename(columns={"sire": "s_", "ds": "b_"})
    X = X.merge(pedk, on="horse_id", how="left").sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    yr = X["date"].dt.year.to_numpy()
    byv = pd.to_numeric(X["horse_id"].str[:4], errors="coerce").to_numpy()
    for ver, col in (("add", "maxcls"), ("dam_adj", "maxcls_adj")):
        E1 = effects(H[H["per"] == 1], col, "sire", "ds").rename(columns={"sire": "s_", "ds": "b_"})
        z = X[["s_", "b_"]].merge(E1[["s_", "b_", "eff"]], on=["s_", "b_"], how="left")["eff"].fillna(0).to_numpy()
        m = (yr >= 2021) & (byv >= 2017)
        S = X[m].reset_index(drop=True)
        zz = zcol(S, z[m])
        b, bs = fit_boot(S, np.c_[S["lp"], S["lq"], zz], 200)
        res[f"market_{ver}"] = {"rows": int(m.sum()), "coef": round(float(b[2]), 4), "ci": np.percentile(bs[:, 2], [2.5, 97.5]).round(4).tolist()}
        print("market", ver, res[f"market_{ver}"], flush=True)
    path = data.OUT / "nicks.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
