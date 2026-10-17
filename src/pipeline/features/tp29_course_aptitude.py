"""tp-29: 期待とずれた馬（期待外れ / 期待以上）のグループ差分から作る、コース別の向き/不向き指標。

グループの定義（ユーザー提案）。期待順位 = 人気（市場版）か、モデルの勝率順位（モデル期待版）。
  G1（期待外れ）: 期待順位 1〜3 位で、着順 6 着以降、かつ 3 着馬との着差（走破タイム差）が 0.3 秒以上
  G2（期待以上）: 期待順位 6 位以降で 3 着以内、または 4 着以降でも 3 着馬との着差が 0.2 秒以内（3 着以内とみなす）
除外: 着順 ≤ 0（取消・除外・中止・失格）、走破タイム ≤ 0 または欠損、期待順位 ≤ 0（人気なし）、3 着馬が定まらないレース。
同着は 3 着馬の時計が同じなのでそのまま扱う。

コース単位（3 階層）: k1 = 場|面|距離|内外、k2 = 場|面|距離帯、k3 = 面|距離帯。
属性（トークン）: 種牡馬・母父・騎手・調教師の ID、脚質、枠・馬番（頭数で正規化）、年齢・性別、斤量・馬体重・増減、前走間隔、クラス変更、距離の適合差、同コース経験。
各コース × 属性トークンについて、G2 と G1 の出現率の差を縮約付き対数オッズ比で測り、Fisher 検定 + BH-FDR で有意なものだけを使う。

リーク防止: 年 Y の行は『年 Y より前の年のラベル』だけで作った統計で採点する（年単位の expanding）。人気は過去レースのラベルを作る材料としてだけ使い、
当日の人気は一切使わない。馬自身の履歴は日付が厳密に前の行だけを数える。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SLOW_RANK = 6          # 着順 6 着以降
G1_RANK_MAX = 3        # 期待順位 1〜3
G2_RANK_MIN = 6        # 期待順位 6 位以降
G1_GAP = 0.3
G2_GAP = 0.2
ALPHA = 30.0           # 縮約の強さ（行数換算）
Q_FDR = 0.10
MIN_CNT = 4            # 検定に回す最小件数（G1+G2）
LOR_CLIP = 3.0         # 対数比の上下限

TOKEN_COLS = ["sire", "dam_sire", "jockey", "trainer", "style", "gate", "bracket", "age", "sex", "jw", "bw", "wtchg", "rest", "clsstep", "distgap", "nco"]
LEVELS = ("k1", "k2", "k3")


# ------------------------------------------------------------------ ラベル

def make_groups(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    """df: race_id, horse_id, finish_position, time_sec, <rank_col>（期待順位、1 が最上位）。
    戻り値（df と同じ index）: g1, g2（bool）, gap3（3 着馬との着差）, labelable（ラベルを付けられる行）。"""
    fin = pd.to_numeric(df["finish_position"], errors="coerce")
    t = pd.to_numeric(df["time_sec"], errors="coerce")
    rk = pd.to_numeric(df[rank_col], errors="coerce")
    ok = (fin > 0) & (t > 0) & (rk > 0)
    race = df["race_id"].astype(str)
    t3 = t.where(ok & (fin == 3)).groupby(race).transform("min")
    gap = (t - t3).round(2)
    lab = ok & t3.notna()
    g1 = lab & (rk <= G1_RANK_MAX) & (fin >= SLOW_RANK) & (gap >= G1_GAP)
    g2 = lab & (rk >= G2_RANK_MIN) & ((fin <= 3) | ((fin >= 4) & (gap <= G2_GAP)))
    return pd.DataFrame({"g1": g1.to_numpy(), "g2": g2.to_numpy(), "gap3": gap.to_numpy(), "labelable": lab.to_numpy()}, index=df.index)


# ------------------------------------------------------------------ コースと属性

def dist_band(distance) -> pd.Series:
    d = pd.to_numeric(distance, errors="coerce")
    return pd.cut(d, [0, 1300, 1600, 2000, 2400, 99999], labels=["S", "M", "I", "L", "X"]).astype("object")


def course_keys(df: pd.DataFrame, variant: pd.Series | None = None) -> pd.DataFrame:
    surf = df["surface"].astype(str).str.replace("ダート", "ダ", regex=False)
    band = dist_band(df["distance"]).fillna("?").astype(str)
    v = (variant if variant is not None else pd.Series("?", index=df.index)).fillna("?").astype(str).replace({"": "-"})
    ven = df["venue_code"].astype(str).str.zfill(2)
    dist = pd.to_numeric(df["distance"], errors="coerce").fillna(0).astype(int).astype(str)
    return pd.DataFrame({"k1": ven + "|" + surf + "|" + dist + "|" + v, "k2": ven + "|" + surf + "|" + band, "k3": surf + "|" + band}, index=df.index)


def _bin(x: pd.Series, edges: list[float], names: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(x, errors="coerce"), edges, labels=names).astype("object")


def attribute_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """属性トークン（'名前=値' の文字列、欠損は NaN）。df は学習表の列 + sire_id, dam_sire_id を持つ。"""
    t = pd.DataFrame(index=df.index)

    def tag(name: str, s: pd.Series) -> pd.Series:
        s = s.astype("object")
        return s.where(s.isna(), name + "=" + s.astype(str))

    n = pd.to_numeric(df["field_size"], errors="coerce").replace(0, np.nan)
    t["sire"] = tag("sire", df["sire_id"].replace("", np.nan))
    t["dam_sire"] = tag("dam_sire", df["dam_sire_id"].replace("", np.nan))
    t["jockey"] = tag("jockey", df["jockey_id"].replace("", np.nan))
    t["trainer"] = tag("trainer", df["trainer_id"].replace("", np.nan))
    t["style"] = tag("style", _bin(df["a_style"], [-1e9, 0.25, 0.5, 0.75, 1e9], ["front", "mid_front", "mid_back", "back"]))
    pos = pd.to_numeric(df["horse_number"], errors="coerce") / n
    t["gate"] = tag("gate", _bin(pos, [0, 1 / 3, 2 / 3, 1.01], ["in", "mid", "out"]))
    brk = pd.to_numeric(df["bracket_number"], errors="coerce")
    t["bracket"] = tag("bracket", _bin(brk, [0, 2, 4, 6, 8], ["b12", "b34", "b56", "b78"]))
    sa = df["sex_age"].astype(str)
    t["age"] = tag("age", _bin(pd.to_numeric(sa.str[1:], errors="coerce"), [0, 2, 3, 4, 5, 99], ["a2", "a3", "a4", "a5", "a6p"]))
    t["sex"] = tag("sex", sa.str[:1].replace({"": np.nan}))
    t["jw"] = tag("jw", _bin(df["jockey_weight"], [0, 52, 54, 56, 58, 99], ["w_le52", "w_53_54", "w_55_56", "w_57_58", "w_59p"]))
    t["bw"] = tag("bw", _bin(df["weight"], [100, 440, 480, 520, 700], ["bw_lt440", "bw_440_480", "bw_480_520", "bw_gt520"]))
    t["wtchg"] = tag("wtchg", _bin(df["d_wt_chg_now"], [-99, -6, -2, 2, 6, 99], ["wc_m6", "wc_m2", "wc_0", "wc_p2", "wc_p6"]))
    first = pd.to_numeric(df["d_nprior"], errors="coerce") == 0
    rest = _bin(df["d_rest"], [-1, 14, 35, 70, 140, 1e5], ["r_le14", "r_15_35", "r_36_70", "r_71_140", "r_gt140"])
    t["rest"] = tag("rest", rest.where(~first, "r_first"))
    t["clsstep"] = tag("clsstep", _bin(df["c_cls_step"], [-99, -0.5, 0.5, 99], ["cls_down", "cls_same", "cls_up"]))
    t["distgap"] = tag("distgap", _bin(df["b_dist_gap"], [-1e9, -300, 300, 1e9], ["dg_shorter", "dg_same", "dg_longer"]))
    t["nco"] = tag("nco", _bin(df["b_n_course"], [-1, 0, 1, 3, 1e5], ["nco_0", "nco_1", "nco_2_3", "nco_4p"]))
    return t[TOKEN_COLS]


# ------------------------------------------------------------------ 統計表

def _long(rows: pd.DataFrame, level: str, g: str) -> pd.DataFrame:
    parts = []
    for c in TOKEN_COLS:
        s = rows[c]
        m = s.notna()
        if m.any():
            parts.append(pd.DataFrame({"k": rows.loc[m, level].to_numpy(), "tok": s[m].to_numpy()}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"k": [], "tok": []})
    out["g"] = g
    return out


def _bh(p: np.ndarray, q: float) -> np.ndarray:
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thr = q * (np.arange(1, m + 1) / m)
    ok = p[order] <= thr
    if not ok.any():
        return np.zeros(m, bool)
    cut = np.max(np.flatnonzero(ok))
    sig = np.zeros(m, bool)
    sig[order[: cut + 1]] = True
    return sig


def level_table(S: pd.DataFrame, level: str, parent: str | None, alpha: float = ALPHA, q: float = Q_FDR, min_cnt: int = MIN_CNT) -> pd.DataFrame:
    """S: ラベル（g1 または g2）のある行（トークン列と k1,k2,k3 を持つ）。戻り値: k, tok, n1, n2, N1, N2, lor（縮約）, p, sig。"""
    from scipy.stats import fisher_exact

    g1, g2 = S[S["g1"]], S[S["g2"]]
    L = pd.concat([_long(g1, level, "n1"), _long(g2, level, "n2")], ignore_index=True)
    if L.empty:
        return pd.DataFrame(columns=["k", "tok", "n1", "n2", "N1", "N2", "lor", "p", "sig", "xlor", "xp", "xsig"])
    cnt = L.groupby(["k", "tok", "g"]).size().unstack("g", fill_value=0)
    for c in ("n1", "n2"):
        if c not in cnt.columns:
            cnt[c] = 0
    cnt = cnt[["n1", "n2"]].reset_index()
    tot1 = g1.groupby(level).size().rename("N1")
    tot2 = g2.groupby(level).size().rename("N2")
    cnt = cnt.merge(tot1, left_on="k", right_index=True, how="left").merge(tot2, left_on="k", right_index=True, how="left").fillna({"N1": 0, "N2": 0})
    # 親階層（または全体）での出現率に縮約する
    if parent is None:
        a1 = _long(g1, "k3", "n1").groupby("tok").size()
        a2 = _long(g2, "k3", "n2").groupby("tok").size()
        s1 = (a1 / max(len(g1), 1)).rename("ps1")
        s2 = (a2 / max(len(g2), 1)).rename("ps2")
        cnt = cnt.merge(s1, left_on="tok", right_index=True, how="left").merge(s2, left_on="tok", right_index=True, how="left")
    else:
        kmap = S[[level, parent]].drop_duplicates().set_index(level)[parent]
        cnt["pk"] = cnt["k"].map(kmap)
        p1 = _long(g1, parent, "n1").groupby(["k", "tok"]).size().rename("c1").reset_index()
        p2 = _long(g2, parent, "n2").groupby(["k", "tok"]).size().rename("c2").reset_index()
        pt1 = g1.groupby(parent).size().rename("T1")
        pt2 = g2.groupby(parent).size().rename("T2")
        p1 = p1.merge(pt1, left_on="k", right_index=True)
        p2 = p2.merge(pt2, left_on="k", right_index=True)
        p1["ps1"] = p1["c1"] / p1["T1"]
        p2["ps2"] = p2["c2"] / p2["T2"]
        cnt = cnt.merge(p1[["k", "tok", "ps1"]].rename(columns={"k": "pk"}), on=["pk", "tok"], how="left").merge(p2[["k", "tok", "ps2"]].rename(columns={"k": "pk"}), on=["pk", "tok"], how="left")
    cnt[["ps1", "ps2"]] = cnt[["ps1", "ps2"]].fillna(0.0)
    r1 = (cnt["n1"] + alpha * cnt["ps1"] + 0.5) / (cnt["N1"] + alpha + 1.0)
    r2 = (cnt["n2"] + alpha * cnt["ps2"] + 0.5) / (cnt["N2"] + alpha + 1.0)
    cnt["lor"] = np.log(r2 / r1).clip(-LOR_CLIP, LOR_CLIP)         # 片方のグループに全く出ない属性で極端な値にならないよう、疑似件数 0.5 と上下限を置く
    p = np.ones(len(cnt))
    cand = np.flatnonzero(((cnt["n1"] + cnt["n2"]) >= min_cnt).to_numpy() & (cnt["N1"] > 0).to_numpy() & (cnt["N2"] > 0).to_numpy())
    n1, n2, N1, N2 = (cnt[c].to_numpy() for c in ("n1", "n2", "N1", "N2"))
    for i in cand:
        p[i] = fisher_exact([[n2[i], N2[i] - n2[i]], [n1[i], N1[i] - n1[i]]])[1]
    sig = np.zeros(len(cnt), bool)
    if len(cand):
        sig[cand] = _bh(p[cand], q)
    cnt["p"] = p
    cnt["sig"] = sig
    # コース固有の超過: 他のコース全体での G2/G1 の差（全コース共通の『人気馬と穴馬の違い』）を引いた差の差（difference-in-differences）
    from scipy.stats import norm

    NT1, NT2 = float(len(g1)), float(len(g2))
    tn1 = cnt.groupby("tok")["n1"].transform("sum")
    tn2 = cnt.groupby("tok")["n2"].transform("sum")
    o1, o2 = tn1 - cnt["n1"], tn2 - cnt["n2"]
    O1, O2 = NT1 - cnt["N1"], NT2 - cnt["N2"]
    lk = np.log((cnt["n2"] + .5) / (cnt["N2"] - cnt["n2"] + .5)) - np.log((cnt["n1"] + .5) / (cnt["N1"] - cnt["n1"] + .5))
    lo = np.log((o2 + .5) / (O2 - o2 + .5)) - np.log((o1 + .5) / (O1 - o1 + .5))
    var = sum(1.0 / (c + .5) for c in (cnt["n2"], cnt["N2"] - cnt["n2"], cnt["n1"], cnt["N1"] - cnt["n1"], o2, O2 - o2, o1, O1 - o1))
    z = (lk - lo) / np.sqrt(var)
    xp = np.ones(len(cnt))
    xp[cand] = 2 * norm.sf(np.abs(z.to_numpy()[cand]))
    lor_out = np.log(((o2 + .5) / (O2 + 1)) / ((o1 + .5) / (O1 + 1)))
    cnt["xlor"] = (cnt["lor"] - lor_out).clip(-LOR_CLIP, LOR_CLIP)
    xsig = np.zeros(len(cnt), bool)
    if len(cand):
        xsig[cand] = _bh(xp[cand], q)
    cnt["xp"] = xp
    cnt["xsig"] = xsig
    return cnt[["k", "tok", "n1", "n2", "N1", "N2", "lor", "p", "sig", "xlor", "xp", "xsig"]]


def build_tables(S: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"k1": level_table(S, "k1", "k2"), "k2": level_table(S, "k2", "k3"), "k3": level_table(S, "k3", None)}


def _score_level(rows: pd.DataFrame, level: str, table: pd.DataFrame, excess: bool = False) -> pd.DataFrame:
    sig = table[table["xsig" if excess else "sig"]][["k", "tok", "xlor" if excess else "lor"]].rename(columns={"xlor": "lor"})
    out = pd.DataFrame(0.0, index=rows.index, columns=["sum", "pos", "neg", "n"])
    if sig.empty:
        return out
    parts = []
    for c in TOKEN_COLS:
        s = rows[c]
        m = s.notna()
        if m.any():
            parts.append(pd.DataFrame({"i": rows.index[m], "k": rows.loc[m, level].to_numpy(), "tok": s[m].to_numpy()}))
    long = pd.concat(parts, ignore_index=True).merge(sig, on=["k", "tok"], how="inner")
    if long.empty:
        return out
    long["pos"] = long["lor"].clip(lower=0)
    long["neg"] = long["lor"].clip(upper=0)
    agg = long.groupby("i").agg(sum=("lor", "sum"), pos=("pos", "sum"), neg=("neg", "sum"), n=("lor", "size"))
    out.loc[agg.index, ["sum", "pos", "neg", "n"]] = agg[["sum", "pos", "neg", "n"]].to_numpy()
    return out


def horse_history(F: pd.DataFrame) -> pd.DataFrame:
    """馬自身の過去（日付が厳密に前）の G1 / G2 の回数。全体・同じ k2・同じ面。F は date, horse_id, g1, g2, k2, surf を持つ。"""
    F = F.sort_values(["horse_id", "date", "race_id"], kind="stable")
    out = pd.DataFrame(index=F.index)
    g1, g2 = F["g1"].astype(int), F["g2"].astype(int)
    one = pd.Series(1, index=F.index)
    for name, keys in (("all", ["horse_id"]), ("k2", ["horse_id", "k2"]), ("surf", ["horse_id", "surf"])):
        gb = lambda s: s.groupby([F[k] for k in keys]).cumsum() - s
        out[f"hz_g1_{name}"] = gb(g1)
        out[f"hz_g2_{name}"] = gb(g2)
        out[f"hz_n_{name}"] = gb(one)
    return out


def build_features(F: pd.DataFrame, years: list[int] | None = None, alpha: float = ALPHA) -> tuple[pd.DataFrame, dict[int, dict[str, pd.DataFrame]]]:
    """F: race_id, horse_id, year, date, g1, g2, surf, k1, k2, k3 とトークン列。
    戻り値: (F と同じ index の特徴量, 年ごとの統計表)。年 Y の行は Y より前の年のラベルだけで採点する。"""
    ys = sorted(F["year"].unique()) if years is None else years
    names = [f"{p}{l}_{s}" for p in ("apt", "aptx") for l in (1, 2, 3) for s in ("sum", "pos", "neg", "n")]
    feats = pd.DataFrame(index=F.index, columns=names, dtype="float64")
    tables: dict[int, dict[str, pd.DataFrame]] = {}
    for y in ys:
        S = F[(F["year"] < y) & (F["g1"] | F["g2"])]
        rows = F[F["year"] == y]
        if len(S) < 50 or rows.empty:
            continue
        tb = build_tables(S)
        tables[int(y)] = tb
        for i, lv in enumerate(LEVELS, start=1):
            for pre, ex in (("apt", False), ("aptx", True)):
                sc = _score_level(rows, lv, tb[lv], excess=ex)
                for s in ("sum", "pos", "neg", "n"):
                    feats.loc[rows.index, f"{pre}{i}_{s}"] = sc[s].to_numpy()
    # 学習側に過去が無い年（最初の年）は欠損のまま
    for pre in ("apt", "aptx"):
        for i in (1, 2, 3):
            tot = feats[f"{pre}{i}_pos"] + feats[f"{pre}{i}_neg"]
            feats[f"{pre}{i}_cls"] = np.sign(tot).where(tot.abs() >= 0.5, 0.0)
            feats.loc[feats[f"{pre}{i}_sum"].isna(), f"{pre}{i}_cls"] = np.nan
    hh = horse_history(F)
    feats = pd.concat([feats, hh.reindex(F.index)], axis=1)
    return feats, tables
