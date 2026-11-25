"""RC-4 の候補の列: 初出走の馬（JRA で前走が無い馬・新馬と未勝利）の、目録に無い as-of の成績。

- 列（その走の日より前の、初出走の走だけで数える。同じ日は入れない）:
  調教師の初出走の成績（trainer_id）・種牡馬の産駒の初出走の成績（父の名前）・母の過去の産駒の初出走の成績（兄姉。ped_anc の gen1 pos1）・
  生産者の初出走の成績（horse_profile の breeder）。
- 2015 年以前の走が表に無いので、2016〜17 年は数が少ない（as-of で数えるため）。
- 出力: <COURSE_OUT>/rules_council/debut_cols.parquet（full.parquet の行の並びで、初出走でない行は欠け）。費用 0 円（手元の表だけ）。
使い方: python -m src.research.rules_council.debut [bound]（nice -n 19）
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.rules_council import engine as E
from src.research.rules_council import rc_frame as F

PED = F.CR.REPO / "data/local/modeling/training_pipeline/factory/src/ped_anc.parquet"
PROFILE = F.CR.REPO / "data/local/modeling/training_pipeline/factory/conn/horse_profile.parquet"
KEYS = {"trainer": "調教師", "sire": "種牡馬の産駒", "dam": "母の産駒（兄姉）", "breeder": "生産者"}
MIN_N = 10
MIN_N_KEY = {"trainer": 10, "sire": 10, "dam": 2, "breeder": 10}   # 母の産駒は少ないので 2 頭から


def debut_rows(fr: pd.DataFrame) -> pd.DataFrame:
    """初出走の走: starts_prev = 0 で、新馬か未勝利。frame は 2016 年からなので、2016 年の未勝利は 2015 年に走った馬が混ざる（偽の初出走）。
    そのため 2016 年は新馬だけにする。同じ馬が 2 行以上あれば最初の 1 走だけ。"""
    sp = pd.to_numeric(fr["starts_prev"], errors="coerce")
    cls = pd.to_numeric(fr["cls"], errors="coerce")
    yr = pd.to_numeric(fr["year"], errors="coerce")
    deb = fr[(sp == 0) & ((cls == 0) | ((cls == 1) & (yr >= 2017)))].copy()
    return deb.sort_values("date").drop_duplicates("horse_id", keep="first")


def build() -> pd.DataFrame:
    fr = pd.read_parquet(F.CR.OUT / "v3" / "frame.parquet",
                         columns=["race_id", "horse_id", "date", "year", "cls", "starts_prev", "trainer_id", "f_name", "fin"],
                         filters=[("year", "<=", F.YEAR_END)])
    fr["race_id"], fr["horse_id"] = fr["race_id"].astype(str), fr["horse_id"].astype(str)
    deb = debut_rows(fr)
    a = pd.read_parquet(PED)
    d1 = a[(a["gen"] == 1) & (a["pos"] == 1)].drop_duplicates("horse_id")
    dam = pd.Series(d1["anc_id"].astype(str).to_numpy(), index=d1["horse_id"].astype(str).to_numpy())
    prof = pd.read_parquet(PROFILE, columns=["horse_id", "breeder"]).drop_duplicates("horse_id")
    prof["horse_id"] = prof["horse_id"].astype(str)
    deb["trainer"] = deb["trainer_id"].astype(str)
    deb["sire"] = deb["f_name"].astype(str)
    deb["dam"] = deb["horse_id"].map(dam)
    deb = deb.merge(prof, on="horse_id", how="left")
    fin = pd.to_numeric(deb["fin"], errors="coerce")
    deb["w"] = (fin == 1).astype(float)
    deb["t3"] = ((fin >= 1) & (fin <= 3)).astype(float)
    deb["dt"] = pd.to_datetime(deb["date"])
    deb = deb.sort_values("dt").reset_index(drop=True)
    for k in KEYS:
        key = deb[k].where(deb[k].notna() & (deb[k].astype(str) != "nan") & (deb[k].astype(str) != ""))
        g = deb.assign(_k=key).dropna(subset=["_k"])
        # 同じ日の走を入れない as-of の累計（日ごとに足してから、日をずらす）
        day = g.groupby(["_k", "dt"]).agg(n=("w", "size"), w=("w", "sum"), t3=("t3", "sum")).reset_index().sort_values(["_k", "dt"])
        for c in ("n", "w", "t3"):
            day[f"c_{c}"] = day.groupby("_k")[c].cumsum() - day[c]
        m = g[["_k", "dt"]].reset_index().merge(day[["_k", "dt", "c_n", "c_w", "c_t3"]], on=["_k", "dt"], how="left").set_index("index")
        for c in ("n", "w", "t3"):
            deb.loc[m.index, f"{k}_{c}"] = m[f"c_{c}"].to_numpy()
    keep = ["race_id", "horse_id"] + [f"{k}_{c}" for k in KEYS for c in ("n", "w", "t3")]
    return deb[keep]


def cmd_bound() -> dict:
    """条件 3: 市場を使わない上限（φ = 0）を、探す側 70% だけで出す。初出走の馬の中で、列の水準（as-of の勝率の上位）の勝率 ÷ 全体の勝率。
    帯（再現 T-60 の一桁・穴馬）ごとに、同じ帯の中の比も出す（オッズは層にだけ使う）。"""
    full = pd.read_parquet(F.OUTW / "full.parquet", columns=["race_id", "horse_id", "kaisai", "year", "win", "top3", "cls", "q"])
    full["horse_id"] = full["horse_id"].astype(str)
    pre = np.load(F.OUTW / "full_pre.npy")[0]
    full["pre"] = pre
    cols = pd.read_parquet(F.OUTW / "debut_cols.parquet")
    x = full.merge(cols, on=["race_id", "horse_id"], how="inner")
    x = x[E.search70(x["kaisai"].to_numpy()) & (x["pre"] <= 50)].copy()         # 探す側 70%・再現 ≤ 50 倍
    out = {"runs": int(len(x)), "win_rate_all": float(x["win"].mean()), "cols": {}}
    for k, nm in KEYS.items():
        n, w = x[f"{k}_n"], x[f"{k}_w"]
        rate = (w + 1) / (n + 10)                              # 縮めた as-of の勝率（事前 10%。母の産駒のように数が少ない列ほど強く縮む）
        ok = n >= MIN_N_KEY[k]
        r = {"known_share": float(ok.mean())}
        for band, m in {"一桁": x["pre"] < 10, "穴馬": x["pre"] >= 10}.items():
            y = x[m & ok].assign(rate=rate[m & ok])
            if len(y) < 200:
                continue
            yr = y["year"].astype(int).to_numpy()
            ev = yr >= 2018                                           # 線は前の年だけで決めるので、2016〜17 は数えない（線の無い年も数えない。下で）
            y = y.assign(_r=y["rate"] + 1e-12 * np.random.default_rng(0).random(len(y)))   # 同順位は乱数で切る（10% と 20% を同じ組にしない）
            th = {}
            for yy in range(2018, F.YEAR_END + 1):
                prev = y["_r"].to_numpy()[yr < yy]
                if len(prev) >= 100:                                  # 前の年の分かる走が少ない年は数えない
                    th[yy] = (np.quantile(prev, 0.8), np.quantile(prev, 0.9))
            t80 = np.array([th[v][0] if v in th else np.inf for v in yr])
            t90 = np.array([th[v][1] if v in th else np.inf for v in yr])
            ev = ev & np.isfinite(t80)
            ye = y[ev]
            base, qb = float(ye["win"].mean()), float(ye["q"].mean())        # 比べる元は、同じ帯・同じ年（2018〜）の列の分かる走
            r[band] = {"runs_known_2018on": int(len(ye)), "win_rate_band_known": base}
            for nm_, tt in (("top20", t80), ("top10", t90)):
                t = y[(y["_r"].to_numpy() >= tt) & ev]
                lift, Q = float(t["win"].mean() / base), float(t["q"].mean() / qb)
                k_ = float(t["win"].sum())
                # Q: 上位の群の確定の q の平均 ÷ 比べる元の q の平均（オッズだけ）。lift ÷ Q は探す側の市場比の粗い値（RC-4 の R1 の一致で足した）
                r[band] |= {f"{nm_}_runs": int(len(t)), f"{nm_}_wins": k_, f"{nm_}_lift": lift, f"{nm_}_Q": Q, f"{nm_}_lift_over_Q": lift / Q,
                            f"{nm_}_se": float((lift / Q) / np.sqrt(max(k_, 1)))}
        out["cols"][nm] = r
    # 付く馬の偏り（着順なし）: 生産者・母が付く率を、初出走の年と、その後の出走の数ごとに
    fr = pd.read_parquet(F.CR.OUT / "v3" / "frame.parquet", columns=["horse_id", "year", "starts_prev", "cls"], filters=[("year", "<=", F.YEAR_END)])
    fr["year"] = fr["year"].astype(int)
    fr["horse_id"] = fr["horse_id"].astype(str)
    runs = fr.groupby("horse_id").size()
    dd = debut_rows(fr.assign(date=fr["year"].astype(str)))
    dd["later"] = dd["horse_id"].map(runs) - 1
    prof = set(pd.read_parquet(PROFILE, columns=["horse_id"])["horse_id"].astype(str))
    a_ = pd.read_parquet(PED)
    damset = set(a_[(a_["gen"] == 1) & (a_["pos"] == 1)]["horse_id"].astype(str))
    dd["has_breeder"], dd["has_dam"] = dd["horse_id"].isin(prof), dd["horse_id"].isin(damset)
    dd["later_group"] = pd.cut(dd["later"], [-1, 0, 2, 5, 10, 10_000], labels=["0", "1-2", "3-5", "6-10", "11+"]).astype(str)
    out["coverage_by_year"] = dd.groupby("year")[["has_breeder", "has_dam"]].mean().round(3).to_dict("index")
    out["coverage_by_later_runs"] = dd.groupby("later_group")[["has_breeder", "has_dam"]].mean().round(3).to_dict("index")
    o19 = dd[dd["year"].astype(int) <= 2019]
    out["coverage_by_later_runs_2016_19"] = o19.groupby("later_group")[["has_breeder", "has_dam"]].mean().round(3).to_dict("index")
    out["note"] = ("市場を使わない上限（φ = 0）: 列の上位の勝率 ÷ 同じ帯の列の分かる初出走の勝率（2018〜24）。上位の線はその年より前の分布だけで決める（as-of の分位）。市場が列を全く値段に入れていないときの市場比の上限にあたる。"
                   "探す側 70% だけ（確かめの側 30% の着順は読まない）。列の水準の上位は as-of の縮めた勝率の 80%・90% 点（その帯の中）")
    p = F.CR.REPO / "docs/operations/retrain_trials/rules_council_rc4_bound.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main() -> None:
    import sys
    if sys.argv[1:] == ["bound"]:
        cmd_bound()
        return
    D = build()
    D.to_parquet(F.OUTW / "debut_cols.parquet", index=False)
    print("debut runs", len(D), {k: float((D[f"{k}_n"] >= MIN_N_KEY[k]).mean()) for k in KEYS})


if __name__ == "__main__":
    main()
