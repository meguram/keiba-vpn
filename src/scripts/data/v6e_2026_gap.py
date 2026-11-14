"""2026 年の本番 v6e で、市場に対する上積みが消えた理由の切り分け（読み取りのみ・手元のファイルだけ。GCS・GCP は読まない）。

python -m src.scripts.data.v6e_2026_gap [--out DIR]

見るもの（docs/operations/retrain_trials/v6e_2026_gap_20261114.md）:
1. 作り方の差: 同じモデル・同じレースを、当日の推論（shadow・predictions_v6）と後から作った予測（backfill）で突き合わせる。
   ◎ の一致率、勝率の相関・差の分布。
2. 予測の出どころ × 期間ごとの、市場（確定オッズ）に対する上積み。
   - 勝ち馬の対数損失（モデル − 市場。負ならモデルが良い）
   - Benter 型の混合 softmax(a·log p + b·log q) の a（市場に上乗せしてモデルに情報が残っているか。0 なら残っていない）
   - ◎ の単勝の回収率と、市場の 1 番人気の単勝の回収率（確定オッズ × 100 円で近似。同着は無視）
   2025 年は学習側の経路で作った test の予測、2026 年は推論の経路（当日・後から作った予測）。
3. 2026 年の月別の推移（同じ作り方の s18 の候補と並べる）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
INF = ROOT / "data/local/inference"
MOD = ROOT / "data/local/modeling"
B16 = Path("/home/jovyan/work/retrain_out/base_2016")
TABLES = Path(os.environ.get("FT_TABLES", str(ROOT / "data/page_reference/tables")))
EXCLUDED = {"2026-11-07", "2026-11-08"}          # 推論の履歴の年のずれ（config/frozen_tracking.json の excluded_dates）

# 出どころ: (名前, 種類, 読み方, 中身の説明)
SOURCES = {
    "v6e_prod_test2025": ("file", MOD / "training_pipeline/final_eval_v6e/test2025_predictions.parquet", "本番 v6e（2020〜2024 学習）の 2025 年 test。学習側の経路"),
    "v6e_2016_test2025": ("file", B16 / "final_eval_v6e16/test2025_predictions.parquet", "s18 の候補（2016〜2024 学習）の 2025 年 test。学習側の経路"),
    "v6e_prod_backfill2026": ("dir", INF / "predictions_backfill_old_bundle_v6_2025_20261112094014", "本番 v6e（2020〜2025 学習）を 2026 年に後から当てた予測（10/24〜25 に作成）"),
    "v6e_2016_backfill2026": ("dir", INF / "predictions_backfill/bundle_v6_2025", "s18 の候補（2016〜2025 学習）を 2026 年に後から当てた予測（11/11〜12 に作成）"),
    "v6e_prod_live2026": ("dir", INF / "predictions_v6", "本番 v6e の当日の予測（発走約 60 分前、10/25〜）"),
    "v4_backfill2026": ("dir", MOD / "betting/tp36/pred2026", "v4（bundle_v4_2025）を 2026 年に後から当てた予測（tp36_pred2026）"),
    "v4_live2026": ("dir", INF / "predictions_v3", "v4 の当日の予測（shadow）"),
}
CHECKS = {   # 作り方の差: (当日, 後から)
    "v4": (INF / "predictions_v3", INF / "predictions_backfill_check/bundle_v4_2025"),
    "v6e_prod_old_vs_check": (INF / "predictions_backfill_old_bundle_v6_2025_20261112094014", INF / "predictions_backfill_check/bundle_v6_2025"),
}


def read_dir(base: Path) -> pd.DataFrame:
    fs = list(base.glob("*.parquet")) + list(base.glob("*/*.parquet"))
    if not fs:
        return pd.DataFrame(columns=["race_id", "horse_id", "p_win"])
    d = pd.concat([pd.read_parquet(f, columns=["race_id", "horse_id", "p_win"]) for f in fs], ignore_index=True)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    return d.drop_duplicates(["race_id", "horse_id"], keep="last")


def load_source(name: str) -> pd.DataFrame:
    kind, path, _ = SOURCES[name]
    if kind == "file":
        d = pd.read_parquet(path, columns=["race_id", "horse_id", "p_win"])
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
        return d
    return read_dir(path)


def results(years) -> pd.DataFrame:
    fs = []
    for y in years:
        f = TABLES / str(y) / "race_result_flat.parquet"
        r = pd.read_parquet(f, columns=["race_id", "date", "horse_id", "horse_number", "finish_position", "odds", "popularity"])
        fs.append(r)
    r = pd.concat(fs, ignore_index=True)
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    for c in ("finish_position", "odds", "popularity"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r["date"] = pd.to_datetime(r["date"]).dt.strftime("%Y-%m-%d")
    return r[r["odds"] > 0]                        # 走った馬（取消は確定オッズが無い）


def race_frame(pred: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """走った馬が予測に全部そろうレースだけ。p は走った馬で正規化、q は確定オッズの逆数を正規化。"""
    d = res.merge(pred, on=["race_id", "horse_id"], how="left")
    ok = d.groupby("race_id")["p_win"].apply(lambda s: s.notna().all() and len(s) >= 3)
    d = d[d["race_id"].isin(ok[ok].index)].copy()
    d["p"] = d["p_win"] / d.groupby("race_id")["p_win"].transform("sum")
    inv = 1.0 / d["odds"]
    d["q"] = inv / inv.groupby(d["race_id"]).transform("sum")
    d["win"] = (d["finish_position"] == 1).astype(float)
    return d[d.groupby("race_id")["win"].transform("sum") == 1]       # 勝ち馬がちょうど 1 頭のレースだけ（同着は除く）


def benter_a(d: pd.DataFrame) -> tuple[float, float]:
    """softmax(a·log p + b·log q) の最尤推定 (a, b)（レースごとの条件付きロジット）。"""
    lp, lq = np.log(d["p"].clip(1e-6).to_numpy()), np.log(d["q"].clip(1e-6).to_numpy())
    w = d["win"].to_numpy()
    _, inv = np.unique(d["race_id"].to_numpy(), return_inverse=True)

    def nll(ab):
        z = ab[0] * lp + ab[1] * lq
        m = np.zeros(inv.max() + 1)
        np.maximum.at(m, inv, z)
        e = np.exp(z - m[inv])
        s = np.bincount(inv, weights=e)
        return -float((w * (z - m[inv] - np.log(s[inv]))).sum())

    r = minimize(nll, x0=[0.3, 0.8], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    return float(r.x[0]), float(r.x[1])


def benter_ci(d: pd.DataFrame, reps: int = 60, seed: int = 0) -> tuple[float, float]:
    """係数 a のレース単位のブートストラップ 95% 区間（reps 回。重いので回数は少なめ）。"""
    rng = np.random.default_rng(seed)
    ids = d["race_id"].unique()
    g = {r: x for r, x in d.groupby("race_id")}
    vals = []
    for _ in range(reps):
        pick = rng.choice(ids, size=len(ids), replace=True)
        bd = pd.concat([g[r].assign(race_id=f"{r}#{i}") for i, r in enumerate(pick)], ignore_index=True)
        vals.append(benter_a(bd)[0])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def race_ll(d: pd.DataFrame) -> pd.DataFrame:
    """レースごとの勝ち馬の対数損失（モデル・市場）。"""
    w = d[d["win"] == 1]
    return pd.DataFrame({"race_id": w["race_id"].to_numpy(), "ll_m": -np.log(w["p"].clip(1e-6)).to_numpy(), "ll_q": -np.log(w["q"].clip(1e-6)).to_numpy()})


def did(hi25: pd.DataFrame, lo25: pd.DataFrame, hi26: pd.DataFrame, lo26: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """差の差: [(2026 の層 hi − 2026 の層 lo) − (2025 の層 hi − 2025 の層 lo)] の『モデル − 市場』の対数損失。4 つの集まりを別々に再標本化した 95% 区間。"""
    rng = np.random.default_rng(seed)
    parts = [(x["ll_m"] - x["ll_q"]).to_numpy() for x in (hi26, lo26, hi25, lo25)]
    point = (parts[0].mean() - parts[1].mean()) - (parts[2].mean() - parts[3].mean())
    bs = np.zeros(n_boot)
    for i, (sgn, x) in enumerate(zip((1, -1, -1, 1), parts)):
        bs += sgn * x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return {"did": float(point), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}


def two_sample(x: pd.DataFrame, y: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """別々のレースの集まり x と y の『モデル − 市場』の対数損失の平均の差（x − y）と 95% 区間。"""
    rng = np.random.default_rng(seed)
    a, b = (x["ll_m"] - x["ll_q"]).to_numpy(), (y["ll_m"] - y["ll_q"]).to_numpy()
    bs = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1) - b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(1)
    return {"diff": float(a.mean() - b.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}


def paired(dx: pd.DataFrame, dy: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """同じレースでの対の比較: モデル x とモデル y の勝ち馬の対数損失の差（x − y）の 95% 区間。"""
    m = race_ll(dx).merge(race_ll(dy), on="race_id", suffixes=("_x", "_y"))
    v = (m["ll_m_x"] - m["ll_m_y"]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return {"races": int(len(v)), "diff": float(v.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}


def summary(d: pd.DataFrame, n_boot: int = 500, seed: int = 0, ci_a: bool = False) -> dict:
    """レース単位の指標。単勝の払戻は確定オッズ × 100 円で近似。"""
    rid = d["race_id"].to_numpy()
    top = d.sort_values(["race_id", "p", "horse_number"], ascending=[True, False, True]).groupby("race_id").head(1).set_index("race_id")
    fav = d.sort_values(["race_id", "q", "horse_number"], ascending=[True, False, True]).groupby("race_id").head(1).set_index("race_id")
    w = d[d["win"] == 1].drop_duplicates("race_id").set_index("race_id")
    races = top.index
    ll_m, ll_q = -np.log(w.loc[races, "p"].clip(1e-6)).to_numpy(), -np.log(w.loc[races, "q"].clip(1e-6)).to_numpy()
    ret_m = (top["win"] * top["odds"] * 100).to_numpy()
    ret_q = (fav.loc[races, "win"] * fav.loc[races, "odds"] * 100).to_numpy()
    agree = (top["horse_number"].to_numpy() == fav.loc[races, "horse_number"].to_numpy())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(races), size=(n_boot, len(races)))
    dll = ll_m - ll_q
    dr = (ret_m - ret_q) / 100
    a, b = benter_a(d) if len(races) >= 100 else (np.nan, np.nan)
    a_lo, a_hi = benter_ci(d) if (ci_a and len(races) >= 100) else (np.nan, np.nan)
    return {"races": int(len(races)), "ll_model": float(ll_m.mean()), "ll_market": float(ll_q.mean()), "d_ll": float(dll.mean()),
            "d_ll_lo": float(np.quantile(dll[idx].mean(1), 0.025)), "d_ll_hi": float(np.quantile(dll[idx].mean(1), 0.975)),
            "top1_win": float(top["win"].mean()), "fav_win": float(fav.loc[races, "win"].mean()), "top1_is_fav": float(agree.mean()),
            "roi_top1": float(ret_m.mean() / 100), "roi_fav": float(ret_q.mean() / 100), "d_roi": float(dr.mean()),
            "d_roi_lo": float(np.quantile(dr[idx].mean(1), 0.025)), "d_roi_hi": float(np.quantile(dr[idx].mean(1), 0.975)),
            "roi_top1_not_fav": float(ret_m[~agree].mean() / 100) if (~agree).any() else None, "n_not_fav": int((~agree).sum()),
            "benter_a": a, "benter_b": b, "benter_a_lo": a_lo, "benter_a_hi": a_hi}


def method_check(live: Path, back: Path, res: pd.DataFrame) -> dict:
    """同じモデル・同じレースの当日の予測と後から作った予測。"""
    a, b = read_dir(live), read_dir(back)
    m = a.merge(b, on=["race_id", "horse_id"], suffixes=("_live", "_back"))
    if m.empty:
        return {"rows": 0}
    m = m.merge(res[["race_id", "horse_id", "horse_number"]], on=["race_id", "horse_id"], how="inner")
    t1 = m.sort_values(["race_id", "p_win_live", "horse_number"], ascending=[True, False, True]).groupby("race_id").head(1).set_index("race_id")["horse_number"]
    t2 = m.sort_values(["race_id", "p_win_back", "horse_number"], ascending=[True, False, True]).groupby("race_id").head(1).set_index("race_id")["horse_number"]
    diff = (m["p_win_live"] - m["p_win_back"]).abs()
    days = sorted({str(x)[:6] for x in m["race_id"]})            # race_id の先頭（年・場）。日付ではない
    return {"rows": int(len(m)), "races": int(m["race_id"].nunique()), "corr": float(m["p_win_live"].corr(m["p_win_back"])),
            "mean_abs_diff": float(diff.mean()), "p95_abs_diff": float(diff.quantile(0.95)), "max_abs_diff": float(diff.max()),
            "top1_agree": float((t1 == t2.reindex(t1.index)).mean()), "race_id_year_venue": days[:5],
            "dirs_live_days": sorted(p.name for p in live.iterdir() if p.is_dir() and any((back / p.name).glob("*.parquet")))}


def attr_missing(res: pd.DataFrame, days_glob: str = "2026-*") -> pd.DataFrame:
    """推論の日ごとの特徴量（state/<日>/v4_day.parquet。v4・v5・v6e の推論が共通に使う V4Features の馬自身の群）から、
    走った馬（確定オッズのある馬）のうち『産地の列が欠けている馬の割合』をレースごとに。日付はフォルダの名前（state/<YYYY-MM-DD>）。"""
    fs = sorted((INF / "state").glob(f"{days_glob}/v4_day.parquet"))
    if not fs:
        return pd.DataFrame(columns=["race_id", "date", "attr_na"])
    d = pd.concat([pd.read_parquet(f, columns=["race_id", "horse_id", "attr_birthplace"]).assign(date=f.parent.name) for f in fs], ignore_index=True)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    d = d.merge(res[["race_id", "horse_id"]], on=["race_id", "horse_id"], how="inner")
    return d.groupby(["race_id", "date"])["attr_birthplace"].apply(lambda x: float(x.isna().mean())).rename("attr_na").reset_index()


def attrs_coverage(res: pd.DataFrame) -> pd.DataFrame:
    """産地・誕生日の表（factory/self/horse_attrs.parquet。推論の V4Features が読む）に、走った馬の産地が無い割合（年月ごと）。"""
    import os

    from src.pipeline.features import serving_v4 as sv4

    a = pd.read_parquet(sv4.ATTRS_PATH, columns=["horse_id", "birthplace"])
    have = set(a.loc[a["birthplace"].notna(), "horse_id"].astype(str))
    r = res.assign(ym=res["date"].str[:7])
    out = r.groupby("ym")["horse_id"].apply(lambda x: float(1 - x.isin(have).mean())).rename("not_in_table").reset_index()
    out["table"] = str(sv4.ATTRS_PATH)
    out["table_mtime"] = pd.Timestamp(os.path.getmtime(sv4.ATTRS_PATH), unit="s").strftime("%Y-%m-%d")
    return out


def newcomer_share(years=range(2016, 2027)) -> pd.DataFrame:
    """レースごとの『その年に初めて走った馬の割合』（race_result_flat の 2016 年〜、走った行で初出走の年を決める）。
    JRA の結果表だけなので、地方から移ってきた馬は JRA で初めて走った年が初出走になる。
    2025 年（学習側の表がそろう）と 2026 年（産地・誕生日の表が新しい馬を含まない）を、同じ物差しで層に分けるため。"""
    fs = []
    for y in years:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            fs.append(pd.read_parquet(f, columns=["race_id", "date", "horse_id", "odds"]))
    r = pd.concat(fs, ignore_index=True)
    r["horse_id"], r["race_id"] = r["horse_id"].astype(str), r["race_id"].astype(str)
    r["y"] = pd.to_datetime(r["date"]).dt.year
    r = r[pd.to_numeric(r["odds"], errors="coerce") > 0]                 # 走った行だけで初出走の年を決める（取消だけの年は数えない）
    first = r.groupby("horse_id")["y"].min()
    r["new"] = (r["horse_id"].map(first) == r["y"]).astype(float)
    return r.groupby("race_id")["new"].mean().rename("new_share").reset_index()


def new_band(x: float) -> str:
    return "初出走の年の馬 10% 未満" if x < 0.1 else ("初出走の年の馬 10〜49%" if x < 0.5 else "初出走の年の馬 50% 以上")


def attr_band(x: float) -> str:
    return "産地の欠け 0%" if x == 0 else ("産地の欠け 1〜49%" if x < 0.5 else "産地の欠け 50% 以上")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(B16 / "tp_rank/cond/v6e_2026_gap"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = results([2025, 2026])
    rows, frames = [], {}
    for name in SOURCES:
        d = race_frame(load_source(name), res)
        d = d[~d["date"].isin(EXCLUDED)]
        frames[name] = d
        rows.append({"source": name, "period": "all", **summary(d, ci_a=True)})
        if name.endswith("2026") or name.endswith("2025"):          # 2025 年（学習側の経路）も月別に見て、もともとの落ち込みと比べる（月別の a は区間を付けない。各約 250 レースでぶれが大きい）
            for mth, g in d.groupby(d["date"].str[:7]):
                if g["race_id"].nunique() >= 50:
                    rows.append({"source": name, "period": mth, **summary(g)})
    extras: dict = {}
    # 同じレースにそろえた比較（対の差の区間つき）
    pairs = {"prod_backfill_vs_2016_backfill": ("v6e_prod_backfill2026", "v6e_2016_backfill2026"),
             "prod_live_vs_2016_backfill": ("v6e_prod_live2026", "v6e_2016_backfill2026"),
             "prod_test2025_vs_2016_test2025": ("v6e_prod_test2025", "v6e_2016_test2025")}
    for key, (x, y) in pairs.items():
        common = set(frames[x]["race_id"]) & set(frames[y]["race_id"])
        fx, fy = frames[x][frames[x]["race_id"].isin(common)], frames[y][frames[y]["race_id"].isin(common)]
        for n, f in ((x, fx), (y, fy)):
            rows.append({"source": n, "period": f"common:{key}", **summary(f, ci_a=len(common) >= 1000)})
        extras[f"paired_ll:{key}"] = paired(fx, fy)
    # 産地・誕生日の表の欠け（走った馬のうち表に無い割合、年月ごと）と、推論の特徴量の欠け（月ごと）
    cov = attrs_coverage(res)
    cov.to_csv(out / "attrs_coverage.csv", index=False)
    extras["attrs_not_in_table_by_month"] = {r.ym: round(r.not_in_table, 3) for r in cov.itertuples()}
    am = attr_missing(res).rename(columns={"date": "state_date"})
    am.to_csv(out / "attr_missing.csv", index=False)
    extras["serving_attr_na_by_month"] = {k: round(float(v), 3) for k, v in am.groupby(am["state_date"].str[:7])["attr_na"].mean().items()}
    for name in ("v6e_prod_backfill2026", "v6e_2016_backfill2026", "v4_backfill2026"):
        d = frames[name].merge(am, on="race_id", how="inner")
        for band, g in d.groupby(d["attr_na"].map(attr_band)):
            rows.append({"source": name, "period": f"attr:{band}", **summary(g)})
            for half, h in g.groupby(np.where(g["date"] < "2026-06-01", "1〜5月", "6月〜")):
                if h["race_id"].nunique() >= 100:
                    rows.append({"source": name, "period": f"attr:{band}:{half}", **summary(h)})
    # 新しい馬の割合の層（2025 年と 2026 年を同じ物差しで）と、差の差
    ns = newcomer_share()
    nb = {}
    for name in ("v6e_prod_test2025", "v6e_2016_test2025", "v6e_prod_backfill2026", "v6e_2016_backfill2026", "v4_backfill2026"):
        d = frames[name].merge(ns, on="race_id", how="inner")
        for band, g in d.groupby(d["new_share"].map(new_band)):
            rows.append({"source": name, "period": f"new:{band}", **summary(g, ci_a=True)})
        nb[name] = {"hi": race_ll(d[d["new_share"] >= 0.1]), "lo": race_ll(d[d["new_share"] < 0.1])}
    for m25, m26 in (("v6e_prod_test2025", "v6e_prod_backfill2026"), ("v6e_2016_test2025", "v6e_2016_backfill2026")):
        extras[f"did_newcomer:{m26}"] = did(nb[m25]["hi"], nb[m25]["lo"], nb[m26]["hi"], nb[m26]["lo"])
        extras[f"lo_band_2026_minus_2025:{m26}"] = two_sample(nb[m26]["lo"], nb[m25]["lo"])
    checks = {k: method_check(live, back, res) for k, (live, back) in CHECKS.items()}
    df = pd.DataFrame(rows)
    df.to_csv(out / "gap.csv", index=False)
    (out / "checks.json").write_text(json.dumps({"method_checks": checks, **extras}, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(json.dumps({"method_checks": checks, **extras}, ensure_ascii=False, indent=1))
    print(df[["source", "period", "races", "d_ll", "d_ll_lo", "d_ll_hi", "top1_is_fav", "roi_top1", "roi_fav", "d_roi", "benter_a", "benter_a_lo", "benter_a_hi"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
