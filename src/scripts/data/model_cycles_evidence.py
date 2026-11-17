"""第 2 段（市場とのずれを学習するモデル）の「事象と根拠」を実データで示す（2026-11-17、ユーザーの方針「すべての分析に事象と根拠を」）。

    python -m src.scripts.data.model_cycles_evidence NAME [--pred PATH] [--full FULLNAME]

- NAME: 出力の名前。予測は market/oof/NAME.parquet（開発の検証の OOF）か、--pred で渡した parquet（例: 全期間の 2016〜2024 学習の 2025 年 test）。
- ◎ は p≥0.05 の中でずれ r=log(p/q) の 1 位（r_p05。選び方は 2022 年の OOF で決めた）。
- 注意: q は確定オッズの市場の勝率（mk.frame の q_raw）。買う時点（T-60 など）の q ではないので、回収率と「勝率 ÷ q」は実際より良く出うる。
  発走前のオッズでの効きは 2026 年の推論の入力（apply2026 の t60・t15・t05）で別に測る。
- 出す表（market/evidence/NAME.json）:
  1. ◎ の人気帯・クラス・頭数・年ごとの件数・勝率・q の平均・「勝率 ÷ q」（= Σ勝ち ÷ Σq）と、その 95% 区間（レースを単位にした再抽出）・単勝の回収率と区間
     年は「選んだ年（2022）」と「確かめる年」を分けて記す。区分の列が欠けて表から落ちた ◎ の数も記す。
  2. 同じ人気帯の全馬の「勝率 ÷ q」（市場が人気帯の平均で当たっているかの基準）
  3. ずれ r の 10 分位（全馬）の「勝率 ÷ q」と区間（レースを単位にした再抽出）
  4. --full: 全期間のモデル（offset のモデルに限る）の特徴量の重要度（total_gain を seed ごとに割合にして平均）。分割の多い連続の列ほど大きく出る点に注意
- GCS は読まない。
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.scripts.data import model_cycles as mc
from src.scripts.data import model_cycles_market as mk

KEY = ["race_id", "horse_id"]
SELECT_YEAR = 2022
POP_BINS, POP_LABELS = [0, 1, 2, 3, 5, 9, 99], ["1", "2", "3", "4-5", "6-9", "10+"]


def race_boot_ratio(g: pd.DataFrame, num: str, den: str, n_boot: int = 1000, seed: int = 0) -> list:
    """Σnum ÷ Σden の点推定と 95% 区間。レースを単位に再抽出する（同じレースの馬は互いに相関するため）。"""
    a = g.groupby("race_id")[[num, den]].sum()
    if len(a) == 0 or a[den].sum() <= 0:
        return [None, None, None]
    x, y = a[num].to_numpy(float), a[den].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    bs = x[idx].sum(1) / y[idx].sum(1)
    return [float(x.sum() / y.sum()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def group_table(sel: pd.DataFrame, by: str) -> list[dict]:
    out = []
    for k, g in sel.groupby(by, observed=True):
        n = len(g)
        hit = g[g["win"] == 1]
        out.append({by: str(k), "n": int(n), "win": float(g["win"].mean()), "q_mean": float(g["q"].mean()),
                    "win_over_q": race_boot_ratio(g, "win", "q"), "roi_tan": race_boot_ratio(g.assign(stake=100.0), "ret_tan", "stake"),
                    "avg_pay_when_hit": float(hit["ret_tan"].mean()) if len(hit) else None, "place": float((g["finish_position"] <= 3).mean())})
    return out


def class_group(cls: pd.Series) -> np.ndarray:
    """race_class の区分。重賞・特別は『オープン』の中に入る（この列では分けられない）。空・欠けは『クラス不明』。"""
    c = cls.fillna("").astype(str)
    return np.select([c.str.contains("新馬"), c.str.contains("未勝利"), c.str.contains("１勝|1勝"), c.str.contains("２勝|2勝"), c.str.contains("３勝|3勝"),
                      c.str.contains("オープン")], ["新馬", "未勝利", "1勝", "2勝", "3勝", "オープン（重賞・特別を含む）"], "クラス不明")


def importance(full: str) -> dict:
    import xgboost as xgb

    fd = mk.MK / "full" / full
    fm = json.loads((fd / "meta.json").read_text(encoding="utf-8"))
    spec = fm.get("spec", {})
    if spec.get("offset") in (None, "none"):
        raise SystemExit(f"{full} は offset なしのモデル（市場からのずれではなく勝率そのものを学ぶ）。重要度を『ずれの説明』として出さない")
    shares: list[dict[str, float]] = []
    for s in range(fm["seeds"]):
        b = xgb.XGBClassifier()
        b.load_model(str(fd / f"win_s{s}.json"))
        sc = b.get_booster().get_score(importance_type="total_gain")
        z = sum(sc.values())
        shares.append({k: v / z for k, v in sc.items()})
    cols = set().union(*shares)
    avg = {c: float(np.mean([sh.get(c, 0.0) for sh in shares])) for c in cols}
    return {"model": full, "offset": spec.get("offset"), "offset_noise_sd": spec.get("offset_noise_sd"), "seeds": fm["seeds"],
            "note": "total_gain を seed ごとに割合にして平均。分割に使われやすい連続の列ほど大きく出るので、列の効きの強さの目安にとどめる",
            "top30": [{"col": k, "share": v} for k, v in sorted(avg.items(), key=lambda x: -x[1])[:30]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--pred", default="", help="予測の parquet（race_id・horse_id・p_win）。省略時は market/oof/NAME.parquet")
    ap.add_argument("--full", default="")
    a = ap.parse_args()
    if a.pred:
        pred = pd.read_parquet(a.pred)
        pred["race_id"], pred["horse_id"] = pred["race_id"].astype(str), pred["horse_id"].astype(str)
        pred = pred[[c for c in pred.columns if c in KEY or c.startswith("p_")]]   # 着順・オッズは frame が市場の表から付ける
        mk.read_any = lambda _n: pred.copy()          # frame は read_any で予測を読む
    d = mk.frame(a.name)                                  # 走った馬がそろうレース、q は確定の市場の勝率（q_raw）、r = log(p/q)
    meta = pd.read_parquet(mc.CYC / "meta.parquet", columns=KEY + ["race_class", "field_size"])
    meta["race_id"], meta["horse_id"] = meta["race_id"].astype(str), meta["horse_id"].astype(str)
    d = d.merge(meta.drop_duplicates(KEY), on=KEY, how="left")
    d["year"] = d["race_id"].str[:4].astype(int)
    d["win"] = (d["finish_position"] == 1).astype(float)    # 着順の欠け（競走中止など）は外れ
    d["pop_band"] = pd.cut(d["popularity"], POP_BINS, labels=POP_LABELS)
    sel = mk.pick(d, "r_p05").copy()
    sel["class_grp"] = class_group(sel["race_class"])
    sel["field_band"] = pd.cut(sel["field_size"], [0, 10, 14, 18], labels=["〜10 頭", "11〜14 頭", "15〜18 頭"])
    sel["year_role"] = np.where(sel["year"] == SELECT_YEAR, f"{SELECT_YEAR}（選び方を決めた年）", sel["year"].astype(str) + "（選び方 r_p05 だけの確かめ。offset・seed の採用はこの年も見た）")
    n = len(sel)
    res = {"name": a.name, "pred": a.pred or f"market/oof/{a.name}.parquet", "races": int(sel["race_id"].nunique()),
           "years": sorted(int(y) for y in sel["year"].unique()),
           "rule": "r_p05（p≥0.05 の中でずれ r=log(p/q) の 1 位。選び方は 2022 年の OOF で決めた）",
           "caveat": "外に出した確かめは --pred の 2025 年 test だけ（OOF の 2022〜2024 は offset・seed の採用に使った標本）。q は確定オッズの市場の勝率。買う時点の q ではないので、回収率と勝率 ÷ q は実際より良く出うる。区間はレースを単位にした再抽出",
           "overall": group_table(sel.assign(all="全体"), "all")}
    for by in ("pop_band", "class_grp", "field_band", "year_role"):
        res[by] = group_table(sel, by)
        res[f"{by}_dropped"] = int(n - sum(r["n"] for r in res[by]))   # 区分の列が欠けて表から落ちた ◎ の数
    res["all_horses_by_pop_band"] = {str(k): {"n": int(len(g)), "win": float(g["win"].mean()), "q_mean": float(g["q"].mean()),
                                              "win_over_q": race_boot_ratio(g, "win", "q"), "roi_tan": float(g["ret_tan"].sum() / (100 * len(g)))}
                                     for k, g in d.groupby("pop_band", observed=True)}
    d["dec"] = pd.qcut(d["r"].rank(method="first"), 10, labels=False)
    res["r_deciles"] = [{"decile": int(k), "n": int(len(g)), "r_mean": float(g["r"].mean()), "win": float(g["win"].mean()), "q_mean": float(g["q"].mean()),
                         "win_over_q": race_boot_ratio(g, "win", "q"), "roi_tan": float(g["ret_tan"].sum() / (100 * len(g)))} for k, g in d.groupby("dec")]
    if a.full:
        res["importance"] = importance(a.full)
    out = mk.MK / "evidence"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{a.name}.json"
    prev = json.loads(dst.read_text(encoding="utf-8")).get("pred") if dst.exists() else None
    if prev is not None and prev != res["pred"]:
        raise SystemExit(f"{dst} は別の予測（{prev}）の結果。名前を変える（例: {a.name}__test2025）")
    dst.write_text(json.dumps(mk._no_nan(res), ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    print(f"書き出し: {out / (a.name + '.json')}（{res['races']} レース）")


if __name__ == "__main__":
    sys.exit(main())
