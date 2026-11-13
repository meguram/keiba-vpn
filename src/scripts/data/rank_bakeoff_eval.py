"""rank_bakeoff の評価: 並びの指標と、予測 1 番手を買ったときの回収率（4 種類）を、基準（v6e のアンサンブル）と比べる。

python -m src.scripts.data.rank_bakeoff_eval [--years 2022,2023,2024] [--names A1_... ...] [--boot 1000] [--sub oof|oof_s3|ens|test2025]
（RANK_BASE=2016 で 2016 年起点の土台。--sub: oof=1 seed、oof_s3=3 seed、ens=v6e に足したアンサンブル、test2025=2025 年の 1 回きりの確認）

回収率（ユーザー指定 2026-11-13）:
- 単勝: 予測 1 番手を 100 円
- 複勝: 予測 1 番手を 100 円
- 馬連の流し: 予測 1 番手 → 2・3 番手（2 点・200 円）
- ワイドの流し: 予測 1 番手 → 2・3 番手（2 点・200 円）
回収率 = 払戻の合計 ÷ 購入金額の合計。同着で払戻が複数あれば当たった分をすべて足す。払戻の無いレースは除く。
予測の順位は p_win の大きい順（同じ値は馬番の小さい順）。OOF は出走した馬だけなので取消の返還は起きない。

並びの指標: 1 番手の勝率・1 番手の複勝率・上位 3 頭のうち 3 着以内の頭数・NDCG@5（関連度 6-着順、6 着以下 0、指数の利得）。
区間はレース単位のブートストラップ（差の 95% 区間）。相手は同じ seed 数の基準（matched_ref。LightGBM ほかは R0、XGBoost は B0、
調整の変種は元の設定の候補）と、v6e のアンサンブル（test2025 では final_eval_v6e16 の 2025 年の予測）。
出力: tp_rank/eval/<sub>/summary.json と summary.md。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
# RANK_BASE=2016 なら 2016〜2025 年の土台（retrain_out/base_2016）の OOF・基準（段 9 の v6e）を読む
V = (Path(os.environ.get("REBUILD_BASE") or "/home/jovyan/work/retrain_out/base_2016") if os.environ.get("RANK_BASE") == "2016"
     else ROOT / "data/local/modeling/training_pipeline_v4")
RANK = V / "tp_rank"
BASELINE = {"v6e_ens": V / "tp27_v6e/ens/v6.parquet"}
V6E_MEMBERS = ["lgb_extras_ped", "xgb_extras_ped", "lgb_extras", "lgb_softmax", "linear", "mtl_nn_extras_ped", "mtl_nn_s"]
PAYOFF = ROOT / "data/local/modeling/betting/payoff.parquet"
BETS = ["tansho", "fukusho", "umaren", "wide"]
COST = {"tansho": 100, "fukusho": 100, "umaren": 200, "wide": 200}
METRICS = ["top1_win", "top1_place", "top3_in3", "ndcg5"] + [f"roi_{b}" for b in BETS]


def _pair(a: int, b: int) -> str:
    x, y = sorted((int(a), int(b)))
    return f"{x}-{y}"


def load_payoff(path: Path = PAYOFF) -> dict[tuple[str, str], dict[str, int]]:
    """{(race_id, bet_type): {combo: 払戻（100 円あたり）}}。馬連・ワイドの組は馬番の昇順 a-b（数として並べる）。
    同着は組ごとに別の行なので、当たった組をすべて足せば同着の払戻も入る。"""
    pay = pd.read_parquet(path, columns=["race_id", "bet_type", "combo", "payout"])
    pay = pay[pay["bet_type"].isin(BETS)]
    out: dict[tuple[str, str], dict[str, int]] = {}
    for rid, bt, combo, v in zip(pay["race_id"].astype(str), pay["bet_type"], pay["combo"].astype(str), pay["payout"]):
        if bt in ("umaren", "wide"):
            a, b = combo.split("-")
            combo = _pair(int(a), int(b))
        out.setdefault((rid, bt), {})[combo] = int(v)
    return out


def ndcg_at(fin_by_pred: np.ndarray, fin_all: np.ndarray, k: int = 5) -> float:
    """予測順に並べた着順 fin_by_pred と、そのレースの全着順 fin_all から NDCG@k（関連度 6-着順、利得 2^rel-1）。"""
    rel = lambda f: np.clip(6 - np.where(np.isfinite(f), f, 99), 0, None)   # noqa: E731
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    g = (2 ** rel(fin_by_pred[:k]) - 1) @ disc[: min(k, len(fin_by_pred))]
    ideal = np.sort(rel(fin_all))[::-1][:k]
    ig = (2 ** ideal - 1) @ disc[: len(ideal)]
    return float(g / ig) if ig > 0 else np.nan


def race_table(oof: pd.DataFrame, keys: pd.DataFrame, pay: dict) -> pd.DataFrame:
    """レースごとの指標と、4 券種の購入額・払戻。oof: race_id, horse_id, p_win。"""
    d = oof[["race_id", "horse_id", "p_win"]].merge(keys, on=["race_id", "horse_id"], how="inner")
    d = d.sort_values(["race_id", "p_win", "horse_number"], ascending=[True, False, True], kind="stable")
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        if len(g) < 3:
            continue
        hn = g["horse_number"].to_numpy().astype(int)
        fin = g["finish_position"].to_numpy().astype("float64")
        r = {"race_id": rid, "year": int(g["year"].iloc[0]), "top1_win": float(fin[0] == 1), "top1_place": float(fin[0] <= 3),
             "top3_in3": float(np.sum(fin[:3] <= 3)), "ndcg5": ndcg_at(fin, fin)}
        have = all((rid, bt) in pay for bt in BETS)       # 4 券種の払戻がそろったレースだけ回収率に入れる
        r["has_pay"] = have
        if have:
            r["ret_tansho"] = pay[(rid, "tansho")].get(str(hn[0]), 0)
            r["ret_fukusho"] = pay[(rid, "fukusho")].get(str(hn[0]), 0)
            for bt in ("umaren", "wide"):
                tbl = pay[(rid, bt)]
                r[f"ret_{bt}"] = tbl.get(_pair(hn[0], hn[1]), 0) + tbl.get(_pair(hn[0], hn[2]), 0)
            for bt in BETS:
                r[f"cost_{bt}"] = COST[bt]
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(t: pd.DataFrame) -> dict:
    s = {m: float(t[m].mean()) for m in ("top1_win", "top1_place", "top3_in3", "ndcg5")}
    p = t[t["has_pay"]]
    for bt in BETS:
        s[f"roi_{bt}"] = float(p[f"ret_{bt}"].sum() / p[f"cost_{bt}"].sum())
    s["n_races"], s["n_pay"] = int(len(t)), int(len(p))
    return s


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """a − b（同じレースに揃えたうえでレース単位で再標本化）の 95% 区間。"""
    m = a.merge(b, on="race_id", suffixes=("_a", "_b"))
    rng = np.random.default_rng(seed)
    out = {}
    n = len(m)                                                   # 並びの指標は全レース（表の水準と同じ母集団）
    idx = rng.integers(0, n, size=(n_boot, n))
    for k in ("top1_win", "top1_place", "top3_in3", "ndcg5"):
        x = (m[f"{k}_a"] - m[f"{k}_b"]).to_numpy()
        bs = np.nanmean(x[idx], axis=1)
        out[k] = (float(np.nanmean(x)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
    m = m[m["has_pay_a"] & m["has_pay_b"]]                       # 回収率は 4 券種の払戻がそろったレースだけ
    n = len(m)
    idx = rng.integers(0, n, size=(n_boot, n))
    for bt in BETS:
        ra, rb, c = m[f"ret_{bt}_a"].to_numpy(), m[f"ret_{bt}_b"].to_numpy(), m[f"cost_{bt}_a"].to_numpy()
        bs = ra[idx].sum(1) / c[idx].sum(1) - rb[idx].sum(1) / c[idx].sum(1)
        out[f"roi_{bt}"] = (float(ra.sum() / c.sum() - rb.sum() / c.sum()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
    out["n"] = n
    return out


# 1 段目で比べる相手。候補は 1 seed なので、同じ 1 seed・同じ入力の勝ちの二値 R0_lgb_bin_win と比べる
# （v6e の単体メンバーは seed 0・1・2 の平均なので、そのまま比べると候補だけ不利になる）。
# R0 自体は v6e の lgb_extras_ped と比べ、入力が一致しているかの確認にする（seed の数だけ違う）。
STAGE1_REF = "R0_lgb_bin_win"
STAGE1_REF_XGB = "B0_xgb_bin_win"


def matched_ref(name: str) -> str | None:
    import re

    tuned = re.match(r"^([A-Z]\d)t\d+_(.+)$", name)
    if tuned:                                        # 2 段目の木の設定の調整（A6t1_… など）は、元の設定の同じ候補（同じ seed 0）と比べる
        return f"{tuned.group(1)}_{tuned.group(2)}"
    if name.startswith("v6e_"):
        return "v6e_ens"
    if name == STAGE1_REF:
        return "v6e_lgb_extras_ped"
    if name == STAGE1_REF_XGB:
        return "v6e_xgb_extras_ped"
    if name.startswith("B"):                         # XGBoost の候補は、同じ XGBoost の勝ちの二値 B0（1 seed）と比べる
        return STAGE1_REF_XGB
    return STAGE1_REF


# 2025 年（test）を予測・評価してよい最終候補（2025 年を見る前にコードに書いて commit する）。2026-11-13 の比較では B3・A6。
# 予測を作ってしまった他の名前（B3t2 など）は、評価でも読まない。
TEST2025_FINALISTS = ("B3_xgb_bin_top3", "A6_lgb_xendcg_top5")


def check_test2025_name(name: str) -> None:
    if name not in TEST2025_FINALISTS:
        raise SystemExit(f"{name} は 2025 年の最終候補（TEST2025_FINALISTS）に無い。先に一覧に足して commit してから予測する")


def average_seed_oofs(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """seed ごとの OOF（keys・p_win・p_top2・p_top3）を平均する。行（keys）が全部そろっていなければ止める。"""
    idx = frames[0].set_index(keys).index
    for f in frames[1:]:
        if not f.set_index(keys).index.equals(idx):
            raise ValueError("seed ごとの OOF の行がそろわない")
    cols = ["p_win", "p_top2", "p_top3"]
    out = sum(f.set_index(keys)[cols] for f in frames) / len(frames)
    return out.reset_index()


def candidates(names: list[str] | None, sub: str = "oof") -> dict[str, Path]:
    if sub == "test2025":                            # 2025 年の 1 回きりの確認: 基準は v6e の test 2025 の予測（2024 年までで学習）。単体メンバーは無い
        for n in names or []:
            if n != "v6e_ens":
                check_test2025_name(n)               # 一覧に無い名前は評価もしない
        c = {"v6e_ens": V / "final_eval_v6e16/test2025_predictions.parquet"}
        for p in sorted((RANK / sub).glob("*.parquet")):
            if p.stem in TEST2025_FINALISTS:          # 最終候補の一覧にあるものだけ（誤って作った予測は読まない）
                c[p.stem] = p
        keep = (set(names) | {"v6e_ens"} | {matched_ref(n) for n in names}) if names else None
        return {k: v for k, v in c.items() if keep is None or k in keep}
    c = dict(BASELINE)
    for m in V6E_MEMBERS:
        p = V / "tp27_v6e/oof" / f"{m}__v4.parquet"
        if p.exists():
            c[f"v6e_{m}"] = p
    for p in sorted((RANK / sub).glob("*.parquet")):
        c[p.stem] = p
    if names:
        keep = set(names) | {"v6e_ens"} | {matched_ref(n) for n in names}
        keep |= {matched_ref(n) for n in list(keep) if n}          # R0 を残したらその相手も残す
        c = {k: v for k, v in c.items() if k in keep}
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2022,2023,2024")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--sub", default="oof", help="tp_rank の下の oof / smoke")
    a = ap.parse_args()
    years = {2025} if a.sub == "test2025" else {int(y) for y in a.years.split(",")}   # test2025 は 2025 年だけ
    keys = pd.read_parquet(RANK / "keys.parquet")
    keys = keys[keys["year"].isin(years)]
    pay = load_payoff()
    tables = {}
    for name, p in candidates(a.names, a.sub).items():
        o = pd.read_parquet(p)
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        tables[name] = race_table(o, keys, pay)
    common = set.intersection(*(set(t["race_id"]) for t in tables.values()))
    n_full = len(set(tables["v6e_ens"]["race_id"]))
    for k, t in tables.items():                                   # fold の欠けた OOF があると全体の比較が一部の年だけになるので知らせる
        if len(set(t["race_id"])) < n_full:
            print(f"警告: {k} は {len(set(t['race_id']))} レース（v6e は {n_full}）。共通の {len(common)} レースだけで比べる")
    tables = {k: t[t["race_id"].isin(common)] for k, t in tables.items()}
    res = {}
    for name, t in tables.items():
        r = {"all": summarize(t), "by_year": {int(y): summarize(g) for y, g in t.groupby("year")}}
        ref = matched_ref(name)
        if ref == STAGE1_REF_XGB and ref not in tables and STAGE1_REF in tables:
            print(f"警告: {name} の相手 {STAGE1_REF_XGB} の OOF が無いので {STAGE1_REF} と比べる")
            ref = STAGE1_REF
        r["ref"] = ref if ref in tables else None
        if name != "v6e_ens":
            r["diff_vs_v6e"] = boot_diff(t, tables["v6e_ens"], a.boot)
        if r["ref"] and r["ref"] != "v6e_ens":
            r["diff_vs_ref"] = boot_diff(t, tables[r["ref"]], a.boot)
        res[name] = r
    out = RANK / "eval" / (a.sub if a.sub != "oof" else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    head = ["| 候補 | 比べる相手 | 1番手勝率 | 1番手複勝率 | 上位3頭の3着内 | NDCG@5 | 単勝 | 複勝 | 馬連流し | ワイド流し |", "|---|---|---|---|---|---|---|---|---|---|"]
    lines = [f"# rank_bakeoff 評価（{','.join(map(str, sorted(years)))}、共通 {len(common)} レース）", ""]
    for key, title in (("diff_vs_ref", "同じ seed 数の基準との差（候補は R0・B0、調整の変種は元の設定の候補。相手の列を参照）"), ("diff_vs_v6e", "v6e のアンサンブルとの差")):
        lines += [f"## {title}", ""] + head
        for name, r in sorted(res.items(), key=lambda kv: -kv[1]["all"]["ndcg5"]):
            s = r["all"]
            cells = [f"{s['top1_win']:.4f}", f"{s['top1_place']:.4f}", f"{s['top3_in3']:.4f}", f"{s['ndcg5']:.4f}"] + [f"{s[f'roi_{b}']:.3f}" for b in BETS]
            d = r.get(key)
            if d:
                cells = [f"{c}<br>({d[m][1]:+.4f}〜{d[m][2]:+.4f})" for c, m in zip(cells, METRICS)]
            ref = (r["ref"] or "－") if key == "diff_vs_ref" else "v6e_ens"
            lines.append(f"| {name} | {ref if d else '－'} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["括弧は差（候補 − 相手）の 95% 区間（レース単位のブートストラップ）。回収率は払戻 ÷ 購入額で、選ぶのには使わない（参考）。",
              "1 段目の候補は 1 seed。相手は同じ 1 seed・同じ入力の勝ちの二値（LightGBM ほかは R0、XGBoost は B0）。R0・B0 の行は v6e の lgb_extras_ped・xgb_extras_ped（3 seed）との差で、入力の一致の確認。"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
