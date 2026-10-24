"""予測値のレンジ別の成績（較正表）。予測確率の帯ごとに、頭数・平均予測・実際の発生率・差・95% 信頼区間と、単勝・複勝の回収率（参考）を出す。

python -m src.scripts.data.accuracy_calibration
入力: final_eval_v3|v4/test2025_predictions.parquet（test 2025、2020〜2024 学習・一度だけの評価）、
      tp27/ens/v4_e2.parquet（v4 の valid 2022〜2024、leave-one-year-out）、v3 は ensemble_v3 の spec と各メンバーの OOF から合成
      確定結果は data/page_reference/tables/<年>/race_result_flat.parquet（着順・単勝/複勝の払戻）
出力: data/local/modeling/training_pipeline/accuracy_summary/calibration.json
オッズは特徴量に使っていない。回収率は『その帯の全馬を毎回 100 円買う』場合の参考値で、買い目戦略の評価ではない。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "data/local/modeling/training_pipeline"
V4 = ROOT / "data/local/modeling/training_pipeline_v4"
OUT = OLD / "accuracy_summary"
KEY = ["race_id", "horse_id"]
EDGES = {
    "win": [0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0001],
    "top2": [0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0001],
    "top3": [0, 0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 1.0001],
}
K = {"win": 1, "top2": 2, "top3": 3}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def _int(s: str) -> int | None:
    s = s.replace(",", "").strip()
    return int(s) if s.isdigit() else None


def _split_digits(s: str, n: int) -> list[int] | None:
    """複勝の払戻は桁をつなげた文字列（例 '130110140'）。n 個に分け、各 100〜99999 で分け方が一意のときだけ返す。"""
    s = s.replace(",", "").strip()
    found: list[list[int]] = []

    def rec(i: int, left: int, acc: list[int]) -> None:
        if len(found) > 1:
            return
        if left == 0:
            if i == len(s):
                found.append(list(acc))
            return
        for ln in (3, 4, 5):
            if i + ln <= len(s) and s[i] != "0":
                v = int(s[i:i + ln])
                if v % 10 == 0 and v >= 100:
                    rec(i + ln, left - 1, acc + [v])

    rec(0, n, [])
    return found[0] if len(found) == 1 else None


def load_results(years: list[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        d = pd.read_parquet(ROOT / f"data/page_reference/tables/{y}/race_result_flat.parquet", columns=["race_id", "horse_id", "horse_number", "finish_position", "payoff"])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    d["finish_position"] = pd.to_numeric(d["finish_position"], errors="coerce")
    d["horse_number"] = pd.to_numeric(d["horse_number"], errors="coerce")
    return d


def _items(x) -> list[tuple[int, int]]:
    """払戻の項目（リスト形式: 馬ごとの {numbers, payout}）を [(馬番, 払戻円)] にする。"""
    out = []
    for it in x if isinstance(x, list) else []:
        n, v = str(it.get("numbers", "")).strip(), _int(str(it.get("payout", "")))
        if n.isdigit() and v:
            out.append((int(n), v))
    return out


def payouts(res: pd.DataFrame) -> tuple[dict, dict]:
    """単勝: {(race_id, 馬番): 払戻円}、複勝: {(race_id, 馬番): 払戻円}（取れたレースだけ）。
    払戻はリスト形式と、桁をつなげた文字列形式（複勝）がある。後者は着順どおりの 1〜3 着の馬に割り当て、分け方が一意のときだけ使う。"""
    tan, fuku = {}, {}
    for rid, g in res.groupby("race_id", sort=False):
        raw = g["payoff"].iloc[0]
        if raw is None:
            continue
        try:
            j = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        t = j.get("単勝")
        if isinstance(t, list):
            for n, v in _items(t):
                tan[(rid, n)] = v
        elif isinstance(t, dict):
            nums, pay = str(t.get("numbers", "")).split(), _int(str(t.get("payout", "")))
            if len(nums) == 1 and nums[0].isdigit() and pay:
                tan[(rid, int(nums[0]))] = pay
        f = j.get("複勝")
        if isinstance(f, list):
            for n, v in _items(f):
                fuku[(rid, n)] = v
        elif isinstance(f, dict):
            placed = g[(g["finish_position"] >= 1) & (g["finish_position"] <= 3)].sort_values("finish_position")
            for cnt in (3, 2):
                if len(placed) == cnt and placed["finish_position"].tolist() == list(range(1, cnt + 1)):
                    pays = _split_digits(str(f.get("payout", "")), cnt)
                    digits = "".join(str(int(x)) for x in placed["horse_number"])
                    if pays and digits == str(f.get("numbers", "")).replace(" ", ""):
                        for hn, pv in zip(placed["horse_number"], pays):
                            fuku[(rid, int(hn))] = pv
                        break
    return tan, fuku


def table(P: pd.DataFrame, tan: dict, fuku: dict, ok_tan: set, ok_fuku: set) -> dict:
    out = {}
    for t, k in K.items():
        p = P[f"p_{t}"].to_numpy()
        y = (P["finish_position"].to_numpy() <= k).astype(int)
        bins = pd.cut(p, EDGES[t], right=False, include_lowest=True)
        rows = []
        for iv in bins.categories:
            m = np.asarray(bins == iv)
            n = int(m.sum())
            if n == 0:
                continue
            kk = int(y[m].sum())
            lo, hi = wilson(kk, n)
            sub = P[m]
            # 回収率: その帯の全馬に毎回 100 円。払戻が取れたレースの馬だけで集計
            rt = sub[sub["race_id"].isin(ok_tan)]
            tan_ret = float(sum(tan.get((r, int(h)), 0) for r, h in zip(rt["race_id"], rt["horse_number"]))) / (100 * len(rt)) if len(rt) else None
            rf = sub[sub["race_id"].isin(ok_fuku)]
            fuku_ret = float(sum(fuku.get((r, int(h)), 0) for r, h in zip(rf["race_id"], rf["horse_number"]))) / (100 * len(rf)) if len(rf) else None
            rows.append({"lo": float(iv.left), "hi": float(min(iv.right, 1.0)), "n": n, "mean_pred": float(p[m].mean()), "actual": kk / n, "diff": float(p[m].mean() - kk / n),
                         "ci_lo": lo, "ci_hi": hi, "tan_roi": tan_ret, "tan_n": int(len(rt)), "fuku_roi": fuku_ret, "fuku_n": int(len(rf))})
        out[t] = rows
    return out


def v3_valid_oof() -> pd.DataFrame | None:
    """v3 の valid 2022〜2024: メンバーの OOF を ensemble_v3 の spec で合成（EnsembleBundle.predict と同じ合成式）。"""
    try:
        from src.pipeline.models import ensemble as en
        from src.pipeline.models import win_models as wm

        spec = json.loads((OLD / "ensemble_v3/ensemble_spec.json").read_text(encoding="utf-8"))
        names = spec["models"]
        base = None
        per = {}
        for n in names:
            o = pd.read_parquet(OLD / f"oof/model_oof/{n}_tl3.parquet")
            o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
            per[n] = o.set_index(KEY)
            base = per[n].index if base is None else base.intersection(per[n].index)
        race = np.array([i[0] for i in base])
        out = {}
        for t, total in (("win", 1.0), ("top2", 2.0), ("top3", 3.0)):
            Pm = np.stack([per[n].loc[base, f"p_{t}"].to_numpy() for n in names], axis=1)
            out[f"p_{t}"] = wm.normalize_to_total(race, en.apply_spec(spec["targets"][t], Pm), total)
        d = pd.DataFrame(out, index=base).reset_index()
        d.columns = KEY + list(out.keys())
        return d
    except Exception as e:  # noqa: BLE001
        _log(f"v3 valid の合成に失敗: {type(e).__name__}: {e}")
        return None


def main() -> None:
    t0 = time.time()
    res = load_results([2022, 2023, 2024, 2025])
    tan, fuku = payouts(res)
    ok_tan = {r for r, _ in tan}      # 払戻を取れたレース（そのレースの全馬が集計対象）
    ok_fuku = {r for r, _ in fuku}
    races = res["race_id"].nunique()
    cover = {"races": int(races), "tan_races": len(ok_tan), "fuku_races": len(ok_fuku)}
    _log(f"払戻の取得: {cover}")
    fin = res[KEY + ["horse_number", "finish_position"]]
    sets = {}
    for ver in ("v3", "v4"):
        T = pd.read_parquet(OLD / f"final_eval_{ver}/test2025_predictions.parquet")[KEY + ["p_win", "p_top2", "p_top3"]]
        T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
        sets[(ver, "test")] = T.merge(fin, on=KEY, how="inner")
    V4v = pd.read_parquet(V4 / "tp27/ens/v4_e2.parquet")[KEY + ["p_win", "p_top2", "p_top3"]]
    V4v["race_id"], V4v["horse_id"] = V4v["race_id"].astype(str), V4v["horse_id"].astype(str)
    sets[("v4", "valid")] = V4v.merge(fin, on=KEY, how="inner")
    V3v = v3_valid_oof()
    if V3v is not None:
        sets[("v3", "valid")] = V3v.merge(fin, on=KEY, how="inner")
    rep = {"payout_coverage": cover, "edges": EDGES, "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tables": {}}
    for (ver, split), P in sets.items():
        P = P[np.isfinite(P["finish_position"].to_numpy())]
        from sklearn.metrics import roc_auc_score

        met = {}
        for tg, kk in K.items():
            yy = (P["finish_position"].to_numpy() <= kk).astype(float)
            pp = np.clip(P[f"p_{tg}"].to_numpy(), 1e-9, 1 - 1e-9)
            met[tg] = {"logloss": float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))), "auc": float(roc_auc_score(yy, pp))}
        rep["tables"][f"{ver}_{split}"] = {"rows": int(len(P)), "races": int(P["race_id"].nunique()), "metrics": met, "bins": table(P, tan, fuku, ok_tan, ok_fuku)}
        _log(f"{ver} {split}: {len(P):,} 行")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"完了 {time.time() - t0:.0f}s → {OUT / 'calibration.json'}")


if __name__ == "__main__":
    main()
