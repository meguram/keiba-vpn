"""ラベルなしのドリフト検知（先行指標）: 結果が出る前に（出馬表確定・予測の直後に）出せる指標。

  予測確率の分布（勝率・複勝率・最大勝率・エントロピー・頭数別）を、基準期間（学習の OOF と test①）と比べる: PSI・KS・分位点・尖り。
  入力データの質: 新しい ID の出現率（初出走の馬・学習期間に出ていない騎手/調教師）、調教ページの被覆（drift_coverage）。

入力は予測の表（race_id, horse_id, p_win, p_top3, date, field_size, jockey_id, trainer_id）。着順・オッズは不要。
指標は報告・警報のみ。モデルや閾値の選択には使わない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import drift_metrics as M


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def band_field(n) -> pd.Series:
    return pd.cut(pd.to_numeric(n, errors="coerce"), [0, 9, 13, 16, 99], labels=["~9頭", "10-13頭", "14-16頭", "17頭~"]).astype(str)


def dist_metrics(g: pd.DataFrame) -> dict:
    """予測の分布の要約（1 窓ぶん）。"""
    if len(g) == 0:
        return {}
    mx = g.groupby("race_id")["p_win"].max()
    out = {"races": int(g["race_id"].nunique()), "rows": int(len(g)), "mean_top_p": float(mx.mean()), "entropy_rel": M.entropy_rel(g, "p_win"),
           "logit_sd": float(_logit(g["p_win"]).std()), "share_p_lt_001": float((g["p_win"] < 0.01).mean()),
           "mean_p_top3": float(g["p_top3"].mean()) if "p_top3" in g else float("nan")}
    if "field_size" in g:
        fb = band_field(g.drop_duplicates("race_id").set_index("race_id")["field_size"].reindex(mx.index))
        out["mean_top_p_by_field"] = {k: float(v) for k, v in mx.groupby(fb.to_numpy()).mean().items()}
    return out


def dist_shift(base: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """基準と今回の勝率・複勝率の分布比較（PSI は基準の分位で区切る。最大勝率はレースごと）。"""
    out = {}
    for col in ("p_win", "p_top3"):
        if col not in base or col not in cur:
            continue
        a, b = base[col].to_numpy(float), cur[col].to_numpy(float)
        out[col] = {"psi": M.psi_numeric(a, b), "ks": M.ks_stat(a, b), "mean_base": float(a.mean()), "mean_cur": float(b.mean()),
                    "q_base": M.quantile_table(a), "q_cur": M.quantile_table(b)}
    ma, mb = base.groupby("race_id")["p_win"].max().to_numpy(), cur.groupby("race_id")["p_win"].max().to_numpy()
    out["max_p_win"] = {"psi": M.psi_numeric(ma, mb), "ks": M.ks_stat(ma, mb), "mean_base": float(ma.mean()), "mean_cur": float(mb.mean())}
    return out


def new_id_rates(cur: pd.DataFrame, first_run: pd.Series, week_start: str, known_jockeys: set[str], known_trainers: set[str]) -> dict:
    """今回の出走馬のうち、初出走の馬（これまで走った記録が無い）と、学習期間に出ていない騎手・調教師の割合。"""
    h = cur["horse_id"].astype(str)
    fr = h.map(first_run)
    new_horse = (fr.isna() | (fr >= week_start)).mean()
    out = {"rows": int(len(cur)), "new_horse": float(new_horse)}
    if "jockey_id" in cur:
        out["new_jockey"] = float((~cur["jockey_id"].astype(str).isin(known_jockeys)).mean())
    if "trainer_id" in cur:
        out["new_trainer"] = float((~cur["trainer_id"].astype(str).isin(known_trainers)).mean())
    return out


def load_pred_with_card(root: Path, cfg: dict, years: list[int]) -> pd.DataFrame:
    """予測 + 出馬表(race_shutuba_flat)の表。結果が無くても作れる（先行指標用）。"""
    from src.scripts.data.drift_data import load_predictions

    pred = load_predictions(cfg, root, years)
    parts = []
    for y in years:
        p = root / "data/page_reference/tables" / str(y) / "race_shutuba_flat.parquet"
        if p.exists():
            x = pd.read_parquet(p, columns=["race_id", "date", "horse_id", "field_size", "jockey_id", "trainer_id"])
            parts.append(x)
    if not parts:
        return pred.iloc[0:0]
    card = pd.concat(parts, ignore_index=True)
    for c in ("race_id", "horse_id"):
        card[c] = card[c].astype(str)
    card["date"] = pd.to_datetime(card["date"], errors="coerce")
    card = card.drop_duplicates(["race_id", "horse_id"])
    return pred.merge(card, on=["race_id", "horse_id"], how="inner")


def known_ids(root: Path, train_to: int) -> tuple[set[str], set[str], pd.Series]:
    """学習期間(〜train_to 年)に出ていた騎手・調教師の ID 集合と、馬ごとの初出走日。"""
    tables = root / "data/page_reference/tables"
    j, t, fr = set(), set(), []
    for y in range(2020, 2100):
        p = tables / str(y) / "race_result_flat.parquet"
        if not p.exists():
            break
        x = pd.read_parquet(p, columns=["horse_id", "date", "jockey_id", "trainer_id"])
        fr.append(x.assign(date=pd.to_datetime(x["date"], errors="coerce"), horse_id=x["horse_id"].astype(str))[["horse_id", "date"]])
        if y <= train_to:
            j |= set(x["jockey_id"].dropna().astype(str))
            t |= set(x["trainer_id"].dropna().astype(str))
    first = pd.concat(fr).dropna().groupby("horse_id")["date"].min().dt.strftime("%Y-%m-%d")
    return j, t, first
