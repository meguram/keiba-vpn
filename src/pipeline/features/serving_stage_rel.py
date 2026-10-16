"""v3 の推論用: レース内の相対化列・前段モデル 3 列（s_time_pred / s_time_gap / s_rank_pred）・追い切り tl_* 列を作る。

学習側（``tp05_data.load_dataset`` / ``win_models.add_race_relative`` / tp-04 / tp-19）と同じ式を共有し、二重実装にしない。
- 相対化: ``race_relative`` が唯一の実装（学習側もここを呼ぶ）。出走馬の base 列だけから作る。
- 前段モデル: tp-04 の採用は P5（走破タイム差）と P6（正規化着順）で、どちらも「前段なし」= 入力は base 175 列のみ。
  学習時の列は時系列 OOF（expanding）で、推論時は最終モデル（2020〜2025 の全行で学習）が未知の行を予測するので性質が合う。
- tl_*: 参照表（直近 12 か月の時計の基準）+ 出走馬の馬別調教の記録から ``training_serving.tl_a_for_race`` で作る。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

SUFFIX = ("_rz", "_rrank", "_rgap")
_NAME_RE = re.compile(r"^(?P<src>.+?)(?P<part>_rz|_rrank|_rgap)$")


# ───────────────────────── レース内の相対化
def race_relative(df: pd.DataFrame, cols: Sequence[str], parts: Sequence[str] = SUFFIX) -> pd.DataFrame:
    """各列について、レース内の z（``_rz``）・順位（``_rrank``。降順で 1 位 = 最大）・最大との差（``_rgap``）を作る。
    結果（着順など）は使わず、as-of の特徴量どうしの比較だけ。列の順は列ごとに parts の順。"""
    g = df.groupby("race_id", sort=False)
    out: dict[str, Any] = {}
    for c in cols:
        x = df[c].astype("float64")
        if "_rz" in parts:
            mu, sd = g[c].transform("mean"), g[c].transform("std")
            out[c + "_rz"] = ((x - mu) / sd.replace(0, np.nan)).astype("float32")
        if "_rrank" in parts:
            out[c + "_rrank"] = g[c].rank(ascending=False, method="average").astype("float32")
        if "_rgap" in parts:
            out[c + "_rgap"] = (x - g[c].transform("max")).astype("float32")
    return pd.DataFrame(out, index=df.index)


def split_relative_name(name: str) -> tuple[str, str] | None:
    m = _NAME_RE.match(name)
    return (m.group("src"), m.group("part")) if m else None


def relative_by_names(df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """モデルの列名（``f3_r_rz`` など）から元の列と種類を読み取って作る。返す列の順は names のまま。"""
    need: dict[str, list[str]] = {}
    for n in names:
        sp = split_relative_name(n)
        if sp is None:
            raise ValueError(f"相対化の列名ではありません: {n}")
        need.setdefault(sp[0], []).append(sp[1])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"相対化の元の列がありません: {missing}")
    parts: dict[str, pd.DataFrame] = {}
    for c, ps in need.items():
        parts[c] = race_relative(df, [c], ps)
    out = pd.concat([parts[split_relative_name(n)[0]][[n]] for n in names], axis=1)
    return out


# ───────────────────────── 前段モデル（P5・P6）
STAGE_TARGETS = {"s_time_pred": "t_time_dev", "s_rank_pred": "t_rank_norm"}
STAGE_COLS = ["s_time_pred", "s_time_gap", "s_rank_pred"]
MISSING_LEVEL = "(なし)"


class StageModels:
    """走破タイム差・正規化着順を base 列から予測する LightGBM 2 本。``s_time_gap`` は予測を使ってレース内で作る。"""

    def __init__(self, feature_cols: list[str], cat_levels: dict[str, list[str]], boosters: dict[str, Any]):
        self.feature_cols = list(feature_cols)
        self.cat_levels = cat_levels
        self.boosters = boosters

    @classmethod
    def fit(cls, X: pd.DataFrame, targets: dict[str, np.ndarray], threads: int, seed: int = 0, n_rounds: int | None = None,
            params: dict | None = None) -> "StageModels":
        """X: base 列（カテゴリ列は category 型）。targets: s_* の名前 → 目的変数（欠損の行は学習から外す）。tp-04 の ``oof_regression`` と同じ設定。"""
        import lightgbm as lgb

        from src.pipeline.models import stage_models as sm

        p = dict(sm.REG_PARAMS, **(params or {}))
        cats = [c for c in X.columns if str(X[c].dtype) == "category"]
        levels = {c: [str(v) for v in X[c].cat.categories] for c in cats}
        boosters = {}
        for name, y in targets.items():
            y = np.asarray(y, dtype=float)
            ok = ~np.isnan(y)
            boosters[name] = lgb.train(dict(p, seed=seed, num_threads=threads), lgb.Dataset(X.loc[ok], label=y[ok], categorical_feature=cats or "auto"),
                                       num_boost_round=n_rounds or sm.N_ROUNDS)
        return cls(list(X.columns), levels, boosters)

    def _cast(self, X: pd.DataFrame) -> pd.DataFrame:
        miss = [c for c in self.feature_cols if c not in X.columns]
        if miss:
            raise KeyError(f"前段モデルの入力列がありません: {miss[:10]}（{len(miss)} 列）")
        Xm = X[self.feature_cols].copy()
        for c, lv in self.cat_levels.items():
            s = Xm[c].astype(str).where(Xm[c].notna())
            if MISSING_LEVEL in lv:                      # 学習時は欠損を「(なし)」にしている（ablation.load_table）
                s = s.fillna(MISSING_LEVEL)
            Xm[c] = pd.Categorical(s, categories=lv)
        return Xm

    def predict(self, X: pd.DataFrame, race_id: Sequence[Any]) -> pd.DataFrame:
        """戻り値: s_time_pred, s_time_gap, s_rank_pred（X と同じ並び・index）。s_time_gap = s_time_pred − レース内の最小。"""
        Xm = self._cast(X)
        out = pd.DataFrame(index=X.index)
        out["s_time_pred"] = self.boosters["s_time_pred"].predict(Xm)
        out["s_time_gap"] = out["s_time_pred"] - out["s_time_pred"].groupby(np.asarray(race_id)).transform("min")
        out["s_rank_pred"] = self.boosters["s_rank_pred"].predict(Xm)
        return out[STAGE_COLS].astype("float32")

    def save(self, d: str | Path, extra: dict | None = None) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for name, b in self.boosters.items():
            b.save_model(str(d / f"{name}.txt"))
        (d / "stage_meta.json").write_text(json.dumps({"feature_cols": self.feature_cols, "cat_levels": self.cat_levels, "models": list(self.boosters),
                                                       **(extra or {})}, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, d: str | Path) -> "StageModels":
        import lightgbm as lgb

        d = Path(d)
        meta = json.loads((d / "stage_meta.json").read_text(encoding="utf-8"))
        return cls(meta["feature_cols"], meta["cat_levels"], {n: lgb.Booster(model_file=str(d / f"{n}.txt")) for n in meta["models"]})


# ───────────────────────── tl_*（追い切り）
TL_PREFIX = "tl_"


def tl_reference_path(base: str | Path) -> Path:
    return Path(base) / "tl_reference.parquet"


def build_tl_reference(training_laps: pd.DataFrame, out: str | Path | None = None, window_months: int = 12, min_n: int = 200) -> pd.DataFrame:
    """追い切り表から参照表を作る（月ごとに『その月より前の 12 か月』の時計の基準）。出力先を渡せば保存する。"""
    from src.pipeline.features import training_serving as ts

    ref = ts.build_reference_table(training_laps, window_months, min_n)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        ref.to_parquet(out, index=False)
    return ref


def tl_columns_for_races(races: pd.DataFrame, records: dict[str, dict[str, Any] | None], ref: pd.DataFrame) -> pd.DataFrame:
    """races: race_id・horse_id・date の表（出走馬）。records: horse_id → 馬別調教ページの記録（``storage.load('horse_training', id)``）。
    戻り値: race_id・horse_id + ``tl_*`` 列（学習の ``cand_TL_A`` を ``tl_`` 付きにしたもの）。記録の無い馬は欠損の行。"""
    from src.pipeline.features import training_serving as ts

    r = races[["race_id", "horse_id", "date"]].copy()
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    r["date"] = pd.to_datetime(r["date"]).dt.strftime("%Y-%m-%d")
    parts = []
    for d, g in r.groupby("date", sort=False):
        got = ts.tl_a_for_race({h: records.get(h) for h in g["horse_id"].unique()}, d, ref)
        m = g[["race_id", "horse_id"]].merge(got.reset_index(), on="horse_id", how="left") if len(got) else g[["race_id", "horse_id"]].copy()
        parts.append(m)
    out = pd.concat(parts, ignore_index=True) if parts else r[["race_id", "horse_id"]]
    rename = {c: TL_PREFIX + c for c in out.columns if c not in ("race_id", "horse_id")}
    return out.rename(columns=rename)
