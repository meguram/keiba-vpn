"""位置取り予想（pos_model 試行 6）の推論。出走馬の履歴だけで当日の特徴量を作り、最終モデル（final_pos）で予測する。

学習の ``pos_model.build()`` は表全体のバッチ（約 8 分・1.3GB）なので、推論は『その日より前の走の集計（PosState）』から同じ式で作る。
共通にできる式（行ごとの列・レース内の相対・追い切りの標準化・コース×枠のキー・pp_*）は ``pos_model`` の関数を呼び、
馬ごとの履歴・騎手/調教師/父/母父/コースの as-of 平均だけをここで持つ。学習表との一致は tests/pipeline/test_pos_serving.py と
ホストの過去日の一致確認（``python -m src.scripts.data.pos_serving_parity``）で確かめる。

有効化: ``KEIBA_POS_MODEL_DIR``（final_pos の置き場）。未設定なら使わない（既定は今の位置取り予想=馬柱の集計のまま）。

組み込みの約束（費用）: ``PosState.from_tables`` は 2016 年以降の結果表を全部読む重い処理。Cloud Run の起動ごとには動かさず、
1 日 1 回ジョブかホストで作って小さな成果物にして保存し、API はそれを 1 回読むだけにする。GCP の外から GCS を繰り返し読まない。
出馬表に渡す行は平地（芝・ダート）で、土日をまとめて作るときは日曜分の state に土曜の走が入らない（学習では入る）ことに注意。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.scripts.data import pos_model as pm

CONSTS_FILE = "pos_consts.json"
PRIORS = (("j_pf", "jockey_id", 30.0), ("t_pf", "trainer_id", 30.0), ("s_pf", "sire", 50.0), ("ds_pf", "dam_sire", 50.0))
CAT_FEATURES = ("course", "venue_c", "cls", "going")
REQUIRED_CARD = ("race_id", "date", "venue", "surface", "distance", "field_size", "horse_number", "horse_id", "sex_age", "jockey_weight", "jockey_id", "trainer_id",
                 "grade", "track_condition")
OIKIRI_COLS = ("oi_1f", "oi_3f", "oi_4f", "oi_hill", "oi_eval", "oi_days")


def model_dir() -> Path | None:
    v = (os.environ.get("KEIBA_POS_MODEL_DIR") or "").strip()
    return Path(v) if v else None


def enabled() -> bool:
    return model_dir() is not None


def make_consts(features: pd.DataFrame) -> dict[str, Any]:
    """学習表（pos_model の features.parquet）から、推論に必要な定数（縮め先の全体平均・追い切りの標準化の値）を作る。学習時に 1 回。"""
    f = features
    _, oz = pm.oikiri_z(f[["race_id", "year", "oi_hill", "oi_1f", "oi_3f", "oi_4f", "oi_eval"]].copy())
    return {"m_pf": float(f["pos_first"].mean()), "oikiri": oz}


def load_consts(d: str | Path | None = None) -> dict[str, Any]:
    return json.loads(((Path(d) if d else model_dir()) / CONSTS_FILE).read_text(encoding="utf-8"))


def _blank_ped(df: pd.DataFrame) -> pd.DataFrame:
    """出馬表の父・母父は学習表でも空文字（sire/dam_sire はほぼ全期間で空）。同じキー（空文字）で数える。"""
    for c in ("sire", "dam_sire"):
        df[c] = df[c].fillna("").astype(str) if c in df.columns else ""
    return df


class PosState:
    """date < ``before`` の走の集計。馬ごとの直近 5 走・馬×芝ダ×距離帯の初角位置の和と件数・各キーの初角位置の和と件数。"""

    def __init__(self, sums: dict[str, pd.DataFrame], horses: pd.DataFrame, same: pd.DataFrame, consts: dict[str, Any], last_date: pd.Timestamp):
        self.sums, self.horses, self.same, self.consts, self.last_date = sums, horses, same, consts, last_date

    @classmethod
    def from_runs(cls, df: pd.DataFrame, consts: dict[str, Any], before: pd.Timestamp | None = None) -> "PosState":
        """df: ``pos_model.load_runs`` の出力に ``f3_dev`` を足したもの。before を渡すと、その日より前の走だけで集計する。"""
        m = consts["m_pf"]
        df = _blank_ped(df.copy())
        if "j_pf" not in df.columns:
            df["j_pf"] = pm.asof_mean(df, "jockey_id", k=30.0, m=m)
        d = df if before is None else df[df["date"] < before]
        d = d.copy()
        d["_ck"], d["_ckg"] = pm.course_gate_keys(d)
        sums = {k: d.groupby(k)["pos_first"].agg(["sum", "count"]) for k in ("jockey_id", "trainer_id", "sire", "dam_sire", "_ck", "_ckg")}
        d = d.sort_values(["horse_id", "date", "race_id"])
        g = d.groupby("horse_id", sort=False)
        d["_k"] = g.cumcount(ascending=False) + 1
        t = d[d["_k"] <= 5]
        cols: dict[str, pd.DataFrame] = {}
        for name, n in (("pf", 5), ("pl", 3), ("p2", 3), ("fd", 3), ("jp", 3)):
            src = {"pf": "pos_first", "pl": "pos_last", "p2": "pos_second", "fd": "f3_dev", "jp": "j_pf"}[name]
            w = t[t["_k"] <= n].pivot(index="horse_id", columns="_k", values=src).reindex(columns=range(1, n + 1))
            w.columns = [f"{name}{k}" for k in w.columns]
            cols[name] = w
        last = t[t["_k"] == 1].set_index("horse_id")[["date", "distance", "surface", "jockey_id", "weight"]].add_prefix("last_")
        horses = pd.concat(list(cols.values()) + [last, g.size().rename("runs")], axis=1)
        sk = d["horse_id"] + "|" + d["surface"] + "|" + d["dist_band"].astype(str)
        same = d.groupby(sk)["pos_first"].agg(["sum", "count"])
        return cls(sums, horses, same, consts, d["date"].max())

    @classmethod
    def from_tables(cls, tables: Path | None, consts: dict[str, Any], years: list[int], before: pd.Timestamp | None = None) -> "PosState":
        if min(years) != 2016:
            raise ValueError("as-of の累積と f3_dev の縮め先は 2016 年からの全期間を読む前提（学習と同じ）。years は 2016 年から始める")
        df = pm.load_runs(tables, years)
        df["f3_dev"] = pm.race_front_pace(df, tables=tables, years=years)
        return cls.from_runs(df, consts, before)


def _asof(state: PosState, name: str, keys: pd.Series, k: float) -> np.ndarray:
    s = state.sums[name].reindex(keys.to_numpy())
    cs, cn = s["sum"].fillna(0.0).to_numpy(), s["count"].fillna(0.0).to_numpy()
    return (cs + k * state.consts["m_pf"]) / (cn + k)


def card_features(card: pd.DataFrame, state: PosState, oikiri: pd.DataFrame | None = None, blank_pedigree: bool = True,
                  fill_weight: bool = True, fill_going: bool = True) -> pd.DataFrame:
    """出走馬の表から ``pos_model.FEATURES`` を作る。card は ``load_runs`` と同じ列名
    （race_id・date・venue・surface・distance・field_size・horse_number・horse_id・sex_age・jockey_weight・jockey_id・trainer_id・weight・weight_change・
    grade・track_condition。sire/dam_sire は無くてよい）。oikiri: ``race_oikiri_flat`` と同じ列の表（無ければ追い切りの列は欠損）。
    blank_pedigree=False は学習表と同じ値の確認用（結果表の行は父が '?' の行がある）。戻り値は card と同じ並びで、race_id・horse_id・FEATURES の列。"""
    missing = [c for c in REQUIRED_CARD if c not in card.columns]
    if missing:
        raise KeyError(f"出馬表に必要な列が無い: {missing}")
    df = card.copy().reset_index(drop=True)
    day = pd.to_datetime(df["date"]).min()
    if pd.notna(state.last_date) and state.last_date >= day:
        raise ValueError(f"state に出走日以降の走が入っている（state の最終日 {state.last_date.date()} ≥ 出走日 {day.date()}）。PosState は before=出走日 で作る")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    if df["distance"].dropna().mod(1).eq(0).all():      # NaN が混ざった float だと course のキーが '1600.0' になり学習の '1600' と合わない
        df["distance"] = df["distance"].astype("Int64")
    for c, width in (("jockey_id", 5), ("trainer_id", 5)):   # 数値で来た ID は頭の 0 が落ちている
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("Int64").astype(str).str.zfill(width)
    if blank_pedigree:                      # 学習の表は父・母父がほぼ全期間で空。出馬表の父名を入れると集計のキーが合わなくなるので空に揃える
        df["sire"], df["dam_sire"] = "", ""
    df = _blank_ped(df)
    df["race_id"], df["horse_id"], df["jockey_id"], df["trainer_id"] = (df[c].astype(str) for c in ("race_id", "horse_id", "jockey_id", "trainer_id"))
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["dist_band"] = pd.cut(df["distance"], [0, 1300, 1700, 2100, 9999], labels=False).astype(float)
    H = state.horses.reindex(df["horse_id"].to_numpy())
    H.index = df.index
    runs = H["runs"].fillna(0)
    P = H[[f"pf{k}" for k in range(1, 6)]]
    w = pm.WEIGHTS
    num = sum(wk * P[f"pf{k}"].fillna(0) for wk, k in zip(w, range(1, 6)))
    den = sum(wk * P[f"pf{k}"].notna() for wk, k in zip(w, range(1, 6)))
    df["h_pf_w5"] = (num / den).where(den > 0)
    df["h_pf_1"], df["h_pf_2"] = P["pf1"], P["pf2"]
    df["h_pf_3"] = P[["pf1", "pf2", "pf3"]].mean(axis=1)
    df["h_pf_sd5"], df["h_pf_min5"] = P.std(axis=1), P.min(axis=1)
    df["h_pl_3"] = H[["pl1", "pl2", "pl3"]].mean(axis=1)
    df["h_p2_3"] = H[["p21", "p22", "p23"]].mean(axis=1)
    for k in (1, 2, 3):
        df[f"h_f3d_{k}"] = H[f"fd{k}"]
    df["h_es_3"] = pd.concat([(1 - H[f"pf{k}"]) - 0.1 * H[f"fd{k}"] for k in (1, 2, 3)], axis=1).mean(axis=1)
    df["h_runs"] = runs.astype(int)
    df["h_days"] = (df["date"] - pd.to_datetime(H["last_date"])).dt.days
    df["h_dist_chg"] = df["distance"] - H["last_distance"].astype(float)
    df["h_surf_chg"] = (df["surface"] != H["last_surface"]).astype(float).where(runs > 0)
    sk = df["horse_id"] + "|" + df["surface"] + "|" + df["dist_band"].astype(str)
    sm = state.same.reindex(sk.to_numpy())
    df["h_pf_same"] = (sm["sum"] / sm["count"]).where(sm["count"].fillna(0) > 0).to_numpy()
    for out, key, k in PRIORS:
        df[out] = _asof(state, key, df[key], k)
    df["debut"] = (df["h_runs"] == 0).astype(int)
    df["h_jk_same"] = (df["jockey_id"] == H["last_jockey_id"]).astype(float).where(runs > 0)
    df["j_pf_diff"] = df["j_pf"] - H[["jp1", "jp2", "jp3"]].mean(axis=1)
    ck, ckg = pm.course_gate_keys(df)
    df["cg_pf"] = _asof(state, "_ckg", ckg, 50.0) - _asof(state, "_ck", ck, 50.0)
    if fill_going:                          # 前夜の出馬表には当日の馬場が無い。学習は常に実際の馬場が入る（良 76%）ので、欠損は最頻の良で埋める
        df["track_condition"] = df["track_condition"].where(df["track_condition"].notna() & (df["track_condition"].astype(str).str.strip() != ""), "良")
    if fill_weight:                         # 前夜の出馬表には馬体重が無い。欠損のままだと 0kg 扱いになり MAE が約 6% 悪化する（誤差分析）ので、前走の体重・増減 0 で埋める
        wt = pd.to_numeric(df["weight"], errors="coerce") if "weight" in df.columns else pd.Series(np.nan, index=df.index)
        miss = wt.isna()
        df["weight"] = wt.where(~miss, pd.to_numeric(H["last_weight"], errors="coerce"))
        chg = pd.to_numeric(df["weight_change"], errors="coerce") if "weight_change" in df.columns else pd.Series(np.nan, index=df.index)
        df["weight_change"] = chg.where(~(miss & chg.isna()), 0.0).where(df["weight"].notna())
    df = pm.row_columns(df)
    df = pm.race_level_columns(df)
    if oikiri is not None and len(oikiri):
        df = df.merge(pm.oikiri_from_frame(oikiri), on=["race_id", "horse_id"], how="left")
    else:
        for c in OIKIRI_COLS:
            df[c] = np.nan
    df, _ = pm.oikiri_z(df, state.consts["oikiri"])
    return df[["race_id", "horse_id"] + pm.FEATURES]


class PosModel:
    """final_pos（1 段目 × 乱数 3・2 段目 × 乱数 5・LambdaRank）で位置取り予想（0〜1。小さいほど前）を出す。"""

    def __init__(self, meta: dict[str, Any], stage1: list[Any], stage2: list[Any], rank: Any):
        self.meta, self.stage1, self.stage2, self.rank = meta, stage1, stage2, rank

    @classmethod
    def load(cls, d: str | Path | None = None) -> "PosModel":
        import lightgbm as lgb

        d = Path(d) if d else model_dir()
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if list(meta["features"]) != list(pm.FEATURES):
            raise ValueError("final_pos の特徴量の並びが今の pos_model.FEATURES と違う（学習し直すか、コードを学習時の版に合わせる）")
        b = lambda n: lgb.Booster(model_file=str(d / n))  # noqa: E731
        return cls(meta, [b(f"stage1_s{s}.txt") for s in meta["stage1_seeds"]], [b(f"stage2_s{s}.txt") for s in meta["stage2_seeds"]],
                   b("rank.txt") if meta["rank_blend"] > 0 else None)

    def _cast(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.meta["cats"]:
            X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=self.meta["levels"][c])
        return X

    def predict(self, feats: pd.DataFrame) -> pd.Series:
        """feats: ``card_features`` の出力（レース単位でまとめて渡す）。戻り値: pos_pred（feats と同じ index）。"""
        X = self._cast(feats[pm.FEATURES])
        p1 = np.clip(np.mean([b.predict(X) for b in self.stage1], axis=0), 0, 1)
        t = pd.DataFrame({"race_id": feats["race_id"].to_numpy(), "p1": p1}, index=feats.index)
        S = pm._stage2_cols(t)
        X2 = pd.concat([X, S], axis=1)[self.meta["stage2_cols"]]
        p = np.clip(np.mean([b.predict(X2) for b in self.stage2], axis=0), 0, 1)
        blend = float(self.meta["rank_blend"])
        if blend > 0 and self.rank is not None:
            v = pd.DataFrame({"race_id": feats["race_id"].to_numpy(), "s": self.rank.predict(X2)})
            n = v.groupby("race_id")["s"].transform("count")
            rk = ((v.groupby("race_id")["s"].rank(ascending=False) - 1) / (n - 1).clip(lower=1)).to_numpy()
            p = (1 - blend) * p + blend * rk
        return pd.Series(p, index=feats.index, name="pos_pred")


def pos_pred_for_card(card: pd.DataFrame, state: PosState, model: PosModel, oikiri: pd.DataFrame | None = None, blank_pedigree: bool = True,
                      fill_weight: bool = True, fill_going: bool = True) -> pd.DataFrame:
    """race_id・horse_id・pos_pred と pp_* 4 列。"""
    f = card_features(card, state, oikiri, blank_pedigree, fill_weight, fill_going)
    out = f[["race_id", "horse_id"]].copy()
    out["pos_pred"] = model.predict(f).to_numpy()
    return pd.concat([out, pm.pp_columns(out)], axis=1)
