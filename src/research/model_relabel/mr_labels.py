"""拡張ラベル（担当 model-relabel）: 結果表の着差の文字列の累積から、勝ち・複勝の拡張ラベルと大敗の印を作る。

- 着差（race_result_flat の margin）は直前の馬からの差。着順に並べて勝ち馬から累積する。換算は config/model_relabel/spec.json。
- y_w: 1 着、または勝ち馬からの累積 ≤ 0.5 馬身。y_p: 3 着内、または（自分の累積 − 3 着馬の累積の最大）≤ 0.5 馬身。y_p1: 同じ作りで ≤ 1.0 馬身（11/27 に追加）。
- big（大敗）: 6 着以下かつ time_sec − 勝ち馬の time_sec ≥ 1.0 秒。
- 拡張しないレース（元のラベルのまま）: 2 着以下に空の着差・1 着に着差・降着の印（finish_position_raw の「降」。2016〜19 年だけ列がある）・
  着順で後ろの馬の時計が前の馬より速い。2016〜19 年の降着 12 レースは、降着の印を除いた 3 つの決まりで全部拾えた（2020 年以降は印が無い）。
- 同着の組の中は、着差が「同着」の行を後ろに並べてから累積する（入力の行の順に頼らない）。
- 学習の側だけで使う。推論の経路（src/pipeline/）からは import しない（tests/research/test_model_relabel.py で確かめる）。

費用 0 円（手元の結果表だけ。GCS を読まない）。
  python -m src.research.model_relabel.mr_labels table
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "config/model_relabel/spec.json"
TABLES = Path(os.environ.get("KEIBA_TABLES_DIR") or ROOT / "data/page_reference/tables")
OUT = Path(os.environ.get("MODEL_RELABEL_OUT") or "/home/jovyan/work/model_relabel_out")
EPS = 1e-9


def load_spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def _part(s: str, conv: dict) -> float:
    if s in conv:
        return float(conv[s])
    m = re.fullmatch(r"(?:(\d+)\.)?(\d+)/(\d+)", s)
    if m:
        return int(m.group(1) or 0) + int(m.group(2)) / int(m.group(3))
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    raise ValueError(f"着差の文字列が読めない: {s!r}")


def margin_lengths(s, conv: dict) -> float:
    """着差の文字列を馬身に換算する。空は NaN。「1/2+ハナ」は和。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip()
    if s == "":
        return np.nan
    return float(sum(_part(p.strip(), conv) for p in s.split("+")))


def build_labels(res: pd.DataFrame, spec: dict | None = None, conv_key: str = "convert_A") -> pd.DataFrame:
    """結果表（race_id・horse_id・finish_position・margin・time_sec、あれば finish_position_raw）から、
    y1・y3・y_w・y_p・y_p1・big・cum（勝ち馬からの累積）・dt（勝ち馬との時計の差）・ext_ok を返す（入力の行の順）。"""
    spec = spec or load_spec()
    conv = spec["margin"][conv_key]
    thr = float(spec["margin"]["within_lengths"])
    d = res[["race_id", "horse_id", "finish_position", "margin", "time_sec"]].copy()
    d["_ord"] = np.arange(len(d))
    raw = res["finish_position_raw"].astype(str) if "finish_position_raw" in res.columns else pd.Series("", index=res.index)
    d["_demoted"] = raw.str.contains("降").to_numpy()
    fin = pd.to_numeric(d["finish_position"], errors="coerce").fillna(-1).to_numpy()
    d["finish_position"] = fin
    d["m"] = d["margin"].map(lambda s: margin_lengths(s, conv))
    d["_tie"] = (d["margin"].astype(str).str.strip() == "同着").astype(int)
    ok_row = fin > 0
    f = d[ok_row].sort_values(["race_id", "finish_position", "_tie", "_ord"], kind="mergesort").copy()
    g = f.groupby("race_id", sort=False)
    # 拡張してよいレースか
    blank_after = (f["finish_position"] > 1) & f["m"].isna()
    winner_has = (f["finish_position"] == 1) & f["m"].notna() & (f["margin"].astype(str).str.strip() != "同着")
    t = f["time_sec"].to_numpy(float)
    t_prev = g["time_sec"].shift(1).to_numpy(float)
    time_back = np.isfinite(t_prev) & (t + EPS < t_prev)                  # 後ろの着順の馬の時計が前の馬より速い
    bad = pd.Series(blank_after.to_numpy() | winner_has.to_numpy() | time_back | f["_demoted"].to_numpy(), index=f.index)
    bad_race = bad.groupby(f["race_id"]).transform("any").to_numpy()
    f["cum"] = g["m"].transform(lambda s: s.fillna(0.0).cumsum())
    f.loc[f["finish_position"] == 1, "cum"] = 0.0
    c3 = f[f["finish_position"] <= 3].groupby("race_id")["cum"].max()      # 3 着内の馬の累積の最大（3 着の同着・2 着の同着で 3 着が無いときも）
    f["cum3"] = f["cum"] - f["race_id"].map(c3)
    wt = f[f["finish_position"] == 1].groupby("race_id")["time_sec"].min()
    f["dt"] = f["time_sec"] - f["race_id"].map(wt)
    fp = f["finish_position"].to_numpy()
    y1 = fp == 1
    y3 = fp <= 3
    ext_ok = ~bad_race
    yw = y1 | (ext_ok & (f["cum"].to_numpy() <= thr + EPS))
    c3v = f["cum3"].to_numpy()
    yp = y3 | (ext_ok & np.isfinite(c3v) & (c3v <= thr + EPS))
    thr_p1 = float(spec["margin"].get("within_lengths_place_1", 1.0))
    f["y_p1"] = y3 | (ext_ok & np.isfinite(c3v) & (c3v <= thr_p1 + EPS))
    big = (fp >= 6) & (f["dt"].to_numpy() >= 1.0 - EPS)
    f["y1"], f["y3"], f["y_w"], f["y_p"], f["big"], f["ext_ok"] = y1, y3, yw, yp, big, ext_ok
    out = d[["race_id", "horse_id", "finish_position", "_ord"]].copy()
    cols = ["y1", "y3", "y_w", "y_p", "y_p1", "big", "ext_ok", "cum", "dt"]
    out = out.merge(f[["_ord"] + cols], on="_ord", how="left").sort_values("_ord")
    for c in ("y1", "y3", "y_w", "y_p", "y_p1", "big"):
        out[c] = out[c].fillna(False).astype(bool)
    race_ok = f.groupby("race_id")["ext_ok"].all()
    out["ext_ok"] = out["race_id"].map(race_ok).fillna(True).astype(bool)     # 中止の馬にもレースの値を引き継ぐ
    return out.drop(columns="_ord").reset_index(drop=True)


def load_results(years=range(2016, 2025), ledger: bool = True) -> pd.DataFrame:
    """開発の行（2016〜24）の結果表を読む。2025 年以降は読まない。2022〜24 は開発の検証の期間なので台帳に残す。"""
    if ledger:
        from src.ops import data_ledger as DL
        tok = DL.guard(tool="src/research/model_relabel/mr_labels.py:load_results", family="model_relabel", version="labels-mr-4",
                       period=(f"{min(years)}-01-01", f"{max(years)}-12-31"), purpose="reproduce", reads=("着順", "人気"), frozen_paths=[SPEC],
                       reason="拡張ラベル（着差・時計）を学習の教師として作り、本命決着の印（確定の人気）を学習の行の選びにだけ使う。成績は数えない")
        tok.finish()
    frames = []
    for y in years:
        assert int(y) <= 2024, "2025 年以降の結果表は開かない"
        p = TABLES / str(y) / "race_result_flat.parquet"
        import pyarrow.parquet as pq
        names = pq.ParquetFile(p).schema_arrow.names
        cols = ["race_id", "horse_id", "finish_position", "margin", "time_sec", "field_size", "surface", "grade", "race_class", "popularity"]
        cols += [c for c in ("finish_position_raw",) if c in names]
        frames.append(pd.read_parquet(p, columns=cols))
    return pd.concat(frames, ignore_index=True)


def labels_dev(conv_key: str = "convert_A") -> pd.DataFrame:
    res = load_results()
    lab = build_labels(res, conv_key=conv_key)
    lab["race_id"], lab["horse_id"] = lab["race_id"].astype(str), lab["horse_id"].astype(str)
    for c in ("field_size", "surface", "grade", "race_class"):
        lab[c] = res[c].to_numpy()
    lab["year"] = lab["race_id"].str[:4].astype(int)
    lab["chalk_race"] = chalk_races(res)
    return lab


def chalk_races(res: pd.DataFrame) -> np.ndarray:
    """1〜3 着の馬がすべて 3 番人気以内だったレースの印（行ごと）。学習の行を除く版（_nc）にだけ使う。特徴量には入れない。
    確定の人気・着順で決まるので、発走前には分からない。評価からは除かない（評価は全レース。11/27 統括役）。"""
    fp = pd.to_numeric(res["finish_position"], errors="coerce")
    pop = pd.to_numeric(res["popularity"], errors="coerce")
    top = (fp >= 1) & (fp <= 3)              # 同着で 4 頭以上いるときは全員（spec の chalk_definition）
    rid = res["race_id"]
    mx = pop.where(top).groupby(rid).transform("max")
    n_top = top.groupby(rid).transform("sum")
    n_pop = (top & pop.notna()).groupby(rid).transform("sum")
    return ((n_top > 0) & (n_pop == n_top) & (mx <= 3)).to_numpy()      # 3 着内の人気が 1 頭でも欠けたら印を付けない


def rate_table(lab: pd.DataFrame) -> dict:
    """拡張で 1 になる割合（年・クラス・頭数の帯・芝ダ）。完走馬だけで数える。"""
    f = lab[lab["finish_position"] > 0].copy()
    f["fs_band"] = pd.cut(f["field_size"], [0, 10, 13, 16, 18], labels=["≤10", "11-13", "14-16", "17-18"])
    f["cls"] = f["grade"].replace("", "（grade が空）").fillna("不明")

    def agg(by):
        t = f.groupby(by, observed=True).agg(n=("y1", "size"), y1=("y1", "mean"), y_w=("y_w", "mean"), y3=("y3", "mean"),
                                             y_p=("y_p", "mean"), y_p1=("y_p1", "mean"), big=("big", "mean"))
        return {str(k): {c: (round(float(v), 4) if c != "n" else int(v)) for c, v in r.items()} for k, r in t.iterrows()}

    races = lab.drop_duplicates("race_id")
    no_ext = lab.groupby("race_id")["ext_ok"].all()
    return {"rows_finished": int(len(f)), "races": int(len(races)), "races_no_extend": int((~no_ext).sum()),
            "all": agg(lambda x: "全体"), "year": agg("year"), "class": agg("cls"), "field_size": agg("fs_band"), "surface": agg("surface"),
            "secondary_time": {"time_le_0.1_2nd_plus": round(float(((f["finish_position"] > 1) & (f["dt"] <= 0.1 + EPS)).mean()), 4),
                               "crosstab_yw_vs_time": pd.crosstab(f["y_w"], (f["finish_position"] == 1) | (f["dt"] <= 0.1 + EPS)).to_dict()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "save"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "save":
        for k in ("convert_A", "convert_B_sensitivity"):
            tmp = OUT / f"labels_{k}.tmp.parquet"       # 学習が読んでいる最中に置き換えても壊れないように
            labels_dev(k).to_parquet(tmp, index=False)
            os.replace(tmp, OUT / f"labels_{k}.parquet")
        print("saved", OUT)
        return
    lab = labels_dev()
    tab = rate_table(lab)
    labB = labels_dev("convert_B_sensitivity")
    tab["A_vs_B"] = {"y_w_diff": int((lab["y_w"] != labB["y_w"]).sum()), "y_p_diff": int((lab["y_p"] != labB["y_p"]).sum())}
    s = json.dumps(tab, ensure_ascii=False, indent=1, default=str)
    (OUT / "label_rates.json").write_text(s, encoding="utf-8")
    print(s[:3000])


if __name__ == "__main__":
    main()
