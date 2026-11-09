"""母（同じ母の兄姉）の成績と、smartrc の系統を、予想モデルの特徴量に入れたときの上積みを、tp31 と同じ手順で測る（調査。本番は触らない）。

2026-11-09 ユーザー指示: 系統（smartrc）と、母・兄弟の新馬成績の統計量が、馬の予測モデルに採用できるかを調べる。
手順は tp31_eval の stage1 と同じ: valid 2022・2023、seed 1 本の LightGBM で、V4 の特徴量全体（tp31/v4_frame.parquet）に群を足し、
V4 との差（Δwin・95% 上限 win_hi）で判定する。基準（V4）の結果は tp31/stage1 のものを使い回す。市場オッズ・人気は使わない（tp31 と同じ）。

群（どれも「その日より前」の出走だけから。同じ日の出走は含めない）:
  SIB_dam:      同じ母の兄姉の 新馬戦の頭数・3着以内率・(3着以内 − 3/頭数) の縮約、全出走の (3着以内 − 3/頭数) の縮約と件数。母は 5 代血統の世代 1・位置 1。
  LINE_smartrc: smartrc の小系統・大系統・国別タイプ（カテゴリ。smartrc で空欄の種牡馬は netkeiba の系統）と、
                小系統 × 芝ダ × 距離帯 / 小系統 × 芝ダ × 良・道悪 の (3着以内 − 3/頭数) の縮約。
使い方:
    python -m src.scripts.data.tp_sib_line_eval groups     # 群を tp31/groups/ に作る
    python -m src.scripts.data.tp_sib_line_eval stage1     # 評価（数十分）。結果は tp31/stage1_sibline/results.json
    python -m src.scripts.data.tp_sib_line_eval stage2     # 2024 年・seed 2 本での確認。tp31/stage2_sibline/results.json
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd

from src.scripts.data import tp31_eval as e

KEY = e.KEY
GROUPS = ["SIB_dam", "LINE_smartrc"]
LABEL = "stage1_sibline"


def _runs() -> pd.DataFrame:
    from src.research.columns.compare_series import jump_mask

    R = e.ROOT / "data/page_reference/tables"
    cols = ["race_id", "date", "horse_id", "surface", "distance", "finish_position", "odds", "race_name", "race_class", "track_condition"]
    years = [y for y in range(2016, 2027) if (R / str(y) / "race_result_flat.parquet").exists()]
    r = pd.concat([pd.read_parquet(R / str(y) / "race_result_flat.parquet", columns=cols) for y in years], ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"]) & ~jump_mask(r) & (r["odds"] > 0)].copy()
    r["horse_id"], r["race_id"] = r["horse_id"].astype(str), r["race_id"].astype(str)
    r["date"] = pd.to_datetime(r["date"].astype(str).str[:10])
    fin = pd.to_numeric(r["finish_position"], errors="coerce")
    r["top3"] = fin.between(1, 3).astype(float)
    r["st"] = r.groupby("race_id")["race_id"].transform("size")
    r["r3"] = r["top3"] - np.minimum(1, 3 / r["st"])
    r = r.sort_values(["horse_id", "date", "race_id"])
    r["hn"] = r.groupby("horse_id").cumcount()
    return r.sort_values("date").reset_index(drop=True)


def _shrunk(r: pd.DataFrame, key: pd.Series, val: pd.Series, M: float) -> tuple[np.ndarray, np.ndarray]:
    from src.pipeline.features import factory_ped as fp

    S, N = fp.asof_sums(r, key.where(key.notna()), val.to_numpy())
    return S / (N + M), N


def cmd_groups(a) -> None:
    from src.pipeline.features import factory_ped as fp
    from src.scraper import sire_line_store, smartrc_sire_lines as ssl

    r = _runs()
    A, _ = fp.build_ped_tables(r["horse_id"].unique())
    r["dam"] = r["horse_id"].map(fp._pick(A, 1, 1, "dam"))
    r["sire"] = r["horse_id"].map(fp._pick(A, 1, 0, "sire"))
    deb = r["hn"] == 0
    G = r[KEY].copy()
    s, n = _shrunk(r, r["dam"], r["top3"].where(deb), 0.0)
    G["sib_deb_n"] = n
    G["sib_deb_t3"] = np.where(n > 0, s, np.nan)
    G["sib_deb_r3"], _ = _shrunk(r, r["dam"], r["r3"].where(deb), 3.0)
    G["sib_all_r3"], G["sib_all_n"] = _shrunk(r, r["dam"], r["r3"], 10.0)
    G.drop_duplicates(KEY).to_parquet(e.OUT / "groups" / "SIB_dam.parquet", index=False)
    e._log(f"SIB_dam: {len(G)} 行、兄姉の記録あり {float((G['sib_all_n'] > 0).mean()):.3f}")

    L = ssl.load()["sires"]
    nk = sire_line_store.labels()

    def lab(field: str):
        return r["sire"].map(lambda s: (L.get(str(s)) or {}).get(field) or nk.get(str(s)))
    r["l_small"], r["l_large"], r["l_type"] = lab("small_name"), lab("large_name"), lab("type_name")
    band = pd.cut(pd.to_numeric(r["distance"], errors="coerce"), [0, 1400, 1800, 2200, 9999], labels=["短", "マ", "中", "長"]).astype(str)
    wet = np.where(r["track_condition"].isin(["稍重", "重", "不良"]), "wet", "dry")
    H = r[KEY].copy()
    for c in ("l_small", "l_large", "l_type"):
        H[c] = r[c].astype("category")
    H["line_sd_r3"], H["line_sd_n"] = _shrunk(r, r["l_small"] + "|" + r["surface"] + "|" + band, r["r3"], 200.0)
    H["line_gd_r3"], _ = _shrunk(r, r["l_small"] + "|" + r["surface"] + "|" + pd.Series(wet, index=r.index), r["r3"], 200.0)
    H.drop_duplicates(KEY).to_parquet(e.OUT / "groups" / "LINE_smartrc.parquet", index=False)
    e._log(f"LINE_smartrc: {len(H)} 行、系統あり {float(H['l_small'].notna().mean()):.3f}")


def cmd_stage1(a) -> None:
    d = e.OUT / LABEL
    d.mkdir(parents=True, exist_ok=True)
    src = e.OUT / "stage1"
    if not (d / "base.parquet").exists() and (src / "base.parquet").exists():      # 同じ valid 年・seed の基準を使い回す
        shutil.copy(src / "base.parquet", d / "base.parquet")
        base = json.loads((src / "results.json").read_text(encoding="utf-8"))["base"]
        (d / "results.json").write_text(json.dumps({"base": base}, ensure_ascii=False, indent=1), encoding="utf-8")
    e._stage(LABEL, [2022, 2023], [0], GROUPS + ["ALL_PASSED"], a)


def cmd_stage2(a) -> None:
    """第 2 段: valid 2024、seed 2・3。基準は tp31/stage2 のものを使い回す。"""
    d = e.OUT / "stage2_sibline"
    d.mkdir(parents=True, exist_ok=True)
    src = e.OUT / "stage2"
    if not (d / "base.parquet").exists() and (src / "base.parquet").exists():
        shutil.copy(src / "base.parquet", d / "base.parquet")
        base = json.loads((src / "results.json").read_text(encoding="utf-8"))["base"]
        (d / "results.json").write_text(json.dumps({"base": base}, ensure_ascii=False, indent=1), encoding="utf-8")
    e._stage("stage2_sibline", [2024], [2, 3], GROUPS, a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["groups", "stage1", "stage2"])
    a = ap.parse_args()
    {"groups": cmd_groups, "stage1": cmd_stage1, "stage2": cmd_stage2}[a.cmd](a)


if __name__ == "__main__":
    main()
