"""前の夜の段の列の作り（血統予想家）。研究の道具は読むだけで、起動時に sha256 を照合する（会議の 1 号の条件 4）。

- ダートの 4 列（dirt_sire・dirt_bms・dirt_mat5・dirt_lv5）は、前向きの記録（forward_record.dirt_aptitude）と同じ関数で作る。
  値は記録の CSV と一致する（verify で |差| ≤ 1e-6 を確かめる）。
- 芝の牝系 5 代（turf_mat5）は、群 A を「芝の行」にして同じ mat_cols で作る（dirt_deep.setup の aspect=turf と同じ。未検証の規則）。
- 理由の文の材料: 3〜4 代の母の側の牡の祖先ごとの「血量 × 偏差」の寄与（mat_cols の中身と同じ計算）。
- as-of は「その日より厳密に前」の結果だけ。オッズ・着順は使わない（その日の行は未確定）。GCS を読まない。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
from src.research.pedigree_persona.store import JST  # noqa: E402  （t60 が pandas を読まずに使えるよう、store に置く）
# 前向きの登録（prereg_dirt_mat5.md・log.md の 849・1221 行）で凍結した道具の sha256。違えば止める
FROZEN_SHA256 = {
    "src/research/pedigree_lab/aptitude.py": "8039df5aa8418a1167a64af44ff9407c7b50eb2f3568f7ede6b1ea91c808e5f8",
    "src/research/pedigree_lab/aptitude_checks.py": "0685fea2cf91f7b392ba207107e51acde21b08a80f2f645e5bb8d5e236dc10fc",
    "src/research/pedigree_lab/forward_record.py": "6bdaff28a15257cf0df7bcf653d2b9744011791c99ca3c80801293377a36e5c5",
    # 前の夜の段が呼ぶ、残りの研究の道具（forward_record の凍結の一覧に入るもの。2026-11-27 の版で固定。critic 2）
    "src/research/pedigree_lab/build_base.py": "668b1035388462aeb246f99b44ed486f176adf88fb2923c75d693510cfd8aa41",
    "src/research/pedigree_lab/data.py": "5e51b366ba3e1bdb87d2208a048441f0a16d05de1a3e483d906bf3fc079ce354",
    "src/research/pedigree_lab/features4.py": "99c1b54685f2f0e90734c386d6062b704b0f15ce2a48342cf5a47ca8f6f506d3",
    "src/research/pedigree_lab/features2.py": "698af75356ca48341eb271eb0a9ce03bbec41d227a7c5ac1d13f455803e8ab7f",
    "src/research/pedigree_lab/gates.py": "82e7dd2e953bfa56aaf5f0ba0f0e79ba772607169fe2b6b9838a744693253863",
    "src/research/pedigree_lab/scene_dist.py": "7ca5c60b3d54acc9f95625a815720cce01a8894b1d90bc4548598ff25f81a988",
    "src/research/pedigree_lab/final_eval.py": "c994f3aceadfed600029bf33e092f3cec986da7d2f5b704d9908321433a1ccf2",
}
RAW_COLS = ["dirt_sire", "dirt_bms", "dirt_mat5", "dirt_lv5", "turf_mat5"]
REASON_GENS = (3, 4)          # 理由の文に名前を出す祖先の代（どの血統表にも多く出る祖先が毎回並ばないように）


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen(root: Path = ROOT) -> None:
    """研究の道具の sha256 を登録の値と照合する。違えば SystemExit（推論に合わせて書き換えない）。"""
    for rel, h in FROZEN_SHA256.items():
        got = sha256_of(root / rel)
        if got != h:
            raise SystemExit(f"研究の道具が凍結の版と違う: {rel}（{got[:16]}… ≠ {h[:16]}…）。前向きの登録が無効になるので止める")


def contributions(F: pd.DataFrame, A: np.ndarray, L: pd.DataFrame, rows: np.ndarray) -> dict[int, list[tuple[str, int, float]]]:
    """rows（F の行番号）の馬ごとに、3〜4 代の祖先の寄与（血量 × 偏差 / 血量の和）を大きい順に返す。mat_cols と同じ計算。"""
    from src.research.pedigree_lab import aptitude as ap_

    row = L["row"].to_numpy()
    mA0, mT0 = ap_.base_means(F, A, [])
    d = ap_.apt_by(L, A[row], ["anc"], base=(mA0[row], mT0[row]))
    T = pd.DataFrame({"row": row, "anc": L["anc"].to_numpy(), "gen": L["gen"].to_numpy(), "w": L["bw"].to_numpy(), "d": d})
    T = T[T["row"].isin(set(int(x) for x in rows))]
    T["c"] = T["w"] * T["d"] / T.groupby("row")["w"].transform("sum")
    out: dict[int, list[tuple[str, int, float]]] = {}
    for r, g in T[T["gen"].isin(REASON_GENS) & (T["c"] > 0)].groupby("row"):
        g = g.sort_values(["c", "anc"], ascending=[False, True])
        out[int(r)] = [(str(a), int(n), float(c)) for a, n, c in zip(g["anc"], g["gen"], g["c"])]
    return out


def ancestor_names(horse_id: str, ids: set[str]) -> dict[str, str]:
    """その馬の 5 代血統の JSON から、祖先の名前を引く（手元のファイルだけ）。"""
    from src.research.pedigree_lab import build_base

    for base in build_base.PED_DIRS:
        f = base / horse_id[:4] / f"{horse_id}.json"
        if f.exists():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            return {str(a.get("horse_id")): str(a.get("name") or "") for a in d.get("ancestors") or [] if str(a.get("horse_id")) in ids}
    return {}


def has_entries(day: pd.Timestamp) -> bool:
    """その日の芝・ダートの出馬表が手元にあるか（重い計算の前に確かめる）。"""
    from src.research.pedigree_lab import build_base

    p = build_base.TABLES / str(day.year) / "race_shutuba_flat.parquet"
    if not p.exists():
        return False
    S = pd.read_parquet(p, columns=["date", "surface"])
    return bool(((pd.to_datetime(S["date"]) == day) & S["surface"].isin(["芝", "ダート"])).any())


def build_day(day: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """その日の出馬表の馬ごとの生の値・理由の材料・事実の札の材料を作る。forward_record.main と同じ前処理（模型の当てはめは無い）。"""
    check_frozen()
    from src.research.pedigree_lab import aptitude as ap_
    from src.research.pedigree_lab import aptitude_checks as ck
    from src.research.pedigree_lab import build_base, data, features4, forward_record, gates, scene_dist

    runs = build_base.load_runs()
    runs = runs[runs["date"] < day]
    E = forward_record.entries(day)
    if E.empty:
        raise SystemExit(f"{day.date()} の出馬表が無い")
    r = data._prep(pd.concat([runs, E], ignore_index=True))
    ped = data.load_ped()
    miss = sorted(set(E["horse_id"]) - set(ped["horse_id"]))
    if miss:
        ped = pd.concat([ped, build_base.load_ped(miss)], ignore_index=True)
    X, kinfo = features4.build(r, ped, sub=True)
    frozen = json.loads((data.OUT / "kinfo_v4.json").read_text())
    if any(abs(kinfo[k] - frozen[k]) > 1e-6 for k in frozen if k != "rho") or kinfo["rho"] != frozen["rho"]:
        raise SystemExit("縮めの k が凍結の第 4 版と違う（作りがずれている）")
    X["split"] = -1
    Fsc = gates.scenes(X[gates.FCOLS].copy())
    D = scene_dist.marks(X[list(dict.fromkeys(gates.FCOLS + ["dband", "going"]))].copy())
    X["FD"], X["DB0"] = Fsc["FD"].to_numpy(), D["DB0"].to_numpy()
    dm = forward_record.dirt_aptitude(X, ped)               # 前向きの記録と同じ 4 列
    F = X[["date", "horse_id", "surf", "fpct", "sire", "ds"]].reset_index(drop=True)
    L = ap_.maternal_long(F, drop_common=False, ped=ped)
    A_t = (F["surf"] == "T").to_numpy()
    turf, _ = ck.mat_cols(F, A_t, [], L)
    is_day = (X["date"] == day).to_numpy()
    rows = np.flatnonzero(is_day)
    A_d, _ = ap_.group_a(F.assign(venue="", distance=0, going="", rage="", dchg=""), "dirt")
    con_d = contributions(F, A_d, L, rows[(F["surf"].to_numpy()[rows] == "D")])
    con_t = contributions(F, A_t, L, rows[(F["surf"].to_numpy()[rows] == "T")])
    T = X[is_day].copy()
    for k, v in dm.items():
        T[k] = v[is_day]
    T["turf_mat5"] = turf[is_day]
    T["no_pedigree"] = (T["sire"].isna() | T["sire"].eq("")).to_numpy()
    T["layoff_y"] = T["layoff"].eq("Y").to_numpy()
    T["new_race"] = T["cls"].eq("NEW").to_numpy()
    n_sire = X.loc[~is_day & X["sire"].fillna("").ne(""), "sire"].value_counts()     # 父の産駒の、その日より前の走りの数
    T["sire_runs"] = T["sire"].map(n_sire).fillna(0).astype(int).to_numpy()
    anc = []
    for i in rows:
        c = con_d.get(int(i)) if F["surf"].iat[i] == "D" else con_t.get(int(i))
        anc.append(json.dumps(c[:4] if c else [], ensure_ascii=False))
    T["anc_top"] = anc
    S = pd.read_parquet(build_base.TABLES / str(day.year) / "race_shutuba_flat.parquet", columns=["race_id", "horse_id", "horse_name", "start_time"])
    S["horse_id"] = S["horse_id"].astype(str)
    T = T.merge(S.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    out = T[["race_id", "date", "venue", "surface", "distance", "horse_number", "horse_id", "horse_name", "start_time", "FD", "layoff_y",
             "career", "new_race", "sire_runs", "no_pedigree", *RAW_COLS, "anc_top"]].copy()
    out["date"] = out["date"].dt.date.astype(str)
    src_tab = build_base.TABLES / str(day.year) / "race_shutuba_flat.parquet"
    meta = {"made_at": datetime.now(JST).isoformat(timespec="seconds"), "asof": "その日より厳密に前",
            "inputs": {"runs_rows_before_day": int(len(runs)), "runs_last_date": str(runs["date"].max().date()),
                       "shutuba_mtime": datetime.fromtimestamp(src_tab.stat().st_mtime, JST).isoformat(timespec="seconds"),
                       "ped_rows": int(len(ped))},
            "frozen_sha256": {k: v[:16] for k, v in FROZEN_SHA256.items()},
            "races": int(out["race_id"].nunique()), "horses": int(len(out)), "no_pedigree": int(out["no_pedigree"].sum())}
    return out.sort_values(["race_id", "horse_number"]).reset_index(drop=True), meta
