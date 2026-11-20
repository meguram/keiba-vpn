"""走法軸の点数（1〜10）ごとの勝率・複勝率を、対象（クラス・馬場・距離・コース）別に集計する（読むだけ・JSON を書く）。

入力（ホストのローカルだけ。GCS は読まない）:
  data/local/running_profile/axis_scores10.parquet … 各走の「直前時点」の 10 点満点（build_axis_scores10.py）。環境変数 KEIBA_AXIS_SCORES10 で差し替え可
  data/page_reference/tables/<年>/race_result_flat.parquet … 馬場・人気・頭数・着順
集計: 点数を floor で 1〜10 の 10 区間にし、区間ごとに 走数・勝ち数・複勝数（8 頭以上は 3 着以内、5〜7 頭は 2 着以内）と、
  「頭数・人気順が同じ馬の過去の平均（2016〜2023 年。人気から期待される勝率・複勝率）」の合計を数える。
期間: A = 2024-01-01〜2025-12-31（主）、B = 2026 年（確認だけ。テストセット②。選択には使わない）。
使い方（ホストで）: cd <repo> && nice -n 19 /opt/conda/bin/python -I src/scripts/data/axis_score_curves.py OUT.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from src.research.race import running_map as rm  # noqa: E402

TABLE_DIR = REPO / "data/page_reference/tables"
HIST = Path(os.environ.get("KEIBA_AXIS_SCORES10") or REPO / "data/local/running_profile/axis_scores10.parquet")
PERIODS = {"A": ("2024-01-01", "2025-12-31"), "B": ("2026-01-01", "2026-12-31")}
EXP_YEARS = range(2016, 2024)          # 人気から期待される率を作る期間（評価期間を含めない）
N_MIN_FIELD = 5
SHRINK = 40.0
COURSE_MIN_RUNS = 400                  # 個別のコースを出す最低の走数（期間 A）
EXTRA_LABELS = {"slope": ("急坂適性（中山・阪神の過去走から）", "ゴール前の坂で落ちない", "坂で落ちる")}
RELABEL = {"wett": "道悪適性（芝）", "wetd": "道悪適性（ダート）"}
ONLY_SURFACE = {"wett": "芝", "wetd": "ダート"}
CLASS_BANDS = [("未勝利・新馬", ("未勝利", "新馬")), ("1勝", ("1勝",)), ("2・3勝", ("2勝", "3勝")), ("OP以上", ("OP", "L", "G3", "G2", "G1"))]
GOING = {"良": "良", "稍": "稍重", "稍重": "稍重", "重": "重・不良", "不良": "重・不良"}
GOING_ORDER = ["良", "稍重", "重・不良"]
DIST_ORDER = ["〜1200m", "1300〜1400m", "1500〜1700m", "1800〜2000m", "2100〜2400m", "2500m〜"]


def dist_band(d: float) -> str:
    d = int(d)
    return DIST_ORDER[0] if d <= 1200 else DIST_ORDER[1] if d <= 1400 else DIST_ORDER[2] if d <= 1700 else DIST_ORDER[3] if d <= 2000 else DIST_ORDER[4] if d <= 2400 else DIST_ORDER[5]


def class_of(g: object) -> str | None:
    for name, members in CLASS_BANDS:
        if g in members:
            return name
    return None


def load_results(years) -> pd.DataFrame:
    parts = []
    for y in years:
        p = TABLE_DIR / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["race_id", "horse_id", "popularity", "field_size", "track_condition", "finish_position"]))
    d = pd.concat(parts, ignore_index=True)
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    for c in ("popularity", "field_size", "finish_position"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    n = d.groupby("race_id")["race_id"].transform("size")
    d["field_size"] = d["field_size"].where(d["field_size"] > 0, n)
    return d.drop_duplicates(["race_id", "horse_id"])


def place_cut(nf: np.ndarray) -> np.ndarray:
    return np.where(nf >= 8, 3, 2)


def expected_tables(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """[頭数 5〜18][人気順 1〜18] の 勝率・複勝率（過去の平均。隣の人気順へ寄せて均す）。"""
    f = train[(train["finish_position"] > 0) & (train["field_size"] >= N_MIN_FIELD) & (train["popularity"] >= 1)].copy()
    f = f[f["popularity"] <= f["field_size"]]
    f["nb"] = f["field_size"].clip(upper=18).astype(int)
    f["pp"] = f["popularity"].clip(upper=18).astype(int)
    f["win"] = (f["finish_position"] == 1).astype(float)
    f["plc"] = (f["finish_position"] <= place_cut(f["field_size"].to_numpy())).astype(float)
    out = []
    for col in ("win", "plc"):
        T = np.zeros((19, 19))
        for n in range(N_MIN_FIELD, 19):
            g = f[f["nb"] == n]
            s = np.array([g.loc[g["pp"] == k, col].sum() for k in range(1, n + 1)])
            c = np.array([(g["pp"] == k).sum() for k in range(1, n + 1)], float)
            allm = g[col].mean()
            raw = np.where(c > 0, s / np.maximum(c, 1), allm)
            nb = np.convolve(np.pad(raw, 1, mode="edge"), [1 / 3, 1 / 3, 1 / 3], mode="valid")
            T[n, 1:n + 1] = (s + SHRINK * nb) / (c + SHRINK)
        out.append(T)
    return out[0], out[1]


def main() -> None:
    out_path = Path(sys.argv[1])
    h = pd.read_parquet(HIST)
    h["race_id"] = h["race_id"].astype(str)
    h["horse_id"] = h["horse_id"].astype(str)
    h["date"] = pd.to_datetime(h["date"])
    axes = [a for a in rm.AXES if a[0] in h.columns and a[0] != "split"]
    meta_axes = [{"key": a[0], "label": RELABEL.get(a[0], a[3]), "high": a[4], "low": a[5]} for a in axes]
    if "slope" in h.columns:
        meta_axes.append({"key": "slope", "label": EXTRA_LABELS["slope"][0], "high": EXTRA_LABELS["slope"][1], "low": EXTRA_LABELS["slope"][2]})
    keys = [a["key"] for a in meta_axes]

    res = load_results(range(2016, 2027))
    Tw, Tp = expected_tables(res[res["race_id"].str[:4].astype(int).isin(list(EXP_YEARS))])

    df = h.merge(res[["race_id", "horse_id", "popularity", "field_size", "track_condition"]], on=["race_id", "horse_id"], how="left")
    df = df[(df["finish_position"] > 0) & (df["field_size"] >= N_MIN_FIELD) & (df["popularity"] >= 1) & df["surface"].isin(["芝", "ダート"])].copy()
    df["period"] = np.select([df["date"].between(*map(pd.Timestamp, PERIODS["A"])), df["date"] >= pd.Timestamp(PERIODS["B"][0])], [0, 1], -1)
    df = df[df["period"] >= 0].reset_index(drop=True)
    nf = df["field_size"].to_numpy().astype(int)
    nb = np.clip(nf, N_MIN_FIELD, 18)
    pp = np.minimum(df["popularity"].to_numpy().astype(int), nb)
    stat = {"n": np.ones(len(df)), "w": (df["finish_position"] == 1).to_numpy(float),
            "p": (df["finish_position"].to_numpy() <= place_cut(nf)).astype(float), "ew": Tw[nb, pp], "ep": Tp[nb, pp]}
    order = ["n", "w", "p", "ew", "ep"]
    per = df["period"].to_numpy().astype(int)
    bins = {k: np.minimum(np.floor(df[k].to_numpy(float)), 10).clip(1, 10) - 1 for k in keys}
    ok = {k: ~np.isnan(df[k].to_numpy(float)) & ((df["surface"] == ONLY_SURFACE[k]).to_numpy() if k in ONLY_SURFACE else True) for k in keys}

    surf = df["surface"].map({"芝": "芝", "ダート": "ダ"})
    df["cls"] = df["grade"].map(class_of)
    df["going"] = df["track_condition"].map(GOING)
    df["dband"] = [f"{s}{dist_band(d)}" for s, d in zip(surf, df["distance"])]
    df["course"] = [f"{v}{s}{int(d)}m" for v, s, d in zip(df["venue"], surf, df["distance"])]
    df["sg"] = [f"{s}・{g}" if isinstance(g, str) else None for s, g in zip(surf, df["going"])]

    def cell(mask: np.ndarray) -> dict:
        r = {}
        for k in keys:
            m = mask & ok[k]
            if m.sum() == 0:
                continue
            idx = per[m] * 10 + bins[k][m].astype(int)
            a = np.stack([np.bincount(idx, weights=stat[s][m], minlength=20) for s in order])   # 5 × 20（期間 A の 10 区間 → 期間 B の 10 区間）
            r[k] = {"A": [[int(x) for x in a[0, :10]], [int(x) for x in a[1, :10]], [int(x) for x in a[2, :10]], [round(float(x), 1) for x in a[3, :10]], [round(float(x), 1) for x in a[4, :10]]],
                    "B": [[int(x) for x in a[0, 10:]], [int(x) for x in a[1, 10:]], [int(x) for x in a[2, 10:]], [round(float(x), 1) for x in a[3, 10:]], [round(float(x), 1) for x in a[4, 10:]]]}
        return r

    def group_list(col: str, labels: list[str]) -> list[dict]:
        return [{"key": s, "label": s, "data": cell((df[col] == s).to_numpy())} for s in labels if (df[col] == s).any()]

    dims: dict[str, dict] = {}
    dims["all"] = {"label": "全体", "groups": [{"key": "全体", "label": "全体", "data": cell(np.ones(len(df), bool))}]}
    dims["class"] = {"label": "クラス", "groups": group_list("cls", [c for c, _ in CLASS_BANDS])}
    dims["going"] = {"label": "馬場（芝ダ × 馬場状態）", "groups": group_list("sg", [f"{s}・{g}" for s in ("芝", "ダ") for g in GOING_ORDER])}
    dims["dist"] = {"label": "距離（芝ダ × 距離帯）", "groups": group_list("dband", [f"{s}{b}" for s in ("芝", "ダ") for b in DIST_ORDER])}
    cn = df[df["period"] == 0].groupby(["course", "surface", "venue", "distance"]).size().reset_index(name="n")
    cn = cn[cn["n"] >= COURSE_MIN_RUNS].sort_values(["surface", "venue", "distance"], ascending=[False, True, True])
    dims["course"] = {"label": "コース（場 × 芝ダ × 距離）", "groups": group_list("course", list(cn["course"]))}

    meta = {"built": date.today().isoformat(), "history_file": str(HIST), "history_mtime": pd.Timestamp(HIST.stat().st_mtime, unit="s").isoformat(),
            "periods": PERIODS, "runs": {"A": int((per == 0).sum()), "B": int((per == 1).sum())},
            "axis_runs": {k: {"A": int((ok[k] & (per == 0)).sum()), "B": int((ok[k] & (per == 1)).sum())} for k in keys},
            "going_values": sorted(str(x) for x in df["track_condition"].dropna().unique()), "grades": sorted(str(x) for x in df["grade"].dropna().unique()),
            "exp_years": [EXP_YEARS.start, EXP_YEARS.stop - 1], "course_min_runs": COURSE_MIN_RUNS}
    out_path.write_text(json.dumps({"meta": meta, "axes": meta_axes, "dims": dims}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("wrote", out_path, f"{out_path.stat().st_size:,} bytes", meta["runs"])


if __name__ == "__main__":
    main()
