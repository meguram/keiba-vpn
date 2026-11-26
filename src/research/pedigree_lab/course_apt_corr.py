"""コース適性点とダートの牝系の点の相関（着順で当てない 0 円の確かめ。persona_design.md 付録・ユーザーの決定 4）。

    KEIBA_SESSION=pedigree-persona OMP_NUM_THREADS=4 nice -n 19 /opt/conda/bin/python -m src.research.pedigree_lab.course_apt_corr

数を見る前に決めた作り（2026-11-26・pedigree-persona。critic の指摘を入れた版。相関の数はまだ 1 つも見ていない）:
- コース c = 場 × 芝ダ × 距離。京都は改修の前（2020-11-01 より前）を「京都旧」として別のコースにする。
- 群 A_c = コース c の行。父・母父・牝系 5 代の適性を、aptitude.py の凍結の関数（apt_by・maternal_long）と
  aptitude_checks.mat_cols のまま作る（鍵は芝ダで分けない。ダートの牝系 5 代と同じ作り）。値はコース c の行にだけ配る。
- コース適性点 C = 3 列それぞれをレースの中で中心化し、測る行の sd で割って 1:1:1 で足した値（当てはめない）。
  0〜100 の点は C の順位を写すだけなので、相関は C で測る。ダートの側も同じ作りで D = 父 + 母父 + 牝系 5 代。
- 測る行: 前向きの登録の開発の行と同じ、2017〜24 年の split 0・1 のダートの行。2016 年を含む版は副。
  血統の無い馬は、生の値を 0 にしてから中心化する（開発の行と同じ。登録の判定の「中心化の後に 0」とは違う）。
- 主（打ち切りの判定）: Pearson の r（C, D）。同じ作りどうしの比べ。r ≥ 0.8 なら付録は打ち切り。
  ユーザーの決定の文面どおりの r（C, dirt_mat5）は並べて書くが、C の 3 分の 1 しか同じ列が無いので、上限は約 0.58（1/√3）で、
  打ち切りの判定には使わない（critic 1）。r（コースの牝系 5 代, dirt_mat5）も並べる。
- 非対称の注記: コースごとの件数は少なく K=200 で 0 へ強く縮むので、r < 0.8 でも「コース適性点に独自の情報がある」とは言えない（critic 2）。
- 着順は as-of の産駒の成績の集計（研究の列と同じ）にだけ使い、行の着順で当てはめない。2025 年以降は読まない（base_frame が切る）。
- 台帳の担当名は pl_ledger.py の直書きで pedigree-lab になる（凍結の一覧のファイルなので直さない）。reason に担当名を書く。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.research.pedigree_lab import aptitude as ap_
from src.research.pedigree_lab import aptitude_checks as ck
from src.research.pedigree_lab import data

KYOTO_NEW = pd.Timestamp("2020-11-01")


def course_key(F: pd.DataFrame) -> np.ndarray:
    v = F["venue"].astype(str).to_numpy().copy()
    v[(v == "京都") & (F["date"] < KYOTO_NEW).to_numpy()] = "京都旧"
    return pd.Series(v).str.cat([F["surf"].astype(str).to_numpy(), F["distance"].astype(int).astype(str).to_numpy()], sep="|").to_numpy()


def summarize(S: pd.DataFrame, cols: dict[str, np.ndarray]) -> dict:
    """S の行で、6 列をレースの中で中心化・sd で割り、相関を出す。"""
    rid = S["race_id"].to_numpy()

    def z(x: np.ndarray) -> np.ndarray:
        v = x - pd.Series(x).groupby(rid).transform("mean").to_numpy()
        sd = v.std()
        return v / (sd if sd > 0 else 1.0)
    Z = {k: z(v) for k, v in cols.items()}
    C = Z["c_sire"] + Z["c_bms"] + Z["c_mat5"]
    D = Z["d_sire"] + Z["d_bms"] + Z["d_mat5"]

    def r(a, b) -> float:
        return round(float(np.corrcoef(a, b)[0, 1]), 4)
    by = {}
    keys = course_key(S)
    for c in sorted(set(keys)):
        ii = np.where(keys == c)[0]
        if len(ii) >= 1000:
            by[c] = [int(len(ii)), r(C[ii], D[ii])]
    return {
        "rows": int(len(S)),
        "primary_pearson_C_vs_D": r(C, D),
        "spearman_C_vs_D": round(float(pd.Series(C).corr(pd.Series(D), method="spearman")), 4),
        "literal_pearson_C_vs_dirt_mat5（上限 約 0.58）": r(C, Z["d_mat5"]),
        "pearson_course_mat5_vs_dirt_mat5": r(Z["c_mat5"], Z["d_mat5"]),
        "pearson_course_sire_vs_dirt_sire": r(Z["c_sire"], Z["d_sire"]),
        "pearson_course_bms_vs_dirt_bms": r(Z["c_bms"], Z["d_bms"]),
        "by_course_C_vs_D": ({"min": min(v[1] for v in by.values()), "max": max(v[1] for v in by.values()), "rows": by} if by else {}),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="時間を測るための試し: このコースだけ作る（台帳を開かない・結果は書かない）")
    a = p.parse_args()
    if not a.only:
        data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "course_apt_corr",
                                       "コース適性点とダートの牝系 5 代の点の相関（着順で当てない・記述。r ≥ 0.8 なら付録を打ち切る。担当 pedigree-persona）"))
    F = ap_.base_frame()
    L = ap_.maternal_long(F, drop_common=False)
    ckey = course_key(F)
    dirt_rows = (F["surf"] == "D").to_numpy()
    courses = sorted(set(ckey[dirt_rows]))
    if a.only:
        t0 = time.time()
        A = ckey == a.only
        ap_.apt_by(F, A, ["sire"])
        ck.mat_cols(F, A, [], L)
        print(f"試し {a.only}: {time.time() - t0:.0f} 秒（全 {len(courses)} 本）", flush=True)
        return
    # ダートの 3 列（前向きの登録と同じ作り: forward_record.dirt_aptitude）
    A_d, keys_d = ap_.group_a(F, "dirt")
    cols = {"d_sire": ap_.apt_by(F, A_d, ["sire"] + keys_d), "d_bms": ap_.apt_by(F, A_d, ["ds"] + keys_d)}
    cols["d_mat5"], _ = ck.mat_cols(F, A_d, keys_d, L)
    # コースごとの 3 列（値はそのコースの行にだけ配る）。相関はダートの行で測るので、ダートのコースだけ作る
    for k in ("c_sire", "c_bms", "c_mat5"):
        cols[k] = np.full(len(F), np.nan)
    for i, c in enumerate(courses):
        A = ckey == c
        s = ap_.apt_by(F, A, ["sire"])
        b = ap_.apt_by(F, A, ["ds"])
        m, _ = ck.mat_cols(F, A, [], L)
        cols["c_sire"][A], cols["c_bms"][A], cols["c_mat5"][A] = s[A], b[A], m[A]
        print(f"[{i + 1}/{len(courses)}] {c} 行 {int(A.sum())}", flush=True)
    dev = dirt_rows & F["split"].isin([0, 1]).to_numpy()
    main_rows = dev & (F["date"] >= pd.Timestamp("2017-01-01")).to_numpy()
    res = {
        "courses": len(courses),
        "rule": "主の r（C, D）≥ 0.8 なら付録（コース適性点の買い判断）を打ち切る。r < 0.8 でも独自の情報があるとは言えない（K=200 の縮み）",
        "main_2017_2024": summarize(F[main_rows].reset_index(drop=True), {k: v[main_rows] for k, v in cols.items()}),
        "sub_2016_2024": summarize(F[dev].reset_index(drop=True), {k: v[dev] for k, v in cols.items()}),
    }
    res["stop_appendix"] = res["main_2017_2024"]["primary_pearson_C_vs_D"] >= 0.8
    path = data.OUT / "course_apt_corr.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
