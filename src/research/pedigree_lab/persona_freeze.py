"""血統予想家の閾値と芝の sd を作って凍結する（1 回だけ流す研究の CLI。着順は読まない・記述）。

    KEIBA_SESSION=pedigree-persona OMP_NUM_THREADS=4 nice -n 19 /opt/conda/bin/python -m src.research.pedigree_lab.persona_freeze

数を見る前に決めた作り（2026-11-27・pedigree-persona。persona_impl_plan.md の「閾値」）:
- 列: ダートは dirt_mat5（前向きの記録と同じ mat_cols）、芝は turf_mat5（群 A = 芝の行）。base_frame（2024 年まで）で as-of に作る。
- sd: ダートは前向きの登録の開発の定数 0.00539186。芝は同じ作り（2017〜24 年の split 0・1 の芝の行・レースの中で中心化）で 1 回測る。
  ダートも同じ作りで測り、登録の定数との比を確かめに残す（0.98〜1.02 の外なら止める）。
- 閾値の母集団: 2024 年の split 0・1 の行（保留の split 2 は使わない）で、新馬戦を除き、確定の単勝オッズがある行。
  T-60 のオッズは再現（pre_odds.simulate・5 回）。人気は rules.pop_rank、点は rules.score（推論と同じ関数）。
  6 番人気以下の点を 5 回ぶん合わせ、上位 10%（90 分位）を閾値にする。芝ダごとに 1 つ。
- 出力: config/pedigree_persona/frozen.json（rules.py の sha256 を入れる）。満期まで作り直さない。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.research.pedigree_lab import aptitude as ap_
from src.research.pedigree_lab import aptitude_checks as ck
from src.research.pedigree_lab import data, pre_odds

DIRT_SD = 0.00539186


def main() -> None:
    from src.research.pedigree_persona import columns as pc
    from src.research.pedigree_persona import rules as R

    pc.check_frozen()
    if R.FROZEN_PATH.exists():
        raise SystemExit(f"{R.FROZEN_PATH} は既にある（凍結は作り直さない。直すなら版を分ける）")
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "persona_freeze",
                                   "血統予想家の閾値（2024 年の 6 番人気以下の上位 10%）と芝の sd の凍結。着順は読まない（as-of の祖先の集計にだけ使う）。担当 pedigree-persona"))
    F = ap_.base_frame()
    extra = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "popularity", "cls"])
    F = F.merge(extra.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    L = ap_.maternal_long(F, drop_common=False)
    raw = {}
    A_d, keys_d = ap_.group_a(F, "dirt")
    raw["ダート"], _ = ck.mat_cols(F, A_d, keys_d, L)
    raw["芝"], _ = ck.mat_cols(F, (F["surf"] == "T").to_numpy(), [], L)
    no_ped = (F["sire"].isna() | F["sire"].eq("")).to_numpy()
    surf = F["surf"].map({"D": "ダート", "T": "芝"}).to_numpy()
    dev = F["split"].isin([0, 1]).to_numpy() & (F["date"] >= pd.Timestamp("2017-01-01")).to_numpy()
    sd = {}
    for s in ("ダート", "芝"):
        m = dev & (surf == s)
        v = raw[s][m] - pd.Series(raw[s][m]).groupby(F["race_id"].to_numpy()[m]).transform("mean").to_numpy()
        sd[s] = float(v.std())
    ratio = sd["ダート"] / DIRT_SD
    if not 0.98 <= ratio <= 1.02:
        raise SystemExit(f"ダートの sd が登録の定数と合わない（比 {ratio:.4f}）。作りがずれている")
    pool = pre_odds.load_drift("t60")
    thr, counts = {}, {}
    for s in ("ダート", "芝"):
        m = (F["date"].dt.year == 2024).to_numpy() & F["split"].isin([0, 1]).to_numpy() & (surf == s) & F["cls"].ne("NEW").to_numpy() \
            & (pd.to_numeric(F["odds"], errors="coerce") > 0).to_numpy()
        S = F[m].assign(_raw=raw[s][m], _np=no_ped[m]).sort_values(["race_id", "horse_number"], kind="stable").reset_index(drop=True)
        rid = S["race_id"].to_numpy()
        z = R.score(S["_raw"].to_numpy(), rid, S["_np"].to_numpy(), DIRT_SD if s == "ダート" else sd["芝"])
        sims = pre_odds.simulate(rid, S["odds"].to_numpy(float), pd.to_numeric(S["popularity"], errors="coerce").to_numpy(float), pool, n=5)
        zz = np.concatenate([z[R.pop_rank(rid, o, S["horse_number"].to_numpy()) >= R.POP_MIN] for o in sims])
        thr[s] = float(np.quantile(zz, 1 - R.TOP_FRAC))
        counts[s] = {"rows": int(len(S)), "races": int(pd.Series(rid).nunique()), "pop6_rows_per_sim": round(len(zz) / 5, 1)}
    doc = {
        "version": R.RULES_VERSION, "made_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
        "rules_sha256": R.file_sha256(R.Path(R.__file__)),
        "code_sha256": {"persona_freeze.py": R.file_sha256(R.Path(__file__)), "columns.py": R.file_sha256(R.Path(pc.__file__))},
        "sd": {"ダート": DIRT_SD, "芝": round(sd["芝"], 8)}, "sd_dirt_measured": round(sd["ダート"], 8), "sd_dirt_ratio": round(ratio, 4),
        "threshold": {k: round(v, 6) for k, v in thr.items()},
        "population": "2024 年の split 0・1・新馬戦を除く・T-60 は再現（pre_odds.simulate 5 回）・6 番人気以下・上位 10%",
        "counts": counts, "pop_min": R.POP_MIN, "top_frac": R.TOP_FRAC, "max_picks": R.MAX_PICKS,
    }
    R.FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    R.FROZEN_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False), flush=True)
    data.finish_all(str(R.FROZEN_PATH))


if __name__ == "__main__":
    main()
