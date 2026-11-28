"""血統の合い具合の列の評価（計画の第 2 版。物差し・評価の行・Benter・中止の馬の扱いは model_relabel.mr_eval の関数をそのまま使う）。

    KEIBA_SESSION=aptitude-group OMP_NUM_THREADS=2 flock /tmp/claude-1000/heavy.lock nice -n 19 python -m src.research.aptitude_group.megu_eval gate0
    ... megu_eval main        # 門 0 を越えたときだけ（本物・偽 3 組・形）

- 比べ: 足す先 v5bag_mfs（model-relabel の基準・s_* を外した版）の勝ちの元のラベル（orig）の予測と、列を足した版の予測。
  選び（本数・較正）は mr_eval.nested_time と同じ入れ子（t−1 年で選ぶ・t−2 年で較正）。Benter の α・β は t−1 年で当てて t 年で数える。
- 評価の年 2019〜24（読みの主は 2021〜24）。区間は開催日で束ねた対のブートストラップ（mr_eval._boot・2,000 回）。
  _boot の ci_bonf12（族 12 個の割引き）は model-relabel の作りで、この研究では読まない（ci95 だけを読む）。
- 門 0: 偽 1 組（null101）と基準の対の差の SE から、見つけられる最小の差 MDE = 2.8 × SE。MDE > 0.0005 なら本物を流さずに閉じる。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.aptitude_group import common
from src.research.model_relabel import mr_eval as ME
from src.research.model_relabel import mr_train as TR
from src.research.pedigree_lab import data

YEARS = list(range(2019, 2025))
LAM = 0.5
BASE_FS = "v5bag_mfs"
MDE_LINE = 0.0005


def evaluate(fs: str, base: pd.DataFrame, sp: dict) -> dict:
    """1 つの特徴量の版の、外側の年ごとの選んだ予測と Benter の行（族は orig だけ）。"""
    rounds = sp["select"]["rounds_v5bag"]
    pr = ME.load_pred(fs, "win", "orig", base)
    fam = {"base": {f"orig|r{r}": pr[f"p_r{r}"].to_numpy(float) for r in rounds}}
    import copy
    sp2 = copy.deepcopy(sp)
    sp2["penalty"]["P_w"]["w"] = []                          # 固定の w の群を作らない（orig の族だけ・critic 7 回目の 1）
    orig = ME.candidates
    ME.candidates = lambda *a, **k: fam                      # mr_eval の入れ子の選びを、orig の族だけで使う
    try:
        res = ME.nested_time(fs, "win", YEARS, [LAM], sp2, base)[LAM]
    finally:
        ME.candidates = orig
    return res


def compare(res_a: dict, res_b: dict, base: pd.DataFrame, years) -> dict:
    """a − b の差（負 = a が良い）: Benter の 2 段の勝ち馬の −log p と、モデルだけの勝ち馬の −log p。中止の馬の 3 通り。"""
    out = {"benter": ME.paired_boot_benter(res_a, "base", res_b, "base", years),
           "model": ME.paired_boot(res_a, base, "win", "base", "base", years, res_b=res_b)}
    comb = {"years": {t: {"base": res_b["years"][t]["base"], "add": res_a["years"][t]["base"]} for t in years}}
    dn = ME.dnf_sensitivity(comb, base, "win", years)
    out["dnf"] = {h: dn[f"{h}_minus_base"]["add"] for h in ("none", "min", "median")}
    out["dnf_direction_agree"] = dn["direction_agree"]["add"]
    return out


def se_of(ci: list[float]) -> float:
    return (ci[1] - ci[0]) / 3.92


def cmd_gate0() -> dict:
    sp = TR.spec()
    base = ME.base_frame(2024)
    rb = evaluate(BASE_FS, base, sp)
    rn = evaluate("v5bag_mfs_aptnull101", base, sp)
    c = compare(rn, rb, base, YEARS)
    se = se_of(c["benter"]["ci95"])
    mde = 2.8 * se
    out = {"null101_minus_base": c, "se_benter": round(se, 6), "mde": round(mde, 6), "line": MDE_LINE,
           "go": bool(mde <= MDE_LINE), "years": YEARS}
    print(json.dumps(out, ensure_ascii=False, default=str), flush=True)
    return out


def cmd_main() -> dict:
    sp = TR.spec()
    base = ME.base_frame(2024)
    rb = evaluate(BASE_FS, base, sp)
    out: dict = {"years": YEARS}
    res = {fs: evaluate(fs, base, sp) for fs in ("v5bag_mfs_apt", "v5bag_mfs_aptnull101", "v5bag_mfs_aptnull102", "v5bag_mfs_aptnull103")}
    for fs, r in res.items():
        out[f"{fs}_minus_base"] = compare(r, rb, base, YEARS)
        out[f"{fs}_minus_base_2021_24"] = ME.paired_boot_benter(r, "base", rb, "base", [2021, 2022, 2023, 2024])
    # 本物 − 偽の平均（Benter の行で、偽 3 組の nll の平均を b にする）
    rows = []
    for t in YEARS:
        ra = res["v5bag_mfs_apt"]["years"][t]["base"]["_b"]
        nl = [res[f"v5bag_mfs_aptnull{s}"]["years"][t]["base"]["_b"][["key", "nll"]] for s in (101, 102, 103)]
        m = ra[["key", "date", "nll"]]
        for i, x in enumerate(nl):
            m = m.merge(x.rename(columns={"nll": f"n{i}"}), on="key")
        rows.append(pd.DataFrame({"date": m["date"], "a": m["nll"], "b": m[["n0", "n1", "n2"]].mean(1)}))
    out["real_minus_nullmean"] = ME._boot(pd.concat(rows), 2000, 20261128)
    # 形: レースの級（2・3 歳の未勝利・新馬・それ以外）ごとの Benter の改善（レースの単位・全頭の 2 段の nll）
    out["shape"] = shape(res["v5bag_mfs_apt"], rb)
    print(json.dumps(out, ensure_ascii=False, default=str), flush=True)
    return out


def shape(ra: dict, rb: dict) -> dict:
    """レースの級ごとの Benter の 2 段の勝ち馬の −log p の差（a − b）。級は race_result の race_class の文字から規則で決める。"""
    cls = {}
    for y in range(2019, 2025):
        r = pd.read_parquet(f"{common.TABLES}/{y}/race_result_flat.parquet", columns=["race_id", "race_class"]).drop_duplicates("race_id")   # 2019 年の race 表には race_class が無い
        for rid, rc in zip(r["race_id"].astype(str), r["race_class"].fillna("")):
            young = ("２歳" in rc) or ("３歳" in rc and "以上" not in rc)
            cls[rid] = "新馬" if "新馬" in rc else "2・3 歳の未勝利" if (young and "未勝利" in rc) else "それ以外"
    out = {}
    for name in ("2・3 歳の未勝利", "新馬", "それ以外"):
        a = b = n = 0.0
        for t in YEARS:
            x = ra["years"][t]["base"]["_b"].merge(rb["years"][t]["base"]["_b"][["key", "nll"]], on="key", suffixes=("", "_b"))
            k = (x["key"].astype(str).str.split("_").str[0].map(cls) == name).to_numpy()     # 鍵は race_id_horse_id（mr_eval.benter_rows）
            a += float(x.loc[k, "nll"].sum())
            b += float(x.loc[k, "nll_b"].sum())
            n += float(k.sum())
        out[name] = {"diff": round((a - b) / max(n, 1), 6), "n": int(n)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate0", "main"])
    a = ap.parse_args()
    common.open_token("select", f"megu_eval_{a.cmd}_v1", "血統の合い具合の列の評価（勝ち・Benter・2019〜24・回収率は数えない）")
    out = cmd_gate0() if a.cmd == "gate0" else cmd_main()
    p = common.OUT / f"megu_eval_{a.cmd}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    data.finish_all(str(p))


if __name__ == "__main__":
    main()
