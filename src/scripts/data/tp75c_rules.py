"""tp-75 第 2 ラウンドの追加（2c）: 血統の系統ルールを、完全な母方 10 世代表で再測定する。

python -m src.scripts.data.tp75c_rules prereg     # 原子条件・列挙規則・候補総数を tp75c_prereg.json に書く（評価の前にコミットする。結果の列は使わない）
python -m src.scripts.data.tp75c_rules eval       # 検定・採用・クラスタ・ペルソナ（第 1・第 2 ラウンドの採用と合算）の再評価
python -m src.scripts.data.tp75c_rules reeval FROZEN.json [--years 2026]

前提: 共有表 maternal_lineage.parquet（tp75c_maternal.py build）。列は血統表の構造だけから作った（成績・オッズは使わない）。
父方の系統は対照（sire_main / sire_coarse だけ）。旧ルール（不完全な表に基づく母母父 dds_*）は台帳に『不完全な表（旧）』として残し、完全な表の結果と並べる。
多重比較は、第 1 ラウンドの検定可能なルールと 2c の候補の合算（族ごと）。全ラウンド合算（第 2 ラウンドの候補も足した）の q も併記する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp75_rules as T1
from src.scripts.data import tp75b_rules as T2

HERE = Path(__file__).parent
OUT = T2.OUT
PREREG_C = HERE / "tp75c_prereg.json"
MAT = Path(os.environ.get("TP75C_OUT", "/home/jovyan/work/keiba-vpn/data/local/modeling/shared_cache/maternal_lineage")) / "maternal_lineage.parquet"
MIN_RUNS_C = int(os.environ.get("TP75C_MIN_RUNS", "400"))
MIN_RACES_C = int(os.environ.get("TP75C_MIN_RACES", "120"))
DEEP_Q = 0.80
CONTEXT_DIMS = {"venue", "surface", "dist", "going", "season", "cls", "sex", "age", "bracket", "field", "lay", "style", "geld", "jockey", "p_dist", "p_surf", "p_cls", "p_fin"}     # L,C の文脈
CONTEXT_DIMS_TRIPLE = {"venue", "surface", "dist", "going", "season", "cls"}     # L,C,C' の文脈（組合せの爆発を避けるため少数に絞る）
MAT_COLS_PREFIX = ("mmf_", "mmmf_", "mff_", "near_", "deep_", "share_near_", "share_deep_", "hf", "mat_cl", "mat_complete")


def make_extra_c():
    base = T2.make_extra()

    def extra(d: pd.DataFrame) -> pd.DataFrame:
        d = base(d)
        m = pd.read_parquet(MAT)
        keep = ["horse_id"] + [c for c in m.columns if c != "horse_id" and c.startswith(MAT_COLS_PREFIX)]
        return d.merge(m[keep], on="horse_id", how="left")

    return extra


def atom_specs_c(d: pd.DataFrame) -> tuple[list[dict], dict]:
    """2c の原子条件。kind = L（血統の系統）/ C（文脈）。new = 第 2 ラウンドに無い原子。"""
    r2_names = {a["name"] for a in json.loads((HERE / "tp75b_prereg.json").read_text(encoding="utf-8"))["atoms"]}
    base = T2.atom_specs(d)
    A: list[dict] = []

    def add(kind: str, dim: str, name: str, expr: str) -> None:
        A.append({"name": name, "dim": dim, "expr": expr, "kind": kind, "new": name not in r2_names})

    for a in base:
        if a["dim"] == "dsire_line":
            add("L", a["dim"], a["name"], a["expr"])
        elif a["dim"] == "sire_line" and (a["name"].startswith("sire_main=") or a["name"].startswith("sire_coarse=")):
            add("L", a["dim"], a["name"], a["expr"])          # 父方は対照（main・coarse だけ）
        elif a["dim"] in CONTEXT_DIMS:
            add("C", a["dim"], a["name"], a["expr"])
    # 母母父・母母母父・母父父の系統（完全な母方 10 世代表から）
    for key, dim in (("mmf", "mmf_line"), ("mmmf", "mmmf_line"), ("mff", "mff_line")):
        for lvl in ("main", "coarse", "fine"):
            col = f"{key}_{lvl}"
            if col in d.columns and not (key != "mmf" and lvl == "fine"):
                for v in sorted(d[col].dropna().astype(str).unique()):
                    add("L", dim, f"{col}={v}", f'{col}=="{v}"')
    levels = json.loads((MAT.parent / "README.json").read_text(encoding="utf-8"))["coarse_levels"]
    cuts = {}
    uniq = d.drop_duplicates("horse_id")
    for li, lv in enumerate(levels):
        if lv == "その他":
            continue
        add("L", f"near_L{li:02d}", f"mat_near_{lv}", f"near_L{li:02d}>=1")
        cut = float(uniq.loc[uniq["mat_complete"] == True, f"share_deep_L{li:02d}"].quantile(DEEP_Q))      # noqa: E712
        cuts[lv] = round(cut, 4)
        if cut > 0:
            add("L", f"deep_L{li:02d}", f"mat_deep_heavy_{lv}", f"share_deep_L{li:02d}>={cut:.4f}")
    for c in d.columns:
        if c.startswith("hf") and c.endswith("_present"):
            k = c[2:4]
            for typ in ("present", "cross", "deeponly"):
                add("L", f"hf{k}", f"mat_hf{k}_{typ}", f"hf{k}_{typ}==1")
    for K in (8, 24):
        for i in range(K):
            add("L", "mat_cl", f"mat_cl{K}_{i}", f"mat_cl{K}=={i}")
    return A, {"deep_cuts": cuts, "deep_quantile": DEEP_Q}


def enumerate_c(M: np.ndarray, dims: list[str], kinds: list[str], new: list[bool], rc: np.ndarray, disc: np.ndarray, min_runs: int, min_races: int, known: set[bytes]) -> dict:
    """形: L / L,L' / L,C / L,C,C'（L = 血統の系統、C = 文脈。C,C' は CONTEXT_DIMS_TRIPLE の次元だけ）。第 2 ラウンドに無い原子を 1 つ以上含むものだけ。"""
    Md, rcd = M[:, disc], rc[disc]
    n = M.shape[0]
    st = {"atoms": n, "frequent_atoms": 0, "pairs": 0, "triples": 0, "dropped_no_new_atom": 0, "duplicates_dropped": 0}
    f1 = [i for i in range(n) if Md[i].sum() >= min_runs and T2._races(rcd, Md[i]) >= min_races]
    st["frequent_atoms"] = len(f1)
    L = [i for i in f1 if kinds[i] == "L"]
    Cc = [i for i in f1 if kinds[i] == "C"]
    C2 = [i for i in Cc if kinds[i + 0] == "C" and dims[i] in CONTEXT_DIMS_TRIPLE]
    cands: list[tuple[int, ...]] = []
    seen = set(known)

    def ok_add(c: tuple[int, ...], mask: np.ndarray) -> None:
        if not any(new[i] for i in c):
            st["dropped_no_new_atom"] += 1
            return
        h = hashlib.md5(np.packbits(mask).tobytes()).digest()
        if h in seen:
            st["duplicates_dropped"] += 1
            return
        seen.add(h)
        cands.append(tuple(sorted(c)))

    def sup_ok(m: np.ndarray) -> bool:
        return bool(np.count_nonzero(m) >= min_runs and T2._races(rcd, m) >= min_races)

    pair: set[tuple[int, int]] = set()
    for i in L:
        ok_add((i,), Md[i])
    for a_, i in enumerate(L):
        for j in L[a_ + 1:] + Cc:
            if dims[i] != dims[j]:
                m = Md[i] & Md[j]
                if sup_ok(m):
                    pair.add((min(i, j), max(i, j)))
                    ok_add((i, j), m)
    for a_, i in enumerate(C2):
        for j in C2[a_ + 1:]:
            if dims[i] != dims[j] and sup_ok(Md[i] & Md[j]):
                pair.add((min(i, j), max(i, j)))
    st["pairs"] = len(pair)
    for (i, j) in sorted(pair):
        if i in set(C2) and j in set(C2) and kinds[i] == "C" and kinds[j] == "C":
            mij = Md[i] & Md[j]
            for k in L:
                if dims[k] in (dims[i], dims[j]):
                    continue
                if (min(i, k), max(i, k)) in pair and (min(j, k), max(j, k)) in pair:
                    m = mij & Md[k]
                    if sup_ok(m):
                        st["triples"] += 1
                        ok_add((i, j, k), m)
    st["candidates"] = len(cands)
    return {"candidates": cands, "stats": st}


def prereg_c() -> dict:
    d = T1.load_table(extra=make_extra_c())
    specs, cuts = atom_specs_c(d)
    M = T2.atom_masks(d, specs)
    disc = np.isin(d["year"].to_numpy(), T1.DISC)
    rc = d["rc"].to_numpy()
    en = enumerate_c(M, [a["dim"] for a in specs], [a["kind"] for a in specs], [a["new"] for a in specs], rc, disc, MIN_RUNS_C, MIN_RACES_C, T2.known_round1_hashes(d, disc))
    names = sorted(T2.cand_name(specs, c) for c in en["candidates"])
    sup = M[:, disc].sum(1)
    pre2 = json.loads((HERE / "tp75b_prereg.json").read_text(encoding="utf-8"))
    spec = {
        "version": 1, "round": "2c",
        "design": {
            "purpose": "血統の系統ルール（母系・母父・母母父の系統、母方の高頻度種牡馬、母方クラスタ）を、完全な母方 10 世代表（99.99%）で再測定する。父方の系統は対照（sire_main・sire_coarse だけ）。",
            "market_free": "原子条件は血統表の構造と事前に分かる文脈だけ。人気・オッズ・市場確率・想定オッズ・期待値は使わない。",
            "source_table": "data/local/modeling/shared_cache/maternal_lineage/maternal_lineage.parquet（tp75c_maternal.py）。完全な母方 10 世代表（path_fm が M で始まる 1,023 位置）から作成。",
            "forms": ["L", "L,L'", "L,C", "L,C,C'"], "form_note": "L = 血統の系統の原子、C = 文脈の原子（L,C の C: 競馬場・面・距離・馬場・季節・クラス・性別・年齢・枠・頭数・間隔・脚質・去勢明け・騎手・前走の距離/面/クラス/着順。L,C,C' の C: 競馬場・面・距離・馬場・季節・クラスだけ）。L,L',C と 4 条件以上は、組合せの爆発と標本不足のため打ち切り（最初の試算で L,L',C を含めると候補が数百万に膨らみ、評価できないため。試算は結果を使わない）。",
            "enumeration": {"method": "Apriori 型の頻出集合（サブセットがすべて頻出のものだけ延ばす）。同じ次元の原子は同じ連言に入れない", "min_runs_discovery": MIN_RUNS_C, "min_races_discovery": MIN_RACES_C,
                            "support_uses": "発見期間（2022–24）の該当走数・レース数だけ（結果は使わない）", "dedup": "第 1 ラウンドのルールと該当行が同じものは除く。第 2 ラウンドに無い原子を 1 つ以上含むものだけ（第 2 ラウンドの原子だけの連言は第 2 ラウンドで検定済み）"},
            "deep_heavy": f"深い母系（世代 6〜10）で系統 X の割合が、母方が完全な全出走馬の上位 {int((1 - DEEP_Q) * 100)}% 以内（構造だけで決めた閾値）",
            "high_frequency_sires": "母方 10 世代に現れる割合が 5〜60% の種牡馬のうち、割合の高い順に 20 頭（構造だけ）。現れる／2 回以上（クロス）／世代 6〜10 にだけ現れる",
            "clusters": "近い・深い母系の系統（coarse）の割合の KMeans（K=8, 24、乱数の種 20261022、母方が完全な全出走馬で当てはめ。構造だけ）",
            "adoption": "第 2 ラウンドと同じ（残差、A/B、再現、増分（親・第 1 ラウンド・第 2 ラウンドの最終採用）、クラスタ）。BH の母数は第 1 ラウンドの検定可能 624 型と 2c の候補の合算（族ごと）。全ラウンド合算（第 2 ラウンドの候補を足した）の q も併記",
            "usable": "順位付け・凍結に使うのは、最終採用のうち BH 頑健（q<0.10）なものだけ。第 2 ラウンドで、A/B・再現・増分だけの採用が 2025 年の標本外で偶然の期待とほぼ同じだった（同符号 52%）一方、BH 頑健なものだけが再現した（67〜83%）ことを受けた、第 2 ラウンドの結果を見た後の取り決め。2c の結果を見る前に固定する",
            "old_rules": "不完全な表に基づく旧ルール（第 2 ラウンドの母母父 dds_*）は台帳に『不完全な表（旧）』として残し、完全な表の結果と並べる",
            "parameters": cuts,
        },
        "counts": {"atoms": len(specs), "atoms_new": int(sum(a["new"] for a in specs)), "min_support_runs": MIN_RUNS_C, "min_support_races": MIN_RACES_C, **en["stats"],
                   "round1_testable": 624, "round2_candidates": pre2["counts"]["candidates"], "combined_tests_per_family": 624 + en["stats"]["candidates"],
                   "all_rounds_tests_per_family": 624 + pre2["counts"]["candidates"] + en["stats"]["candidates"], "families": 4},
        "candidates_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
        "atoms": [{"name": a["name"], "dim": a["dim"], "kind": a["kind"], "new": a["new"], "expr": a["expr"], "support_disc_runs": int(s)} for a, s in zip(specs, sup)],
    }
    PREREG_C.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidates_rc.txt").write_text("\n".join(names), encoding="utf-8")
    return spec


def run_eval_c(out_dir: Path) -> dict:
    d = T1.load_table(extra=make_extra_c())
    specs, _cuts = atom_specs_c(d)
    M = T2.atom_masks(d, specs)
    disc = np.isin(d["year"].to_numpy(), T1.DISC)
    rc = d["rc"].to_numpy()
    en = enumerate_c(M, [a["dim"] for a in specs], [a["kind"] for a in specs], [a["new"] for a in specs], rc, disc, MIN_RUNS_C, MIN_RACES_C, T2.known_round1_hashes(d, disc))
    cands = en["candidates"]
    pre = json.loads(PREREG_C.read_text(encoding="utf-8"))
    names = sorted(T2.cand_name(specs, c) for c in cands)
    assert hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest() == pre["candidates_sha256"], "候補が事前登録と一致しない"
    ledger1 = json.loads((out_dir / "ledger.json").read_text(encoding="utf-8"))
    ledger2 = json.loads((out_dir / "ledger_r2.json").read_text(encoding="utf-8"))
    res2 = json.loads((out_dir / "results_r2.json").read_text(encoding="utf-8"))
    ctx = T1.Ctx(d)
    fams = T1.FAMILIES
    uniq, rcd = np.unique(rc[disc], return_inverse=True)
    nrace = len(uniq)
    field_d = np.bincount(rcd, minlength=nrace)
    e1d, e3d = d["e_m1"].to_numpy()[disc], d["e_m3"].to_numpy()[disc]
    Md = M[:, disc]
    rows = []
    for c in cands:
        m = Md[c[0]].copy()
        for k in c[1:]:
            m &= Md[k]
        z1, z3, nn, _ = T2.fast_disc_z(e1d, e3d, rcd, nrace, m)
        nsel = np.bincount(rcd[m], minlength=nrace)
        rows.append({"c": c, "n": nn, "z1": z1, "z3": z3, "race_level": bool((nsel[rcd[m]] == field_d[rcd[m]]).mean() >= 0.9)})
    n2 = res2["prereg_counts"]["candidates"]
    for fam, (col, sign) in fams.items():
        p1 = [r["adopt"][fam]["p"] for r in ledger1 if fam in r["adopt"]]
        pc_ = [1.0 if r["race_level"] else 1 - T1.phi(sign * (r["z1"] if col == "e_m1" else r["z3"])) for r in rows]
        q = T1.bh(np.array(p1 + pc_))
        # 全ラウンド合算: 第 2 ラウンドの候補（p は ledger_r2 の有望分だけ保存。残りは p≈1 として足す＝保守的でない側。p の下限として q_all は参考）
        p2 = [1 - T1.phi(r["win_resid_z"] * sign) if col == "e_m1" else 1 - T1.phi(r["top3_resid_z"] * sign) for r in ledger2]
        pad = [1.0] * max(n2 - len(p2), 0)
        qa = T1.bh(np.array(p1 + pc_ + p2 + pad))
        for k, r in enumerate(rows):
            r.setdefault("q", {})[fam] = float(q[len(p1) + k])
            r.setdefault("q_all", {})[fam] = float(qa[len(p1) + k])
    prom = [k for k, r in enumerate(rows) if (not r["race_level"]) and max(abs(r["z1"]), abs(r["z3"])) > T1.ZB]
    rules = [{"id": "R2c/" + T2.cand_name(specs, rows[k]["c"]), "family": "R2c_order%d" % len(rows[k]["c"]), "name": T2.cand_name(specs, rows[k]["c"]), "expr": T2.cand_expr(specs, rows[k]["c"]),
              "source": "2c（完全な母方 10 世代表）", "market_free": True, "from_tp72": False} for k in prom]
    full = T1.evaluate_rules(d, rules)
    for r, k in zip(full, prom):
        r["c"] = rows[k]["c"]; r["q2"] = rows[k]["q"]; r["q_all"] = rows[k]["q_all"]
    for r in full:
        r["adopt"] = {}
        for fam, (col, sign) in fams.items():
            z = sign * (r["stats"]["disc"][col]["z"] or 0.0)
            tier = "A" if z > T1.ZA else "B" if z > T1.ZB else None
            rep = T1._years_ok(r, col, sign)
            r["adopt"][fam] = {"z": float(z), "p": float(1 - T1.phi(z)), "q": float(r["q2"][fam]), "q_all": float(r["q_all"][fam]), "tier": tier, "replicated": rep,
                               "robust_bh": bool(r["q2"][fam] < T1.FDR), "robust_bh_all": bool(r["q_all"][fam] < T1.FDR), "adopted": bool(tier is not None and rep), "effect": r["stats"]["disc"][col]["mean"]}
    # 増分: 親、第 1 ラウンド、第 2 ラウンドの最終採用（重なり）
    ad_prior = {fam: [x for x in ledger1 if x["adopt"].get(fam, {}).get("adopted")] for fam in fams}
    prior_masks = {fam: [(x["id"], T1.rule_mask(d, x["expr"]) & ctx.periods["disc"]) for x in ad_prior[fam]] for fam in fams}
    for fam in fams:
        for x in ledger2:
            if x[fam].get("final"):
                prior_masks[fam].append((x["id"], T1.rule_mask(d, x["expr"]) & ctx.periods["disc"]))
    for r in full:
        mask = T1.rule_mask(d, r["expr"])
        md = mask & ctx.periods["disc"]
        for fam, (col, sign) in fams.items():
            a = r["adopt"][fam]
            if not a["adopted"]:
                continue
            inc = {}
            if len(r["c"]) >= 2:
                for drop in range(len(r["c"])):
                    parent = [x for j, x in enumerate(r["c"]) if j != drop]
                    pm = M[parent[0]].copy()
                    for x in parent[1:]:
                        pm &= M[x]
                    inc[T2.cand_name(specs, tuple(parent))] = T2.inc_z(ctx, md, pm & ctx.periods["disc"], col, sign)
            a["inc_parents"] = inc
            a["inc_parent_ok"] = all(v > T1.ZB for v in inc.values()) if inc else True
            ov = []
            for rid, rm in prior_masks[fam]:
                inter = int((md & rm).sum())
                if inter == 0:
                    continue
                n_c, n_r = int(md.sum()), int(rm.sum())
                if inter / (n_c + n_r - inter) >= T1.OVERLAP_JAC or inter / min(n_c, n_r) >= T1.OVERLAP_CONT:
                    only_new = md & ~rm
                    zn = sign * (T1.cstat(ctx, only_new, col)["z"] or 0.0) if only_new.sum() >= 50 else -9.0
                    ov.append((rid, float(zn)))
            a["inc_prior"] = ov
            a["inc_prior_ok"] = all(zv > T1.ZB for _, zv in ov) if ov else True
            a["final"] = bool(a["adopted"] and a["inc_parent_ok"] and a["inc_prior_ok"])
            a["usable"] = bool(a["final"] and a["robust_bh"])
    summary = {}
    for fam in fams:
        ad = [r for r in full if r["adopt"][fam]["adopted"]]
        fin = [r for r in ad if r["adopt"][fam].get("final")]
        summary[fam] = {"usable": sum(1 for r in fin if r["adopt"][fam]["robust_bh"]), "candidates": len(rows), "promising": len(prom), "adopted_before_increment": len(ad), "pass_parent_increment": sum(1 for r in ad if r["adopt"][fam]["inc_parent_ok"]),
                        "pass_prior_increment": sum(1 for r in ad if r["adopt"][fam]["inc_prior_ok"]), "final": len(fin), "final_A": sum(r["adopt"][fam]["tier"] == "A" for r in fin),
                        "final_robust_bh": sum(r["adopt"][fam]["robust_bh"] for r in fin), "final_robust_bh_all": sum(r["adopt"][fam]["robust_bh_all"] for r in fin),
                        "by_form_final": {k: sum(1 for r in fin if len(r["c"]) == k) for k in (1, 2, 3)}, "combined_tests": pre["counts"]["combined_tests_per_family"],
                        "all_rounds_tests": pre["counts"]["all_rounds_tests_per_family"], "expected_false_pos_at_1.645": 0.05 * pre["counts"]["combined_tests_per_family"]}
    # 旧ルール（不完全な表に基づく母母父 dds_*）と、完全な表の mmf_* の並べ替え
    old = [x for x in ledger2 if "dds_" in x["id"]]
    old_rows = [{"id": x["id"], "n_disc": x["n_disc"], "win_resid_pt": x["win_resid_pt"], "top3_resid_pt": x["top3_resid_pt"], "win_z": x["win_resid_z"], "top3_z": x["top3_resid_z"],
                 "z_2025_win": x["z_2025_win"], "z_2025_top3": x["z_2025_top3"], **{f"{f}_final": x[f]["final"] for f in fams}} for x in old]
    new_mmf = [r for r in full if "mmf_" in r["name"] or "mmmf_" in r["name"]]
    # クラスタ: 第 1 ラウンド + 第 2 ラウンドの最終採用 + 2c の最終採用
    clusters = {}
    for fam in fams:
        rows_f = []
        for x in ledger1:
            if x["adopt"].get(fam, {}).get("adopted"):
                rows_f.append({"id": x["id"], "expr": x["expr"], "source": x["source"], "stats": T2._rebuild_stats(d, ctx, x), "adopt": {fam: x["adopt"][fam]}, "round": 1})
        for x in ledger2:
            if x[fam].get("final") and x[fam].get("robust_bh"):
                st2 = T2._rebuild_stats(d, ctx, x)
                col2 = "e_m1" if fam.startswith("ev") else "e_m3"
                rows_f.append({"id": x["id"], "expr": x["expr"], "source": "第 2 ラウンド", "stats": st2,
                               "adopt": {fam: {**x[fam], "z": x["win_resid_z"] if fam.startswith("ev") else x["top3_resid_z"], "effect": st2["disc"][col2]["mean"]}}, "round": 2})
        for r in full:
            if r["adopt"][fam].get("usable"):
                rows_f.append({"id": r["id"], "expr": r["expr"], "source": r["source"], "stats": r["stats"], "adopt": {fam: r["adopt"][fam]}, "round": "2c", "c": r["c"]})
        clusters[fam] = T1.cluster_set(d, rows_f, fam)
        for c in clusters[fam]:
            c["cover"] = float(c["union"][ctx.periods["disc"]].mean())
    res = {"summary": summary, "prereg_counts": pre["counts"], "enumeration": en["stats"], "old_rules_incomplete_table": old_rows}
    res["clusters"] = {fam: [{"rep": c["rep"], "round": c["rep_row"]["round"], "n_members": len(c["members"]), "members": c["members"], "weight": T1.cluster_weight(c, fam), "cover": c["cover"],
                              "expr": c["rep_row"]["expr"], "tier": c["rep_row"]["adopt"][fam]["tier"], "robust_bh": c["rep_row"]["adopt"][fam].get("robust_bh"),
                              "effect_pt": 100 * c["rep_row"]["adopt"][fam]["effect"]} for c in clusters[fam]] for fam in fams}
    res["v3"] = T1.evaluate_personas(d, ctx, clusters)
    res["dose_response"] = T2.dose_response(d, ctx, clusters)
    rep_of = {fam: {m: c["rep"] for c in clusters[fam] for m in c["members"]} for fam in fams}
    led = []
    for r in full:
        s = r["stats"]
        rec = {"id": r["id"], "order": len(r["c"]), "expr": r["expr"], "n_disc": s["disc"]["e_m1"]["n"], "races_disc": s["disc"]["e_m1"]["races"],
               "win_resid_pt": None if s["disc"]["e_m1"]["mean"] is None else 100 * s["disc"]["e_m1"]["mean"], "win_resid_z": s["disc"]["e_m1"]["z"],
               "top3_resid_pt": None if s["disc"]["e_m3"]["mean"] is None else 100 * s["disc"]["e_m3"]["mean"], "top3_resid_z": s["disc"]["e_m3"]["z"],
               "z_2025_win": s["y2025"]["e_m1"]["z"], "z_2025_top3": s["y2025"]["e_m3"]["z"], "z_2026_win": s["y2026"]["e_m1"]["z"], "z_2026_top3": s["y2026"]["e_m3"]["z"]}
        for fam in fams:
            a = r["adopt"][fam]
            rec[fam] = {"tier": a["tier"], "adopted": a["adopted"], "final": a.get("final", False), "usable": a.get("usable", False), "robust_bh": a["robust_bh"], "robust_bh_all": a["robust_bh_all"], "q_combined": a["q"], "q_all": a["q_all"],
                        "z": a["z"], "inc_parents": a.get("inc_parents"), "inc_round1": a.get("inc_prior"), "cluster": rep_of[fam].get(r["id"])}
        led.append(rec)
    (out_dir / "ledger_rc.json").write_text(json.dumps(led, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (out_dir / "results_rc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg", "eval", "reeval"])
    ap.add_argument("frozen", nargs="?")
    ap.add_argument("--years", nargs="*", type=int, default=[2026])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.cmd == "prereg":
        print(json.dumps(prereg_c()["counts"], ensure_ascii=False, indent=1))
    elif a.cmd == "eval":
        print(json.dumps(run_eval_c(Path(a.out))["summary"], ensure_ascii=False, indent=1))
    else:
        print(json.dumps(T1.reeval(Path(a.frozen), a.years, loader=lambda: T1.load_table(extra=make_extra_c())), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
