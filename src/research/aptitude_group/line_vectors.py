"""サイアーライン (A) と牝系 (B) のベクトル（記述・可視化。ユーザーの指定・11/27。log.md「(A)(B) の固定」）。

    KEIBA_SESSION=aptitude-group OMP_NUM_THREADS=2 flock /tmp/claude-1000/heavy.lock nice -n 19 python -m src.research.aptitude_group.line_vectors [--boot 50]
    （--boot は (B) で約 600 万行の表を 50 回 × 2 種作り直す数時間の重い処理。load 8 以下・空き 20GB 以上を確かめてから、heavy.lock で 1 本だけ）

- (A) サイアーライン: 種牡馬 S が父系の直系に入る馬の走り。重みは 父 1・父の父 1/2・父の父の父 1/4（3 代まで）。
- (B) 牝系: S が母の側の 5 代以内の牡の位置（母父・母の母の父・…）に入る馬の走り。重みは血量 2^−代（母父 1/4・…・5 代 1/32）。
  母父（代 2）だけの値は、sire_vectors の母父のベクトル（bms）として別にある。
- 作りは sire_vectors と同じ（13 本・条件の対比・調和の件数・傾き）。行の重みを stats の列 w で渡す。
- 縮め: S の値を、S の父の同じ種類の値へ擬似件数 K で縮める（父はその父へ。3 段で 0 へ）。父の関係は 5 代の血統（anc5）の位置から作る。
  注意: 父の値には S 自身の走りも入る（(A) では父の「父の父」の位置に S の産駒が重み 1/2、(B) も同じ形）。sire_vectors の shrink（中系統に本人が入る）と
  同じ扱い。日本で走る息子の少ない父ほど、縮め先が S 自身に近くなり、縮めは弱く効く（記述の値として残す。予測のモデルの確かめでは S の系統を抜く）。
- 尺度: 中核（重み付きの走数 300 以上）の、重み付きの走数の重みの平均と sd で z。対象は重み付きの走数 50 以上。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.aptitude_group import common
from src.research.aptitude_group.sire_vectors import DIMS, K, MIN_RUNS, PED5, SHOW_RUNS, clean, names, runs, stats, uncertainty
from src.research.pedigree_lab import data
from src.research.pedigree_lab.aptitude import MAT_POS

LINE_POS = {(1, 0): 1.0, (2, 0): 0.5, (3, 0): 0.25}          # (代, 位置): 重み（父系の直系）
MAT_W = {(g, p): 0.5 ** g for g, ps in MAT_POS.items() for p in ps}   # 母の側の牡の祖先（代 2〜5）: 血量


def pedigree_links(keep_horses: set[str] | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """使う位置（LINE_POS・MAT_W）だけの祖先の長い表と、祖先の id → その父の id の対応（anc5 の位置 (g, p) の父は (g+1, 2p)）。
    長い表は keep_horses（走りのある馬）に絞る（critic 4 回目: 全位置・全馬だと数百万のタプルになる）。"""
    ped = data.load_ped()[["horse_id", "anc5"]]
    want = set(LINE_POS) | set(MAT_W)
    rows, parent = [], {}
    for h, a in zip(ped["horse_id"], ped["anc5"].fillna("")):
        d = {}
        for t in a.split(";"):
            if t:
                g, p, i = t.split(":")
                d[(int(g), int(p))] = i
        for (g, p), i in d.items():
            f = d.get((g + 1, 2 * p))
            if f and i not in parent:
                parent[i] = f
        if keep_horses is None or h in keep_horses:
            rows.extend((h, g, p, i) for (g, p), i in d.items() if (g, p) in want)
    return pd.DataFrame(rows, columns=["horse_id", "gen", "pos", "anc"]), parent


def long_frame(F: pd.DataFrame, P: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """走りの行を、祖先の位置ごとに重みを付けて複製した長い表（列 key・w）。同じ祖先が 2 つの位置にあれば重みを足す。"""
    wmap = pd.Series({g * 100 + p: w for (g, p), w in weights.items()})
    code = P["gen"].to_numpy() * 100 + P["pos"].to_numpy()
    sel = P.assign(w=pd.Series(code).map(wmap).to_numpy()).dropna(subset=["w"])
    sel = sel.groupby(["horse_id", "anc"], as_index=False)["w"].sum()
    R = F.reset_index().rename(columns={"index": "row"})[["row", "horse_id"]].merge(sel, on="horse_id", how="inner")
    L = F.iloc[R["row"].to_numpy()].reset_index(drop=True)
    L["key"], L["w"] = R["anc"].to_numpy(), R["w"].to_numpy()
    return L


def shrink_chain(S: pd.DataFrame, parent: dict[str, str], levels: int = 3) -> pd.DataFrame:
    """S の値を、父の同じ種類の値へ K で縮める（levels 段。一番上は 0 へ）。"""
    ids = S.index
    v = {nm: S[("d", nm)] / (S[("n", nm)] + K) for nm in DIMS}
    par = ids.to_series().map(parent)
    for _ in range(levels):
        v = {nm: (S[("d", nm)] + K * par.map(v[nm]).fillna(0.0)) / (S[("n", nm)] + K) for nm in DIMS}
    V = pd.DataFrame(v, index=ids)
    for nm in DIMS:
        V[f"n_{nm}"] = S[("n", nm)]
    V["runs"] = S["runs"]
    return V


def zfix(V: pd.DataFrame, mu: dict | None = None, sd: dict | None = None):
    if mu is None:
        core = V["runs"] >= MIN_RUNS
        w = V.loc[core, "runs"].to_numpy(float)
        mu = {d: float(np.average(V.loc[core, d], weights=w)) for d in DIMS}
        sd = {d: float(np.sqrt(np.average((V.loc[core, d] - mu[d]) ** 2, weights=w))) or 1.0 for d in DIMS}
    Z = pd.DataFrame({d: (V[d] - mu[d]) / sd[d] for d in DIMS}, index=V.index)
    return Z, mu, sd


def build(F: pd.DataFrame, P: pd.DataFrame, parent: dict, weights: dict) -> pd.DataFrame:
    return shrink_chain(stats(long_frame(F, P, weights), "key"), parent)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=0)
    a = ap.parse_args()
    common.open_token("select", "line_vectors_v1", "サイアーライン (A) と牝系 (B) のベクトル（記述・可視化・開発の行 2016〜24）")
    F = runs()
    P, parent = pedigree_links(set(F["horse_id"]))
    F = F[["horse_id", "meeting", "gc", "surf", "venue", "distance", "going", "course_type", "direction", "early", "ts", "lx", "acc_z", "gain_r",
           "sire", "ss", "L4", "ds", "bms_ss", "bms_L4"]].copy()          # 長い表（約 600 万行）を軽くするため、集計に要る列だけ
    out: dict = {"dims": DIMS, "k": K, "min_runs": MIN_RUNS, "show_runs": SHOW_RUNS, "line_pos": {str(k): v for k, v in LINE_POS.items()}}
    scales = {}
    for nm, wts in (("A", LINE_POS), ("B", MAT_W)):
        V = build(F, P, parent, wts)
        V = V[V["runs"] >= SHOW_RUNS]
        Z, mu, sd = zfix(V)
        scales[nm] = (mu, sd, list(V.index))
        unc = uncertainty(V)
        out[nm] = {h: {"runs": round(float(V.loc[h, "runs"]), 1), "core": bool(V.loc[h, "runs"] >= MIN_RUNS), "unc": round(float(unc.loc[h]), 3),
                       "v": [round(float(Z.loc[h, d]), 3) for d in DIMS]} for h in V.index}
        print(nm, len(V), int((V["runs"] >= MIN_RUNS).sum()), flush=True)
    want = set(out["A"]) | set(out["B"])
    nm_map = names(want, F)
    # names は父・母父などの列だけを見るので、母の側の深い祖先は、血統の長い表でその祖先を持つ馬の JSON から拾う
    cand = P[P["anc"].isin(want - set(nm_map))].groupby("anc")["horse_id"].apply(lambda s: list(s.head(5)))
    bad = 0
    for anc, hs in cand.items():                      # 子孫の 5 頭まで試す（critic 4 回目）
        for h in hs:
            f = PED5 / h[:4] / f"{h}.json"
            if not f.exists():
                continue
            try:
                hit = [x for x in json.loads(f.read_text(encoding="utf-8")).get("ancestors", []) if x.get("horse_id") == anc and x.get("name")]
            except (OSError, ValueError):
                bad += 1
                continue
            if hit:
                nm_map[anc] = clean(hit[0]["name"])
                break
    if bad:
        print(f"[名前] 読めない JSON {bad} 件", flush=True)
    out["names"] = nm_map
    p = common.OUT / "line_vectors.json"
    p.write_text(json.dumps(out, ensure_ascii=False))
    if a.boot:
        meets = F["meeting"].astype(str).to_numpy()
        idx = pd.Series(np.arange(len(F))).groupby(meets).indices
        mk = np.array(sorted(idx))
        res = {}
        for nm, wts in (("A", LINE_POS), ("B", MAT_W)):
            mu, sd, ids = scales[nm]
            B = np.full((a.boot, len(ids), 6), np.nan)
            for b in range(a.boot):
                rng = np.random.default_rng(9090 + b)
                Fb = F.iloc[np.concatenate([idx[mk[i]] for i in rng.choice(len(mk), len(mk), replace=True)])].reset_index(drop=True)
                Vb = build(Fb, P, parent, wts).reindex(ids)
                Z, _, _ = zfix(Vb, mu, sd)
                Z["wet"] = (Z["wet_T"] + Z["wet_D"]) / 2
                B[b] = Z[["dist", "surf", "wet", "pos", "ts", "gear"]].to_numpy()
                print(nm, "boot", b + 1, flush=True)
            res[f"{nm}_ids"], res[f"{nm}_B"] = np.array(ids), B
        np.savez_compressed(common.OUT / "line_vectors_boot.npz", **res)
    print({k: len(out[k]) for k in ("A", "B")}, "named", len(nm_map))
    data.finish_all(str(p))


if __name__ == "__main__":
    main()
