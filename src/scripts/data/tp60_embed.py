"""tp-60 血統グラフ上の『構成』埋め込み（ID を持ち込まない近さ）と、既知ペアでの検証。

python -m src.scripts.data.tp60_embed graph          # 親子の辺（5gen JSON の全体）→ <TP60_OUT>/embed/graph.parquet
python -m src.scripts.data.tp60_embed embed          # 変種ごとの埋め込み（競走馬 35,794 頭）→ embed/E_<variant>.npy（SVD 32 次元）
python -m src.scripts.data.tp60_embed known          # 既知ペア（全兄弟・同系・父だけ違う産駒）の近さの順位・AUC → embed/known_pairs.json

設計（近さを『祖先の構成』で決める）:
  節点 v の構成ベクトル E_t[v]: t=0 は節点ごとのランダムベクトル Z0[v]（葉）。t≥1 は両親が分かる節点なら 0.5×(E_{t-1}[父] + E_{t-1}[母])、
  片親だけ分かるときは 0.5×E_{t-1}[分かる親] + 0.5×Z0[v]、どちらも不明なら Z0[v]（創始者として止まる）。
  E_t[h] は『h の t 世代前の祖先の枠（各 2^-t）に載る Z0 の和』= t 世代目の血量で重み付けた祖先の構成。
  馬の埋め込みは Σ_{t=tmin..T} γ^(t-tmin) E_t[h]。tmin=2 なら 1 世代目（父・母）の ID は入らず、父は父の両親（祖父母）の構成に展開される。
    → 全兄弟は祖先の枠が同一なので距離 0、父だけが違う産駒（同じ母×ディープインパクト／ブラックタイド）も、父どうしが全兄弟なら距離 0。
  tmin=1 の変種は『親の ID を持ち込む』対照（従来の祖先ベクトルに近い）。
  ランダム射影（次元 d）で、10 世代 × 全祖先の疎行列を作らずに構成のカーネル（共通祖先の血量の積和）を保つ。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTD = Path(os.environ.get("TP60_OUT") or "/home/jovyan/work/tp60_out")
EMB = OUTD / "embed"
PED5 = ROOT / "data/local/horse_pedigree_5gen"

VARIANTS = {
    "A_t2_T10_g100": {"tmin": 2, "T": 10, "gamma": 1.0, "leaf": "none"},
    "B_t2_T10_g070": {"tmin": 2, "T": 10, "gamma": 0.7, "leaf": "none"},
    "C_t2_T5_g100": {"tmin": 2, "T": 5, "gamma": 1.0, "leaf": "none"},
    "D_t2_T10_idf": {"tmin": 2, "T": 10, "gamma": 1.0, "leaf": "idf"},
    "E_t1_T10_g100": {"tmin": 1, "T": 10, "gamma": 1.0, "leaf": "none"},          # 対照: 親の ID を持ち込む
    "F_t1_T5_idf": {"tmin": 1, "T": 5, "gamma": 1.0, "leaf": "idf"},                # 対照: 従来の祖先ベクトル（5 世代・IDF）に近い
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ 純粋関数（テスト対象）
def compute_E(sire: np.ndarray, dam: np.ndarray, d: int, tmin: int, T: int, gamma: float, leaf_w: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """節点ごとの構成埋め込み（Σ_{t=tmin..T} γ^(t-tmin) E_t を、重みの総和で割る）。sire/dam: 節点コード（不明は −1）。"""
    n = len(sire)
    rng = np.random.default_rng(seed)
    Z0 = rng.standard_normal((n, d)).astype(np.float32) / np.sqrt(d)
    if leaf_w is not None:
        Z0 *= leaf_w[:, None].astype(np.float32)
    has_s, has_d = sire >= 0, dam >= 0
    si, di = np.where(has_s, sire, 0), np.where(has_d, dam, 0)
    E = Z0.copy()
    acc = np.zeros_like(Z0)
    wsum = 0.0
    if tmin <= 0:
        acc += E
        wsum += 1.0
    for t in range(1, T + 1):
        En = np.where((has_s & has_d)[:, None], 0.5 * (E[si] + E[di]),
                      np.where(has_s[:, None], 0.5 * E[si] + 0.5 * Z0, np.where(has_d[:, None], 0.5 * E[di] + 0.5 * Z0, Z0)))
        E = En.astype(np.float32)
        if t >= tmin:
            w = gamma ** (t - tmin)
            acc += w * E
            wsum += w
    return acc / max(wsum, 1e-9)


def cosine_rows(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(A, axis=1) + 1e-12
    vn = np.linalg.norm(v) + 1e-12
    return (A @ v) / (an * vn)


def partner_rank(E: np.ndarray, pool: np.ndarray, a: int, b: int) -> tuple[int, float]:
    """節点 a から見た b の近さの順位（pool の中で、a 自身を除く。同点は上位に数えない）と順位の割合。"""
    s = cosine_rows(E[pool], E[a])
    mask = pool != a
    sb = float(cosine_rows(E[b:b + 1], E[a])[0])
    rank = int((s[mask] > sb + 1e-6).sum()) + 1
    return rank, rank / int(mask.sum())


def pair_auc(E: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    def sims(pr):
        a, b = E[pr[:, 0]], E[pr[:, 1]]
        return (a * b).sum(1) / ((np.linalg.norm(a, axis=1) + 1e-12) * (np.linalg.norm(b, axis=1) + 1e-12))
    sp, sn = sims(pos), sims(neg)
    allv = np.concatenate([sp, sn])
    ranks = pd.Series(allv).rank().to_numpy()
    n1, n0 = len(sp), len(sn)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ 実データ
def cmd_graph(a) -> None:
    EMB.mkdir(parents=True, exist_ok=True)
    parents: dict[str, tuple[str, str]] = {}
    names: dict[str, str] = {}
    n_files = 0
    for d in sorted(os.listdir(PED5)):
        p = PED5 / d
        if not p.is_dir():
            continue
        for f in os.listdir(p):
            if not f.endswith(".json"):
                continue
            try:
                js = json.loads((p / f).read_text(encoding="utf-8"))
            except Exception:
                continue
            h = str(js.get("horse_id") or f[:-5])
            tree = {(int(r["generation"]), int(r["position"])): str(r.get("horse_id") or "") for r in js.get("ancestors") or [] if r.get("generation") and r.get("horse_id")}
            for r in js.get("ancestors") or []:
                if r.get("horse_id") and r.get("name") and str(r["horse_id"]) not in names:
                    names[str(r["horse_id"])] = str(r["name"])
            tree[(0, 0)] = h
            for (g, pos), nid in list(tree.items()):
                if g > 4 or not nid:
                    continue
                s_, d_ = tree.get((g + 1, 2 * pos), ""), tree.get((g + 1, 2 * pos + 1), "")
                if nid not in parents:
                    parents[nid] = (s_, d_)
                else:
                    ps, pd_ = parents[nid]
                    parents[nid] = (ps or s_, pd_ or d_)
            n_files += 1
            if n_files % 10000 == 0:
                _log(f"{n_files:,} ファイル")
    rows = [(k, v[0], v[1]) for k, v in parents.items()]
    # 親として現れるだけの節点も入れる
    known = set(parents)
    extra = {x for _, s, d in rows for x in (s, d) if x and x not in known}
    rows += [(x, "", "") for x in extra]
    G = pd.DataFrame(rows, columns=["node_id", "sire_id", "dam_id"])
    G.to_parquet(EMB / "graph.parquet", index=False)
    (EMB / "node_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    _log(f"グラフ: 節点 {len(G):,}（親が分かる {(G['sire_id'] != '').sum():,}、両親 {((G['sire_id'] != '') & (G['dam_id'] != '')).sum():,}）、ファイル {n_files:,}")


def load_graph():
    G = pd.read_parquet(EMB / "graph.parquet")
    code = {n: i for i, n in enumerate(G["node_id"])}
    sire = G["sire_id"].map(lambda x: code.get(x, -1) if x else -1).to_numpy(dtype=np.int64)
    dam = G["dam_id"].map(lambda x: code.get(x, -1) if x else -1).to_numpy(dtype=np.int64)
    return G, code, sire, dam


def leaf_weights(G: pd.DataFrame, code: dict, horses: np.ndarray, sire: np.ndarray, dam: np.ndarray, kind: str) -> np.ndarray | None:
    if kind == "none":
        return None
    # 葉の出現頻度（競走馬の 10 世代以内の枠に載る回数ではなく、簡便に『競走馬の祖先として現れる馬数』の IDF）。10 世代の血量表から
    bs = pd.read_parquet(EMB.parent / "ped" / "blood_share.parquet", columns=["anc_id"])
    df = bs["anc_id"].value_counts()
    n_h = len(horses)
    w = np.ones(len(G), dtype=np.float32)
    idx = pd.Series(code).reindex(df.index).dropna().astype(int)
    w[idx.to_numpy()] = np.log((n_h + 1) / (df.reindex(idx.index).to_numpy() + 1)).astype(np.float32).clip(0.1, None)
    return w


def cmd_embed(a) -> None:
    G, code, sire, dam = load_graph()
    horses = pd.read_parquet(Path(os.environ.get("TP38_OUT") or "/home/jovyan/work/tp38_out") / "static" / "horses.parquet")["horse_id"].astype(str).to_numpy()
    hc = np.array([code.get(h, -1) for h in horses])
    _log(f"節点 {len(G):,}、競走馬 {len(horses):,}（グラフにある {(hc >= 0).sum():,}）")
    d = int(os.environ.get("TP60_EMB_D", 256))
    for name, v in VARIANTS.items():
        if a.only and name not in a.only.split(","):
            continue
        lw = leaf_weights(G, code, horses, sire, dam, v["leaf"])
        E = compute_E(sire, dam, d, v["tmin"], v["T"], v["gamma"], lw, seed=7)
        np.save(EMB / f"Efull_{name}.npy", E.astype(np.float16))
        H = E[np.where(hc >= 0, hc, 0)]
        H[hc < 0] = 0
        Hc = H - H[hc >= 0].mean(0)
        u, s, vt = np.linalg.svd(Hc[hc >= 0].astype(np.float32), full_matrices=False)
        P = np.zeros((len(horses), 32), dtype=np.float32)
        P[hc >= 0] = (u[:, :32] * s[:32]).astype(np.float32)
        np.save(EMB / f"E32_{name}.npy", P)
        ev = (s[:32] ** 2).sum() / (s ** 2).sum()
        _log(f"{name}: SVD32 の説明割合 {ev:.3f}")
    pd.Series(horses).to_frame("horse_id").to_parquet(EMB / "horses.parquet", index=False)


def resolve_known(G: pd.DataFrame, code: dict, names: dict[str, str], sire: np.ndarray, dam: np.ndarray) -> dict:
    """名前 → ID の解決を一度だけ行い、親子関係で検証する。結果は known_ids.json に記録。"""
    def find(pat, want_prefix=None):
        c = [i for i, n in names.items() if re.search(pat, n) and i in code]
        if want_prefix:
            c = [i for i in c if i.startswith(want_prefix)]
        return c
    di = find(r"^ディープインパクト", "2002")
    bt = find(r"^ブラックタイド", "2001")
    sw = find(r"サドラーズウェルズ|Sadler'?s Wells")
    nu = find(r"ヌレイエフ|Nureyev")
    nd = find(r"ノーザンダンサー|Northern Dancer")
    out = {"candidates": {"DI": di, "BT": bt, "SW": sw, "Nureyev": nu, "ND": nd}}
    pick = {}
    for k, c in (("DI", di), ("BT", bt)):
        pick[k] = c[0] if c else None
    if pick["DI"] and pick["BT"]:
        a, b = code[pick["DI"]], code[pick["BT"]]
        out["DI_BT_same_parents"] = bool(sire[a] == sire[b] and dam[a] == dam[b] and sire[a] >= 0)
    for k, c in (("SW", sw), ("Nureyev", nu)):
        ok = [x for x in c if sire[code[x]] >= 0 and G.loc[sire[code[x]], "node_id"] in nd]
        pick[k] = ok[0] if ok else None
    out["picked"] = pick
    return out


def cmd_known(a) -> None:
    G, code, sire, dam = load_graph()
    names = json.loads((EMB / "node_names.json").read_text(encoding="utf-8"))
    horses = pd.read_parquet(EMB / "horses.parquet")["horse_id"].astype(str).to_numpy()
    hs = pd.read_parquet(EMB.parent / "ped" / "horse_static.parquet")
    known = resolve_known(G, code, names, sire, dam)
    (EMB / "known_ids.json").write_text(json.dumps(known, ensure_ascii=False, indent=1), encoding="utf-8")
    pool = np.flatnonzero((sire >= 0) & (dam >= 0))
    pick = known["picked"]
    rng = np.random.default_rng(3)
    # 全兄弟ペア（競走馬: 同じ父・同じ母）、父だけ違う産駒ペア（同じ母 × 父が DI / BT）、同系（SW–Nureyev）、ランダム
    hc = np.array([code.get(h, -1) for h in hs["horse_id"].astype(str)])
    hs = hs.assign(c=hc)
    hs = hs[hs["c"] >= 0]
    fs = hs.dropna(subset=["sire_id", "dam_id"]).groupby(["sire_id", "dam_id"])["c"].apply(list)
    fs = fs[fs.map(len) >= 2]
    pos_fs = np.array([(l[0], l[1]) for l in fs.tolist()])[:5000]
    pairs_fd = None
    if pick.get("DI") and pick.get("BT"):
        d_ = hs[hs["sire_id"].isin([pick["DI"], pick["BT"]])]
        g = d_.groupby("dam_id")
        rows = []
        for dam_id, sub in g:
            a_ = sub[sub["sire_id"] == pick["DI"]]["c"].tolist()
            b_ = sub[sub["sire_id"] == pick["BT"]]["c"].tolist()
            if a_ and b_:
                rows.append((a_[0], b_[0]))
        pairs_fd = np.array(rows) if rows else None
    allc = hs["c"].to_numpy()
    neg = np.column_stack([rng.choice(allc, 20000), rng.choice(allc, 20000)])
    same_sire = hs.dropna(subset=["sire_id"]).groupby("sire_id")["c"].apply(list)
    pos_ss = np.array([(l[0], l[1]) for l in same_sire.tolist() if len(l) >= 2 and l[0] != l[1]])
    res = {"ids": known["picked"], "n_pool": int(len(pool)), "n_full_sib_pairs": int(len(pos_fs)), "n_father_differs_pairs": int(0 if pairs_fd is None else len(pairs_fd)), "variants": {}}
    for name in list(VARIANTS) + ["N_node2vec"]:
        f = EMB / f"Efull_{name}.npy"
        if not f.exists():
            continue
        E = np.load(f).astype(np.float32)
        r: dict = {}
        if pick.get("DI") and pick.get("BT"):
            rk, fr = partner_rank(E, pool, code[pick["DI"]], code[pick["BT"]])
            r["DI_BT_rank"], r["DI_BT_frac"] = rk, fr
            r["DI_BT_cos"] = float(cosine_rows(E[code[pick["BT"]]:code[pick["BT"]] + 1], E[code[pick["DI"]]])[0])
        if pick.get("SW") and pick.get("Nureyev"):
            rk, fr = partner_rank(E, pool, code[pick["SW"]], code[pick["Nureyev"]])
            r["SW_Nureyev_rank"], r["SW_Nureyev_frac"] = rk, fr
            r["SW_Nureyev_cos"] = float(cosine_rows(E[code[pick["Nureyev"]]:code[pick["Nureyev"]] + 1], E[code[pick["SW"]]])[0])
        r["auc_full_sib_vs_random"] = pair_auc(E, pos_fs, neg)
        r["auc_same_sire_vs_random"] = pair_auc(E, pos_ss[:5000], neg)
        r["cos_full_sib_mean"] = float(np.mean([cosine_rows(E[b:b + 1], E[a_])[0] for a_, b in pos_fs[:500]]))
        if pairs_fd is not None and len(pairs_fd) >= 3:
            r["auc_father_differs_vs_random"] = pair_auc(E, pairs_fd, neg)
            r["cos_father_differs_mean"] = float(np.mean([cosine_rows(E[b:b + 1], E[a_])[0] for a_, b in pairs_fd[:500]]))
            # 父だけ違う産駒の順位（競走馬の中で）
            ranks = []
            hpool = hs["c"].to_numpy()
            for a_, b in pairs_fd[:30]:
                s_ = cosine_rows(E[hpool], E[a_])
                sb = float(cosine_rows(E[b:b + 1], E[a_])[0])
                ranks.append(float(((s_ > sb + 1e-6).sum() + 1) / len(hpool)))
            r["father_differs_rank_frac_median"] = float(np.median(ranks))
        res["variants"][name] = r
        _log(f"{name}: {r}")
    (EMB / "known_pairs.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("graph")
    e = sub.add_parser("embed")
    e.add_argument("--only")
    sub.add_parser("known")
    a = ap.parse_args()
    {"graph": cmd_graph, "embed": cmd_embed, "known": cmd_known}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
