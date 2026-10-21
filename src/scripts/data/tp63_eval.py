"""tp-63: V5 の 7 メンバーの多様性の棚卸しと、追加メンバー（tp63_members の OOF）を足したときの合成の増分。

python -m ... inventory                       # 7 メンバーの性能・相関・残差相関・重み・1 本外したときの劣化
python -m ... eval NAME[,NAME...] [--modes loyo,forward]    # 『V5 の 7 + そのメンバー』（現行合成 P_ と L3 型）を V5 の現行合成と比べる
python -m ... combo NAME1,NAME2 --label X     # 複数メンバーを同時に足す
入力: V5 の 7 メンバー OOF（tp27_v5/oof）+ 追加メンバー（$TP63_OUT/ev/members）。valid 2022〜2024（140,865 行）。
評価の枠は tp-51 と同じ（LOYO / 前進、レース単位の paired bootstrap、採用 = 上限 < 0・効果 ≥ 0.0002・連対複勝の悪化 ≤ 0.0002）。
test 2025・2026 年は使わない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KEIBA_ROOT", Path(__file__).resolve().parents[3]))
V4 = ROOT / "data/local/modeling/training_pipeline_v4"
OUT = Path(os.environ.get("TP63_OUT") or "/home/jovyan/work/tp63_out")
EV = OUT / "ev"
os.environ.setdefault("TP51_MEMBERS_DIR", str(V4 / "tp27_v5/oof"))
os.environ.setdefault("TP51_OUT", str(EV))
os.environ.setdefault("TP51_MEMBERS_SUBDIR", "members")
os.environ.setdefault("KEIBA_ROOT", str(ROOT))
try:
    from src.scripts.data import tp51_stack as S  # noqa: E402
except ImportError:
    sys.path.insert(0, "/home/jovyan/work/tp51_code")
    import tp51_stack as S  # type: ignore  # noqa: E402

KEY = S.KEY
TARGETS = S.TARGETS
MEMBERS = S.MEMBERS
BASE_METHOD = S.BASE_METHOD
RES = EV / "tp63_results.json"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_res() -> dict:
    return json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {}


def _save_res(res: dict) -> None:
    EV.mkdir(parents=True, exist_ok=True)
    RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────── 合成の方式（メンバーの組を引数にできる版）
def fn_cross_lr(members, C: float = 0.1):
    """クロス対象 LR（3 対象 × メンバー数の logit を入力に、対象ごとに LR）。tp-51 の S3_lr_cross を、メンバーの組を指定できるようにしたもの。"""
    from sklearn.linear_model import LogisticRegression

    def fn(tr, va):
        Ptr, Pva = S.member_arrays(tr, members), S.member_arrays(va, members)
        Xtr = np.hstack([S.logit(Ptr[t]) for t in TARGETS])
        Xva = np.hstack([S.logit(Pva[t]) for t in TARGETS])
        Y = S._ys(tr)
        return {t: LogisticRegression(C=C, max_iter=1000).fit(Xtr, Y[t]).predict_proba(Xva)[:, 1] for t in TARGETS}

    return fn


def schemes_for(members: list[str]) -> dict:
    spec = S.fn_spec(BASE_METHOD, members)
    l3 = S.fn_avg([spec, fn_cross_lr(members)])
    return {
        "P": spec,                                                       # 現行の合成（勝率 weighted_logit、連対・複勝 greedy）
        "L3win2": S.fn_mix({"win": l3, "top2": l3, "top3": spec}),      # 勝率・連対は L3、複勝は現行（tp-51 の S11e 型）
    }


def base_oofs(df: pd.DataFrame) -> dict:
    """V5 の 7 メンバーの現行合成（S0）と L3 型（S11e）の OOF（LOYO / 前進）。無ければ作る。"""
    out = {}
    sc = schemes_for(MEMBERS)
    for mode in ("loyo", "forward"):
        for nm, key in (("P", "S0"), ("L3win2", "S11e")):
            p = EV / f"oof_base_{key}_{mode}.parquet"
            if p.exists():
                out[(key, mode)] = pd.read_parquet(p)
            else:
                out[(key, mode)] = S.run_scheme(df, sc[nm], mode)
                EV.mkdir(parents=True, exist_ok=True)
                out[(key, mode)].to_parquet(p, index=False)
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# ───────────────────────── 棚卸し
def cmd_inventory(a) -> None:
    from src.pipeline.models import ensemble as en

    os.environ["TP51_MEMBERS_SUBDIR"] = "no_extra_members"
    S.OUT = EV / "stack_inv"
    df = S.load_frame()
    fin = df["finish_position"].to_numpy()
    fin_map = df.set_index(KEY)["finish_position"]
    Y = {t: (fin <= k).astype(float) for t, k in (("win", 1), ("top2", 2), ("top3", 3))}
    rep: dict = {"rows": int(len(df)), "members": {}}
    P = S.member_arrays(df)
    ll = lambda p, y: float(-(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))).mean())     # noqa: E731
    for i, m in enumerate(MEMBERS):
        rep["members"][m] = {t: ll(P[t][:, i], Y[t]) for t in TARGETS}
    # logit 相関と残差相関
    for t in TARGETS:
        Z = S.logit(P[t])
        R = Y[t][:, None] - P[t]
        rep[f"logit_corr_{t}"] = np.round(np.corrcoef(Z.T), 4).tolist()
        rep[f"resid_corr_{t}"] = np.round(np.corrcoef(R.T), 4).tolist()
        ev = np.linalg.eigvalsh(np.corrcoef(R.T))
        rep[f"resid_participation_ratio_{t}"] = float(ev.sum() ** 2 / (ev ** 2).sum())      # 実質的に独立なメンバー数の目安
        off = np.corrcoef(R.T)[np.triu_indices(len(MEMBERS), 1)]
        rep[f"resid_corr_mean_{t}"] = float(off.mean())
        rep[f"resid_corr_min_{t}"] = float(off.min())
    # 重み（全年で当てはめ）
    w = {}
    for t in TARGETS:
        st = en.fit_method(BASE_METHOD[t], P[t], S.logit(P[t]), Y[t])
        w[t] = np.round(np.asarray(st if not isinstance(st, dict) else st.get("weights")), 4).tolist() if st is not None else None
    rep["weights_all_years"] = w
    # LOYO の重み（年ごと）
    loyo_w = {}
    for yr in sorted(df["year"].unique()):
        tr = df["year"].to_numpy() != yr
        loyo_w[int(yr)] = {t: np.round(np.asarray(en.fit_method(BASE_METHOD[t], P[t][tr], S.logit(P[t][tr]), Y[t][tr])), 4).tolist() for t in ("win",)}
    rep["loyo_weights_win"] = loyo_w
    # 1 本外したときの劣化
    full = S.run_scheme(df, S.fn_spec(BASE_METHOD, MEMBERS), "loyo")
    rep["drop_one"] = {}
    for m in MEMBERS:
        rest = [x for x in MEMBERS if x != m]
        o = S.run_scheme(df, S.fn_spec(BASE_METHOD, rest), "loyo")
        c = S.compare(full, o, fin_map, f"drop {m}")
        rep["drop_one"][m] = {k: c[k] for k in ("Δwin", "win_hi", "Δtop2", "Δtop3")}
        _log(f"drop {m}: Δwin {c['Δwin']:+.5f} Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f}")
    # 現行合成の LogLoss（LOYO）
    rep["composite_loyo"] = {t: float(ll(full[f"p_{t}"].to_numpy(), full.merge(df[KEY + ["finish_position"]], on=KEY)["finish_position"].to_numpy() <= k)) for t, k in (("win", 1), ("top2", 2), ("top3", 3))}
    res = _load_res()
    res["inventory"] = rep
    _save_res(res)
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith(("logit_corr", "resid_corr_" + "win"))}, ensure_ascii=False)[:3000])


def cmd_inventory2(a) -> None:
    """メンバーの『合成からのずれ』（logit の差）の相関 — 残差相関はラベルの雑音で 0.99 に張り付くので、モデル出力どうしの違いで多様性を測る。"""
    os.environ["TP51_MEMBERS_SUBDIR"] = "no_extra_members"
    S.OUT = EV / "stack_inv"
    df = S.load_frame()
    P = S.member_arrays(df)
    rep = {}
    for t in TARGETS:
        Z = S.logit(P[t])
        D = Z - Z.mean(1, keepdims=True)           # 7 メンバーの平均 logit からのずれ
        C = np.corrcoef(D.T)
        off = C[np.triu_indices(len(MEMBERS), 1)]
        ev = np.linalg.eigvalsh(np.cov(D.T))[::-1]
        rep[t] = {"dev_corr": np.round(C, 3).tolist(), "dev_corr_mean": float(off.mean()), "dev_std": np.round(D.std(0), 4).tolist(),
                  "pc_var_share": np.round(ev / ev.sum(), 4).tolist(), "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
                  "logit_std_overall": float(Z.std())}
        _log(f"{t}: ずれの相関の平均 {rep[t]['dev_corr_mean']:.3f}、主成分の分散比 {rep[t]['pc_var_share'][:4]}、実効次元 {rep[t]['participation_ratio']:.2f}")
    res = _load_res()
    res["inventory2"] = rep
    _save_res(res)


# ───────────────────────── 追加メンバーの評価
def standalone(df: pd.DataFrame, name: str, base: dict) -> dict:
    """単独の LogLoss と、V5 の現行合成（LOYO）・各メンバーとの相関。"""
    fin = df["finish_position"].to_numpy()
    cols = {t: f"{name}__{t}" for t in TARGETS}
    out = {}
    s0 = base[("S0", "loyo")].merge(df[KEY + list(cols.values())], on=KEY)
    for t, k in (("win", 1), ("top2", 2), ("top3", 3)):
        y = (df["finish_position"].to_numpy() <= k).astype(float)
        p = df[cols[t]].to_numpy()
        out[f"ll_{t}"] = float(-(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))).mean())
        ys0 = (s0.merge(df[KEY + ["finish_position"]], on=KEY)["finish_position"].to_numpy() <= k).astype(float)
        c = s0[f"p_{t}"].to_numpy()
        pm = s0[cols[t]].to_numpy()
        out[f"logit_corr_composite_{t}"] = _corr(S.logit(c), S.logit(pm))
        out[f"resid_corr_composite_{t}"] = _corr(ys0 - c, ys0 - pm)
        out[f"resid_corr_members_mean_{t}"] = float(np.mean([_corr(y - df[f"{m}__{t}"].to_numpy(), y - p) for m in MEMBERS]))
    return out


def eval_members(names: list[str], label: str, modes: list[str], force: bool) -> None:
    stage = EV / f"stage_{label}"
    stage.mkdir(parents=True, exist_ok=True)
    for n in names:
        link = stage / f"{n}.parquet"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(EV / "members" / f"{n}.parquet")
    os.environ["TP51_MEMBERS_SUBDIR"] = stage.name
    df = S.load_frame()
    fin_map = df.set_index(KEY)["finish_position"]
    _log(f"データ {df.shape}")
    base = base_oofs(df)
    res = _load_res()
    members = MEMBERS + names
    sc = schemes_for(members)
    sa = {}
    for n in names:
        sa[n] = standalone(df, n, base)
        res.setdefault("standalone", {})[n] = sa[n]
        _log(f"{n} 単独: win {sa[n]['ll_win']:.5f} top2 {sa[n]['ll_top2']:.5f} top3 {sa[n]['ll_top3']:.5f} 合成との残差相関(win) {sa[n]['resid_corr_composite_win']:.3f}")
    for mode in modes:
        for nm, key in (("P", "S0"), ("L3win2", "S11e")):
            k = f"{label}|{nm}|{mode}"
            if k in res and not force:
                continue
            t0 = time.time()
            oof = S.run_scheme(df, sc[nm], mode)
            oof.to_parquet(EV / f"oof_{label}_{nm}_{mode}.parquet", index=False)
            c0 = S.compare(base[("S0", mode)], oof, fin_map, f"{label} {nm} vs S0 {mode}")      # V5 の現行合成に対する増分
            c1 = S.compare(base[("S11e", mode)], oof, fin_map, f"{label} {nm} vs L3 {mode}")    # V5 の L3 型に対する増分
            res[k] = {"vs_S0": {kk: c0[kk] for kk in ("Δwin", "win_hi", "Δtop2", "Δtop3", "採用")}, "vs_L3": {kk: c1[kk] for kk in ("Δwin", "win_hi", "Δtop2", "Δtop3", "採用")},
                      "members": names}
            _log(f"{k}: vs現行 Δwin {c0['Δwin']:+.5f} (hi {c0['win_hi']:+.5f}) Δtop2 {c0['Δtop2']:+.5f} Δtop3 {c0['Δtop3']:+.5f} 採用={c0['採用']} | vs L3 Δwin {c1['Δwin']:+.5f} (hi {c1['win_hi']:+.5f}) 採用={c1['採用']} ({time.time() - t0:.0f}s)")
            _save_res(res)


def cmd_peryear(a) -> None:
    """採用に近い組を年ごとに（L3 型の基準 S11e 比）。前進の 2024 年は『2022〜23 年で合成を学習して 2024 年を予測』で、運用に最も近い。"""
    df = S.load_frame()
    fin_map = df.set_index(KEY)["finish_position"]
    res = _load_res()
    out = res.setdefault("peryear", {})
    for lab in a.labels.split(","):
        for mode in ("loyo", "forward"):
            pb, pn = EV / f"oof_base_S11e_{mode}.parquet", EV / f"oof_{lab}_L3win2_{mode}.parquet"
            if not (pb.exists() and pn.exists()):
                continue
            b, o = pd.read_parquet(pb), pd.read_parquet(pn)
            for y in sorted(set(b["year"]) & set(o["year"])):
                c = S.compare(b[b["year"] == y], o[o["year"] == y], fin_map, f"{lab} {mode} {y}")
                out[f"{lab}|{mode}|{y}"] = {k: c[k] for k in ("Δwin", "win_hi", "Δtop2", "Δtop3")}
                _log(f"{lab}|{mode}|{y}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f}")
    _save_res(res)


def cmd_eval(a) -> None:
    for n in a.names.split(","):
        eval_members([n], n, a.modes.split(","), a.force)


def cmd_combo(a) -> None:
    eval_members(a.names.split(","), a.label, a.modes.split(","), a.force)


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("inventory")
    sp.add_parser("inventory2")
    e = sp.add_parser("eval")
    e.add_argument("names")
    e.add_argument("--modes", default="loyo,forward")
    e.add_argument("--force", action="store_true")
    py = sp.add_parser("peryear")
    py.add_argument("labels")
    c = sp.add_parser("combo")
    c.add_argument("names")
    c.add_argument("--label", required=True)
    c.add_argument("--modes", default="loyo,forward")
    c.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"inventory": cmd_inventory, "inventory2": cmd_inventory2, "eval": cmd_eval, "combo": cmd_combo, "peryear": cmd_peryear}[a.cmd](a)


if __name__ == "__main__":
    main()
