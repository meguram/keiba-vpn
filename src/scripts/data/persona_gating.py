"""予想家ペルソナ別サブモデル＋条件別ゲーティングの上限見積り（調査用。大きなモデルの学習はしない）。

python -m src.scripts.data.persona_gating run [--out DIR]

入力: tp-51 のグループ別専門家の OOF（gm_oof/exp_*.parquet、V4 の特徴量グループ + COND だけの LightGBM。valid 2022・2023）と
全列モデル（full.parquet）。ペルソナ代理 = 該当グループの専門家の『レース内で正規化した勝つ確率の対数』の平均。
評価: レース単位の NLL（勝ち馬の対数確率）。学習は『一方の年で当てはめ、もう一方の年で評価』（2 方向）。条件付きロジット（パラメータ数 7〜80）。
市場オッズ・人気は使わない。test 2025・2026 年は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(os.environ.get("TP51_OUT") or "/home/jovyan/work/tp51_out")
OOF = ROOT / "gm_oof"
FEAT = Path(os.environ.get("PERSONA_FEAT") or "data/local/features")
OUT = Path(os.environ.get("PERSONA_OUT") or "/home/jovyan/work/persona_out")

PERSONAS: dict[str, list[str]] = {
    "ABILITY": ["HIST_RESULT", "STATE", "EXP", "CONN_JKTR"],        # 能力重視家（過去の成績・状態・経験）
    "LAP": ["STAGE_AUX", "FIELD_PACE", "HIST_SPEED"],               # ラップ予想家（前段の位置取り・上がり予測、展開、速度履歴）
    "TRACKQ": ["AFFINITY", "FIELD_ENV"],                            # 馬場質予想家（条件別の適性、開催時期・環境）。実測の馬場質列は V4 に無く、代理は弱い
    "PED": ["PED"],                                                 # 血統予想家
    "BIAS": ["COND"],                                               # トラックバイアス予想家（枠・馬番・当日前日の勝ち馬上がり差 track_bias_*）
    "FIELD": ["FIELD_MEM", "FIELD_DEMO", "FIELD_CONN"],             # 参考: 出走メンバー構成
}
DIST_BANDS = [(0, 1400, "短"), (1401, 1800, "マ"), (1801, 2200, "中"), (2201, 9999, "長")]


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _race_logprob(p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """確率 p をレース内で正規化した対数確率（idx = レース番号の連番、行はレースごとに連続）。"""
    s = np.log(np.clip(p, 1e-9, None))
    m = np.maximum.reduceat(s, starts(idx))
    e = np.exp(s - m[idx])
    z = np.add.reduceat(e, starts(idx))
    return s - m[idx] - np.log(z)[idx]


def starts(idx: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])


def load_frame() -> pd.DataFrame:
    full = pd.read_parquet(OOF / "full.parquet")[["race_id", "horse_id", "year", "p_win"]].rename(columns={"p_win": "full"})
    df = full.copy()
    cache: dict[str, pd.Series] = {}
    for g in sorted({g for gs in PERSONAS.values() for g in gs}):
        x = pd.read_parquet(OOF / f"exp_{g}.parquet")[["race_id", "horse_id", "p_win"]]
        df = df.merge(x.rename(columns={"p_win": f"g_{g}"}), on=["race_id", "horse_id"], how="left")
    ranks = pd.concat([pd.read_parquet(FEAT / "target" / "rank_tbl" / str(y) / "rank.parquet") for y in (2022, 2023)])
    df = df.merge(ranks, on=["race_id", "horse_id"], how="left")
    cols = {}
    for name in ("surface", "distance", "track_condition", "field_size", "venue", "race_class", "date"):
        parts = [pd.read_parquet(FEAT / "race_tbl" / str(y) / f"shutuba_{name}.parquet") for y in (2022, 2023)]
        cols[name] = pd.concat(parts).rename(columns={f"shutuba_{name}": name})
    for name, t in cols.items():
        df = df.merge(t, on="race_id", how="left")
    df = df.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    return df


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    ok = df.groupby("race_id")["rank"].transform(lambda r: (r == 1).sum() == 1)
    df = df[ok].reset_index(drop=True)
    ridx = pd.factorize(df["race_id"])[0]
    df["_r"] = ridx
    lp = {"full": _race_logprob(df["full"].to_numpy(), ridx)}
    for k, gs in PERSONAS.items():
        sc = np.mean([np.log(np.clip(df[f"g_{g}"].to_numpy(), 1e-9, None)) for g in gs], axis=0)
        lp[k] = _race_logprob(np.exp(sc), ridx)
    for k, v in lp.items():
        df[f"lp_{k}"] = v
    df["win"] = (df["rank"] == 1).astype(int)
    d = pd.to_datetime(df["date"].astype(str).str[:10], errors="coerce")
    df["season"] = d.dt.month.map(lambda m: "春" if m in (3, 4, 5) else "夏" if m in (6, 7, 8) else "秋" if m in (9, 10, 11) else "冬")
    df["dband"] = pd.cut(df["distance"].astype(float), [0, 1400, 1800, 2200, 9999], labels=["短", "マ", "中", "長"]).astype(str)
    df["sd"] = df["surface"].astype(str).str[:1] + df["dband"]
    tc = df["track_condition"].astype(str)
    df["going"] = np.where(tc.str.contains("良"), "良", np.where(tc.str.contains("稍"), "稍重", "重不良"))
    rc = df["race_class"].astype(str)
    df["cls"] = np.where(rc.str.contains("新馬|未勝利"), "未勝利", np.where(rc.str.contains("1勝|2勝|500|1000"), "下級条件",
                         np.where(rc.str.contains("3勝|1600|OP|オープン|リステッド"), "上級条件", "重賞他")))
    fs = df["field_size"].astype(float)
    df["fsz"] = np.where(fs <= 12, "~12", np.where(fs <= 15, "13-15", "16~"))
    df["sg"] = df["surface"].astype(str).str[:1] + df["going"]
    meta = {"personas": list(PERSONAS), "n_rows": int(len(df)), "n_races": int(df["_r"].nunique())}
    return df, meta


def race_nll(df: pd.DataFrame, lp: np.ndarray) -> pd.Series:
    """レースごとの NLL（勝ち馬の対数確率の符号反転）。"""
    return pd.Series(-lp[df["win"].to_numpy() == 1], index=df.loc[df["win"] == 1, "_r"].to_numpy())


class CL:
    """条件付きロジット: レースごとの softmax(X θ + Σ_c 1[cell=c] X δ_c)。δ には ridge（階層縮約）。"""

    def __init__(self, X: np.ndarray, ridx: np.ndarray, win: np.ndarray, cell: np.ndarray | None, lam: float):
        self.X, self.ridx, self.win, self.lam = X, ridx, win, lam
        self.st = starts(ridx)
        self.d = X.shape[1]
        self.cell = cell
        self.C = 0 if cell is None else int(cell.max()) + 1
        if self.C:
            self.OH = np.zeros((len(cell), self.C), dtype=np.float64)
            self.OH[np.arange(len(cell)), cell] = 1.0

    def _scores(self, th):
        s = self.X @ th[: self.d]
        if self.C:
            D = th[self.d:].reshape(self.C, self.d)
            s = s + np.einsum("ij,ij->i", self.X, self.OH @ D)
        return s

    def nll_grad(self, th):
        s = self._scores(th)
        m = np.maximum.reduceat(s, self.st)
        e = np.exp(s - m[self.ridx])
        z = np.add.reduceat(e, self.st)
        p = e / z[self.ridx]
        nr = len(self.st)
        f = -(s[self.win == 1] - (m + np.log(z))).sum() / nr
        r = (p - self.win)
        g_w = (self.X * r[:, None]).sum(0) / nr
        g = [g_w]
        if self.C:
            g.append(((self.OH[:, :, None] * (self.X * r[:, None])[:, None, :]).sum(0)).ravel() / nr)
        g = np.concatenate(g)
        if self.C:
            dl = th[self.d:]
            f += self.lam * float(dl @ dl) / nr
            g[self.d:] += 2 * self.lam * dl / nr
        f += 1e-4 * float(th[: self.d] @ th[: self.d]) / nr
        g[: self.d] += 2e-4 * th[: self.d] / nr
        return f, g

    def fit(self, th0=None):
        n = self.d + self.C * self.d
        th = np.zeros(n) if th0 is None else th0
        r = minimize(lambda t: self.nll_grad(t), th, jac=True, method="L-BFGS-B", options={"maxiter": 300})
        return r.x

    def score(self, th, X, cell):
        s = X @ th[: self.d]
        if self.C and cell is not None:
            D = th[self.d:].reshape(self.C, self.d)
            s = s + np.einsum("ij,ij->i", X, D[cell])
        return s


def feats(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    return np.column_stack([df[f"lp_{k}"].to_numpy() for k in keys])


def eval_scores(df: pd.DataFrame, s: np.ndarray) -> pd.Series:
    lp = _race_logprob(np.exp(s - s.max()), df["_r"].to_numpy())
    return race_nll(df, lp)


def boot(d: pd.Series, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a = d.to_numpy()
    m = a.mean()
    bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)])
    return float(m), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cross_year(df: pd.DataFrame, keys: list[str], cell_col: str | None, lam: float) -> pd.Series:
    """2 方向の交差年評価。返り値 = 評価年のレースごとの NLL（全列モデル単体との比較用に連結）。"""
    out = []
    for tr_y, te_y in ((2022, 2023), (2023, 2022)):
        tr, te = df[df.year == tr_y].reset_index(drop=True), df[df.year == te_y].reset_index(drop=True)
        tr["_r"] = pd.factorize(tr["race_id"])[0]
        te["_r"] = pd.factorize(te["race_id"])[0]
        cats = None
        ctr = cte = None
        if cell_col:
            cats = sorted(df[cell_col].astype(str).unique())
            mp = {c: i for i, c in enumerate(cats)}
            ctr, cte = tr[cell_col].astype(str).map(mp).to_numpy(), te[cell_col].astype(str).map(mp).to_numpy()
        m = CL(feats(tr, keys), tr["_r"].to_numpy(), tr["win"].to_numpy(), ctr, lam)
        th = np.zeros(m.d + m.C * m.d)
        th[0] = 1.0
        th = m.fit(th)
        s = m.score(th, feats(te, keys), cte)
        nl = eval_scores(te, s)
        nl.index = te.loc[te["win"] == 1, "race_id"].to_numpy()
        out.append(nl)
    return pd.concat(out)


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _log("読み込み")
    df, meta = build(load_frame())
    keys_p = list(PERSONAS)
    res: dict = {"meta": meta}
    # 1. 全列モデル単体と各ペルソナ単体（レース単位 NLL）
    base_nll = {}
    for y in (2022, 2023):
        d = df[df.year == y].reset_index(drop=True)
        d["_r"] = pd.factorize(d["race_id"])[0]
        base_nll[y] = eval_scores(d, d["lp_full"].to_numpy())
        base_nll[y].index = d.loc[d["win"] == 1, "race_id"].to_numpy()
    base = pd.concat(base_nll.values())
    res["full_nll"] = float(base.mean())
    per = {}
    for k in keys_p:
        nl = []
        for y in (2022, 2023):
            d = df[df.year == y].reset_index(drop=True)
            d["_r"] = pd.factorize(d["race_id"])[0]
            v = eval_scores(d, d[f"lp_{k}"].to_numpy())
            v.index = d.loc[d["win"] == 1, "race_id"].to_numpy()
            nl.append(v)
        nl = pd.concat(nl)
        per[k] = {"nll": float(nl.mean()), "d_vs_full": boot(nl - base)}
    res["persona_alone"] = per
    # 2. 固定重みスタック（全列 + ペルソナ）と、条件別ゲーティング
    keys = ["full"] + keys_p
    fixed = cross_year(df, keys, None, 0.0)
    res["fixed_stack"] = boot(fixed - base)
    regimes = ["sd", "going", "cls", "fsz", "season", "venue", "sg"]
    gated = {}
    for rg in regimes:
        best = None
        for lam in (3.0, 30.0, 300.0, 3000.0):
            g = cross_year(df, keys, rg, lam)
            gate_d = boot(g - fixed)
            gated[f"{rg}|lam{lam:g}"] = {"d_vs_fixed": gate_d, "d_vs_full": boot(g - base)}
        _log(f"gating {rg} 完了")
    res["gated"] = gated
    # 3. 条件別の『最良ペルソナ』の入れ替わり（ペルソナ単体の NLL が最小のもの）と、年をまたぐ安定性
    sw = {}
    for rg in regimes:
        tab = {}
        for y in (2022, 2023):
            d = df[df.year == y].reset_index(drop=True)
            d["_r"] = pd.factorize(d["race_id"])[0]
            cells = d.loc[d["win"] == 1, rg].astype(str).to_numpy()
            nl = {k: eval_scores(d, d[f"lp_{k}"].to_numpy()).to_numpy() for k in keys_p}
            for c in sorted(set(cells)):
                msk = cells == c
                tab.setdefault(c, {})[y] = {k: float(nl[k][msk].mean()) for k in keys_p} | {"n": int(msk.sum())}
        sw[rg] = tab
    res["persona_by_cell"] = sw
    # 4. ペルソナ間の不一致（不確実性）と全列モデルの誤差
    rows = []
    for y in (2022, 2023):
        d = df[df.year == y].reset_index(drop=True)
        d["_r"] = pd.factorize(d["race_id"])[0]
        L = np.column_stack([d[f"lp_{k}"].to_numpy() for k in keys_p[:5]])
        d["dev"] = L.std(1)
        nl = eval_scores(d, d["lp_full"].to_numpy())
        top = d.loc[d.groupby("_r")["lp_full"].idxmax()].set_index("_r")
        top = top.assign(pf=np.exp(top["lp_full"]), nll=nl.reindex(top.index).to_numpy(), year=y)
        rows.append(top[["win", "pf", "dev", "nll", "year"]])
    dd = pd.concat(rows)
    dd["tercile"] = pd.qcut(dd["dev"], 3, labels=["低", "中", "高"])
    res["disagreement"] = {
        str(t): {"races": int(len(x)), "top1_hit": float(x["win"].mean()), "top1_pred": float(x["pf"].mean()), "nll": float(x["nll"].mean())}
        for t, x in dd.groupby("tercile", observed=True)
    }
    dd["pq"] = pd.qcut(dd["pf"], 4, labels=False)
    res["disagreement_by_pred_quartile"] = {
        str(q): {str(t): float(y["win"].mean() - y["pf"].mean()) for t, y in pd.concat([x, pd.Series(pd.qcut(x["dev"], 3, labels=["低", "中", "高"]), name="tt")], axis=1).groupby("tt", observed=True)}
        for q, x in dd.groupby("pq")
    }
    (out / "persona_gating.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    df.drop(columns=[c for c in df.columns if c.startswith("g_")]).to_parquet(out / "persona_frame.parquet")
    _log("完了")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    run(Path(a.out))


if __name__ == "__main__":
    main()
