"""tp-83: 出走馬全員のめぐ指数の履歴 → 能力の分布 → レース内の同時確率・差/形の特徴量 の評価。tp-42 と同じ土台（V4 の特徴量表・基準 OOF・段階評価）を使う。

python -m src.scripts.data.tp83_eval calib    # 能力の分布の当たり具合（2022〜2023 年の各走の実績指数を、過去だけの μ・σ で予測。κ・外れ値の重み下げ・t 分布の自由度の選び方）
python -m src.scripts.data.tp83_eval groups   # 群を作って tp83[_<variant>]/groups/*.parquet に保存（被覆率・母集団の定数も）
python -m src.scripts.data.tp83_eval stack    # 軽い一次選別: V4 アンサンブル OOF の log p_win に特徴量を 1 つ足す条件付きロジット（較正段階での使い方に相当）
python -m src.scripts.data.tp83_eval stage1   # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分（特徴量として足す使い方）
python -m src.scripts.data.tp83_eval stage2   # 第 2 段: 第 1 段を通った群を valid 2024、新 seed 2 本で確認
python -m src.scripts.data.tp83_eval member   # 第 3 段: アンサンブルのメンバー 1 つ（TP42_MEMBER）に群（TP42_ADD）を足して LOYO で OOF
python -m src.scripts.data.tp83_eval ens      # 第 3 段: 差し替えたメンバーでの LOYO アンサンブル（TP42_MEMBERS）
test 2025・2026 年は使わない。市場オッズ・人気は使わない。環境変数: TP83_VARIANT（出力先を分ける）, TP83_LAG_DAYS（指数の反映遅れ。既定 0）,
TP83_KAPPA_MULT, TP83_HUBER_K, TP83_T_DF, TP83_SD_MULT（能力の分布の補正。既定は補正なし。calib の結果で決める）, TP83_GROUPS（段階評価に使う群。既定は全群）, TP39_THREADS, TP39_LOAD_LIMIT, TP39_MAX_HEAVY。
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from src.pipeline.features import factory_self as fs  # noqa: E402
from src.pipeline.features import tp39_growth as tg  # noqa: E402
from src.pipeline.features import tp83_fieldsim as fsim  # noqa: E402
from src.scripts.data import tp39_eval as te  # noqa: E402
from src.scripts.data import tp42_eval as te42  # noqa: E402

KEY = ["race_id", "horse_id"]
VARIANT = os.environ.get("TP83_VARIANT", "")
OUT = te.V / (f"tp83_{VARIANT}" if VARIANT else "tp83")
LAG_DAYS = int(os.environ.get("TP83_LAG_DAYS", "0"))
GROUP_NAMES = ["FS_A", "FS_B", "FS_ALL", "FC_A", "FC_B", "FC_ALL"]
te.OUT = OUT                       # tp39 / tp42 の段階評価をそのまま使い、出力先だけ替える
te42.OUT = OUT
te42.VARIANT = VARIANT


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _opts() -> dict:
    env = {"kappa_mult": "TP83_KAPPA_MULT", "huber_k": "TP83_HUBER_K", "t_df": "TP83_T_DF", "sd_mult": "TP83_SD_MULT"}
    return {k: float(os.environ[v]) for k, v in env.items() if os.environ.get(v)}


def _groups() -> list[str]:
    return [g for g in os.environ.get("TP83_GROUPS", "").split(",") if g] or GROUP_NAMES


def cmd_groups(a) -> None:
    runs = fs.load_runs(te.TABLES, list(range(2020, 2026)))
    megu = tg.load_megu(te42.MEGU_ROOT, list(range(2020, 2026)))
    d = fsim.prepare(runs, megu)
    _log(f"走 {len(d):,} 行 / 指数あり {int(d['a'].notna().sum()):,} 行 / megu: {megu.attrs['megu'].get('dir')} {megu.attrs['megu'].get('version')} 反映遅れ={LAG_DAYS}日")
    t0 = time.time()
    gs, consts = fsim.build_all(d, LAG_DAYS, _opts())
    _log(f"母集団の定数 {consts} ({time.time() - t0:.0f}s)")
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    cov = {}
    yr = d["date"].dt.year.to_numpy()
    for n, t in gs.items():
        t.to_parquet(gdir / f"{n}.parquet", index=False)
        cols = [c for c in t.columns if c not in KEY]
        by = {int(y): float(t.loc[yr == y, cols].notna().mean().mean()) for y in (2021, 2022, 2023, 2024, 2025)}
        cov[n] = {"cols": len(cols), "columns": cols, "nonnull_rate_by_year": by}
        _log(f"{n}: {len(cols)} 列 被覆 {by}")
    te._jsave(OUT / "consts.json", {"megu": megu.attrs["megu"], "variant": VARIANT, "lag_days": LAG_DAYS, "population": consts})
    te._jsave(OUT / "coverage.json", cov)


def cmd_calib(a) -> None:
    """valid 年（2022・2023）の各走について、その走の実績指数 a を、as-of の (μ, σ) で予測したときの当たり具合。test 2025・2026 は使わない。
    見るもの: 走数別の二乗誤差、z=(a-μ)/σ の標準偏差（1 より大きい＝σ が小さすぎ）、|z|>2.5 の割合（正規なら 1.2%）、平均の対数尤度（正規／t）。"""
    import numpy as np
    from scipy.stats import t as tdist

    runs = fs.load_runs(te.TABLES, list(range(2020, 2026)))
    megu = tg.load_megu(te42.MEGU_ROOT, list(range(2020, 2026)))
    d = fsim.prepare(runs, megu)
    base = fsim.population_constants(d, LAG_DAYS)
    _log(f"母集団の定数 {base}")
    yr = d["date"].dt.year.to_numpy()
    ev = np.isin(yr, [2022, 2023]) & d["a"].notna().to_numpy()
    y = d["a"].to_numpy(float)[ev]
    buckets = {"1走": lambda n: n == 1, "2走": lambda n: n == 2, "3走": lambda n: n == 3, "4走": lambda n: n == 4, "5走": lambda n: n >= 5}
    dfs = (0, 4, 6, 10)

    def nll(z, sd, df):
        if df <= 2:
            return float(np.mean(0.5 * z ** 2 + np.log(sd) + 0.5 * np.log(2 * np.pi)))
        sc = np.sqrt((df - 2.0) / df)
        return float(np.mean(-tdist.logpdf(z / sc, df) + np.log(sc) + np.log(sd)))

    rows = []
    for km in (0.1, 0.25, 0.5, 1.0):
        for hk in (0.0, 2.5):
            c = {**base, "kappa": base["kappa"] * km, "huber_k": hk}
            c["sd_n_mult"] = fsim.fit_sd_by_n(d, c, LAG_DAYS)         # 2021 年だけで合わせる
            mu, sd, n = fsim.ability(d, c, LAG_DAYS)
            mu, sd, n = mu[ev], sd[ev], n[ev]
            z = (y - mu) / sd
            row = {"kappa_mult": km, "huber_k": hk, "sd_n_mult": c["sd_n_mult"], "by_n": {}}
            tot = {df: 0.0 for df in dfs}
            cnt = 0
            for name, f in buckets.items():
                m = f(n)
                if not m.any():
                    continue
                e = {"n": int(m.sum()), "rmse": float(np.sqrt(np.mean((y[m] - mu[m]) ** 2))), "z_std": float(np.std(z[m])),
                     "tail": float(np.mean(np.abs(z[m]) > 2.5)), **{f"nll_df{df}": nll(z[m], sd[m], df) for df in dfs}}
                row["by_n"][name] = e
                for df in dfs:
                    tot[df] += e[f"nll_df{df}"] * e["n"]
                cnt += e["n"]
            row["nll_all"] = {f"df{df}": tot[df] / cnt for df in dfs}
            row["rmse_all"] = float(np.sqrt(np.mean((y[n >= 1] - mu[n >= 1]) ** 2)))
            rows.append(row)
            _log(f"κ×{km} huber_k={hk} σ倍率 {[round(x, 2) for x in c['sd_n_mult'][1:]]}: rmse {row['rmse_all']:.3f} 対数損失(正規/t4/t6/t10) " + " / ".join(f"{row['nll_all'][f'df{df}']:.4f}" for df in dfs)
                 + " | z_std " + " ".join(f"{k}:{v['z_std']:.2f}" for k, v in row["by_n"].items())
                 + " | tail " + " ".join(f"{k}:{v['tail']:.3f}" for k, v in row["by_n"].items()))
    best = min(((r["nll_all"][f"df{df}"], r["kappa_mult"], r["huber_k"], df) for r in rows for df in dfs))
    _log(f"対数損失が最小: κ×{best[1]} huber_k={best[2]} t_df={best[3]} ({best[0]:.4f})")
    te._jsave(OUT / "calib.json", {"population": base, "rows": rows, "best": {"nll": best[0], "kappa_mult": best[1], "huber_k": best[2], "t_df": best[3]}})


def cmd_stage1(a) -> None:
    te._stage(os.environ.get("TP83_LABEL") or "stage1", [2022, 2023], [0], _groups(), "stage1")


def cmd_stage2(a) -> None:
    te._stage(os.environ.get("TP83_LABEL") or "stage2", [2024], [2, 3], _groups(), "stage2")


def cmd_stack(a) -> None:
    if not (OUT / "nprior.parquet").exists():
        shutil.copy(te.V / "tp39" / "nprior.parquet", OUT / "nprior.parquet")
    te.GROUP_NAMES = ["FS_ALL", "FC_ALL"]
    te.cmd_stack(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "groups", "stack", "stage1", "stage2", "member", "ens"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"calib": cmd_calib, "groups": cmd_groups, "stack": cmd_stack, "stage1": cmd_stage1, "stage2": cmd_stage2,
     "member": lambda x: _delegate(te42.cmd_member, x),
     "ens": lambda x: _delegate(te42.cmd_ens, x)}[a.cmd](a)


def _delegate(fn, a) -> None:
    os.environ.setdefault("TP42_ADD", "FS_ALL")
    fn(a)


if __name__ == "__main__":
    main()
