"""親モデルの「ラベル × 学習モデル」の比較: 上位 k 頭の並びに集中する学習の候補を、v6e と同じ入力・同じ fold で OOF にする。

python -m src.scripts.data.rank_bakeoff list
python -m src.scripts.data.rank_bakeoff keys                       # 評価用の鍵（馬番・着順）を保存
python -m src.scripts.data.rank_bakeoff member NAME [--seeds 1 | --seed N] [--force] [--smoke]   # --seed N は 1 seed だけ（oof_seeds/）
python -m src.scripts.data.rank_bakeoff combine NAME [--seeds 1,2]   # oof/（seed 0）と oof_seeds/ を平均 → oof_s3/
python -m src.scripts.data.rank_bakeoff ens [--add NAME] [--sub oof_s3]   # v6e の 7 メンバー（+ 候補）の LOYO アンサンブル → ens/
python -m src.scripts.data.rank_bakeoff test2025 NAME [--seeds 3]   # 2025 年を 1 回だけ予測（最終候補 TEST2025_FINALISTS のみ）→ test2025/

- 入力は v6e のメンバーと同じ（V6_STAGE=e・V6_TAG=e・V5_GROUPS=CY,BEST,RB を既定にする。v6_members と同じ前処理）。
  RANK_BASE=2016 なら 2016〜2025 年の土台（retrain_out/base_2016、parent_2016 と同じ向け替え）で回し、出力は base_2016/tp_rank/。
  木のモデルは lgb_extras_ped と同じ列、NN は nn_s と同じ列（追加群は重要度上位 NN_TOP 列）。
- fold は valid 2022〜2024（年の前進）。2026 年は使わない。2025 年は test2025 だけ（2024-12-22 より前で学習）。
- 出力は data/local/modeling/training_pipeline_v4/tp_rank/oof/NAME.parquet（tp27_v6e/ には書かない）。
- --smoke は最初の fold だけ・木の数を減らして、最後まで通るかを確かめる（出力は tp_rank/smoke/）。
- 学習の中身は src/pipeline/models/rank_recipes.py。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# RANK_BASE=2016: 2016〜2025 年の土台（retrain_out/base_2016。parent_2016 と同じ向け替え）で回す。出力は BASE/tp_rank。
BASE_2016 = os.environ.get("RANK_BASE") == "2016"
os.environ.setdefault("V6_TAG", "e")
os.environ.setdefault("V5_GROUPS", "CY,BEST,RB")
if BASE_2016:
    from src.scripts.data import parent_2016 as _p16

    _p16.prepare()
    for _k, _v in {"TP_BASE_DIR": _p16.BASE, "TP_WORK": _p16.BASE, "TP25C_OUT": _p16.BASE / "tp25c", "TP25E_OUT": _p16.BASE / "tp25e",
                   "TP27_FAC_FIELD": _p16.BASE / "factory/field", "V6_OUT": _p16.BASE / "v6", "V5_TP38_OUT": _p16.BASE / "tp38",
                   "TP27_OUT": _p16.BASE / f"tp27_v6{os.environ['V6_TAG']}"}.items():
        os.environ[_k] = str(_v)
    os.environ.pop("V6_STAGE", None)                 # 前段は TP_WORK の oof から読む（parent_2016.main と同じ）
    from src.scripts.data import rebuild_groups_2016 as _rg

    _rg.redirect()                                   # 読み先の表・fold の学習の始まり（2016 年）を BASE に向ける
else:
    os.environ.setdefault("V6_STAGE", "e")

import pandas as pd  # noqa: E402

from src.scripts.data import v6_members  # noqa: E402  （環境変数の設定と作業ディレクトリの差し替え）
from src.scripts.data import tp27_eval as t27  # noqa: E402

if BASE_2016:
    from src.scripts.data import v5_common as _v5c

    v6_members.OUT = t27.OUT = _p16.BASE / f"tp27_v6{os.environ['V6_TAG']}"
    os.environ["TP27_OUT"] = str(t27.OUT)            # v6_members の import が本番の tp27_v6e を入れるので戻す
    _v5c.V = _p16.BASE
    OUT = _p16.BASE / "tp_rank"
else:
    OUT = v6_members.V / "tp_rank"
KEY = ["race_id", "horse_id"]

# 名前 -> (土台のメンバー, rank_recipes の cfg)
CONFIGS: dict[str, tuple[str, dict]] = {
    "R0_lgb_bin_win": ("lgb_extras_ped", {"model": "lgb", "kind": "binary", "k": 1}),   # 基準: 勝ちの二値（候補と同じ seed 数の lgb_extras_ped 相当。1 段目はこれと比べる。v6e の lgb_extras_ped（seed 0・1・2）との差で入力の一致も確かめる）
    "A1_lgb_bin_top3": ("lgb_extras_ped", {"model": "lgb", "kind": "binary", "k": 3}),
    "A2_lgb_bin_top5": ("lgb_extras_ped", {"model": "lgb", "kind": "binary", "k": 5}),
    "A3_lgb_lrank_top3": ("lgb_extras_ped", {"model": "lgb", "kind": "lambdarank", "k": 3, "gain": "exp", "truncation": 5}),
    "A4_lgb_lrank_top5_exp": ("lgb_extras_ped", {"model": "lgb", "kind": "lambdarank", "k": 5, "gain": "exp", "truncation": 10}),
    "A5_lgb_lrank_top5_lin": ("lgb_extras_ped", {"model": "lgb", "kind": "lambdarank", "k": 5, "gain": "linear", "truncation": 10}),
    "A6_lgb_xendcg_top5": ("lgb_extras_ped", {"model": "lgb", "kind": "xendcg", "k": 5}),
    "A7_lgb_reg_cap6": ("lgb_extras_ped", {"model": "lgb", "kind": "regression", "cap": 6}),
    "B1_xgb_ndcg_top5": ("xgb_extras_ped", {"model": "xgb", "objective": "rank:ndcg", "k": 5}),
    "B2_xgb_pairwise_top5": ("xgb_extras_ped", {"model": "xgb", "objective": "rank:pairwise", "k": 5}),
    "B3_xgb_bin_top3": ("xgb_extras_ped", {"model": "xgb", "kind": "binary", "k": 3}),               # A1 と同じラベルを XGBoost（GPU）で
    "B0_xgb_bin_win": ("xgb_extras_ped", {"model": "xgb", "kind": "binary", "k": 1}),                # B 系の基準: 勝ちの二値（1 seed の xgb_extras_ped 相当）
    # 2 段目の木の設定の調整（1 回だけ。valid の中で、同じ seed 0 の元の設定と比べる）。調整済みの値（hpo の best）は勝ちの二値向けで、
    # lambda_l2≈100 はランキングの目的関数（勾配の大きさが違う）には正則化が強すぎる可能性がある
    "A6t1_lgb_xendcg_top5": ("lgb_extras_ped", {"model": "lgb", "kind": "xendcg", "k": 5, "lgb_over": {"lambda_l2": 5.0, "num_leaves": 31}}),
    "A6t2_lgb_xendcg_top5": ("lgb_extras_ped", {"model": "lgb", "kind": "xendcg", "k": 5, "lgb_over": {"lambda_l2": 5.0, "num_leaves": 31}, "n_rounds": 1200}),
    "B3t1_xgb_bin_top3": ("xgb_extras_ped", {"model": "xgb", "kind": "binary", "k": 3, "xgb_over": {"reg_lambda": 5.0, "colsample_bytree": 0.5}}),
    "B3t2_xgb_bin_top3": ("xgb_extras_ped", {"model": "xgb", "kind": "binary", "k": 3, "xgb_over": {"max_depth": 7, "n_estimators": 1200}}),
    "C1_cat_yetirank_top5": ("lgb_extras_ped", {"model": "cat", "loss": "YetiRank", "k": 5}),
    "N1_nn_pl_top5": ("nn_s", {"model": "nn", "kind": "pl", "k": 5, "hidden": (128, 64), "epochs": 14}),
    "N2_nn_listnet_top5": ("nn_s", {"model": "nn", "kind": "listnet", "k": 5, "hidden": (128, 64), "epochs": 14}),
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _inputs(base: str, nn: bool):
    """cmd_member と同じ入力（土台のメンバーの X + TL + aux_B + 追加群）。NN は cmd_mtl と同じ（aux_B なし・追加群は上位 NN_TOP 列）。"""
    from src.scripts.data import tp25e_eval as e

    v6_members.patch()
    df, meta, folds, members, TLX = e.load_v3()
    _recipe, params, X = members[base]
    X = pd.concat([X, TLX], axis=1)
    parts = [X]
    if not nn:                                       # NN は mtl_nn（tp27_eval.cmd_mtl）と同じく補助ヘッドの列を入れない
        aux = pd.read_parquet(t27.AUX_DIR / "aux_B.parquet")
        aux["race_id"], aux["horse_id"] = aux["race_id"].astype(str), aux["horse_id"].astype(str)
        m = df[KEY].merge(aux, on=KEY, how="left")
        A = m[[c for c in m.columns if c not in KEY]]
        A.index = df.index
        parts.append(A)
    parts.append(t27._extra_for_ensemble(df, "core", t27.NN_TOP if nn else None))
    X = t27._drop_by_env(pd.concat(parts, axis=1))
    X = X.loc[:, ~X.columns.duplicated()]
    return df, meta, folds, params, X


def _smoke_params(cfg: dict, params: dict) -> tuple[dict, dict]:
    cfg, params = dict(cfg), dict(params)
    if cfg["model"] == "lgb":
        params["n_rounds"] = 20
        if "n_rounds" in cfg:                        # 調整の変種の木の数の上書きも簡易実行では 20 にする
            cfg["n_rounds"] = 20
    elif cfg["model"] == "xgb":
        params["n_estimators"] = 20
        if "n_estimators" in (cfg.get("xgb_over") or {}):
            cfg["xgb_over"] = {**cfg["xgb_over"], "n_estimators": 20}
    elif cfg["model"] == "cat":
        cfg["iterations"] = 20
    elif cfg["model"] == "nn":
        cfg["epochs"] = 1
    return cfg, params


def cmd_member(a) -> None:
    from src.pipeline.models import rank_recipes, zoo_runner

    base, cfg = CONFIGS[a.name]
    out_dir = OUT / ("smoke" if a.smoke else ("oof_seeds" if a.seed is not None else "oof"))
    p = out_dir / (f"{a.name}__seed{a.seed}.parquet" if a.seed is not None else f"{a.name}.parquet")
    if p.exists() and not a.force:
        _log(f"{a.name} 既存（--force で作り直す）")
        return
    rank_recipes.patch_zoo()
    df, meta, folds, params, X = _inputs(base, cfg["model"] == "nn")
    params = dict(params)
    if cfg["model"] == "nn":
        params = {}
    if a.smoke:
        cfg, params = _smoke_params(cfg, params)
        folds = folds[:1]
    seeds = [a.seed] if a.seed is not None else list(range(a.seeds))
    t27._wait_for_load()
    t0 = time.time()
    res, oof = zoo_runner.run_oof(a.name, "rk", X, meta, folds, seeds, t27.THREADS, params={**params, "rk": cfg})
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(p, index=False)
    info = {"name": a.name, "base": base, "cfg": cfg, "seeds": seeds, "cols": int(X.shape[1]), "seconds": round(time.time() - t0),
            "smoke": bool(a.smoke), "logloss_win": float(res["logloss_win"].mean())}
    (out_dir / f"{a.name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log(f"{a.name}: {info['seconds']}s cols={info['cols']} 行={len(oof)}")


def cmd_combine(a) -> None:
    """seed ごとの OOF の確率を平均する（run_oof の seed 平均と同じく、レース内で正規化した確率の平均）。"""
    from src.scripts.data.rank_bakeoff_eval import average_seed_oofs

    paths = [OUT / "oof" / f"{a.name}.parquet"] + [OUT / "oof_seeds" / f"{a.name}__seed{s}.parquet" for s in a.seeds.split(",")]
    out = average_seed_oofs([pd.read_parquet(p) for p in paths], KEY + ["year"])
    d = OUT / "oof_s3"
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / f"{a.name}.parquet", index=False)
    _log(f"{a.name}: {len(paths)} seed を平均 → {d / (a.name + '.parquet')}")


V6E_MEMBERS = ["lgb_extras_ped", "xgb_extras_ped", "lgb_extras", "lgb_softmax", "linear", "mtl_nn_extras_ped", "mtl_nn_s"]


def cmd_ens(a) -> None:
    """2 段目 (a): v6e の 7 メンバーに候補を 1 つ足したアンサンブル（tp27_eval.cmd_ens と同じ LOYO・同じ方法）を作り、
    v6e のアンサンブル（同じ道具で作り直したもの）と勝率・連対・複勝の LogLoss を比べる（feature_bakeoff.compare_oof の判定）。
    出力は tp_rank/ens/（基準の tp27_v6e/ens には書かない）。--add を省くと v6e の 7 メンバーだけ（作り直しの一致の確認用）。"""
    import numpy as np

    from src.pipeline.models import ensemble as en
    from src.pipeline.models import feature_bakeoff as fb
    from src.pipeline.models import tp05_data

    D = tp05_data.load_dataset(Path(t27.WORK_V))
    meta = D["meta"]
    base = meta[KEY + ["finish_position", "year"]].copy()
    base["race_id"], base["horse_id"] = base["race_id"].astype(str), base["horse_id"].astype(str)
    srcs = [(f"{m}__v4", t27.OUT / "oof" / f"{m}__v4.parquet") for m in V6E_MEMBERS]
    if a.add:
        srcs.append((a.add, OUT / a.sub / f"{a.add}.parquet"))
    names = []
    for n, path in srcs:
        o = pd.read_parquet(path)[KEY + ["p_win", "p_top2", "p_top3"]].copy()
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        before = len(base)
        base = base.merge(o.rename(columns={f"p_{t}": f"{n}__{t}" for t in ("win", "top2", "top3")}), on=KEY, how="inner")
        if names and len(base) != before:            # 候補の OOF に欠けがあると、重みと正規化が一部の行で決まり v6e と公平に比べられない
            raise SystemExit(f"{n} を足すと行が {before} → {len(base)} に減る（OOF の欠け）")
        names.append(n)
    race, years, fin = base["race_id"].to_numpy(), base["year"].to_numpy(), base["finish_position"].to_numpy()
    TOT = {"win": 1.0, "top2": 2.0, "top3": 3.0}
    Y = {"win": (fin == 1).astype(float), "top2": (fin <= 2).astype(float), "top3": (fin <= 3).astype(float)}
    METHOD = {"win": "weighted_logit", "top2": "greedy", "top3": "greedy"}
    ENS = base[KEY + ["year"]].copy()
    spec = {}
    for t in TOT:
        P = np.stack([base[f"{n}__{t}"].to_numpy() for n in names], axis=1)
        llv, pred = en.loyo_eval(METHOD[t], P, years, race, Y[t], TOT[t])
        ENS[f"p_{t}"] = pred
        st = en.fit_method(METHOD[t], P, en.logit(P), Y[t])
        spec[t] = {"method": METHOD[t], "loyo": float(llv), "weights": dict(zip(names, [round(float(x), 4) for x in st]))}
        _log(f"{t} LOYO {llv:.5f} 重み {spec[t]['weights']}")
    name = "v6" + (f"+{a.add}" if a.add else "_rebuilt")
    d = OUT / "ens"
    d.mkdir(parents=True, exist_ok=True)
    ENS.to_parquet(d / f"{name}.parquet", index=False)
    ref = pd.read_parquet(t27.OUT / "ens" / "v6.parquet")[KEY + ["p_win", "p_top2", "p_top3"]]
    ref["race_id"], ref["horse_id"] = ref["race_id"].astype(str), ref["horse_id"].astype(str)
    fin_map = base.set_index(KEY)["finish_position"]
    cmp = fb.compare_oof(ref.merge(ENS[KEY], on=KEY), ENS[KEY + ["p_win", "p_top2", "p_top3"]], fin_map, name)
    cmp = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in cmp.items()}
    (d / f"{name}.json").write_text(json.dumps({"spec": spec, "vs_v6e": cmp}, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"{name} vs v6e: " + str({k: (round(v, 5) if isinstance(v, float) else v) for k, v in cmp.items()}))


def cmd_test2025(a) -> None:
    """2025 年（test）を 1 回だけ予測する。学習は 2024-12-22 より前（v6e の fit_final_v6 --fit-end 2024 と同じく 10 日空ける）、seed は 0〜(n-1) の平均。
    出力 tp_rank/test2025/NAME.parquet。作り直さない（選択に使わない 1 回きりの確認）。基準は final_eval_v6e16/test2025_predictions.parquet。"""
    import numpy as np

    from src.pipeline.models import rank_recipes
    from src.pipeline.models import win_models as wm

    from src.scripts.data.rank_bakeoff_eval import check_test2025_name

    if not BASE_2016:                                # 基準（final_eval_v6e16）は 2016 年起点の土台にしかない。1 回きりの出力を別の土台に書かない
        raise SystemExit("test2025 は RANK_BASE=2016 で回す")
    check_test2025_name(a.name)                      # 先に commit した最終候補の一覧にある名前だけ
    out = OUT / "test2025" / f"{a.name}.parquet"
    if out.exists():
        raise SystemExit(f"{out} は作成済み（2025 年は 1 回きり。作り直さない）")
    base, cfg = CONFIGS[a.name]
    df, meta, folds, params, X = _inputs(base, cfg["model"] == "nn")
    params = {} if cfg["model"] == "nn" else dict(params)
    params.pop("rk", None)
    dates = pd.to_datetime(meta["date"])
    fin = meta["finish_position"].to_numpy()
    tr = np.flatnonzero(dates < pd.Timestamp("2025-01-01") - pd.Timedelta(days=10))
    va = np.flatnonzero((meta["year"].to_numpy() == 2025) & np.isfinite(fin.astype(float)))
    race = meta["race_id"].to_numpy()
    ytr = {"win": (fin[tr] == 1).astype(int), "top2": (fin[tr] <= 2).astype(int), "top3": (fin[tr] <= 3).astype(int), "finish": fin[tr]}
    t27._wait_for_load()
    t0 = time.time()
    win = np.zeros(len(va))
    for seed in range(a.seeds):
        o = rank_recipes.fit_predict(cfg, X.iloc[tr], ytr, race[tr], X.iloc[va], race[va], t27.THREADS, seed=seed, params=params)
        win += wm.normalize_to_total(race[va], o["win"], 1.0)
        _log(f"{a.name} seed {seed} 済み（{time.time() - t0:.0f}s）")
    P = meta.iloc[va][KEY + ["year"]].copy().reset_index(drop=True)
    P["p_win"] = win / a.seeds
    P["p_top2"] = np.nan                               # 順位の比較だけに使う（確率としては使わない）
    P["p_top3"] = np.nan
    out.parent.mkdir(parents=True, exist_ok=True)
    P.to_parquet(out, index=False)
    (out.with_suffix(".json")).write_text(json.dumps({"name": a.name, "cfg": cfg, "seeds": a.seeds, "train_rows": int(len(tr)), "test_rows": int(len(va)),
                                                      "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log(f"{a.name}: test 2025 {len(va)} 行（学習 {len(tr)} 行）→ {out}")


def cmd_keys(_a) -> None:
    """評価用の鍵: race_id・horse_id・馬番・着順・年（全行）。"""
    from src.pipeline.models import tp05_data

    D = tp05_data.load_dataset(Path(os.environ["TP_WORK"]))
    df = D["df"]
    k = df[["race_id", "horse_id", "horse_number", "finish_position", "year"]].copy()
    k["race_id"], k["horse_id"] = k["race_id"].astype(str), k["horse_id"].astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    k.to_parquet(OUT / "keys.parquet", index=False)
    _log(f"keys: {len(k)} 行 → {OUT / 'keys.parquet'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list")
    sp.add_parser("keys")
    m = sp.add_parser("member")
    m.add_argument("name", choices=sorted(CONFIGS))
    m.add_argument("--seeds", type=int, default=1)
    m.add_argument("--force", action="store_true")
    m.add_argument("--smoke", action="store_true")
    m.add_argument("--seed", type=int, default=None, help="この seed 1 つだけを回す（出力 oof_seeds/NAME__seedN。seed を並行で回して combine でまとめる）")
    e = sp.add_parser("ens", help="2 段目 (a): v6e の 7 メンバー + 候補 1 つの LOYO アンサンブル（出力 tp_rank/ens/）")
    e.add_argument("--add", default="", help="足す候補の名前（tp_rank/<sub>/<名前>.parquet）。省くと v6e の作り直しだけ")
    e.add_argument("--sub", default="oof_s3", help="oof_s3（3 seed）。調整の変種（seed 0 だけ）は oof")
    t = sp.add_parser("test2025", help="2025 年を 1 回だけ予測（2024-12-22 より前で学習、seed 平均）。出力 tp_rank/test2025/")
    t.add_argument("name", choices=sorted(CONFIGS))
    t.add_argument("--seeds", type=int, default=3)
    c = sp.add_parser("combine", help="oof/NAME（seed 0）と oof_seeds/NAME__seedN を平均して oof_s3/NAME にする")
    c.add_argument("name", choices=sorted(CONFIGS))
    c.add_argument("--seeds", default="1,2")
    a = ap.parse_args()
    if a.cmd == "list":
        for n, (b, c) in CONFIGS.items():
            print(n, b, c)
    elif a.cmd == "ens":
        cmd_ens(a)
    elif a.cmd == "test2025":
        cmd_test2025(a)
    elif a.cmd == "combine":
        cmd_combine(a)
    elif a.cmd == "keys":
        cmd_keys(a)
    else:
        cmd_member(a)


if __name__ == "__main__":
    main()
