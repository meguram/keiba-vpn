"""親モデルの「ラベル × 学習モデル」の比較: 上位 k 頭の並びに集中する学習の候補を、v6e と同じ入力・同じ fold で OOF にする。

python -m src.scripts.data.rank_bakeoff list
python -m src.scripts.data.rank_bakeoff keys                       # 評価用の鍵（馬番・着順）を保存
python -m src.scripts.data.rank_bakeoff member NAME [--seeds 1] [--force] [--smoke]

- 入力は v6e のメンバーと同じ（V6_STAGE=e・V6_TAG=e・V5_GROUPS=CY,BEST,RB を既定にする。v6_members と同じ前処理）。
  RANK_BASE=2016 なら 2016〜2025 年の土台（retrain_out/base_2016、parent_2016 と同じ向け替え）で回し、出力は base_2016/tp_rank/。
  木のモデルは lgb_extras_ped と同じ列、NN は nn_s と同じ列（追加群は重要度上位 NN_TOP 列）。
- fold は valid 2022〜2024（年の前進）。test 2025・2026 年は使わない。
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
    elif cfg["model"] == "xgb":
        params["n_estimators"] = 20
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
    paths = [OUT / "oof" / f"{a.name}.parquet"] + [OUT / "oof_seeds" / f"{a.name}__seed{s}.parquet" for s in a.seeds.split(",")]
    frames = [pd.read_parquet(p).set_index(KEY + ["year"]) for p in paths]
    idx = frames[0].index
    for f in frames[1:]:
        if not f.index.equals(idx):
            raise SystemExit(f"行がそろわない: {a.name}")
    out = sum(f[["p_win", "p_top2", "p_top3"]] for f in frames) / len(frames)
    d = OUT / "oof_s3"
    d.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_parquet(d / f"{a.name}.parquet", index=False)
    _log(f"{a.name}: {len(frames)} seed を平均 → {d / (a.name + '.parquet')}")


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
    c = sp.add_parser("combine", help="oof/NAME（seed 0）と oof_seeds/NAME__seedN を平均して oof_s3/NAME にする")
    c.add_argument("name", choices=sorted(CONFIGS))
    c.add_argument("--seeds", default="1,2")
    a = ap.parse_args()
    if a.cmd == "list":
        for n, (b, c) in CONFIGS.items():
            print(n, b, c)
    elif a.cmd == "combine":
        cmd_combine(a)
    elif a.cmd == "keys":
        cmd_keys(a)
    else:
        cmd_member(a)


if __name__ == "__main__":
    main()
