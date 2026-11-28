"""拡張ラベルの学習（担当 model-relabel）: v5bag と v6e の二値の代理を、ラベルと重みだけ替えて学ぶ。

- 特徴量の表は cycles16_fix3 の X_xgb（読むだけ）。v5bag の列は megu_cycles の `_rich_matrix("v5")`、v6e の代理は
  final_v6e16 の束の xgb_extras_ped の feature_cols（1,231 列）。木の設定は v5bag の BAG_SETS・XGB_PARAMS、
  v6e は xgb_extras_ped の params に固定する（選び直さない。config/model_relabel/spec.json）。
- 目標: win（元 y1・拡張 y_w）・top3（元 y3・拡張 y_p）・notbig（P-ord の m_a = 大敗でない）。
- 変種: orig（元のラベル）・ext_w1・ext_w2・ext_w4（拡張のラベル、大敗の負例の重み w）・ext1_w1・ext1_w2・ext1_w4（複勝の 1 馬身の版 y_p1）・
  *_nc（本命決着のレースを学習の行からだけ除く版。v5bag の orig・ext_w*。予測・較正・評価は全レース）。
- 分け:
  - time: 年 t（2017〜24）は t より前の年で学ぶ。
  - meeting: 開催の 5 分割（docs/operations/model_relabel/meeting_folds.csv）。学びの側の calib_k=1 の開催は学びに入れず較正に回す。
- 途中の木の予測（spec の rounds）を全部保存する（打ち切りを L で選ぶため）。部品ごとに保存して再開できる。
- 漏れの注意（点検役・11/27）: v5bag にも s_time_pred・s_time_gap・s_rank_pred（leave-one-year-out の上流の学習。後の年の着順が入る）と、
  sr_・f3_・f2_・e_・pc_ の族（2021-12 までの全体の当てはめ）がある。ラベルどうしの比べ（同じ列）には効かないが、値の水準は楽観に寄る。
  s_* だけ外した版が v5bag_mfs。
- v6e の代理の注意: 束の 1,231 列には、v5 で外した列が 148 列ある（rb_*・mtl_* は未来の年を含む上流の学習・集計、
  tl_f_untimed_later はその走の後の値）。本番の列のまま比べるので外さないが、値の水準と Benter の改善分は汚れる。主の判定は v5bag で行う。
  乱数は 1 本（束は 5 本）、カテゴリの番号は X_xgb の cat.codes（束の cat_levels は当てない）。
- 2025 年以降の行は読み込んだ直後に落とす。費用 0 円（手元の表だけ。GCS を読まない）。nice -n 19 で流す。
  KEIBA_SESSION=model-relabel nice -n 19 python -m src.research.model_relabel.mr_train run --fs v5bag --target win --variant orig
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.model_relabel import mr_labels as LB

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "config/model_relabel/spec.json"
OUT = LB.OUT
PARTS = OUT / "parts"
V6E_BUNDLE = Path(os.environ.get("MODEL_RELABEL_V6E_BUNDLE")
                  or "/home/jovyan/work/retrain_out/base_2016/final_v6e16/bundle_v6_2025/bundle.json")
FOLDS_CSV = ROOT / "docs/operations/model_relabel/meeting_folds.csv"
YEARS = range(2017, 2025)
# v5bag_mf で外す列: 追い切り評価の 3 列（公開の意見で人気と重なりうる。mr-5）と course_type（推論で 100% 空・学習だけ値。mr-8 点検役）
MARKET_POLICY_DROP = ("oikiri_eval", "oikiri_eval_rank", "oikiri_eval_gap", "course_type")
LEAK_S_COLS = ("s_time_pred", "s_time_gap", "s_rank_pred")      # leave-one-year-out の上流の学習（後の年の着順が入る）。v5bag_mfs で外す（mr-8）
MARKET_CORR = ROOT / "docs/operations/model_relabel/market_corr_2016_2021.csv"      # commit して凍結の照合に入れる
MARKET_CHOICE = ROOT / "docs/operations/model_relabel/market_choice.json"
NTHREAD = int(os.environ.get("MODEL_RELABEL_NTHREAD") or 4)


def spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def fs_configs(fs: str) -> tuple[list[dict], list[int]]:
    """木の設定（固定）と保存する本数の一覧。"""
    from src.research.megu_cycles import mc_data as MC
    sp = spec()["select"]
    if fs.startswith("v5bag"):
        sets = [dict(MC.XGB_PARAMS, **bs) for bs in MC.BAG_SETS]
        rounds = [int(r) for r in sp["rounds_v5bag"]]
    elif fs == "v6e":
        b = json.loads(V6E_BUNDLE.read_text(encoding="utf-8"))
        m = next(x for x in b["members"] if x["name"] == "xgb_extras_ped")
        p = m["params"]
        sets = [dict(objective="binary:logistic", tree_method="hist", device="cuda", eval_metric="logloss", seed=0,
                     max_depth=int(p["max_depth"]), learning_rate=float(p["learning_rate"]), subsample=float(p["subsample"]),
                     colsample_bytree=float(p["colsample_bytree"]), reg_lambda=float(p["reg_lambda"]),
                     min_child_weight=float(p["min_child_weight"]))]
        rounds = [int(r) for r in sp["rounds_v6e"]]
        assert rounds[-1] == int(p["n_estimators"])
    else:
        raise ValueError(fs)
    for s in sets:
        s["nthread"] = NTHREAD
    return sets, rounds


def load_matrix(fs: str):
    """X（float32）と meta（race_id・horse_id・year・finish_position）を year<=2024 の行だけ返す。"""
    from src.research.megu_cycles import mc_data as MC
    if fs.startswith("v5bag"):
        X, meta, names = MC._rich_matrix("v5")
        if fs.startswith("v5bag_mf"):
            # 市場を確実に外す版（11/27 ユーザーの指定・longshot-ai の market_free_columns.md の規則 8）: v5 の 1,083 列に市場の列は 0。
            # 公開の意見で人気と重なりうる追い切り評価の 3 列を方針で外す
            drop = [c for c in names if c in MARKET_POLICY_DROP]
            assert len(drop) == len(MARKET_POLICY_DROP), drop
            X = X.drop(columns=drop)
            names = [c for c in names if c not in MARKET_POLICY_DROP]
        if fs == "v5bag_mfs" or fs.startswith("v5bag_mfs_apt"):
            drop = [c for c in names if c in LEAK_S_COLS]
            assert len(drop) == len(LEAK_S_COLS), drop
            X = X.drop(columns=drop)
            names = [c for c in names if c not in LEAK_S_COLS]
        if fs in ("v5bag_mfc05", "v5bag_mfc03"):
            # 市場の logit と相関の高い列を外す版（spec の market_corr。2016〜21 で測った表 market_corr_2016_2021.csv）
            th = {"v5bag_mfc05": 0.5, "v5bag_mfc03": 0.3}[fs]
            t = pd.read_csv(MARKET_CORR)
            assert set(names) <= set(t["column"]), "相関の表に無い列がある"      # 表は mr-6 の v5bag_mf（course_type を含む 1,080 列）で測った
            drop = set(t.loc[t["abs_rho"] >= th, "column"])
            X = X.drop(columns=[c for c in names if c in drop])
            names = [c for c in names if c not in drop]
        if fs.startswith("v5bag_mfs_apt"):
            # aptitude-group の血統ベクトルの合い具合の 9 列（src/research/aptitude_group/megu_cols.py。年ごとの切り口・as-of）を足す。
            # v5bag_mfs_apt = 本物、v5bag_mfs_aptnull{101,102,103} = 偽（血統の塊を生年の中で入れ替えた列）。2016 年・障害の行は NaN
            from src.research.aptitude_group import megu_cols as AG
            from src.research.aptitude_group import common as AGC
            suf = fs[len("v5bag_mfs_apt"):]
            src = [c if not suf else f"{c}__{suf}" for c in AG.COLS]
            ap = pd.read_parquet(AGC.OUT / "megu_cols.parquet", columns=["race_id", "horse_id"] + src)
            k = meta[["race_id", "horse_id"]].astype(str).merge(ap, on=["race_id", "horse_id"], how="left")
            assert len(k) == len(X)
            yr_ = pd.to_numeric(meta["year"]).to_numpy()
            cover = k[src[0]].notna().to_numpy()[yr_ >= 2017].mean()
            assert cover > 0.85, f"血統の列の当たりが少ない（{cover:.3f}）: 鍵の型・書式を確かめる"     # 2016 年は空・障害は空（critic 7 回目の 4）
            for c, s_ in zip(AG.COLS, src):
                X[c] = k[s_].to_numpy(np.float32)
            names = list(names) + list(AG.COLS)
        if fs == "v5bag_ped":
            # 血統の 4 列（mr_ped.py。凍結の第 4 版の as-of の作り・推論の側も同じ関数）を足す。feat_v4 に無い走は NaN
            from src.research.model_relabel import mr_ped
            ped = pd.read_parquet(mr_ped.DEST)
            k = meta[["race_id", "horse_id"]].astype(str).merge(ped, on=["race_id", "horse_id"], how="left")
            assert len(k) == len(X)
            for c in mr_ped.PED_COLS:
                X[c] = k[c].to_numpy(np.float32)
            names = list(names) + mr_ped.PED_COLS
    else:
        import pyarrow.parquet as pq
        cols = json.loads(V6E_BUNDLE.read_text(encoding="utf-8"))
        cols = next(x for x in cols["members"] if x["name"] == "xgb_extras_ped")["feature_cols"]
        yr_all = pd.to_numeric(pq.read_table(MC.META_RICH, columns=["year"]).column("year").to_pandas()).to_numpy()
        keep = yr_all <= 2024
        meta = pd.read_parquet(MC.META_RICH, filters=[("year", "<=", 2024)])
        assert len(meta) == keep.sum()
        X = pd.read_parquet(MC.X_RICH, columns=cols)
        X = X[keep].reset_index(drop=True)
        for c in X.columns:
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].cat.codes.where(X[c].notna(), np.nan)
            elif not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")
        X, names = X.astype(np.float32), cols
    meta = meta.reset_index(drop=True)
    assert pd.to_numeric(meta["year"]).max() <= 2024
    return X, meta, names


def attach_labels(meta: pd.DataFrame, conv_key: str = "convert_A", max_year: int = 2024) -> pd.DataFrame:
    """meta の行の順に、結果表の拡張ラベルを付ける（race_id・horse_id で結ぶ。行数と y1 の一致を確かめる）。"""
    path = OUT / f"labels_{conv_key}.parquet"
    lab = pd.read_parquet(path, filters=[("year", "<=", max_year)]) if path.exists() else LB.labels_dev(conv_key)
    lab = lab[lab["year"] <= max_year].drop_duplicates(["race_id", "horse_id"])
    key = meta[["race_id", "horse_id"]].astype(str)
    m = key.merge(lab[["race_id", "horse_id", "y1", "y3", "y_w", "y_p", "y_p1", "big", "ext_ok", "chalk_race"]], on=["race_id", "horse_id"], how="left")
    assert len(m) == len(meta)
    miss = m["y1"].isna()
    fin = pd.to_numeric(meta["finish_position"], errors="coerce").to_numpy()
    # 結果表に無い行（取消など）は負例・大敗でない
    assert (fin[miss.to_numpy()] != 1).all(), "結果表に無い勝ち馬がいる"
    for c in ("y1", "y3", "y_w", "y_p", "y_p1", "big"):
        m[c] = m[c].fillna(False).astype(bool)
    # 本命決着の印はレースの値（結果表に無い行＝取消などにもレースの値を引き継ぐ）
    m["chalk_race"] = m["race_id"].map(lab.groupby("race_id")["chalk_race"].any()).fillna(False).astype(bool).to_numpy()
    m["ext_ok"] = m["ext_ok"].fillna(True).astype(bool)
    agree = (m["y1"].to_numpy() == (fin == 1)).mean()
    assert agree > 0.999, f"meta と結果表の 1 着の一致が低い: {agree}"
    m["year"] = pd.to_numeric(meta["year"]).to_numpy(int)
    m["meeting"] = m["race_id"].str[:8]
    return m


def target_and_weight(lab: pd.DataFrame, target: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    big = lab["big"].to_numpy()
    variant = variant.removesuffix("_nc")          # _nc（本命決着のレースを学習から除く版）はラベル・重みは同じ。行の除き方は run で
    if target == "notbig":
        assert variant == "orig"
        return (~big).astype(np.float32), np.ones(len(lab), np.float32)
    col = {"win": ("y1", "y_w"), "top3": ("y3", "y_p")}[target]
    if variant.startswith("ext1_"):                      # 複勝の 1 馬身の版（11/27 に追加）
        assert target == "top3"
        col = ("y3", "y_p1")
    if variant == "orig":
        return lab[col[0]].to_numpy(np.float32), np.ones(len(lab), np.float32)
    w = float(variant.split("_w")[1])
    y = lab[col[1]].to_numpy(np.float32)
    wt = np.where(big & (y == 0), w, 1.0).astype(np.float32)
    return y, wt


def _part_path(split, fs, target, variant, conv, bag, unit) -> Path:
    return PARTS / split / f"{fs}_{target}_{variant}_{conv}" / f"bag{bag}_{unit}.npy"


RACE_DAYS = {"1128", "1129", "1130", "1205", "1206"}      # 土日のほかの開催日（月曜の開催など）


def blocked(now=None) -> bool:
    """開催日（土日と RACE_DAYS）の 07:30〜17:00（JST）は重い学習を止める。"""
    import datetime as dt
    now = now or dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    day = now.weekday() >= 5 or now.strftime("%m%d") in RACE_DAYS
    hm = now.hour * 100 + now.minute
    return day and 730 <= hm < 1700


def _fingerprint(lab: pd.DataFrame, target: str, variant: str, sets: list[dict], rounds: list[int], names=None) -> dict:
    """部品の作りの印（spec の版・ラベルの中身・木の設定・本数）。違う印の部品は混ぜない。"""
    import hashlib
    y, wt = target_and_weight(lab, target, variant)
    h = hashlib.md5(np.ascontiguousarray(np.stack([y, wt])).tobytes()).hexdigest()
    fp = {"spec_version": spec()["version"], "label_md5": h, "sets": sets, "rounds": rounds}
    if names is not None:                          # 特徴量の列の一覧（外す列を後で変えたら混ぜない）
        fp["cols_md5"] = hashlib.md5("\n".join(map(str, names)).encode()).hexdigest()
    if variant.endswith("_nc"):                    # 学習の行の除き方も印に入れる
        fp["chalk_md5"] = hashlib.md5(np.ascontiguousarray(lab["chalk_race"].to_numpy(bool)).tobytes()).hexdigest()
    return fp


def _check_manifest(d: Path, fp: dict) -> None:
    m = d / "manifest.json"
    d.mkdir(parents=True, exist_ok=True)
    if m.exists():
        def strip(x, keys):    # 比べるのはラベルの md5・木の設定・本数・列の一覧。スレッドの数（11/27 に 4 → 2）と spec の版（記録だけ）は値に関わらない
            x = json.loads(json.dumps(x))
            x.pop("spec_version", None)
            for st in x.get("sets", []):
                st.pop("nthread", None)
            return {k: v for k, v in x.items() if k in keys}
        old = json.loads(m.read_text(encoding="utf-8"))
        keys = set(old) & set(fp)          # 列の一覧の印（cols_md5）は 11/27 夜から。前の部品は列が v5 の決まりで固定だったので、その鍵は比べない
        if set(old) - set(fp) or strip(old, keys) != strip(fp, keys):
            raise SystemExit(f"部品の作りの印が違う（古い部品を混ぜない）: {d}")
    else:
        m.write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")


def run(fs: str, target: str, variant: str, split: str = "time", conv: str = "convert_A", only_unit: str | None = None) -> Path:
    import xgboost as xgb
    from src.ops import data_ledger as DL
    t0 = time.time()
    tok = DL.guard(tool="src/research/model_relabel/mr_train.py:run", family="aptitude_group" if fs.startswith("v5bag_mfs_apt") else "model_relabel",
                   version=f"train-{split}-{fs}-{target}-{variant}-{conv}",
                   period=("2016-01-01", "2024-12-31"), purpose="reproduce", reads=("着順",),
                   frozen_paths=[SPEC] + ([MARKET_CORR, MARKET_CHOICE] if fs.startswith("v5bag_mfc")
                                          else [MARKET_CHOICE] if fs == "v5bag_mfs" and variant != "orig" else []),
                   reason="拡張ラベルの前へ進める学習の教師（着順・着差・時計）を読むだけ。成績は数えない")
    X, meta, names = load_matrix(fs)
    lab = attach_labels(meta, conv)
    y, wt = target_and_weight(lab, target, variant)
    yr = lab["year"].to_numpy()
    sets, rounds = fs_configs(fs)
    if split == "time":
        units = [(str(t), yr < t, yr == t) for t in YEARS]
        if variant.endswith("_nc"):          # 本命決着（1〜3 着がすべて 3 番人気以内）のレースを学習の行からだけ除く。予測は全行
            keep = ~lab["chalk_race"].to_numpy()
            units = [(u, tr & keep, te) for u, tr, te in units]
    else:
        f = pd.read_csv(FOLDS_CSV, dtype={"meeting": str}).set_index("meeting")
        mt = lab["meeting"].map(f["fold"]).to_numpy()
        assert not np.isnan(mt.astype(float)).any()
        units = []
        for j in range(5):
            cal = lab["meeting"].map(f[f"calib_{j}"]).to_numpy() == 1
            units.append((f"f{j}", (mt != j) & ~cal, (mt == j) | cal))     # 予測は外側の折と較正の側の両方
    todo = [u for u in units if u[0] == only_unit] if only_unit else units
    names_fp = list(names)
    if fs.startswith("v5bag_mfs_apt"):              # 列の中身（megu_cols.parquet）を作り直したら、古い部品を混ぜない（critic 7 回目の 5）
        import hashlib
        from src.research.aptitude_group import common as AGC
        names_fp.append("megu_cols_md5:" + hashlib.md5((AGC.OUT / "megu_cols.parquet").read_bytes()).hexdigest())
    _check_manifest(_part_path(split, fs, target, variant, conv, 0, "x").parent, _fingerprint(lab, target, variant, sets, rounds, names_fp))
    for bi, params in enumerate(sets):
        for unit, tr, te in todo:
            p = _part_path(split, fs, target, variant, conv, bi, unit)
            if p.exists():
                continue
            if blocked():
                tok.finish(status="error", error="開催日の日中で止めた")
                raise SystemExit("開催日の 07:30〜17:00 なので止める（部品は保存済み。続きから流せる）")
            m = xgb.train(params, xgb.DMatrix(X[tr], label=y[tr], weight=wt[tr]), rounds[-1])
            dte = xgb.DMatrix(X[te])
            P = np.stack([m.predict(dte, iteration_range=(0, r)) for r in rounds]).astype(np.float32)
            tmp = p.with_suffix(".tmp.npy")
            np.save(tmp, P)
            os.replace(tmp, p)
            print(f"  {split} {fs} {target} {variant} bag{bi} {unit}: 学習 {int(tr.sum())} → 予測 {int(te.sum())}（{round(time.time() - t0)} 秒）",
                  flush=True)
    dest = assemble(split, fs, target, variant, conv, lab, units, len(sets), rounds)
    tok.finish(result_path=dest)
    return dest


def assemble(split, fs, target, variant, conv, lab, units, n_bags, rounds) -> Path:
    """部品を束ねて、行ごとの予測（本数ごと・袋の平均）の表にする。"""
    if split == "meeting":
        return _assemble_meeting(fs, target, variant, conv, lab, units, n_bags, rounds)
    out = lab[["race_id", "horse_id", "year", "meeting"]].copy()
    P = np.full((len(rounds), len(lab)), np.nan, np.float32)
    done = True
    for unit, tr, te in units:
        parts = [_part_path(split, fs, target, variant, conv, bi, unit) for bi in range(n_bags)]
        if not all(p.exists() for p in parts):
            done = False
            continue
        P[:, te] = sum(np.load(p) for p in parts) / n_bags
    for k, r in enumerate(rounds):
        out[f"p_r{r}"] = P[k]
    dest = OUT / "pred" / split / f"{fs}_{target}_{variant}_{conv}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not done:
        dest = dest.with_suffix(".partial.parquet")      # 未完の表は評価に入れない（mr_eval は完了の名前だけを読む）
    out.to_parquet(dest, index=False)
    print("assembled", dest, "完了" if done else "途中", flush=True)
    return dest


def _assemble_meeting(fs, target, variant, conv, lab, units, n_bags, rounds) -> Path:
    """meeting の分けは、折 j ごとに（外側の予測・較正の側の予測）を縦に積んだ表にする（列 fold・role）。"""
    f = pd.read_csv(FOLDS_CSV, dtype={"meeting": str}).set_index("meeting")
    fold = lab["meeting"].map(f["fold"]).to_numpy()
    rows = []
    done = True
    for unit, tr, te in units:
        j = int(unit[1:])
        parts = [_part_path("meeting", fs, target, variant, conv, bi, unit) for bi in range(n_bags)]
        if not all(p.exists() for p in parts):
            done = False
            continue
        acc = sum(np.load(p) for p in parts) / n_bags
        sub = lab.loc[te, ["race_id", "horse_id", "year", "meeting"]].copy()
        sub["fold"] = j
        sub["role"] = np.where(fold[te] == j, "outer", "calib")
        for k, r in enumerate(rounds):
            sub[f"p_r{r}"] = acc[k]
        rows.append(sub)
    dest = OUT / "pred" / "meeting" / f"{fs}_{target}_{variant}_{conv}.parquet"
    if not done:
        dest = dest.with_suffix(".partial.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return dest
    pd.concat(rows, ignore_index=True).to_parquet(dest, index=False)
    print("assembled", dest, flush=True)
    return dest


JOBS = [f"{fs}:{t}:{v}" for fs in ("v5bag", "v6e", "v5bag_ped")
        for t, vs in (("win", ("orig", "ext_w1", "ext_w2", "ext_w4")),
                      ("top3", ("orig", "ext_w1", "ext_w2", "ext_w4", "ext1_w1", "ext1_w2", "ext1_w4")), ("notbig", ("orig",)))
        for v in vs] + [f"v5bag:{t}:{v}_nc" for t in ("win", "top3") for v in ("orig", "ext_w1", "ext_w2", "ext_w4")] \
    + [f"v5bag_mf:{t}:{v}" for t in ("win", "top3") for v in ("orig", "ext_w1", "ext_w2", "ext_w4")] + ["v5bag_mf:notbig:orig"]


def mfc_jobs() -> list[str]:
    """相関で外す版の 8 本（元と market_choice.json の w × 勝ち・複勝 × 2 段）。w は v5bag_mf の select の後に決まる。"""
    p = ROOT / "docs/operations/model_relabel/market_choice.json"
    if not p.exists():
        return []
    ch = json.loads(p.read_text(encoding="utf-8"))
    return [f"{fs}:{t}:{v}" for fs in ("v5bag_mfs", "v5bag_mfc05", "v5bag_mfc03") for t in ("win", "top3") for v in ("orig", f"ext_w{ch[t]}")]


def status() -> None:
    """時間の順の全 53 本（と相関で外す版の 8 本）の済み・残り（部品の数）。止まっても続きから流せる（部品ごとに保存）。"""
    done, left = [], []
    for j in JOBS + mfc_jobs():
        fs, t, v = j.split(":")
        n_bags = 1 if fs == "v6e" else 3
        d = PARTS / "time" / f"{fs}_{t}_{v}_convert_A"
        have = len(list(d.glob("bag*_*.npy"))) - len(list(d.glob("*.tmp.npy"))) if d.exists() else 0
        (done if have >= n_bags * len(YEARS) else left).append(f"{j}（部品 {have}/{n_bags * len(YEARS)}）")
    print(f"済み {len(done)} 本・残り {len(left)} 本")
    print("残り: " + " ".join(x.split("（")[0] for x in left))
    for x in left:
        print("  ", x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "labels", "status"])
    ap.add_argument("--fs", choices=["v5bag", "v5bag_ped", "v5bag_mf", "v5bag_mfs", "v5bag_mfc05", "v5bag_mfc03", "v6e",
                                     "v5bag_mfs_apt", "v5bag_mfs_aptnull101", "v5bag_mfs_aptnull102", "v5bag_mfs_aptnull103"], default="v5bag")
    ap.add_argument("--target", choices=["win", "top3", "notbig"], default="win")
    ap.add_argument("--variant", default="orig")
    ap.add_argument("--split", choices=["time", "meeting"], default="time")
    ap.add_argument("--conv", default="convert_A")
    ap.add_argument("--only-unit", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "status":
        return status()
    if a.cmd == "labels":
        dests = [OUT / f"labels_{k}.parquet" for k in ("convert_A", "convert_B_sensitivity")]
        if any(d.exists() for d in dests):   # 上書きしない（mr_labels の save と同じ。先に全部を確かめる）
            raise SystemExit(f"{[str(d) for d in dests if d.exists()]} がある。上書きしない")
        for k in ("convert_A", "convert_B_sensitivity"):
            dest = OUT / f"labels_{k}.parquet"
            tmp = OUT / f"labels_{k}.tmp.parquet"
            LB.labels_dev(k).to_parquet(tmp, index=False)
            os.replace(tmp, dest)
        print("labels saved")
        return
    run(a.fs, a.target, a.variant, a.split, a.conv, a.only_unit)


if __name__ == "__main__":
    main()
