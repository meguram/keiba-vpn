"""tp-37: 追い切りの『残りの余地』を、V4 全体（tp31 の V4 フレーム 1,062 列）に対して試す。

python -m src.scripts.data.tp37_eval build      # 群ごとの特徴量と離散キー（KEYS）を tp37/groups/*.parquet に保存（追い切り表は tp35 と同じ training_laps.parquet）
python -m src.scripts.data.tp37_eval coverage   # 被覆と、新しい標準化の基本統計（調教師のいつもの速さの分布、クラスタの中身など）
python -m src.scripts.data.tp37_eval stage1     # 第 1 段: valid 2022+2023、seed 1 本。V4 に対する増分（追加 19 群 + TL_A の時計列を置き換える 2 通り）
python -m src.scripts.data.tp37_eval stage2     # 第 2 段: 通過群を valid 2024、新 seed 2 本で確認 + 同時追加
python -m src.scripts.data.tp37_eval resid      # V4 の OOF（2022〜24）の勝ち残差が、追い切りの条件・交互作用で説明できるか（レース単位、BH 補正）
python -m src.scripts.data.tp37_eval bet        # V4 の予測 × 追い切り条件の絞り込みの回収率（レース単位。reality check）
python -m src.scripts.data.tp37_eval report     # /tmp/tp37_report.{json,md}
test 2025・2026 は選択に使わない（2025 は参考列のみ）。市場オッズ・人気は特徴量に使わない（馬券の判断にだけ使う）。環境変数: TP37_THREADS（既定 6）, TP37_LOAD_LIMIT（既定 16）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp35_eval as E  # noqa: E402  (環境変数 TP_BASE_DIR などを設定する)
from src.scripts.data import tp27_eval as t  # noqa: E402

ROOT = E.ROOT
V = E.V
OUT = Path(os.environ.get("TP37_OUT") or V / "tp37")
KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP37_THREADS", 6))
LOAD_LIMIT = float(os.environ.get("TP37_LOAD_LIMIT", 16))
E.THREADS, E.LOAD_LIMIT = THREADS, LOAD_LIMIT
BET = E.ROOT / "data/local/modeling/betting"

TEXT_GROUPS = ["D7_text_flags", "D7_text_lift"]
TEXT_EXTRA = ["D8_text_ctx"]
GROUPS = ["TRZ", "TRZ_SELF", "CLZ", "I1_cls_cond", "I2_zt4_cond_int", "I3_zt4_style_surf", "I4_cls_grade", "I5_zt1_courseexact", "I6_clu_zt4_cond", "I7_kind_int_days_cond",
          "I8_trainer_zt4", "I9_zacc_style_dist", "I10_habit_cond", "I11_clu_course", "D1_between", "D2_change", "D3_rider", "D4_eval", "D5_dow_pos"]
TL_Z_COLS = ["tl_f_t4z", "tl_f_t1z", "tl_f_accelz", "tl_w1_t4z", "tl_w1_t1z", "tl_w1_accelz", "tl_w2_t4z", "tl_w2_t1z", "tl_w2_accelz",
             "tl_f_t4z_vs_self", "tl_f_t1z_vs_self", "tl_f_accelz_vs_self", "tl_mean_t4z_14", "tl_best_t1z_14", "tl_best_t1z_28"]
BATCHES = [["I3_zt4_style_surf", "I4_cls_grade", "I5_zt1_courseexact"], ["I6_clu_zt4_cond", "I7_kind_int_days_cond", "I8_trainer_zt4"], ["I9_zacc_style_dist", "I10_habit_cond", "I11_clu_course"],
           ["D1_between", "D2_change"], ["D3_rider", "D4_eval", "D5_dow_pos"]]
VARIANTS = {"TRZ_repl": (["TRZ"], TL_Z_COLS), "TRZ_CLZ_repl": (["TRZ", "TRZ_SELF", "CLZ"], TL_Z_COLS)}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jsave(p: Path, d) -> None:
    E._jsave(p, d)


def _jload(p: Path):
    return E._jload(p)


# ───────────────────────── build
def cmd_build(a) -> None:
    from src.pipeline.features import tp37_workout as W

    df, base, meta, folds, lp = t.setup()
    laps = pd.read_parquet(E.LAPS)
    ret = E._returns(df)
    runs = E._runs(df)
    runs["a_style"] = df["a_style"].to_numpy()
    runs["grade"] = df["grade"].astype(str).where(df["grade"].notna()).to_numpy()
    _log(f"追い切り表 {len(laps):,} 本、レース行 {len(runs):,}")
    feats = W.build_features(runs, laps, ret, fit_before="2021-01-01")
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    cover = {}
    for g, f in feats.items():
        assert (f["race_id"].to_numpy() == df["race_id"].astype(str).to_numpy()).all() and (f["horse_id"].to_numpy() == df["horse_id"].to_numpy()).all()
        f.to_parquet(gdir / f"{g}.parquet", index=False)
        cover[g] = {"cols": int(f.shape[1] - 2), "non_null_share": float(f.drop(columns=KEY).notna().mean().mean())}
        _log(f"{g}: {cover[g]}")
    _jsave(OUT / "build_meta.json", {"laps_rows": int(len(laps)), "rows": int(len(df)), "groups": cover, "n_groups": len(GROUPS), "variants": list(VARIANTS)})


def cmd_build_text(a) -> None:
    """記者コメント（tp37_comments extract の結果）から作る群 D7 だけを保存する（他の群のファイルは触らない）。"""
    from src.pipeline.features import tp37_workout as W

    df, base, meta, folds, lp = t.setup()
    com = pd.read_parquet(OUT / "training_comments.parquet")
    laps = pd.read_parquet(E.LAPS)
    runs = E._runs(df)
    runs["a_style"] = df["a_style"].to_numpy()
    runs["grade"] = df["grade"].astype(str).where(df["grade"].notna()).to_numpy()
    feats = W.build_features(runs, laps, E._returns(df), fit_before="2021-01-01", comments=com)
    only = [x for x in os.environ.get("TP37_TEXT_ONLY", "").split(",") if x]
    for g in (only or TEXT_GROUPS + TEXT_EXTRA):
        f = feats[g]
        assert (f["race_id"].to_numpy() == df["race_id"].astype(str).to_numpy()).all()
        f.to_parquet(OUT / "groups" / f"{g}.parquet", index=False)
        _log(f"{g}: cols={f.shape[1] - 2} 非欠損 {float(f.drop(columns=KEY).notna().mean().mean()):.3f}（コメント {len(com):,} 行）")


def _load_group(g: str, df: pd.DataFrame) -> pd.DataFrame:
    T = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
    T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
    m = df[KEY].merge(T, on=KEY, how="left")
    assert len(m) == len(df)
    m = m[[c for c in m.columns if c not in KEY]]
    m.index = df.index
    return m


# ───────────────────────── coverage
def cmd_coverage(a) -> None:
    df, *_ = t.setup()
    y = pd.to_datetime(df["date"]).dt.year.to_numpy()
    rep: dict = {}
    for g in ("TRZ", "CLZ", "D1_between", "D3_rider", "D4_eval"):
        f = _load_group(g, df)
        rep[g] = {c: {int(k): round(float(f[c].notna().groupby(y).mean().get(k, np.nan)), 3) for k in sorted(set(y))} for c in list(f.columns)[:3]}
    K = _load_group("KEYS", df) if (OUT / "groups/KEYS.parquet").exists() else None
    if K is not None:
        rep["clu_dist"] = {str(k): int(v) for k, v in K["clu"].value_counts(dropna=False).items()}
    f = _load_group("TRZ", df)
    rep["trz_stats"] = {c: {"mean": float(f[c].mean()), "std": float(f[c].std()), "p05": float(f[c].quantile(0.05)), "p95": float(f[c].quantile(0.95))} for c in ("f_z0_t4", "f_dev_t4", "f_ztr_t4", "f_ztrs_t4", "f_ztr_t1", "f_ztr_acc")}
    rep["corr_with_tl"] = {}
    X0 = E._frame(df)
    for c_new, c_old in (("f_z0_t4", "tl_f_t4z"), ("f_ztr_t4", "tl_f_t4z"), ("f_z0_t1", "tl_f_t1z"), ("f_ztr_t1", "tl_f_t1z"), ("f_z0_acc", "tl_f_accelz"), ("f_ztr_acc", "tl_f_accelz")):
        x, yv = f[c_new].to_numpy(dtype=float), X0[c_old].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(yv)
        rep["corr_with_tl"][f"{c_new}~{c_old}"] = float(np.corrcoef(x[ok], yv[ok])[0, 1])
    _jsave(OUT / "coverage.json", rep)
    for k, v in rep["corr_with_tl"].items():
        _log(f"{k}: {v:+.3f}")
    _log(json.dumps(rep["trz_stats"], ensure_ascii=False))


# ───────────────────────── stage1 / stage2
def _run_subset(name, X, meta, folds, lp, years, seeds):
    return E._run_subset(name, X, meta, folds, lp, years, seeds)


def _stage(label: str, years: list[int], seeds: list[int], groups: list[str]) -> None:
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = E._frame(df)
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = _jload(resp)
    bp = d / "base.parquet"
    if bp.exists() and "base" in res:
        oof0 = pd.read_parquet(bp)
    else:
        lv, oof0 = _run_subset("V4", X0, meta, folds, lp, years, seeds)
        oof0.to_parquet(bp, index=False)
        res["base"] = {"levels": lv, "cols": int(X0.shape[1])}
        _jsave(resp, res)
    steps = res.setdefault("groups", {})
    for g in groups:
        if g in steps:
            continue
        if g.startswith("BATCH["):
            G = pd.concat([_load_group(x, df) for x in g[6:-1].split("+")], axis=1)
            drop = []
        elif g in ("ALL_TEXT", "ALL_TEXT3"):
            G = pd.concat([_load_group(x, df) for x in (TEXT_GROUPS if g == "ALL_TEXT" else TEXT_GROUPS + TEXT_EXTRA)], axis=1)
            drop = []
        elif g.startswith("ALL"):
            names = [x for x in groups if not x.startswith("ALL") and x not in VARIANTS]
            G = pd.concat([_load_group(x, df) for x in names], axis=1)
            drop = []
        elif g in VARIANTS:
            add, drop = VARIANTS[g]
            G = pd.concat([_load_group(x, df) for x in add], axis=1)
            drop = [c for c in drop if c in X0.columns]
        else:
            G = _load_group(g, df)
            drop = []
        dup = set(G.columns) & set(X0.columns)
        assert not dup, f"列名が V4 と重複: {sorted(dup)[:5]}"
        X = pd.concat([X0.drop(columns=drop), G], axis=1)
        lv, oof = _run_subset(f"V4+{g}", X, meta, folds, lp, years, seeds)
        c = t._cmp(oof0, oof, fin_map, g)
        steps[g] = {"cols": int(G.shape[1]), "dropped": len(drop), "levels": lv, "vs_V4": c}
        _jsave(resp, res)
        _log(f"  {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")


def cmd_stage1(a) -> None:
    text = [g for g in TEXT_GROUPS + TEXT_EXTRA if (OUT / "groups" / f"{g}.parquet").exists()]
    _stage("stage1", [2022, 2023], [0], text + GROUPS + list(VARIANTS))


def cmd_stage1batch(a) -> None:
    """残りの群を、3〜4 群ずつのバッチ（BATCH[a+b+c]）で画面にかける（1 本あたり約 25 分）。バッチが第 1 段を通れば、構成群を個別に第 2 段へ進める。"""
    groups = [f"BATCH[{'+'.join(b)}]" for b in BATCHES] + [g for g in TEXT_EXTRA if (OUT / "groups" / f"{g}.parquet").exists()] + list(VARIANTS)
    _stage("stage1", [2022, 2023], [0], groups)


def _expand(g: str) -> list[str]:
    return g[6:-1].split("+") if g.startswith("BATCH[") else [g]


def _passed() -> list[str]:
    r = _jload(OUT / "stage1/results.json")
    out: list[str] = []
    for g, s in r.get("groups", {}).items():
        if s["vs_V4"]["Δwin"] < -0.00012 and s["vs_V4"]["win_hi"] < 0.0001:
            out += [x for x in _expand(g) if x not in out]
    return out


def cmd_stage2(a) -> None:
    passed = _passed()
    _log(f"第 1 段を通過: {passed}")
    _jsave(OUT / "stage1/passed.json", passed)
    if not passed:
        return
    text = [g for g in passed if g in TEXT_GROUPS + TEXT_EXTRA]
    pref = [g for g in ("D4_eval", "D3_rider", "D5_dow_pos", "D1_between", "D2_change") if g in passed]
    rest = [g for g in passed if g not in TEXT_GROUPS + TEXT_EXTRA and g not in pref]
    comp = []
    for b in BATCHES:                                                   # 同じバッチの構成群がすべて通過している場合は、バッチ全体も第 2 段に進める
        if all(x in passed for x in b):
            comp.append(f"BATCH[{'+'.join(b)}]")
    t_all = (["ALL_TEXT"] if len([g for g in text if g in TEXT_GROUPS]) > 1 else []) + (["ALL_TEXT3"] if "D8_text_ctx" in text and len(text) > 2 else [])
    singles = [g for g in passed if g not in VARIANTS]
    order = text + t_all + pref[:3] + [c for c in comp if "D3_rider" in c] + pref[3:] + [c for c in comp if "D1_between" in c] + rest
    if os.environ.get("TP37_STAGE2_ALL") == "1" and len(singles) > 1:
        order += [f"ALL[{len(singles)}]"]
    _stage("stage2", [2024], [2, 3], list(dict.fromkeys(order)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "build_text", "coverage", "stage1", "stage1batch", "stage2", "resid", "bet", "followup", "final2026", "tilt", "report", "chapter"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fn = {"build": cmd_build, "build_text": cmd_build_text, "stage1batch": cmd_stage1batch, "coverage": cmd_coverage, "stage1": cmd_stage1, "stage2": cmd_stage2}
    if a.cmd in fn:
        fn[a.cmd](a)
    else:
        from src.scripts.data import tp37_bet as B

        {"resid": B.cmd_resid, "bet": B.cmd_bet, "followup": B.cmd_followup, "final2026": B.cmd_final2026, "tilt": B.cmd_tilt, "report": B.cmd_report, "chapter": B.cmd_chapter}[a.cmd](a)


if __name__ == "__main__":
    main()
