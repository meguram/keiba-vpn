"""前向きの記録（1 号の指示 1。宣言には使わない・将来の評価の材料を貯めるだけ）。手で流す道具（cron・常駐は入れない）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.forward_record --date 2026-11-28

- その日より前の結果（race_result_flat）だけで as-of の特徴量を作り、その日の出馬表（race_shutuba_flat）の馬に、
  凍結の第 4 版（2 代母の単位の 5 折の平均）の勝率を当てる。初ダート（FD）・初の距離帯（DB0）の印も付ける。
- 出力: docs/operations/pedigree_lab/forward/<日付>.csv（commit の時刻で発走前を示す）。オッズは使わない。
- 0 円（手元の表と 5 代血統の JSON だけ。GCS を読まない）。開催日の 07:30〜17:00 には流さない。本番・画面には出さない。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree_lab import build_base, data, features4, final_eval, gates, scene_dist

ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "docs/operations/pedigree_lab/forward"
JST = timezone(timedelta(hours=9))


def entries(day: pd.Timestamp) -> pd.DataFrame:
    S = pd.read_parquet(build_base.TABLES / str(day.year) / "race_shutuba_flat.parquet")
    S["date"] = pd.to_datetime(S["date"])
    S = S[(S["date"] == day) & S["surface"].isin(["芝", "ダート"])].copy()
    S["horse_id"] = S["horse_id"].astype(str)
    S["finish_position"] = 99            # 未確定（as-of の集計では、その日より後の行が無いので使われない）
    S["finish_status"] = None
    S["popularity"] = np.nan
    S["payoff"] = None
    S["odds"] = np.nan                   # オッズは使わない
    S["meeting"] = S["race_id"].str[:8]
    S["sex"] = S["sex_age"].str[0]
    S["age"] = pd.to_numeric(S["sex_age"].str[1:], errors="coerce")
    return S.reindex(columns=[*build_base.COLS, "meeting", "sex", "age"])


def dirt_aptitude(X: pd.DataFrame, ped: pd.DataFrame) -> dict[str, np.ndarray]:
    """ダートの適性の 4 列（父・母父・牝系 5 代・牝系の家系の質）。aptitude・aptitude_checks の凍結の作りのまま（as-of・自分を引く）。
    ダートでない行の値も出すが、記録の評価はダートの行だけで行う（log の事前の登録）。"""
    from src.research.pedigree_lab import aptitude as ap_
    from src.research.pedigree_lab import aptitude_checks as ck
    F = X[["date", "horse_id", "surf", "fpct", "sire", "ds"]].reset_index(drop=True)
    A, keys = ap_.group_a(F.assign(venue="", distance=0, going="", rage="", dchg=""), "dirt")
    L = ap_.maternal_long(F, drop_common=False, ped=ped)
    mat5, lv5 = ck.mat_cols(F, A, keys, L)
    return {"dirt_sire": ap_.apt_by(F, A, ["sire"] + keys), "dirt_bms": ap_.apt_by(F, A, ["ds"] + keys),
            "dirt_mat5": mat5, "dirt_lv5": lv5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--backfill", action="store_true", help="発走前に commit できなかった日を後から作る（meta に backfill: true。prereg_dirt_mat5.md）")
    a = ap.parse_args()
    day = pd.Timestamp(a.date)
    # 台帳（統括役の答え・11/26）: 2016-01-01〜2026-11-20 は reproduce（凍結の第 4 版の再現。見た回数に数えない。凍結のファイルを渡す）、
    # 2026-11-21〜前日は forward。凍結の模型は公開のリポジトリに置かないので、sha256 は理由の文に書く
    frozen = [ROOT / "src/research/pedigree_lab" / f for f in
              ("forward_record.py", "gates.py", "features4.py", "features2.py", "data.py", "final_eval.py", "scene_dist.py", "build_base.py",
               "model.py", "hyp_tests.py", "bets.py", "pl_ledger.py", "aptitude.py", "aptitude_checks.py")]
    # 凍結の模型を sha256 で照合する（違えば止める。critic 4 回目の 6）
    import hashlib
    fz = data.OUT / "freeze_v4_gate.sha256"
    if not fz.exists():
        raise SystemExit("freeze_v4_gate.sha256 が無い（凍結の模型を照合できない）")
    for line in fz.read_text().splitlines():
        h, name = line.split(maxsplit=1)
        if hashlib.sha256((data.OUT / name.strip()).read_bytes()).hexdigest() != h:
            raise SystemExit(f"凍結の模型と違う: {name}")
    sha = fz.read_text().split()[0][:16]
    data.TOKENS.append(data._guard(("2016-01-01", "2026-11-20"), "reproduce", f"forward_{a.date}_v4",
                                   "凍結の第 4 版で前向きの出馬表の血統の点を作るための as-of の特徴量の元。着順は産駒の成績の集計にだけ使い、"
                                   f"評価・選び直し・閾値の決めに使わない（模型 models_v4_pure_dd・kf の sha {sha}…・freeze_v4_gate.sha256）",
                                   frozen_paths=frozen))
    if day > pd.Timestamp("2026-11-21"):
        data.TOKENS.append(data._guard(("2026-11-21", str((day - pd.Timedelta(days=1)).date())), "forward", f"forward_{a.date}_recent",
                                       f"前向きの記録 {a.date} の as-of の特徴量の元（2026-11-21〜前日。評価ではない）"))
    runs = build_base.load_runs()
    runs = runs[runs["date"] < day]
    E = entries(day)
    if E.empty:
        raise SystemExit(f"{a.date} の出馬表が無い")
    r = data._prep(pd.concat([runs, E], ignore_index=True))
    ped = data.load_ped()
    miss = sorted(set(E["horse_id"]) - set(ped["horse_id"]))
    if miss:
        ped = pd.concat([ped, build_base.load_ped(miss)], ignore_index=True)
    X, kinfo = features4.build(r, ped, sub=True)
    frozen = json.loads((data.OUT / "kinfo_v4.json").read_text())
    if any(abs(kinfo[k] - frozen[k]) > 1e-6 for k in frozen if k != "rho") or kinfo["rho"] != frozen["rho"]:
        raise SystemExit("縮めの k が凍結の第 4 版と違う（作りがずれている。critic 7）")
    X["split"] = -1                      # gates.FCOLS の列をそろえる（前向きの行は分けに入らない）
    F = gates.scenes(X[gates.FCOLS].copy())
    D = scene_dist.marks(X[list(dict.fromkeys(gates.FCOLS + ["dband", "going"]))].copy())
    X["FD"], X["DB0"] = F["FD"].to_numpy(), D["DB0"].to_numpy()
    # ダートの牝系 5 代の適性の候補（11/26 に凍結。統括役の承認・記録だけ）。生の値を残し、標準化は評価の時に行う
    dm = dirt_aptitude(X, ped)
    T = X[X["date"] == day].copy()
    for k, v in dm.items():
        T[k] = v[(X["date"] == day).to_numpy()]
    p = final_eval.predict(T, "v4")
    T["p_win"] = p
    T["rank"] = T.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)
    out = T[["race_id", "date", "venue", "surface", "distance", "horse_number", "horse_id", "p_win", "rank", "FD", "DB0", *dm]].copy()
    out["p_win"] = out["p_win"].round(5)
    for k in dm:
        out[k] = out[k].round(6)
    out["date"] = out["date"].dt.date.astype(str)
    out["no_pedigree"] = (T["sire"].isna() | T["sire"].eq("")).to_numpy()      # 血統の JSON が無い馬（critic 6）
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{a.date}.csv"
    if path.exists():
        raise SystemExit(f"{path.name} は既にある（記録は後から変えない。prereg_dirt_mat5.md）")
    # 発走前かどうかは道具が決める: 作った時刻がその日の 07:30 JST 以降なら後から作った日（自己申告と食い違えば止める。critic）
    late = datetime.now(JST) >= datetime(day.year, day.month, day.day, 7, 30, tzinfo=JST)
    if late != bool(a.backfill):
        raise SystemExit(f"--backfill の印が作った時刻と合わない（その日の 07:30 以降={late}）")
    meta = {"made_at": datetime.now(JST).isoformat(timespec="seconds"), "model": "v4_pure_dd（freeze_v4_gate.sha256）",
            "races": int(out["race_id"].nunique()), "horses": len(out), "no_pedigree": int(out["no_pedigree"].sum()),
            "backfill": bool(a.backfill)}
    out.sort_values(["race_id", "rank"]).to_csv(path, index=False)
    (OUTDIR / f"{a.date}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
