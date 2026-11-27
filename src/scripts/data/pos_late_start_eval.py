"""過去位置取りの集計の比べ（出遅れの扱い）。着順・オッズ・払戻は使わない。的は次走の初角の相対位置だけ。

python -m src.scripts.data.pos_late_start_eval            # → /tmp/pos_late_start/report.json

事前に固定した決まり（数を見る前。2026-11-27）:
- 位置取り = 初角の通過順位 ÷ field_size（本番の past_position と同じ。初角の無い走〔通過順が空〕は使わない）。平地と障害は別。
- 履歴 = 的の走より前の、同じ馬の直近 WINDOW=8 走（新しい順）。的の走より後の走は使わない（出遅れ疑いの判定の中央値も履歴の中だけ）。
- 的の行 = 2016〜2024 年の走のうち、履歴が 1 走以上ある走。2025 年以降は使わない。
- 分け方 = 開催（race_id の先頭 8 桁）で int(md5(先頭8桁).hexdigest()[-1], 16) % 2 が 0 なら探す側、1 なら確かめる側。
- 候補 = M0 今の値（past_position.horse_position）・M1 中央値・M2 KDE の頂上・M3 重み付き KDE の頂上（DECAY=0.85）・
  M4 出遅れ疑いを除いた加重平均・M5 M4 から挽回の条件を外した版（パラメータは position_estimators の定数。格子 0.01・同じ密度は前の位置を採る）。
- 主の的 = 2023〜24 年の走のうち、備考の表（remarks_flags）に載り（戦績ページを取れた走）、「出遅」が無い走。副 = 全走。参考 = 「出遅」の付いた走。
  備考は 2023-10 以降も走った馬に偏る。主と副で結論が割れたら、その旨を報告に書く。
- 採り方: 確かめる側の主の的で、平均絶対誤差が最小の候補を採る。M1〜M5 それぞれの M0 との差を、開催単位のブートストラップ
  （B=2000・乱数の seed 7331）で両側の p 値にし、Holm（5 本）で 5% に届かなければ「差は偶然の範囲」とする。
  最小が M0 か、最小の候補と M0 の差が偶然の範囲なら、M2 と M3 のうち確かめる側の主の的で誤差が小さい方を採る
  （ユーザーの案。それが M0 より有意に悪いときだけ M0 のまま）。
- 層: 履歴の走数（1〜2・3〜7・8）と、履歴の出遅れ疑いの率（25% 以上か）。
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.race_analysis import position_estimators as PE
from src.research.race_analysis.past_position import WINDOW, horse_position

ROOT = Path(__file__).resolve().parents[3]
TABLES = Path(os.environ.get("POS_EVAL_TABLES") or ROOT / "data/page_reference/tables")
REMARKS = Path(os.environ.get("POS_EVAL_REMARKS") or ROOT / "data/local/modeling/training_pipeline_v4/slowreal/remarks_flags.parquet")
OUT = Path(os.environ.get("POS_EVAL_OUT") or "/tmp/pos_late_start")
CANDS = ["M0", "M1", "M2", "M3", "M4", "M5"]
HOW = {"M1": "median", "M2": "kde", "M3": "kde_w", "M4": "mean_nolate", "M5": "mean_nofar"}
BOOT_B, BOOT_SEED = 2000, 7331
HIGH_LATE = 0.25


def side_of(race_id: str) -> int:
    return int(hashlib.md5(race_id[:8].encode()).hexdigest()[-1], 16) % 2


def load() -> pd.DataFrame:
    cols = ["race_id", "date", "surface", "horse_id", "passing_order", "field_size", "last_3f", "finish_position"]
    parts = []
    for f in sorted(glob.glob(str(TABLES / "*/race_result_flat.parquet"))):
        if int(Path(f).parent.name) > 2024:
            continue
        parts.append(pd.read_parquet(f, columns=cols))
    df = pd.concat(parts, ignore_index=True)
    df["date"] = df["date"].astype(str).str[:10]
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    df = df[fp.notna() & (fp > 0)].drop(columns=["finish_position"])     # 本番の load_runs と同じく、着順が確定した走だけ（着順の値は使わない）
    fs = pd.to_numeric(df["field_size"], errors="coerce")
    corners = df["passing_order"].astype("string").str.findall(r"\d+")
    first = pd.to_numeric(corners.str[0], errors="coerce")
    last = pd.to_numeric(corners.str[-1], errors="coerce").where(corners.str.len() >= 2)
    l3 = pd.to_numeric(df["last_3f"], errors="coerce")
    l3_pct = l3.groupby(df["race_id"]).rank(method="min") / fs          # 上がりの順位は、初角で絞る前の完走した全頭で付ける
    ok = first.between(1, fs)
    df = df.assign(fs=fs, r=first / fs, r_last=(last / fs).where(last.between(1, fs)), l3_pct=l3_pct)[ok].copy()
    df["hurdle"] = df["surface"].astype(str).str.startswith("障")
    rem = pd.read_parquet(REMARKS, columns=["horse_id", "race_id", "slow_rem", "has_remarks"]).drop_duplicates(["horse_id", "race_id"])
    df = df.merge(rem, on=["horse_id", "race_id"], how="left")
    return df.sort_values(["horse_id", "hurdle", "date", "race_id"]).reset_index(drop=True)


def rows(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (_, _), g in df.groupby(["horse_id", "hurdle"], sort=False):
        r, rl, lp = g["r"].tolist(), g["r_last"].tolist(), g["l3_pct"].tolist()
        rid, dt, sr, hr = g["race_id"].tolist(), g["date"].tolist(), g["slow_rem"].tolist(), g["has_remarks"].tolist()
        for t in range(1, len(g)):
            lo = max(0, t - WINDOW)
            first = r[lo:t][::-1]
            last = [None if pd.isna(x) else x for x in rl[lo:t][::-1]]
            l3p = [None if pd.isna(x) else x for x in lp[lo:t][::-1]]
            late = PE.late_flags(first, last, l3p)
            far = PE.late_flags(first, last, l3p, need_recover=False)
            rec = {"race_id": rid[t], "year": int(dt[t][:4]), "y": r[t], "n": len(first),
                   "late_n": int(sum(late)), "far_n": int(sum(far)), "slow_rem": sr[t], "has_rem": hr[t],
                   "M0": horse_position(first).value, "m0_excl": len(horse_position(first).excluded)}
            for k, how in HOW.items():
                rec[k] = PE.estimate(first, how, far if k == "M5" else late)
            rec["bimodal"] = len(first) >= PE.MIN_RUNS and len(PE.kde_peaks(first)) > 1
            out.append(rec)
    return pd.DataFrame(out)


def boot_p(d: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> float:
    """開催単位のブートストラップで、誤差の差（a − b）の両側 p 値。"""
    diff = (d[a] - d["y"]).abs() - (d[b] - d["y"]).abs()
    g = diff.groupby(d["race_id"].str[:8]).agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(s), size=(BOOT_B, len(s)))
    m = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))


def mae_raw(d: pd.DataFrame) -> dict[str, float]:
    return {k: float((d[k] - d["y"]).abs().mean()) for k in CANDS}


def mae(d: pd.DataFrame) -> dict[str, float]:
    return {k: round(v, 4) for k, v in mae_raw(d).items()} | {"rows": int(len(d))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    R = rows(df)
    R["side"] = R["race_id"].map(side_of)
    R["late_rate"] = R["late_n"] / R["n"]
    R.to_parquet(OUT / "rows.parquet")
    main_t = R["year"].between(2023, 2024) & R["has_rem"].notna() & (R["slow_rem"] == 0)
    rep: dict = {"rows": int(len(R)), "m0_excludes_share": round(float((R["m0_excl"] > 0).mean()), 4),
                 "late_any_share": round(float((R["late_n"] > 0).mean()), 4), "far_any_share": round(float((R["far_n"] > 0).mean()), 4), "bimodal_share": round(float(R["bimodal"].mean()), 4)}
    for side, name in ((0, "search"), (1, "verify")):
        S = R[R["side"] == side]
        mt = main_t[R["side"] == side]
        sec: dict = {"main": mae(S[mt]), "all": mae(S), "slow_rem1": mae(S[S["slow_rem"] == 1])}
        for lab, m in (("n1_2", S["n"] <= 2), ("n3_7", S["n"].between(3, 7)), ("n8", S["n"] >= 8),
                       ("late_high", S["late_rate"] >= HIGH_LATE), ("late_any", S["late_n"] > 0), ("bimodal", S["bimodal"])):
            sec[f"all_{lab}"] = mae(S[m])
            sec[f"main_{lab}"] = mae(S[mt & m])
        rep[name] = sec
    V = R[(R["side"] == 1) & main_t]
    rng = np.random.default_rng(BOOT_SEED)
    ps = {k: boot_p(V, k, "M0", rng) for k in CANDS[1:]}
    order = sorted(ps, key=ps.get)
    holm, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, ps[k] * (len(order) - i)))
        holm[k] = round(run, 4)
    rep["verify_p_vs_M0"] = {k: round(v, 4) for k, v in ps.items()}
    rep["verify_holm_vs_M0"] = holm
    vm = mae_raw(V)                                            # 選ぶのは丸める前の値
    best = min(CANDS, key=lambda k: vm[k])
    if not (best != "M0" and holm[best] < 0.05):          # M0 より有意に良い候補が最小でなければ、M2・M3 の良い方（有意に悪ければ M0）
        alt = min(("M2", "M3"), key=lambda k: vm[k])
        best = "M0" if (vm[alt] > vm["M0"] and holm[alt] < 0.05) else alt
    rep["adopted"] = best
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
