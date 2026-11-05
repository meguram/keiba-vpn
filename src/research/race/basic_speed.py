"""基礎スピード（走法マップの 6 軸目）の 1 走ごとの材料。

集計は良馬場のレース・走だけ（ユーザー指定 2026-11-05。稍重〜不良は含水で時計が変わるので、速さの基準にも馬の走にも入れない）。
定義（ユーザー指定 2026-11-05）: 馬場が速い（良馬場・開幕週など走破タイムが速くなりやすい）ときに、いつも通りの走りができるタイプ
（追走を難なくこなし、自分の走りができる）。低い馬は一般に足が遅く、時計のかかる馬場（開催が進んだ馬場・ダート・パワー型）が向く。
Hペース耐性（消耗耐性）とは別物: あちらは基礎スピードのレンジを超えて速い流れになり、終いに脚が残らない中でも走れるか。

作り方: 各走に (p, s) を付ける。
  p  そのレース内の着順スコア（+1 = 1 着 … -1 = 最下位。レースごとに平均 0）。
  s  その日・その競馬場・その芝ダの「速さ」の標準化（大 = 速い）。各レースの勝ち時計（200m あたり）から、
     コース・距離・クラスごとの中央値を引いたものを、同じ日・場・芝ダの「そのレース自身を除く」平均にし、芝ダごとに標準化。
     自分のレースの時計は s に入れない（その走の結果で自分の s が動くのを防ぐ）。同じ日の他のレースが 3 つ未満なら s は欠測。
馬ごとの基礎スピード = 直近の走での (p, s) の共分散（速い馬場で成績が上がるほど高い）。共分散の組み立て（加重・引き寄せ）は
running_profile.profiles_as_of / profile_table がやるので、ここでは 1 走ごとの p, s を作る。
中央値の基準は全期間で測っている（未来の走の値は使っていないが、基準の中央値だけは全期間）。

検証（2022〜2025、次走の着順スコア p を、直近 10 走までの共分散 D・次走の馬場の速さ s・両者の積で回帰。host の /tmp/bs_eval.py）:
  積の係数 +0.0092（t = 5.1）。速い馬場（s > 0.7）では D 上位 20% の馬が下位 20% より p が +0.015、
  遅い馬場（s < -0.7）では -0.042。方向は定義どおり（高い馬は速い馬場で上がり、時計のかかる馬場で下がる）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASIC_SCALE = 90.0       # 積 p×s の尺度（馬ごとのプロファイルの値が 50 ± 3 程度に収まるように）
STORE_VERSION = "2"
MIN_PAR_RACES = 15       # コース・距離・クラスの基準タイムを取るのに要るレース数
MIN_DAY_RACES = 3        # 同じ日・場・芝ダの他のレースがこれ未満なら速さは欠測
MIN_FIELD = 5
_JUMP = r"障害|ジャンプ|JS\(|JG|J・G"
_COLS = ["race_id", "date", "venue", "surface", "distance", "grade", "race_name", "track_condition", "finish_position", "horse_id", "time_sec"]


def _read(years: Iterable[int], base: Path) -> pd.DataFrame:
    parts = []
    for y in years:
        f = base / str(y) / "race_result_flat.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=_COLS))
    if not parts:
        return pd.DataFrame(columns=_COLS)
    d = pd.concat(parts, ignore_index=True)
    d = d[d["surface"].isin(["芝", "ダート"]) & ~d["race_name"].fillna("").str.contains(_JUMP)].copy()
    for c in ("distance", "finish_position", "time_sec"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["finish_position"] > 0) & (d["track_condition"] == "良")].copy()      # 良馬場だけ（ユーザー指定 2026-11-05）
    d["horse_id"] = d["horse_id"].astype(str)
    d["race_id"] = d["race_id"].astype(str)
    d["date"] = pd.to_datetime(d["date"])
    return d.reset_index(drop=True)


def compute(runs: pd.DataFrame) -> pd.DataFrame:
    """runs（平地・着順あり）から 1 走ごとの p, s, ax_basic。列: race_id, horse_id, bs_p, bs_s（積 ax_basic = BASIC_SCALE × p × s は running_profile.attach_cov が作る）。"""
    d = runs.copy()
    n = d.groupby("race_id")["horse_id"].transform("size")
    d["bs_p"] = np.where(n >= MIN_FIELD, ((n + 1) / 2 - d["finish_position"]) / ((n - 1) / 2).replace(0, np.nan), np.nan)

    r = d.drop_duplicates("race_id")[["race_id", "date", "venue", "surface", "distance", "grade"]].copy()
    win = d[(d["finish_position"] == 1) & (d["time_sec"] > 0)].drop_duplicates("race_id")[["race_id", "time_sec"]]
    r = r.merge(win, on="race_id", how="left")
    r["grade"] = r["grade"].fillna("")
    r["tpm"] = r["time_sec"] / r["distance"] * 200                      # 200m あたりの勝ち時計（秒）
    g = r.groupby(["venue", "surface", "distance", "grade"])["tpm"]
    r["dev"] = (r["tpm"] - g.transform("median")).where(g.transform("count") >= MIN_PAR_RACES)       # 正 = 遅い
    k = ["date", "venue", "surface"]
    own = r["dev"].notna().astype(int)
    tot = r.assign(_v=r["dev"].fillna(0), _c=own).groupby(k)[["_v", "_c"]].transform("sum")
    others = tot["_c"] - own
    r["day_loo"] = ((tot["_v"] - r["dev"].fillna(0)) / others.replace(0, np.nan)).where(others >= MIN_DAY_RACES)
    s = pd.Series(np.nan, index=r.index)
    for _, idx in r.groupby("surface").groups.items():
        x = r.loc[idx, "day_loo"]
        s.loc[idx] = -(x - x.mean()) / x.std()                          # 大 = 速い
    r["bs_s"] = s
    out = d[["race_id", "horse_id", "bs_p"]].merge(r[["race_id", "bs_s"]], on="race_id", how="left")
    return out.drop_duplicates(["race_id", "horse_id"]).reset_index(drop=True)


def load(years: Iterable[int], base: Path, store: Path, sig: dict) -> pd.DataFrame:
    """保存した 1 走ごとの p, s を読む。出走記録のファイルが変わっていたら作り直す（全体で数秒）。"""
    mf = store.with_suffix(".json")
    try:
        if store.exists() and mf.exists():
            meta = json.loads(mf.read_text(encoding="utf-8"))
            if meta.get("version") == STORE_VERSION and meta.get("source") == sig:
                return pd.read_parquet(store)
    except (OSError, ValueError):
        pass
    out = compute(_read(years, base))
    store.parent.mkdir(parents=True, exist_ok=True)
    tmp = store.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(store)
    mf.write_text(json.dumps({"version": STORE_VERSION, "source": sig, "rows": len(out)}, ensure_ascii=False), encoding="utf-8")
    return out
