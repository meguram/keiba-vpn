"""開発の読み込み（2016〜2024 年だけ）と、as-of の集計の部品。

- `load_dev()` は 2025-01-01 より前だけを返す（固定）。2025・2026 年は `load_final()` だけが読み、台帳 eval_only を通す。
- `asof()` は「そのレースの日より厳密に前・その馬自身の過去走を引いた」鍵ごとの和と件数を返す。
費用 0 円（手元の parquet だけ）。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("PEDLAB_OUT", "/home/jovyan/work/pedigree_lab_out"))
DEV_END = pd.Timestamp("2024-12-31")
FAMILY = "pedigree_lab"


def _guard(period: tuple[str, str], purpose: str, version: str, reason: str, frozen_paths=()):
    from src.research.pedigree_lab import pl_ledger      # 台帳の入口は pl_ledger だけ（test_data_ledger の登録）
    return pl_ledger.open_token(period, purpose, version, reason, frozen_paths)


def _prep(r: pd.DataFrame) -> pd.DataFrame:
    r = r[r["finish_position"] >= 1].copy()                    # 取消・除外（-1）を落とす
    r["horse_id"] = r["horse_id"].astype(str)
    r = r.sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True)
    # 取り込みの誤り: 同じ日に同じ馬が 2 レースに出ている（中身の同じレースの写し。例 202408060108）。後のレースを丸ごと落とす
    dup = r.duplicated(["horse_id", "date"], keep="first")
    r = r[~r["race_id"].isin(r.loc[dup, "race_id"].unique())].reset_index(drop=True)
    n =r.groupby("race_id")["race_id"].transform("size")
    r["n_run"] = n
    r["win"] = (r["finish_position"] == 1).astype(np.int8)
    r["top3"] = (r["finish_position"] <= 3).astype(np.int8)
    r["ex_win"] = r["win"] - 1.0 / n                            # 頭数で割った期待との差
    r["ex_top3"] = r["top3"] - np.minimum(3, n) / n
    r["fpct"] = (r["finish_position"] - 1) / (n - 1).clip(lower=1)   # 0=1 着、1=最下位
    return r


TOKENS: list = []      # 台帳の Token（道具の終わりに finish_all() で閉じる）


def finish_all(result_path: str | None = None) -> None:
    while TOKENS:
        try:
            TOKENS.pop().finish(result_path=result_path)
        except Exception as exc:  # 記録の失敗で研究を止めない
            print(f"[台帳] finish に失敗: {exc}")


def load_dev(reason: str = "段 1・2 の開発（2016〜2024 の開催の無作為の分け）", version: str = "dev") -> pd.DataFrame:
    TOKENS.append(_guard(("2016-01-01", "2024-12-31"), "select", version, reason))
    r = pd.read_parquet(OUT / "runs.parquet", filters=[("date", "<=", DEV_END)])     # 2025 年以降は読み込みの時点で切る
    assert r["date"].max() <= DEV_END
    return _prep(r)


def load_final(version: str, reason: str) -> pd.DataFrame:
    """最後の 1 回（2025〜2026）。as-of の集計のため 2016 年からの全期間を返す。"""
    TOKENS.append(_guard(("2025-01-01", "2026-11-23"), "eval_only", version, reason))
    return _prep(pd.read_parquet(OUT / "runs.parquet"))


def load_ped() -> pd.DataFrame:
    p = pd.read_parquet(OUT / "ped.parquet")
    p["horse_id"] = p["horse_id"].astype(str)
    return p


def split_meetings(meetings: pd.Series, seed: int = 20261125, fr=(0.70, 0.15, 0.15)) -> pd.Series:
    """開催（race_id の先頭 8 桁）単位の無作為の分け。0=学習 1=検証 2=保留。"""
    u = np.array(sorted(pd.unique(meetings)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    lab = np.zeros(len(u), dtype=np.int8)
    a, b = int(len(u) * fr[0]), int(len(u) * (fr[0] + fr[1]))
    lab[perm[a:b]] = 1
    lab[perm[b:]] = 2
    return meetings.map(dict(zip(u, lab))).astype(np.int8)


def asof(df: pd.DataFrame, keys: list[str], vals: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """鍵ごとに「その日より厳密に前」の和（列ごと）と件数。その馬自身の、同じ鍵での過去走は引く。

    df は date・horse_id と keys・vals を持つ。鍵に欠け（空・NaN）がある行は件数 0。
    返す: (sums [len(df), len(vals)], counts [len(df)])
    """
    k = df[keys].copy()
    bad = np.zeros(len(df), dtype=bool)
    for c in keys:
        s = k[c]
        bad |= s.isna().to_numpy() | (s.astype(str) == "").to_numpy()
    kid = pd.MultiIndex.from_frame(k.astype(str)).factorize()[0] if len(keys) > 1 else pd.factorize(k[keys[0]].astype(str))[0]
    d = df["date"].to_numpy("datetime64[D]").astype(np.int64)
    V = df[vals].to_numpy(float)
    one = np.ones((len(df), 1))
    VV = np.hstack([V, one])
    # 鍵 × 日 で合計 → 鍵の中で日の順に累積（その日を含まない）
    g = pd.DataFrame(VV, columns=[*vals, "_n"])
    g["kid"], g["d"] = kid, d
    day = g.groupby(["kid", "d"], sort=True).sum()
    cum = day.groupby(level=0).cumsum() - day
    idx = pd.MultiIndex.from_arrays([kid, d])
    tot = cum.reindex(idx).to_numpy()
    # 自分の過去走（同じ鍵・前の日）。1 日に 1 走なので、日の順の累積から自分を引く
    hid = pd.factorize(df["horse_id"])[0]
    # 自分の引きは「同じ鍵・同じ馬で 1 日 1 行」が前提。重なると今回の値を余分に引く（第 5 版 E の漏れ）。重なりは止める
    dup = pd.DataFrame({"k": kid, "h": hid, "d": d})[~bad].duplicated().any()
    if dup:
        raise ValueError("asof: 同じ鍵・同じ馬・同じ日の行が複数ある（自分の引きが今回の値を含む）")
    h = pd.DataFrame(VV, columns=[*vals, "_n"])
    h["kid"], h["hid"], h["d"] = kid, hid, d
    h = h.sort_values(["kid", "hid", "d"], kind="stable")
    own = h.groupby(["kid", "hid"])[[*vals, "_n"]].cumsum() - h[[*vals, "_n"]]
    own = own.sort_index().to_numpy()
    res = tot - own
    res[bad] = 0.0
    return res[:, :-1], res[:, -1]
