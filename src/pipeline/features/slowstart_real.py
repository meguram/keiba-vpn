"""出遅れ率（実ラベル版）の特徴量。ラベルは戦績ページの備考にある『出遅』（horse_race_remarks）。

実ラベルの入手には癖がある。戦績ページを取れたのは「直近 3 年に走った馬」だけなので、
  * 取れた馬の戦績は全期間のラベルが揃う（引退が早い馬は 2020〜2022 年の走でもラベルが無い）。
  * 取れた馬に偏る（2023-10 以降に走った馬）ので、2020〜2022 年の走の『ラベルがあるか』自体が将来（その後も走るか）の情報になる。
そのため、ラベルが無い走の特徴量は全て欠損にし、学習・検証はラベルのある走に限る（slowreal_eval）。キーは horse_id / jockey_id / trainer_id / 種牡馬 ID。
全て『その走の日付より厳密に前』のラベルだけから作る（merge_asof の allow_exact_matches=False。累積和は『自分を引く』形にしない）。
市場オッズ・人気は使わない。

ラベルの時代差: 備考の『出遅』の付く割合は年とともに下がる（2022 年 17% → 2025 年 9% → 2026 年 5%。付け方の変化か記入の遅れ）。
生の率はこの差をそのまま持つので、各走のラベルから『その走の月より前 3 か月の全体の率』を引いた超過（slow_exc）で作った群 SX_* も作り、
生の率の群 SS_* と比べる（両方作って良い方を採る）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import factory_slowstart as fs

LABEL = "slow_rem"
LABEL_EXC = "slow_exc"
MIN_BASE_EVENTS = 1500           # 月ごとの全体の率を作るのに要る直近 3 か月のラベル件数
P0 = 0.025                       # 備考に『出遅』が付く走の割合（全体）。as-of の全体の率の出発点

GROUP_ORDER = ["SS_horse", "SS_horse_ctx", "SS_gate", "SS_jockey", "SS_trainer", "SS_sire", "SS_misc", "SS_damage", "SS_rel",
               "SX_horse", "SX_horse_ctx", "SX_gate", "SX_jockey", "SX_trainer", "SX_sire", "SX_misc", "SX_rel"]
REL_BASE = ["ss_h_rate", "ss_h_r10", "ss_g_tert_rate", "ss_g_num_rate", "ss_jk_rate", "ss_tr_rate"]


def parse_remarks(h: str, history: list[dict]) -> list[dict]:
    """horse_race_remarks の race_history → 1 走 1 行（フラグだけ。備考の生テキストは持ち出さない）。"""
    out = []
    for r in history or []:
        rm = str(r.get("remarks") or "")
        out.append({"horse_id": h, "race_id": str(r.get("race_id") or ""), "date": r.get("date"), "start_index": r.get("start_index"),
                    "slow_rem": float("出遅" in rm), "has_remarks": float(len(rm) > 0),
                    "trouble_rem": float(any(w in rm for w in ("不利", "落馬", "競走中止", "躓", "挟まれ")))})
    return out


def row_labels(frame: pd.DataFrame, rem: pd.DataFrame) -> pd.Series:
    """frame の各走（race_id, horse_id）にラベル（0/1）を付ける。戦績ページが無い馬・戦績に出ていない走は欠損（＝不明）。"""
    r = rem.drop_duplicates(["horse_id", "race_id"])[["horse_id", "race_id", LABEL]]
    m = frame[["horse_id", "race_id"]].merge(r, on=["horse_id", "race_id"], how="left")
    return pd.Series(m[LABEL].to_numpy(dtype=float), index=frame.index, name=LABEL)


def event_table(frame: pd.DataFrame, rem: pd.DataFrame) -> pd.DataFrame:
    """馬×日付のラベル表（as-of の履歴）。frame にある走は frame の日付、2020 年より前の走は備考側の日付を使う。"""
    known = set(zip(frame["horse_id"], frame["race_id"]))
    r = rem.copy()
    r["date"] = pd.to_datetime(r["date"].astype(str).str.replace("/", "-"), errors="coerce")
    inframe = pd.Series([(h, x) in known for h, x in zip(r["horse_id"], r["race_id"])], index=r.index)
    d = frame[["horse_id", "race_id", "date"]].merge(rem.drop_duplicates(["horse_id", "race_id"])[["horse_id", "race_id", LABEL]], on=["horse_id", "race_id"], how="inner")
    old = r[~inframe & r["date"].notna() & (r["date"] < frame["date"].min())][["horse_id", "date", LABEL]]
    return pd.concat([d[["horse_id", "date", LABEL]], old], ignore_index=True).drop_duplicates(["horse_id", "date"], keep="last")


def _drop_group_counts(g: pd.DataFrame) -> pd.DataFrame:
    """馬以外の集団の件数列は落とす（取れた馬が時代で違うので、件数の水準が年で違い、年を当てる手がかりになる）。"""
    keep = [c for c in g.columns if not (c.endswith("_n") and c != "ss_h_n")]
    return g[keep]


def relative_features(frame: pd.DataFrame, groups: dict[str, pd.DataFrame], prefix: str = "ss") -> pd.DataFrame:
    """同じレースの出走馬の中での相対（差と順位）。出遅れは、自分の率より『メンバーの中での位置』で効くことがある。"""
    src = pd.concat([g for g in groups.values()], axis=1)
    cols = [c for c in (b.replace("ss_", prefix + "_", 1) for b in REL_BASE) if c in src.columns]
    out = {}
    race = frame["race_id"].to_numpy()
    for c in cols:
        s = pd.Series(src[c].to_numpy(dtype=float), index=frame.index)
        g = s.groupby(race)
        out[f"{prefix}_rel_{c}_diff"] = (s - g.transform("mean")).to_numpy()
        out[f"{prefix}_rel_{c}_rank"] = g.rank(ascending=False, method="average", pct=True).to_numpy()
    k = pd.Series(src[f"{prefix}_h_rate"].to_numpy(dtype=float), index=frame.index)
    out[f"{prefix}_rel_field_sum_h"] = k.groupby(race).transform("sum").to_numpy()
    return pd.DataFrame(out, index=frame.index)


def monthly_base(events: pd.DataFrame, dates: pd.Series, window_months: int = 3, min_events: int | None = None) -> np.ndarray:
    """各日付について『その日の月より前の window_months か月』の全体の出遅れ率（その月の初日より前のラベルだけ。月内の他の走は使わない）。
    件数が min_events に満たない月は、それ以前の全期間の率（無ければ P0）で代える。"""
    min_events = MIN_BASE_EVENTS if min_events is None else min_events
    e = events.dropna(subset=[LABEL])
    per = e["date"].dt.to_period("M")
    g = e.groupby(per)[LABEL].agg(["sum", "size"]).sort_index()
    allm = pd.period_range(g.index.min() if len(g) else pd.Period("2018-01", "M"), pd.Period("2027-12", "M"), freq="M")
    g = g.reindex(allm).fillna(0.0)
    s3 = g["sum"].rolling(window_months, min_periods=1).sum().shift(1)
    n3 = g["size"].rolling(window_months, min_periods=1).sum().shift(1)
    sa = g["sum"].cumsum().shift(1)
    na = g["size"].cumsum().shift(1)
    rate = (s3 / n3).where(n3 >= min_events, (sa / na).where(na >= min_events, P0))
    return rate.reindex(dates.dt.to_period("M")).to_numpy(dtype=float)


def _rename(g: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    return g.rename(columns={c: c.replace(old, new, 1) for c in g.columns})


def build(frame: pd.DataFrame, events: pd.DataFrame, parents: pd.DataFrame | None, p0: float = P0) -> dict[str, pd.DataFrame]:
    """frame: prepare 済み（LABEL 列＝その走のラベル。欠損＝不明）。events: 馬×日付×LABEL。戻り値: 群名 → 特徴量表（frame と同じ index）。
    SS_* は生の率、SX_* は時代の出遅れ率を引いた超過の率。ラベルが不明の走の特徴量は全て欠損にする。"""
    groups = fs.build_groups(frame, events, LABEL, parents, p0=p0)
    groups["SS_damage"] = _damage(frame)
    groups = {k: _drop_group_counts(v) for k, v in groups.items()}
    groups["SS_rel"] = relative_features(frame, groups, "ss")
    # 時代差を引いた超過の率
    fr2 = frame.copy()
    ev2 = events.copy()
    fr2[LABEL_EXC] = fr2[LABEL] - monthly_base(events, fr2["date"])
    ev2[LABEL_EXC] = ev2[LABEL] - monthly_base(events, ev2["date"])
    gx = fs.build_groups(fr2, ev2, LABEL_EXC, parents, p0=0.0)
    gx = {_rename_group(k): _rename(_drop_group_counts(v), "ss_", "sx_").drop(columns=["sx_h_n"], errors="ignore") for k, v in gx.items()}      # 馬の走数は SS_horse と同じ
    gx["SX_rel"] = relative_features(frame, gx, "sx")
    groups.update(gx)
    unknown = frame[LABEL].isna().to_numpy()
    for k, g in groups.items():
        g = g.copy()
        g.loc[unknown, :] = np.nan
        groups[k] = g
    return groups


def _rename_group(k: str) -> str:
    return k.replace("SS_", "SX_", 1)


def _damage(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame[["horse_id", "date", "finish_position", "field_size"]].copy()
    f[LABEL] = frame[LABEL].to_numpy()
    f["fin_pct"] = (f["finish_position"].astype(float) - 1) / (f["field_size"].astype(float) - 1).clip(lower=1)
    f.loc[~(f["finish_position"].astype(float) > 0), "fin_pct"] = np.nan
    return fs.damage(f, "fin_pct", LABEL)
