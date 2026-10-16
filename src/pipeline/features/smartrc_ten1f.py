"""SmartRC のテン1F（スタート後 1F のタイム）から作る as-of の特徴量と、前段モデルの正解。

SmartRC の smartrc_race（GCS）は 1 レースにつき runners（出走馬と過去5走 h1_〜h5_）と fullresults（馬ごとの全戦績）を持つ。
fullresults は取得時点（2026 年）までの戦績を含むので、**特徴量には rdate < 当該レース日 の行だけ**を使う。
当該レース自身の行（rdate == 当該レース日）は正解（ターゲット）にだけ使い、特徴量には入れない。
結合キーは (race_id, horse_id)。SmartRC の hcode は netkeiba の horse_id（10 桁）と同じ。

単位: ten1f は 1/10 秒の文字列（'132' = 13.2 秒）。
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

TEN_MIN, TEN_MAX = 90, 200            # 9.0〜20.0 秒の範囲外は欠損（0 や ' ' を含む）
PREFIX = "sx_"

# fullresults から残す列（戦績 1 行）
FR_KEYS = ("rcode", "rdate", "place", "trackkind", "range", "ten1f", "corner1", "corner2", "corner3", "corner4",
           "entry", "uno", "rank", "stime", "furlong3", "f3rank", "interval")
# runners から残す列（発走前の値）
RUNNER_KEYS = ("ten1f_best", "ten1f_best_rank", "ten_has", "ten_has_rank", "ten_pat", "agari_has", "agari_has_rank", "agari_pat")
H_FIELDS = ("rcode", "ten1f", "ten1f_rank", "corner1", "range", "trackkind", "rank")


def parse_ten1f(v: Any) -> float | None:
    """'132' → 13.2（秒）。空・非数・範囲外は None。"""
    s = str(v).strip() if v is not None else ""
    if not s.isdigit():
        return None
    n = int(s)
    return n / 10.0 if TEN_MIN <= n <= TEN_MAX else None


def _int(v: Any) -> int | None:
    s = str(v).strip() if v is not None else ""
    return int(s) if s.isdigit() else None


# ── 抽出（smartrc_race 1 レース → 行）──────────────────────────────────────────

def extract_race(d: dict, race_id: str) -> tuple[list[dict], list[dict]]:
    """(runner 行, fullresults 行) を返す。runner 行は発走前の値だけ。fullresults 行は馬×戦績（重複は呼び出し側で除く）。"""
    rcode = str(d.get("rcode") or "")
    date = rcode[:8]
    runners, careers = [], []
    for r in d.get("runners") or []:
        hid = str(r.get("hcode") or "").strip()
        if not hid:
            continue
        row = {"race_id": race_id, "horse_id": hid, "date": date}
        for k in RUNNER_KEYS:
            row[k] = r.get(k)
        for i in range(1, 6):
            for f in H_FIELDS:
                row[f"h{i}_{f}"] = r.get(f"h{i}_{f}")
        runners.append(row)
    for hid, rows in (d.get("fullresults") or {}).items():
        for x in rows or []:
            careers.append({"horse_id": str(hid).strip(), **{k: x.get(k) for k in FR_KEYS}})
    return runners, careers


def _first_corner(df: pd.DataFrame) -> pd.Series:
    """最初に記録されているコーナーの通過順（corner1〜4 のうち 0 でない最初の値）。

    1200m などは第1コーナーが無く corner1 が 0 で、通過順は corner3・4 から始まるため、corner1 だけでは約 57% が欠損になる。"""
    c = pd.concat([df[f"corner{i}"].map(_int) for i in (1, 2, 3, 4)], axis=1).astype("float64")
    return c.where(c > 0).bfill(axis=1).iloc[:, 0]


def career_table(rows: Iterable[dict]) -> pd.DataFrame:
    """fullresults 行を型付きの表にし、(horse_id, rcode) で重複を除く。c1 は『最初に記録されているコーナーの通過順』。"""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=["horse_id", "rcode", "rdate", "ten", "c1", "entry", "range", "trackkind"])
    out = pd.DataFrame({
        "horse_id": df["horse_id"].astype(str),
        "rcode": df["rcode"].astype(str),
        "rdate": df["rdate"].astype(str),
        "ten": df["ten1f"].map(parse_ten1f).astype("float64"),
        "c1": _first_corner(df),
        "entry": df["entry"].map(_int).astype("float64"),
        "range": df["range"].map(_int).astype("float64"),
        "trackkind": df["trackkind"].map(_int).astype("float64"),
    })
    out["c1_rel"] = out["c1"] / out["entry"].where(out["entry"] > 0)
    return out.drop_duplicates(["horse_id", "rcode"]).reset_index(drop=True)


# ── 正解（当該レース自身の行。特徴量には使わない）──────────────────────────────────

def race_targets(entries: pd.DataFrame, career: pd.DataFrame, min_cover: float = 0.8) -> pd.DataFrame:
    """entries(race_id, horse_id, date) → ten_actual / c1_rel_actual と、レース内で引いた ten_rel。

    date は 'YYYY-MM-DD' か 'YYYYMMDD'。そのレースの出走馬のうち ten が取れた馬の割合が min_cover 未満のレースは ten_rel を欠損にする。"""
    e = entries[["race_id", "horse_id", "date"]].copy()
    e["rdate"] = pd.to_datetime(e["date"]).dt.strftime("%Y%m%d")
    c = career[["horse_id", "rdate", "ten", "c1_rel"]].rename(columns={"ten": "ten_actual", "c1_rel": "c1_rel_actual"})
    out = e.merge(c, on=["horse_id", "rdate"], how="left").drop(columns=["rdate"])
    g = out.groupby("race_id")["ten_actual"]
    cover = g.transform(lambda s: s.notna().mean())
    mean = g.transform("mean")
    out["ten_rel_actual"] = (out["ten_actual"] - mean).where(cover >= min_cover)
    return out


# ── 特徴量（rdate < 当該レース日だけ）─────────────────────────────────────────────

def build_features(entries: pd.DataFrame, career: pd.DataFrame, runners: pd.DataFrame | None = None) -> pd.DataFrame:
    """entries(race_id, horse_id, date, distance, surface) の各行について、日付より**前**の戦績から sx_ 特徴量を作る。

    runners があれば、発走前の ten1f_best_rank / h?_ten1f_rank（過去5走のテン順位）も足す。"""
    e = entries[["race_id", "horse_id", "date", "distance", "surface"]].copy()
    e["_d"] = pd.to_datetime(e["date"]).dt.strftime("%Y%m%d")
    e["_i"] = np.arange(len(e))
    c = career[["horse_id", "rdate", "ten", "c1_rel", "range", "trackkind"]].dropna(subset=["rdate"])
    m = e[["_i", "horse_id", "_d", "distance", "surface"]].merge(c, on="horse_id", how="inner")
    m = m[m["rdate"] < m["_d"]]                                      # ← 先読み防止はここ（厳密に前）
    m = m.sort_values(["_i", "rdate"], ascending=[True, False])
    m["_k"] = m.groupby("_i").cumcount()                              # 0 = 直近
    tv = m[m["ten"].notna()].copy()
    tv["_kt"] = tv.groupby("_i").cumcount()                           # テンが取れた走だけの順位
    out = pd.DataFrame(index=e["_i"].to_numpy())
    g = tv.groupby("_i")["ten"]
    out[f"{PREFIX}ten_n"] = g.size()
    out[f"{PREFIX}ten_best"] = g.min()
    out[f"{PREFIX}ten_last"] = tv[tv["_kt"] == 0].set_index("_i")["ten"]
    for n in (3, 5):
        sel = tv[tv["_kt"] < n].groupby("_i")["ten"]
        out[f"{PREFIX}ten_mean{n}"] = sel.mean()
    out[f"{PREFIX}ten_min5"] = tv[tv["_kt"] < 5].groupby("_i")["ten"].min()
    out[f"{PREFIX}ten_sd5"] = tv[tv["_kt"] < 5].groupby("_i")["ten"].std()
    # 今回と似た条件（同じ芝/ダート・距離 ±200m）での平均
    surf_kind = tv["surface"].map(lambda s: 0.0 if str(s).startswith("芝") else (1.0 if str(s).startswith("ダ") else np.nan))
    sim = tv[(tv["trackkind"] == surf_kind) & ((tv["range"] - tv["distance"]).abs() <= 200)]
    out[f"{PREFIX}ten_sim_mean"] = sim.groupby("_i")["ten"].mean()
    out[f"{PREFIX}ten_sim_n"] = sim.groupby("_i").size()
    # 序盤の位置取り（corner1 / 頭数）
    cv = m[m["c1_rel"].notna()].copy()
    cv["_kc"] = cv.groupby("_i").cumcount()
    out[f"{PREFIX}c1_rel_mean3"] = cv[cv["_kc"] < 3].groupby("_i")["c1_rel"].mean()
    out[f"{PREFIX}c1_rel_last"] = cv[cv["_kc"] == 0].set_index("_i")["c1_rel"]
    out[f"{PREFIX}n_prior"] = m.groupby("_i").size()
    out = out.reindex(np.arange(len(e)))
    out[f"{PREFIX}ten_n"] = out[f"{PREFIX}ten_n"].fillna(0)
    out[f"{PREFIX}n_prior"] = out[f"{PREFIX}n_prior"].fillna(0)
    out[f"{PREFIX}ten_sim_n"] = out[f"{PREFIX}ten_sim_n"].fillna(0)
    out["race_id"] = e["race_id"].to_numpy()
    out["horse_id"] = e["horse_id"].to_numpy()

    # 発走前のテン順位（runners）
    if runners is not None and len(runners):
        r = runners.drop_duplicates(["race_id", "horse_id"]).copy()
        hr = [pd.to_numeric(r[f"h{i}_ten1f_rank"].astype(str).str.strip(), errors="coerce") for i in range(1, 6)]
        r[f"{PREFIX}tenrank_mean5"] = pd.concat(hr, axis=1).mean(axis=1)
        r[f"{PREFIX}tenbest_rank"] = pd.to_numeric(r["ten1f_best_rank"].astype(str).str.strip(), errors="coerce")
        out = out.merge(r[["race_id", "horse_id", f"{PREFIX}tenrank_mean5", f"{PREFIX}tenbest_rank"]], on=["race_id", "horse_id"], how="left")
    return add_race_relative(out)


def add_race_relative(f: pd.DataFrame) -> pd.DataFrame:
    """同じレースの出走馬どうしで比べた値（ペース圧の代理）。基準は mean3 → 無ければ mean5 → last。"""
    base = f[f"{PREFIX}ten_mean3"].fillna(f[f"{PREFIX}ten_mean5"]).fillna(f[f"{PREFIX}ten_last"])
    g = base.groupby(f["race_id"])
    mn, mu, sd = g.transform("min"), g.transform("mean"), g.transform("std")
    n = g.transform("count")
    ok = n >= 3
    f[f"{PREFIX}rel_gap_fast"] = (base - mn).where(ok)
    f[f"{PREFIX}rel_z"] = ((base - mu) / sd.replace(0, np.nan)).where(ok)
    f[f"{PREFIX}rel_rank_pct"] = g.rank(method="min", pct=True).where(ok)
    f[f"{PREFIX}race_ten_mean"] = mu.where(ok)
    f[f"{PREFIX}race_ten_sd"] = sd.where(ok)
    near = ((base - mn) <= 0.3).astype("float64").where(base.notna())
    f[f"{PREFIX}race_n_fast03"] = near.groupby(f["race_id"]).transform("sum").where(ok)
    c1 = f[f"{PREFIX}c1_rel_mean3"]
    front = (c1 <= 0.25).astype("float64").where(c1.notna())
    f[f"{PREFIX}race_n_front"] = front.groupby(f["race_id"]).transform("sum").where(c1.groupby(f["race_id"]).transform("count") >= 3)
    f[f"{PREFIX}race_ten_known"] = (n / f.groupby("race_id")["horse_id"].transform("size")).astype("float64")
    return f


CORE = [f"{PREFIX}{s}" for s in ("ten_n", "ten_best", "ten_last", "ten_mean3", "ten_mean5", "ten_min5", "ten_sd5", "ten_sim_mean", "ten_sim_n",
                                  "c1_rel_mean3", "c1_rel_last", "n_prior")]
RELATIVE = [f"{PREFIX}{s}" for s in ("rel_gap_fast", "rel_z", "rel_rank_pct", "race_ten_mean", "race_ten_sd", "race_n_fast03", "race_n_front", "race_ten_known")]
RUNNER_RANKS = [f"{PREFIX}tenrank_mean5", f"{PREFIX}tenbest_rank"]
# SmartRC が事前に計算して持つ要約（定義が公開されておらず、as-of かどうかを検証できないので、主評価には入れない）
UNVERIFIED = ["ten_has", "ten_has_rank", "ten_pat"]
