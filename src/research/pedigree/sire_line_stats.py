"""父／母方（母父・母母父・母母母父）の系統・種牡馬ごとの、実データの成績表（2026-11-13 ユーザー依頼）。

集計の対象（画面で切り替え）:
  all   … 全馬の全レース（2016〜2026 の平地。着順あり・5 頭以上）
  top_h … 2 勝クラス以上（2勝・3勝・OP・L・重賞）を 1 回でも走った馬の全レース
  top_r … 2 勝クラス以上のレースだけ
位置: 父・母父・母母父・母母母父（5 代血統の (1,0)・(2,2)・(3,6)・(4,14)。系統の補完は sire_dam_influence.pedigree_lines と同じ）。
粒度: 小系統・大系統（smartrc）・種牡馬。群に載せる条件は、その対象で 30 頭以上（種牡馬はさらに 100 走以上）。
指標（METRICS）: 基本（勝率・複勝率・回収率・人気以上に走るか）、賞金・クラス（本賞金の概算）、位置取り、上がり 3F、距離、芝ダ・馬場、
  枠順、年齢・間隔、回り・コース、馬体重、走法軸（2023〜、2 勝クラス以上の走だけ。対象によらず同じ走。TS上限は 3 走以上の馬の値を 1 頭 1 回）。
  率は「その条件の走だけの平均」（例: 芝の複勝率 = 芝の走の複勝率）。各値には、その値に使った走数（頭数の指標は頭数）を添える。
頭数は、そのレースで着順の付いた頭数（取消・除外を除く）。休み明けの間隔は、絞り込む前の全出走（障害・中止を含む）で計算する。
1 頭あたりの累計本賞金・クラスへの到達は 2016〜2026 の期間の中だけ（期間の前後で経歴が切れる）。
賞金: 手元のレース結果に賞金の列が無いので、src/pipeline/megu_index/career_prize.py の PRIZE_TABLE（クラス × 着順 1〜5 着の
  本賞金の概算、万円）で走ごとに推定する。重賞はレースごとの差が大きく、実際の額とは違う。
使い方: python -m src.research.pedigree.sire_line_stats
  手元のデータだけを読む（GCS・GCP に触れない）。
出力: data/local/pedigree/sire_line_stats.json（build_sire_line_hub_html.py の「系統・種牡馬の成績表」タブの材料）
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")   # 他のセッションの学習と CPU を分け合う

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.pipeline.megu_index.career_prize import PRIZE_TABLE  # noqa: E402
from src.research.pedigree import sire_dam_influence as SD  # noqa: E402

OUT = SD.REPO / "data/local/pedigree/sire_line_stats.json"
POPS = {"all": "全馬の全レース", "top_h": "2勝クラス以上を走った馬の全レース", "top_r": "2勝クラス以上のレースだけ"}
GRANS = {"小": "小系統", "大": "大系統", "id": "種牡馬"}
MIN_HORSES, MIN_RUNS_SIRE = 30, 100
DBANDS = [(0, 1400, "〜1400"), (1401, 1800, "1500〜1800"), (1801, 2200, "1900〜2200"), (2201, 9999, "2300〜")]
REST_DAYS = 70

# (キー, 表示名, まとまり, 書式, 単位) 書式: pct = 割合、num = 数、man = 万円、sec = 秒、z = 標準化した値、kg
METRICS: list[tuple[str, str, str, str]] = [
    ("win", "勝率", "基本", "pct"), ("top2", "連対率", "基本", "pct"), ("top3", "複勝率", "基本", "pct"),
    ("tan_ret", "単勝回収率", "基本", "pct"), ("fuku_ret", "複勝回収率", "基本", "pct"), ("pct", "着順率（1=1着）", "基本", "num"),
    ("over_pop", "複勝率 − 人気からの見込み", "基本", "pct"),
    ("prize", "1走あたり本賞金（概算）", "賞金・クラス", "man"), ("h_prize", "1頭あたり累計本賞金（概算）", "賞金・クラス", "man"),
    ("h_2win", "2勝クラス出走馬の割合", "賞金・クラス", "pct"), ("h_op", "OP 出走馬の割合", "賞金・クラス", "pct"),
    ("h_grade", "重賞出走馬の割合", "賞金・クラス", "pct"), ("h_gwin", "重賞勝ち馬の割合", "賞金・クラス", "pct"),
    ("pos1", "初角の位置（0=先頭）", "位置取り", "num"), ("nige", "逃げ率（初角1番手）", "位置取り", "pct"),
    ("front3", "前1/3の率（初角）", "位置取り", "pct"), ("back3", "後ろ1/3の率（初角）", "位置取り", "pct"),
    ("pos4", "最終角の位置（0=先頭）", "位置取り", "num"), ("push", "押し上げ（初角−最終角）", "位置取り", "num"),
    ("l3_rank", "上がりの順位（0=最速）", "上がり3F", "num"), ("l3_best", "上がり最速の率", "上がり3F", "pct"),
    ("l3_top3", "上がり3位以内の率", "上がり3F", "pct"), ("l3_diff", "上がり − レース平均（秒）", "上がり3F", "sec"),
    ("l3_top3_fuku", "上がり3位以内の走の複勝率", "上がり3F", "pct"),
    ("dist", "平均の距離（m）", "距離", "num"),
    *[(f"share_d{i}", f"{lab} の出走割合", "距離", "pct") for i, (_, _, lab) in enumerate(DBANDS)],
    *[(f"top3_d{i}", f"{lab} の複勝率", "距離", "pct") for i, (_, _, lab) in enumerate(DBANDS)],
    *[(f"top3_{sf}_d{i}", f"{sfn}{lab} の複勝率", "距離", "pct") for sf, sfn in (("t", "芝"), ("d", "ダ")) for i, (_, _, lab) in enumerate(DBANDS)],
    ("turf_share", "芝の出走割合", "芝ダ・馬場", "pct"), ("top3_t", "芝の複勝率", "芝ダ・馬場", "pct"), ("top3_d", "ダートの複勝率", "芝ダ・馬場", "pct"),
    *[(f"top3_{sf}_{c}", f"{sfn}・{cn} の複勝率", "芝ダ・馬場", "pct") for sf, sfn in (("t", "芝"), ("d", "ダ"))
      for c, cn in (("g", "良"), ("y", "稍重"), ("h", "重・不良"))],
    *[(f"top3_w{w}", f"{w}枠 の複勝率", "枠順", "pct") for w in range(1, 9)],
    ("top3_in", "内枠（1〜3）の複勝率", "枠順", "pct"), ("top3_mid", "中（4〜5）の複勝率", "枠順", "pct"),
    ("top3_out", "外枠（6〜8）の複勝率", "枠順", "pct"), ("in_minus_out", "内枠 − 外枠", "枠順", "pct"),
    *[(f"top3_a{a}", f"{a}{'歳以上' if a == 5 else '歳'} の複勝率", "年齢・間隔", "pct") for a in (2, 3, 4, 5)],
    ("top3_rest", "休み明け（10週以上）の複勝率", "年齢・間隔", "pct"), ("top3_norest", "間隔 10 週未満の複勝率", "年齢・間隔", "pct"),
    ("top3_R", "右回りの複勝率", "回り・コース", "pct"), ("top3_L", "左回りの複勝率", "回り・コース", "pct"),
    ("top3_steep", "急坂（中山・阪神・中京）の複勝率", "回り・コース", "pct"), ("top3_flat", "平坦な会場の複勝率", "回り・コース", "pct"),
    ("weight", "平均の馬体重（kg）", "馬体重", "num"), ("top3_wup", "体重 +10kg 以上の複勝率", "馬体重", "pct"),
    ("top3_wdown", "体重 −10kg 以下の複勝率", "馬体重", "pct"),
    *[(f"ax_{v}", v, "走法軸", "z") for v in list(SD.AXIS_NAMES.values()) + ["TS上限"]],
]
GROUPS = ["基本", "賞金・クラス", "位置取り", "上がり3F", "距離", "芝ダ・馬場", "枠順", "年齢・間隔", "回り・コース", "馬体重", "走法軸"]
HORSE_METRICS = {"h_prize", "h_2win", "h_op", "h_grade", "h_gwin"}


# ---------------------------------------------------------------- 走ごとの値
def fuku_payouts(payoff: pd.Series) -> dict[tuple[str, int], float]:
    """レースごとの配当の JSON から、(race_id, 馬番) → 複勝の払い戻し（100 円あたり）"""
    out = {}
    for rid, js in payoff.items():
        try:
            d = json.loads(js) if isinstance(js, str) else (js or {})
        except Exception:  # noqa: BLE001
            continue
        f = d.get("複勝") or []
        for e in f if isinstance(f, list) else [f]:
            try:
                out[(rid, int(str(e.get("numbers", "")).strip()))] = float(str(e.get("payout", "")).replace(",", ""))
            except Exception:  # noqa: BLE001
                continue
    return out


def prize_of(grade: str, pos: int) -> float:
    t = PRIZE_TABLE.get("OP" if grade == "L" else grade)
    return float(t.get(int(pos), 0.0)) if t else 0.0


def prev_gap_days(raw: pd.DataFrame) -> pd.Series:
    """馬ごとの前走からの日数。絞り込む前の全出走（障害・少頭数・中止を含む。取消・除外は除く）で計算する。
    取消・除外の判定: finish_status が scratched / excluded、または状態が無く着順 -1 で通過順が空の行（dnf は中止で、走ったとみなす）。
    index は (race_id, horse_id)。期間の最初の走は NaN"""
    cols = ["race_id", "horse_id", "date", "finish_position"] + [c for c in ("passing_order", "finish_status") if c in raw]
    a = raw[cols].copy()
    a["horse_id"] = a.horse_id.astype(str)
    a["race_id"] = a.race_id.astype(str)
    pos = pd.to_numeric(a.finish_position, errors="coerce")
    st = a["finish_status"].astype(str) if "finish_status" in a else pd.Series("nan", index=a.index)
    po = a["passing_order"].fillna("").astype(str).str.strip() if "passing_order" in a else pd.Series("", index=a.index)
    # 取消・除外も中止も着順は -1。状態の列があればそれで分け、無い年（2020〜2023 など）は「通過順が空」を取消・除外とみなす
    started = (pos >= 1) | ((pos == -1) & (st.isin(["dnf"]) | (~st.isin(["scratched", "excluded"]) & (po != ""))))
    a = a[started]
    a["d"] = pd.to_datetime(a.date, errors="coerce")
    a = a.dropna(subset=["d"]).sort_values(["horse_id", "d"]).drop_duplicates(["race_id", "horse_id"])
    gap = a.groupby("horse_id").d.diff().dt.days
    return pd.Series(gap.to_numpy(), index=pd.MultiIndex.from_arrays([a.race_id, a.horse_id]))


def run_table(raw: pd.DataFrame) -> pd.DataFrame:
    """平地・着順あり・完走 5 頭以上の走に、指標の元になる列を付ける（条件に当たらない走は NaN）。
    頭数は、そのレースで着順の付いた頭数（field_size は取消・除外の行も数えるため使わない）"""
    gaps = prev_gap_days(raw)
    r = raw.copy()
    r["horse_id"] = r.horse_id.astype(str)
    r["race_id"] = r.race_id.astype(str)
    r = r[r.surface.isin(["芝", "ダート"]) & r.grade.isin(SD.GRADES)]
    r["pos"] = pd.to_numeric(r.finish_position, errors="coerce")
    r["field"] = r.pos.ge(1).groupby(r.race_id).transform("sum")
    r = r[(r.pos >= 1) & (r.field >= 5)].copy()
    n1 = r.field - 1
    r["win"] = (r.pos == 1).astype(float)
    r["top2"] = (r.pos <= 2).astype(float)
    r["top3"] = (r.pos <= 3).astype(float)
    r["pct"] = 1 - (r.pos - 1) / n1
    odds = pd.to_numeric(r.odds, errors="coerce")
    r["tan_ret"] = np.where(odds.notna(), r.win * odds, np.nan)
    fp = fuku_payouts(r.drop_duplicates("race_id").set_index("race_id").payoff)
    has_pay = {rid for rid, _ in fp}   # 複勝の配当が読めたレース
    hn = pd.to_numeric(r.horse_number, errors="coerce")
    pay = [fp.get((a, int(b))) if b == b else None for a, b in zip(r.race_id, hn)]
    # 配当の載っていない馬は外れ（7 頭以下は 2 着までしか払わないので、その 3 着も 0）。配当が読めないレースだけ NaN
    r["fuku_ret"] = [(p / 100 if p else 0.0) if rid in has_pay else np.nan for rid, p in zip(r.race_id, pay)]
    # 人気からの複勝の見込み（人気 × 頭数の帯の、全体の複勝率）
    popu = pd.to_numeric(r.popularity, errors="coerce")
    fb = pd.cut(r.field, [0, 10, 14, 99], labels=False)
    key = popu.astype("Int64").astype(str) + "_" + fb.astype(str)
    r["over_pop"] = r.top3 - r.top3.groupby(key).transform("mean")
    r.loc[popu.isna(), "over_pop"] = np.nan
    r["prize"] = [prize_of(g, p) for g, p in zip(r.grade, r.pos)]
    # 位置取り（通過順。初角と最終角）
    po = r.passing_order.astype(str).str.split("-")
    first = pd.to_numeric(po.str[0], errors="coerce")
    last = pd.to_numeric(po.str[-1], errors="coerce")
    r["pos1"] = (first - 1) / n1
    r["pos4"] = (last - 1) / n1
    r["push"] = r.pos1 - r.pos4
    r["nige"] = np.where(first.notna(), (first == 1).astype(float), np.nan)
    r["front3"] = np.where(first.notna(), (r.pos1 <= 1 / 3).astype(float), np.nan)
    r["back3"] = np.where(first.notna(), (r.pos1 >= 2 / 3).astype(float), np.nan)
    # 上がり 3F（全頭の中の順位）
    l3 = pd.to_numeric(r.last_3f, errors="coerce")
    rk = l3.groupby(r.race_id).rank(method="min")
    r["l3_rank"] = (rk - 1) / n1
    r["l3_best"] = np.where(l3.notna(), (rk == 1).astype(float), np.nan)
    r["l3_top3"] = np.where(l3.notna(), (rk <= 3).astype(float), np.nan)
    r["l3_diff"] = l3 - l3.groupby(r.race_id).transform("mean")
    r["l3_top3_fuku"] = np.where(rk <= 3, r.top3, np.nan)
    # 距離・芝ダ・馬場
    r["dist"] = r.distance.astype(float)
    turf = r.surface == "芝"
    r["turf_share"] = turf.astype(float)
    r["top3_t"] = np.where(turf, r.top3, np.nan)
    r["top3_d"] = np.where(~turf, r.top3, np.nan)
    for i, (lo, hi, _) in enumerate(DBANDS):
        m = r.distance.between(lo, hi)
        r[f"share_d{i}"] = m.astype(float)
        r[f"top3_d{i}"] = np.where(m, r.top3, np.nan)
        r[f"top3_t_d{i}"] = np.where(m & turf, r.top3, np.nan)
        r[f"top3_d_d{i}"] = np.where(m & ~turf, r.top3, np.nan)
    cond = r.track_condition.map({"良": "g", "稍重": "y", "重": "h", "不良": "h"})
    for sf, msf in (("t", turf), ("d", ~turf)):
        for c in ("g", "y", "h"):
            r[f"top3_{sf}_{c}"] = np.where(msf & (cond == c), r.top3, np.nan)
    # 枠順
    br = pd.to_numeric(r.bracket_number, errors="coerce")
    for w in range(1, 9):
        r[f"top3_w{w}"] = np.where(br == w, r.top3, np.nan)
    r["top3_in"] = np.where(br.between(1, 3), r.top3, np.nan)
    r["top3_mid"] = np.where(br.between(4, 5), r.top3, np.nan)
    r["top3_out"] = np.where(br.between(6, 8), r.top3, np.nan)
    # 年齢・間隔
    age = pd.to_numeric(r.sex_age.astype(str).str[1:], errors="coerce")
    for a in (2, 3, 4, 5):
        r[f"top3_a{a}"] = np.where((age >= 5) if a == 5 else (age == a), r.top3, np.nan)
    gap = pd.Series(gaps.reindex(pd.MultiIndex.from_arrays([r.race_id, r.horse_id])).to_numpy(), index=r.index)
    r["top3_rest"] = np.where(gap >= REST_DAYS, r.top3, np.nan)
    r["top3_norest"] = np.where(gap < REST_DAYS, r.top3, np.nan)
    # 回り・コース
    r["top3_R"] = np.where(r.direction == "右", r.top3, np.nan)
    r["top3_L"] = np.where(r.direction == "左", r.top3, np.nan)
    r["top3_steep"] = np.where(r.venue.isin(SD.STEEP), r.top3, np.nan)
    r["top3_flat"] = np.where(r.venue.isin(SD.FLATV), r.top3, np.nan)
    # 馬体重
    wt = pd.to_numeric(r.weight, errors="coerce")
    wc = pd.to_numeric(r.weight_change, errors="coerce")
    r["weight"] = wt.where(wt > 300)
    r["top3_wup"] = np.where(wc >= 10, r.top3, np.nan)
    r["top3_wdown"] = np.where(wc <= -10, r.top3, np.nan)
    return r


def axis_runs(axes: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """走法軸を、2 勝クラス以上の走だけで 芝ダ × 距離帯 × 会場 × クラス の中で標準化した走ごとの値（sire_dam_influence と同じ）"""
    ax = axes.copy()
    ax["horse_id"] = ax.horse_id.astype(str)
    ax["race_id"] = ax.race_id.astype(str)
    if "abn" in ax:
        ax = ax[ax.abn != True]  # noqa: E712
    ax = ax.merge(runs[["race_id", "horse_id", "grade"]], on=["race_id", "horse_id"], how="inner")
    ax = ax[ax.grade.isin(SD.TOP)].copy()
    ax["dband"] = pd.cut(ax.distance, [0, 1400, 1800, 2200, 4000], labels=False)
    key = [ax.surface, ax.dband, ax.venue, ax.grade]
    out = ax[["race_id", "horse_id"]].copy()
    for c, v in SD.AXIS_NAMES.items():
        g = ax.groupby(key)[c]
        out[f"ax_{v}"] = ((ax[c] - g.transform("mean")) / g.transform("std")).replace([np.inf, -np.inf], np.nan)
    # TS上限: 馬ごとの上位 2 走の平均（2 勝クラス以上の走が 3 走以上の馬だけ。sire_dam_influence と同じ）。
    # 群の平均が 1 頭 1 回になるよう、その馬の最初の走にだけ付ける（走数 = 頭数）
    g = out.groupby("horse_id")["ax_TS"]
    top2 = g.apply(lambda s: s.nlargest(2).mean())[g.count() >= 3]
    out["ax_TS上限"] = np.where(~out.horse_id.duplicated(), out.horse_id.map(top2), np.nan)
    return out


# ---------------------------------------------------------------- 集計
def horse_level(r: pd.DataFrame) -> pd.DataFrame:
    g = r.groupby("horse_id")
    return pd.DataFrame({
        "h_prize": g.prize.sum(),
        "h_2win": g.grade.apply(lambda s: float(s.isin(SD.TOP).any())),
        "h_op": g.grade.apply(lambda s: float(s.isin({"OP", "L", "G3", "G2", "G1"}).any())),
        "h_grade": g.grade.apply(lambda s: float(s.isin({"G3", "G2", "G1"}).any())),
        "h_gwin": r.assign(gw=r.grade.isin({"G3", "G2", "G1"}) & (r.pos == 1)).groupby("horse_id").gw.max().astype(float),
    })


def aggregate(r: pd.DataFrame, key: pd.Series, hl: pd.DataFrame, min_horses: int, min_runs: int) -> pd.DataFrame:
    """key（走ごとの群の名前）で、走の指標の平均と、馬の指標の平均。値の列 m と、その走数・頭数の列 n_m"""
    run_cols = [m for m, *_ in METRICS if m not in HORSE_METRICS and m in r.columns]
    k = key.loc[r.index]
    ok = k.notna()
    rr, kk = r.loc[ok, run_cols], k[ok]
    g = rr.groupby(kk.to_numpy())
    mean, cnt = g.mean(), g.count()
    horses = r.loc[ok].groupby(kk.to_numpy()).horse_id.nunique()
    runs = g.size()
    hk = pd.Series(kk.to_numpy(), index=r.loc[ok, "horse_id"].to_numpy())
    hk = hk[~hk.index.duplicated()]
    hm = hl.loc[hk.index].groupby(hk.to_numpy())
    out = mean.join(hm.mean()).join(cnt.add_prefix("n_")).join(hm.count().add_prefix("n_"))
    out["in_minus_out"] = out.top3_in - out.top3_out
    out["n_in_minus_out"] = np.minimum(out.n_top3_in, out.n_top3_out)
    out["horses"], out["runs"] = horses, runs
    return out[(out.horses >= min_horses) & (out.runs >= min_runs)]


def pack(df: pd.DataFrame, names: dict | None = None) -> list:
    """群ごとに [名前, 頭数, 走数, 値..., 走数（値ごと）...]（JSON を小さく）"""
    keys = [m for m, *_ in METRICS]
    rows = []
    for idx, row in df.sort_values("horses", ascending=False).iterrows():
        vals = [None if pd.isna(row.get(m)) else round(float(row[m]), 4) for m in keys]
        ns = [0 if pd.isna(row.get("n_" + m)) else int(row["n_" + m]) for m in keys]
        label = names.get(idx, idx) if names else idx
        rows.append([str(label), int(row.horses), int(row.runs), vals, ns])
    return rows


def populations(r: pd.DataFrame) -> dict[str, pd.Series]:
    """集計の対象 3 つの行の印。all = 全馬の全レース、top_h = 2 勝クラス以上を 1 回でも走った馬の全レース（下のクラスの走も含む）、
    top_r = 2 勝クラス以上のレースの走だけ（2勝・3勝・OP・L・重賞。SD.TOP）"""
    return {"all": pd.Series(True, index=r.index),
            "top_h": r.horse_id.isin(set(r.loc[r.grade.isin(SD.TOP), "horse_id"])),
            "top_r": r.grade.isin(SD.TOP)}


def run(out: Path = OUT) -> dict:
    want = ["race_id", "date", "horse_id", "surface", "distance", "direction", "track_condition", "venue", "grade", "finish_position",
            "field_size", "passing_order", "last_3f", "sex_age", "odds", "popularity", "bracket_number", "horse_number", "weight",
            "weight_change", "payoff", "finish_status"]
    parts = []
    for y in range(2016, 2027):
        f = SD.FLAT / f"{y}/race_result_flat.parquet"
        if f.exists():
            names = set(pq.read_schema(f).names)   # 年によって無い列がある（finish_status は 2020〜2023 に無い）
            parts.append(pd.read_parquet(f, columns=[c for c in want if c in names]))
    raw = pd.concat(parts, ignore_index=True)
    r = run_table(raw)
    print(f"走 {len(r)}・馬 {r.horse_id.nunique()}", file=sys.stderr)
    if SD.AXES.exists():
        r = r.merge(axis_runs(pd.read_parquet(SD.AXES), r), on=["race_id", "horse_id"], how="left")
    ped = SD.pedigree_lines(sorted(r.horse_id.unique()))
    pops = populations(r)
    res = {"built": datetime.now(SD.JST).strftime("%Y-%m-%d %H:%M"), "pops": POPS, "grans": GRANS, "positions": list(SD.POS),
           "metrics": [{"key": m, "label": lab, "group": grp, "fmt": f} for m, lab, grp, f in METRICS], "groups": GROUPS,
           "min_horses": MIN_HORSES, "min_runs_sire": MIN_RUNS_SIRE, "data": {}, "overall": {}, "n": {}}
    for pk, pm in pops.items():
        rp = r[pm]
        hl = horse_level(rp)
        res["n"][pk] = {"runs": int(len(rp)), "horses": int(rp.horse_id.nunique())}
        res["overall"][pk] = pack(aggregate(rp, pd.Series("全体", index=rp.index), hl, 0, 0))[0]
        res["data"][pk] = {}
        for p in SD.POS:
            res["data"][pk][p] = {}
            for gk in GRANS:
                col = f"{p}_{gk}" if gk != "id" else f"{p}_id"
                key = rp.horse_id.map(ped[col])
                names = None
                if gk == "id":
                    nm = ped[[f"{p}_id", f"{p}_name", f"{p}_小"]].dropna(subset=[f"{p}_id"]).drop_duplicates(f"{p}_id").set_index(f"{p}_id")
                    names = {i: f"{row[f'{p}_name'] or i}（{row[f'{p}_小'] or '系統不明'}）" for i, row in nm.iterrows()}
                agg = aggregate(rp, key, hl, MIN_HORSES, MIN_RUNS_SIRE if gk == "id" else 0)
                res["data"][pk][p][gk] = pack(agg, names)
            print(pk, p, {g: len(v) for g, v in res["data"][pk][p].items()}, file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"保存: {out}（{out.stat().st_size / 1e6:.1f} MB）", file=sys.stderr)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    run(ap.parse_args().out)


if __name__ == "__main__":
    main()
