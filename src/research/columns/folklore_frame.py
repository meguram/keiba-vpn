"""めぐコラム『通説シリーズ』の集計の土台: 出走ごとに、予想の時点で分かる条件の列をまとめた表を作る。

通説（docs/html/modeling/chapters/folklore_hypotheses.html の一覧）を 1 つずつ確かめるため、
前走との違い（距離・面・場・間隔・騎手）、前走の内容（着順・着差・上がり順位・4角位置・ペース）、
騎手・調教師の状況（その日の成績・コンビの実績・格）、最終追い切り（乗り手・コース・強さ）、
SmartRC の陣営情報（生産者・東西・ブリンカー）などを、出走 1 行ごとに持たせる。

すべて as-of（その出走より前の情報だけ）。今走の結果（着順・オッズ）は目的変数と比較の基準にだけ使う。

対象: 集計は 2020〜2025 年の JRA 平地（compare_series.load と同じ除外）。前走などの履歴は 2016 年から読む。

使い方:
    python -m src.research.columns.folklore_frame          # 作って data/local/research/folklore/frame.parquet に保存
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.columns.compare_series import FLAT_DIR, jump_mask

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "data/local/research/folklore"
FRAME = OUT_DIR / "frame.parquet"
TRAINING = REPO / "data/local/modeling/training_pipeline_v4/tp35/training_laps_fresh.parquet"
HIST_YEARS = range(2016, 2026)
YEARS = range(2020, 2026)

CLASS_RANK = {"新馬": 0, "未勝利": 0, "1勝": 1, "2勝": 2, "3勝": 3, "OP": 4, "L": 4, "G3": 5, "G2": 6, "G1": 7}
OVERSEAS_SKIP = set("札幌 函館 福島 新潟 東京 中山 中京 京都 阪神 小倉".split())
LOCAL_TRACKS = set("門別 札幌 盛岡 水沢 浦和 船橋 大井 川崎 金沢 笠松 名古屋 園田 姫路 高知 佐賀 帯広 ばんえい".split())


def _q(order: str, starters: int, which: int) -> float:
    """通過順の which 番目（0 = 最初のコーナー、-1 = 最後）を (位置−1)/(頭数−1) に直す。"""
    try:
        parts = [int(x) for x in str(order).split("-") if x.strip().isdigit()]
        if not parts or starters < 4:
            return np.nan
        return (parts[which] - 1) / (starters - 1)
    except Exception:
        return np.nan


def _pace(s) -> tuple[float, float]:
    try:
        d = json.loads(s) if isinstance(s, str) else (s or {})
        return float(d.get("first_half_3f")), float(d.get("second_half_3f"))
    except Exception:
        return np.nan, np.nan


def _margin_sec(df: pd.DataFrame) -> pd.Series:
    win = df.loc[df["finish_position"] == 1].groupby("race_id")["time_sec"].min()
    m = df["time_sec"] - df["race_id"].map(win)
    return m.where((df["time_sec"] > 0) & m.between(0, 10))         # 10 秒を超える差はタイムの取り違えとみなす


def history() -> pd.DataFrame:
    cols = ["race_id", "date", "venue", "surface", "distance", "direction", "grade", "race_class", "weather", "track_condition",
            "field_size", "race_name", "pace", "finish_position", "bracket_number", "horse_number", "horse_id", "sex_age",
            "jockey_weight", "jockey_name", "jockey_id", "time_sec", "passing_order", "last_3f", "odds", "popularity",
            "weight", "weight_change", "trainer_name", "trainer_id"]
    r = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=cols) for y in HIST_YEARS], ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"]) & ~jump_mask(r) & (r["odds"] > 0) & (r["popularity"] > 0)].copy()
    r = r.drop_duplicates(["race_id", "horse_id"])
    r["horse_id"] = r["horse_id"].astype(str)
    r["date"] = pd.to_datetime(r["date"])
    r["starters"] = r.groupby("race_id")["race_id"].transform("size")
    r["top3"] = r["finish_position"].between(1, 3).astype(float)
    r["win"] = (r["finish_position"] == 1).astype(float)
    r["q1"] = [_q(o, s, 0) for o, s in zip(r["passing_order"], r["starters"])]
    r["q4"] = [_q(o, s, -1) for o, s in zip(r["passing_order"], r["starters"])]
    r["l3f_rank"] = r.groupby("race_id")["last_3f"].rank(method="min")
    r["margin_sec"] = _margin_sec(r)
    fh = r.drop_duplicates("race_id").set_index("race_id")["pace"].map(_pace)
    r["pace_diff"] = r["race_id"].map(fh.map(lambda t: t[0] - t[1]))        # 前半3F − 後半3F（負 = 前が速い = ハイペース）
    # コースごとに前後半の差の普通は違う（短距離ダートは前が速いのが普通）ので、同じコースの平均との差にする
    r["pace_rel"] = r["pace_diff"] - r.groupby(r["venue"] + r["surface"] + r["distance"].astype(str))["pace_diff"].transform("mean")
    r["cls"] = r["grade"].map(CLASS_RANK)
    r["kai_day"] = r["race_id"].str[8:10].astype(int)                     # 開催の何日目
    r["race_no"] = r["race_id"].str[10:12].astype(int)
    r["sex"] = r["sex_age"].str[0]
    r["age"] = pd.to_numeric(r["sex_age"].str[1:], errors="coerce")
    r["gate_rel"] = (r["horse_number"] - 1) / (r["field_size"] - 1).clip(lower=1)
    r["course"] = r["venue"] + r["surface"] + r["distance"].astype(str)
    r["going_bad"] = r["track_condition"].isin(["重", "不良"])
    return r.drop(columns=["pace"])


def _cum_before(r: pd.DataFrame, keys: list[str], x: pd.Series) -> pd.Series:
    """keys ごとに、今の行より前（行の並び順）の x の合計。"""
    x = x.fillna(0).astype(float)
    return x.groupby([r[k] for k in keys]).cumsum() - x


def add_prev(r: pd.DataFrame) -> pd.DataFrame:
    r = r.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    g = r.groupby("horse_id")
    for c in ["date", "distance", "surface", "venue", "direction", "finish_position", "popularity", "margin_sec", "l3f_rank",
              "q1", "q4", "pace_diff", "pace_rel", "track_condition", "jockey_id", "jockey_name", "trainer_id", "jockey_weight", "cls",
              "race_name", "grade", "field_size", "weight", "course", "going_bad", "sex_age", "starters"]:
        r[f"p_{c}"] = g[c].shift()
    r["p2_finish_position"] = g["finish_position"].shift(2)
    r["p2_jockey_id"] = g["jockey_id"].shift(2)
    r["interval"] = (r["date"] - r["p_date"]).dt.days
    r["nstart"] = g.cumcount()                                              # JRA 平地の出走数（今走を含まない）
    # 休み明けから何戦目（間隔 70 日以上を休み明けとする。初出走は 0）
    layoff = (r["interval"] >= 70).astype(int)
    blk = layoff.groupby(r["horse_id"]).cumsum()
    r["since_layoff"] = np.where(blk > 0, r.groupby([r["horse_id"], blk]).cumcount() + 1, 0)
    r["dist_diff"] = r["distance"] - r["p_distance"]
    r["cls_diff"] = r["cls"] - r["p_cls"]
    r["weight_rel"] = r["jockey_weight"] - r["p_jockey_weight"]
    # 同コース・道悪の過去の 3 着内の回数（今走を含まない）
    r["course_top3_before"] = _cum_before(r, ["horse_id", "course"], r["top3"])
    r["course_runs_before"] = r.groupby(["horse_id", "course"]).cumcount()
    bad = r["top3"].where(r["going_bad"], 0.0)
    r["bad_top3_before"] = _cum_before(r, ["horse_id"], bad)
    r["bad_runs_before"] = _cum_before(r, ["horse_id"], r["going_bad"].astype(float))
    r["l3f_best_count"] = _cum_before(r, ["horse_id"], (r["l3f_rank"] == 1).astype(float))
    # 継続騎乗の回数（今走の騎手が直前から何走続けて乗っているか）
    same = (r["jockey_id"] == r["p_jockey_id"]).astype(int)
    run = same.groupby(((same == 0) | (r["horse_id"] != r["horse_id"].shift())).cumsum()).cumsum()
    r["jockey_streak"] = np.where(r["nstart"] > 0, run, 0)
    return r


def add_people(r: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の状況: 前年の勝利数の順位（格）、その日の前のレースまでの成績、その日の騎乗数、コンビの実績。"""
    r["year"] = r["date"].dt.year
    wins = r.groupby(["year", "jockey_id"])["win"].sum().reset_index()
    wins["jrank"] = wins.groupby("year")["win"].rank(ascending=False, method="min")
    wins["year"] += 1                                                      # 前年の順位を今年に付ける
    r = r.merge(wins[["year", "jockey_id", "jrank"]], on=["year", "jockey_id"], how="left")
    pw = wins.rename(columns={"jockey_id": "p_jockey_id", "jrank": "p_jrank"})[["year", "p_jockey_id", "p_jrank"]]
    r = r.merge(pw, on=["year", "p_jockey_id"], how="left")
    tw = r.groupby(["year", "trainer_id"])["win"].sum().reset_index()
    tw["trank"] = tw.groupby("year")["win"].rank(ascending=False, method="min")
    tw["year"] += 1
    r = r.merge(tw[["year", "trainer_id", "trank"]], on=["year", "trainer_id"], how="left")
    # その日の騎乗数と、そのレースより前の勝ち数
    day = r["date"].dt.strftime("%Y%m%d")
    r["jday_rides"] = r.groupby([day, "jockey_id"])["race_id"].transform("size")
    r = r.sort_values(["date", "race_id"]).reset_index(drop=True)
    key = day + r["jockey_id"].astype(str)
    r["_k"] = key
    r["jday_wins_before"] = _cum_before(r, ["_k"], r["win"])
    # 調教師 × 騎手のコンビの過去の騎乗数・3 着内（今走を含まない）
    pair = r["trainer_id"].astype(str) + "_" + r["jockey_id"].astype(str)
    r["pair_runs"] = r.groupby(pair).cumcount()
    r["_pair"] = pair
    r["pair_top3"] = _cum_before(r, ["_pair"], r["top3"])
    r["trainer_runs"] = r.groupby("trainer_id").cumcount()
    r["trainer_top3"] = _cum_before(r, ["trainer_id"], r["top3"])
    r["jockey_runs"] = r.groupby("jockey_id").cumcount()
    r["jockey_top3"] = _cum_before(r, ["jockey_id"], r["top3"])
    r = r.drop(columns=["_k", "_pair"])
    # 前走の騎手が今回は同じレースの別の馬に乗っている（主戦が他を選んだ）
    rj = r.groupby("race_id")["jockey_id"].apply(set).to_dict()
    r["prev_jockey_elsewhere"] = [(pj in rj[rid]) and (pj != j) for rid, pj, j in zip(r["race_id"], r["p_jockey_id"], r["jockey_id"])]
    # 同じ厩舎の出走頭数・その中の人気順
    r["stable_n"] = r.groupby(["race_id", "trainer_id"])["horse_id"].transform("size")
    r["stable_pop_rank"] = r.groupby(["race_id", "trainer_id"])["popularity"].rank(method="first")
    return r


def add_same_day_bias(r: pd.DataFrame) -> pd.DataFrame:
    """同じ日・同じ場・同じ面で、そのレースより前に終わったレースの 3 着内馬の、枠の位置・4角位置の平均（その日の偏り）。"""
    t3 = r[r["top3"] == 1].copy()
    t3["day"] = t3["date"].dt.strftime("%Y%m%d")
    agg = t3.groupby(["day", "venue", "surface", "race_no"]).agg(g=("gate_rel", "sum"), q=("q4", "sum"), n=("gate_rel", "size"),
                                                                  nq=("q4", "count")).reset_index().sort_values("race_no")
    for c in ["g", "q", "n", "nq"]:
        agg[f"c{c}"] = agg.groupby(["day", "venue", "surface"])[c].cumsum() - agg[c]    # そのレースより前の合計
    agg["day_gate"] = agg["cg"] / agg["cn"].replace(0, np.nan)
    agg["day_q4"] = agg["cq"] / agg["cnq"].replace(0, np.nan)
    agg["day_prior_races"] = agg["cn"] / 3
    r["day"] = r["date"].dt.strftime("%Y%m%d")
    return r.merge(agg[["day", "venue", "surface", "race_no", "day_gate", "day_q4", "day_prior_races"]], on=["day", "venue", "surface", "race_no"], how="left")


def _rider_is_jockey(rider: str, jockey: str) -> bool:
    """追い切りの乗り手（『鮫島駿』『ルメール』など略記）が、本番の騎手名（『鮫島克駿』）と同じ人か。"""
    if not isinstance(rider, str) or not isinstance(jockey, str):
        return False
    a = rider.replace("．", "").replace(".", "").strip()
    b = jockey.replace("．", "").replace(".", "").strip()
    if not a or a in ("助手", "師", "見習") or not b:
        return False
    if b.startswith(a):
        return True
    return len(a) >= 3 and b.startswith(a[:-1]) and a[-1] in b[len(a) - 1:]


def add_training(r: pd.DataFrame) -> pd.DataFrame:
    """最終追い切り（レースの前日までで最後のもの）と 1 週前、本数、乗り手が本番の騎手か。"""
    t = pd.read_parquet(TRAINING, columns=["horse_id", "train_date", "race_date", "kind", "center", "rider", "leg_color",
                                           "rank", "evaluation", "l1", "accel", "t4", "days_before"])
    t = t[(t["days_before"] >= 1) & (t["days_before"] <= 60)].copy()
    t["horse_id"] = t["horse_id"].astype(str)
    t["race_date"] = pd.to_datetime(t["race_date"])
    t = t.sort_values(["horse_id", "race_date", "days_before"])
    last = t.groupby(["horse_id", "race_date"]).first().reset_index()
    last = last.rename(columns={c: f"w_{c}" for c in ["kind", "center", "rider", "leg_color", "rank", "evaluation", "l1", "accel", "t4", "days_before"]})
    wk = t[t["days_before"].between(5, 11)].groupby(["horse_id", "race_date"])["rider"].apply(list).rename("w1_riders").reset_index()
    n30 = t[t["days_before"] <= 30].groupby(["horse_id", "race_date"]).agg(
        w_n30=("days_before", "size"), w_hard30=("leg_color", lambda s: s.isin(["一杯", "強め", "G強", "G一", "直強", "直一"]).sum())).reset_index()
    any_rider = t[t["days_before"] <= 30].groupby(["horse_id", "race_date"])["rider"].apply(list).rename("w30_riders").reset_index()
    r = r.merge(last.drop(columns=["train_date"]), left_on=["horse_id", "date"], right_on=["horse_id", "race_date"], how="left").drop(columns=["race_date"])
    for x in (wk, n30, any_rider):
        r = r.merge(x, left_on=["horse_id", "date"], right_on=["horse_id", "race_date"], how="left").drop(columns=["race_date"])
    r["w_last_by_jockey"] = [_rider_is_jockey(a, b) for a, b in zip(r["w_rider"], r["jockey_name"])]
    r["w_wk_by_jockey"] = [any(_rider_is_jockey(a, b) for a in (xs if isinstance(xs, list) else [])) for xs, b in zip(r["w1_riders"], r["jockey_name"])]
    r["w_any_by_jockey"] = [any(_rider_is_jockey(a, b) for a in (xs if isinstance(xs, list) else [])) for xs, b in zip(r["w30_riders"], r["jockey_name"])]
    r["w_has"] = r["w_kind"].notna()
    return r.drop(columns=["w1_riders", "w30_riders"])


def add_smartrc(r: pd.DataFrame) -> pd.DataFrame:
    cols = ["race_id", "hcode", "blinker", "tozai", "producer", "owner", "f_name", "mf_name", "f_llcode", "mf_llcode", "f_country", "mf_country"]
    s = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "smartrc_race_flat.parquet", columns=cols) for y in YEARS], ignore_index=True)
    s = s.rename(columns={"hcode": "horse_id"}).drop_duplicates(["race_id", "horse_id"])
    s["horse_id"] = s["horse_id"].astype(str)
    return r.merge(s, on=["race_id", "horse_id"], how="left")


_OVERSEAS_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\s+(\S+)")


def add_overseas(r: pd.DataFrame) -> pd.DataFrame:
    """出馬表の過去 5 走（race_shutuba_past）から、直前の 1 走が海外か、と海外の場名。"""
    rows = []
    for y in YEARS:
        p = pd.read_parquet(FLAT_DIR / str(y) / "race_shutuba_past_flat.parquet", columns=["race_id", "horse_id", "past_races"])
        for rid, hid, pr in zip(p["race_id"], p["horse_id"], p["past_races"]):
            try:
                races = json.loads(pr) if isinstance(pr, str) else list(pr or [])
            except Exception:
                continue
            if not races:
                continue
            raw = races[0].get("raw", "") if isinstance(races[0], dict) else ""
            m = _OVERSEAS_RE.match(raw)
            v = m.group(1) if m else ""
            ov = bool(v) and v not in OVERSEAS_SKIP and v not in LOCAL_TRACKS and not v.isdigit()
            anyov = False
            for x in races:
                mm = _OVERSEAS_RE.match(x.get("raw", "") if isinstance(x, dict) else "")
                vv = mm.group(1) if mm else ""
                anyov |= bool(vv) and vv not in OVERSEAS_SKIP and vv not in LOCAL_TRACKS
            rows.append((rid, str(hid), ov, v if ov else "", v in LOCAL_TRACKS, anyov))
    o = pd.DataFrame(rows, columns=["race_id", "horse_id", "prev_overseas", "prev_overseas_venue", "prev_local", "past5_overseas"])
    return r.merge(o.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")


def expected_top3(r: pd.DataFrame) -> pd.Series:
    """オッズ・人気・頭数から見込まれる 3 着内の確率: 人気順（14 番人気以上はまとめる）× 同じ人気の中での単勝オッズの 5 分位
    × 頭数（〜8・9〜11・12〜14・15〜）の区分ごとの、2020〜2025 年の 3 着内率の実績。

    オッズの式（ロジスティック回帰）では人気順ごとに ±1.5pt ずれ、人気になりやすい条件が『人気以上に走る』ように見えた。
    区分の平均なら、人気・オッズ・頭数のどの向きにもずれが残らない。区分の出走が 200 未満なら人気 × 頭数の平均を使う。
    """
    pop = r["popularity"].clip(upper=14).astype(int)
    fb = pd.cut(r["starters"], [0, 8, 11, 14, 99], labels=False)
    oq = r.groupby(pop)["odds"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    key3 = pop.astype(str) + "_" + oq.astype(str) + "_" + fb.astype(str)
    key2 = pop.astype(str) + "_" + fb.astype(str)
    m3 = r.groupby(key3)["top3"].agg(["mean", "size"])
    m2 = r.groupby(key2)["top3"].mean()
    e = key3.map(m3["mean"]).where(key3.map(m3["size"]) >= 200, key2.map(m2))
    return e.astype(float)


def build() -> pd.DataFrame:
    import time
    t0 = time.time()

    def step(name: str) -> None:
        print(f"  {name}: {time.time() - t0:.0f} 秒", flush=True)

    r = history(); step("履歴")
    r = add_prev(r); step("前走")
    r = add_people(r); step("騎手・調教師")
    r = add_same_day_bias(r); step("当日の偏り")
    r = r[r["date"].dt.year.isin(list(YEARS))].copy()
    r = add_training(r); step("追い切り")
    r = add_smartrc(r); step("SmartRC")
    r = add_overseas(r); step("海外")
    q = 1.0 / r["odds"]
    r["implied"] = q / q.groupby(r["race_id"]).transform("sum")
    r["exp_top3"] = expected_top3(r)
    r["ret"] = np.where(r["finish_position"] == 1, r["odds"], 0.0)
    return r


def load_frame() -> pd.DataFrame:
    return pd.read_parquet(FRAME)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r = build()
    for c in r.columns:
        if r[c].dtype == object:
            vals = set(r[c].dropna().unique().tolist()) if r[c].dropna().map(type).isin([bool, np.bool_]).all() else None
            if vals is not None and vals <= {True, False}:
                r[c] = r[c].astype("boolean")            # 真偽（欠測あり）は真偽のまま
            else:
                r[c] = r[c].astype(str).where(r[c].notna(), None)
    r.to_parquet(FRAME, index=False)
    print(f"{FRAME}: {len(r):,} 行 × {r.shape[1]} 列")


if __name__ == "__main__":
    main()
