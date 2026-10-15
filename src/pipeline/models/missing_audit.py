"""学習表の特徴量ごとの欠損を、原因別に分類する（tp-01b 欠損監査）。

欠損している (race_id, horse_number) × 特徴量 の 1 セルごとに、次のどれかを付ける（上から順に最初に当てはまったもの）。

  正当           … その特徴量の定義上、値がありえない（初出走で過去走なし、開催初日で前日傾向なし、障害レース 等）
  上流の欠損     … 元になる別の特徴量が欠損しているため（例: 差分の片側が欠損）
  生データ未取得 … そのレース自体の生データ（race_index など）が無い
  生データ欠け   … レースの生データはあるが、その馬の行が無い
  生データの値が空 … 馬の行はあるが、該当項目が空／0（データ元が値を出していない）
  加工で欠損     … 生データに値があるのに、特徴量が欠損している（加工・結合・型変換のエラー疑い）
  要確認         … 上のどれにも分類できない（根拠不足）

「正当」は定義に基づく判断で、**ルールが実データと整合しているか**（そのルールに当てはまる行が本当に欠損しているか）を
``validate_legit_rules`` で併せて確認する。整合していないルールは正当の根拠にならない。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Iterable

import polars as pl

LEGIT = "正当"
UPSTREAM = "上流の欠損"
RAW_RACE = "生データ未取得"
RAW_HORSE = "生データ欠け"
RAW_EMPTY = "生データの値が空"
PROC = "加工で欠損"
UNKNOWN = "要確認"
CAUSES = [LEGIT, UPSTREAM, RAW_RACE, RAW_HORSE, RAW_EMPTY, PROC, UNKNOWN]
KEYS = ["race_id", "horse_number"]
INDEX_COLS = ["speed_max", "speed_avg", "speed_distance", "speed_course"]


@dataclass(frozen=True)
class Step:
    cause: str
    reason: str
    cond: pl.Expr


def _chain(steps: list[Step], default: tuple[str, str] = (UNKNOWN, "どの条件にも当てはまらない")) -> tuple[pl.Expr, pl.Expr]:
    """(cause 式, reason 式)。steps の先頭から最初に当てはまったものを採用する。"""
    def build(pick):
        e = None
        for s in steps:
            e = pl.when(s.cond).then(pl.lit(pick(s))) if e is None else e.when(s.cond).then(pl.lit(pick(s)))
        d = pl.lit(default[0] if pick is _cause else default[1])
        return d if e is None else e.otherwise(d)
    return build(_cause), build(_reason)


def _cause(s: Step) -> str:
    return s.cause


def _reason(s: Step) -> str:
    return s.reason


# ── 生データの存在（キー単位）────────────────────────────────────────────────

def surface_group_expr(col: str = "surface") -> pl.Expr:
    s = pl.col(col).cast(pl.Utf8).fill_null("")
    return pl.when(s.str.contains("障")).then(pl.lit("obstacle")).when(s.str.contains("芝")).then(pl.lit("turf")).when(s.str.contains("ダ")).then(pl.lit("dirt")).otherwise(pl.lit("unknown"))


def _scan_years(tables_dir: Path, table: str, years: Iterable[int | str], cols: list[str]) -> pl.LazyFrame:
    frames = []
    for y in years:
        p = tables_dir / str(y) / f"{table}_flat.parquet"
        if p.exists():
            have = pl.scan_parquet(p).collect_schema().names()
            frames.append(pl.scan_parquet(p).select([c for c in cols if c in have]))
    if not frames:
        return pl.LazyFrame({c: [] for c in cols})
    return pl.concat(frames, how="diagonal_relaxed")


def raw_presence(tables_dir: str | Path, years: Iterable[int | str]) -> dict[str, pl.DataFrame]:
    """生データの flat テーブルから、キー単位の「行がある／値がある」を作る。"""
    td = Path(tables_dir)
    out: dict[str, pl.DataFrame] = {}

    ix = _scan_years(td, "race_index", years, ["race_id", "horse_number", *INDEX_COLS, "speed_recent"]).with_columns(
        pl.col("horse_number").cast(pl.Int64, strict=False))
    nums = pl.col("speed_recent").cast(pl.Utf8).str.extract_all(r"-?\d+(?:\.\d+)?").list.eval(pl.element().cast(pl.Float64, strict=False))
    ix = ix.with_columns(
        *[pl.col(c).is_not_null().alias(f"ix_nn_{c}") for c in INDEX_COLS],
        *[(nums.list.get(k - 1, null_on_oob=True).fill_null(0) > 0).alias(f"ix_recent_nn_{k}") for k in (1, 2, 3)],
    ).select(KEYS + [f"ix_nn_{c}" for c in INDEX_COLS] + [f"ix_recent_nn_{k}" for k in (1, 2, 3)])
    ixd = ix.group_by(KEYS).agg(pl.all().any()).with_columns(ix_row=pl.lit(True)).collect()
    out["index"] = ixd
    out["index_races"] = ixd.select("race_id").unique().with_columns(ix_race=pl.lit(True))

    pa = _scan_years(td, "race_shutuba_past", years, ["race_id", "horse_number", "past_races", "training"]).with_columns(
        pl.col("horse_number").cast(pl.Int64, strict=False))
    pa = pa.with_columns(
        (pl.col("past_races").cast(pl.Utf8).fill_null("").str.strip_chars().is_in(["", "[]", "null", "None"]).not_()).alias("pa_nn_past_races"),
        (pl.col("training").cast(pl.Utf8).fill_null("").str.strip_chars().is_in(["", "[]", "null", "None"]).not_()).alias("pa_nn_training"),
    ).select(KEYS + ["pa_nn_past_races", "pa_nn_training"])
    pad = pa.group_by(KEYS).agg(pl.all().any()).with_columns(pa_row=pl.lit(True)).collect()
    out["past"] = pad
    out["past_races"] = pad.select("race_id").unique().with_columns(pa_race=pl.lit(True))

    rr = _scan_years(td, "race_result", years, ["race_id", "horse_number", "finish_position", "last_3f"]).with_columns(
        pl.col("horse_number").cast(pl.Int64, strict=False))
    out["winner_l3f"] = (rr.filter(pl.col("finish_position") == 1).group_by("race_id")
                         .agg(pl.col("last_3f").is_not_null().any().alias("winner_l3f_nn")).collect())
    return out


# ── 馬場傾向の前提（日・場・路面ごと）────────────────────────────────────────

def track_bias_context(df: pl.DataFrame, winner_l3f: pl.DataFrame) -> pl.DataFrame:
    """race_id ごとに: 当日その前に同じ(日,場,路面)のレースが何件あり、勝ち馬の上り3Fが何件取れているか／前日(暦日)はどうか。"""
    races = (df.select("race_id", "date", "venue_code", "surface", "start_time").unique(subset=["race_id"])
             .with_columns(pl.col("date").cast(pl.Utf8).str.to_date(strict=False).alias("_d"),
                           pl.col("venue_code").cast(pl.Utf8), surface_group_expr().alias("_sg"),
                           pl.col("start_time").cast(pl.Utf8).alias("_st"))
             .join(winner_l3f, on="race_id", how="left")
             .with_columns(pl.col("winner_l3f_nn").fill_null(False).cast(pl.Int64).alias("_l3f")))
    races = races.sort(["_d", "venue_code", "_sg", "_st", "race_id"])
    g = ["_d", "venue_code", "_sg"]
    races = races.with_columns(
        pl.int_range(pl.len()).over(g).alias("tb_prior_races"),
        (pl.col("_l3f").cum_sum().over(g) - pl.col("_l3f")).alias("tb_prior_l3f_nn"),
    )
    day = races.group_by(g).agg(pl.len().alias("_n"), pl.col("_l3f").sum().alias("_nn"))
    prev = day.with_columns((pl.col("_d") + pl.duration(days=1)).alias("_d")).rename({"_n": "tb_prev_races", "_nn": "tb_prev_l3f_nn"})
    return races.join(prev, on=g, how="left").with_columns(
        pl.col("tb_prev_races").fill_null(0), pl.col("tb_prev_l3f_nn").fill_null(0),
    ).select("race_id", "tb_prior_races", "tb_prior_l3f_nn", "tb_prev_races", "tb_prev_l3f_nn")


# ── 特徴量ごとのルール ────────────────────────────────────────────────────────

def build_context(df: pl.DataFrame, presence: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """学習表に、生データの存在フラグと馬場傾向の前提を結合する。"""
    ctx = df.with_columns(pl.col("horse_number").cast(pl.Int64))
    ctx = (ctx.join(presence["index"], on=KEYS, how="left").join(presence["index_races"], on="race_id", how="left")
           .join(presence["past"], on=KEYS, how="left").join(presence["past_races"], on="race_id", how="left")
           .join(track_bias_context(ctx, presence["winner_l3f"]), on="race_id", how="left"))
    flags = [c for c in ctx.columns if c.startswith(("ix_", "pa_"))]
    return ctx.with_columns([pl.col(c).fill_null(False) for c in flags])


def _prior_rule(prefix: str, who: str) -> list[Step]:
    n = pl.col(f"{prefix}_prior_races")
    return [Step(LEGIT, f"{who}の過去実績なし（prior_races=0）", n == 0),
            Step(PROC, "過去実績があるのに率が欠損（履歴統計の加工）", n > 0)]


def feature_rules(ctx_cols: set[str]) -> dict[str, list[Step]]:
    c = pl.col
    rules: dict[str, list[Step]] = {}

    debut = c("horse_prior_races") == 0
    for f in INDEX_COLS:
        # 初出走でも指数がある馬が多い（実測で約 3 割）ため、初出走は『正当』にしない。生データの有無を先に見る。
        rules[f] = [
            Step(RAW_RACE, "そのレースの race_index が無い", ~c("ix_race")),
            Step(RAW_HORSE, "race_index にこの馬の行が無い（初出走）", ~c("ix_row") & debut),
            Step(RAW_HORSE, "race_index にこの馬の行が無い（過去走あり）", ~c("ix_row")),
            Step(RAW_EMPTY, "行はあるが値が空（初出走）", ~c(f"ix_nn_{f}") & debut),
            Step(RAW_EMPTY, "行はあるが値が空（過去走あり）", ~c(f"ix_nn_{f}")),
            Step(PROC, "生データに値があるのに特徴量が欠損", c(f"ix_nn_{f}")),
        ]
    for k in (1, 2, 3):
        rules[f"speed_recent_{k}"] = [
            Step(RAW_RACE, "そのレースの race_index が無い", ~c("ix_race")),
            Step(RAW_HORSE, "race_index にこの馬の行が無い（初出走）", ~c("ix_row") & debut),
            Step(RAW_HORSE, "race_index にこの馬の行が無い（過去走あり）", ~c("ix_row")),
            Step(RAW_EMPTY, f"行はあるが {k} 走前の指数が 0/空（過去走が {k} 走未満）", ~c(f"ix_recent_nn_{k}") & (c("horse_prior_races") < k)),
            Step(RAW_EMPTY, f"行はあるが {k} 走前の指数が 0/空（過去走は十分）", ~c(f"ix_recent_nn_{k}")),
            Step(PROC, "生データに値があるのに特徴量が欠損", c(f"ix_recent_nn_{k}")),
        ]
    rules["past_races"] = [
        Step(LEGIT, "初出走（過去走なし）", c("horse_prior_races") == 0),
        Step(RAW_RACE, "そのレースの shutuba_past が無い", ~c("pa_race")),
        Step(RAW_HORSE, "shutuba_past にこの馬の行が無い", ~c("pa_row")),
        Step(RAW_EMPTY, "行はあるが past_races が空", ~c("pa_nn_past_races")),
        Step(PROC, "生データに値があるのに特徴量が欠損", c("pa_nn_past_races")),
    ]
    rules["training"] = [
        Step(RAW_RACE, "そのレースの shutuba_past が無い", ~c("pa_race")),
        Step(RAW_HORSE, "shutuba_past にこの馬の行が無い", ~c("pa_row")),
        Step(RAW_EMPTY, "行はあるが調教が空", ~c("pa_nn_training")),
        Step(PROC, "生データに値があるのに特徴量が欠損", c("pa_nn_training")),
    ]
    for p, who in (("jockey", "騎手"), ("trainer", "調教師")):
        for suf in ("win_rate", "top3_rate"):
            rules[f"{p}_prior_{suf}"] = _prior_rule(p, who)
    for suf in ("win_rate", "top3_rate", "avg_finish"):
        rules[f"horse_prior_{suf}"] = _prior_rule("horse", "馬")

    sdp, pdv = "track_bias_winner_3f_same_day_prior", "track_bias_winner_3f_prev_day"
    rules[sdp] = [
        Step(LEGIT, "その日その場その路面で時刻が最初のレース（直前の勝ち馬がいない）", c("tb_prior_races") == 0),
        Step(RAW_EMPTY, "直前のレースはあるが、勝ち馬の上り3F が生データに無い", c("tb_prior_l3f_nn") == 0),
        Step(PROC, "時刻順では直前のレースがあるのに欠損（実装の並びが 馬場×距離帯ブロック順 のため）", c("tb_prior_l3f_nn") > 0),
    ]
    rules[pdv] = [
        Step(LEGIT, "前日(暦日)に同じ場・路面の開催なし（開催初日・土曜など）", c("tb_prev_races") == 0),
        Step(RAW_EMPTY, "前日の開催はあるが、勝ち馬の上り3F が生データに無い", c("tb_prev_l3f_nn") == 0),
        Step(PROC, "前日の勝ち馬の上り3F はあるのに欠損", c("tb_prev_l3f_nn") > 0),
    ]
    both = c(sdp).is_null() & c(pdv).is_null()
    rules["track_bias_winner_3f_diff"] = [
        Step(UPSTREAM, "当日傾向または前日傾向が欠損", c(sdp).is_null() | c(pdv).is_null()),
        Step(PROC, "両方あるのに差が欠損", pl.lit(True)),
    ]
    rules["track_bias_winner_3f_weighted"] = [
        Step(UPSTREAM, "当日傾向と前日傾向の両方が欠損", both),
        Step(PROC, "どちらかはあるのに合成値が欠損", pl.lit(True)),
    ]
    for ped in ("sire", "dam_sire"):
        rules[f"{ped}_x_track_bias_diff"] = [
            Step(UPSTREAM, "馬場傾向の差が欠損", c("track_bias_winner_3f_diff").is_null()),
            Step(UPSTREAM, "血統の路面別勝率が欠損", c(f"{ped}_prior_win_rate_surface").is_null()),
            Step(PROC, "元は揃っているのに交互作用が欠損", pl.lit(True)),
        ]
        rules[f"{ped}_x_track_bias_weighted"] = [
            Step(UPSTREAM, "馬場傾向の合成値が欠損", c("track_bias_winner_3f_weighted").is_null()),
            Step(UPSTREAM, "血統の路面別勝率が欠損", c(f"{ped}_prior_win_rate_surface").is_null()),
            Step(PROC, "元は揃っているのに交互作用が欠損", pl.lit(True)),
        ]
        rules[f"{ped}_prior_win_rate_surface"] = [
            Step(UNKNOWN, "血統名が無い／初出の血統の可能性（要確認）", pl.lit(True)),
        ]
    return {k: v for k, v in rules.items() if k in ctx_cols}


# ── 履歴特徴量（tp-02）: 必要な過去走数を実データから推定して判定 ─────────────────

def infer_min_history(df: pl.DataFrame, feature: str, count_col: str, *, max_n: int = 12, tol: float = 0.02) -> int | None:
    """「count_col が n 以上の行では欠損率が tol 以下」になる最小の n。見つからなければ None（過去走数だけでは説明できない）。"""
    for n in range(0, max_n + 1):
        sub = df.filter(pl.col(count_col) >= n)
        if sub.height == 0:
            return None
        if sub[feature].is_null().mean() <= tol:
            return n
    return None


def history_feature_names(df: pl.DataFrame, exclude: set[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and df[c].null_count() > 0]


def classify_history(df: pl.DataFrame, features: list[str], *, count_col: str = "d_nprior", l3f_col: str = "e_n_l3f") -> tuple[pl.DataFrame, pl.DataFrame]:
    """履歴特徴量の欠損を分類する。戻り値: (原因別件数, 推定した必要過去走数の表)。"""
    rows, thr = [], []
    for f in features:
        k = infer_min_history(df, f, count_col)
        kl = None
        if k is not None and l3f_col in df.columns:
            sub = df.filter(pl.col(count_col) >= k)
            kl = infer_min_history(sub, f, l3f_col)
        thr.append({"feature": f, "必要過去走数(推定)": k, "必要な上り3F有効走数(推定)": kl})
        nul = df.filter(pl.col(f).is_null())
        if nul.height == 0:
            continue
        steps = []
        if k is not None:
            steps.append(Step(LEGIT, f"過去走が {k} 走未満（推定）", pl.col(count_col) < k))
        if kl is not None and kl > 0:
            steps.append(Step(RAW_EMPTY, f"過去走の上り3F有効数が {kl} 未満（推定）。生データ側の欠け", pl.col(l3f_col) < kl))
        cause, reason = _chain(steps, (UNKNOWN, "過去走は足りているのに欠損（条件付き特徴量か加工エラー）"))
        res = nul.select(cause.alias("原因"), reason.alias("理由")).group_by("原因", "理由").len().rename({"len": "件数"})
        rows.append(res.with_columns(pl.lit(f).alias("feature"), pl.lit(df.height).alias("全行数")))
    summary = pl.concat(rows) if rows else pl.DataFrame({"原因": [], "理由": [], "件数": [], "feature": [], "全行数": []})
    return summary, pl.DataFrame(thr)


# ── 実行 ──────────────────────────────────────────────────────────────────────

def classify(ctx: pl.DataFrame, rules: dict[str, list[Step]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """戻り値: (原因別件数 [feature, 原因, 理由, 件数, 全行数], 要調査の行 [feature, race_id, horse_number, 原因, 理由])。"""
    summaries, suspects = [], []
    for f, steps in rules.items():
        nul = ctx.filter(pl.col(f).is_null())
        if nul.height == 0:
            continue
        cause, reason = _chain(steps)
        lab = nul.select("race_id", "horse_number", cause.alias("原因"), reason.alias("理由"))
        summaries.append(lab.group_by("原因", "理由").len().rename({"len": "件数"}).with_columns(pl.lit(f).alias("feature"), pl.lit(ctx.height).alias("全行数")))
        suspects.append(lab.filter(pl.col("原因").is_in([PROC, UNKNOWN])).with_columns(pl.lit(f).alias("feature")))
    s = pl.concat(summaries) if summaries else pl.DataFrame()
    d = pl.concat(suspects) if suspects else pl.DataFrame()
    return s, d


def validate_legit_rules(ctx: pl.DataFrame, rules: dict[str, list[Step]]) -> pl.DataFrame:
    """「正当」ルールの整合: そのルールに当てはまる行のうち、実際に欠損している割合（低ければルールが実データと合っていない）。"""
    out = []
    for f, steps in rules.items():
        prev: pl.Expr | None = None
        for s in steps:
            if s.cause == LEGIT:
                m = ctx.filter(s.cond if prev is None else (s.cond & ~prev))
                out.append({"feature": f, "ルール": s.reason, "該当行": m.height,
                            "実際に欠損": int(m[f].is_null().sum()) if m.height else 0})
            prev = s.cond if prev is None else (prev | s.cond)
    r = pl.DataFrame(out) if out else pl.DataFrame({"feature": [], "ルール": [], "該当行": [], "実際に欠損": []})
    return r.with_columns((pl.col("実際に欠損") / pl.col("該当行").clip(lower_bound=1)).alias("欠損率"))


def row_accounting(tables_dir: str | Path, years: Iterable[int | str], train_keys: pl.DataFrame) -> pl.DataFrame:
    """出馬表の行のうち学習表に入らなかったものを理由別に数える。"""
    td = Path(tables_dir)
    sb = _scan_years(td, "race_shutuba", years, ["race_id", "horse_number", "venue_code"]).with_columns(
        pl.col("horse_number").cast(pl.Int64, strict=False)).collect()
    sb = sb.filter(pl.col("venue_code").cast(pl.Utf8).str.to_integer(strict=False).is_between(1, 10)).unique(subset=KEYS)
    rr = _scan_years(td, "race_result", years, ["race_id", "horse_number", "finish_position"]).with_columns(
        pl.col("horse_number").cast(pl.Int64, strict=False)).collect()
    res_races = rr.select("race_id").unique().with_columns(has_result_race=pl.lit(True))
    tk = train_keys.select(KEYS).with_columns(pl.col("horse_number").cast(pl.Int64)).with_columns(in_train=pl.lit(True))
    m = (sb.join(tk, on=KEYS, how="left").join(res_races, on="race_id", how="left")
         .join(rr.unique(subset=KEYS), on=KEYS, how="left"))
    cause = (pl.when(pl.col("in_train").is_not_null()).then(pl.lit("学習表に入っている"))
             .when(pl.col("has_result_race").is_null()).then(pl.lit("生データ未取得: そのレースの結果が無い"))
             .when(pl.col("finish_position").is_null()).then(pl.lit("正当: 結果に行が無い（出走取消・競走除外など）"))
             .when(pl.col("finish_position") <= 0).then(pl.lit("正当: 着順が 0 以下（競走中止・失格など）"))
             .otherwise(pl.lit("要確認: 結果に着順があるのに学習表に無い")))
    return m.select(cause.alias("区分")).group_by("区分").len().rename({"len": "行数"}).sort("行数", descending=True)


# ── ユニーク値による識別 ──────────────────────────────────────────────────────

SENTINEL_TOKENS = {"-1", "0", "", "-", "--", "nan", "none", "null", "n/a"}
# 着順欄に入りうる、数字ではない正規のトークン（netkeiba の表記）。ここに無い文字列は「解析失敗」の疑い。
FINISH_STATUS_TOKENS = {"取": "出走取消", "除": "競走除外", "中": "競走中止", "失": "失格", "降": "降着"}


def value_profile(df: pl.DataFrame, columns: list[str] | None = None, *, top: int = 6) -> pl.DataFrame:
    """列ごとのユニーク値の要約。欠損の代用値（-1・0・空文字・"-" など）や、数値のはずの列に混じる文字列を見つける。"""
    rows = []
    n = df.height
    for c in columns or df.columns:
        s = df[c]
        vc = s.drop_nulls().cast(pl.Utf8).value_counts(sort=True) if s.len() else None
        sent = 0
        non_numeric = 0
        top_vals = []
        if vc is not None and vc.height:
            cnt_col = [x for x in vc.columns if x != c][0]
            vals = vc[c].to_list()
            cnts = vc[cnt_col].to_list()
            top_vals = [f"{v}:{k}" for v, k in list(zip(vals, cnts))[:top]]
            sent = sum(k for v, k in zip(vals, cnts) if str(v).strip().lower() in SENTINEL_TOKENS)
            if s.dtype in (pl.Utf8, pl.Categorical):
                num = sum(k for v, k in zip(vals, cnts) if _is_num(v))
                if num >= 0.9 * sum(cnts):
                    non_numeric = sum(k for v, k in zip(vals, cnts) if not _is_num(v) and str(v).strip().lower() not in SENTINEL_TOKENS)
        rows.append({"column": c, "dtype": str(s.dtype), "null": s.null_count(), "unique": s.n_unique(),
                     "代用値らしい値(-1,0,空,-)": sent, "数値列に混じる文字列": non_numeric, "上位の値": " | ".join(top_vals)})
    return pl.DataFrame(rows).with_columns((pl.col("代用値らしい値(-1,0,空,-)") / max(n, 1)).alias("代用値の割合"))


def _is_num(v: object) -> bool:
    try:
        float(str(v))
        return True
    except ValueError:
        return False


def finish_status_evidence(rr: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """race_result の着順が 0 以下の行を、周辺の列（タイム・上り3F・通過順）から推定して分ける。

    元のトークン（取/除/中/失/降 か、解析失敗か）はパーサが -1 にまとめて捨てているため、確定はできない。
    タイムが無い → 取消・除外・中止と整合 / タイムがある → 失格・降着か解析失敗のどちらか（要・元データ確認）。
    戻り値: (区分別件数, 要確認の行)。
    """
    nf = rr.filter(pl.col("finish_position") <= 0)
    has_time = (pl.col("time_sec").fill_null(0) > 0)
    cat = (pl.when(has_time).then(pl.lit("タイムあり: 失格・降着か解析失敗（要確認）"))
           .otherwise(pl.lit("タイムなし: 出走取消・競走除外・競走中止と整合")))
    summary = nf.select(cat.alias("区分")).group_by("区分").len().rename({"len": "行数"}).sort("行数", descending=True)
    pos = rr.filter(pl.col("finish_position") > 0)
    gaps = (pos.group_by("race_id").agg(pl.col("finish_position").max().alias("最大着順"), pl.len().alias("着順付きの頭数"))
            .filter(pl.col("最大着順") > pl.col("着順付きの頭数")))
    suspects = nf.filter(has_time).select("race_id", "horse_number", "finish_position", "time_sec", "finish_time", "passing_order")
    return summary, pl.concat([suspects.with_columns(pl.lit("着順<=0なのにタイムあり").alias("種別"),
                                                     pl.lit(None, dtype=pl.Int64).alias("最大着順"), pl.lit(None, dtype=pl.Int64).alias("着順付きの頭数")),
                               gaps.with_columns(pl.lit("着順に欠番").alias("種別"))], how="diagonal_relaxed")


def track_bias_chronology_check(df: pl.DataFrame, winner_l3f: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """「当日の直前傾向」が時刻順になっているか。その日その場その路面で**時刻が最初**のレースに値が入っていたら、
    まだ行われていない後続レースの結果を使っている（先読み）。戻り値: (件数の要約, 該当レース)。"""
    ctx = track_bias_context(df, winner_l3f)
    r = df.select("race_id", "date", "venue_code", "surface", "start_time", "track_bias_winner_3f_same_day_prior").unique(subset=["race_id"]).join(ctx, on="race_id")
    first = r.filter(pl.col("tb_prior_races") == 0)
    leak = first.filter(pl.col("track_bias_winner_3f_same_day_prior").is_not_null())
    summary = pl.DataFrame({"区分": ["時刻が最初のレース（本来、直前の傾向は無い）", "うち値が入っている（後続レースの結果を使った疑い）", "うち欠損（正しい）"],
                            "レース数": [first.height, leak.height, first.height - leak.height]})
    return summary, leak.select("race_id", "date", "venue_code", "surface", "start_time", "track_bias_winner_3f_same_day_prior").sort("race_id")
