"""小さなメモリのマシンで、当日の出走馬の特徴量だけを作るための「サービング用バンドル」。

全履歴を使う計算（``build_from_runs``）は 2〜3GB を使う。当日の特徴量は出走馬の履歴だけで決まるので、
次のように分ける。

  大きなマシン（毎日）  全履歴から特徴量を作る → fit した値・過去走の素材・Elo の最終状態をバンドルに書き出す
  小さなマシン（当日）  バンドルから出走馬の履歴だけを読み、fit を省いて今日の行の特徴量を作る（数十 MB）

バンドル（``data/local/modeling/feature_store/serving_bundle/``）:
  meta.json        作成時刻・履歴の最終日・行数・馬数
  params.pkl       P（fit した値）、S（段階ごとの fit 結果と日ごとの馬場差）、Elo の最終状態
  runs.arrow       過去走の素材（Arrow IPC・無圧縮。馬 ID・日付順。メモリマップで読むので、触れた分だけがメモリに載る）
  horse_index.json 馬 ID → [開始行, 行数]

ここで作る特徴量は、全履歴で作った値と一致する（tests/pipeline/test_horse_history_serving.py が確認する）。
"""

from __future__ import annotations

import json
import pickle
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from .common import apply_repair, clean_runs
from .elo import elo_prerace
from .inference import card_to_raw_runs
from .match import build_frozen
from .profile import prepare

JST = timezone(timedelta(hours=9))
VERSION = 3

# 凍結した段階（history〜clusters）が過去走の行から読む列。参照されない列（騎手 ID・馬番・パース中間値など）は持たない。
# 足りない列は tests/pipeline/test_horse_history_serving.py が検出する。
SERVING_COLUMNS = [
    "E", "age", "class_level", "date", "direction", "dist_band", "distance", "finish_position", "fs_mean", "fs_std", "fs_top3", "going",
    "horse_id", "is_top3", "is_turf", "is_wet", "is_win", "jockey_weight", "l3f", "l3f_cnt", "l3f_cov", "l3f_gap", "l3f_min", "l3f_mu",
    "l3f_rk", "l3f_sd", "l3f_top3", "l3f_z", "l3f_z_back", "l3f_z_front", "logd", "n_fin", "n_pre", "pace_bal", "pos_early", "pos_late",
    "r_eff", "r_pre", "race_id", "res", "res2", "res_w", "rpci", "sex", "standing", "start_time", "surface", "time_sec", "venue_code", "wt",
    "_var",
]
EMPTY_RM = pl.DataFrame(schema={"horse_id": pl.Utf8, "run_date": pl.Date, "l3f_fix": pl.Float64, "weight_fix": pl.Float64, "src_date": pl.Date})


def default_bundle_dir() -> Path:
    import os

    return Path(os.environ.get("KEIBA_SERVING_BUNDLE_DIR") or Path(__file__).resolve().parents[4] / "data" / "local" / "modeling" / "feature_store" / "serving_bundle")


def write_bundle(out_dir: Path, F_cols: list[str], P: dict, S: dict, elo_state: dict, extras: dict | None = None) -> dict:
    """build_with_state の結果をバンドルとして書き出す（一時ディレクトリに作ってから置き換える）。"""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    F0 = S["F0"].join(S["sr_var"], on=["race_id", "horse_id"], how="left").select(SERVING_COLUMNS)
    state = {k: v for k, v in S.items() if k not in ("F0", "sr_var")}
    with (tmp / "params.pkl").open("wb") as f:
        pickle.dump({"version": VERSION, "P": P, "S": state, "elo": elo_state, "extras": extras or {}}, f, protocol=4)
    F0 = F0.sort(["horse_id", "date", "race_id"])
    F0.write_ipc(tmp / "runs.arrow", compression="uncompressed")
    idx = F0.with_row_index("i").group_by("horse_id", maintain_order=True).agg(pl.col("i").min().alias("s"), pl.len().alias("n"))
    (tmp / "horse_index.json").write_text(json.dumps({h: [s, n] for h, s, n in idx.iter_rows()}), encoding="utf-8")
    meta = {"version": VERSION, "built_at": datetime.now(JST).isoformat(timespec="seconds"), "rows": F0.height,
            "horses": F0["horse_id"].n_unique(), "history_last_date": str(F0["date"].max()), "columns": F0.columns,
            "elo_delta": elo_state["delta"]}
    (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return meta


class Bundle:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        with (self.path / "params.pkl").open("rb") as f:
            d = pickle.load(f)
        if d.get("version") != VERSION:
            raise ValueError(f"バンドルの版が違う: {d.get('version')} (期待 {VERSION})")
        self.P = d["P"]
        self.S = dict(d["S"], frozen=True)
        self.rating, self.cnt, self.delta = d["elo"]["rating"], d["elo"]["cnt"], d["elo"]["delta"]
        self.graded_names: dict[str, str] = d.get("extras", {}).get("graded_names", {})
        self._runs = None  # pyarrow.Table（メモリマップ）
        self._index: dict[str, list[int]] = {}

    def horses(self, horse_ids: list[str]) -> pl.DataFrame:
        """出走馬の過去走の素材（保存時点の最新まで）。無圧縮の Arrow ファイルをメモリマップで読み、該当の行だけを取り出す。"""
        import pyarrow as pa  # polars の scan_ipc は全体を読み込んでしまうため、pyarrow のメモリマップを使う

        if self._runs is None:
            self._runs = pa.ipc.open_file(pa.memory_map(str(self.path / "runs.arrow"))).read_all()  # ゼロコピー
            self._index = json.loads((self.path / "horse_index.json").read_text(encoding="utf-8"))
        parts = [self._runs.slice(*self._index[h]) for h in horse_ids if h in self._index]
        return pl.from_arrow(pa.concat_tables(parts)) if parts else pl.DataFrame()


def compute_entrants(bundle: Bundle, card: pl.DataFrame, graded_names: dict[str, str] | None = None) -> tuple[pl.DataFrame, dict]:
    """card（出馬表の行）のレースの特徴量を、バンドルから出走馬の履歴だけを読んで作る。"""
    t0 = time.perf_counter()
    today_raw = card_to_raw_runs(card, graded_names)
    bad = today_raw.filter(
        pl.col("date").is_null() | (pl.col("date").cast(pl.Utf8).fill_null("") == "")
        | ~(pl.col("distance").cast(pl.Float64, strict=False) > 0) | pl.col("surface").cast(pl.Utf8).fill_null("").eq(""))
    if bad.height:
        raise ValueError(f"出馬表のレース情報（日付・面・距離）が空のレース: {sorted(bad['race_id'].unique().to_list())}")
    today = apply_repair(clean_runs(today_raw), EMPTY_RM)
    race_ids = today["race_id"].unique().to_list()
    day = today["date"].min()
    elo_t = elo_prerace(bundle.rating, bundle.cnt, today, bundle.delta)
    F0_today = prepare(today, elo_t)
    hist = bundle.horses(today["horse_id"].unique().to_list())
    if hist.height:
        hist = hist.filter((pl.col("date") < day) & ~pl.col("race_id").is_in(race_ids))
    if "_var" not in F0_today.columns:
        F0_today = F0_today.with_columns(pl.lit(None, pl.Float64).alias("_var"))
    F0 = pl.concat([hist, F0_today], how="diagonal_relaxed") if hist.height else F0_today
    F = build_frozen(F0, bundle.P, bundle.S)
    out = F.filter(pl.col("race_id").is_in(race_ids))
    info = {"races": len(race_ids), "rows": out.height, "history_rows": hist.height,
            "weights_known_ratio": float((today_raw["weight"].cast(pl.Float64) > 0).fill_null(False).mean()) if today_raw.height else 0.0,
            "going_known": bool(card["track_condition"].is_not_null().all()) if "track_condition" in card.columns else False,
            "build_sec": round(time.perf_counter() - t0, 1), "engine": "bundle"}
    return out, info
