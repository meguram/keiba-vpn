"""
血統メタクラスタリング (notebooks/pedigree/bloodline_subgroup_analysis.ipynb §21〜§22)
の最終アーティファクトを `data/research/bloodline_meta_cluster/` に書き出すビルダー。

ノートブックの全コードセルを抽出し、headless (matplotlib Agg + plt.show 抑制) で
順次実行することで、§22-4 までの結果と保存処理が同時に走る。

Usage:
    python -m src.research.pedigree.build_meta_cluster_artifacts

実行後、`data/research/bloodline_meta_cluster/` 以下に以下が生成される:
    unified.parquet
    l2_profiles.json
    overall_raw.json
    non_main_root_of.json
    non_main_group_labels.json
    sid_to_main.json
    sid_to_name.json
    scaler.json
    knn_data.json
    meta.json
"""
from __future__ import annotations

import io
import json
import os
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NB_PATH = ROOT / "notebooks/pedigree/bloodline_subgroup_analysis.ipynb"
TBL_DIR = ROOT / "data/local/tables"
IDX_DIR = ROOT / "data/research/pedigree_race_index"
ART_DIR = ROOT / "data/research/bloodline_meta_cluster"


def _extract_cells() -> str:
    """ノートブックの全コードセルを連結して 1 つの Python スクリプトを返す。"""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    parts: list[str] = [
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as _plt_mod",
        "_plt_mod.show = lambda *a, **k: None",
        "from IPython.display import display, HTML, clear_output",
        "import warnings; warnings.filterwarnings('ignore')",
        "",
    ]
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        parts.append(f"# === cell {i} ===")
        parts.append(src)
        parts.append("")
    return "\n".join(parts)


def build(quiet: bool = False) -> Path:
    """ノートブックを実行してアーティファクトを生成。"""
    art_dir = ROOT / "data/research/bloodline_meta_cluster"

    script = _extract_cells()
    nb_dir = NB_PATH.parent

    # ノートブックは notebooks/pedigree/ から実行されることを前提にした相対パスを使う
    cwd_orig = Path.cwd()
    os.chdir(nb_dir)

    g: dict = {"__name__": "__main__", "__file__": str(NB_PATH)}
    try:
        if quiet:
            buf = io.StringIO()
            with redirect_stdout(buf):
                exec(compile(script, str(NB_PATH), "exec"), g)
        else:
            exec(compile(script, str(NB_PATH), "exec"), g)
    finally:
        os.chdir(cwd_orig)

    if not (art_dir / "meta.json").exists():
        raise RuntimeError(
            f"アーティファクト生成失敗: {art_dir / 'meta.json'} が見つかりません"
        )

    # Web 統計エンドポイント用の records / horse_to_sire / 特殊適性タグを追加生成
    build_race_records(art_dir, quiet=quiet)
    build_horse_to_sire(art_dir, quiet=quiet)
    build_special_tags(art_dir, quiet=quiet)

    return art_dir


# ───────────────────────────────────────────────────────────
#  race_records.parquet ビルダー
#   全年の race_result_flat から、Web 統計用に必要な列だけを抽出し、
#   payoff dict から複勝払戻 (yen) を horse_number 単位で展開して付与。
# ───────────────────────────────────────────────────────────


_PLACE_KEY_CANDIDATES = ("複勝", "fukusho", "place")


# ── ペース / lap_type 判定 (notebook §20-1 と同等ロジック) ─────────
def _parse_pace_obj(p):
    if p is None:
        return None
    if isinstance(p, str):
        try:
            import json as _json
            p = _json.loads(p)
        except Exception:
            return None
    if not isinstance(p, dict):
        return None
    return p


def _parse_lap_list(s):
    """'[12.3, 11.8, ...]' 形式の文字列 / list を float リストに正規化"""
    if s is None:
        return None
    if isinstance(s, list):
        try:
            return [float(x) for x in s if x is not None]
        except (TypeError, ValueError):
            return None
    if isinstance(s, str):
        import ast
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [float(x) for x in v if x is not None]
        except Exception:
            return None
    return None


def _is_gear_change_race(laps) -> bool:
    """ギアチェンジレースの判定。

    定義 (ユーザー要件):
      L3F 区間に入る際に 12.0 → 11.0 のような急加速 (>=0.7秒) があり、
      その加速がL3F内で平均的に持続している (L3F平均 ≤ pre_l3f − 0.5秒)
      かつ、絶対水準として L3F 平均が 11.83秒 (=l3f 35.5) 以下。

    laps: 1F (200m) ごとの通過タイム配列。
    """
    if not isinstance(laps, list) or len(laps) < 4:
        return False
    try:
        pre = float(laps[-4])             # L3F 直前の 1F
        l3f_first = float(laps[-3])       # L3F 入り口 1F
        l3f_avg = sum(float(x) for x in laps[-3:]) / 3.0
    except (TypeError, ValueError):
        return False
    if pre <= 0 or l3f_first <= 0 or l3f_avg <= 0:
        return False
    accel_burst = pre - l3f_first         # 入り口での 1F 加速幅
    accel_avg   = pre - l3f_avg           # L3F 全体での加速幅 (持続)
    return (accel_burst >= 0.7) and (accel_avg >= 0.5) and (l3f_avg <= 11.83)


def _classify_lap_type(pace_dict, laps) -> str | None:
    """RPCI / L3FR / midR の組み合わせで 4 分類する。"""
    if pace_dict is None:
        return None
    try:
        first3f = float(pace_dict.get("first_half_3f", 0))
        last3f = float(pace_dict.get("last_3f", pace_dict.get("l3f", 0)))
    except (TypeError, ValueError):
        return None
    if first3f <= 0 or last3f <= 0:
        return None
    rpci = first3f / last3f if last3f else None
    # ラスト 3F 率 (低いほど後半瞬発)
    l3fr = last3f / (first3f + last3f) if (first3f + last3f) else None
    # 中盤 (lap_times の中央寄り)
    midr = None
    if isinstance(laps, list) and len(laps) >= 5:
        n = len(laps)
        mid = laps[n // 3 : 2 * n // 3]
        if mid:
            midr = sum(mid) / len(mid)

    # 分類: notebook §20-1 と同じ
    if rpci is None or l3fr is None:
        return None
    if rpci >= 1.04 and l3fr <= 0.49:
        return "スロー瞬発力"
    if rpci <= 0.97:
        return "持久力勝負"
    if 0.97 < rpci < 1.04 and (midr is None or midr < 12.5):
        return "持続力勝負"
    return "その他"


def _parse_place_payouts(payoff) -> dict[int, float]:
    """payoff (dict) → {horse_number: 複勝払戻金額(円)}"""
    if payoff is None:
        return {}
    if isinstance(payoff, str):
        try:
            payoff = json.loads(payoff)
        except Exception:
            return {}
    if not isinstance(payoff, dict):
        return {}
    rows = None
    for k in _PLACE_KEY_CANDIDATES:
        if k in payoff:
            rows = payoff[k]
            break
    if rows is None:
        return {}
    if isinstance(rows, dict):
        rows = [rows]
    out: dict[int, float] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        n_str = str(r.get("numbers", "")).strip()
        p_str = str(r.get("payout", "")).replace(",", "").strip()
        if not n_str or not p_str:
            continue
        try:
            hnum = int(n_str.split()[0])
            payout = float(p_str)
        except (ValueError, IndexError):
            continue
        out[hnum] = payout
    return out


def build_race_records(art_dir: Path, quiet: bool = False) -> Path:
    """全年の race_result_flat を結合し、Web 統計に必要な最小列セットを保存。

    保存列:
        race_id, date, venue, surface, distance, track_condition, grade, race_class,
        field_size, horse_id, horse_number, finish_position, popularity, odds (単勝倍率),
        place_payout_yen (複勝払戻 / 100 円賭け基準, 圏外は NaN)
    """
    files = sorted(TBL_DIR.glob("*/race_result_flat.parquet"))
    if not files:
        if not quiet:
            print(f"[race_records] race_result_flat ファイルなし: {TBL_DIR}/*/race_result_flat.parquet")
        return art_dir / "race_records.parquet"

    cols_in = [
        "race_id", "date", "venue", "surface", "distance", "track_condition",
        "grade", "race_class", "race_name", "field_size", "horse_id", "horse_number",
        "finish_position", "popularity", "odds", "passing_order", "last_3f", "time_sec",
    ]

    parts: list[pd.DataFrame] = []
    for f in files:
        if not quiet:
            print(f"[race_records] {f.parent.name}/{f.name}")
        df = pd.read_parquet(f)
        keep = [c for c in cols_in if c in df.columns]
        df = df[keep].copy()

        # 複勝払戻金 + lap_type (ペース由来) + ギアチェンジレース判定 は
        # race_result_race.parquet (race 単位) から取得
        payoff_map_per_race: dict[str, dict[int, float]] = {}
        lap_type_map: dict[str, str] = {}
        gear_change_set: set[str] = set()
        race_path = f.parent / "race_result_race.parquet"
        if race_path.exists():
            try:
                race_cols = ["race_id", "payoff"]
                rdf_check = pd.read_parquet(race_path, columns=None)
                for c in ("pace", "lap_times"):
                    if c in rdf_check.columns:
                        race_cols.append(c)
                rdf = pd.read_parquet(race_path, columns=race_cols)
                for row in rdf.itertuples(index=False):
                    rid = str(row.race_id)
                    payoff_map_per_race[rid] = _parse_place_payouts(row.payoff)
                    if "pace" in race_cols and "lap_times" in race_cols:
                        pace_d = _parse_pace_obj(getattr(row, "pace", None))
                        laps   = _parse_lap_list(getattr(row, "lap_times", None))
                        lt = _classify_lap_type(pace_d, laps)
                        if lt:
                            lap_type_map[rid] = lt
                        if _is_gear_change_race(laps):
                            gear_change_set.add(rid)
            except Exception as e:
                if not quiet:
                    print(f"[race_records]   warn: {race_path.name} 読込失敗 ({e})")
        # lap_type を flat レコードに付与
        if lap_type_map:
            df["lap_type"] = df["race_id"].astype(str).map(lap_type_map)
        else:
            df["lap_type"] = pd.NA
        if gear_change_set:
            df["is_gear_change_race"] = df["race_id"].astype(str).isin(gear_change_set)
        else:
            df["is_gear_change_race"] = False

        if "race_id" in df.columns and "horse_number" in df.columns:
            place_yen = []
            for rid, hnum in zip(df["race_id"].astype(str), df["horse_number"]):
                try:
                    hn = int(hnum)
                except (TypeError, ValueError):
                    place_yen.append(np.nan); continue
                place_yen.append(payoff_map_per_race.get(rid, {}).get(hn, np.nan))
            df["place_payout_yen"] = place_yen
        else:
            df["place_payout_yen"] = np.nan

        parts.append(df)

    out_df = pd.concat(parts, ignore_index=True)
    # 型整理
    out_df["race_id"] = out_df["race_id"].astype(str)
    out_df["horse_id"] = out_df["horse_id"].astype(str)
    if "date" in out_df.columns:
        out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce")
    for c in ("finish_position", "popularity", "horse_number", "field_size"):
        if c in out_df.columns:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce").astype("Int64")
    for c in ("distance", "odds", "place_payout_yen", "last_3f", "time_sec"):
        if c in out_df.columns:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce")

    # ── センチネル値クリーニング (-1, -999, 0 等は欠損扱い) ────────
    if "finish_position" in out_df.columns:
        out_df["finish_position"] = out_df["finish_position"].where(
            out_df["finish_position"] > 0, pd.NA
        )
    if "popularity" in out_df.columns:
        out_df["popularity"] = out_df["popularity"].where(
            out_df["popularity"].between(1, 30), pd.NA
        )
    if "horse_number" in out_df.columns:
        out_df["horse_number"] = out_df["horse_number"].where(
            out_df["horse_number"] > 0, pd.NA
        )
    if "odds" in out_df.columns:
        out_df["odds"] = out_df["odds"].where(
            (out_df["odds"] > 0) & (out_df["odds"] <= 999.9), np.nan
        )
    if "place_payout_yen" in out_df.columns:
        out_df["place_payout_yen"] = out_df["place_payout_yen"].where(
            out_df["place_payout_yen"] > 0, np.nan
        )
    if "last_3f" in out_df.columns:
        out_df["last_3f"] = out_df["last_3f"].where(
            (out_df["last_3f"] > 30) & (out_df["last_3f"] < 50), np.nan
        )
    if "time_sec" in out_df.columns:
        out_df["time_sec"] = out_df["time_sec"].where(
            (out_df["time_sec"] > 30) & (out_df["time_sec"] < 600), np.nan
        )

    # ── 派生列: age_class (race_class から年齢区分を抽出) ───────────
    if "race_class" in out_df.columns:
        rc = out_df["race_class"].fillna("").astype(str)
        out_df["age_class"] = pd.NA
        out_df.loc[rc.str.contains("２歳") | rc.str.contains("2歳"), "age_class"] = "juvenile"
        out_df.loc[(rc.str.contains("３歳") | rc.str.contains("3歳"))
                   & ~rc.str.contains("以上"), "age_class"] = "three_only"
        out_df.loc[rc.str.contains("以上") & (rc.str.contains("３歳") | rc.str.contains("3歳")),
                   "age_class"] = "mixed_3up"
        out_df.loc[rc.str.contains("以上") & (rc.str.contains("４歳") | rc.str.contains("4歳")),
                   "age_class"] = "older"

    # ── 派生列: interval_days (前走からの日数) ───────────
    if "horse_id" in out_df.columns and "date" in out_df.columns:
        out_df = out_df.sort_values(["horse_id", "date"]).reset_index(drop=True)
        prev = out_df.groupby("horse_id")["date"].shift(1)
        out_df["interval_days"] = (out_df["date"] - prev).dt.days
        out_df["interval_days"] = pd.to_numeric(
            out_df["interval_days"], errors="coerce"
        ).clip(lower=0, upper=999).astype("Int64")

    # ── 派生列: last_3f_rank (race 内の上がり 3F 順位 / field_size) ───────────
    if "last_3f" in out_df.columns and "race_id" in out_df.columns:
        out_df["last_3f_rank"] = (
            out_df.groupby("race_id")["last_3f"].rank(method="min", ascending=True)
            .astype("Float64")
        )
        out_df["last_3f_pct"] = (
            out_df["last_3f_rank"] / out_df["field_size"].astype("Float64")
        )

    # ── 派生列: 高速馬場判定 ───────────
    # 同条件 (venue × surface × distance × track_condition) ごとに 1着タイムの
    # 分布を取り、そのレースの 1着タイムが下位 (=速い側) 10% に入る場合「高速馬場」
    if {"time_sec", "race_id", "venue", "surface", "distance",
        "track_condition", "finish_position"}.issubset(out_df.columns):
        winners = out_df[out_df["finish_position"] == 1].copy()
        winners = winners[["race_id", "venue", "surface", "distance",
                           "track_condition", "time_sec"]].dropna()
        if not winners.empty:
            winners["_grp_pct"] = (
                winners.groupby(["venue", "surface", "distance", "track_condition"])
                ["time_sec"].rank(method="min", pct=True, ascending=True)
            )
            hi_speed_set = set(
                winners.loc[winners["_grp_pct"] <= 0.10, "race_id"].astype(str)
            )
            out_df["is_high_speed_race"] = out_df["race_id"].astype(str).isin(hi_speed_set)
        else:
            out_df["is_high_speed_race"] = False
    else:
        out_df["is_high_speed_race"] = False

    out_path = art_dir / "race_records.parquet"
    out_df.to_parquet(out_path, index=False)
    if not quiet:
        n_finished = int(out_df["finish_position"].notna().sum())
        def _cov(col): return out_df[col].notna().mean() * 100 if col in out_df.columns else 0.0
        n_juvenile = int((out_df.get("age_class") == "juvenile").sum()) if "age_class" in out_df.columns else 0
        n_older    = int((out_df.get("age_class") == "older").sum()) if "age_class" in out_df.columns else 0
        n_gear     = int(out_df.get("is_gear_change_race", pd.Series([False]*len(out_df))).sum())
        n_hispeed  = int(out_df.get("is_high_speed_race",  pd.Series([False]*len(out_df))).sum())
        print(
            f"[race_records] ✓ {out_path.name}: {len(out_df):,} rows ({n_finished:,} 完走), "
            f"odds {_cov('odds'):.1f}% / "
            f"place_payout {_cov('place_payout_yen'):.1f}% / "
            f"popularity {_cov('popularity'):.1f}% / "
            f"lap_type {_cov('lap_type'):.1f}% / "
            f"passing_order {_cov('passing_order'):.1f}% / "
            f"last_3f {_cov('last_3f'):.1f}% / "
            f"time_sec {_cov('time_sec'):.1f}% / "
            f"interval_days {_cov('interval_days'):.1f}%\n"
            f"               age: 二歳戦={n_juvenile:,} 古馬戦={n_older:,} / "
            f"ギアチェンジレース={n_gear:,} / 高速馬場レース={n_hispeed:,}"
        )
    return out_path


def build_horse_to_sire(art_dir: Path, quiet: bool = False) -> Path:
    """horse_id → 直接の父 (cat=1, gen=1) のマップを保存。"""
    cats_path = IDX_DIR / "horse_pedigree_cats.parquet"
    if not cats_path.exists():
        if not quiet:
            print(f"[horse_to_sire] {cats_path} がないのでスキップ")
        return art_dir / "horse_to_sire.parquet"
    cats = pd.read_parquet(
        cats_path,
        columns=["horse_id", "cat", "gen", "stallion_id", "stallion_name"],
    )
    direct = cats[(cats["cat"] == 1) & (cats["gen"] == 1)].copy()
    direct["horse_id"] = direct["horse_id"].astype(str)
    direct["stallion_id"] = direct["stallion_id"].astype(str)
    direct = direct.drop_duplicates("horse_id")
    out = direct[["horse_id", "stallion_id", "stallion_name"]]
    out_path = art_dir / "horse_to_sire.parquet"
    out.to_parquet(out_path, index=False)
    if not quiet:
        print(f"[horse_to_sire] ✓ {out_path.name}: {len(out):,} 頭")
    return out_path


# ───────────────────────────────────────────────────────────
#  特殊適性タグ ビルダー
#   L2 メタクラスタ (総合プロファイル) と直交する「タグ」を各種牡馬に付与。
#   - 脚質 / 馬場 / ペース / 距離 / コース形態 の 5 カテゴリ × 計 17 タグ
#   - 各タグ条件下の Empirical Bayes 補正勝率を計算
#   - 全種牡馬中 Top 20% を `is_top20` フラグで付与
#   - 絶対値 (rate_eb) も併記 → UI で両軸表示
# ───────────────────────────────────────────────────────────


# 急坂コース: ゴール前坂を「急」と呼べる場のみ。
# 中山 (高低差2.2m, 急勾配) / 阪神 (1.8m, 内回り急坂) / 中京 (約2m).
# 小倉は緩い上り (急坂とは呼ばれない) ため除外し、平坦扱いに変更。
_STEEP_VENUES = ("中山", "阪神", "中京")
# 内外区別のない場 (1コースのみ): 福島・札幌・函館・東京・中京・小倉
# (大回り = 直線長 vs 小回り は course_type ではなく venue で判定)
_PURE_BIG_TURN_VENUES = ("東京", "新潟")          # 純粋な大回り場 (直線も長い)
_PURE_SMALL_TURN_VENUES = ("福島", "札幌", "函館")  # 純粋な小回り場
_LONG_STRAIGHT_VENUES = ("東京", "新潟")           # 直線 500m+ の場 (東京525m/新潟外659m)

# course_profiles.json で未指定の中山・新潟の内外を公知情報で補完するオーバーライド
# (key, surface, distance) -> "内"/"外"
_COURSE_TYPE_OVERRIDES: dict[tuple[str, str, int], str] = {
    # 中山芝の内外 (有馬2500=内, AJCC・宝塚NK2200=外, 中山金杯2000=内, 弥生1800=内, 朝日杯1600=外)
    ("中山", "芝", 1200): "内",
    ("中山", "芝", 1600): "外",
    ("中山", "芝", 1800): "内",
    ("中山", "芝", 2000): "内",
    ("中山", "芝", 2200): "外",
    ("中山", "芝", 2500): "内",
    ("中山", "芝", 3600): "外",
    # 新潟芝の補完 (course_profiles にない部分)
    ("新潟", "芝", 1200): "内",
    ("新潟", "芝", 1800): "内",
    ("新潟", "芝", 2200): "内",
    # 新潟2400 は秋華賞トライアル等で外回りも使われるが、レアなので主流の内に寄せる
    ("新潟", "芝", 2400): "内",
}
# 距離区分 (一般的な日本競馬の感覚に合わせ、ステイヤー閾値を 2400m に引き上げ)
#   短距離 ~1400m  / マイル 1400-1800m
#   中距離 1800-2400m (ダービー2400 / JC2400 / 有馬2500 も中距離寄り)
#   ステイヤー 2400m+ (天皇賞春3200 / 菊花賞3000 / 有馬2500 / ダイヤモンドS3400)
_DIST_BINS = [0, 1400, 1800, 2400, 9999]
_DIST_LABELS = ["短距離", "マイル", "中距離", "長距離"]

# tag 定義: (tag_id, label, category, filter フラグ列)
# カテゴリ: 脚質 / 馬場 / ペース / 距離 / コース / 枠順 / 上がり / ローテ / 季節 / 年齢
TAG_DEFS = [
    # 脚質
    ("runstyle_nige",     "逃げ巧者",       "脚質",     "_is_nige"),
    ("runstyle_senko",    "先行型",         "脚質",     "_is_senko"),
    ("runstyle_sashi",    "差し馬",         "脚質",     "_is_sashi"),
    ("runstyle_oikomi",   "追い込み巧者",   "脚質",     "_is_oikomi"),
    # 馬場
    ("track_heavy",       "芝道悪巧者",     "馬場",     "_is_heavy"),
    ("track_good",        "芝良馬場巧者",   "馬場",     "_is_good"),
    ("track_high_speed",  "芝高速馬場巧者", "馬場",     "_is_high_speed"),
    # ペース (lap_type)
    ("pace_high",         "ハイペース巧者", "ペース",   "_is_pace_high"),
    ("pace_slow",         "スロー瞬発力",   "ペース",   "_is_pace_slow"),
    ("pace_long",         "ロンスパ巧者",   "ペース",   "_is_pace_long"),
    # 距離 (専用判定: 絶対勝率 + 出走シェア >= 10%, min_n=100)
    ("dist_sprint",       "スプリンター",   "距離",     "_is_sprint"),
    ("dist_mile",         "マイラー",       "距離",     "_is_mile"),
    ("dist_mid",          "中距離型",       "距離",     "_is_mid"),
    ("dist_stay",         "ステイヤー",     "距離",     "_is_stay"),
    # コース形態 (course_profiles.json + 公知 OVERRIDE で内/外回りを厳密判定)
    ("course_steep",      "急坂巧者",       "コース",   "_is_steep"),
    ("course_flat",       "平坦巧者",       "コース",   "_is_flat"),
    ("course_big_turn",   "外回り (大回り) 得意",  "コース",   "_is_big_turn"),
    ("course_small_turn", "内回り (小回り) 得意",  "コース",   "_is_small_turn"),
    ("course_long_str",   "直線長コース",   "コース",   "_is_long_straight"),
    # 枠順
    ("draw_inner",        "内枠得意",       "枠順",     "_is_inner"),
    ("draw_outer",        "外枠得意",       "枠順",     "_is_outer"),
    # 上がり
    ("speed_top",         "トップスピード", "上がり",   "_is_top_speed"),
    ("speed_gear",        "ギアチェンジ",   "上がり",   "_is_gear_change"),
    # ローテ
    ("rotate_fresh",      "休養明け強し",   "ローテ",   "_is_fresh_layoff"),
    ("rotate_repeat",     "叩き良化型",     "ローテ",   "_is_stack_improver"),
    # 季節
    ("season_summer",     "夏馬",           "季節",     "_is_summer"),
    # 年齢
    ("age_juvenile",      "二歳戦巧者",     "年齢",     "_is_juvenile_race"),
    ("age_older",         "古馬戦巧者",     "年齢",     "_is_older_race"),
]

MIN_TAG_N = 30
TOP_PCT = 0.20

# 距離タグ専用パラメータ:
#   距離タグ (sprint/mile/mid/stay) は「自身ベースライン比 (lift)」での判定を行うと
#   母系選抜バイアスで結果が直感に反する (例: ロードカナロアの長距離 192戦が「ステイヤー」判定)。
#   そこで距離タグだけは:
#     1) 出走数が十分大きい (n >= MIN_N_DIST_TAG)
#     2) その種牡馬の全出走に占めるシェア >= MIN_SHARE_DIST
#     3) 絶対勝率が条件平均の 110% 以上 (rate_eb >= cond_global * 1.10)
#   の 3 条件すべてを満たした場合のみ is_top20=True とする。
MIN_N_DIST_TAG = 100        # 距離タグ最小サンプル
MIN_SHARE_DIST = 0.10       # 距離タグの出走シェア下限 (= 全出走の 10% 以上)
DIST_TAG_IDS = {"dist_sprint", "dist_mile", "dist_mid", "dist_stay"}


def _load_course_type_map() -> dict[tuple[str, str, int], str]:
    """course_profiles.json + 公知 OVERRIDE から
    (venue_name, surface, distance) -> "内"/"外"/"" のマッピングを構築。

    内外区別のない場/距離は "" を返す (大半の地方場/中京/東京/福島 等)。
    """
    import re as _re
    path = ROOT / "data/knowledge/course_profiles.json"
    mapping: dict[tuple[str, str, int], str] = {}
    if path.exists():
        prof = json.loads(path.read_text())
        for vid, v in prof.get("venues", {}).items():
            vname = v.get("name")
            if not vname:
                continue
            db = v.get("draw_bias", {})
            if not isinstance(db, dict):
                continue
            for key, det in db.items():
                m = _re.match(r"(turf|dirt)_(\d+)(?:_(int|ext))?$", key)
                if not m:
                    continue
                surface = "芝" if m.group(1) == "turf" else "ダート"
                dist = int(m.group(2))
                suf = m.group(3)
                ct = ""
                if suf == "int":
                    ct = "内"
                elif suf == "ext":
                    ct = "外"
                if isinstance(det, dict):
                    ct_in = det.get("course_type", "")
                    if ct_in in ("内", "外"):
                        ct = ct_in
                mapping[(vname, surface, dist)] = ct
    # 公知情報で OVERRIDE
    mapping.update(_COURSE_TYPE_OVERRIDES)
    return mapping


def _attach_course_type(df: pd.DataFrame) -> pd.DataFrame:
    """df['venue'] / df['surface'] / df['distance'] から course_type ('内'/'外'/'') を付与。"""
    cmap = _load_course_type_map()
    if not cmap:
        df["course_type"] = ""
        return df
    venue = df["venue"].astype(str)
    surface = df["surface"].astype(str)
    distance = pd.to_numeric(df["distance"], errors="coerce").astype("Int64")

    def _lookup(v: str, s: str, d) -> str:
        if pd.isna(d):
            return ""
        try:
            return cmap.get((v, s, int(d)), "")
        except (ValueError, TypeError):
            return ""

    df["course_type"] = [
        _lookup(v, s, d) for v, s, d in zip(venue, surface, distance)
    ]
    return df


def _running_style_label(po, fs):
    """passing_order ('1-1-1-1' 等) と field_size から脚質ラベル"""
    if not isinstance(po, str) or not po:
        return None
    parts = po.strip().split("-")
    try:
        last = int(parts[-1])
    except (ValueError, IndexError):
        return None
    try:
        f = float(fs)
    except (TypeError, ValueError):
        return None
    if f <= 0 or pd.isna(f):
        return None
    if last == 1:
        return "逃げ"
    if last <= 3:
        return "先行"
    pct = last / f
    if pct <= 0.5:
        return "好位"
    if pct <= 0.75:
        return "差し"
    return "追込"


def build_special_tags(
    art_dir: Path,
    quiet: bool = False,
    *,
    out_name: str = "stallion_tags.parquet",
    cluster_out_name: str = "cluster_tags.json",
    min_tag_n: int = MIN_TAG_N,
    min_baseline_n: int | None = None,
    skip_cluster: bool = False,
    log_label: str = "special_tags",
) -> Path:
    """race_records + horse_to_sire から、種牡馬 × 特殊条件 EB勝率 とタグを算出して保存。

    Args:
        out_name: stallion_tags の出力ファイル名 (例: 'stallion_tags_full.parquet')
        cluster_out_name: cluster_tags JSON の出力ファイル名
        min_tag_n: 各タグ条件下での種牡馬の最低出走数
        min_baseline_n: ベースライン EB勝率算出に必要な最低出走数 (None なら min_tag_n*2)
        skip_cluster: True なら L2 集計をスキップ (full モード用)
        log_label: ログ接頭辞

    出力:
        stallion_tags.parquet:
            stallion_id, stallion_name, tag_id, tag_label, category,
            n_records, wins, rate_raw, rate_eb, pct_rank, is_top20
        cluster_tags.json (skip_cluster=False のみ):
            { "<L2>": [ {tag_id, tag_label, category, n_in_top20, n_eligible,
                         pct_in_top20, mean_rate_eb}, ... ] }
    """
    rec_path = art_dir / "race_records.parquet"
    h2s_path = art_dir / "horse_to_sire.parquet"
    unified_path = art_dir / "unified.parquet"
    nm_root_path = art_dir / "non_main_root_of.json"
    sid_name_path = art_dir / "sid_to_name.json"
    if not rec_path.exists() or not h2s_path.exists():
        if not quiet:
            print(f"[{log_label}] 必要なアーティファクトが不足")
        return art_dir / out_name
    if (not skip_cluster) and (not unified_path.exists()):
        if not quiet:
            print(f"[{log_label}] L2 集計に unified.parquet が必要")
        return art_dir / out_name
    if min_baseline_n is None:
        min_baseline_n = min_tag_n * 2

    # 1. データロード
    rec = pd.read_parquet(rec_path)
    h2s = pd.read_parquet(h2s_path)
    sire_of = dict(zip(h2s["horse_id"].astype(str), h2s["stallion_id"].astype(str)))
    sname_map = json.loads(sid_name_path.read_text()) if sid_name_path.exists() else {}

    # 2. 派生列を計算
    df = rec.copy()
    df["horse_id"] = df["horse_id"].astype(str)
    df["stallion_id"] = df["horse_id"].map(sire_of)
    df = df[df["stallion_id"].notna()]
    df = df[df["finish_position"].notna() & (df["finish_position"] > 0)]
    df["win"] = (df["finish_position"] == 1).astype(int)

    # 脚質
    style = [
        _running_style_label(po, fs)
        for po, fs in zip(df.get("passing_order", pd.Series([None]*len(df))),
                          df.get("field_size", pd.Series([None]*len(df))))
    ]
    df["style"] = style
    df["_is_nige"]   = df["style"] == "逃げ"
    df["_is_senko"]  = df["style"] == "先行"
    df["_is_sashi"]  = df["style"].isin(["好位", "差し"])
    df["_is_oikomi"] = df["style"] == "追込"

    # 馬場 (芝限定: surface == "芝" のレースのみで判定)
    # 「道悪/良馬場/高速馬場」は競馬用語として通常「芝の」コンディションを指すため、
    # ダート競走を混ぜると意味が変質する (例: ダートの稍重〜重は標準走行水準で道悪と呼ばない)
    is_turf = df["surface"] == "芝"
    df["_is_heavy"] = is_turf & df["track_condition"].isin(["重", "不良"])
    df["_is_good"]  = is_turf & (df["track_condition"] == "良")

    # ペース
    df["_is_pace_high"] = df["lap_type"] == "持久力勝負"
    df["_is_pace_slow"] = df["lap_type"] == "スロー瞬発力"
    df["_is_pace_long"] = df["lap_type"] == "持続力勝負"

    # 距離 (right=False で左閉右開: 2400m はステイヤー, 1800m は中距離 と直感的に)
    #   短距離 = [0, 1400)  / マイル = [1400, 1800)
    #   中距離 = [1800, 2400) / 長距離 = [2400, 9999)
    dist = pd.to_numeric(df["distance"], errors="coerce")
    df["dist_cat"] = pd.cut(dist, bins=_DIST_BINS, labels=_DIST_LABELS, right=False)
    df["_is_sprint"] = df["dist_cat"] == "短距離"
    df["_is_mile"]   = df["dist_cat"] == "マイル"
    df["_is_mid"]    = df["dist_cat"] == "中距離"
    df["_is_stay"]   = df["dist_cat"] == "長距離"

    # コース形態 (course_profiles.json + OVERRIDE で内/外回りを厳密判定)
    df = _attach_course_type(df)
    is_outer = (df["course_type"] == "外")
    is_inner = (df["course_type"] == "内")
    # 急坂/平坦 は venue ベース (中山・阪神・中京 = 急坂)
    df["_is_steep"]      = df["venue"].isin(_STEEP_VENUES)
    df["_is_flat"]       = ~df["venue"].isin(_STEEP_VENUES)
    # 大回り (= 外回り or 純粋大回り場): 東京/新潟 + 京都・阪神・中山の「外」
    df["_is_big_turn"]   = df["venue"].isin(_PURE_BIG_TURN_VENUES) | is_outer
    # 小回り (= 内回り or 純粋小回り場): 福島・札幌・函館 + 京都・阪神・中山の「内」 + 中京・小倉
    df["_is_small_turn"] = (
        df["venue"].isin(_PURE_SMALL_TURN_VENUES)
        | is_inner
        | df["venue"].isin(["中京", "小倉"])
    )
    # 直線長コース: 東京 (525m) / 新潟外 (659m) と京都・阪神・中山の「外」回り (300-400m台で純粋直線長は東京/新潟のみ)
    # → 東京/新潟全 + 阪神外 (473m) + 京都外 (404m) を「直線長レース」として広めに採用
    df["_is_long_straight"] = (
        df["venue"].isin(_LONG_STRAIGHT_VENUES)
        | (is_outer & df["venue"].isin(["京都", "阪神"]))
    )

    # ── 枠順 (内枠 1-3 / 外枠 上位 1/4) ────────────
    hn = pd.to_numeric(df.get("horse_number"), errors="coerce")
    fs = pd.to_numeric(df.get("field_size"), errors="coerce")
    df["_is_inner"] = (hn <= 3) & hn.notna()
    # 外枠: hn が field_size の最後 1/4 (例: 16頭立て → 13-16番)
    df["_is_outer"] = hn.notna() & fs.notna() & (hn > (fs * 0.75))

    # ── 上がり (race 内 last_3f 順位) ────────────
    if "last_3f_pct" in df.columns:
        l3 = pd.to_numeric(df["last_3f_pct"], errors="coerce")
        # トップスピード: race 内 last_3f 上位 25%
        df["_is_top_speed"] = (l3 <= 0.25) & l3.notna()
    else:
        df["_is_top_speed"] = False

    # ギアチェンジ: race 単位フラグ (lap_times 由来) — 入り口で 0.7 秒以上加速し
    # L3F 平均が pre_l3f より 0.5 秒以上速く、絶対水準 11.83 秒以下のレース
    if "is_gear_change_race" in df.columns:
        df["_is_gear_change"] = df["is_gear_change_race"].astype(bool)
    else:
        df["_is_gear_change"] = False

    # ── 高速馬場 (race 単位フラグ × 芝限定) ────────────
    if "is_high_speed_race" in df.columns:
        df["_is_high_speed"] = df["is_high_speed_race"].astype(bool) & is_turf
    else:
        df["_is_high_speed"] = False

    # ── 年齢区分 (race_class 由来) ────────────
    if "age_class" in df.columns:
        df["_is_juvenile_race"] = df["age_class"] == "juvenile"
        df["_is_older_race"]    = df["age_class"] == "older"
    else:
        df["_is_juvenile_race"] = False
        df["_is_older_race"]    = False

    # ── ローテ ────────────
    if "interval_days" in df.columns:
        ivd = pd.to_numeric(df["interval_days"], errors="coerce")
        df["_is_fresh_layoff"] = (ivd >= 60) & ivd.notna()       # 中休後初戦 (60+日)
        df["_is_stack_improver"] = (ivd >= 14) & (ivd <= 28) & ivd.notna()  # 中1-3週連戦
    else:
        df["_is_fresh_layoff"] = False
        df["_is_stack_improver"] = False

    # ── 季節 ────────────
    if "date" in df.columns:
        month = df["date"].dt.month
        df["_is_summer"] = month.isin([6, 7, 8, 9])
    else:
        df["_is_summer"] = False

    # 3. EB shrinkage
    global_win = float(df["win"].mean())
    PRIOR_N = 30

    # 種牡馬ごとの「全体 EB勝率」をベースラインとして算出
    # → タグ条件下の勝率を "lift = 条件勝率 / 自分のベースライン" で評価することで
    #    「逃げ馬は全体的に勝率高い」のような脚質効果を相殺し、純粋な「特性」を抽出
    base_g = df.groupby("stallion_id")["win"].agg(["sum", "count"]).rename(
        columns={"sum": "wins", "count": "n"})
    base_g = base_g[base_g["n"] >= min_baseline_n]  # ベースラインはより安定に
    base_g["rate_eb"] = (base_g["wins"] + global_win * PRIOR_N) / (base_g["n"] + PRIOR_N)
    sire_baseline: dict[str, float] = base_g["rate_eb"].to_dict()

    # 種牡馬全体の出走数マップ (share 計算 + 距離タグ判定用)
    sire_n_total: dict[str, int] = base_g["n"].to_dict()

    # 4. タグ集計
    records = []
    for tag_id, tag_label, category, flag_col in TAG_DEFS:
        sub = df[df[flag_col]]
        if sub.empty:
            continue
        # タグ条件下での全体勝率 (全種牡馬合算) → タグ条件のグローバル基準
        cond_global = float(sub["win"].mean()) if len(sub) else global_win

        g = sub.groupby("stallion_id").agg(
            wins=("win", "sum"),
            n=("win", "size"),
        ).reset_index()

        # 距離タグは選抜バイアス対策で min_n を厳しく
        is_dist_tag = tag_id in DIST_TAG_IDS
        min_n_this = max(min_tag_n, MIN_N_DIST_TAG) if is_dist_tag else min_tag_n
        g = g[g["n"] >= min_n_this]
        if len(g) == 0:
            continue

        g["rate_raw"] = g["wins"] / g["n"]
        g["rate_eb"] = (g["wins"] + cond_global * PRIOR_N) / (g["n"] + PRIOR_N)
        g["baseline_eb"] = g["stallion_id"].map(sire_baseline)
        g = g.dropna(subset=["baseline_eb"])
        if len(g) == 0:
            continue
        # 出走シェア: そのタグ条件下の出走数 / その種牡馬の全出走数
        g["sire_n_total"] = g["stallion_id"].map(sire_n_total).astype("Float64")
        g["share"] = g["n"].astype("Float64") / g["sire_n_total"].clip(lower=1)
        # lift = タグ条件下勝率 / その種牡馬の全体勝率 (1.0 = 平均並み, >1.0 = 得意)
        g["lift"] = g["rate_eb"] / g["baseline_eb"].clip(lower=1e-6)
        # 絶対値判定: 条件下平均の 10% 超え
        g["is_strong_abs"] = g["rate_eb"] >= cond_global * 1.10
        # 相対値判定: 自分の全体勝率比 1.15 以上 (= 顕著に得意)
        g["is_relative_strong"] = g["lift"] >= 1.15
        # 「専門家」フラグ:
        #   距離タグ → 絶対値 + 出走シェア >= 10% (lift は使わない: 選抜バイアス回避)
        #   その他   → 絶対値 + 相対値の両方
        if is_dist_tag:
            g["is_specialist"] = g["is_strong_abs"] & (g["share"] >= MIN_SHARE_DIST)
        else:
            g["is_specialist"] = g["is_strong_abs"] & g["is_relative_strong"]
        # Top20% ランキング: 複合スコア (rate_eb × lift) でランキング
        g["combo_score"] = g["rate_eb"] * g["lift"]
        g["pct_rank"] = g["combo_score"].rank(pct=True)
        # is_top20:
        #   距離タグ → specialist (絶対値+シェア) を満たせば自動的に top20=True
        #              (lift Top20% 制約は選抜バイアスを再導入してしまうため外す)
        #   その他   → specialist かつ combo Top20%
        if is_dist_tag:
            g["is_top20"] = g["is_specialist"]
        else:
            g["is_top20"] = (g["pct_rank"] >= (1.0 - TOP_PCT)) & g["is_specialist"]

        for _, r in g.iterrows():
            sid = str(r["stallion_id"])
            records.append({
                "stallion_id": sid,
                "stallion_name": sname_map.get(sid, sid),
                "tag_id": tag_id,
                "tag_label": tag_label,
                "category": category,
                "n_records": int(r["n"]),
                "wins": int(r["wins"]),
                "rate_raw": float(r["rate_raw"]),
                "rate_eb": float(r["rate_eb"]),
                "baseline_eb": float(r["baseline_eb"]),
                "cond_global": float(cond_global),
                "lift": float(r["lift"]),
                "share": float(r["share"]) if pd.notna(r["share"]) else 0.0,
                "sire_n_total": int(r["sire_n_total"]) if pd.notna(r["sire_n_total"]) else 0,
                "combo_score": float(r["combo_score"]),
                "pct_rank": float(r["pct_rank"]),
                "is_top20": bool(r["is_top20"]),
                "is_strong_abs": bool(r["is_strong_abs"]),
                "is_relative_strong": bool(r["is_relative_strong"]),
                "is_specialist": bool(r["is_specialist"]),
            })

    tags_df = pd.DataFrame(records)

    # ── 距離タグの排他化 ──
    # シェア最大だと「新馬戦・条件戦が多い短距離」が常に勝ってしまうので、
    # 「rate_eb (条件下勝率, EB shrinkage済) が最も高い距離区分」を主距離とする。
    # 加えて share>=10% / rate_eb>=cond_global*1.10 / n>=100 を既に通過した候補のみが対象。
    # 最終的に「is_specialist=True かつ rate_eb 最大の距離タグ」だけを is_top20=True に残す。
    if not tags_df.empty:
        dist_mask = tags_df["tag_id"].isin(DIST_TAG_IDS)
        dist_part = tags_df[dist_mask].copy()
        if not dist_part.empty:
            # specialist (= absolute strong + share threshold) を満たすものに限定
            spec = dist_part[dist_part["is_specialist"]].copy()
            primary_set: set = set()
            if not spec.empty:
                spec = spec.sort_values(
                    ["stallion_id", "rate_eb", "share"], ascending=[True, False, False]
                )
                primary_idx = spec.drop_duplicates("stallion_id", keep="first").index
                primary_set = set(primary_idx)
            # 距離タグのうち primary 以外は is_top20 を強制 False
            new_top = tags_df["is_top20"].copy()
            for idx in tags_df[dist_mask].index:
                new_top.loc[idx] = idx in primary_set
            tags_df["is_top20"] = new_top

    tags_path = art_dir / out_name
    tags_df.to_parquet(tags_path, index=False)

    if skip_cluster:
        if not quiet:
            n_uniq = tags_df["stallion_id"].nunique() if not tags_df.empty else 0
            n_top = int(tags_df["is_top20"].sum()) if not tags_df.empty else 0
            print(
                f"[{log_label}] ✓ {out_name}: {len(tags_df):,} 行 "
                f"({n_uniq} 種牡馬 × {len(TAG_DEFS)} タグ候補, top20={n_top})"
            )
        return tags_path

    # 5. クラスタ別 (L2) タグ集計
    unified = pd.read_parquet(unified_path)
    nm_root = json.loads(nm_root_path.read_text()) if nm_root_path.exists() else {}

    def _expand(entity_id: str, main_group: str) -> list[str]:
        if main_group == "非主流":
            return [sid for sid, root in nm_root.items() if str(root) == str(entity_id)]
        return [str(entity_id)]

    cluster_tags: dict[str, list[dict]] = {}
    for L2 in sorted(unified["L2"].unique()):
        if int(L2) < 0:
            continue
        sub_u = unified[unified["L2"] == L2]
        sids: list[str] = []
        for _, r in sub_u.iterrows():
            sids.extend(_expand(str(r["entity_id"]), str(r["main_group"])))
        sids = list(set(sids))
        sub_tags = tags_df[tags_df["stallion_id"].isin(sids)]
        items: list[dict] = []
        for tag_id, tag_label, category, _ in TAG_DEFS:
            tdf = sub_tags[sub_tags["tag_id"] == tag_id]
            n_eligible = int(len(tdf))
            n_in_top = int(tdf["is_top20"].sum())
            n_strong_abs = int(tdf["is_strong_abs"].sum()) if "is_strong_abs" in tdf.columns else 0
            mean_eb = float(tdf["rate_eb"].mean()) if n_eligible > 0 else None
            mean_lift = float(tdf["lift"].mean()) if n_eligible > 0 and "lift" in tdf.columns else None
            items.append({
                "tag_id": tag_id,
                "tag_label": tag_label,
                "category": category,
                "n_eligible": n_eligible,
                "n_in_top20": n_in_top,
                "n_strong_abs": n_strong_abs,
                "pct_in_top20": (n_in_top / n_eligible) if n_eligible > 0 else None,
                "mean_rate_eb": mean_eb,
                "mean_lift": mean_lift,
            })
        cluster_tags[str(int(L2))] = items

    ct_path = art_dir / cluster_out_name
    ct_path.write_text(json.dumps(cluster_tags, ensure_ascii=False, indent=2))

    if not quiet:
        print(
            f"[{log_label}] ✓ {tags_path.name}: {len(tags_df):,} 行 "
            f"({tags_df['stallion_id'].nunique()} 種牡馬 × {len(TAG_DEFS)} タグ候補) "
            f"/ Top20% 判定基準=lift"
        )
        # サンプル: 各タグの Top 3 種牡馬 (combo_score 基準, is_top20 のみ)
        for tag_id, tag_label, _, _ in TAG_DEFS:
            top = (tags_df[(tags_df["tag_id"] == tag_id) & tags_df["is_top20"]]
                   .sort_values("combo_score", ascending=False).head(3))
            if len(top):
                exs = ", ".join(
                    f"{r['stallion_name']}({r['rate_eb']*100:.1f}%, lift={r['lift']:.2f})"
                    for _, r in top.iterrows()
                )
                print(f"  [{tag_label}] top: {exs}")
            else:
                print(f"  [{tag_label}] (該当なし)")
        print(f"[cluster_tags] ✓ {ct_path.name}: {len(cluster_tags)} L2")

    return tags_path


def build_all_stallion_tags(art_dir: Path, quiet: bool = False) -> Path:
    """父系ツリー上の全種牡馬を対象に、より緩い閾値でタグを集計して保存。

    通常の `build_special_tags` (n>=30/baseline n>=60) より閾値を緩め、
    出走数が少ない種牡馬にも可能な限りタグを付与する。pedigree-map のツリー UI
    で「全種牡馬の馬名横にタグを表示」する用途のために `stallion_tags_full.parquet`
    を生成する。
    """
    return build_special_tags(
        art_dir,
        quiet=quiet,
        out_name="stallion_tags_full.parquet",
        min_tag_n=15,            # タグ条件下 15 出走 (緩め)
        min_baseline_n=30,       # ベースライン 30 出走 (緩め)
        skip_cluster=True,       # L2 集計はメイン版でカバー済
        log_label="all_stallion_tags",
    )


def main():
    quiet = "--quiet" in sys.argv or "-q" in sys.argv
    print(f"[build_meta_cluster_artifacts] notebook: {NB_PATH}")
    art_dir = build(quiet=quiet)
    print(f"\n[build_meta_cluster_artifacts] ✓ 完了: {art_dir}")
    for p in sorted(art_dir.glob("*")):
        size = p.stat().st_size
        print(f"  {p.name:>32}  {size:>10,} bytes")


if __name__ == "__main__":
    main()
