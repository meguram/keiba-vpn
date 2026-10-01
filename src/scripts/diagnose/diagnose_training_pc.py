"""【学習PC で実行】特徴量ストア・学習済みモデル・環境を調査してレポート(JSON)を出す。

目的: 「このPCでしか分からないこと」を集め、``decide.py``（開発PC）が判断できるようにする。
  - 特徴量ストアの実サイズ・列数・年・parquet の行グループ構成（行フィルタ読み込みが効くか）
  - 実データでの推論向け読み込み（``load_rows_for_keys``）の時間とメモリ
  - 学習済みアンサンブルのファイルサイズ・ロード時間・推論メモリ
  - ライブラリ版（VPS/GCP の実行環境と揃っているかの比較元）
  - 特徴量ビルダーの状態、動的特徴量（T-45 時点で初めて決まるもの）のラベル付け用 CSV

使い方（リポジトリのルートで。読み取りのみで、ストアやモデルは変更しない）:
  python -m src.scripts.diagnose.diagnose_training_pc --out reports/training_pc.json
  python -m src.scripts.diagnose.diagnose_training_pc --model-dir models/ensemble --sample-races 3 \
      --feature-sample 200 --labels-csv reports/dynamic_feature_labels.csv

出力したレポートと（記入済みの）ラベルCSVを開発PCへ持ち帰り、``decide`` に渡す。
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from src.scripts.diagnose.common import (
    add_check,
    env_presence,
    library_versions,
    memory_info,
    new_report,
    print_summary,
    run_check,
    run_inference_probe,
    save_report,
    timed,
    current_rss_mb,
)

# 名前から「開催当日に初めて決まる/変わる」可能性が高い特徴量を推定するキーワード（CSVの初期値のみ。最終判断は人が記入）
DYNAMIC_HINTS = (
    "odds", "ninki", "popularity", "weight", "bataiju", "zogen", "track_condition", "baba", "going",
    "scratch", "cancel", "jockey_change", "declared", "post_position", "waku", "gate", "horse_number",
    "umaban", "weather", "tenko",
    "オッズ", "人気", "馬体重", "増減", "馬場", "取消", "除外", "枠", "馬番", "天候",
)


def guess_dynamic(name: str) -> bool:
    low = name.lower()
    return any(h in low or h in name for h in DYNAMIC_HINTS)


def _open_store(args):
    from src.pipeline.features.feature_store import FeatureStore

    return FeatureStore(base_dir=args.base_dir)


def _store_or_none(args):
    """特徴量ディレクトリが存在するときだけストアを返す（無ければ呼び出し側で skip）。"""
    store = _open_store(args)
    return store if store._features_dir.is_dir() else None


def check_environment(_report):
    mem = memory_info()
    import os

    libs = library_versions()
    return "ok", f"メモリ{mem['mem_total_mb']}MB / CPU{os.cpu_count()} / libs={libs}", {
        **mem, "cpu_count": os.cpu_count(), "libraries": libs,
        "env_configured": env_presence(["KEIBA_MODEL_STORE", "KEIBA_FEATURE_BUILDER", "GCS_PROJECT_ID"]),
    }


NO_STORE = ("skip", "特徴量ストアがありません（--base-dir を確認。学習データを作ったPCで実行）", {})


def check_feature_store(report, args):
    store = _store_or_none(args)
    if store is None:
        return NO_STORE
    cols = store.list_columns()
    if not cols:
        return "ng", "特徴量ストアに列がありません（base_dir を確認）", {"n_columns": 0}

    sizes: list[int] = []
    blocks: dict[str, int] = {}
    years: set[str] = set()
    rows_by_col: list[int] = []
    for c in cols:
        meta = store.column_info(c) or {}
        blocks[meta.get("table_block", "?")] = blocks.get(meta.get("table_block", "?"), 0) + 1
        rows_by_col.append(int(meta.get("rows") or 0))
        for y, rel in (meta.get("year_paths") or {}).items():
            years.add(str(y))
            p = store._features_dir / rel
            if p.is_file():
                sizes.append(p.stat().st_size)
        rel = meta.get("rel_path")
        if rel and (store._features_dir / rel).is_file():
            sizes.append((store._features_dir / rel).stat().st_size)
    total_mb = sum(sizes) / 2**20
    detail = f"{len(cols)}列 / 年{sorted(years)[:1]}〜{sorted(years)[-1:]} / 合計{total_mb:.0f}MB / ブロック{blocks}"
    return "ok", detail, {
        "n_columns": len(cols), "blocks": blocks, "years": sorted(years), "total_mb": round(total_mb, 1),
        "n_files": len(sizes), "median_file_mb": round(statistics.median(sizes) / 2**20, 2) if sizes else 0,
        "max_file_mb": round(max(sizes) / 2**20, 2) if sizes else 0,
        "max_rows_per_column": max(rows_by_col) if rows_by_col else 0,
    }


def check_parquet_layout(report, args):
    """行グループ数と race_id の統計の有無。1行グループ=述語プッシュダウンが効かず全件読みになる。"""
    import pyarrow.parquet as pq

    store = _store_or_none(args)
    if store is None:
        return NO_STORE
    sample = store.list_columns()[: args.layout_sample]
    rg_counts, has_stats, checked = [], 0, 0
    for c in sample:
        meta = store.column_info(c) or {}
        rels = list((meta.get("year_paths") or {}).values()) or ([meta["rel_path"]] if meta.get("rel_path") else [])
        for rel in rels[:1]:
            p = store._features_dir / rel
            if not p.is_file():
                continue
            pf = pq.ParquetFile(p)
            rg_counts.append(pf.num_row_groups)
            checked += 1
            try:
                idx = pf.schema_arrow.get_field_index("race_id")
                st = pf.metadata.row_group(0).column(idx).statistics
                has_stats += int(bool(st and st.has_min_max))
            except Exception:  # noqa: BLE001
                pass
    if not checked:
        return "skip", "確認できるparquetがありません", {}
    med = statistics.median(rg_counts)
    status = "ok" if med > 1 else "warn"
    return status, f"行グループ数(中央値)={med} / race_id統計あり {has_stats}/{checked}", {
        "row_groups_median": med, "race_id_stats_ratio": round(has_stats / checked, 2), "files_checked": checked,
    }


def check_row_read(report, args):
    """実データで推論向けの行フィルタ読み込みを測る（load_columns との比較つき）。"""
    store = _store_or_none(args)
    if store is None:
        return NO_STORE
    cols = [c for c in store.list_columns() if (store.column_info(c) or {}).get("table_block") == "race_horse_tbl"]
    if not cols:
        return "skip", "race_horse_tbl の列がありません", {}
    cols = cols[: args.feature_sample]
    years = store.available_years() or sorted({y for c in cols for y in (store.column_info(c) or {}).get("year_paths", {})})
    if not years:
        return "skip", "年が特定できません", {}
    year = years[-1]
    first = store.load_column(cols[0], years=[year])
    race_ids = list(dict.fromkeys(first["race_id"].astype(str)))[-args.sample_races:]

    rss0 = current_rss_mb()
    rows, t_rows = timed(store.load_rows_for_keys, cols, race_ids, [year])
    rss_rows = current_rss_mb() - rss0

    rss1 = current_rss_mb()
    _, t_cols = timed(store.load_columns, cols, years=[year])
    rss_cols = current_rss_mb() - rss1

    per_race = t_rows / max(1, len(race_ids))
    ratio = t_cols / t_rows if t_rows else 0
    scale = 1000 / max(1, len(cols))
    status = "ok" if per_race * scale < 3 else "warn"
    return status, (
        f"{len(cols)}列×{len(race_ids)}レース: 行フィルタ {t_rows:.2f}s(+{rss_rows:.0f}MB) / "
        f"全読み {t_cols:.2f}s(+{rss_cols:.0f}MB) → 1000列換算 {per_race * scale:.1f}s/レース"
    ), {
        "columns": len(cols), "races": len(race_ids), "rows_sec": round(t_rows, 3), "rows_mem_mb": round(rss_rows),
        "full_sec": round(t_cols, 3), "full_mem_mb": round(rss_cols), "speedup": round(ratio, 1),
        "est_sec_per_race_1000cols": round(per_race * scale, 2), "result_rows": int(len(rows)),
    }


def check_model(report, args):
    d = Path(args.model_dir)
    if not d.is_dir():
        return "skip", f"モデルディレクトリなし: {d}（--model-dir で指定）", {}
    files = {p.name: p.stat().st_size for p in d.iterdir() if p.is_file()}
    total_mb = sum(files.values()) / 2**20
    probe = run_inference_probe(d, rows=args.probe_rows)
    detail = (
        f"合計{total_mb:.0f}MB / ロード{probe['load_sec']}s / 推論ピークRSS {probe['peak_rss_mb']}MB / "
        f"予測{probe['predict_sec_median']}s(1レース) / 特徴量{probe['n_features']}列"
    )
    return "ok", detail, {
        "total_mb": round(total_mb, 1), "files_mb": {k: round(v / 2**20, 1) for k, v in files.items()}, **probe,
    }


def check_builder(report, args):
    import os

    name = (os.environ.get("KEIBA_FEATURE_BUILDER") or "pseudo").strip().lower()
    try:
        from src.pipeline.features.pseudo_builder import get_feature_builder

        builder = get_feature_builder()
    except NotImplementedError as e:
        return "warn", f"KEIBA_FEATURE_BUILDER={name} は未実装: {e}", {"builder": name, "is_pseudo": False, "implemented": False}
    is_pseudo = getattr(builder, "name", "") == "pseudo"
    return ("warn" if is_pseudo else "ok"), f"ビルダー={getattr(builder, 'name', type(builder).__name__)}", {
        "builder": getattr(builder, "name", name), "is_pseudo": is_pseudo, "implemented": True,
    }


def write_label_template(args) -> Path | None:
    """特徴量ごとに dynamic/static を記入するためのCSVを出力する（既存ファイルは上書きしない）。"""
    path = Path(args.labels_csv)
    if path.exists():
        return path
    store = _store_or_none(args)
    cols = store.list_columns() if store else []
    if not cols:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "table_block", "source", "guess", "label"])
        for c in cols:
            meta = store.column_info(c) or {}
            w.writerow([c, meta.get("table_block", ""), meta.get("source", ""), "dynamic" if guess_dynamic(c) else "static", ""])
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/training_pc.json")
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--model-dir", default="models/ensemble")
    ap.add_argument("--sample-races", type=int, default=3)
    ap.add_argument("--feature-sample", type=int, default=200, help="行フィルタ読み込みを測る列数（多いほど正確・遅い）")
    ap.add_argument("--layout-sample", type=int, default=30)
    ap.add_argument("--probe-rows", type=int, default=18)
    ap.add_argument("--labels-csv", default="reports/dynamic_feature_labels.csv")
    args = ap.parse_args(argv)

    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    report = new_report("training_pc")
    run_check(report, "environment", check_environment)
    run_check(report, "feature_store", lambda r: check_feature_store(r, args))
    run_check(report, "parquet_layout", lambda r: check_parquet_layout(r, args))
    run_check(report, "feature_row_read", lambda r: check_row_read(r, args))
    run_check(report, "model", lambda r: check_model(r, args))
    run_check(report, "builder", lambda r: check_builder(r, args))
    try:
        p = write_label_template(args)
        add_check(report, "dynamic_label_template", "ok" if p else "skip", f"{p}（label列に dynamic/static を記入して開発PCへ）" if p else "特徴量なし")
    except Exception as e:  # noqa: BLE001
        add_check(report, "dynamic_label_template", "error", f"{type(e).__name__}: {e}")

    print_summary(report)
    print(f"\nレポート: {save_report(report, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
