"""各モデルの予測性能を事前計算して JSON に保存する（監視ポータル /models が読むだけ）。読み取り専用で、GCS・本番データは書き換えない。

    python -m src.scripts.data.model_perf_cache build [--models ens_v3,ens_v4,ens_v5] [--no-importance] [--no-prod]
    python -m src.scripts.data.model_perf_cache show

出力: data/local/meta/model_perf/{index.json, <key>.json}（KEIBA_MODEL_PERF_DIR で変更）
 - test2025 … final_eval_vN/test2025_predictions.parquet から再計算（LogLoss・Brier・AUC・校正・Top1 的中率）
 - eval_series … 学習時の年別検証（ensemble*/fold_metrics の CSV があるもの）
 - prod_series … 2026 年の予測（predictions_backfill / predictions_v3）× 実着順（rank_tbl）の週別集計。2026 はテストセット②（選択に使わない）
 - importance … LightGBM 系メンバーのゲイン重要度（特徴量上位・群別）。XGB・NN・線形は含めない
重い処理（重要度の読み込み）は heavy_run.sh 経由で実行する: /tmp/heavy_run.sh model-perf python -m src.scripts.data.model_perf_cache build
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(os.environ.get("KEIBA_REPO_ROOT") or Path(__file__).resolve().parents[3])
JST = timezone(timedelta(hours=9))
TP = REPO_ROOT / "data" / "local" / "modeling" / "training_pipeline"
INF = Path(os.environ.get("KEIBA_INFERENCE_DIR") or REPO_ROOT / "data" / "local" / "inference")
RANK_TBL = REPO_ROOT / "data" / "local" / "features" / "target" / "rank_tbl"
OUT_DIR = Path(os.environ.get("KEIBA_MODEL_PERF_DIR") or REPO_ROOT / "data" / "local" / "meta" / "model_perf")
GROUPS_JSON = REPO_ROOT / "config" / "model_feature_groups.json"

# アンサンブルの版 → 評価出力・バンドル・推論の置き場
ENSEMBLES: dict[str, dict[str, Any]] = {
    "ens_v2": {"label": "アンサンブル v2", "eval": "final_eval_v2", "bundle": None, "mlflow": None, "backfill": None},
    "ens_v3": {"label": "アンサンブル v3", "eval": "final_eval_v3", "bundle": "final_v3/bundle_2025", "mlflow": None, "backfill": None},
    "ens_v4": {"label": "アンサンブル v4", "eval": "final_eval_v4", "bundle": "final_v4/bundle_v4_2025", "mlflow": "keiba-ensemble-v4", "backfill": "bundle_v4_2025"},
    "ens_v5": {"label": "アンサンブル v5", "eval": "final_eval_v5", "bundle": "final_v5/bundle_v5_2025", "mlflow": None, "backfill": "bundle_v5_2025"},
}
# 学習時の年別検証の CSV（モデル名の列 'model' で絞る）
FOLD_FILES: dict[str, tuple[str, str]] = {
    "ens_v2": ("ensemble_v2/ensemble_v2_fold_metrics.csv", "ensemble_v2"),
}
LOW_SAMPLE_RACES = 30
HEADS = (("win", "p_win", 1), ("top2", "p_top2", 2), ("top3", "p_top3", 3))


def _now() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)


# ───────────────────────── 指標 ─────────────────────────

def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = pd.Series(p).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[list[dict], float]:
    """等頻度の bins 区間ごとの (予測平均, 実測率)。ECE = 件数加重の |予測-実測|。"""
    if len(p) < bins * 5:
        return [], float("nan")
    order = np.argsort(p)
    out, ece = [], 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        pm, om = float(p[chunk].mean()), float(y[chunk].mean())
        out.append({"n": int(len(chunk)), "pred": round(pm, 5), "obs": round(om, 5), "lo": round(float(p[chunk].min()), 5), "hi": round(float(p[chunk].max()), 5)})
        ece += len(chunk) / len(p) * abs(pm - om)
    return out, ece


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def head_metrics(df: pd.DataFrame, with_reliability: bool = False) -> dict[str, Any]:
    """df: race_id, finish_position(or rank), p_win, p_top2, p_top3。"""
    fin = df["finish_position"].to_numpy(dtype=float)
    out: dict[str, Any] = {}
    for name, col, k in HEADS:
        y = (fin <= k).astype(float)
        p = df[col].to_numpy(dtype=float)
        m: dict[str, Any] = {"logloss": _f(logloss(y, p)), "brier": _f(brier(y, p)), "auc": _f(auc(y.astype(int), p))}
        rel, ece = reliability(y, p)
        m["ece"] = _f(ece)
        if with_reliability:
            m["reliability"] = rel
        out[name] = m
    return out


def top1_stats(df: pd.DataFrame) -> dict[str, Any]:
    """レース単位: 勝率最大の馬が 1 着 / 3 着以内だった割合。"""
    idx = df.groupby("race_id")["p_win"].idxmax()
    top = df.loc[idx]
    n = int(len(top))
    h1 = int((top["finish_position"] == 1).sum())
    h3 = int((top["finish_position"] <= 3).sum())
    return {"n_races": n, "hit": _f(h1 / n) if n else None, "ci": wilson(h1, n), "in_top3": _f(h3 / n) if n else None}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["finish_position", "p_win", "p_top2", "p_top3"]).copy()
    d = d[d["finish_position"] >= 1]
    sizes = d.groupby("race_id")["horse_id"].transform("size")
    return d[sizes >= 2]


# ───────────────────────── 評価時（テスト 2025・年別検証） ─────────────────────────

def build_test2025(eval_dir: str) -> dict[str, Any] | None:
    p = TP / eval_dir / "test2025_predictions.parquet"
    if not p.exists():
        return None
    d = clean(pd.read_parquet(p))
    if d.empty:
        return None
    res: dict[str, Any] = {"n_rows": int(len(d)), "n_races": int(d["race_id"].nunique()), "metrics": head_metrics(d, with_reliability=True), "top1": top1_stats(d)}
    rp = TP / eval_dir / "report.json"
    if rp.exists():
        try:
            rep = json.loads(rp.read_text(encoding="utf-8"))
            res["reported_logloss"] = rep.get("logloss")
            res["reported_vs_prev"] = {k: v for k, v in rep.items() if k.startswith("vs_")}
        except (OSError, ValueError):
            pass
    return res


def build_eval_series(key: str) -> list[dict]:
    rel = FOLD_FILES.get(key)
    if not rel:
        return []
    path = TP / rel[0]
    if not path.exists():
        return []
    d = pd.read_csv(path)
    if "model" in d.columns:
        d = d[d["model"] == rel[1]]
    rows = []
    for _, r in d.iterrows():
        fold = str(r.get("fold") or "")
        m = re.search(r"(\d{4})", fold)
        rows.append({"label": fold, "year": int(m.group(1)) if m else None, "win_logloss": _f(r.get("logloss_win")), "top2_logloss": _f(r.get("logloss_top2")),
                     "top3_logloss": _f(r.get("logloss_top3")), "top1_hit": _f(r.get("top1_hit")), "ece_win": _f(r.get("ece_win")),
                     "auc_win": _f(r.get("auc_win")), "brier_win": _f(r.get("brier_win")), "source": "ensemble 年別検証"})
    return sorted(rows, key=lambda x: (x["year"] or 0))


# ───────────────────────── 本番期間（2026・週別） ─────────────────────────

def _load_ranks() -> pd.DataFrame:
    frames = []
    for p in sorted(RANK_TBL.glob("2026/rank.parquet")):
        frames.append(pd.read_parquet(p, columns=["race_id", "horse_id", "rank"]))
    if not frames:
        return pd.DataFrame(columns=["race_id", "horse_id", "finish_position"])
    d = pd.concat(frames, ignore_index=True)
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    return d.rename(columns={"rank": "finish_position"})


def _week_of(day: str) -> tuple[str, str]:
    d = date(int(day[:4]), int(day[4:6]), int(day[6:8]))
    mon = d - timedelta(days=d.weekday())
    iso = mon.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}", mon.isoformat()


def _read_day(dirpath: Path, ranks: pd.DataFrame) -> pd.DataFrame | None:
    frames = []
    for p in dirpath.glob("*.parquet"):
        try:
            f = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        if not {"horse_id", "p_win", "p_top2", "p_top3"} <= set(f.columns):
            continue
        f = f[["horse_id", "p_win", "p_top2", "p_top3"]].copy()
        f["race_id"] = p.stem
        frames.append(f)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["horse_id"] = d["horse_id"].astype(str)
    d = d.merge(ranks, on=["race_id", "horse_id"], how="inner")
    return clean(d) if not d.empty else None


def build_prod_series(model_dirs: list[tuple[str, Path]], ranks: pd.DataFrame) -> list[dict]:
    """model_dirs: [(出どころの説明, <日付>/<race>.parquet を持つ親ディレクトリ)]。週ごとにレース単位で集計。"""
    by_week: dict[str, list[pd.DataFrame]] = {}
    week_start: dict[str, str] = {}
    for _, root in model_dirs:
        if not root.exists():
            continue
        for day_dir in sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)):
            d = _read_day(day_dir, ranks)
            if d is None:
                continue
            wk, ws = _week_of(day_dir.name)
            by_week.setdefault(wk, []).append(d)
            week_start[wk] = ws
    rows = []
    for wk in sorted(by_week):
        d = pd.concat(by_week[wk], ignore_index=True)
        m = head_metrics(d)
        t1 = top1_stats(d)
        # 1 レースあたりの勝率 LogLoss の標準誤差（レースの平均を 1 件として）
        y = (d["finish_position"] == 1).astype(float).to_numpy()
        p = np.clip(d["p_win"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
        per = pd.Series(-(y * np.log(p) + (1 - y) * np.log(1 - p))).groupby(d["race_id"].to_numpy()).mean()
        n = int(len(per))
        se = float(per.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        rows.append({"week": wk, "start": week_start[wk], "n_races": n, "low_sample": n < LOW_SAMPLE_RACES,
                     "win_logloss": m["win"]["logloss"], "win_logloss_se": _f(se), "top2_logloss": m["top2"]["logloss"], "top3_logloss": m["top3"]["logloss"],
                     "top1_hit": t1["hit"], "top1_ci": t1["ci"], "win_ece": m["win"]["ece"], "win_brier": m["win"]["brier"]})
    return rows


def prod_total(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    n = sum(r["n_races"] for r in rows)
    w = lambda k: _f(sum((r[k] or 0) * r["n_races"] for r in rows) / n) if n else None  # noqa: E731
    return {"n_races": n, "weeks": len(rows), "win_logloss": w("win_logloss"), "top2_logloss": w("top2_logloss"), "top3_logloss": w("top3_logloss"), "top1_hit": w("top1_hit")}


# ───────────────────────── 特徴量の寄与率 ─────────────────────────

def _group_rules() -> tuple[list[tuple[str, re.Pattern]], str]:
    cfg = json.loads(GROUPS_JSON.read_text(encoding="utf-8"))
    return [(r["group"], re.compile(r["regex"])) for r in cfg["rules"]], cfg.get("default", "その他")


def group_of(name: str, rules: list[tuple[str, re.Pattern]], default: str) -> str:
    for g, rx in rules:
        if rx.search(name):
            return g
    return default


def build_importance(bundle_rel: str, top_n: int = 30) -> dict[str, Any] | None:
    import lightgbm as lgb  # ホストの学習環境にある。無ければ呼び出し側が『未計算』にする

    bdir = TP / bundle_rel
    bj = bdir / "bundle.json"
    if not bj.exists():
        return None
    bundle = json.loads(bj.read_text(encoding="utf-8"))
    rules, default = _group_rules()
    result: dict[str, Any] = {"members": [], "heads": {}}
    for head in ("win", "top3"):
        acc: dict[str, float] = {}
        used = 0
        for mem in bundle.get("members", []):
            if mem.get("recipe") != "lgb_binary":
                continue
            cols = list(mem.get("feature_cols") or [])
            files = sorted((bdir / mem["name"]).glob(f"seed*/lgb_{head}.txt"))
            member_acc: dict[str, float] = {}
            for f in files:
                try:
                    booster = lgb.Booster(model_file=str(f))
                except Exception:  # noqa: BLE001
                    continue
                names = booster.feature_name()
                gain = booster.feature_importance(importance_type="gain").astype(float)
                if gain.sum() <= 0:
                    continue
                gain = gain / gain.sum()
                if names and names[0].startswith("Column_") and len(names) == len(cols):
                    names = cols
                for nm, g in zip(names, gain):
                    member_acc[nm] = member_acc.get(nm, 0.0) + float(g) / len(files)
            if member_acc:
                used += 1
                if head == "win":
                    result["members"].append({"name": mem["name"], "seeds": len(files), "n_features": len(cols)})
                for nm, g in member_acc.items():
                    acc[nm] = acc.get(nm, 0.0) + g
        if not used:
            continue
        tot = sum(acc.values()) or 1.0
        feats = sorted(((nm, g / tot) for nm, g in acc.items()), key=lambda x: -x[1])
        gshare: dict[str, float] = {}
        for nm, g in feats:
            gr = group_of(nm, rules, default)
            gshare[gr] = gshare.get(gr, 0.0) + g
        result["heads"][head] = {
            "members_used": used,
            "top_features": [{"name": nm, "group": group_of(nm, rules, default), "share": round(g, 5)} for nm, g in feats[:top_n]],
            "by_group": [{"group": g, "share": round(s, 5)} for g, s in sorted(gshare.items(), key=lambda x: -x[1])],
            "n_features_used": int(sum(1 for _, g in feats if g > 0)),
            "unclassified_share": round(gshare.get(default, 0.0), 5),
        }
    result["note"] = "LightGBM 系メンバーのゲイン重要度（各メンバーの合計を 1 に揃えて平均）。XGBoost・NN・線形は含めない。群は列名の接頭辞からの分類（config/model_feature_groups.json）。重要度は『予測への効き方』で、因果ではない。"
    return result if result["heads"] else None


# ───────────────────────── 付随情報 ─────────────────────────

def bundle_meta(bundle_rel: str | None) -> dict[str, Any]:
    if not bundle_rel:
        return {}
    bdir = TP / bundle_rel
    meta: dict[str, Any] = {"bundle": bundle_rel}
    fi = bdir / "fit_info.json"
    if fi.exists():
        try:
            j = json.loads(fi.read_text(encoding="utf-8"))
            meta.update({k: j.get(k) for k in ("fit_end", "train_rows", "seeds", "n_features", "members", "groups", "ens", "ens_source", "seconds")})
            meta["fit_info_mtime"] = datetime.fromtimestamp(fi.stat().st_mtime, JST).isoformat(timespec="minutes")
        except (OSError, ValueError):
            pass
    return meta


def mlflow_runs() -> dict[str, Any]:
    """MLflow カタログのモデルごとに、直近の run の指標（取れなければ空）。"""
    out: dict[str, Any] = {}
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        from src.pipeline.mlflow.catalog import MODEL_CATALOG

        uri = os.environ.get("MLFLOW_TRACKING_URI") or ""
        if not uri:
            db = REPO_ROOT / "mlflow" / "runs" / "mlflow.db"
            uri = f"sqlite:///{db}" if db.exists() else ""
        if not uri:
            return {}
        mlflow.set_tracking_uri(uri)
        cli = MlflowClient()
        for key, spec in MODEL_CATALOG.items():
            exp = cli.get_experiment_by_name(spec.experiment_name)
            runs = []
            if exp is not None:
                for r in cli.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=30):
                    mets = {k: _f(v) for k, v in (r.data.metrics or {}).items()}
                    runs.append({"run": r.info.run_id[:8], "start": datetime.fromtimestamp(r.info.start_time / 1000, JST).isoformat(timespec="minutes"),
                                 "status": r.info.status, "metrics": {k: v for k, v in list(mets.items())[:12] if v is not None}})
            out[key] = {"title": spec.title, "description": spec.description, "lifecycle": spec.lifecycle.value, "registered_name": spec.registered_name,
                        "experiment": spec.experiment_name, "runs": list(reversed(runs))}
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"MLflow を読めませんでした: {type(exc).__name__}: {exc}\n")
    return out


# ───────────────────────── 組み立て ─────────────────────────

def build_all(models: list[str], *, importance: bool = True, prod: bool = True) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranks = _load_ranks() if prod else pd.DataFrame()
    shadow_map: dict[str, str] = {}
    try:
        shadow_map = json.loads((INF / "predictions_backfill" / "_shadow_models.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    index: list[dict] = []
    for key in models:
        spec = ENSEMBLES[key]
        entry: dict[str, Any] = {"key": key, "label": spec["label"], "kind": "ensemble", "built_at": _now(), "meta": bundle_meta(spec["bundle"])}
        entry["meta"]["mlflow_name"] = spec["mlflow"]
        entry["test2025"] = build_test2025(spec["eval"])
        entry["eval_series"] = build_eval_series(key)
        entry["prod_series"], entry["prod_sources"] = [], []
        if prod and spec["backfill"]:
            root = INF / "predictions_backfill" / spec["backfill"]
            entry["prod_series"] = build_prod_series([(f"predictions_backfill/{spec['backfill']}", root)], ranks)
            entry["prod_sources"] = [f"predictions_backfill/{spec['backfill']}"]
        entry["prod_total"] = prod_total(entry["prod_series"])
        entry["importance"] = None
        if importance and spec["bundle"]:
            try:
                entry["importance"] = build_importance(spec["bundle"])
            except Exception as exc:  # noqa: BLE001
                entry["importance_error"] = f"{type(exc).__name__}: {exc}"[:200]
        (OUT_DIR / f"{key}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        index.append({"key": key, "label": spec["label"], "kind": "ensemble", "built_at": entry["built_at"]})
        print(f"{key}: test2025={'あり' if entry['test2025'] else 'なし'} 年別={len(entry['eval_series'])} 本番週={len(entry['prod_series'])} 重要度={'あり' if entry['importance'] else 'なし'}")
    # shadow（現行の影運用）: predictions_v3 は日付ごとにモデルが変わりうるので別系列にする
    if prod:
        entry = {"key": "shadow", "label": "shadow（predictions_v3・日付ごとの設定）", "kind": "shadow", "built_at": _now(), "meta": {"shadow_models": shadow_map},
                 "test2025": None, "eval_series": [], "importance": None,
                 "prod_series": build_prod_series([("predictions_v3", INF / "predictions_v3")], ranks), "prod_sources": ["predictions_v3"]}
        entry["prod_total"] = prod_total(entry["prod_series"])
        (OUT_DIR / "shadow.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        index.append({"key": "shadow", "label": entry["label"], "kind": "shadow", "built_at": entry["built_at"]})
        print(f"shadow: 本番週={len(entry['prod_series'])}")
    ml = mlflow_runs()
    (OUT_DIR / "mlflow.json").write_text(json.dumps({"built_at": _now(), "models": ml}, ensure_ascii=False, indent=1), encoding="utf-8")
    idx = {"built_at": _now(), "models": index, "mlflow_models": list(ml)}
    (OUT_DIR / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    return idx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--models", default=",".join(ENSEMBLES))
    b.add_argument("--no-importance", action="store_true")
    b.add_argument("--no-prod", action="store_true")
    sub.add_parser("show")
    a = ap.parse_args(argv)
    if a.cmd == "show":
        p = OUT_DIR / "index.json"
        print(p.read_text(encoding="utf-8") if p.exists() else "未計算です")
        return 0
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ENSEMBLES]
    if bad:
        print(f"不明なモデル: {bad}（選べるもの: {list(ENSEMBLES)}）", file=sys.stderr)
        return 2
    build_all(models, importance=not a.no_importance, prod=not a.no_prod)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
