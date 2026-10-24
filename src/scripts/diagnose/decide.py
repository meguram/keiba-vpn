"""【開発PC で実行】学習PC・VPS の調査レポートを読み、設計上の判断と次の作業を出す。

レポートが無い項目は「未判定」とし、どの環境でどのコマンドを実行すべきかを示す。
判断ルールは関数ごとに分かれており、しきい値は ``THRESHOLDS`` で調整できる。

  python -m src.scripts.diagnose.decide --training reports/training_pc.json --vps reports/vps.json \
      --labels reports/dynamic_feature_labels.csv --out reports/decision.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.scripts.diagnose.common import check_values, load_report

THRESHOLDS = {
    "inference_margin_ok_mb": 300,      # 空き - ピーク がこれ以上なら余裕あり
    "inference_margin_min_mb": 100,     # これ未満ならGCPへ切替
    "swap_min_mb": 1024,
    "row_read_sec_per_race_max": 3.0,   # 1000列換算の行読み込み秒/レース
    "gcs_objects_per_page": 5,          # 1ページ表示で読むGCSオブジェクト数の想定
    "page_gcs_budget_ms": 500,
    "cloud_sql_median_max_ms": 50,
    "model_mb_large": 500,
    "dynamic_share_high": 0.10,
    "egress_usd_per_gb": 0.12,
}

VERDICTS = ("go", "caution", "switch", "unknown")
RUN_HINT = {
    "training_pc": "学習PCで `python -m src.scripts.diagnose.diagnose_training_pc`",
    "vps": "VPSで `python -m src.scripts.diagnose.diagnose_vps`（--netkeiba-trial 付き）",
}


def _decision(id_: str, title: str, verdict: str, summary: str, actions: list[str] | None = None, evidence: dict | None = None) -> dict:
    assert verdict in VERDICTS
    return {"id": id_, "title": title, "verdict": verdict, "summary": summary, "actions": actions or [], "evidence": evidence or {}}


def _unknown(id_: str, title: str, need: str) -> dict:
    return _decision(id_, title, "unknown", f"判断材料が未取得: {RUN_HINT[need]} を実行してレポートを持ち帰る", [RUN_HINT[need]])


def _mm(v: str) -> str:
    return ".".join(v.split(".")[:2])


def decide_inference_location(training: dict | None, vps: dict | None, t: dict = THRESHOLDS) -> dict:
    title = "T-45推論の実行場所（VPS or GCP）"
    if not vps:
        return _unknown("inference_location", title, "vps")
    sysv, inf = check_values(vps, "system"), check_values(vps, "inference")
    avail, peak = sysv.get("mem_available_mb"), inf.get("peak_rss_mb")
    if avail is None or peak is None:
        return _unknown("inference_location", title, "vps")
    margin = avail - peak
    ev = {"mem_available_mb": avail, "inference_peak_mb": peak, "margin_mb": margin, "swap_mb": sysv.get("swap_total_mb"), "model": inf.get("model_label")}
    actions: list[str] = []
    if (sysv.get("swap_total_mb") or 0) < t["swap_min_mb"]:
        actions.append("スワップを2GB用意する（OOM回避の保険）")
    if margin >= t["inference_margin_ok_mb"]:
        return _decision("inference_location", title, "go", f"VPSで実行可（空き{avail}MB − ピーク{peak}MB = 余裕{margin}MB）", actions, ev)
    if margin >= t["inference_margin_min_mb"]:
        actions += ["推論を `docker run --memory` で上限付き・`nice` で低優先度にする", "開催日は推論中に重いバッチを止める"]
        return _decision("inference_location", title, "caution", f"VPSで実行可だが余裕が小さい（余裕{margin}MB）", actions, ev)
    actions += ["推論をGCP（Cloud Tasks + Cloud Run）に切り替える: `predict_race_day --mode enqueue`"]
    return _decision("inference_location", title, "switch", f"VPSのメモリが不足（余裕{margin}MB）→ GCPで実行", actions, ev)


def decide_feature_read(training: dict | None, t: dict = THRESHOLDS) -> dict:
    title = "推論時の特徴量の読み方"
    if not training:
        return _unknown("feature_read", title, "training_pc")
    rr, lay = check_values(training, "feature_row_read"), check_values(training, "parquet_layout")
    if not rr:
        return _unknown("feature_read", title, "training_pc")
    est = rr["est_sec_per_race_1000cols"]
    ev = {**rr, **lay}
    actions = []
    if lay and lay.get("row_groups_median", 2) <= 1:
        actions.append("parquet を race_id 順に並べ替え、行グループ分割(例: 1万行)して書き直す（行フィルタを効かせる）")
    if est <= t["row_read_sec_per_race_max"]:
        return _decision("feature_read", title, "go", f"`FeatureStore.load_rows_for_keys` で足りる（1000列換算 {est}s/レース、全読みの{rr['speedup']}倍速）", actions, ev)
    actions.append("推論用スナップショット（レース単位の1ファイル）を前日に作っておき、T-45はそれを読むだけにする")
    return _decision("feature_read", title, "caution", f"行フィルタ読み込みでも遅い（1000列換算 {est}s/レース）→ レース単位スナップショットを採用", actions, ev)


def decide_library_alignment(training: dict | None, vps: dict | None) -> dict:
    title = "ライブラリ版の一致（学習PC ↔ VPS）"
    if not training or not vps:
        return _unknown("library_alignment", title, "vps" if training else "training_pc")
    a, b = check_values(training, "environment").get("libraries"), check_values(vps, "system").get("libraries")
    if not a or not b:
        return _unknown("library_alignment", title, "vps")
    diffs = [f"{k}: 学習PC {a[k]} / VPS {b.get(k, '未インストール')}" for k in a if _mm(a[k]) != _mm(b.get(k, ""))]
    if not diffs:
        return _decision("library_alignment", title, "go", "major.minor が全て一致", evidence={"libraries": a})
    return _decision("library_alignment", title, "switch", "版が不一致（モデルの読み込みが拒否される）: " + "; ".join(diffs),
                     ["requirements を学習PCの版に固定し、VPS/Dockerイメージを再ビルドする"], {"diffs": diffs})


def decide_scraping_location(vps: dict | None) -> dict:
    title = "スクレイピングをVPSで実行してよいか（案A）"
    if not vps:
        return _unknown("scraping_location", title, "vps")
    n = check_values(vps, "netkeiba")
    if not n.get("tried"):
        return _decision("scraping_location", title, "unknown", "netkeiba への試行が未実施", [RUN_HINT["vps"]])
    if n.get("blocked"):
        return _decision("scraping_location", title, "switch", f"VPSのIPはブロックの疑い（HTTP {n.get('status_code')}）→ 案B（GCP Cloud Run Jobs等）を検討",
                         ["スクレイピングを学習PC(自宅IP) or GCP に移す案を再検討する", "`docs/operations/vps-gcp-responsibilities.md` 論点A を更新"], n)
    return _decision("scraping_location", title, "go", f"VPSから取得可（HTTP {n.get('status_code')}）。案Aで確定", ["継続運用で429/403が出ないかは SCRAPE_QUEUE_STAGGER_SEC(=1.0) 等で間隔を維持"], n)


def decide_page_latency(vps: dict | None, t: dict = THRESHOLDS) -> dict:
    title = "ページ表示の遅延（GCS/Cloud SQL → VPS）"
    if not vps:
        return _unknown("page_latency", title, "vps")
    g, d = check_values(vps, "gcs_latency"), check_values(vps, "cloud_sql_latency")
    if not g.get("median_ms"):
        return _unknown("page_latency", title, "vps")
    est = g["median_ms"] * t["gcs_objects_per_page"]
    actions, verdict = [], "go"
    if est > t["page_gcs_budget_ms"]:
        verdict = "caution"
        actions += ["ページ用の集約オブジェクト（1ページ=1オブジェクト）を作り、予測後にVPSキャッシュへ先読みする"]
    if d.get("median_ms", 0) > t["cloud_sql_median_max_ms"]:
        verdict = "caution"
        actions.append(f"Cloud SQL 中央{d['median_ms']}ms: 一覧・検索結果はVPSのRedisにキャッシュする")
    msg = f"GCS 1件 中央{g['median_ms']}ms → 1ページ{t['gcs_objects_per_page']}件で約{est}ms（予算{t['page_gcs_budget_ms']}ms）"
    return _decision("page_latency", title, verdict, msg, actions, {**g, "cloud_sql": d, "est_page_ms": est})


def decide_model_size(training: dict | None, vps: dict | None, t: dict = THRESHOLDS) -> dict:
    title = "学習済みモデルの容量と配布"
    if not training:
        return _unknown("model_size", title, "training_pc")
    m = check_values(training, "model")
    if not m:
        return _unknown("model_size", title, "training_pc")
    mb = m["total_mb"]
    cost = mb / 1024 * t["egress_usd_per_gb"]
    actions = ["モデルは版ごとにVPSへキャッシュし、同じ版を再ダウンロードしない（`KEIBA_MODEL_CACHE_DIR`）"]
    disk = check_values(vps, "system").get("disk_free_gb") if vps else None
    if disk is not None and mb / 1024 * 3 > disk:
        actions.append(f"VPSのディスク空き{disk}GBに対し世代保持が重い: 旧版を削除する")
    if mb > t["model_mb_large"]:
        actions.append("モデルが大きい: 木の本数削減・MLP縮小・不要なベースモデルの除外で精度と容量の兼ね合いを検討")
        verdict = "caution"
    else:
        verdict = "go"
    return _decision("model_size", title, verdict, f"合計{mb}MB（1回の配布で送信料金 約${cost:.3f}）。ロード{m.get('load_sec')}s", actions, m)


def read_labels(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def decide_dynamic_features(rows: list[dict] | None, builder: dict | None = None, t: dict = THRESHOLDS) -> dict:
    title = "動的特徴量（T-45時点で初めて決まる特徴量）の割合"
    if not rows:
        return _decision("dynamic_features", title, "unknown", "ラベルCSV未提出: 学習PCで出力した CSV の label 列に dynamic/static を記入して持ち帰る", [RUN_HINT["training_pc"]])
    labeled = [r for r in rows if (r.get("label") or "").strip() in ("dynamic", "static")]
    src = labeled if labeled else [{"label": r.get("guess", "")} for r in rows]
    n_dyn = sum(1 for r in src if r["label"] == "dynamic")
    share = n_dyn / len(src)
    unl = len(rows) - len(labeled)
    note = f"（ラベル未記入{unl}件は名前からの推定値を使用）" if unl else ""
    ev = {"total": len(rows), "dynamic": n_dyn, "share": round(share, 3), "unlabeled": unl}
    if share <= t["dynamic_share_high"]:
        return _decision("dynamic_features", title, "go", f"動的 {n_dyn}/{len(src)}列（{share:.0%}）{note}。静的部分は前日に事前計算して保存し、T-45は動的列だけ計算する",
                         ["静的特徴量のレース単位スナップショットを前日に生成するジョブを追加", "T-45は動的列のみ計算してスナップショットと結合"], ev)
    return _decision("dynamic_features", title, "caution", f"動的 {n_dyn}/{len(src)}列（{share:.0%}）{note}。T-45での計算量が大きい",
                     ["動的列の計算時間を実測し、T-45前(T-60など)に途中まで計算して、オッズ・馬体重だけ最後に反映する二段構えを検討"], ev)


def decide_builder(training: dict | None) -> dict:
    title = "特徴量ビルダー"
    if not training:
        return _unknown("builder", title, "training_pc")
    b = check_values(training, "builder") or next((c.get("values") for c in training["checks"] if c["id"] == "builder"), {})
    if b.get("is_pseudo") or not b.get("implemented", True):
        return _decision("builder", title, "caution", "本番の特徴量ビルダーが未実装（疑似ビルダーのみ）",
                         ["増分計算(全件再計算しない)の本ビルダーを実装し `get_feature_builder` に登録", "学習時と同じ入力処理（FeatureStore）を共有する"], b)
    return _decision("builder", title, "go", f"ビルダー={b.get('builder')}", evidence=b)


def decide_all(training: dict | None, vps: dict | None, labels: list[dict] | None) -> list[dict]:
    return [
        decide_inference_location(training, vps),
        decide_scraping_location(vps),
        decide_library_alignment(training, vps),
        decide_feature_read(training),
        decide_page_latency(vps),
        decide_model_size(training, vps),
        decide_dynamic_features(labels),
        decide_builder(training),
    ]


_MARK = {"go": "確定", "caution": "要対応", "switch": "変更", "unknown": "未判定"}


def render_markdown(decisions: list[dict], training: dict | None, vps: dict | None) -> str:
    lines = ["# 調査結果にもとづく判断", ""]
    for name, r in (("学習PC", training), ("VPS", vps)):
        lines.append(f"- {name}レポート: " + (f"{r['host']}（{r['created_at']}）" if r else "未提出"))
    lines += ["", "| 判断 | 結果 | 内容 |", "|---|---|---|"]
    for d in decisions:
        lines.append(f"| {d['title']} | **{_MARK[d['verdict']]}** | {d['summary']} |")
    todo = [(d["title"], a) for d in decisions for a in d["actions"]]
    if todo:
        lines += ["", "## 次の作業", ""]
        seen = set()
        for title, a in todo:
            if a not in seen:
                seen.add(a)
                lines.append(f"- [ ] {a}（{title}）")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training", help="学習PCのレポート(JSON)")
    ap.add_argument("--vps", help="VPSのレポート(JSON)")
    ap.add_argument("--labels", help="動的/静的ラベルCSV")
    ap.add_argument("--out", default="reports/decision.md")
    args = ap.parse_args(argv)

    def _load(p: Any):
        return load_report(p) if p and Path(p).is_file() else None

    training, vps = _load(args.training), _load(args.vps)
    labels = read_labels(args.labels) if args.labels and Path(args.labels).is_file() else None
    md = render_markdown(decide_all(training, vps, labels), training, vps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
