#!/usr/bin/env bash
# T-029: T-45 の予測出力に win_prob / place_prob / show_prob が入ること
# 任意: RACE_ID=<GCS に race_shutuba がある未来/当日のレースID> で実データの出馬表を使って確認（GCS 読み取りのみ・保存しない）
source "$(dirname "$0")/_lib.sh"
TITLE="T-029 T-45 予測出力の確率フィールド"

echo "$TITLE"
check "pytest: 開催日ワークフロー一式（疑似アンサンブル・疑似ビルダー・メモリ上の保存先）" \
  python3 -m pytest tests/pipeline/test_race_day_workflow.py -q -p no:cacheprovider

if [ -z "${RACE_ID:-}" ]; then
  skip "実データ確認（RACE_ID 未指定。例: RACE_ID=202606010101 bash $0）"
else
  py_check "実出馬表で predict_race(persist=False) の全頭に win/place/show がある" "$RACE_ID" "$TMP_DIR" <<'PY'
import sys
from pathlib import Path
from src.pipeline.features.pseudo_builder import PseudoFeatureBuilder
from src.pipeline.inference import race_day_workflow as wf
from src.pipeline.models.ensemble_predictor import EnsemblePredictor
from src.scripts.maintenance.make_pseudo_ensemble import train_pseudo_ensemble
from src.scraper.storage import HybridStorage

rid, tmp = sys.argv[1], Path(sys.argv[2])
names = train_pseudo_ensemble(tmp / "pseudo", 40, n_rows=400)
res = wf.predict_race(rid, HybridStorage(), builder=PseudoFeatureBuilder(n_features=40),
                      predictor=EnsemblePredictor.load(tmp / "pseudo"), persist=False)
if res.get("status") != "success":
    print("         ", res.get("status"), res.get("error")); sys.exit(2)
preds = res["predictions"]
need = ("win_prob", "place_prob", "show_prob", "pred_score", "normalized_score")
missing = [p["horse_number"] for p in preds if any(k not in p for k in need)]
print(f"         頭数={len(preds)} 欠落馬番={missing} 保存={res.get('persisted')}")
print(f"         place合計={sum(p['place_prob'] for p in preds):.3f}(≈2) show合計={sum(p['show_prob'] for p in preds):.3f}(≈3)")
ok = not missing and res.get("persisted") is False and abs(sum(p["show_prob"] for p in preds) - 3) < 0.1
sys.exit(0 if ok else 1)
PY
fi
finish
