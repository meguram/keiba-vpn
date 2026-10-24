#!/usr/bin/env bash
# T-013: 連対率・複勝率が Harville 式（固定倍率ではない）で出ること／計算が速いこと
# 任意: RACE_ID=<GCS に予測キャッシュがあるレースID> を付けると実データでも確認する（GCS 読み取りのみ）
source "$(dirname "$0")/_lib.sh"
TITLE="T-013 Harville 式による確率導出"

echo "$TITLE"
check "pytest: Harville（ベクトル化と旧実装の数値一致）" \
  python3 -m pytest tests/utils/test_race_probabilities.py -q -p no:cacheprovider
check "pytest: 推論パイプラインの place/show が Harville で導出される" \
  python3 -m pytest tests/pipeline/test_inference_pipeline_probabilities.py -q -p no:cacheprovider

py_check "18頭の複勝率計算が 1ms 未満（この環境の実測は下に表示）" <<'PY'
import sys, time
import numpy as np
from src.utils.race_probabilities import _softmax_win_probs, harville_top3_prob
w = _softmax_win_probs(np.random.default_rng(0).normal(size=18))
n = 2000
t = time.perf_counter()
for _ in range(n):
    harville_top3_prob(w)
ms = (time.perf_counter() - t) / n * 1000
print(f"         18頭 1回あたり {ms:.4f} ms（旧実装は約 3.9 ms）")
sys.exit(0 if ms < 1.0 else 1)
PY

if [ -z "${RACE_ID:-}" ]; then
  skip "実データ確認（RACE_ID 未指定。例: RACE_ID=202606010101 bash $0）"
else
  py_check "実レースの予測キャッシュで place/show の合計が 2/3 に近い" "$RACE_ID" <<'PY'
import sys
from src.pipeline.inference.race_prediction_service import load_cached
from src.pipeline.inference import inference_pipeline as ip
from src.scraper.storage import HybridStorage
rid = sys.argv[1]
cached = load_cached(HybridStorage(), rid)
if not cached or not cached.get("predictions"):
    print(f"         予測キャッシュが無い: {rid}"); sys.exit(3)
out = ip._map_stage1_to_spec(cached, model_version="verify")["horses"]
w = sum(h["win_prob"] for h in out); p = sum(h["place_prob"] for h in out); s = sum(h["show_prob"] for h in out)
print(f"         頭数={len(out)} 勝率合計={w:.3f} 連対合計={p:.3f}(≈2) 複勝合計={s:.3f}(≈3)")
for h in sorted(out, key=lambda x: -x["win_prob"])[:5]:
    print(f"         馬番{h['post_no']}: win={h['win_prob']:.3f} place={h['place_prob']:.3f} show={h['show_prob']:.3f}")
ok = abs(w - 1) < 0.02 and abs(p - 2) < 0.05 and abs(s - 3) < 0.08 and all(0 <= h["win_prob"] <= h["place_prob"] <= h["show_prob"] <= 1 for h in out)
sys.exit(0 if ok else 1)
PY
fi
finish
