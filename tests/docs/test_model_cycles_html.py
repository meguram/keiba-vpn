"""親モデルの改善サイクルのログ（docs/html/operations/model_cycles.html）の生成のテスト。"""

from __future__ import annotations

import json
import re

from src.scripts.docs import build_model_cycles_html as b


def _cycle(cid: str, verdict: str, diff=None, phase: str = "main") -> dict:
    res = {"races": 100, "ref": None if diff is None else "c00", "metrics": {m[0]: 0.5 for m in b.METRICS},
           "ref_metrics": None if diff is None else {m[0]: 0.4 for m in b.METRICS},
           "diff": {} if diff is None else {"ll": diff, "top1_win": [0.01, -0.01, 0.03]},
           "by": {} if diff is None else {"ll_2022": diff}}
    return {"id": cid, "phase": phase, "date": "2026-11-16 03:00", "hypothesis": f"仮説<{cid}>", "change": "変更", "spec": {"a": 1},
            "result": res, "parity": "ずれ 0 件", "verdict": verdict, "reason": "理由", "next": "次"}


LOG = {"title": "親モデルの改善サイクル", "request": "依頼", "status": "中断中", "rules": ["決まり"],
       "baselines": [{"name": "本番 v6e", "ll": 1.9, "top1_win": 0.33, "top1_place": 0.64, "roi_tansho": 0.9, "roi_fukusho": 0.9, "roi_umaren": 0.9, "roi_wide": 0.9}],
       "best": {"name": "候補", "note": "メモ"}, "vs_prod": "差", "conclusion": "結論",
       "cycles": [_cycle("c00", "基準", phase="before_fix"), _cycle("c01", "不採用", [0.001, -0.002, 0.004], "before_fix"),
                  _cycle("c05", "採用", [-0.002, -0.003, -0.001])]}


def test_render_has_sections_chart_and_no_external_assets():
    h = b.render(LOG, updated="2026-11-16")
    assert "最終更新: 2026-11-16" in h
    for cid in ("c00", "c01", "c05"):
        assert f'id="{cid}"' in h
    assert h.index(b.PHASES["before_fix"]) < h.index(b.PHASES["main"])          # 直す前の参考が先
    assert "<svg" in h and h.count("<circle") == 2                                # 差の付いた 2 回だけ点を打つ
    assert "<script" not in h and "<link" not in h and 'src="http' not in h       # 外部の JS・CSS を読まない
    assert "prefers-color-scheme:dark" in h and 'name="viewport"' in h
    assert "仮説&lt;c00&gt;" in h                                                 # エスケープ
    assert "−0.0020（−0.0030〜−0.0010）" in h                                    # 差の書式（全角の負号）
    assert "+1.0pt（−1.0〜+3.0）" in h                                           # 率の差は %pt


def test_chart_handles_no_diffs():
    assert "まだ無い" in b.chart([_cycle("c00", "基準")])


def test_committed_html_matches_log():
    """記録（JSON）を足したら HTML を作り直していること（日付の行を除いて一致）。"""
    log = json.loads(b.SRC.read_text(encoding="utf-8"))
    want = b.render(log, updated="X")
    have = b.OUT.read_text(encoding="utf-8")
    strip = lambda s: re.sub(r"最終更新: [^。]*。", "", s)  # noqa: E731
    assert strip(want) == strip(have), "python -m src.scripts.docs.build_model_cycles_html で作り直す"
