"""best_analysis が running_map.py の AXES・MAP_KEYS・WET_AXIS を文字列として読めること（書き方を変えると空になり、ページから軸が消える）。"""
from __future__ import annotations

from src.research.race import running_map as rm
from src.scripts.docs import best_analysis as BA


def test_軸は走法マップのコードと同じ並びで読める():
    mapped, wet, shown_only = BA._running_axes_from_code()
    assert mapped and wet and shown_only
    names = {a[0]: a[3] for a in rm.AXES}
    assert mapped == [names[k] for k in rm.MAP_KEYS]
    assert wet == sorted({names[k] for k in rm.WET_AXIS.values()})
    assert sorted(shown_only) == sorted(names[k] for k in names if k not in rm.MAP_KEYS and k not in rm.WET_AXIS.values())


def test_好走距離レンジは軸にも色分けの選択肢にも入らない():
    mapped, wet, shown_only = BA._running_axes_from_code()
    assert rm.GD_KEY not in [a[0] for a in rm.AXES]
    assert not any("好走距離" in n for n in mapped + wet + shown_only)
