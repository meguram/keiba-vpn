"""分析モジュールの登録。画面のタブはこの順で並ぶ。新しい分析は、モジュールを作ってここに足す。"""

from __future__ import annotations

from src.research.race_analysis.course_stats import GATE, JOCKEY, SIRE, SIRE_LINE, CourseStatsModule
from src.research.race_analysis.past_position import PastPositionModule

MODULES = (
    CourseStatsModule(JOCKEY),
    CourseStatsModule(GATE),
    CourseStatsModule(SIRE),
    CourseStatsModule(SIRE_LINE),
    PastPositionModule(),
)
