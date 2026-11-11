"""分析モジュールの登録。画面のタブはこの順で並ぶ。新しい分析は、モジュールを作ってここに足す。"""

from __future__ import annotations

from src.research.race_analysis.course_stats import GATE, JOCKEY, SIRE, CourseStatsModule
from src.research.race_analysis.past_position import PastPositionModule
from src.research.race_analysis.running_map_module import RunningMapModule
from src.research.race_analysis.trainer_pattern import TrainerPatternModule

MODULES = (
    CourseStatsModule(JOCKEY),
    CourseStatsModule(GATE),
    CourseStatsModule(SIRE),
    PastPositionModule(),
    RunningMapModule(),
    TrainerPatternModule(),
)
