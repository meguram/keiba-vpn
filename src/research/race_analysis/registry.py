"""分析モジュールの登録。画面のタブはこの順で並ぶ。新しい分析は、モジュールを作ってここに足す。"""

from __future__ import annotations

from src.research.race_analysis.course_stats import GATE, JOCKEY, SIRE, SIRE_LINE, Block, CourseStatsModule
from src.research.race_analysis.past_position import PastPositionModule
from src.research.race_analysis.running_map_module import RunningMapModule

MODULES = (
    CourseStatsModule(JOCKEY),
    CourseStatsModule(GATE),
    CourseStatsModule(
        SIRE,
        (Block(SIRE, "course", "sire", "種牡馬"), Block(SIRE_LINE, "course", "line", "系統", extra_text="系統名", needs_lines=True)),
        description="出走馬の父（種牡馬）と、その父の系統（サンデーサイレンス系など）の産駒が、今回と同じコースでこれまで残してきた成績。系統の群は、netkeiba の『○○系』の取得がすべて終わってから表示されます。",
    ),
    PastPositionModule(),
    RunningMapModule(),
)
