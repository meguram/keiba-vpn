"""分析モジュールの登録。画面にはこの順で並ぶ。新しい分析は、モジュールを作ってここに足す。"""

from __future__ import annotations

from src.research.race_analysis.jockey_course import JockeyCourseModule

MODULES = (
    JockeyCourseModule(),
)
