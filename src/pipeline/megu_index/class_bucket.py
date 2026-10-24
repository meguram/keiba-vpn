"""めぐ指数 par_time 用クラスバケット。"""

from __future__ import annotations

from src.pipeline.megu_index.career_prize import classify_grade

# par_time セル分割用（6+1 段階。'' はクラス横断プール）
PAR_CLASS_BUCKETS = ("未勝利", "1勝", "2勝", "3勝", "OP", "重賞", "G1", "")

_GRADE_TO_BUCKET: dict[str, str] = {
    "新馬": "未勝利",
    "未勝利": "未勝利",
    "1勝": "1勝",
    "2勝": "2勝",
    "3勝": "3勝",
    "L": "OP",
    "OP": "OP",
    "G3": "重賞",
    "G2": "重賞",
    "G1": "G1",
}


def par_class_bucket(
    grade: str | None,
    race_name: str | None = None,
    race_class: str | None = None,
) -> str:
    """レースの公式クラスから par_time 用バケットを返す。"""
    g = classify_grade(grade, race_name, race_class)
    return _GRADE_TO_BUCKET.get(g, "OP")
