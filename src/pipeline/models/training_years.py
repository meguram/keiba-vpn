"""学習に使う年（2026 年は不完全なため使わない）。pandas などを読み込まずに参照できるよう、定数だけを置く。"""

TRAINING_YEARS: tuple[str, ...] = tuple(str(y) for y in range(2020, 2026))
