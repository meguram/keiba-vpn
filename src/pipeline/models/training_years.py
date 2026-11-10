"""学習に使う年（2026 年は不完全なため使わない）。pandas などを読み込まずに参照できるよう、定数だけを置く。"""

import os

# 起点年。2016〜 の表で学び直した版へ切り替えるまで、本番（v6e）の学習と同じ 2020 を既定にする（切替は配備の段 18）。
FIRST_YEAR: int = int(os.environ.get("KEIBA_FIRST_YEAR") or 2020)
TRAINING_YEARS: tuple[str, ...] = tuple(str(y) for y in range(FIRST_YEAR, 2026))
