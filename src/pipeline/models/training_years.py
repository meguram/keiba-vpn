"""学習に使う年（2026 年は不完全なため使わない）と、推論で読む履歴の年。pandas などを読み込まずに参照できるよう、定数と小さな関数だけを置く。"""

import os

# 起点年。2016〜 の表で学び直した版へ切り替えるまで、本番（v6e）の学習と同じ 2020 を既定にする（切替は配備の段 18）。
FIRST_YEAR: int = int(os.environ.get("KEIBA_FIRST_YEAR") or 2020)
TRAINING_YEARS: tuple[str, ...] = tuple(str(y) for y in range(FIRST_YEAR, 2026))


def serving_years(tables_dir, need: str | None = None) -> list[int]:
    """推論で読む履歴の年: 起点年（FIRST_YEAR。本番のバンドルの学習の起点と同じ）以降で、tables_dir に年のフォルダがある年。
    学習年より後の年（2026 年など）も含む（その日より前の結果は全部数える、という学習時の作りに合わせる）。
    起点年より前の表（2016〜2019 年など）は読まない。読むと通算の数が学習時より大きくなる（2026-11-07〜08 に本番で起きた）。
    need を渡すと、その名前のファイルがある年だけにする。該当する年が無ければ FileNotFoundError（ミラーの欠け・KEIBA_FIRST_YEAR の設定ミス）。"""
    from pathlib import Path

    base = Path(tables_dir)
    ys = sorted(int(p.name) for p in base.iterdir()
                if p.is_dir() and p.name.isdigit() and int(p.name) >= FIRST_YEAR and (need is None or (p / need).exists()))
    if not ys:
        raise FileNotFoundError(f"推論で読む履歴の年が無い: {base} に {FIRST_YEAR} 年以降の年のフォルダ{'（' + need + ' あり）' if need else ''}が無い（KEIBA_FIRST_YEAR={FIRST_YEAR}）")
    return ys
