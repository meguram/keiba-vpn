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


# ---- 起点年の記録と照合（B7。計画: progress/ledger_plan_20261118.md §7） ----
# KEIBA_FIRST_YEAR を付け忘れると、2016 年からの表でも起点年 2020 として分割・履歴が作られる。
# 学習の束に起点年を書き（first_year_stamp）、check_first_year で実行時の FIRST_YEAR と照合する。
# 照合を呼ぶ所（学習・評価の道具、推論の EnsembleBundle.load）は担当と相談して入れる。推論の経路は 11/23 の開催の後（docs/operations/data_access_ledger.md）。
FIRST_YEAR_SOURCE: str = "env" if os.environ.get("KEIBA_FIRST_YEAR") else "default"


def first_year_stamp(train_year_min: int | None = None) -> dict:
    """束の meta（bundle.json・fit_info.json）に書く起点年の記録。train_year_min は実際の学習の行の最小の年（設定でなくデータから）。"""
    ty = list(TRAINING_YEARS)                # 実行中に差し替えられていることがある（rebuild_base_2016.py）。その場合は食い違いを記録する
    out = {"first_year": FIRST_YEAR, "first_year_source": FIRST_YEAR_SOURCE, "training_years": [ty[0], ty[-1]] if ty else []}
    if ty and int(ty[0]) != FIRST_YEAR:
        out["first_year_mismatch"] = f"TRAINING_YEARS は {ty[0]} 年から・FIRST_YEAR は {FIRST_YEAR}（KEIBA_FIRST_YEAR を付けずに年を差し替えた）"
    if train_year_min is not None:
        out["train_year_min"] = int(train_year_min)
    return out


def bundle_first_year(bundle_dir) -> int | None:
    """束の起点年（bundle.json か fit_info.json の first_year）。書かれていない古い束は None。"""
    import json
    from pathlib import Path

    for name in ("bundle.json", "fit_info.json"):
        p = Path(bundle_dir) / name
        if p.exists():
            try:
                v = json.loads(p.read_text(encoding="utf-8")).get("first_year")
            except (OSError, ValueError):
                v = None
            if v is not None:
                return int(v)
    return None


def check_first_year(recorded: int | None, *, where: str, strict: bool = True) -> str | None:
    """記録された起点年（束・表の meta）と実行時の FIRST_YEAR を照合する。
    食い違えば strict は SystemExit、そうでなければ警告の文を返す。記録が無い（古い束）は警告の文だけ返す（止めない）。"""
    if recorded is None:
        return f"[起点年] {where}: 起点年の記録が無い（古い束）。実行時は {FIRST_YEAR}（{FIRST_YEAR_SOURCE}）"
    if int(recorded) != FIRST_YEAR:
        msg = (f"[起点年] {where}: 記録は {recorded} 年起点、実行時の FIRST_YEAR は {FIRST_YEAR}（{FIRST_YEAR_SOURCE}）。"
               f"KEIBA_FIRST_YEAR={recorded} を付けて動かす（履歴・分割の年が食い違う。2026-11-07〜08 に本番で起きた）")
        if strict:
            raise SystemExit(msg)
        return msg
    return None
