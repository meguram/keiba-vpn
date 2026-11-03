"""めぐコラムの主な結論を、集計に使っていない 2026 年（1〜10 月）のデータで確かめ、各記事に「2026年のデータで確かめる」の節を書き込む。

記事の集計は 2020〜2025 年。同じデータで見つけた差は、偶然の上振れを含むことがある。そこで記事の中心の数字（オッズをそろえた差）を
2026 年だけで同じ式で推定し直し、向き・大きさ・95% 区間を並べる。前走をたどる記事は 2025 年から読み、結果は 2026 年の出走だけで見る。

対象（記事 → 確かめる数字）:
  summer-mare        混合戦の 牝 × 夏（6〜8月）の勝率の上積み（オッズ・年齢区分をそろえる）
  first-dirt         初ダート − 通常 の複勝率・勝率（オッズ・コースをそろえる）
  first-blinkers     初ブリンカー − 装着なし の複勝率（芝・ダート別。オッズ・コースをそろえる）
  transport-vs-stay  関西 3 場で 関東馬 − 関西馬 の複勝率（オッズ・コースをそろえる）

使い方:
    python -m src.research.columns.holdout_2026            # 集計して各記事に書き込む
    python -m src.research.columns.holdout_2026 --dry-run
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import src.research.columns.compare_series as cs

REPO = Path(__file__).resolve().parents[3]
COLS = REPO / "frontend/content/columns"
YEAR = 2026
HEADING = "2026年のデータで確かめる"


@contextmanager
def years(rng: range, *modules):
    """compare_series.YEARS（と、それを名前で取り込んだモジュールの YEARS）を一時的に差し替える。"""
    mods = [cs, *modules]
    saved = [m.YEARS for m in mods]
    try:
        for m in mods:
            m.YEARS = rng
        yield
    finally:
        for m, s in zip(mods, saved):
            m.YEARS = s


def fit(formula: str, d: pd.DataFrame, term: str) -> dict[str, float]:
    from src.research.columns.race_context import ols_fit

    f = ols_fit(formula, d)
    e, se = float(f.params[term]), float(f.bse[term])
    return {"e": e, "lo": e - 1.96 * se, "hi": e + 1.96 * se, "p": float(f.pvalues[term]), "n": int(len(d))}


def pt(v: float) -> str:
    return f"{v * 100:+.1f}pt"


def pfmt(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def line(label: str, base: dict[str, float], new: dict[str, float]) -> str:
    return (f"・{label}: 2020〜2025年 {pt(base['e'])}（{pfmt(base['p'])}）→ {YEAR}年 {pt(new['e'])}"
            f"（95% 区間 {pt(new['lo'])}〜{pt(new['hi'])}、{pfmt(new['p'])}）")


def verdict(base: dict[str, float], new: dict[str, float]) -> str:
    if base["p"] >= 0.05:
        return ("元の集計でも差が無く、2026 年も差は見られません。" if new["p"] >= 0.05
                else "元の集計では差が無かったのに 2026 年だけ差が出ており、偶然の可能性が高いと見ています。")
    same = (base["e"] > 0) == (new["e"] > 0)
    if same and new["p"] < 0.05:
        return "同じ向きで、2026 年だけでも偶然とは言いにくい差です。"
    if same and new["lo"] <= base["e"] <= new["hi"]:
        return "同じ向きで、元の大きさも 2026 年の区間に入っています（2026 年だけでは標本が少なく、有意ではありません）。"
    if same:
        return "向きは同じですが、2026 年は小さく、元の大きさは区間の外です。"
    return "2026 年は逆の向きで、元の差が続いているとは言えません。"


# ── 記事ごとの確認 ───────────────────────────────


def summer_mare() -> tuple[str, list[str]]:
    def run(d: pd.DataFrame) -> dict[str, float]:
        x = d[d["sex"].isin(["牝", "牡"]) & d["mixed_sex"]].copy()
        x["mare"] = (x["sex"] == "牝").astype(float)
        x["summer"] = x["month"].isin([6, 7, 8]).astype(float)
        x["lq"] = np.log(x["odds"])
        x["ag"] = np.where(x["age"] == 2, "2歳", np.where(x["age"] == 3, "3歳", "4歳以上"))
        return fit("win ~ mare*summer + lq + I(lq**2) + C(ag)", x, "mare:summer")

    base = run(cs.load())
    with years(range(YEAR, YEAR + 1)):
        new = run(cs.load())
    return "summer-mare", [line("混合戦の牝馬の、夏（6〜8月）の勝率の上積み（オッズ・年齢をそろえる）", base, new), verdict(base, new)]


def first_dirt() -> tuple[str, list[str]]:
    import src.research.columns.first_dirt as fd
    import src.research.columns.race_context as rc

    def run(x: pd.DataFrame, m: str) -> dict[str, float]:
        return fit(f"{m} ~ fd + {fd.ADJ}", x, "fd")

    b = fd.frame()
    base = {m: run(b, m) for m in ("top3", "win")}
    with years(range(YEAR - 1, YEAR + 1), rc):
        x = fd.frame()
    x = x[pd.to_datetime(x["date"]).dt.year == YEAR]
    new = {m: run(x, m) for m in ("top3", "win")}
    return "first-dirt", [line("初ダートの複勝率（通常のダート出走との差、オッズ・コースをそろえる）", base["top3"], new["top3"]), verdict(base["top3"], new["top3"]),
                          line("初ダートの勝率（同じ）", base["win"], new["win"]), verdict(base["win"], new["win"])]


def first_blinkers() -> tuple[str, list[str]]:
    import src.research.columns.first_blinkers as fb

    def run(z: pd.DataFrame, sf: str) -> dict[str, float]:
        w = z[(z["surface"] == sf) & z["grp"].isin(["初装着", "装着なし"])].copy()
        w["fb"] = (w["grp"] == "初装着").astype(float)
        return fit("top3 ~ fb + lq + I(lq**2) + C(key)", w, "fb")

    b = fb.frame()
    base = {sf: run(b, sf) for sf in ("芝", "ダート")}
    with years(range(YEAR - 1, YEAR + 1), fb):
        x = fb.frame()
    x = x[pd.to_datetime(x["date"]).dt.year == YEAR]
    out = []
    for sf in ("芝", "ダート"):
        new = run(x, sf)
        out += [line(f"{sf}の初ブリンカーの複勝率（装着なしとの差、オッズ・コースをそろえる）", base[sf], new), verdict(base[sf], new)]
    return "first-blinkers", out


def transport() -> tuple[str, list[str]]:
    import src.research.columns.transport_vs_stay as tv

    def run(t: pd.DataFrame) -> dict[str, float]:
        w = t[t["venue"].isin(("京都", "阪神", "中京"))].assign(away=lambda d: (d["home"] == "MI").astype(float))
        return fit("top3 ~ away + lq + I(lq**2) + C(key)", w, "away")

    base = run(tv.frame())
    with years(range(YEAR - 1, YEAR + 1), tv):
        x = tv.frame()
    x = x[pd.to_datetime(x["date"]).dt.year == YEAR]
    new = run(x)
    return "transport-vs-stay", [line("関西 3 場（京都・阪神・中京）での 関東馬 − 関西馬 の複勝率（オッズ・コースをそろえる）", base, new), verdict(base, new)]


CHECKS: list[Callable[[], tuple[str, list[str]]]] = [summer_mare, first_dirt, first_blinkers, transport]


# ── 記事への書き込み ─────────────────────────────


def splice(col: dict[str, Any], lines: list[str]) -> dict[str, Any]:
    """「2026年のデータで確かめる」の節（見出し＋本文）を置き換える。無ければ最初の折り畳みの前（無ければ末尾）に足す。"""
    body = [b for b in col["body"]]
    i = next((k for k, b in enumerate(body) if b.get("type") == "heading" and b.get("text") == HEADING), None)
    if i is not None:
        del body[i:i + 2]
    at = next((k for k, b in enumerate(body) if b.get("type") == "accordion"), len(body))
    text = ("記事の数字は 2020〜2025 年で集計しています。同じデータで見つけた差は偶然の上振れを含むことがあるので、"
            f"集計に使っていない {YEAR} 年（1〜10 月）だけで同じ式を当て直しました。\n\n" + "\n".join(lines)
            + f"\n\n{YEAR} 年は 10 か月分で標本が少なく、区間は広めです。")
    body[at:at] = [{"type": "heading", "text": HEADING}, {"type": "text", "text": text}]
    return {**col, "body": body, "updated": "2026-11-03"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="記事の slug（省略時は全部）")
    args = ap.parse_args()
    for check in CHECKS:
        slug, lines = check()
        if args.only and slug not in args.only:
            continue
        print(f"[{slug}]\n  " + "\n  ".join(lines))
        if args.dry_run:
            continue
        path = COLS / f"{slug}.json"
        col = splice(json.loads(path.read_text(encoding="utf-8")), lines)
        text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
        json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
