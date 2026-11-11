"""めぐコラム『厩舎コメントって成績に関連する？』（trainer-comment-wordcloud）を書く。

語ごとの集計（term_stats.csv・summary.json）は src/scripts/data/trainer_comment_words.py の出力。語の選び方と配置は
src/scripts/data/trainer_comment_wordcloud.py（select・layout）と同じものを使い、画面用に座標つきの JSON にする。
原文のコメントは載せない（短い語・句の集計値だけ）。

使い方:
    python -m src.research.columns.trainer_comment_wordcloud <term_stats.csv> <summary.json> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.scripts.data.trainer_comment_wordcloud import H, W, layout, select

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/trainer-comment-wordcloud.json"
NARROW = (480, 640)
TOP_N = 6


def _items(df: pd.DataFrame, w: int, h: int, smin: float, smax: float) -> list[dict]:
    return [
        {
            "label": it["row"].label, "x": round(it["x"], 1), "y": round(it["y"], 1), "size": round(it["size"], 1),
            "n": int(it["row"].n), "rate": round(float(it["row"].good_rate), 4),
            "lift": round(float(it["row"].lift), 3), "adj": round(float(it["row"].adj_lift), 3),
        }
        for it in layout(df, smin, smax, w, h)
    ]


def article(T: pd.DataFrame, summ: dict) -> dict:
    base = float(summ["good_rate"])
    S = select(T, 110, 0.01, 150)
    S_narrow = select(T, 60, 0.01, 250)
    cloud = {
        "type": "wordcloud",
        "title": "厩舎コメントの語・句と好走率（大きさ = 出た数、色 = 好走率が全体の何倍か）",
        "base": round(base, 4),
        "wide": {"w": W, "h": H, "items": _items(S, W, H, 15.0, 84.0)},
        "narrow": {"w": NARROW[0], "h": NARROW[1], "items": _items(S_narrow, NARROW[0], NARROW[1], 14.0, 52.0)},
        "caption": "語・句にカーソルを載せる（スマホはタップ）と、出た数・3着以内率・倍率が出ます。『人気をそろえた倍率』に切り替えると、ほとんどが灰色になります。",
    }
    big = S[S["n"] >= 300]

    def rows(asc: bool) -> list[list[str]]:
        d = big.sort_values("lift", ascending=asc).head(TOP_N)
        return [[r.label, f"{int(r.n):,}", f"{r.good_rate * 100:.1f}%", f"{r.lift:.2f}", f"{r.adj_lift:.2f}"] for r in d.itertuples()]

    cols = ["語・句", "出た数", "3着以内率", "倍率", "人気をそろえた倍率"]
    n_rows = int(summ["n_rows"])
    body = [
        {"type": "stats", "items": [
            {"label": "全体と有意に差がある語・句", "value": f"{summ['n_sig_raw']} 個", "sub": f"{summ['n_terms']:,} 個のうち（そのまま比べたとき）"},
            {"label": "人気をそろえても差が残る語・句", "value": f"{summ['n_sig_adj']} 個", "sub": "単勝人気の順位が同じ馬と比べたとき"},
            {"label": "前半の年に見つけた語が後半でも同じ向き", "value": f"{summ['disc_sig_same_dir_raw_in_val'] * 100:.0f}%", "sub": "人気をそろえると " + f"{summ['disc_sig_same_dir_adj_in_val'] * 100:.0f}%"},
        ]},
        {"type": "text", "text": "厩舎コメントの言葉と成績は、そのまま見ると関係があります。ただし人気（オッズ）をそろえると差は消えるので、人気の見立てにすでに入っている情報で、言葉だけで印を動かす根拠にはなりません。"},
        cloud,
        {"type": "table", "columns": cols, "rows": rows(False), "caption": "好走寄りの語（出た数 300 以上、倍率が高い順）。倍率は、その語を含む馬の3着以内率 ÷ 全体の3着以内率。"},
        {"type": "table", "columns": cols, "rows": rows(True), "caption": "凡走寄りの語（出た数 300 以上、倍率が低い順）。"},
        {"type": "note", "text": (
            "語の切り方は形態素解析（fugashi + unidic-lite）で、同じ文節で隣り合う 1〜3 語の組を数えています。"
            "単独で意味が取れない語は単独では出さず、長い句に含まれる語は長い句にまとめています。"
            "『人気をそろえた倍率』は、実際の3着以内数 ÷ その馬の単勝人気の順位から見込まれる3着以内数です。人気はコメントを読んだ後の市場の評価を含むため、この補正はやや厳しめです。"
            "同じレースの馬は3着以内の数が3頭に固定されて互いに独立ではないため、p 値はやや甘めです。"
            "原文のコメントは載せず、短い語・句の集計値だけを出しています。")},
    ]
    return {
        "slug": "trainer-comment-wordcloud",
        "title": "厩舎コメントって成績に関連する？——言葉で好走は分かれるが、人気に織り込み済み",
        "claim": "厩舎コメントの言葉で、好走する馬としない馬が分かれる",
        "verdict": "partial",
        "summary": (f"『楽しみ』『能力』など前向きな言葉の馬は3着以内が多く、『叩く』『試す』などの馬は少ない傾向が出ました。"
                    "ただし単勝人気をそろえると、差の残る語は 0 個でした。"),
        "categories": ["stable/trainer"],
        "published": "2026-11-11",
        "dataset": {"period": "2020〜2025年", "conditions": "JRA全競馬場の厩舎コメント（結果を示す語を含むものは除く）。3着以内 = good、4着以降 = bad",
                    "sample": f"{n_rows:,} 頭分のコメント"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("summary")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    T = pd.read_csv(a.csv)
    summ = json.loads(Path(a.summary).read_text())
    text = json.dumps(article(T, summ), ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if a.dry_run:
        print(text[:3000])
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
