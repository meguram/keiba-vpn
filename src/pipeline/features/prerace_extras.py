"""レース前に分かる追い切り評価・厩舎コメントから、馬ごとの補助特徴量を作る（GCS の保存データを読み取るだけ）。

- 追い切り評価（race_oikiri の evaluation: A〜D）→ oikiri_eval（A=4〜D=1）、有無のフラグ
- 厩舎コメント（race_trainer_comment の comment）→ コメントの有無・長さ・前向き/後ろ向きの語の数
いずれもそのレースの結果は含まない（前日夕方までに公開される情報）。結果を示す語が混ざっていないかの検査（leak_scan）も行う。
パドック評価は上位 5 頭だけで、公開もレース直前になるため（T-60 の予測に間に合わない）、ここでは使わない。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import pandas as pd

EVAL_MAP = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
POS = ("好調", "良化", "上向き", "仕上がり", "順調", "上々", "ベスト", "絶好", "気配", "動き", "反応", "具合", "デキ", "いい", "良い", "楽しみ", "期待", "自信", "勝負")
NEG = ("太い", "太め", "重い", "ひと叩き", "叩き", "休み明け", "物足り", "不安", "課題", "まだ", "もう少し", "気難しい", "難しい", "クセ", "ズブい", "湿", "渋")
RESULT_WORDS = ("勝てた", "勝った", "勝利", "優勝", "制した", "2着", "3着", "着に", "圏内に入", "ゴール", "快勝", "惜敗", "完敗", "敗れ")


def sentiment(text: str) -> tuple[int, int]:
    t = text or ""
    return sum(t.count(w) for w in POS), sum(t.count(w) for w in NEG)


def has_result_words(text: str) -> bool:
    return any(w in (text or "") for w in RESULT_WORDS)


def extract_race(race_id: str, oikiri: dict | None, comment: dict | None) -> list[dict]:
    rows: dict[str, dict] = {}
    for e in (oikiri or {}).get("entries") or []:
        hid = str(e.get("horse_id") or "")
        if hid:
            ev = EVAL_MAP.get(str(e.get("evaluation") or "").strip().upper()[:1])
            rows.setdefault(hid, {"race_id": race_id, "horse_id": hid})["oikiri_eval"] = ev
    for e in (comment or {}).get("entries") or []:
        hid = str(e.get("horse_id") or "")
        if hid:
            txt = str(e.get("comment") or "")
            pos, neg = sentiment(txt)
            r = rows.setdefault(hid, {"race_id": race_id, "horse_id": hid})
            r.update({"has_comment": 1.0 if txt.strip() else 0.0, "comment_len": float(len(txt)), "comment_pos": float(pos), "comment_neg": float(neg),
                      "comment_result_words": float(has_result_words(txt))})
    return list(rows.values())


def build(race_ids: Iterable[str], storage: Any, workers: int = 6) -> pd.DataFrame:
    def one(rid: str) -> list[dict]:
        try:
            return extract_race(rid, storage.load("race_oikiri", rid), storage.load("race_trainer_comment", rid))
        except Exception:
            return []

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rows in ex.map(one, race_ids):
            out += rows
    return pd.DataFrame(out)
