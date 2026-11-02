"""騎手・調教師の名前の読み（ひらがな）と、50 音順に並べるためのキー。

マスタは data/meta/person_reading/{jockey,trainer}_reading.json（手で確認した読み。netkeiba のフリガナがある調教師はそれを使う）。
レース結果の騎手名は「横山武」「Ｍ．デム」のような略称や、見習いの印（▲△☆★◇）付きで入っているので、
id・正式名・略称（aliases）のどれからでも引けるようにする。

使い方:
    from src.utils.person_reading import jockey_reading, trainer_reading, kana_sort_key, sort_by_reading
    jockey_reading("横山武")            # -> "よこやま たけし"
    jockey_reading("01170")             # id でも引ける
    sorted(names, key=lambda n: kana_sort_key(jockey_reading(n) or n))
    sort_by_reading(names, "jockey")    # 上と同じ（読みの無い名前は末尾）

新しく出てきた騎手・調教師（マスタに無い id）の洗い出し:
    python -m src.utils.person_reading --missing
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

REPO = Path(__file__).resolve().parents[2]
MASTER_DIR = REPO / "data/meta/person_reading"
Kind = Literal["jockey", "trainer"]

_MARKS = re.compile(r"^[▲△☆★◇◆]+")
_SMALL = str.maketrans("ぁぃぅぇぉっゃゅょゎゕゖ", "あいうえおつやゆよわかけ")
_VOWEL = {}
for _row, _v in [("あかさたなはまやらわがざだばぱぁゃゎ", "あ"), ("いきしちにひみりぎじぢびぴぃ", "い"),
                 ("うくすつぬふむゆるぐずづぶぷぅっゅゔ", "う"), ("えけせてねへめれげぜでべぺぇ", "え"),
                 ("おこそとのほもよろをごぞどぼぽぉょ", "お")]:
    for _c in _row:
        _VOWEL[_c] = _v


def normalize_name(name: str) -> str:
    """表記ゆれをならす: 全角英字・記号を半角に（NFKC）、見習いの印と空白を除く。"""
    s = unicodedata.normalize("NFKC", str(name or "")).strip()
    s = _MARKS.sub("", s)
    return re.sub(r"\s+", "", s)


def to_hiragana(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


@lru_cache(maxsize=None)
def _master(kind: Kind) -> tuple[dict[str, dict], dict[str, dict], list[dict]]:
    """(id -> 人, 正規化した表記 -> 人, 全員)。"""
    doc = json.loads((MASTER_DIR / f"{kind}_reading.json").read_text(encoding="utf-8"))
    persons = doc["persons"]
    by_id = {p["id"]: p for p in persons}
    by_alias: dict[str, dict] = {}
    for p in persons:
        for a in [p["name"], *p.get("aliases", [])]:
            by_alias.setdefault(normalize_name(a), p)
    return by_id, by_alias, persons


def lookup(name_or_id: str, kind: Kind) -> dict | None:
    """マスタの 1 人分（id・name・reading・confidence など）。見つからなければ None。"""
    by_id, by_alias, persons = _master(kind)
    key = str(name_or_id or "").strip()
    if key in by_id:
        return by_id[key]
    n = normalize_name(key)
    if not n:
        return None
    if n in by_alias:
        return by_alias[n]
    # 略称の前方一致（例: 「小牧加」→ 小牧加矢太）。当てはまる人が 1 人のときだけ
    hits = [p for p in persons if normalize_name(p["name"]).startswith(n)]
    return hits[0] if len(hits) == 1 else None


def reading(name_or_id: str, kind: Kind) -> str | None:
    """読み（ひらがな。姓と名の間は半角スペース）。

    人が特定できないが、前方一致する全員の姓の読みが同じとき（例: 「角田」→ 角田大和・角田大河）は姓の読みだけを返す。
    それも決まらなければ None。
    """
    p = lookup(name_or_id, kind)
    if p:
        return p["reading"]
    n = normalize_name(name_or_id)
    if not n:
        return None
    surnames = {p["reading"].split(" ")[0] for p in _master(kind)[2] if normalize_name(p["name"]).startswith(n)}
    return surnames.pop() if len(surnames) == 1 else None


def jockey_reading(name_or_id: str) -> str | None:
    return reading(name_or_id, "jockey")


def trainer_reading(name_or_id: str) -> str | None:
    return reading(name_or_id, "trainer")


def _fold(part: str) -> str:
    """50 音順の第 1 キー: 濁点・半濁点を外し、小書きを並字に、長音「ー」を直前の母音にする（辞書の並べ方）。"""
    s = unicodedata.normalize("NFD", to_hiragana(part))
    s = s.replace("゙", "").replace("゚", "")
    s = unicodedata.normalize("NFC", s).translate(_SMALL)
    out: list[str] = []
    for c in s:
        out.append(_VOWEL.get(out[-1], c) if c == "ー" and out else c)
    return "".join(out)


def kana_sort_key(reading_text: str | None) -> tuple:
    """読みを 50 音順に並べるキー。姓 → 名 の順に比べ（「おの」は「おのでら」より前）、
    清音と濁音は同じ位置として扱い、同じなら清音を先にする。None・空は末尾。"""
    if not reading_text:
        return (1, (), ())
    parts = [p for p in re.split(r"[\s・]+", to_hiragana(reading_text.strip())) if p]
    return (0, tuple(_fold(p) for p in parts), tuple(parts))


def sort_by_reading(names: Iterable[str], kind: Kind) -> list[str]:
    """名前（略称・id 可）を 50 音順に並べる。読みの無い名前は末尾に、名前の順で。"""
    return sorted(names, key=lambda n: (kana_sort_key(reading(n, kind)), str(n)))


def missing(since_year: int = 2020) -> dict[str, list[tuple[str, str, int]]]:
    """ローカルのレース結果（data/page_reference/tables/<年>/race_result_flat.parquet）に出て、マスタに無い id。"""
    import pandas as pd

    files = sorted(p for p in (REPO / "data/page_reference/tables").glob("*/race_result_flat.parquet")
                   if p.parent.name.isdigit() and int(p.parent.name) >= since_year)
    out: dict[str, list[tuple[str, str, int]]] = {}
    for kind in ("jockey", "trainer"):
        cols = [f"{kind}_id", f"{kind}_name"]
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True).dropna()
        cnt = df.groupby(cols).size()
        by_id = _master(kind)[0]
        out[kind] = [(i, n, int(c)) for (i, n), c in cnt.items() if str(i) not in by_id]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--missing", action="store_true", help="マスタに無い騎手・調教師を一覧にする")
    ap.add_argument("--since", type=int, default=2020)
    args = ap.parse_args()
    if args.missing:
        for kind, rows in missing(args.since).items():
            print(f"[{kind}] {len(rows)} 件")
            for i, n, c in rows:
                print(f"  {i}\t{n}\t{c}")


if __name__ == "__main__":
    main()
