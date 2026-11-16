"""確定結果（race_result）の保存物に残っている、面（障害）・レース番号・同着の単勝払戻の誤りを、書き出し（export_tables）と解析で直す共通関数。

データ監査 docs/operations/data_audit_20261116.md の C3・C4・C5:
- C3 障害レースの面: 結果のページのコース表記を読み違え、障害が「芝」「ダート」「空」で保存されている（2020 年以降は年 約 125 レースのほぼ全部）。
  レース名（障害・ジャンプ・J・G など）か出馬表の面（障）で「障」に直す。平地を障にしないよう、ヤングJSFR のような名前・2,500m 未満は障にしない。
- C4 レース番号: 古い保存物は round が 0。race_id の末尾 2 桁がレース番号。
- C5 同着の単勝: 1 着同着の単勝が {"numbers": "611", "payout": "1,320630"} のように 1 つに連結されて保存されている
  （2024-01〜2026-03 の再取得の保存形式。popularity が無い。複勝・ワイドの連結は scraper.payoff_repair が扱い、単勝は対象外だった）。
  1 着の 2 頭の馬番の連結で、金額の分け方が 1 通りに決まるときだけ 2 つの要素に分ける。決まらなければ触らない。
"""

from __future__ import annotations

import re
from typing import Any

# 障害レースの名前: 「障害4歳以上未勝利」「新潟ジャンプS(JGIII)」「阪神スプリングJ(JGII)」「J・G1」。
# 素の「JS」は使わない（平地の「ヤングJSFR中京1(1勝)」が年 2 レースある）。出馬表・速報の短い名前（新潟JS・東京HJ・阪神SJ・中山グランドJ）は末尾の J だけ見る。
_STEEPLE_NAME_RE = re.compile(r"障害|ジャンプ|\(\s*J\s*[・.･]?\s*G|J\s*[・.･]\s*G")
_STEEPLE_SHORT_RE = re.compile(r"(?:JS|HJ|SJ|[ァ-ヶー]J)$")
# JRA の障害は最短 2,750m。平地は最長 3,600m なので距離だけでは決めないが、短い距離のものは障にしない（名前の誤判定よけ）
STEEPLE_MIN_DISTANCE = 2500


def _to_int(v: Any) -> int | None:
    try:
        if v is None or (isinstance(v, float) and v != v):
            return None
        return int(float(str(v).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def is_steeplechase_name(race_name: Any, distance: Any = None) -> bool:
    """レース名が障害レースのものか。distance が分かっていて 2,500m 未満なら障害としない。"""
    name = str(race_name or "").strip()
    if not name:
        return False
    d = _to_int(distance)
    if d is not None and 0 < d < STEEPLE_MIN_DISTANCE:
        return False
    if _STEEPLE_NAME_RE.search(name):
        return True
    short = re.sub(r"\s*[\(（][^\)）]*[\)）]\s*$", "", name)   # 末尾の (JGII)・(OP) などを外して短い名前の形を見る
    return bool(_STEEPLE_SHORT_RE.search(short))


def resolve_surface(surface: Any, race_name: Any = None, distance: Any = None, fallback_surface: Any = None) -> Any:
    """面を返す。もとが「障」・出馬表の面が「障」・レース名が障害なら「障」、それ以外はもとの値（障を平地に戻すことはしない）。"""
    if str(surface or "").startswith("障"):
        return "障"
    if str(fallback_surface or "").startswith("障"):
        return "障"
    if is_steeplechase_name(race_name, distance):
        return "障"
    return surface


def round_from_race_id(round_value: Any, race_id: Any) -> Any:
    """レース番号が 0・欠けなら race_id の末尾 2 桁を返す（それ以外はもとの値）。"""
    r = _to_int(round_value)
    rid = str(race_id or "")
    if (r is None or r == 0) and len(rid) == 12 and rid.isdigit():
        return int(rid[-2:])
    return round_value


_PAYOUT_TOKEN = re.compile(r"\d{3}|\d{1,3}(?:,\d{3})+")


def _split_two_payouts(s: str) -> list[tuple[str, str]]:
    """連結された 2 つの金額（3 桁、または桁区切りカンマ付き。100 円以上・10 円単位）の分け方を全て返す。"""
    s = re.sub(r"\s+", "", str(s or "")).replace("円", "")
    out = []
    for i in range(3, len(s) - 2):
        a, b = s[:i], s[i:]
        if not (_PAYOUT_TOKEN.fullmatch(a) and _PAYOUT_TOKEN.fullmatch(b)):
            continue
        va, vb = int(a.replace(",", "")), int(b.replace(",", ""))
        if va >= 100 and vb >= 100 and va % 10 == 0 and vb % 10 == 0:
            out.append((a, b))
    return out


def _winners(entries: Any) -> list[int]:
    out = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        if _to_int(e.get("finish_position")) == 1:
            n = _to_int(e.get("horse_number"))
            if n is not None and n > 0:
                out.append(n)
    return out


def split_deadheat_win(payoff: Any, entries: Any) -> Any:
    """1 着同着の単勝が 1 つに連結されていれば、馬番の昇順の 2 要素の list に分けた払戻を返す（元は変えない）。

    分けるのは、1 着が 2 頭・numbers が 2 頭の馬番の連結（どちらの順でも）・金額の分け方が 1 通りのときだけ。それ以外はそのまま返す。
    """
    if not isinstance(payoff, dict):
        return payoff
    win = payoff.get("単勝")
    if not isinstance(win, dict):
        return payoff
    nums = re.sub(r"\s+", "", str(win.get("numbers") or ""))
    w = _winners(entries)
    if len(w) != 2 or len(set(w)) != 2 or not nums.isdigit():
        return payoff
    a, b = w
    if f"{a}{b}" == f"{b}{a}":   # 1 と 11 の同着は "111" でどちらの順か決まらない（金額を取り違えないよう触らない）
        return payoff
    if nums == f"{a}{b}":
        order = (a, b)
    elif nums == f"{b}{a}":
        order = (b, a)
    else:
        return payoff
    splits = _split_two_payouts(win.get("payout") or "")
    if len(splits) != 1:
        return payoff
    pa, pb = splits[0]
    # 人気は連結の形では保存されていない（分けられない）ので空。複勝・ワイドの補正（payoff_repair.apply_fix）と同じ形
    items = sorted(
        [{"numbers": str(order[0]), "payout": pa, "popularity": ""}, {"numbers": str(order[1]), "payout": pb, "popularity": ""}],
        key=lambda x: int(x["numbers"]),
    )
    out = dict(payoff)
    out["単勝"] = items
    return out
