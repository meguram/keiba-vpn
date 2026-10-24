"""着順（同着を含む）から、各券種で払戻が出る組合せの集合を求める共通関数。

JRA の同着ルール（実データで確認済み。tests/pipeline/test_winning_combos.py に固定）:
- 同着の馬は、そのどれもがその着順として扱われる。上位 k 頭に入る組合せは、同着グループの中の取り方を全て数える。
  例: 3 着同着（A, B）= 複勝は 1・2・A・B の 4 頭、ワイドは 5 組（1-2・1-A・1-B・2-A・2-B。A-B は入らない）、3 連複は {1,2,A}・{1,2,B} の 2 組、3 連単は 1→2→A・1→2→B の 2 組。
  ワイドは『3 連複で当たる組の中の 2 頭』（3 着以内の枠 3 つのうち 2 つを占める組）。3 着を A・B が分け合うと枠は 1 つなので、A-B は当たらない（netkeiba・JRA の公表値で確認: 2024〜2025 年の 3 着同着 18 レースは全て 5 組）。
  2 着同着（A, B）なら 1 着・A・B が 3 枠を占めるので A-B も当たる（3 組）。
  例: 1 着同着（A, B）= 馬単は A→B・B→A、馬連は {A,B}、3 連単は A→B→3 着・B→A→3 着（3 着が同着なら更に増える）。
- 複勝の対象は、出走頭数 8 頭以上は 3 着まで、5〜7 頭は 2 着まで、4 頭以下は発売なし。ワイドは常に 3 着以内の 2 頭。
- 枠連は 1・2 着の枠番の組（同じ枠どうしも組になる）。
- 払戻額は券ごとに公式の値がある（同着では按分で半額などになる）ので、金額はこちらで計算せず、保存された払戻の組ごとの値を使う。

組合せの表現: 単勝・複勝は (馬番,)、順不同の券（馬連・ワイド・3連複）は昇順のタプル、順序のある券（馬単・3連単）は着順のタプル、枠連は昇順の枠番タプル。
"""

from __future__ import annotations

import itertools
import re
from typing import Iterable, Mapping

BET_TYPES = ("tansho", "fukusho", "wakuren", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan")
JP_TO_BET = {"単勝": "tansho", "複勝": "fukusho", "枠連": "wakuren", "馬連": "umaren", "ワイド": "wide", "馬単": "umatan", "三連複": "sanrenpuku", "三連単": "sanrentan"}
UNORDERED = ("umaren", "wide", "sanrenpuku", "wakuren")
ORDERED = ("umatan", "sanrentan")


def finish_groups(finish_by_horse: Mapping[int, int]) -> list[list[int]]:
    """{馬番: 着順} から、着順の昇順の同着グループ（各グループは馬番の昇順）を返す。着順が 0 以下（取消・除外・中止など）の馬は含めない。"""
    by: dict[int, list[int]] = {}
    for h, f in finish_by_horse.items():
        if f is not None and int(f) > 0:
            by.setdefault(int(f), []).append(int(h))
    return [sorted(by[f]) for f in sorted(by)]


def n_places(n_runners: int) -> int:
    """複勝が払い戻される着順の数（8 頭以上 3、5〜7 頭 2、4 頭以下は発売なし=0）。"""
    return 3 if n_runners >= 8 else (2 if n_runners >= 5 else 0)


def _perms_choose(group: list[int], r: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for c in itertools.combinations(group, r):
        out.extend(itertools.permutations(c))
    return out


def top_ordered(groups: list[list[int]], k: int) -> set[tuple[int, ...]]:
    """上位 k 頭の並び（同着グループの内側は並び替え自由）を全て返す。頭数が k に足りなければ空。"""
    res: list[tuple[int, ...]] = [()]
    need = k
    for g in groups:
        if need <= 0:
            break
        take = min(len(g), need)
        opts = _perms_choose(g, take)
        res = [a + b for a in res for b in opts]
        need -= take
    return {t for t in res if len(t) == k}


def top_set(groups: list[list[int]], k: int) -> set[int]:
    """上位 k 着以内に入る馬（同着で k を超えてもその着順の馬は全員含める）。"""
    got: list[int] = []
    for g in groups:
        if len(got) >= k:
            break
        got.extend(g)
    return set(got)


def winning_combos(groups: list[list[int]], bet_type: str, *, n_runners: int | None = None, brackets: Mapping[int, int] | None = None) -> set[tuple[int, ...]]:
    """券種ごとの『払戻が出るはずの組合せ』の集合。n_runners は複勝の対象（発売）判定用（省略時は着順のある頭数）。枠連は brackets={馬番: 枠番} が要る。"""
    n_ranked = sum(len(g) for g in groups)
    n = n_ranked if n_runners is None else n_runners
    if bt_invalid(bet_type):
        raise ValueError(bet_type)
    if bet_type == "tansho":
        return {(h,) for h in groups[0]} if groups else set()
    if bet_type == "fukusho":
        k = n_places(n)
        return {(h,) for h in sorted(top_set(groups, k))} if k else set()
    if bet_type == "wide":
        if n_ranked >= 3:
            return {c for t in top_ordered(groups, 3) for c in itertools.combinations(sorted(t), 2)}
        p = sorted(top_set(groups, 3))
        return {c for c in itertools.combinations(p, 2)} if len(p) >= 2 else set()
    k = {"umaren": 2, "umatan": 2, "sanrenpuku": 3, "sanrentan": 3, "wakuren": 2}[bet_type]
    if n_ranked < k:
        return set()
    ordered = top_ordered(groups, k)
    if bet_type in ORDERED:
        return ordered
    if bet_type == "wakuren":
        if not brackets:
            return set()
        return {tuple(sorted((brackets.get(a, -1), brackets.get(b, -1)))) for a, b in ordered}
    return {tuple(sorted(t)) for t in ordered}


def bt_invalid(bet_type: str) -> bool:
    return bet_type not in BET_TYPES


_SEP = re.compile(r"[>\-→－\s]+")


def parse_combo(bet_type: str, text: str) -> tuple[int, ...] | None:
    """払戻の組合せ文字列（'4 - 7'、'7 → 4'、'4-7'、'7>4'）を正規化したタプルにする。読めなければ None。"""
    try:
        nums = tuple(int(x) for x in _SEP.split(str(text).strip()) if x != "")
    except ValueError:
        return None
    if not nums:
        return None
    return nums if bet_type in ORDERED or len(nums) == 1 else tuple(sorted(nums))


def normalize_payoff(payoff: Mapping | None) -> dict[str, dict[tuple[int, ...], int]]:
    """保存形式の払戻（券種名→{numbers,payout,popularity} か、その配列）を {bet_type: {組合せ: 100 円あたり払戻}} にする。連結（popularity なしの複数行相当）は対象外で、呼び出し側で扱う。"""
    out: dict[str, dict[tuple[int, ...], int]] = {}
    if not isinstance(payoff, Mapping):
        return out
    for jp, val in payoff.items():
        bt = JP_TO_BET.get(jp)
        if not bt:
            continue
        rows = val if isinstance(val, list) else [val]
        for r in rows:
            if not isinstance(r, Mapping):
                continue
            c = parse_combo(bt, r.get("numbers", ""))
            try:
                p = int(str(r.get("payout", "")).replace(",", "").strip())
            except ValueError:
                continue
            if c:
                out.setdefault(bt, {})[c] = p
    return out


def hit_payout(bet_type: str, ticket: Iterable[int], payoff_rows: Mapping[tuple[int, ...], int]) -> int:
    """買い目（組合せ）が当たったときの 100 円あたり払戻。外れは 0。同着で複数の組に払戻があっても、買った組の分だけを返す。"""
    c = tuple(ticket)
    c = c if bet_type in ORDERED or len(c) == 1 else tuple(sorted(c))
    return int(payoff_rows.get(c, 0))


def diff_expected(groups: list[list[int]], bet_type: str, observed: Iterable[tuple[int, ...]], *, n_runners: int | None = None, brackets: Mapping[int, int] | None = None) -> dict[str, list[tuple[int, ...]]]:
    """保存された払戻の組合せと期待集合の差（missing=欠け、extra=余り）。"""
    exp = winning_combos(groups, bet_type, n_runners=n_runners, brackets=brackets)
    obs = set(observed)
    return {"missing": sorted(exp - obs), "extra": sorted(obs - exp)}
