"""種牡馬の系統（父系）の分類。馬の血統（5代）の父系の祖先を父から順にさかのぼり、最初に当たる『系統の祖』でまとめる。

サンプル数が少ない種牡馬を、同じ系統の産駒としてまとめて見るために使う。系統の祖は馬 ID（netkeiba）で持ち、名前では結合しない。
どれにも当たらない馬は OTHER（その他の系統）。
"""

from __future__ import annotations

from typing import Any

OTHER = "other"

# 系統の祖の馬 ID -> 表示名。近い祖ほど優先されるので、孫系統（キングカメハメハ・デインヒル等）は祖より先に当たる。
LINES: dict[str, str] = {
    "000a00033a": "サンデーサイレンス系",
    "2001103460": "キングカメハメハ系",
    "000a001d7e": "キングマンボ系",
    "000a001607": "ミスタープロスペクター系",
    "000a0000d1": "デインヒル系",
    "000a0000d3": "ダンチヒ系",
    "000a001a98": "ストームキャット系",
    "000a00185d": "サドラーズウェルズ系",
    "000a000e04": "ノーザンダンサー系",
    "000a0012bf": "ヘイロー系",
    "000a0012cb": "ロベルト系",
    "000a000f2b": "ヘイルトゥリーズン系",
    "000a001d37": "A.P.インディ系",
    "000a0015fc": "シアトルスルー系",
    "000a000355": "テスコボーイ系",
    "000a000f88": "ナスルーラ系",
}
LABELS: dict[str, str] = {**LINES, OTHER: "その他の系統"}


def line_id_of(ancestors: list[dict[str, Any]]) -> str:
    """血統 JSON の ancestors（generation・position・horse_id）から、父の系統の ID を返す。父から父の父…（位置0）の順に最初に当たった祖。"""
    chain: dict[int, str] = {}
    for a in ancestors or []:
        try:
            if int(a["position"]) == 0 and a.get("horse_id"):
                chain[int(a["generation"])] = str(a["horse_id"])
        except (KeyError, TypeError, ValueError):
            continue
    for g in sorted(chain):
        if chain[g] in LINES:
            return chain[g]
    return OTHER
