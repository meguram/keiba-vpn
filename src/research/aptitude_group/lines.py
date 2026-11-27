"""大系統の札（netkeiba の「○○系」の祖先の鎖から規則で決める。sire_vectors と detailed_lines で共通）。"""
from __future__ import annotations

import json
from pathlib import Path

ORIGIN = Path("/home/jovyan/work/keiba-vpn/data/local/pedigree_lines/sire_line_origin.json")
MIN_LINE = 8
# 表示用の略し方（名前だけ。値には使わない）
ABBR = {"Mr. Prospector": "ミスプロ", "Northern Dancer": "ND", "Halo": "ヘイロー", "Roberto": "ロベルト", "Storm Bird": "ストバ",
        "Nearctic": "ニアクティック", "Danzig": "ダンチヒ", "Native Dancer": "ネイティヴ", "Seattle Slew": "スルー", "Sadler's Wells": "サドラー",
        "Bold Ruler": "ボールド", "Raise a Native": "レイズ", "Turn-to": "ターント", "Nasrullah": "ナスルーラ", "Caro": "カロ",
        "Deputy Minister": "デピュティ", "Nureyev": "ヌレイエフ", "Lyphard": "リファール", "Nearco": "ニアルコ", "Eclipse": "エクリプス",
        "Hail to Reason": "ヘイルトゥ", "Phalaris": "ファラリス", "Darley Arabian": "ダーレー"}


def short(label: str) -> str:
    s = label.replace("系", "").strip()
    return ABBR.get(s, s[:6])


def line_labels(qualified: set[str]) -> dict[str, str]:
    """鎖のある全種牡馬の大系統（数えるのは走数 300 以上の種牡馬だけ。新しい種牡馬にも同じ札が付く）。
    v2: 札 a のすぐ上の 8 頭以上の札 b に、b を最初の札に持つ種牡馬が 8 頭以上あるときだけ、a を b へまとめる（1 回だけ）。"""
    O = json.loads(ORIGIN.read_text(encoding="utf-8"))
    chains = {h: [x[2] for x in v["chain"][1:] if x[2]] for h, v in O.items()}   # 自分の札は代ごとに揺れるので、父から上の札だけ（critic 2 回目の 1）
    cnt: dict[str, int] = {}
    for h, ch in chains.items():
        if h in qualified:
            for lab in set(ch):
                cnt[lab] = cnt.get(lab, 0) + 1
    big = {k for k, v in cnt.items() if v >= MIN_LINE}
    first = {}
    for h, ch in chains.items():
        for lab in ch:
            if lab in big:
                first[h] = lab
                break
    direct: dict[str, int] = {}
    ups: dict[str, dict[str, int]] = {}
    for h, a in first.items():
        if h not in qualified:
            continue
        direct[a] = direct.get(a, 0) + 1
        ch = chains[h]
        above = [lab for lab in ch[ch.index(a) + 1:] if lab in big and lab != a]
        if above:
            ups.setdefault(a, {})
            ups[a][above[0]] = ups[a].get(above[0], 0) + 1
    merge = {}
    for a, u in ups.items():
        b = max(u, key=u.get)
        if direct.get(b, 0) >= MIN_LINE:
            merge[a] = b
    # b 自身がさらに上へまとまるなら、a は b へまとめない（札の名前と中身がずれないように。1 回だけの決まりを保つ・critic 3 回目の 6）
    merge = {a: b for a, b in merge.items() if b not in merge}
    return {h: merge.get(a, a) for h, a in first.items()}


