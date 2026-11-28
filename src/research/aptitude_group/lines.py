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


def line_labels(qualified: set[str], with_ancestors: bool = False) -> dict[str, str]:
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
    out = {h: merge.get(a, a) for h, a in first.items()}
    if with_ancestors == "own":
        # 鎖の中の各 id の「自分の札」（8 頭以上の札だけ・まとめも同じ）。line_of が父から上へたどるときに使う
        own = {}
        for v in O.values():
            for x in v["chain"]:
                if x[2] in big and x[0] not in own:
                    own[x[0]] = merge.get(x[2], x[2])
        return own
    if with_ancestors:
        # 鎖の中の祖先（日本の種牡馬でない古い祖先も）に、同じ規則で札を付ける: その祖先より上の鎖で最初の 8 頭以上の札（まとめも同じ）。
        # 母父・牝系 (B) の古い外国の祖先が「札なし」にならないように（11/27）。鍵のある種牡馬の札は変えない。
        for v in O.values():
            ch = v["chain"]
            for j in range(1, len(ch)):
                i = ch[j][0]
                if i in out:
                    continue
                for x in ch[j + 1:]:
                    if x[2] in big:
                        out[i] = merge.get(x[2], x[2])
                        break
    return out




PARENT_PATH = Path("/home/jovyan/work/aptitude_group_out/parent_links.json")
PED_PATH = Path("/home/jovyan/work/pedigree_lab_out/ped.parquet")
_PARENT: dict[str, str] = {}
_OWN: dict[str, str] = {}


def parent_links() -> dict[str, str]:
    """祖先の id → その父の id（5 代の血統の位置から）。血統の parquet が新しくなったら作り直す（critic 6 回目）。"""
    if not _PARENT:
        if not PARENT_PATH.exists() or PARENT_PATH.stat().st_mtime < PED_PATH.stat().st_mtime:
            from src.research.aptitude_group.line_vectors import pedigree_links
            PARENT_PATH.write_text(json.dumps(pedigree_links(set())[1]))
        _PARENT.update(json.loads(PARENT_PATH.read_text()))
    return _PARENT


def line_of(h: str, lab: dict[str, str], qualified: set[str]) -> str | None:
    """種牡馬 h の大系統。札の表に鍵があればその札（父から上の鎖で最初の 8 頭以上の札）。
    無ければ、父 → 父の父 → … と上へたどり、最初に「自分の札」が 8 頭以上の札である祖先の札を使う
    （鍵のある種牡馬の規則と同じ数え方。lab[父] を使うと 1 代上にずれる・critic 6 回目）。予測のモデルと地図で同じ関数を使う。"""
    if h in lab:
        return lab[h]
    if not _OWN:
        _OWN.update(line_labels(qualified, with_ancestors="own"))
    par = parent_links()
    x = h
    for _ in range(10):
        x = par.get(x)
        if x is None:
            return None
        if x in _OWN:
            return _OWN[x]
    return None
