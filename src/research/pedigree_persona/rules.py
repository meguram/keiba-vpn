"""血統予想家の規則（凍結の対象。会議の 1 号の条件 3）。

このファイルの sha256 は config/pedigree_persona/frozen.json の rules_sha256 に残し、起動時に照合する。
直すときは版（RULES_VERSION）を分け、理由を persona_design.md に書いて、凍結を作り直す（満期の 2028 年後半まで直さない）。

規則（persona_design.md の 2 章。未検証の規則）:
- 対象: 芝・ダートの平地。新馬戦では一覧を出さない。
- 点: ダートは dirt_mat5、芝は turf_mat5 の生の値を、T-60 の時点の出走馬でレースの中で中心化し、血統の無い馬は 0 にして、
  開発の行の sd の定数で割る（前向きの登録の判定と同じ扱い）。
- 選び: T-60 の単勝オッズの順で 6 番人気以下、かつ点が閾値（2024 年の同じ母集団の上位 10%）以上。3 頭以上なら点の上位 2 頭。
- 理由の文: 割合を主にし、名前は 3〜4 代の祖先の寄与の上位から例として 2 頭まで。事実の言い方にする。
- 事実の札: 初ダート・休み明け・出走 1〜2 走・新馬戦の数だけ（効きの主張はしない）。
- 点・閾値・人気の数字は画面に出さない。成績（的中率・回収率）は計算しない・出さない。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

RULES_VERSION = "pp-1"
ROOT = Path(__file__).resolve().parents[3]
FROZEN_PATH = ROOT / "config/pedigree_persona/frozen.json"

POP_MIN = 6                 # 6 番人気以下
TOP_FRAC = 0.10             # 閾値は上位 10%
MAX_PICKS = 2
NEW_SIRE_RUNS = 100         # 父の産駒の走りがこれ未満なら「父の産駒の成績がまだ少ない」と書く
SCORE_COL = {"ダート": "dirt_mat5", "芝": "turf_mat5"}

BANNED = ("回収率100%超", "回収率 100% 超", "もうかる", "儲かる", "儲け", "勝てる", "期待値", "妙味", "お宝", "狙い目", "穴", "推奨", "おすすめ",
          "オススメ", "お勧め", "買い", "買える", "注目", "激走", "必ず", "確実")

TITLE = "血統予想家"
LABEL = "確かめ中の見方"
NOTE_UNVERIFIED = "回収率は確かめていない。成績は 2028 年ごろまで出さない"
INTRO = ("母の側の奥の祖先まで見る予想家。人気の無い馬の中から、血統の点の上位を挙げる。"
         "上位人気では、ほかの見方に任せる。確かめ中の見方で、回収率は確かめていない。")
NOTES = (
    "印は出しません（めぐAI印とは別の見方です）。",
    "一覧は、発走の約 60 分前の単勝オッズの順で 6 番人気以下の馬から選びます。発走の後は表示しません。",
    "母の側の祖先の子孫が少ない馬（外国産の馬など）は、点が 0 に寄るので選ばれにくくなります。",
)
NOTE_TURF = "芝の欄は確かめが弱い・未検証の規則です（開発の期間の記述だけで、前向きにも 2025・26 年でも確かめていません）。"
SURF_WORD = {"ダート": "ダート", "芝": "芝"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(path: Path = FROZEN_PATH, check: bool = True) -> dict[str, Any]:
    """凍結の値（閾値・sd）を読む。規則（このファイル）と列の作り（columns.py）の sha256 が凍結の時と違えば止める。"""
    fz = json.loads(path.read_text(encoding="utf-8"))
    if check:
        want = {"rules.py": fz.get("rules_sha256"), "columns.py": (fz.get("code_sha256") or {}).get("columns.py")}
        for name, h in want.items():
            got = file_sha256(Path(__file__).with_name(name))
            if h != got:
                raise SystemExit(f"{name} が凍結の版と違う（{got[:16]}… ≠ {str(h)[:16]}…）。版を分けて凍結し直す")
    return fz


def score(raw: np.ndarray, race_id: np.ndarray, no_ped: np.ndarray, sd: float) -> np.ndarray:
    """レースの中で中心化し、血統の無い馬は中心化の後に 0、開発の行の sd で割る（学習と推論で同じ関数を使う）。"""
    raw = np.asarray(raw, dtype=float)
    v = raw - pd.Series(raw).groupby(np.asarray(race_id)).transform("mean").to_numpy()
    v = np.where(np.asarray(no_ped, dtype=bool), 0.0, v)
    return v / sd


def pop_rank(race_id: np.ndarray, odds: np.ndarray, horse_number: np.ndarray) -> np.ndarray:
    """単勝オッズの低い順の人気（同じオッズは馬番の若い順）。閾値の母集団と推論で同じ関数を使う。"""
    T = pd.DataFrame({"r": np.asarray(race_id), "o": np.asarray(odds, dtype=float), "h": np.asarray(horse_number, dtype=float)})
    T["i"] = np.arange(len(T))
    T = T.sort_values(["r", "o", "h"], kind="stable")
    T["k"] = T.groupby("r").cumcount() + 1
    return T.sort_values("i")["k"].to_numpy()


def pick(z: np.ndarray, rank: np.ndarray, threshold: float) -> np.ndarray:
    """1 レースの中で選ぶ馬の位置（点の大きい順・最大 2 頭）。"""
    z, rank = np.asarray(z, dtype=float), np.asarray(rank)
    ok = np.flatnonzero((rank >= POP_MIN) & (z >= threshold))
    return ok[np.argsort(-z[ok], kind="stable")][:MAX_PICKS]


_KANA_THEN_LATIN = re.compile(r"^([\u30a0-\u30ff\u3040-\u309f\u4e00-\u9fff・ー]+)[A-Za-z].*$")


def clean_name(name: str) -> str:
    """血統の JSON の名前を画面の形にする（「ジェイドロバリーJade Robbery(米)」→「ジェイドロバリー」・「Wild Again\xa0(米)」→「Wild Again」）。"""
    s = str(name or "").replace("\xa0", " ").strip()
    s = re.sub(r"\s*[（(][^()（）]*[)）]\s*$", "", s).strip()
    m = _KANA_THEN_LATIN.match(s)
    return m.group(1) if m else s


def reason(surface: str, names: list[str], first_dirt: bool, sire_runs: int | None) -> str:
    """理由の文（規則から作る。手で書かない）。"""
    def build(ns: list[str]) -> str:
        w = SURF_WORD.get(surface, surface)
        ex = f"（例: {'・'.join(ns)}）" if ns else ""
        tail = []
        if surface == "ダート" and first_dirt:
            tail.append("今回が初ダート")
        if sire_runs is not None and sire_runs < NEW_SIRE_RUNS:
            tail.append("父の産駒の成績がまだ少ない")
        return f"母の側の 5 代に、{w}のレースで子孫の成績が良い祖先の血が多い{ex}。" + ("・".join(tail) + "。" if tail else "")
    ns = [c for c in (clean_name(n) for n in names[:2]) if c]
    bad = banned_in(build(["○"] * len(ns)))          # 文の型を検査する（祖先の名前は固有名詞なので検査しない）
    if bad:
        raise ValueError(f"禁止の語が入った: {bad}")
    return build(ns)


def facts(surface: str, first_dirt: Iterable[bool], layoff: Iterable[bool], career: Iterable[float], new_race: bool) -> str:
    """レースの事実の札（効きの主張はしない）。"""
    if new_race:
        return "新馬戦"
    c = pd.to_numeric(pd.Series(list(career)), errors="coerce")
    parts = []
    if surface == "ダート":
        parts.append(f"初ダート {int(sum(bool(x) for x in first_dirt))} 頭")
    parts.append(f"休み明け {int(sum(bool(x) for x in layoff))} 頭")
    parts.append(f"出走 1〜2 走 {int(c.isin([1, 2]).sum())} 頭")
    return "・".join(parts)


def banned_in(text: str) -> list[str]:
    return [w for w in BANNED if w in text]


def module(*, race_id: str, surface: str, facts_text: str, state: str, picks: list[dict[str, Any]], asof: str, made_at: str,
           rule_sha: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """めぐ分析の欄の JSON（race_analysis/pedigree_persona/<race_id>.json）。state: pending | ready | none | unavailable | not_applicable。"""
    notes = list(NOTES) + ([NOTE_TURF] if surface == "芝" else [])
    doc = {
        "id": "pedigree_persona", "view": "pedigree_persona", "status": "ok", "label": TITLE, "description": LABEL,
        "version": RULES_VERSION, "race_id": race_id,
        "figure": {"intro": INTRO, "note": NOTE_UNVERIFIED, "surface": surface, "confidence": "弱め" if surface == "芝" else None,
                   "facts": facts_text, "picks_state": state, "picks": picks},
        "notes": notes, "asof": asof, "made_at": made_at, "rule_sha": rule_sha, "inputs": inputs,
    }
    fixed = {k: v for k, v in doc["figure"].items() if k != "picks"}       # 馬の名前・理由の文（reason で検査済み）の外を検査する
    bad = banned_in(json.dumps(fixed, ensure_ascii=False) + "".join(notes) + TITLE + LABEL)
    if bad:
        raise ValueError(f"禁止の語が入った: {bad}")
    return doc
