"""種牡馬ごとの系統を smartrc の分類で持つ台帳（netkeiba の『○○系』より細かい。例: サンデー系を ディープ系・Tサンデー系… に分ける）。

smartrc の馬情報（horses/view）・全戦績（fullresults）には、馬ごとに父の系統コードが入っている:
  f_llcode = 大系統（11 種。例: サンデー系・ノーザンダンサー系）、f_slcode = 小系統（46 種。例: ディープ系・ロベルト系）、
  f_clcode = 国別のタイプ（10 種。例: 日本型サンデー系・米国型ミスプロ系）。
コード → 名前の対応表は smartrc の画面（v3/app.js の kLBloods・kSBloods・kCBloods）から取り、data/meta/sire_lines/smartrc_line_master.json に置く。

台帳の作り方（種牡馬は netkeiba の馬 ID で持ち、名前では結合しない）:
  1. 手元の smartrc の全戦績の表（tp32 の career_eq / runners_eq。馬ごとの f_code・f_*code）から、馬 → 父の系統コード。
  2. 馬 → 父（netkeiba ID）の対応表（horse_sire_map.parquet）とつなぎ、種牡馬ごとに最も多いコードを取る（食い違いは数を残す）。
  3. 1 で 1 頭も分からない種牡馬は、その産駒 1 頭の馬情報を smartrc から取る（fetch。2.2〜4.0 秒間隔、制限を受けたら止める）。
  smartrc で系統が空欄の種牡馬（比較的新しく輸入された海外の種牡馬など。smartrc の画面でも系統欄は空白）は台帳に入れず、
  netkeiba の『○○系』を使う（2026-11-06 ユーザー指示。build_horse_sire_map.line_labels が netkeiba の台帳で補う）。
出力: data/meta/sire_lines/sire_smartrc_lines.json
  {"built": 日時, "complete_at": 全種牡馬を試し終えた日時 or null, "sires": {"<種牡馬ID>": {name, small, small_name, large, large_name, type, type_name, n, from}}}

使い方:
    python -m src.scraper.smartrc_sire_lines build              # 手元のデータから作る（取得なし）
    python -m src.scraper.smartrc_sire_lines fetch --max 100    # 分からない種牡馬を smartrc から補う
    python -m src.scraper.smartrc_sire_lines master             # 系統名・色の対応表を smartrc の画面から取り直す（2 回の取得）
    python -m src.scraper.smartrc_sire_lines colors             # 全種牡馬の色の区分（大系統の色・国系統 日本型/米国型/欧州型 の色）を CSV に出す（取得なし）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
JST = timezone(timedelta(hours=9))
MASTER = REPO / "data/meta/sire_lines/smartrc_line_master.json"
OUT = Path(os.environ.get("KEIBA_SMARTRC_SIRE_LINES") or REPO / "data/meta/sire_lines/sire_smartrc_lines.json")
from src.config.data_paths import MODELING_DIR, PAGE_REF  # noqa: E402

TP32 = MODELING_DIR / "training_pipeline_v4/tp32"   # 学習の成果物（ホストの data/local。共通の定義から取る）
SIRE_MAP = PAGE_REF / "pedigree_race_index/horse_sire_map.parquet"
INTERVAL = (2.2, 4.0)
COLORS_CSV = REPO / "data/meta/sire_lines/sire_smartrc_colors.csv"


def master() -> dict[str, Any]:
    return json.loads(MASTER.read_text(encoding="utf-8"))


def names(codes: dict[str, Any], m: dict[str, Any] | None = None) -> dict[str, Any]:
    """{small, large, type} のコードに名前を付ける。小系統から大系統が分かるときは補う。"""
    m = m or master()
    sm = m["small"].get(str(codes.get("small") or ""), {})
    large = codes.get("large") or sm.get("large")
    return {"small": codes.get("small"), "small_name": sm.get("name"),
            "large": large, "large_name": m["large"].get(str(large or ""), {}).get("name"),
            "type": codes.get("type"), "type_name": m["type"].get(str(codes.get("type") or ""), {}).get("name")}


def load(path: Path | None = None) -> dict[str, Any]:
    path = path or OUT
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"sires": {}}


def labels(path: Path | None = None) -> dict[str, str]:
    """種牡馬 ID → 小系統の名前（めぐ分析の『系統』）。"""
    path = path or OUT
    return {k: v["small_name"] for k, v in load(path).get("sires", {}).items() if v.get("small_name")}


def is_complete(path: Path | None = None) -> bool:
    path = path or OUT
    return bool(load(path).get("complete_at"))


def _save(doc: dict[str, Any], path: Path | None = None) -> None:
    path = path or OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _horse_codes() -> pd.DataFrame:
    """手元の smartrc の表から 馬 ID → 父の系統コード（新しい表を優先）。"""
    frames = []
    for name in ("runners_eq.parquet", "career_eq.parquet"):
        p = TP32 / name
        if not p.exists():
            continue
        try:
            d = pd.read_parquet(p, columns=["hcode", "f_code", "f_name", "f_llcode", "f_slcode", "f_clcode"])
        except Exception:      # 列の無い表は使わない
            continue
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["horse_id", "f_slcode"])
    d = pd.concat(frames, ignore_index=True).dropna(subset=["f_slcode"])
    d["horse_id"] = d["hcode"].astype(str)
    return d.drop_duplicates("horse_id")


def build(path: Path | None = None) -> dict[str, int]:
    """手元のデータから台帳を作り直す（smartrc から取った分＝from が 'fetch' の行は残す）。"""
    path = path or OUT
    m = master()
    old = load(path).get("sires", {})
    sm = pd.read_parquet(SIRE_MAP, columns=["horse_id", "sire_id", "sire_name"]).dropna(subset=["sire_id"])
    sm["horse_id"], sm["sire_id"] = sm["horse_id"].astype(str), sm["sire_id"].astype(str)
    j = sm.merge(_horse_codes(), on="horse_id", how="left")
    sires: dict[str, Any] = {}
    conflicts = 0
    for sid, g in j.groupby("sire_id"):
        g2 = g.dropna(subset=["f_slcode"])
        name = g["sire_name"].dropna().iloc[0] if g["sire_name"].notna().any() else None
        if g2.empty:
            if sid in old and old[sid].get("from") == "fetch":
                sires[sid] = old[sid]
            continue
        key = g2[["f_llcode", "f_slcode", "f_clcode"]].astype(str).agg("|".join, axis=1)
        top = key.value_counts()
        conflicts += int(len(top) > 1)
        ll, sl, cl = top.index[0].split("|")
        sires[sid] = {"name": name, **names({"small": sl, "large": ll, "type": cl}, m), "n": int(len(g2)),
                      "agree": round(float(top.iloc[0] / top.sum()), 3), "from": "smartrc_tables",
                      "smartrc_code": str(g2["f_code"].mode().iloc[0]) if g2["f_code"].notna().any() else None}
    all_sires = set(sm["sire_id"])
    prev = load(path)
    tried = prev.get("tried") or {}
    missing = sorted(all_sires - set(sires))
    # 完了 = 台帳に無い種牡馬を、すべて smartrc で確かめ終えた（空欄だったものは tried に残り、netkeiba の系統を使う）
    done = not [s for s in missing if s not in tried]
    doc = {"built": datetime.now(JST).isoformat(timespec="seconds"), "n_sires": len(all_sires), "sires": sires, "tried": tried,
           "missing": missing, "complete_at": (prev.get("complete_at") or datetime.now(JST).isoformat(timespec="seconds")) if done else None}
    _save(doc, path)
    return {"sires": len(all_sires), "known": len(sires), "missing": len(doc["missing"]), "conflicts": conflicts}


def fetch(max_n: int = 100, path: Path | None = None) -> dict[str, int]:
    """台帳に無い種牡馬を、産駒 1 頭の smartrc 馬情報（horses/view）から補う。全部試し終えたら complete_at を付ける。"""
    path = path or OUT
    from src.scraper.smartrc_client import SmartRCClient, SmartRCRestricted

    m = master()
    doc = load(path)
    sires = doc.setdefault("sires", {})
    tried = doc.setdefault("tried", {})
    sm = pd.read_parquet(SIRE_MAP, columns=["horse_id", "sire_id", "sire_name"]).dropna(subset=["sire_id"])
    sm["horse_id"], sm["sire_id"] = sm["horse_id"].astype(str), sm["sire_id"].astype(str)
    todo = [s for s in sorted(set(sm["sire_id"]) - set(sires)) if s not in tried]
    cli = SmartRCClient(interval=INTERVAL[0], jitter=INTERVAL[1] - INTERVAL[0])
    got = miss = 0
    try:
        for sid in todo[:max_n]:
            kids = sm.loc[sm["sire_id"] == sid, ["horse_id", "sire_name"]]
            rec = None
            for hid in kids["horse_id"].head(2):         # 1 頭目が smartrc に無いときだけ 2 頭目
                h = cli.get_horse(hid)
                if h and h.get("f_slcode"):
                    rec = h
                    break
            if rec:
                sires[sid] = {"name": kids["sire_name"].iloc[0], **names({"small": rec["f_slcode"], "large": rec.get("f_llcode"), "type": rec.get("f_clcode")}, m),
                              "n": 1, "agree": 1.0, "from": "fetch", "smartrc_code": rec.get("f_code")}
                got += 1
            else:
                tried[sid] = datetime.now(JST).date().isoformat()
                miss += 1
            if (got + miss) % 20 == 0:
                _save(doc, path)
    except SmartRCRestricted as e:
        logger.warning("smartrc が制限中のため中断: %s", e)
    rest = [s for s in set(sm["sire_id"]) - set(sires) if s not in tried]
    doc["missing"] = sorted(set(sm["sire_id"]) - set(sires))
    if not rest:
        doc["complete_at"] = datetime.now(JST).isoformat(timespec="seconds")
    _save(doc, path)
    return {"fetched": got, "not_found": miss, "left": len(rest), "known": len(sires)}


COUNTRY_COLORS = {"日本型": "#ddffdd", "米国型": "#BA945B", "欧州型": "#00CCFF"}   # smartrc の画面の色（2026-11-12 に app.js で確認）。取り直しで取れなかったときの補い


def parse_bloods(js: str) -> dict[str, list[dict[str, str]]]:
    """app.js の kLBloods（大系統）・kSBloods（小系統）・kCBloods（国別タイプ）を、項目ごとの辞書のリストにする。"""
    out = {}
    for key in ("kLBloods", "kSBloods", "kCBloods"):
        mm = re.search(key + r":\[(.*?)\]", js)
        if not mm:
            raise RuntimeError(f"{key} が見つかりません（画面の作りが変わった可能性）")
        out[key] = [dict(re.findall(r"(\w+):'([^']*)'", it)) for it in re.findall(r"\{([^{}]*)\}", mm.group(1))]
    return out


def _type_entry(r: dict[str, str]) -> dict[str, Any]:
    country = _country(r["name"])
    return {"name": r["name"], "short": r.get("addr"), "color": r.get("color") or COUNTRY_COLORS.get(country or ""), "country": country}


def _country(type_name: str) -> str | None:
    """国別のタイプ名から国系統（日本型・米国型・欧州型）。smartrc の画面の色分けはこの 3 つ（日本型 #ddffdd・米国型 #BA945B・欧州型 #00CCFF）。"""
    for c in ("日本型", "米国型", "欧州型"):
        if type_name.startswith(c):
            return c
    return None


def colors_table(doc: dict[str, Any] | None = None, m: dict[str, Any] | None = None, names: dict[str, str] | None = None) -> pd.DataFrame:
    """全種牡馬について、smartrc の画面の色の区分を 1 行にする: 種牡馬 ID・名前・大系統（と色）・小系統・国別タイプ・国系統（と色）。
    status: "ok"（系統あり）／"blank"（smartrc で試したが系統が空欄。netkeiba の系統で補う運用）／"unchecked"（台帳に無く、まだ試していない）。
    names は種牡馬 ID → 名前（台帳に名前が無い行に使う。省略時は horse_sire_map から取れれば取る）。"""
    doc = doc or load()
    m = m or master()
    if names is None:
        try:
            sm = pd.read_parquet(SIRE_MAP, columns=["sire_id", "sire_name"]).dropna().drop_duplicates("sire_id")
            names = dict(zip(sm["sire_id"].astype(str), sm["sire_name"].astype(str)))
        except Exception:
            names = {}
    tried = doc.get("tried", {})
    rows = []
    for sid, v in sorted(doc["sires"].items()):
        t = m["type"].get(str(v.get("type") or ""), {})
        rows.append({"sire_id": sid, "name": v.get("name") or names.get(sid), "large_name": v.get("large_name"),
                     "large_color": m["large"].get(str(v.get("large") or ""), {}).get("color"), "small_name": v.get("small_name"),
                     "type_name": v.get("type_name"), "country": t.get("country"), "country_color": t.get("color"), "status": "ok"})
    for sid in doc.get("missing", []):
        rows.append({"sire_id": sid, "name": names.get(sid), "large_name": None, "large_color": None, "small_name": None, "type_name": None,
                     "country": None, "country_color": None, "status": "blank" if sid in tried else "unchecked"})
    return pd.DataFrame(rows)


def refresh_master() -> dict[str, int]:
    """smartrc の画面の app.js から系統名の対応表を取り直す（app.json → app.js の 2 回の取得）。"""
    import requests

    ua = {"User-Agent": "Mozilla/5.0"}
    js_path = requests.get("https://www.smartrc.jp/v3/app.json", headers=ua, timeout=30).json()["js"][0]["path"]
    time.sleep(random.uniform(*INTERVAL))
    s = requests.get(f"https://www.smartrc.jp/v3/{js_path}", headers=ua, timeout=60).text
    out = parse_bloods(s)
    doc = {"source": "smartrc.jp v3 app.js（画面の系統名の対応表 kLBloods/kSBloods/kCBloods）", "fetched": datetime.now(JST).date().isoformat(),
           "large": {r["code"]: {"name": r["name"], "short": r.get("addr"), "color": r.get("color")} for r in out["kLBloods"]},
           "small": {r["code"]: {"name": r["name"], "large": r.get("llcode")} for r in out["kSBloods"]},
           "type": {r["code"]: _type_entry(r) for r in out["kCBloods"]}}
    MASTER.parent.mkdir(parents=True, exist_ok=True)
    MASTER.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k: len(v) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    f = sp.add_parser("fetch")
    f.add_argument("--max", type=int, default=100)
    sp.add_parser("master")
    sp.add_parser("colors")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if a.cmd == "colors":   # 取得なし。台帳と対応表から、全種牡馬の色の区分の一覧を出す
        t = colors_table()
        COLORS_CSV.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(COLORS_CSV, index=False, encoding="utf-8")
        print(json.dumps({"rows": len(t), **t["status"].value_counts().to_dict(), "out": str(COLORS_CSV)}, ensure_ascii=False))
        return 0
    res = build() if a.cmd == "build" else fetch(a.max) if a.cmd == "fetch" else refresh_master()
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
