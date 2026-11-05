"""データセットの全種牡馬について、父系を最深部まで辿り、各段の『○○系』（netkeiba の血統ページの表記）を揃える。

python -m src.scripts.scraping.trace_sire_line_origins            # 足りない分だけ取得して、結果を書く
python -m src.scripts.scraping.trace_sire_line_origins --offline  # 取得せず、手元のデータだけで結果を書く
python -m src.scripts.scraping.trace_sire_line_origins --compact-only  # 手元の出力から、git 管理の小さな版だけを書き直す

1. 父系の鎖: ローカルの 5 世代血統（data/local/horse_pedigree_5gen）から 子 -> 父 を集め、2016 年以降の出走馬の父（種牡馬）から上へ辿る。
   古い馬（18 世紀）は、子孫の 5 世代表では位置がずれて『父』が誤って載ることがある。自分のページで父が無いと分かった馬
   （deep_male_line の no_father。例: 三大始祖）は、そこを鎖の上端にする。
2. 鎖の上端（父の分からない馬）は、その馬自身の血統ページを取り、父が載っていれば鎖を伸ばす（父が載らなくなるまで繰り返す）。
   取ったページは data/local/pedigree_lines/deep_male_line/<id>.json に 5 世代表として置く（共有の horse_pedigree_5gen には書かない）。
3. 系統: 鎖の各馬の『○○系』は、その子の血統ページの父のセルにしか付かないので、台帳（sire_line_map.json）に無い馬は子のページを取る。
4. 出力 data/local/pedigree_lines/sire_line_origin.json:
   {sire_id: {name, chain: [[id, name, line], ...（種牡馬から上へ）], lines: [系統の移り変わり], origin_line: 最も古い系統, root: [id, name]}}
5. 小さな版 data/meta/pedigree_lines/sire_line_origin.json（git の管理下。めぐコラム sire_line_treemap が読む）:
   {sire_id: {name, chain: [[id, name], ...（父から、名前の付いた最も下の系統の祖まで）], lines, origin_line, root}}
   chain は line_path の最後の系統の祖で切る（それより上の祖先は系統名の経路 lines で足りる）。祖先の系統名は持たない。
取得は同時 1 本・一定間隔（既定 3 秒）。アクセス制限の疑いが出たら止める。
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import random
import time
from pathlib import Path

import pandas as pd

from src.scraper import sire_line_store as store

logger = logging.getLogger("trace_sire_line_origins")
REPO = Path(__file__).resolve().parents[3]
PED_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"
DEEP_DIR = REPO / "data/local/pedigree_lines/deep_male_line"
OUT = REPO / "data/local/pedigree_lines/sire_line_origin.json"
COMPACT_OUT = REPO / "data/meta/pedigree_lines/sire_line_origin.json"
MAX_CHAIN = 80


def _anc_map(rec: dict) -> dict:
    return {(a["generation"], a["position"]): a for a in rec.get("ancestors") or [] if a.get("generation") is not None}


def load_links() -> tuple[dict[str, str], dict[str, str], set[str]]:
    """子 -> 父、ID -> 名前、手元で自分の血統ページ（5 世代表）を持つ馬。deep_male_line は共有のデータより優先する。"""
    sire_of: dict[str, str] = {}
    name: dict[str, str] = {}
    own: set[str] = set()

    def add(rec: dict, override: bool) -> None:
        hid = str(rec["horse_id"])
        own.add(hid)
        anc = _anc_map(rec)
        pairs = [(hid, anc.get((1, 0)))] + [((anc.get((g, p)) or {}).get("horse_id"), anc.get((g + 1, p * 2)))
                                             for g in range(1, 5) for p in range(2 ** g)]
        for i, (child, a) in enumerate(pairs):
            if not (child and a and a.get("horse_id")):
                continue
            # 自分のページの 1 世代目の父は確か。それより上の段は、古い馬では位置がずれていることがあるので上書きしない
            if override and i == 0:
                sire_of[str(child)] = str(a["horse_id"])
            else:
                sire_of.setdefault(str(child), str(a["horse_id"]))
            name[str(a["horse_id"])] = a.get("name") or ""

    for f in glob.glob(str(REPO / "data/local/horse_pedigree_5gen/*/*.json")):
        add(json.loads(Path(f).read_text(encoding="utf-8")), False)
    deep_no_father: set[str] = set()
    for f in glob.glob(str(DEEP_DIR / "*.json")):
        rec = json.loads(Path(f).read_text(encoding="utf-8"))
        if rec.get("no_father"):
            deep_no_father.add(str(rec["horse_id"]))
            own.add(str(rec["horse_id"]))
            continue
        add(rec, True)
    for h in deep_no_father:      # 自分のページに父が無い = 鎖の本当の上端
        sire_of.pop(h, None)
    return sire_of, name, own


def dataset_sires(sire_of: dict[str, str]) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for fp in sorted(glob.glob(str(REPO / "data/page_reference/tables/*/race_result_flat.parquet"))):
        if int(Path(fp).parent.name) < 2016:
            continue
        for h in pd.read_parquet(fp, columns=["horse_id"])["horse_id"].astype(str).unique():
            if h in sire_of:
                c[sire_of[h]] += 1
    return c


def chain_of(s: str, sire_of: dict[str, str]) -> list[str]:
    ch = [s]
    while ch[-1] in sire_of and sire_of[ch[-1]] not in ch and len(ch) < MAX_CHAIN:
        ch.append(sire_of[ch[-1]])
    return ch


def fetch_own(client, hid: str) -> bool:
    """馬自身の血統ページを取り、5 世代表として deep_male_line に置く。父の無いページは no_father として残す。"""
    from src.research.pedigree.pedigree_similarity import parse_blood_table_5gen
    from src.scraper.sire_line_extractor import extract_direct_sire, extract_sires_from_ped_html

    html = client.fetch(PED_URL.format(horse_id=hid))
    if not extract_sires_from_ped_html(html) and not parse_blood_table_5gen(html):
        father = None
    else:
        store.record_html(html, hid)      # このページに載る系統（父のセルなど）も台帳に足す
        father = extract_direct_sire(html)
    DEEP_DIR.mkdir(parents=True, exist_ok=True)
    anc = parse_blood_table_5gen(html) if father else []
    rec = {"horse_id": hid, "ancestors": anc, "no_father": not father, "source": "trace_sire_line_origins",
           "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if father and not any(a.get("generation") == 1 and a.get("position") == 0 for a in anc):
        rec["ancestors"] = [{"generation": 1, "position": 0, "horse_id": str(father.get("horse_id")), "name": father.get("name")}] + anc
    (DEEP_DIR / f"{hid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return bool(father)


def build_output(sires: collections.Counter, sire_of: dict[str, str], name: dict[str, str]) -> dict:
    L = store.load()
    out = {}
    for s in sires:
        ch = chain_of(s, sire_of)
        rows = [[h, (L.get(h) or {}).get("name") or name.get(h, ""), (L.get(h) or {}).get("line")] for h in ch]
        lines = []
        for _, _, ln in rows:
            if ln and (not lines or lines[-1] != ln):
                lines.append(ln)
        out[s] = {"name": rows[0][1], "n_runners": int(sires[s]), "chain": rows, "lines": lines,
                  "origin_line": lines[-1] if lines else None, "root": rows[-1][:2]}
    return out


def line_path(lines: list[str | None]) -> list[str]:
    """lines（新しい→古い）を、根から下への経路にする。同じ系統名が 2 回目以降に出たら飛ばす。"""
    out: list[str] = []
    for ln in reversed(lines):
        if ln and ln not in out:
            out.append(ln)
    return out


def compact_record(rec: dict) -> dict:
    """出力の 1 頭分を、小さな版にする。chain は父から、line_path の最後の系統が付いた最も上の馬（その系統の祖）まで。"""
    path = line_path(rec["lines"])
    last = path[-1] if path else None
    chain = rec["chain"]
    founder = max((j for j, c in enumerate(chain) if c[2] == last), default=0)
    return {"name": rec["name"], "chain": [c[:2] for c in chain[1:max(founder, 1) + 1]], "lines": rec["lines"],
            "origin_line": rec["origin_line"], "root": rec["root"]}


def dumps_compact(out: dict) -> str:
    """1 頭 1 行の JSON（差分が読みやすく、小さい）。"""
    rows = [json.dumps(k) + ": " + json.dumps(compact_record(v), ensure_ascii=False, separators=(",", ":"))
            for k, v in out.items()]
    return "{\n" + ",\n".join(rows) + "\n}\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--pace", type=float, default=3.0)
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--compact-only", action="store_true", help="取得も鎖の組み直しもせず、手元の出力から小さな版だけを書く")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.compact_only:
        COMPACT_OUT.parent.mkdir(parents=True, exist_ok=True)
        COMPACT_OUT.write_text(dumps_compact(json.loads(OUT.read_text(encoding="utf-8"))), encoding="utf-8")
        logger.info("書き込み %s", COMPACT_OUT)
        return 0

    client = None
    if not args.offline:
        from src.scraper.client import NetkeibaClient
        from src.scraper.scrape_access_pause import read_access_pause

        if (read_access_pause() or {}).get("active"):
            logger.error("アクセス停止の印が立っているので取得しません")
            return 2
        client = NetkeibaClient(interval=args.pace, cache_dir=None, auto_login=False)

    tried: set[str] = set()      # 取りに行った馬（同じ回で取れなかったものを何度も取り直さない）
    for rnd in range(args.max_rounds):
        sire_of, name, own = load_links()
        sires = dataset_sires(sire_of)
        chains = {s: chain_of(s, sire_of) for s in sires}
        L = store.load()
        tops = sorted({ch[-1] for ch in chains.values()} - own - tried)                    # 上端で、まだ自分のページを見ていない馬
        child_of = {}
        for ch in chains.values():
            for a, b in zip(ch, ch[1:]):
                child_of.setdefault(b, a)
        unlabeled = sorted(h for h in child_of if h not in L and h not in tried)                      # 台帳に無い（子のページを見ていない）馬
        logger.info("第 %d 回: 種牡馬 %d・鎖の馬 %d・上端の未確認 %d・系統の未取得 %d", rnd + 1, len(sires),
                    len({h for ch in chains.values() for h in ch}), len(tops), len(unlabeled))
        if client is None or not (tops or unlabeled):
            break
        try:
            tried.update(tops)
            tried.update(unlabeled)
            for h in tops:
                ok = fetch_own(client, h)
                logger.info("上端 %s %s: %s", h, name.get(h, ""), "父あり→鎖を伸ばす" if ok else "父なし（最深部）")
                time.sleep(args.pace * random.uniform(0.3, 0.7))
            for h in unlabeled:
                st = store.fetch_and_record(client.fetch, PED_URL.format(horse_id=child_of[h]), h, child_of[h])
                logger.info("系統 %s %s: %s", h, name.get(h, ""), store.labels().get(h) or st)
                time.sleep(args.pace * random.uniform(0.3, 0.7))
        except Exception as exc:  # noqa: BLE001
            logger.error("取得を止めます（アクセス制限の疑いなど）: %s", exc)
            return 2

    sire_of, name, _ = load_links()
    sires = dataset_sires(sire_of)
    out = build_output(sires, sire_of, name)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    COMPACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    COMPACT_OUT.write_text(dumps_compact(out), encoding="utf-8")
    c = collections.Counter(v["origin_line"] for v in out.values())
    r = collections.Counter(tuple(v["root"]) for v in out.values())
    logger.info("書き込み %s・%s: 種牡馬 %d / 最も古い系統 %s / 上端 %s", OUT, COMPACT_OUT, len(out), c.most_common(), r.most_common(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
