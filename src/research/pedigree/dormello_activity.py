"""ドルメロ流（8 年周期）の活性値を年単位で計算する。

活性値 = (対象馬の受胎年 - 祖先種牡馬の生年) mod 8。1〜7 は流派によらず同じ値。
0（ドルメロ流）/ 8（自分流）の境界は流派・記事で食い違うため確定させず「要検討ゾーン」とし、
手入力（zone_review.csv の manual_value）が入るまで値は None のままにする。
活性値は (種牡馬, 受胎年) だけで決まるので、手入力もその単位で 1 回ずつ行えばよい。

    python -m src.research.pedigree.dormello_activity build \
        --ped-dirs data/local/horse_pedigree_10gen data/local/horse_pedigree_5gen \
        --birthdays-csv /tmp/horse_birthdays.csv /tmp/ancestor_birthdays.csv --review data/meta/dormello/zone_review.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

CYCLE = 8
GESTATION_DAYS = 340
REVIEW_COLUMNS = ["stallion_id", "stallion_name", "stallion_birth_year", "conception_year", "n_horses", "manual_value", "note"]
_YEAR_ID = re.compile(r"^(19|20)\d{2}\d{6}$")


def conception_year(birth_year: int, birthday: date | None = None) -> int:
    """受胎年。誕生日が分かれば妊娠約 340 日で逆算、無ければ前年（1〜12 月生まれのほぼ全て）。"""
    if birthday is not None:
        return (birthday - timedelta(days=GESTATION_DAYS)).year
    return birth_year - 1


def phase(conception_yr: int, stallion_birth_year: int) -> int:
    return (conception_yr - stallion_birth_year) % CYCLE


def in_review_zone(ph: int) -> bool:
    return ph == 0


def effective_value(ph: int, manual: int | None = None) -> int | None:
    """1〜7 はそのまま。0（= 0 or 8）は手入力があればそれ、無ければ None（未確定）。"""
    if not in_review_zone(ph):
        return ph
    return manual


def birth_year_of(horse_id: str, birthdays: dict[str, str]) -> int | None:
    """生年。birthdays（YYYY-MM-DD）があればそれ、国内 ID なら先頭 4 桁、外国馬・古い馬（000a…）は不明。"""
    b = birthdays.get(horse_id)
    if b:
        return int(b[:4])
    if _YEAR_ID.match(horse_id):
        return int(horse_id[:4])
    return None


def load_birthdays(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["horse_id"]: r["birthday"] for r in csv.DictReader(f) if r.get("birthday")}


def load_manual(path: Path) -> dict[tuple[str, int], tuple[int, str]]:
    """review CSV の手入力。キーは (stallion_id, conception_year)。値は (活性値, メモ)。空欄は未入力。"""
    out: dict[tuple[str, int], tuple[int, str]] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("manual_value") or "").strip()
            if v == "":
                continue
            val = int(v)
            if not 0 <= val <= CYCLE:
                raise ValueError(f"manual_value は 0〜{CYCLE}: {r}")
            out[(r["stallion_id"], int(r["conception_year"]))] = (val, r.get("note") or "")
    return out


def scan_pedigrees(ped_dirs: list[Path], max_gen: int = 10):
    """(対象馬 ID, 世代, 位置, 祖先 ID, 祖先名) を返す。位置が偶数 = 種牡馬。先に書いた dir の馬を優先。"""
    seen: set[str] = set()
    for d in ped_dirs:
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            for f in sorted(sub.glob("*.json")):
                hid = f.stem
                if hid in seen:
                    continue
                seen.add(hid)
                try:
                    anc = json.loads(f.read_text(encoding="utf-8")).get("ancestors") or []
                except Exception:
                    continue
                rows = [
                    (hid, int(a["generation"]), int(a["position"]), a["horse_id"], a.get("name") or "")
                    for a in anc
                    if a.get("horse_id") and int(a["generation"]) <= max_gen and int(a["position"]) % 2 == 0
                ]
                yield hid, rows


def build_review(
    ped_dirs: list[Path],
    birthdays: dict[str, str],
    review_path: Path,
    max_gen: int = 10,
) -> dict:
    """要検討ゾーン (種牡馬, 受胎年) の一覧を作り直す。手入力済みの値・メモは引き継ぐ。"""
    manual = load_manual(review_path)
    zone: dict[tuple[str, int], dict] = {}
    n_targets = n_pairs = n_pairs_known = n_zone_pairs = 0
    male_gen3 = male_gen3_known = 0
    for hid, rows in scan_pedigrees(ped_dirs, max_gen):
        by_h = birth_year_of(hid, birthdays)
        if by_h is None:
            continue
        bday = birthdays.get(hid)
        cy = conception_year(by_h, date.fromisoformat(bday) if bday and len(bday) == 10 else None)
        n_targets += 1
        for _, gen, _, aid, aname in rows:
            n_pairs += 1
            ay = birth_year_of(aid, birthdays)
            if gen <= 3:
                male_gen3 += 1
            if ay is None:
                continue
            n_pairs_known += 1
            if gen <= 3:
                male_gen3_known += 1
            if in_review_zone(phase(cy, ay)):
                n_zone_pairs += 1
                z = zone.setdefault((aid, cy), {"name": aname, "birth_year": ay, "horses": set()})
                z["horses"].add(hid)
    out_rows = []
    for (aid, cy), z in zone.items():
        mv, note = manual.get((aid, cy), ("", ""))
        out_rows.append([aid, z["name"], z["birth_year"], cy, len(z["horses"]), mv, note])
    for (aid, cy), (mv, note) in manual.items():  # 今回の走査に出なかった手入力も落とさない
        if (aid, cy) not in zone:
            out_rows.append([aid, "", "", cy, 0, mv, note])
    out_rows.sort(key=lambda r: (-int(r[4]), r[0], int(r[3])))
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with open(review_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(REVIEW_COLUMNS)
        w.writerows(out_rows)
    return {
        "targets": n_targets,
        "male_ancestor_pairs": n_pairs,
        "pairs_with_birth_year": n_pairs_known,
        "pairs_in_zone": n_zone_pairs,
        "gen<=3_male_pairs": male_gen3,
        "gen<=3_male_pairs_with_birth_year": male_gen3_known,
        "zone_rows": len(zone),
        "manual_entered": sum(1 for r in out_rows if r[5] != ""),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="要検討ゾーンのシートを作り直す（手入力は引き継ぐ）")
    b.add_argument("--ped-dirs", nargs="+", required=True, type=Path)
    b.add_argument("--birthdays-csv", nargs="+", required=True, type=Path,
                   help="horse_id,birthday(YYYY / YYYY-MM / YYYY-MM-DD) の CSV。複数可（後ろが優先）")
    b.add_argument("--review", required=True, type=Path)
    b.add_argument("--max-gen", type=int, default=10)
    a = ap.parse_args(argv)
    births: dict[str, str] = {}
    for p in a.birthdays_csv:
        births.update(load_birthdays(p))
    stats = build_review(a.ped_dirs, births, a.review, a.max_gen)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
