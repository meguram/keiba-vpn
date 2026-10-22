"""dev 環境の UI を表示するための実データ 1 開催日を、各自のローカルに置いて使う道具。

**実データのアーカイブ（``dev_fixture/real_day.tar.gz``）は git 管理しない**（公開リポジトリに netkeiba の取得データを置かない。
netkeiba 利用規約 第14条。``dev_fixture/.gitignore`` で除外）。アーカイブが無い PC では、合成モックだけで dev の UI が動く。

    python3 -m src.scripts.data.dev_fixture ensure      # make dev-data と同じ

  1. モック（``make dev-mock`` が生成する分）が無ければ生成する
  2. ローカルに ``dev_fixture/real_day.tar.gz`` があれば、その実データを dev ストアへ展開する（既にある実データは上書きしない）。
     無ければ何もしない（モックのまま）

自分の PC に実データ日を持つ方法（任意。取得は自分の netkeiba アカウント・自分の責任で）:
    python3 -m src.scripts.scraping.capture_dev_day --date YYYYMMDD --yes   # 取り込み（実アクセスあり）
    python3 -m src.scripts.data.dev_fixture pack --date YYYYMMDD            # dev_fixture/ にローカル保存（git には載らない）

アーカイブの中身: UI の最小動線（週間予測 → レース詳細 → 推奨馬券）に要るカテゴリだけ
（race_lists / race_day_schedule / race_shutuba / race_result / race_odds / race_pair_odds /
 race_predictions / final_odds_prediction）。AI 予測と予測オッズは実馬・実オッズから作ったモック。
アーカイブは再現可能な形式（ファイル順固定・mtime 0）。

注意: 中身は netkeiba から取得したデータの抜粋。再配布・公開しない。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "dev_fixture"
ARCHIVE = FIXTURE_DIR / "real_day.tar.gz"
MANIFEST = FIXTURE_DIR / "MANIFEST.json"

# (ストア, カテゴリ, キー種別) — race: 開催日の各 race_id / date: 開催日そのもの
STORE_DEV_MOCK = "dev_mock"
STORE_PAGE_REF = "page_reference"
CATEGORIES: list[tuple[str, str, str]] = [
    (STORE_PAGE_REF, "race_lists", "date"),
    (STORE_PAGE_REF, "race_day_schedule", "date"),
    (STORE_DEV_MOCK, "race_shutuba", "race"),
    (STORE_DEV_MOCK, "race_result", "race"),
    (STORE_DEV_MOCK, "race_odds", "race"),
    (STORE_DEV_MOCK, "race_pair_odds", "race"),
    (STORE_DEV_MOCK, "race_predictions", "race"),
    (STORE_DEV_MOCK, "final_odds_prediction", "race"),
]


def _roots() -> dict[str, Path]:
    from src.scraper.dev_store import dev_mock_root

    ref = os.environ.get("KEIBA_PAGE_REFERENCE_DIR", "").strip()
    return {
        STORE_DEV_MOCK: dev_mock_root(REPO_ROOT),
        STORE_PAGE_REF: Path(ref).expanduser().resolve() if ref else REPO_ROOT / "data" / "page_reference",
    }


def _path(roots: dict[str, Path], store: str, cat: str, key: str, kind: str) -> Path:
    if kind == "date":
        return roots[store] / cat / f"{key}.json"
    return roots[store] / cat / key[:4] / f"{key}.json"


def _canonical(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def pack(date: str) -> dict[str, Any]:
    """現在の dev ストアから指定日のデータを集めて dev_fixture/ に書く。"""
    roots = _roots()
    lst = roots[STORE_PAGE_REF] / "race_lists" / f"{date}.json"
    if not lst.is_file():
        raise SystemExit(f"{lst} がありません。先に capture_dev_day で {date} を取り込んでください")
    race_ids = [r["race_id"] for r in json.loads(lst.read_text(encoding="utf-8"))["races"]]

    members: dict[str, bytes] = {}
    counts: dict[str, int] = {}
    for store, cat, kind in CATEGORIES:
        keys = [date] if kind == "date" else race_ids
        for key in keys:
            p = _path(roots, store, cat, key, kind)
            if not p.is_file():
                continue
            rel = p.relative_to(roots[store]).as_posix()
            members[f"{store}/{rel}"] = _canonical(json.loads(p.read_text(encoding="utf-8")))
            counts[cat] = counts.get(cat, 0) + 1
    missing = {c: len(race_ids) - n for c, n in counts.items() if _kind_of(c) == "race" and n != len(race_ids)}
    if missing:
        raise SystemExit(f"レース分が揃っていないカテゴリがあります: {missing}")

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name in sorted(members):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = len(members[name]), 0, 0, 0, "", "", 0o644
            tf.addfile(ti, io.BytesIO(members[name]))
    FIXTURE_DIR.mkdir(exist_ok=True)
    with open(ARCHIVE, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())

    manifest = {
        "date": date, "races": len(race_ids), "counts": counts,
        "archive": ARCHIVE.name, "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "note": "netkeiba 取得データの抜粋（出馬表・結果・オッズ）。AI 予測/予測オッズはモック",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _kind_of(cat: str) -> str:
    return next(k for _, c, k in CATEGORIES if c == cat)


def _is_mock_file(p: Path) -> bool:
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("_meta", {}).get("dev_mock"))
    except (OSError, ValueError):
        return False


def install(force: bool = False) -> dict[str, int]:
    """アーカイブを dev ストアへ展開する。既存の実データは force でなければ上書きしない（モックは上書き）。"""
    if not ARCHIVE.is_file():
        raise SystemExit(f"{ARCHIVE} がありません（実データ日は各自のローカルに作る。このファイルの docstring を参照）")
    roots = _roots()
    out = {"written": 0, "kept": 0}
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for m in tf.getmembers():
            store, _, rel = m.name.partition("/")
            if store not in roots or ".." in rel.split("/"):
                raise SystemExit(f"想定外のアーカイブ内パス: {m.name}")
            dest = roots[store] / rel
            if dest.exists() and not force and not _is_mock_file(dest):
                out["kept"] += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tf.extractfile(m).read())  # type: ignore[union-attr]
            out["written"] += 1
    return out


def status() -> dict[str, Any]:
    roots = _roots()
    man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    date = man.get("date", "")
    lst = roots[STORE_PAGE_REF] / "race_lists" / f"{date}.json"
    have = 0
    if lst.is_file():
        ids = [r["race_id"] for r in json.loads(lst.read_text(encoding="utf-8"))["races"]]
        have = sum(_path(roots, STORE_DEV_MOCK, "race_shutuba", i, "race").is_file() for i in ids)
    return {
        "fixture_date": date, "fixture_races": man.get("races"),
        "mock_generated": (roots[STORE_DEV_MOCK] / "race_shutuba").is_dir(),
        "real_day_installed_races": have,
    }


def ensure() -> None:
    """モック未生成なら生成し、実データ日が未展開なら展開する（冪等）。"""
    st = status()
    if not st["mock_generated"]:
        print("[dev_fixture] モックを生成します（make dev-mock 相当）", flush=True)
        from src.scripts.data import make_dev_mock

        old = sys.argv
        sys.argv = ["make_dev_mock"]
        try:
            make_dev_mock.main()
        finally:
            sys.argv = old
        st = status()
    if not ARCHIVE.is_file():
        print("[dev_fixture] OK: 実データのアーカイブはありません（合成モックで動きます）", flush=True)
    elif st["fixture_races"] and st["real_day_installed_races"] < st["fixture_races"]:
        r = install()
        print(f"[dev_fixture] 実データ {st['fixture_date']} を展開しました: 書き込み {r['written']} / 既存維持 {r['kept']}", flush=True)
    else:
        print(f"[dev_fixture] OK: {st}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("pack", help="dev ストアから dev_fixture/ を作る")
    pk.add_argument("--date", required=True, help="YYYYMMDD")
    ins = sub.add_parser("install", help="dev_fixture/ を dev ストアへ展開")
    ins.add_argument("--force", action="store_true", help="既存の実データも上書き")
    sub.add_parser("ensure", help="モック生成 + 実データ展開（冪等）")
    sub.add_parser("status", help="状態を表示")
    args = ap.parse_args(argv)

    if args.cmd == "pack":
        print(json.dumps(pack(args.date), ensure_ascii=False, indent=2))
    elif args.cmd == "install":
        print(install(args.force))
    elif args.cmd == "ensure":
        ensure()
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
