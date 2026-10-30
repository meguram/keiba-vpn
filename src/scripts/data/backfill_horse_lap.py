"""馬ごとの 200m 区間ラップ（db.netkeiba.com の ajax_race_result_horse_laptime）を、平地の全レースぶん取る（ホストで実行）。

python -m src.scripts.data.backfill_horse_lap --years 2026 2025 2024 --out-dir /home/jovyan/acq/horse_lap

1 レース 1 リクエスト。パース結果だけを <out-dir>/horse_lap_<年>.jsonl に追記する（生 HTML は保存しない）。
- 再開できる: 保存済みで頭数が揃っている race_id（--also-done で渡した別ファイルも含む）は飛ばす。例外・頭数不足のレースは保存せず、次回に再取得する。
- ログイン: 個別ラップは会員状態が効いていないと先頭の数頭しか返らない。毎レースの前に「ログイン済み かつ 会員確認 OK」を見て、外れていたら入り直す。
  入り直せなければ取らずに終了する（終了コード 4）。150 リクエストごとのセッション更新で入り直しに失敗した場合もここで拾う。
- 頭数の確認: 取れた馬別ラップの「着順のある馬」が、結果表（race_result_flat）で着順のある馬（除外・取消・中止を除く）と同数のときだけ保存する。
  足りないとき（未掲載・ログイン不全の部分応答）は保存せず、入り直して 1 回だけ取り直す。起動時に保存済みの行も照合し、足りない行は退避して取り直す。
- 止める: <out-dir>/STOP というファイルを置く（次のレースの前で終了）。
- 他の取得とは待ち合わせない（並走してよい）。約 5〜6.5 秒間隔、連続 8 件失敗で中断（終了コード 3）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(os.environ.get("KEIBA_ROOT") or Path(__file__).resolve().parents[3])


def race_ids(years: list[int]) -> list[tuple[int, str]]:
    import pandas as pd

    from src.research.race import running_profile as rp

    out: list[tuple[int, str]] = []
    for y in years:
        runs = rp.load_runs([y])
        if y == 2023:      # 2023 年は重賞だけ馬別ラップが載る
            g = pd.read_parquet(rp.FLAT_DIR / str(y) / "race_result_flat.parquet", columns=["race_id", "grade"])
            runs = runs[runs["race_id"].isin(g.loc[g["grade"].isin(["G1", "G2", "G3"]), "race_id"])]
        runs = runs[runs["date"] <= pd.Timestamp.now().normalize() - pd.Timedelta(days=7)]   # 直近は馬別ラップが載る前なので飛ばす（10/25 開催が未掲載だった）
        ids = runs.drop_duplicates("race_id").sort_values(["date", "race_id"], ascending=False)["race_id"].tolist()
        out += [(y, str(i)) for i in ids]
    return out


def expected_counts(years: list[int]) -> dict[str, int]:
    """race_id → 結果表で着順のある頭数（除外・取消・中止は着順が -1 なので入らない）。"""
    import pandas as pd

    from src.research.race import running_profile as rp

    out: dict[str, int] = {}
    for y in years:
        f = rp.FLAT_DIR / str(y) / "race_result_flat.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["race_id", "finish_position"])
            d = d[pd.to_numeric(d["finish_position"], errors="coerce") > 0]
            out.update({str(k): int(v) for k, v in d.groupby("race_id").size().items()})
    return out


def finishers(d: dict) -> int:
    return sum(1 for e in d.get("entries", []) if (e.get("finish_order") or 0) > 0)


def is_complete(d: dict, expect: dict[str, int]) -> bool:
    n = expect.get(str(d.get("race_id")))
    return bool(n) and finishers(d) == n


def done_ids(paths: list[Path], expect: dict[str, int]) -> set[str]:
    """保存済みのうち頭数が揃っているレース（別ファイルの分も含む）。"""
    s: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if is_complete(d, expect) or (not d.get("entries") and d.get("empty_ok")):
                s.add(str(d["race_id"]))
    return s


def purge_incomplete(path: Path, expect: dict[str, int]) -> int:
    """保存済みの行のうち頭数が揃っていないものを退避して、揃っているものだけ残す。退避した行数を返す。"""
    if not path.exists():
        return 0
    keep, bad = [], []
    seen: set[str] = set()
    for line in path.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            bad.append(line)
            continue
        rid = str(d.get("race_id"))
        if rid in seen:
            continue
        if is_complete(d, expect) or (not d.get("entries") and d.get("empty_ok")):
            keep.append(line)
            seen.add(rid)
        else:
            bad.append(line)
    if not bad:
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path.with_name(f"{path.stem}.incomplete_{stamp}.jsonl").write_text("".join(bad), encoding="utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(keep), encoding="utf-8")
    os.replace(tmp, path)
    return len(bad)


def login_ok(client) -> bool:
    return bool(client.is_logged_in) and client.premium_ok is True


def ensure_login(client, tries: int = 3) -> bool:
    """ログイン済みで会員確認も通っている状態にする。外れていれば入り直す（失敗したら間をあけて繰り返す）。"""
    for i in range(tries):
        if login_ok(client):
            return True
        client._logged_in = False
        client._premium_ok = None
        try:
            client._refresh_session()
            client.login()
        except Exception as e:  # noqa: BLE001
            print(f"ログイン失敗: {str(e)[:120]}", flush=True)
        if login_ok(client):
            print("ログイン OK（会員確認済み）", flush=True)
            return True
        print(f"ログイン不全 premium_ok={client.premium_ok}（{i + 1}/{tries} 回目）", flush=True)
        time.sleep(30 * (i + 1))
    return login_ok(client)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2026, 2025, 2024])
    ap.add_argument("--out-dir", default="/home/jovyan/acq/horse_lap")
    ap.add_argument("--also-done", nargs="*", default=[], help="取得済みとみなす別の jsonl")
    ap.add_argument("--limit", type=int, default=0, help="この回に取る最大レース数（0=すべて）")
    a = ap.parse_args()

    from src.scraper.parsers import DbHorseLaptimeTableParser
    from src.scraper.run import ScraperRunner

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = out_dir / "STOP"
    expect = expected_counts(a.years)
    for y in a.years:
        n = purge_incomplete(out_dir / f"horse_lap_{y}.jsonl", expect)
        if n:
            print(f"{y} 年の保存済みのうち頭数が揃っていない {n} 行を退避した", flush=True)
    todo_all = race_ids(a.years)
    done = done_ids([out_dir / f"horse_lap_{y}.jsonl" for y in a.years] + [Path(p) for p in a.also_done], expect)
    todo = [(y, r) for y, r in todo_all if r not in done]
    print(f"対象 {len(todo_all)} レース / 取得済み {len(todo_all) - len(todo)} / 残り {len(todo)}", flush=True)

    runner = ScraperRunner()
    client = runner.client
    parser = DbHorseLaptimeTableParser()
    if not ensure_login(client):
        print("ログインできないので終了", flush=True)
        raise SystemExit(4)

    def fetch_one(rid: str) -> dict:
        html = client.fetch(parser.AJAX_URL.format(race_id=rid), use_cache=False, encoding="euc-jp")
        return parser.parse(html, race_id=rid)

    streak = got = 0
    for y, rid in todo:
        if stop.exists():
            print("STOP ファイルがあるので終了", flush=True)
            return
        if a.limit and got >= a.limit:
            return
        if not ensure_login(client):
            print("ログインが外れて戻せないので終了", flush=True)
            raise SystemExit(4)
        n_exp = expect.get(rid, 0)
        d, status = {"race_id": rid, "entries": []}, "ERR"
        for attempt in range(2):
            try:
                d = fetch_one(rid)
                n = finishers(d)
                status = "ok" if (n_exp and n == n_exp) else ("EMPTY" if not d.get("entries") else "SHORT")
            except Exception as e:  # noqa: BLE001
                d, status = {"race_id": rid, "error": str(e)[:200], "entries": []}, "ERR"
            if status == "ok" or attempt == 1:
                break
            time.sleep(random.uniform(5.0, 6.5))
            if not ensure_login(client):      # 取り直す前に、入り直す必要がないか見る（150 リクエストごとの更新で外れた場合）
                break
            if status != "ERR":
                client._logged_in = False      # 欠けた応答は会員状態が効いていない疑いがあるので、新しいセッションで取り直す
                if not ensure_login(client):
                    break
        if status == "ok":
            d["expected"] = n_exp
        elif status == "EMPTY" and login_ok(client):
            d["empty_ok"] = True
        if status in ("ok", "EMPTY") and (status == "ok" or d.get("empty_ok")):
            with (out_dir / f"horse_lap_{y}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        streak = 0 if status == "ok" else streak + 1
        got += 1
        print(rid, status, f"{finishers(d)}/{n_exp}", flush=True)
        if streak >= 8:
            print("連続 8 件取れなかったので終了", flush=True)
            raise SystemExit(3)
        time.sleep(random.uniform(5.0, 6.5))


if __name__ == "__main__":
    main()
