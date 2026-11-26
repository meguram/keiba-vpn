"""tp-48: 3連単の実オッズ（race_trifecta_odds）を取り出して表にし、確定払戻との整合を確かめる。

python -m src.scripts.data.tp48_trifecta collect --years 2024 2025 2026     # ストレージ（stg: GCS）から全組合せを parquet へ
python -m src.scripts.data.tp48_trifecta qa --years 2024 2025 2026          # 払戻との一致・Σ(1/odds)・組合せ数・除外リスト
python -m src.scripts.data.tp48_trifecta collect-snap --years 2016 ... 2026  # 確定と発走前（T-60・T-15）を phase・取得時刻つきで手元の parquet へ
python -m src.scripts.data.tp48_trifecta qa --dir data/local/modeling/betting/trifecta_odds --years 2016 ... 2026

出力: data/local/modeling/betting/tp48/trifecta_<年>.parquet（race_id, a, b, c, odds, pop）と qa.json / exclude.json。
オッズは確定後に取得した値で、購入時点の値ではない。

collect-snap の出力: data/local/modeling/betting/trifecta_odds/trifecta_<年>.parquet（1 行 = レース × 段階 × 組合せ。
列 race_id, phase〔final / t60 / t15〕, phase_raw, snapshot_at, scraped_at_jst, source, a, b, c, odds, pop, n_head, expected）と
missing_<年>.json・collect_<年>.json。読む順は (1) 手元の写し data/local/mirror/race_trifecta_odds（0 円）、(2) 既存の tp48 の parquet
（2026-10-20 に集めた確定。開催日が 2026-10-19 より前のレースだけ）、(3) 確定がまだ無いレースだけ GCS の文書を 1 件ずつ読む
（一覧はしない。GCP の外から読むと転送料 約 18 円/GiB。--max-gcs-reads で上限をかける。読んだ文書は <出力>/_docs/ に残し、次の回はそこから読む）。
phase_raw が空の行は、段階を付けずに取った値（過去分の補填は確定後の値。段階を付けない取得を発走前に流した文書もありうる）。
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("TP48_OUT", str(ROOT / "data/local/modeling/betting/tp48")))
TABLES = ROOT / "data/page_reference/tables"
RATE = 0.725


def race_ids(year: int) -> list[str]:
    f = TABLES / str(year) / "race_result_flat.parquet"
    d = pd.read_parquet(f, columns=["race_id"])
    return sorted({str(r) for r in d["race_id"].dropna().unique()})


def doc_to_frame(doc: dict) -> pd.DataFrame | None:
    t = doc.get("trifecta") if isinstance(doc, dict) else None
    if not t:
        return None
    arr = np.asarray(t, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None
    return pd.DataFrame({
        "race_id": str(doc.get("race_id") or ""),
        "a": arr[:, 0].astype(np.int8), "b": arr[:, 1].astype(np.int8), "c": arr[:, 2].astype(np.int8),
        "odds": arr[:, 3].astype(np.float32), "pop": (arr[:, 4] if arr.shape[1] > 4 else np.zeros(len(arr))).astype(np.int16),
    })


def collect(a) -> None:
    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    OUT.mkdir(parents=True, exist_ok=True)
    for year in a.years:
        ids = race_ids(year)
        print(year, "races", len(ids), flush=True)

        def load(rid: str):
            try:
                d = st.load("race_trifecta_odds", rid)
            except Exception as e:      # noqa: BLE001
                return rid, None, str(e)
            return rid, d, ""

        frames, missing = [], []
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for k, (rid, d, err) in enumerate(ex.map(load, ids)):
                f = doc_to_frame(d) if d else None
                if f is None:
                    missing.append(rid)
                else:
                    f["race_id"] = rid
                    frames.append(f.assign(n_head=int(d.get("n_head") or 0), expected=int(d.get("expected_combos") or 0)))
                if k % 500 == 0:
                    print(year, k, len(frames), len(missing), flush=True)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        df.to_parquet(OUT / f"trifecta_{year}.parquet", index=False)
        (OUT / f"missing_{year}.json").write_text(json.dumps(missing), encoding="utf-8")
        print(year, "saved", len(df), "rows /", df["race_id"].nunique() if len(df) else 0, "races; missing", len(missing), flush=True)


TRI_CAT = "race_trifecta_odds"
SNAP_OUT = "data/local/modeling/betting/trifecta_odds"
SNAP_COLS = ["race_id", "phase", "phase_raw", "snapshot_at", "scraped_at_jst", "source", "a", "b", "c", "odds", "pop", "n_head", "expected"]


def doc_to_phase_frames(doc: dict, source: str) -> list[pd.DataFrame]:
    """1 文書 → 段階ごとの表。最上位（最後に取った段階。phase が無ければ確定後の補填）と snapshots の各段階。"""
    out = []
    meta = doc.get("_meta") or {}
    top_raw = doc.get("phase")
    entries = [(str(top_raw) if top_raw else "final", top_raw, doc.get("snapshot_at"), meta.get("scraped_at_jst"), doc)]
    for snap in doc.get("snapshots") or []:
        if isinstance(snap, dict) and snap.get("phase"):
            entries.append((str(snap["phase"]), snap["phase"], snap.get("snapshot_at"), None, snap))
    seen = set()
    for phase, raw, at, scraped, part in entries:
        if phase in seen:      # 最上位と同じ段階が snapshots にも残っていれば、最上位（新しい方）を採る
            continue
        f = doc_to_frame({"race_id": doc.get("race_id"), "trifecta": part.get("trifecta")})
        if f is None:
            continue
        seen.add(phase)
        out.append(f.assign(phase=phase, phase_raw=raw, snapshot_at=at, scraped_at_jst=scraped, source=source,
                            n_head=int(part.get("n_head") or doc.get("n_head") or 0),
                            expected=int(part.get("expected_combos") or doc.get("expected_combos") or 0)))
    return out


def _mirror_path(base: Path, rid: str) -> Path:
    return base / "data/local/mirror" / TRI_CAT / rid[:4] / f"{rid}.json"


def _read_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _open_bucket(base: Path):
    """GCS のバケットと blob の接頭辞・リトライ。テストでは差し替える。"""
    from src.config import data_paths as dp
    from src.scraper.storage import HybridStorage

    return HybridStorage(base_dir=str(base))._get_bucket(), dp.GCS_BASE, HybridStorage._build_retry()


def _has_final(parts: list[pd.DataFrame]) -> bool:
    return any(str(p["phase"].iat[0]) == "final" for p in parts)


TRIFECTA_LIVE_SINCE = "2026-10-19"      # これより前の開催の文書は、確定後の補填だけ（発走前の段階は無い）。既存の tp48 の parquet はこの範囲だけ使う


def collect_snap(a) -> None:
    """確定と発走前の 3連単オッズを年ごとの parquet にする。年の出力が既にあれば飛ばす（--force で作り直し）。

    GCS から読んだ文書は <out>/_docs/<年>/ に残し、次の回はそこから読む（止めて流し直しても転送料を二重に払わない）。
    確定の無いレース（発走前の段階だけ）は no_final_<年>.json に、どこにも無いレースは missing_<年>.json に書く。"""
    base = Path(a.base).resolve()
    out_dir = Path(a.out).resolve() if a.out else base / SNAP_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    old_dir = base / "data/local/modeling/betting/tp48"
    stop_at = pd.Timestamp(a.stop_at) if a.stop_at else None
    gcs: dict = {}
    reads = {"n": 0, "bytes": 0, "errors": 0}

    def gcs_doc(rid: str) -> dict | None:
        """GCS の文書を 1 件読み、_docs に残す。無ければ None。読めなければ errors を数えて None。"""
        if not gcs:
            gcs["bucket"], gcs["base"], gcs["retry"] = _open_bucket(base)
        from google.api_core.exceptions import NotFound

        reads["n"] += 1
        try:
            raw = gcs["bucket"].blob(f"{gcs['base']}/{TRI_CAT}/{rid[:4]}/{rid}.json").download_as_bytes(timeout=60, retry=gcs["retry"])
        except NotFound:
            nf = out_dir / "_docs" / rid[:4] / f"{rid}.404"      # 無いことも残す（流し直しで読み直さない）
            nf.parent.mkdir(parents=True, exist_ok=True)
            nf.write_text("", encoding="utf-8")
            return None
        except Exception as e:      # noqa: BLE001
            errors.append(rid)
            reads["errors"] += 1
            print(f"GCS を読めない {rid}: {e}", flush=True)
            return None
        reads["bytes"] += len(raw)
        cp = out_dir / "_docs" / rid[:4] / f"{rid}.json"
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(raw)
        try:
            d = json.loads(raw)
        except ValueError:
            return None
        return d if isinstance(d, dict) else None

    for year in a.years:
        dst = out_dir / f"trifecta_{year}.parquet"
        if dst.exists() and not a.force:
            print(year, "既にある（飛ばす）", dst, flush=True)
            continue
        rr = pd.read_parquet(base / "data/page_reference/tables" / str(year) / "race_result_flat.parquet", columns=["race_id", "date"])
        rr = rr.dropna(subset=["race_id"]).assign(race_id=lambda x: x["race_id"].astype(str), date=lambda x: x["date"].astype(str).str[:10])
        dates = rr.groupby("race_id")["date"].max().to_dict()
        ids = sorted(dates)
        old = None
        f_old = old_dir / f"trifecta_{year}.parquet"
        if f_old.exists():
            old = {rid: g for rid, g in pd.read_parquet(f_old).groupby("race_id")}
        frames, missing, no_final, errors = [], [], [], []
        src_n = {"mirror": 0, "docs": 0, "tp48_20261020": 0, "gcs": 0}
        for k, rid in enumerate(ids):
            if stop_at is not None and pd.Timestamp.now(tz=stop_at.tz) >= stop_at:
                print(f"期限 {a.stop_at} を過ぎたので止める（{year} 年は書かない。読んだ文書は _docs に残る）", flush=True)
                raise SystemExit(2)
            parts: list[pd.DataFrame] = []
            used: list[str] = []
            docs_p = out_dir / "_docs" / rid[:4] / f"{rid}.json"
            for src, p in (("mirror", _mirror_path(base, rid)), ("docs", docs_p)):
                d = _read_json(p) if p.exists() else None
                got = doc_to_phase_frames(d, src) if d else []
                if got and (_has_final(got), len(got)) > (_has_final(parts), len(parts)):      # 確定のある方、段階の多い方を採る
                    parts, used = got, [src]
            if not _has_final(parts) and old is not None and rid in old and dates[rid] < TRIFECTA_LIVE_SINCE:
                parts = [p for p in parts if str(p["phase"].iat[0]) != "final"] + [
                    old[rid].assign(phase="final", phase_raw=None, snapshot_at=None, scraped_at_jst=None, source="tp48_20261020")]
                used.append("tp48_20261020")
            read_before = docs_p.exists() or docs_p.with_suffix(".404").exists()      # 前の回に GCS で読んだ（無かった）文書は読み直さない
            if not _has_final(parts) and not a.no_gcs and not read_before:
                if reads["n"] >= a.max_gcs_reads:
                    print(f"GCS の読み出しが上限 {a.max_gcs_reads} 件に達したので止める（{year} 年は書かない。読んだ文書は _docs に残る）", flush=True)
                    raise SystemExit(3)
                d = gcs_doc(rid)
                got = doc_to_phase_frames(d, "gcs") if d else []
                if got:
                    parts, used = got, ["gcs"]
                if reads["errors"] >= a.max_gcs_errors:
                    print(f"GCS の読み出しの失敗が {reads['errors']} 件になったので止める（{year} 年は書かない）", flush=True)
                    raise SystemExit(4)
            for s in used:
                src_n[s] += 1
            if parts:
                for p in parts:
                    p["race_id"] = rid
                frames.extend(parts)
                if not _has_final(parts):
                    no_final.append(rid)
            else:
                missing.append(rid)
            if k % 500 == 0:
                print(year, k, src_n, "missing", len(missing), "gcs reads", reads["n"], f"{reads['bytes'] / 2**30:.3f} GiB", flush=True)
        old = None
        df = pd.concat(frames, ignore_index=True)[SNAP_COLS] if frames else pd.DataFrame(columns=SNAP_COLS)
        del frames
        for c in ("phase", "phase_raw", "source", "snapshot_at", "scraped_at_jst"):
            df[c] = df[c].astype("category")
        by_phase = {str(p): int(n) for p, n in df.drop_duplicates(["race_id", "phase"])["phase"].value_counts().items()} if len(df) else {}
        rep = {"year": year, "races": len(ids), "races_by_phase": by_phase, "source": src_n, "missing": len(missing), "no_final": len(no_final), "errors": len(errors),
               "gcs_reads_total": reads["n"], "gcs_gib_total": round(reads["bytes"] / 2**30, 4), "gcs_errors_total": reads["errors"], "rows": int(len(df))}
        (out_dir / f"missing_{year}.json").write_text(json.dumps(missing), encoding="utf-8")
        (out_dir / f"no_final_{year}.json").write_text(json.dumps(no_final), encoding="utf-8")
        (out_dir / f"errors_{year}.json").write_text(json.dumps(errors), encoding="utf-8")
        (out_dir / f"collect_{year}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        if errors:      # 一時的な失敗が残る年は書かない（次の回に失敗したレースだけ読み直す。ほかは _docs から 0 円）
            print(json.dumps(rep, ensure_ascii=False), flush=True)
            print(f"{year} 年は GCS の読み出しの失敗が {len(errors)} 件あるので parquet を書かない（流し直す）", flush=True)
            continue
        tmp = dst.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(dst)      # 置き換えは最後（途中で落ちても「既にある」で飛ばされない）
        del df
        print(json.dumps(rep, ensure_ascii=False), flush=True)


def qa(a) -> None:
    from src.scripts.data.tp36_final2026 import parse_payoff_json
    from src.scraper.payoff_repair import load_fix

    res: dict = {}
    exclude: dict = {}
    out = Path(a.dir).resolve() if a.dir else OUT
    for year in a.years:
        f = out / f"trifecta_{year}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        if "phase" in d.columns:      # collect-snap の出力は確定だけで調べる
            d = d[d["phase"].astype(str) == "final"]
        g = d.groupby("race_id")
        stat = pd.DataFrame({"n_rows": g.size(), "inv_sum": g["odds"].apply(lambda s: float((1.0 / s.clip(lower=1.0)).sum())),
                             "n_head": g["n_head"].first(), "expected": g["expected"].first()})
        stat["combo_ok"] = stat["n_rows"] >= 0.97 * stat["expected"]        # 応答に一部の組合せ（ごく人気薄）が欠ける日がある。97% 未満は除外
        stat["miss_rate"] = 1.0 - stat["n_rows"] / stat["expected"].clip(lower=1)
        stat["inv_ok"] = stat["inv_sum"].between(1.37, 1.39)
        # 払戻との一致: 1 着-2 着-3 着の的中組合せのオッズ×100 と払戻
        if year >= 2026 or year < 2020:      # 整形済みの payoff.parquet は 2020〜2025 年だけ
            pj = pd.read_parquet(TABLES / str(year) / "race_result_flat.parquet", columns=["race_id", "payoff"]).drop_duplicates("race_id")
            fix = load_fix(year)
            hit = {}
            for r in pj.itertuples(index=False):
                h = parse_payoff_json(r.payoff, fix.get(str(r.race_id)))
                hit[str(r.race_id)] = h.get("sanrentan", {})
        else:
            pay = pd.read_parquet(ROOT / "data/local/modeling/betting/payoff.parquet")
            pay = pay[(pay["bet_type"] == "sanrentan") & (pay["race_id"].str.startswith(str(year)[:4]) | True)]
            hit = {}
            for r in pay.itertuples(index=False):
                try:
                    c = tuple(int(x) for x in str(r.combo).replace(">", "-").replace("→", "-").replace(" ", "").split("-"))
                except ValueError:
                    continue
                hit.setdefault(str(r.race_id), {})[c] = float(r.payout)
        odds_map = {rid: dict(zip(zip(x["a"], x["b"], x["c"]), x["odds"])) for rid, x in d.groupby("race_id")}
        match, total, rel = 0, 0, []
        bad_pay = []
        for rid, h in hit.items():
            om = odds_map.get(rid)
            if not om or not h:
                continue
            for c, pay_ in h.items():
                o = om.get(tuple(int(v) for v in c))
                if o is None:
                    bad_pay.append(rid)
                    continue
                total += 1
                r = pay_ / (o * 100.0)
                rel.append(r)
                if abs(r - 1.0) < 0.002:
                    match += 1
                else:
                    bad_pay.append(rid)
        rel = np.array(rel) if rel else np.array([np.nan])
        res[str(year)] = {
            "races": int(len(stat)), "rows": int(len(d)), "combo_ok_rate": float(stat["combo_ok"].mean()), "any_missing_rate": float((stat["n_rows"] < stat["expected"]).mean()), "miss_rate_mean": float(stat["miss_rate"].mean()), "inv_sum_median": float(stat["inv_sum"].median()),
            "inv_sum_p5_p95": [float(stat["inv_sum"].quantile(0.05)), float(stat["inv_sum"].quantile(0.95))], "inv_ok_rate": float(stat["inv_ok"].mean()),
            "hit_checked": int(total), "hit_exact_rate": float(match / max(total, 1)), "hit_ratio_median": float(np.nanmedian(rel)),
            "hit_ratio_p1_p99": [float(np.nanquantile(rel, 0.01)), float(np.nanquantile(rel, 0.99))]}
        ex = set(stat.index[~stat["combo_ok"] | ~stat["inv_ok"]]) | set(bad_pay)
        missing = json.loads((out / f"missing_{year}.json").read_text(encoding="utf-8")) if (out / f"missing_{year}.json").exists() else []
        nf = out / f"no_final_{year}.json"
        exclude[str(year)] = {"missing": missing, "inconsistent": sorted(ex), "no_final": json.loads(nf.read_text(encoding="utf-8")) if nf.exists() else []}
    (out / "qa.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "exclude.json").write_text(json.dumps(exclude, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "qa", "collect-snap"])
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--dir", default="", help="qa: 調べる parquet の置き場（既定は tp48）")
    ap.add_argument("--base", default=str(ROOT), help="collect-snap: データの置き場のルート（data/local・data/page_reference）")
    ap.add_argument("--out", default="", help=f"collect-snap: 出力先（既定は <base>/{SNAP_OUT}）")
    ap.add_argument("--force", action="store_true", help="collect-snap: 既にある年も作り直す")
    ap.add_argument("--no-gcs", action="store_true", help="collect-snap: GCS を読まない（写しと既存の parquet だけ）")
    ap.add_argument("--max-gcs-reads", type=int, default=16000, help="collect-snap: GCS の文書を読む件数の上限（全年の合計）")
    ap.add_argument("--max-gcs-errors", type=int, default=20, help="collect-snap: GCS の読み出しの失敗がこの件数になったら止める")
    ap.add_argument("--stop-at", default="", help="collect-snap: この時刻（例 2026-11-28T06:30+09:00）を過ぎたら止める")
    a = ap.parse_args()
    {"collect": collect, "qa": qa, "collect-snap": collect_snap}[a.cmd](a)


if __name__ == "__main__":
    main()
