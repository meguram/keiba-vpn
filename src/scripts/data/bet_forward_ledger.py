"""本番の試行（11/21〜）の前向きの記録の集計（段階 4）。第 2 回の凍結（frozen_20261117_r2）の判定を、実際の T-60 で決めたまま、確定の払戻で精算する。

python -m src.scripts.data.bet_forward_ledger [--start 2026-11-21] [--end YYYY-MM-DD] [--no-settle]

- 元（どれも手元。GCS は読まない）:
  - 判定: data/local/inference/bets_stage2/<日>/<race_id>.json（candidate_runner が 1 レース 1 回だけ書く。status=buy|skip、bets、honmei、odds_phase）
  - 見張りの記録: data/local/inference/launch_watch/<日>.jsonl の event=system_skip（判定ファイルを書かない）・late_no_write（発走後で書かなかった）・
    scope（その日の対象のレース数。分母）。どれも race_id で重複を除き、判定ファイルのあるレースは除く
  - 精算: 確定の単勝の払戻（bet_cycles.load_payoffs と同じ表。単勝が無ければ race_result_flat から補う）。◎ が取消・除外なら返還（購入 0）
- 判定は書き直さない（ファイルの中身をそのまま使う）。買い方・閾値は凍結のまま。
- 既定の期間は 11/21 から（11/14・15 はリハーサルの日。事前登録の 2026 年の窓でもないが、集計には入れない）。
- 出力: docs/operations/retrain_trials/bet_forward_r2.json（日ごと・月ごと・通算。必須の指標と区間）。--no-settle は払戻を読まずに件数だけ数える（形の確かめ用）。
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
INF = Path("/home/jovyan/work/keiba-vpn/data/local/inference")
OUT = ROOT / "docs/operations/retrain_trials/bet_forward_r2.json"
FROZEN = ROOT / "config/bet_cycles/frozen_20261117_r2.json"


def day_dirs(start: str, end: str | None) -> list[str]:
    s, e = start.replace("-", ""), (end or "99991231").replace("-", "")
    return sorted(p.name for p in (INF / "bets_stage2").iterdir() if p.is_dir() and p.name.isdigit() and s <= p.name <= e)


def load_day(day: str, h: str) -> tuple[list[dict], dict]:
    """判定ファイルと、見張りの記録（system_skip・late_no_write・scope）。system_skip は race_id で重複を除き、判定ファイルのあるレースは除く
    （再起動で同じレースの system_skip が 2 回書かれる・system_skip の後に判定が書かれることがある）。"""
    recs = []
    for f in sorted((INF / "bets_stage2" / day).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("frozen_hash") != h:
            raise SystemExit(f"凍結の hash が違う判定ファイル: {f}（{d.get('frozen_hash')}）")
        recs.append(d)
    decided = {r["race_id"] for r in recs}
    sys_ids, late_ids, scope = set(), set(), None
    w = INF / "launch_watch" / f"{day}.jsonl"
    if w.exists():
        for line in w.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "system_skip" and e.get("race_id"):
                sys_ids.add(e["race_id"])
            elif e.get("event") == "late_no_write" and e.get("race_id"):
                late_ids.add(e["race_id"])
            elif e.get("event") == "scope" and e.get("races") is not None:
                scope = max(scope or 0, int(e["races"]))
    return recs, {"system_skip": len(sys_ids - decided), "late_no_write": len(late_ids - decided), "scope_runner": scope}


def settle(recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """単勝の払戻で精算する（買い方は単勝だけ）。◎ が取消・除外（確定オッズも着順も無い）なら返還（購入 0・払戻 0）。
    払戻の表に単勝が無いレースは、出走表から補う。それでも無ければ精算待ち（nan）。返り値: 購入・払戻。"""
    import pandas as pd
    from src.config.data_paths import TABLES_DIR
    from src.scripts.data import bet_cycles as bc

    buys = [r for r in recs if r.get("status") == "buy"]
    if any(b["bet_type"] != "tansho" for r in buys for b in r.get("bets") or []):
        raise SystemExit("単勝以外の買い目があります（この集計は単勝だけ）")
    ids = {r["race_id"] for r in buys}
    pay = bc.load_payoffs(ids)
    need = {i for i in ids if not (pay.get(i) or {}).get("tansho")}
    flat = None
    if ids:
        yrs = sorted({int(x[:4]) for x in ids})
        flat = pd.concat([pd.read_parquet(Path(TABLES_DIR) / str(y) / "race_result_flat.parquet",
                                          columns=["race_id", "horse_number", "odds", "finish_position", "payoff"]) for y in yrs], ignore_index=True)
        flat["race_id"] = flat["race_id"].astype(str)
        if need:
            bc.fill_payoffs_from_flat(pay, yrs)
    cost, ret = [], []
    for r in recs:
        c = float(sum(int(b["yen"]) for b in (r.get("bets") or []))) if r.get("status") == "buy" else 0.0
        if c == 0:
            cost.append(0.0)
            ret.append(np.nan)
            continue
        g = flat[flat["race_id"] == r["race_id"]] if flat is not None else None
        hn = int((r.get("honmei") or {}).get("horse_number") or r["bets"][0]["combo"][0])
        if g is not None and len(g):
            h = g[pd.to_numeric(g["horse_number"], errors="coerce") == hn]
            if len(h) and pd.isna(pd.to_numeric(h["odds"], errors="coerce")).all() and pd.isna(pd.to_numeric(h["finish_position"], errors="coerce")).all():
                cost.append(0.0)                      # 取消・除外 → 返還
                ret.append(0.0)
                continue
        tp = (pay.get(r["race_id"]) or {}).get("tansho") or {}
        if not tp:
            cost.append(c)
            ret.append(np.nan)                        # 精算待ち（まだ確定していない・払戻が読めない）
            continue
        cost.append(c)
        ret.append(float(sum(tp.get(bc.combo_key(b["bet_type"], b["combo"]), 0) * (int(b["yen"]) // bc.UNIT) for b in r["bets"])))
    return np.array(cost, dtype=float), np.array(ret, dtype=float)


def summary(cost: np.ndarray, ret: np.ndarray, n_all: int, n_sys: int, n_boot: int = 1000, refunded: int = 0) -> dict:
    """n_all: 判定ファイルの数＋システムの見送り＋発走後で書かなかった数。refunded: ◎ の取消で返還になった参戦。"""
    bet = cost > 0
    done = bet & np.isfinite(ret)
    c, r = cost[done], ret[done]
    out = {"races_decided_or_skipped": n_all, "system_skip": n_sys, "bet_races": int(bet.sum()), "refunded": refunded,
           "settled": int(done.sum()), "pending": int((bet & ~np.isfinite(ret)).sum())}
    if done.sum() == 0:
        return out
    W = np.random.default_rng(0).poisson(1.0, size=(n_boot, len(c))).astype(np.float32)
    bs = (W @ r) / np.maximum(W @ c, 1e-12)
    top = np.sort(r)[::-1]
    out.update({"hit_rate": float((r > 0).mean()), "roi": float(r.sum() / c.sum()), "roi_lo": float(np.quantile(bs, 0.025)), "roi_hi": float(np.quantile(bs, 0.975)),
                "roi_ex_top5": float((r.sum() - top[:5].sum()) / c.sum()), "avg_return": float(r.mean()), "median_return": float(np.median(r)),
                "cost": float(c.sum()), "return": float(r.sum())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-11-21")
    ap.add_argument("--end", default=None)
    ap.add_argument("--no-settle", action="store_true")
    a = ap.parse_args()
    h = json.loads(FROZEN.read_text(encoding="utf-8"))["hash"]
    days = day_dirs(a.start, a.end)
    per_day, allc, allr, alln, alls, allf = {}, [], [], 0, 0, 0
    by_month: dict[str, list] = {}
    for d in days:
        recs, w = load_day(d, h)
        if a.no_settle:
            cost = np.array([sum(int(b["yen"]) for b in (r.get("bets") or [])) if r.get("status") == "buy" else 0 for r in recs], dtype=float)
            ret = np.full(len(recs), np.nan)
            nref = 0
        else:
            cost, ret = settle(recs)
            nref = sum(1 for r, c in zip(recs, cost) if r.get("status") == "buy" and c == 0)
        sk = w["system_skip"] + w["late_no_write"]
        n = len(recs) + sk
        per_day[d] = summary(cost, ret, n, w["system_skip"], refunded=nref)
        per_day[d]["late_no_write"] = w["late_no_write"]
        per_day[d]["scope_runner"] = w["scope_runner"]
        per_day[d]["unrecorded"] = (w["scope_runner"] - n) if w["scope_runner"] is not None else None   # 判定も見送りの記録も無いレース（ランナーの停止など）
        per_day[d]["skip_reasons"] = {k: sum(1 for r in recs if r.get("skip_reason") == k) for k in {r.get("skip_reason") for r in recs if r.get("status") == "skip"}}
        m = f"{d[:4]}-{d[4:6]}"
        by_month.setdefault(m, []).append((cost, ret, n, w["system_skip"], nref))
        allc.append(cost)
        allr.append(ret)
        alln += n
        alls += w["system_skip"]
        allf += nref
    months = {m: summary(np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v]), sum(x[2] for x in v), sum(x[3] for x in v),
                         refunded=sum(x[4] for x in v)) for m, v in by_month.items()}
    total = summary(np.concatenate(allc) if allc else np.zeros(0), np.concatenate(allr) if allr else np.zeros(0), alln, alls, refunded=allf)
    res = {"generated_at": date.today().isoformat(), "frozen_hash": h, "start": a.start, "end": a.end, "settled": not a.no_settle,
           "note": "本番の試行の前向きの記録。判定は実際の T-60 のオッズ（candidate_runner の判定ファイルのまま）、精算は確定の払戻。"
                   "システムの原因の見送り（判定ファイルなし）は件数だけ数え、参戦にも見送りにも入れない。",
           "days": per_day, "months": months, "total": total}
    if a.no_settle:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("書いた", OUT.relative_to(ROOT), json.dumps(total, ensure_ascii=False))


if __name__ == "__main__":
    main()
