"""無料ユーザーに出す印（public_mark_type）を、発走 50 分前（T-50）の 1 回で決める（事前登録 config/public_marks.json）。

python -m src.research.public_marks.marks shadow --day 20261129     # 開催日の 17:00 より後に、手元の T-60 と予測から影の記録を書く
python -m src.research.public_marks.marks simulate                  # 止める規則（config/public_marks_stop.json）の模擬（着順は読まない）

- 入力は、予測の行の p_win・p_top3（本番の推論の経路 predictions_v3。T-50 までに最後に書かれたもの）と、
  T-60 の単勝オッズの手元の写し（odds_t60。取得と書き込みが T-50 以前のもの）だけ。確定オッズ・T-50 より後の値・着順・払戻は読まない。
- オッズの取得・書き込み、予測の最後の書き込みが T-50 より後なら、時刻の都合の印なし（wait_odds・wait_pred）にし、ほかの理由と分けて数える。
- 予測の最後の書き込み時刻はファイルの更新時刻で見る（書き直しの履歴は残らない）。T-50 にはあったが後で書き直されたレースも wait_pred になる（影の記録の偏り。門 A で件数を出す）。
- 取消（T-60 の単勝 999.9 以上）の馬は印を付けず「取消」と記録し、ほかの馬の印は付け直さない。
- 影の記録は、設定を最初に commit した日（固めた日）より後の開催日だけ書く。設定は固めた後に変えない（変わっていたら止める）。GCS は読まない（費用 0 円）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "config/public_marks.json"
STOP = ROOT / "config/public_marks_stop.json"
INF = ROOT / "data/local/inference"
OUT = ROOT / "docs/operations/retrain_trials/public_marks_shadow"
JST = timezone(timedelta(hours=9))
_P = json.loads(CFG.read_text(encoding="utf-8"))["params"]       # 区切りはすべて固めた設定から読む
MIN_FIELD, SPLIT_PMAX, STAR_ODDS = int(_P["min_field"]), float(_P["split_pmax"]), float(_P["star_odds"])
HONMEI_ALT_PMIN, SHOW_ALT_P3MIN, STAR_ALT_R = float(_P["honmei_alt_pmin"]), float(_P["show_alt_p3min"]), math.log(float(_P["star_alt_ratio"]))
ANCHOR_PRANK, ANCHOR_POP = int(_P["anchor_prank"]), int(_P["anchor_pop"])
SCRATCH_ODDS = float(_P["scratch_odds"])            # 取消・除外の馬は odds_t60 に 999.9 で残る
DECIDE_BEFORE_MIN = int(_P["decide_before_min"])    # 印は発走 50 分前（T-50）の 1 回で決める（決定 3 の改め、577bf7c7）
ODDS_OLDEST_MIN = int(_P["odds_oldest_min"])        # これより前に取ったオッズは T-60 とみなさない
DAY_DONE_HOUR = int(_P["day_done_hour"])            # 影の記録は開催日の 17:00（JST）より後に流す
KESI_P = _P["kesi"]                                 # 消の線（固めた日に config/kesi_mark.json から写した値）
TIMING = ("wait_odds", "wait_pred")                 # 時刻の都合の印なし（ほかの理由と分けて数える）
P_FIT_END = 2025                                    # p のモデル（bundle_v6_2025）の学習の最後の年


def config_hash(path: Path = CFG) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def harville_top3(pw: np.ndarray) -> np.ndarray:
    """単勝の確率から Harville で 3 着内の確率（人気馬で過大に出る偏りがある）。"""
    pw = np.asarray(pw, float)
    pw = pw / pw.sum()
    n = len(pw)
    out = pw.copy()
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            out[i] += pw[j] * pw[i] / (1 - pw[j])                       # j が 1 着・i が 2 着
            for k in range(n):
                if k in (i, j):
                    continue
                out[i] += pw[k] * pw[j] / (1 - pw[k]) * pw[i] / (1 - pw[k] - pw[j])   # k・j の後に i が 3 着
    return np.minimum(out, 1.0)


def _kesi_cfg() -> dict:
    return {"min_field": int(KESI_P["min_field"]), "min_pop": int(KESI_P["min_pop"]), "p3_max": float(KESI_P["p3_max"])}


def decide(hn, p_win, p_top3, odds, pop) -> dict:
    """1 レースの印。入力は同じ並びの配列（馬番・p_win・p_top3・T-60 の単勝オッズ・T-60 の人気）。取消の馬は渡さない。
    戻り値: marks {馬番: 印}・alt（候補の印）・no_mark（印なしの理由の id）。"""
    hn = [int(x) for x in hn]
    n = len(hn)
    if n < MIN_FIELD:
        return {"marks": {}, "alt": {}, "no_mark": "field"}
    odds = np.asarray(odds, float)
    if len(odds) != n or not np.all(np.isfinite(odds)) or np.any(odds <= 0):
        return {"marks": {}, "alt": {}, "no_mark": "odds"}
    pw = np.asarray(p_win, float)
    if len(pw) != n or not np.all(np.isfinite(pw)) or pw.sum() <= 0:
        return {"marks": {}, "alt": {}, "no_mark": "data"}
    pw = pw / pw.sum()
    if pw.max() < SPLIT_PMAX:
        return {"marks": {}, "alt": {}, "no_mark": "split"}
    q = (1 / odds) / (1 / odds).sum()
    pop = np.asarray([np.nan if x is None else float(x) for x in pop], float)
    order = list(np.argsort(-pw, kind="stable"))
    main2 = set(order[:2])
    marks: dict[int, str] = {hn[order[0]]: "honmei", hn[order[1]]: "pair"}
    used = set(main2)
    p3 = harville_top3(pw)
    sv = max((i for i in range(n) if i not in used), key=lambda i: (p3[i], -i))
    marks[hn[sv]] = "show_val"
    used.add(sv)
    prank = {i: r for r, i in enumerate(order)}
    anc = [i for i in range(n) if i not in used and prank[i] < ANCHOR_PRANK and np.isfinite(pop[i]) and pop[i] <= ANCHOR_POP]
    if anc:
        a = max(anc, key=lambda i: pw[i])
        marks[hn[a]] = "anchor"
        used.add(a)
    r = np.log(pw / q)
    st = [i for i in range(n) if i not in used and odds[i] >= STAR_ODDS and r[i] > 0]
    star_i = max(st, key=lambda i: r[i]) if st else None
    if star_i is not None:
        marks[hn[star_i]] = "star"
        used.add(star_i)
    kc = _kesi_cfg()
    p3m = np.asarray(p_top3, float)
    if n >= kc["min_field"] and np.all(np.isfinite(pop)):          # 人気がそろわないときは消を付けない（今の assign_kesi_mark と同じ）
        for i in range(n):
            if hn[i] not in marks and pop[i] >= kc["min_pop"] and np.isfinite(p3m[i]) and p3m[i] <= kc["p3_max"]:
                marks[hn[i]] = "kesi"
    # 候補（影の記録だけ。表示には使わない）
    ev = pw * odds
    ev_order = sorted((i for i in range(n) if pw[i] >= HONMEI_ALT_PMIN), key=lambda i: (-ev[i], -pw[i]))
    q3 = harville_top3(q)
    alt = {"honmei": hn[ev_order[0]] if ev_order else None, "pair": hn[ev_order[1]] if len(ev_order) > 1 else None}
    sv_alt = [i for i in range(n) if i not in main2 and p3[i] >= SHOW_ALT_P3MIN]        # 主の ◎○ を除く（config の文言どおり）
    alt["show_val"] = hn[max(sv_alt, key=lambda i: p3[i] / q3[i])] if sv_alt else None
    st_alt = [i for i in range(n) if (i not in used or i == star_i) and odds[i] >= STAR_ODDS and r[i] > STAR_ALT_R]
    alt["star"] = hn[max(st_alt, key=lambda i: r[i])] if st_alt else None
    return {"marks": marks, "alt": alt, "no_mark": None}


def _ts(s: str | None) -> datetime | None:
    """ISO の時刻。タイムゾーンの無い文字列は JST とみなす。"""
    if not s:
        return None
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=JST)


def _frozen_date() -> str:
    """設定を固めた日（設定を最初に commit した日、YYYYMMDD）。固めた後に設定が変わっていたら止める。"""
    out = []
    for f in (CFG, STOP):
        rel = str(f.relative_to(ROOT))
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode != 0:
            raise SystemExit(f"{rel} に commit していない変更があります（固めてから流す）")
        log = subprocess.run(["git", "log", "--format=%cs", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.split()
        if not log:
            raise SystemExit(f"{rel} が commit されていません")
        if len(log) > 1:
            raise SystemExit(f"{rel} は固めた後に変わっています（事前登録を変えない。変えるなら会議で新しい登録を立てる）")
        out.append(log[-1].replace("-", ""))
    return max(out)


def race_card(day: str) -> dict[str, dict]:
    """出馬表（race_shutuba の date・start_time）から、その日のレースと発走時刻（JST）。手元の表だけを読む。"""
    import pandas as pd

    from src.config.data_paths import PAGE_REF

    f = Path(PAGE_REF) / "tables" / day[:4] / "race_shutuba_race.parquet"
    if not f.exists():
        return {}
    t = pd.read_parquet(f, columns=["race_id", "date", "start_time"])
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    t = t[t["date"].astype(str).str.replace("/", "-").str[:10].isin([iso, day])]
    out = {}
    for rid, st in zip(t["race_id"].astype(str), t["start_time"]):
        s = str(st or "")
        if len(s) >= 5 and s[2] == ":":
            out[rid] = {"post": datetime(int(day[:4]), int(day[4:6]), int(day[6:]), int(s[:2]), int(s[3:5]), tzinfo=JST)}
    return out


def decide_race(o: dict | None, p, post: datetime | None, pred_mtime: datetime | None) -> dict:
    """1 レースを T-50 で決める。o は odds_t60 の文書、p は馬番つきの予測の表。"""
    if post is None:
        return {"marks": {}, "alt": {}, "no_mark": "data", "decide_at": None, "scratched": []}
    cut = post - timedelta(minutes=DECIDE_BEFORE_MIN)
    base = {"decide_at": cut.isoformat(timespec="seconds"), "scratched": []}
    if o is None or o.get("phase") != "t60" or not o.get("snapshot_at"):
        return {"marks": {}, "alt": {}, "no_mark": "odds", **base}
    snap, wrote = _ts(o.get("snapshot_at")), _ts(o.get("written_at"))
    if wrote is None or snap < post - timedelta(minutes=ODDS_OLDEST_MIN):       # 書き込み時刻が無いと、T-50 に手元にあったか確かめられない
        return {"marks": {}, "alt": {}, "no_mark": "odds", **base}
    avail = max(snap, wrote)                                                     # 手元に書かれた時刻までオッズは使えない
    if avail > cut:
        return {"marks": {}, "alt": {}, "no_mark": "wait_odds", **base}
    if p is None:
        return {"marks": {}, "alt": {}, "no_mark": "data", **base}
    if pred_mtime is None or pred_mtime > cut:
        return {"marks": {}, "alt": {}, "no_mark": "wait_pred", **base}
    ent = {int(e["horse_number"]): e for e in o.get("entries") or []}
    if "horse_number" not in p.columns:
        return {"marks": {}, "alt": {}, "no_mark": "data", **base}
    hall = p["horse_number"].astype(int).tolist()
    scr = sorted(x for x, e in ent.items() if float(e.get("win_odds") or 0) >= SCRATCH_ODDS)    # 取消は T-60 の 999.9 から取る（予測からは抜けていることが多い）
    runners = set(ent) - set(scr)
    if any(x not in ent for x in hall) or not runners <= set(hall):     # 予測にしかいない馬・T-60 にいる取消でない馬が予測にいない → 不整合
        return {"marks": {}, "alt": {}, "no_mark": "data", **base, "scratched": scr}
    keep = [i for i, x in enumerate(hall) if x in runners]
    pk = p.iloc[keep]
    hn = [hall[i] for i in keep]
    res = decide(hn, pk["p_win"].to_numpy(float), pk["p_top3"].to_numpy(float),
                 [ent[x].get("win_odds") for x in hn], [ent[x].get("popularity") for x in hn])
    return {**res, **base, "scratched": scr}


def load_pred(pf: Path, ff: Path):
    """予測の表（馬番つき）と最後の書き込み時刻（JST）。予測の行は horse_id だけなので、同じ日の特徴量で馬番に対応づける。"""
    import pandas as pd

    if not pf.exists():
        return None, None
    pm = datetime.fromtimestamp(pf.stat().st_mtime, JST)                       # 記録の時刻はすべて JST
    p = pd.read_parquet(pf)
    if "horse_number" not in p.columns and ff.exists() and "horse_id" in p.columns:
        fx = pd.read_parquet(ff, columns=["horse_id", "horse_number"]).astype({"horse_id": str})
        p = p.astype({"horse_id": str}).merge(fx, on="horse_id", how="left")
        if p["horse_number"].isna().any():
            p = p.drop(columns=["horse_number"])
    return p, pm


def shadow_day(day: str, inf: Path = INF, out: Path = OUT, check_frozen: bool = True, card: dict | None = None,
               now: datetime | None = None) -> dict:
    """開催日の影の記録（開催日の 17:00 より後に流す）。固めた日より後の開催日だけ。既にある日は書き直さない。
    レースは出馬表から数える（T-60 のファイルが無いレースも「印なし」の行にする）。"""
    import pandas as pd

    if check_frozen and day <= _frozen_date():
        raise SystemExit(f"{day} は設定を固めた日以前の開催日です（入れない）")
    now = now or datetime.now(JST)
    if now < datetime(int(day[:4]), int(day[4:6]), int(day[6:]), DAY_DONE_HOUR, tzinfo=JST):
        raise SystemExit(f"{day} の 17:00 より前です（開催日の後に流す）")
    dst = out / f"{day}.jsonl"
    if dst.exists():
        raise SystemExit(f"{dst} は既にあります（固めた印は書き直さない）")
    card = race_card(day) if card is None else card
    if not card:
        raise SystemExit(f"{day} の出馬表（発走時刻）が手元にありません（空の記録は書かない）")
    if not (inf / "odds_t60" / day).is_dir():
        raise SystemExit(f"{inf / 'odds_t60' / day} がありません（空の記録は書かない）")
    h, hs = config_hash(), config_hash(STOP)
    rows, n_no = [], {}
    for rid in sorted(card):
        post = card[rid]["post"]
        of = inf / "odds_t60" / day / f"{rid}.json"
        pf = inf / "predictions_v3" / day / f"{rid}.parquet"
        ff = inf / "features" / day / f"{rid}.parquet"
        o = json.loads(of.read_text(encoding="utf-8")) if of.exists() else None
        rec = {"race_id": rid, "day": day, "config_hash": h, "stop_hash": hs, "post_time": post.isoformat(timespec="seconds"),
               "t60_at": (o or {}).get("snapshot_at"), "t60_written_at": (o or {}).get("written_at"), "phase": (o or {}).get("phase"),
               "in_training": int(day[:4]) <= P_FIT_END, "tool_sha": _sha(Path(__file__)),
               "inputs_sha": {"odds": _sha(of), "pred": _sha(pf), "features": _sha(ff)}}
        p, pm = load_pred(pf, ff)
        if pm is not None:
            rec["pred_mtime"] = pm.isoformat(timespec="seconds")
        res = decide_race(o, p, post, pm)
        rec.update({"decide_at": res.get("decide_at"), "marks": {str(k): v for k, v in res["marks"].items()}, "alt": res["alt"],
                    "scratched": res.get("scratched") or [], "no_mark": res["no_mark"], "no_mark_timing": res["no_mark"] in TIMING})
        k = res["no_mark"] or "marked"
        n_no[k] = n_no.get(k, 0) + 1
        rows.append(rec)
    out.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {"day": day, "races": len(rows), "by_reason": n_no, "config_hash": h}


def simulate(q_pool: np.ndarray, ratio: float, at: list[int], thr: float = 0.90, n: int = 20000, seed: int = 20261125) -> float:
    """真の比 ratio（◎ の勝率 ÷ 同じ帯の ◎ 以外の馬で測った市場の比）のとき、見る時点 at（◎ のあるレースの数）のどこかで
    excess_mr の点推定 < thr になる（止める）確率。帯の基準は ◎ 以外の多くの馬で測るので、振れを無視して 1 と置く（楽観に寄る前提）。"""
    rng = np.random.default_rng(seed)
    m = max(at)
    qq = rng.choice(q_pool, size=(n, m))
    win = rng.random((n, m)) < np.minimum(qq * ratio, 1.0)
    cw, cq = np.cumsum(win, 1), np.cumsum(qq, 1)
    idx = np.array(at) - 1
    return float(((cw[:, idx] / cq[:, idx]) < thr).any(1).mean())


def honmei_q_pool(days=("20261121", "20261122", "20261123"), inf: Path = INF) -> np.ndarray:
    """模擬に使う ◎ の T-60 の市場の勝率。止める規則の数える単位（◎ のあるレース）に合わせ、decide_race で印が付いたレースの ◎ だけ。着順は読まない。"""
    qs = []
    for d in days:
        card = race_card(d)
        for rid in sorted(card):
            of = inf / "odds_t60" / d / f"{rid}.json"
            if not of.exists():
                continue
            o = json.loads(of.read_text(encoding="utf-8"))
            p, pm = load_pred(inf / "predictions_v3" / d / f"{rid}.parquet", inf / "features" / d / f"{rid}.parquet")
            res = decide_race(o, p, card[rid]["post"], pm)
            hm = [int(k) for k, v in res["marks"].items() if v == "honmei"]
            if not hm:
                continue
            ent = {int(e["horse_number"]): float(e["win_odds"]) for e in o["entries"] if int(e["horse_number"]) not in res["scratched"]}
            inv = {k: 1 / v for k, v in ent.items()}
            qs.append(inv[hm[0]] / sum(inv.values()))
    return np.array(qs)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shadow")
    s.add_argument("--day", required=True)
    sub.add_parser("simulate")
    a = ap.parse_args()
    if a.cmd == "shadow":
        print(json.dumps(shadow_day(a.day), ensure_ascii=False))
        return
    stop = json.loads(STOP.read_text(encoding="utf-8"))
    at = [c["at_races"] for c in stop["checks"]]
    q = honmei_q_pool()
    print(json.dumps({"races": len(q), **{str(r): simulate(q, r, at) for r in (1.1, 1.0, 0.8)}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
