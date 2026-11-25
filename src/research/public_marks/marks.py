"""無料ユーザーに出す印（public_mark_type）を、発走 50 分前（T-50）の 1 回で決める（事前登録 config/public_marks_v2.json）。

python -m src.research.public_marks.marks shadow --day 20261129     # 開催日の 17:00 より後に、手元の T-60 と予測から影の記録を書く
python -m src.research.public_marks.marks simulate                  # 止める規則（config/public_marks_stop_v2.json）の模擬（着順は読まない）

- v2: 印は過不足なく全レースに付ける。予測があり予測待ちでないレースには、必ず ◎○▲★ を頭数まで（重複なし）、△・消は 0 頭以上。
  付ける順は ◎ → ○ → ▲ → ★ → △ → 消。★ の mode は ana・fallback_r・no_odds_p（T-50 までにオッズが無いときは市場と比べず、△・消は付けない）。
- 入力は、予測の行の p_win・p_top3（本番の推論の経路 predictions_v3。T-50 までに最後に書かれたもの）と、
  T-60 の単勝オッズの手元の写し（odds_t60。取得と書き込みが T-50 以前のもの）だけ。確定オッズ・T-50 より後の値・着順・払戻は読まない。
- 予測の最後の書き込みが T-50 より後のレースは「予測待ち」（wait_pred）として記録だけに残す。画面の印なしではなく、測る側の穴
  （本番では T-50 の時点の予測で印が付くが、書き直しの履歴は残らない）。
- 取消（T-60 の単勝 999.9 以上）の馬は印を付けず「取消」と記録し、ほかの馬の印は付け直さない。
- 影の記録は、v2 の設定を最初に commit した日（固めた日）より後の開催日だけ書く。設定は固めた後に変えない（変わっていたら止める）。GCS は読まない（費用 0 円）。
- v1（config/public_marks.json）は使わずに置き換えた（影の記録 0 行・70% の側は 0 回）。
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
CFG = ROOT / "config/public_marks_v2.json"
STOP = ROOT / "config/public_marks_stop_v2.json"
INF = ROOT / "data/local/inference"
OUT = ROOT / "docs/operations/retrain_trials/public_marks_shadow"
JST = timezone(timedelta(hours=9))
_P = json.loads(CFG.read_text(encoding="utf-8"))["params"]       # 区切りはすべて固めた設定から読む
SPLIT_PMAX, STAR_ODDS = float(_P["split_pmax"]), float(_P["star_odds"])
HONMEI_ALT_PMIN, SHOW_ALT_P3MIN, STAR_ALT_R = float(_P["honmei_alt_pmin"]), float(_P["show_alt_p3min"]), math.log(float(_P["star_alt_ratio"]))
ANCHOR_PRANK, ANCHOR_POP = int(_P["anchor_prank"]), int(_P["anchor_pop"])
SCRATCH_ODDS = float(_P["scratch_odds"])            # 取消・除外の馬は odds_t60 に 999.9 で残る
DECIDE_BEFORE_MIN = int(_P["decide_before_min"])    # 印は発走 50 分前（T-50）の 1 回で決める（決定 3 の改め、577bf7c7）
ODDS_OLDEST_MIN = int(_P["odds_oldest_min"])        # これより前に取ったオッズは T-60 とみなさない
DAY_DONE_HOUR = int(_P["day_done_hour"])            # 影の記録は開催日の 17:00（JST）より後に流す
KESI_P = _P["kesi"]                                 # 消の線（固めた日に config/kesi_mark.json から写した値）
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


def decide(hn, p_win, p_top3, odds=None, pop=None) -> dict:
    """1 レースの印（v2）。入力は同じ並びの配列（馬番・p_win・p_top3・T-60 の単勝オッズ・T-60 の人気）。取消の馬は渡さない。
    odds が None（T-50 までにオッズが無い）なら no_odds_p で付ける（市場と比べず、△・消は付けない）。
    戻り値: marks {馬番: 印}・alt（候補）・mode（★ の決め方）・split（割れているか）・status（marked・data）。"""
    if odds is not None and pop is None:
        raise ValueError("odds を渡すときは pop（T-60 の人気）も渡す")
    hn = [int(x) for x in hn]
    n = len(hn)
    pw = np.asarray(p_win, float)
    if n == 0 or len(pw) != n or not np.all(np.isfinite(pw)) or pw.sum() <= 0:
        return {"marks": {}, "alt": {}, "mode": None, "split": None, "status": "data"}
    pw = pw / pw.sum()
    has_odds = odds is not None
    if has_odds:
        odds = np.asarray(odds, float)
        has_odds = len(odds) == n and bool(np.all(np.isfinite(odds))) and bool(np.all(odds > 0))
    order = list(np.argsort(-pw, kind="stable"))
    marks: dict[int, str] = {}
    used: set[int] = set()

    def put(i: int, m: str) -> None:
        marks[hn[i]] = m
        used.add(i)

    put(order[0], "honmei")
    if n >= 2:
        put(order[1], "pair")
    p3 = harville_top3(pw)
    rest = [i for i in range(n) if i not in used]
    if rest:
        put(max(rest, key=lambda i: (p3[i], -i)), "show_val")
    rest = [i for i in range(n) if i not in used]
    mode, star_i = None, None
    if rest:
        if has_odds:
            q = (1 / odds) / (1 / odds).sum()
            r = np.log(pw / q)
            ana = [i for i in rest if odds[i] >= STAR_ODDS and r[i] > 0]
            if ana:
                star_i, mode = max(ana, key=lambda i: r[i]), "ana"
            else:
                star_i, mode = max(rest, key=lambda i: r[i]), "fallback_r"
        else:
            star_i, mode = max(rest, key=lambda i: pw[i]), "no_odds_p"
        put(star_i, "star")
    if has_odds:
        pp = np.asarray([np.nan if x is None else float(x) for x in pop], float)
        prank = {i: k for k, i in enumerate(order)}
        anc = [i for i in range(n) if i not in used and prank[i] < ANCHOR_PRANK and np.isfinite(pp[i]) and pp[i] <= ANCHOR_POP]
        if anc:
            put(max(anc, key=lambda i: pw[i]), "anchor")
        kc = _kesi_cfg()
        p3m = np.asarray(p_top3, float)
        if n >= kc["min_field"] and np.all(np.isfinite(pp)):          # 人気がそろわないときは消を付けない（今の assign_kesi_mark と同じ）
            for i in range(n):
                if i not in used and pp[i] >= kc["min_pop"] and np.isfinite(p3m[i]) and p3m[i] <= kc["p3_max"]:
                    marks[hn[i]] = "kesi"                               # ★ を含む ◎○▲★△ の馬には付けない
    # 候補（影の記録だけ。表示には使わない）
    alt = {}
    if has_odds:
        ev = pw * odds
        ev_order = sorted((i for i in range(n) if pw[i] >= HONMEI_ALT_PMIN), key=lambda i: (-ev[i], -pw[i]))
        q = (1 / odds) / (1 / odds).sum()
        q3 = harville_top3(q)
        r = np.log(pw / q)
        main2 = set(order[:2])
        alt = {"honmei": hn[ev_order[0]] if ev_order else None, "pair": hn[ev_order[1]] if len(ev_order) > 1 else None}
        sv_alt = [i for i in range(n) if i not in main2 and p3[i] >= SHOW_ALT_P3MIN]
        alt["show_val"] = hn[max(sv_alt, key=lambda i: p3[i] / q3[i])] if sv_alt else None
        before_star = {i for i in range(n) if hn[i] in marks and marks[hn[i]] in ("honmei", "pair", "show_val")}
        st_alt = [i for i in range(n) if i not in before_star and odds[i] >= STAR_ODDS and r[i] > STAR_ALT_R]
        alt["star"] = hn[max(st_alt, key=lambda i: r[i])] if st_alt else None
    return {"marks": marks, "alt": alt, "mode": mode, "split": bool(pw.max() < SPLIT_PMAX), "status": "marked"}


def _ts(s: str | None) -> datetime | None:
    """ISO の時刻。タイムゾーンの無い文字列は JST とみなす。"""
    if not s:
        return None
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=JST)


def _frozen_date() -> str:
    """設定を固めた日（v2 の設定を最初に commit した日、YYYYMMDD）。固めた後に設定が変わっていたら止める。"""
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


def _odds_usable(o: dict | None, post: datetime, cut: datetime) -> bool:
    """T-60 のオッズが T-50 に手元にあったか（取得と書き込みが T-50 以前・T-75 より後・phase が t60）。"""
    if o is None or o.get("phase") != "t60" or not o.get("snapshot_at"):
        return False
    snap, wrote = _ts(o.get("snapshot_at")), _ts(o.get("written_at"))
    if wrote is None or snap < post - timedelta(minutes=ODDS_OLDEST_MIN):
        return False
    return max(snap, wrote) <= cut


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def decide_race(o: dict | None, p, post: datetime | None, pred_mtime: datetime | None) -> dict:
    """1 レースを T-50 で決める（v2）。o は odds_t60 の文書、p は馬番つきの予測の表。
    status: marked（印を付けた）・wait_pred・no_pred・data（記録だけ）。"""
    blank = {"marks": {}, "alt": {}, "mode": None, "split": None, "scratched": [], "odds_used": False}
    if post is None:
        return {**blank, "status": "data", "decide_at": None}
    cut = post - timedelta(minutes=DECIDE_BEFORE_MIN)
    base = {"decide_at": cut.isoformat(timespec="seconds")}
    if p is None:
        return {**blank, **base, "status": "no_pred"}
    if pred_mtime is None or pred_mtime > cut:
        return {**blank, **base, "status": "wait_pred"}
    if "horse_number" not in p.columns:
        return {**blank, **base, "status": "data"}
    hall = p["horse_number"].astype(int).tolist()
    usable = _odds_usable(o, post, cut)
    scr: list[int] = []
    if usable:                                          # T-50 に手元にあったオッズだけを使う（取消の判定も、それより後のオッズでは変えない）
        try:
            ent = {int(e["horse_number"]): e for e in (o or {}).get("entries") or []}
        except (KeyError, TypeError, ValueError):
            return {**blank, **base, "status": "data"}
        scr = sorted(x for x, e in ent.items() if _num(e.get("win_odds")) >= SCRATCH_ODDS)    # 取消は T-60 の 999.9 から取る
        runners = set(ent) - set(scr)
        if any(x not in ent for x in hall) or not runners <= set(hall):     # 予測にしかいない馬・T-60 にいる取消でない馬が予測にいない
            return {**blank, **base, "status": "data", "scratched": scr}
        if any(not (0 < _num(ent[x].get("win_odds")) < SCRATCH_ODDS) for x in hall if x in runners):
            usable = False                                                  # 馬はそろうが一部のオッズが欠ける → 市場と比べずに付ける（T-50 に分かっていた取消は使う）
    keep = [i for i, x in enumerate(hall) if x not in scr]
    pk = p.iloc[keep]
    hn = [hall[i] for i in keep]
    if usable:
        res = decide(hn, pk["p_win"].to_numpy(float), pk["p_top3"].to_numpy(float),
                     [ent[x].get("win_odds") for x in hn], [ent[x].get("popularity") for x in hn])
    else:
        res = decide(hn, pk["p_win"].to_numpy(float), pk["p_top3"].to_numpy(float))
    return {**res, **base, "scratched": scr, "odds_used": usable}


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
    レースは出馬表から数える（予測や T-60 のファイルが無いレースも行にする）。"""
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
    for sub in ("predictions_v3", "odds_t60"):
        if not (inf / sub / day).is_dir():
            raise SystemExit(f"{inf / sub / day} がありません（空・全部オッズなしの記録は書かない）")
    h, hs = config_hash(), config_hash(STOP)
    rows, n_st, n_mode = [], {}, {}
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
        rec.update({"decide_at": res.get("decide_at"), "status": res["status"], "mode": res["mode"], "split": res["split"],
                    "odds_used": res["odds_used"], "marks": {str(k): v for k, v in res["marks"].items()}, "alt": res["alt"],
                    "scratched": res.get("scratched") or []})
        n_st[res["status"]] = n_st.get(res["status"], 0) + 1
        if res["mode"]:
            n_mode[res["mode"]] = n_mode.get(res["mode"], 0) + 1
        rows.append(rec)
    out.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {"day": day, "races": len(rows), "by_status": n_st, "by_mode": n_mode, "config_hash": h}


def simulate(q_pool: np.ndarray, ratio: float, at: list[int], thr: float = 0.90, n: int = 20000, seed: int = 20261125) -> float:
    """真の比 ratio（◎ の勝率 ÷ 同じ帯の ◎ 以外の馬で測った市場の比）のとき、見る時点 at（全レースの数）のどこかで
    excess_mr の点推定 < thr になる（止める）確率。帯の基準は ◎ 以外の多くの馬で測るので、振れを無視して 1 と置く（楽観に寄る前提）。"""
    rng = np.random.default_rng(seed)
    m = max(at)
    qq = rng.choice(q_pool, size=(n, m))
    win = rng.random((n, m)) < np.minimum(qq * ratio, 1.0)
    cw, cq = np.cumsum(win, 1), np.cumsum(qq, 1)
    idx = np.array(at) - 1
    return float(((cw[:, idx] / cq[:, idx]) < thr).any(1).mean())


def honmei_q_pool(days=("20261121", "20261122", "20261123"), inf: Path = INF) -> np.ndarray:
    """模擬に使う ◎ の T-60 の市場の勝率。v2 では全レースに ◎ が付くので、decide_race で印を付けたレースのうち、
    T-50 に手元にあった T-60 のオッズで付けたレース（odds_used）の ◎ の q。オッズなしのレースは入れない。着順は読まない。"""
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
            if not hm or not res["odds_used"]:
                continue
            ent = {int(e["horse_number"]): _num(e.get("win_odds")) for e in o["entries"] if int(e["horse_number"]) not in res["scratched"]}
            if hm[0] not in ent or any(v <= 0 for v in ent.values()):
                continue
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
