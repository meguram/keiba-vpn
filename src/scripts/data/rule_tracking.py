"""買い目の決まり（条件で買う・買わない）の前向き追跡。いろいろな決まりを並べて、確定したレースを台帳に積む（読み取りと data/local への書き込みだけ）。

python -m src.scripts.data.rule_tracking track [--from 20260101] [--to 20261231]   # 確定したレースを判定して台帳に追記（同じキーは書かない＝冪等）
python -m src.scripts.data.rule_tracking report                                   # 台帳から集計（latest.json / latest.md / latest.html）

決まりと閾値: config/rule_tracking.json の frozen（2026-11-14 に事前登録）。根拠は docs/operations/retrain_trials/model_rule_backtest_20261114.md。
- 台帳の 1 行 = モデル × 予測の出どころ × レース。◎○▲（勝率の上位 3 頭）、自信の度合い、頭数、馬場、◎ の人気・オッズ（確定と発走前）、
  4 券種の払戻（モデルの買い目と、同じレースを市場の 1 番人気 → 2・3 番人気で買った場合）、決まりごとの「買う／買わない」を持つ。
- 決まりの中身・閾値はハッシュで検査し、変わっていたら追記を拒否する（基準は git に固定した config/rule_tracking_registration.json）。
  決まりを足すのは since（YYYY-MM-DD、台帳の最後の日より後）を付けた追加だけ（さかのぼらない）。
- 発走前オッズは後から補われることがある（pre_odds_table は 14 日間読み直す）ので、forward のレースで発走前オッズが無い間は、
  14 日までは台帳に書かずに次回に回す。除外日（excluded_dates）と区分（eras）は、report のたびに日付で付け直す。
- 期間: track_from 以降の forward（日々の予測）が「前向き」。それより前と、後から作った予測（backfill）は「固定前の参考」。
- オッズ: final = race_result_flat の確定オッズ・人気。pre = 発走前の最後のスナップショット（data/local/research/persona1/pre_odds.parquet。
  週次の pre_odds_table が作る手元の表で、ここでは GCS を読まない）。人気・オッズに関わる決まり（単勝）は、オッズ時点ごとに判定する。
- 費用: 0 円（GCS・GCP を読まない。週次の frozen_tracking_weekly.sh の最後に足す）。
2026 年以降の追跡データは、学習・決まりの選び直しに使わない。約 1,000 レースたまるまでは途中の値で判断しない。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import frozen_tracking as ft

ROOT = Path(__file__).resolve().parents[3]
CFG = Path(os.environ.get("RT_CFG", str(ROOT / "config/rule_tracking.json")))
REG = Path(os.environ.get("RT_REG", str(ROOT / "config/rule_tracking_registration.json")))
PRE_WAIT_DAYS = 21                  # pre_odds_table は最初に読んだ日（確定後の週次、レース日 +1〜7 日）から 14 日読み直すので、レース日 +21 日まで待つ
RETRY_DAYS = 35                     # 払戻が読めない・予測がそろわないなど一時的かもしれない欠けは、レース日 +35 日までは毎回読み直す
OUT_DIR = Path(os.environ.get("RULE_TRACKING_DIR", str(ROOT / "data/local/meta/rule_tracking")))
PRE_ODDS = Path(os.environ.get("RT_PRE_ODDS", str(ROOT / "data/local/research/persona1/pre_odds.parquet")))
BETS = ("tansho", "fukusho", "umaren", "wide")
COST = {"tansho": 100.0, "fukusho": 100.0, "umaren": 200.0, "wide": 200.0}
BET_LABEL = {"tansho": "単勝", "fukusho": "複勝", "umaren": "馬連流し", "wide": "ワイド流し"}
VARIANTS = ("final", "pre")
PRE_PHASES = ("t02", "t05", "t10", "t15", "t60")           # 発走時刻より前のうち、締切に近い順
ODDS_CONDS = {"fav", "odds_lt", "pop2to5", "pop4to5"}       # オッズ時点で判定が変わる条件


# ───────────────────────── 設定と事前登録の検査 ─────────────────────────
def load_cfg(path: Path = CFG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def frozen_hashes(cfg: dict) -> dict:
    """閾値（rules 以外の frozen）と、決まりごとのハッシュ。sources・eras（出どころ・配備の区分）は増えてよいので対象外。"""
    fr = cfg["frozen"]
    return {"thresholds": _h({k: v for k, v in fr.items() if k != "rules"}), "rules": {r["id"]: _h(r) for r in fr["rules"]}}


def _is_date(x) -> bool:
    x = str(x)
    return len(x) == 10 and x[4] == "-" and x[7] == "-" and (x[:4] + x[5:7] + x[8:]).isdigit()


def check_registration(cfg: dict, out_dir: Path, write: bool = True, reg_path: Path | None = None, today: str = "") -> tuple[bool, list[str]]:
    """事前登録（git に固定した registration の閾値・決まりのハッシュ）と比べる。閾値・登録済みの決まりが変わっていたら不一致。
    registration に無い決まりは、since が YYYY-MM-DD で、固定した registered_at と検査した当日（today）より後なら許す（初めて見た時に hashes.jsonl に記録し、
    以後はその記録とも比べる）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = json.loads(Path(reg_path or REG).read_text(encoding="utf-8"))
    cur = frozen_hashes(cfg)
    f = out_dir / "hashes.jsonl"
    recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.exists() else []
    known = dict(reg["rules"])
    for r in recs:
        for i, h in r.get("rules", {}).items():
            known.setdefault(i, h)
    bad = ["thresholds"] if reg["thresholds"] != cur["thresholds"] else []
    bad += [i for i, h in known.items() if cur["rules"].get(i) != h]                 # 登録済み・記録済みの決まりの変更・削除
    by_id = {r["id"]: r for r in cfg["frozen"]["rules"]}
    added = sorted(i for i in cur["rules"] if i not in known)
    floor = max(str(reg["registered_at"]), str(today or datetime.now().strftime("%Y-%m-%d")))
    for i in added:
        sn = by_id[i].get("since")
        if not _is_date(sn):
            bad.append(f"{i}（since が YYYY-MM-DD でない追加）")
        elif sn <= floor:
            bad.append(f"{i}（since {sn} が {floor} より後でない。さかのぼる追加）")
    if not bad and write and (added or not recs):
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "thresholds": cur["thresholds"],
                                 "rules": {i: cur["rules"][i] for i in (added or sorted(cur["rules"]))}, "added": added}, ensure_ascii=False) + "\n")
    return (not bad), bad


# ───────────────────────── 1 レースの判定 ─────────────────────────
def parse_payoffs(raw, fix_rows: list[dict] | None = None) -> dict | None:
    """race_result_flat.payoff → {'tansho': {馬番: 円}, 'fukusho': {...}, 'umaren': {(a,b): 円}, 'wide': {...}}。4 券種のどれかが読めなければ None。
    連結された行（popularity の無い 1 行に複数の組がつながったもの）は読めないとみなす。同着で同じ組が複数行あれば足す。"""
    try:
        js = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
    if fix_rows:                                      # 連結された複勝・ワイドは、payoff_repair の補正表（手元の data/local/meta/payoff_fix）で置き換える
        from src.scraper.payoff_repair import apply_fix

        js = apply_fix(js, fix_rows)
    out = {}
    for jp, key, arity in (("単勝", "tansho", 1), ("複勝", "fukusho", 1), ("馬連", "umaren", 2), ("ワイド", "wide", 2)):
        v = js.get(jp)
        if v is None:
            return None
        if jp in ("複勝", "ワイド") and isinstance(v, dict) and "popularity" not in v:   # 複数行のはずの券種が 1 つに連結された保存（payoff_repair.is_concatenated と同じ判定）
            return None
        d = {}
        for e in (v if isinstance(v, list) else [v]):
            if not isinstance(e, dict):
                return None
            try:
                combo = tuple(int(x) for x in str(e["numbers"]).replace("→", "-").replace(" ", "").split("-"))
                pay = float(str(e["payout"]).replace(",", ""))
            except (KeyError, ValueError, TypeError):
                return None
            if len(combo) != arity or max(combo) > 18:
                return None
            d[combo[0] if arity == 1 else tuple(sorted(combo))] = d.get(combo[0] if arity == 1 else tuple(sorted(combo)), 0.0) + pay
        if not d:
            return None
        out[key] = d
    return out


def bet_returns(picks: list[int], pay: dict) -> dict:
    """◎（picks[0]）の単勝・複勝と、◎→○▲（picks[1]・picks[2]）の馬連・ワイドの払戻（円）。"""
    a = picks[0]
    out = {"tansho": pay["tansho"].get(a, 0.0), "fukusho": pay["fukusho"].get(a, 0.0)}
    for bt in ("umaren", "wide"):
        out[bt] = sum(pay[bt].get(tuple(sorted((a, b))), 0.0) for b in picks[1:3])
    return out


def confidence(p: np.ndarray) -> dict:
    """出走馬で合計 1 に正規化した勝率から、1 番手の確率・1〜2 番手の差・平たさ（正規化エントロピー）。p は降順。"""
    q = p / p.sum()
    nz = q[q > 0]
    return {"p1": float(q[0]), "gap": float(q[0] - q[1]), "flat": float(-(nz * np.log(nz)).sum() / np.log(len(q)))}


def conditions(rec: dict, fr: dict, variant: str) -> dict:
    """台帳の 1 行 → 条件名 → 当たるか（オッズに関わる条件は、その時点のオッズが無ければ None）。"""
    fav = rec.get(f"fav_{variant}") or {}
    pop, odds = fav.get("pop"), fav.get("odds")
    have = pop is not None and odds is not None
    c = {"p1_low": rec["p1"] < fr["p1_low"], "p1_high": rec["p1"] >= fr["p1_high"], "gap_low": rec["gap"] < fr["gap_low"],
         "flat_low": rec["flat"] < fr["flat_low"], "flat_high": rec["flat"] >= fr["flat_high"],
         "field_small": rec["field"] <= fr["field_small_max"], "field_large": rec["field"] >= fr["field_large_min"],
         "going_bad": rec.get("going") == "不良",
         "fav": (pop == 1) if have else None, "odds_lt": (odds < fr["odds_lt"]) if have else None,
         "pop2to5": (2 <= pop <= 5) if have else None, "pop4to5": (4 <= pop <= 5) if have else None}
    return c


def decide(rule: dict, cond: dict) -> bool | None:
    """buy_if（どれかに当たれば買う。空なら全部）と skip_if（どれかに当たれば買わない）。判定に要る条件が None なら None（判定できない）。"""
    need = list(rule["buy_if"]) + list(rule["skip_if"])
    if any(cond.get(k) is None for k in need):
        return None
    buy = (not rule["buy_if"]) or any(cond[k] for k in rule["buy_if"])
    return bool(buy and not any(cond[k] for k in rule["skip_if"]))


def race_record(rid: str, date: str, pred: pd.DataFrame, runners: pd.DataFrame, pay: dict, pre: pd.DataFrame | None) -> tuple[dict | None, str]:
    """1 レースの台帳の中身（決まりの判定の前まで）。runners: 確定の出走馬（馬番・horse_id・人気・オッズ・馬場）。
    走った馬 = 確定オッズのある馬。取消の馬は予測から外して勝率を正規化し直す。走った馬が予測に全部そろわないレースは扱わない
    （◎○▲・頭数・市場の買い目の母集団をそろえるため）。発走前の人気は、走った馬だけでオッズの順に数え直す（取消の馬の分を詰める）。"""
    ran = runners[runners["odds"].notna() & (runners["odds"] > 0)]
    g = pred[["horse_id", "p_win"]].assign(horse_id=lambda x: x["horse_id"].astype(str)).merge(ran, on="horse_id", how="inner")
    if len(ran) < 3:
        return None, "出走 3 頭未満"
    if len(g) < len(ran):
        return None, "予測に無い出走馬がいる"
    if g["p_win"].sum() <= 0:
        return None, "勝率が全部 0"
    g = g.sort_values(["p_win", "horse_number"], ascending=[False, True], kind="stable")
    picks = [int(x) for x in g["horse_number"].head(3)]
    rec = {"race_id": rid, "date": date, "top": picks, "field": int(len(ran)), "going": str(runners["going"].iloc[0]),
           **confidence(g["p_win"].to_numpy(float)), "ret": bet_returns(picks, pay)}
    top = g.iloc[0]
    rec["fav_final"] = {"pop": int(top["popularity"]) if pd.notna(top["popularity"]) and top["popularity"] > 0 else None, "odds": float(top["odds"])}
    mk = ran.sort_values(["popularity", "odds", "horse_number"], kind="stable")
    mpicks = [int(x) for x in mk["horse_number"].head(3)]
    rec["mkt_final"] = {"top": mpicks, "ret": bet_returns(mpicks, pay)}
    rec["fav_pre"], rec["mkt_pre"], rec["pre_phase"] = None, None, None
    if pre is not None and len(pre):
        p = pre[pre["horse_number"].isin(set(ran["horse_number"].astype(int))) & (pre["win_odds"] > 0)]
        if len(p) >= 3:
            p = p.sort_values(["win_odds", "popularity", "horse_number"], kind="stable").assign(rank=lambda x: np.arange(1, len(x) + 1))
            rec["pre_phase"] = str(p["phase"].iloc[0])
            m = p[p["horse_number"] == picks[0]]
            if len(m):
                rec["fav_pre"] = {"pop": int(m["rank"].iloc[0]), "odds": float(m["win_odds"].iloc[0])}
            mp = [int(x) for x in p["horse_number"].head(3)]
            rec["mkt_pre"] = {"top": mp, "ret": bet_returns(mp, pay)}
    return rec, ""


def pick_pre(pre_all: pd.DataFrame, rid: str, post: datetime | None) -> pd.DataFrame | None:
    """そのレースの発走前の最後のスナップショット（PRE_PHASES の順で、発走時刻より前のもの）。"""
    d = pre_all[pre_all["race_id"] == rid]
    if d.empty:
        return None
    for ph in PRE_PHASES:
        s = d[d["phase"] == ph]
        if s.empty:
            continue
        try:
            at = datetime.fromisoformat(str(s["snapshot_at"].iloc[0]))
        except (TypeError, ValueError):
            continue
        if post is None or at < post:
            return s
    return None


# ───────────────────────── 追跡本体 ─────────────────────────
class Results:
    """1 年分の確定の結果（出走馬・人気・確定オッズ・馬場・払戻・発走時刻）。"""

    def __init__(self, year: int, tables: Path = ft.TABLES):
        f = tables / str(year) / "race_result_flat.parquet"
        cols = ["race_id", "date", "start_time", "horse_number", "horse_id", "odds", "popularity", "track_condition", "payoff"]
        r = pd.read_parquet(f, columns=cols) if f.exists() else pd.DataFrame(columns=cols)
        r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
        for c in ("horse_number", "odds", "popularity"):
            r[c] = pd.to_numeric(r[c], errors="coerce")
        r["going"] = r["track_condition"].astype(str).replace({"稍": "稍重", "不": "不良"})
        self.by = {rid: g for rid, g in r.groupby("race_id")}
        try:
            from src.scraper.payoff_repair import load_fix

            self.fix = load_fix(year)                  # 手元の補正表だけを読む（GCS は読まない）。無ければ空
        except Exception:  # noqa: BLE001
            self.fix = {}

    def get(self, rid: str) -> pd.DataFrame | None:
        return self.by.get(rid)


def era_of(cfg: dict, model: str, date: str) -> str:
    rows = sorted((cfg.get("eras") or {}).get(model, []), key=lambda e: e["from"])
    era = ""
    for e in rows:
        if date >= e["from"]:
            era = e["bundle"]
    return era


def run_track(days: list[str], cfg: dict, out_dir: Path, tables: Path = ft.TABLES, pre_path: Path = PRE_ODDS, root: Path = ROOT, verbose: bool = True,
              today: str | None = None) -> dict:
    """台帳に追記する。処理済みの出どころ × 日（done_days.json）は読み直さない。
    forward の出どころで、発走前オッズがまだ無いレースは、PRE_WAIT_DAYS 日までは書かずに次回に回す（後から補われることがある）。"""
    from datetime import date as _date

    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / "ledger.jsonl"
    seen = ft.ledger_keys(ledger)
    fr = cfg["frozen"]
    today = today or _date.today().isoformat()
    done_f = out_dir / "done_days.json"
    done = json.loads(done_f.read_text(encoding="utf-8")) if done_f.exists() else {}
    pre_all = pd.read_parquet(pre_path) if Path(pre_path).exists() else pd.DataFrame(columns=["race_id", "phase", "snapshot_at", "horse_number", "win_odds", "popularity"])
    pre_all["race_id"] = pre_all["race_id"].astype(str)
    res: dict[int, Results] = {}
    new, skipped = [], {}

    def skip(why: str) -> None:
        skipped[why] = skipped.get(why, 0) + 1

    for day in sorted(days):
        y = int(day[:4])
        date = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        for src in cfg["sources"]:
            model, kind = src["model"], src["kind"]
            sk = f"{model}|{kind}|{src['dir']}"
            if day in set(done.get(sk, [])):
                continue
            base = Path(src["dir"]) if Path(src["dir"]).is_absolute() else root / src["dir"]
            if not base.exists():
                continue
            preds = ft.read_pred("x", day, {"x": base})
            if not preds:
                continue
            if y not in res:
                res[y] = Results(y, tables)
            complete = True
            for rid, pred in preds:
                key = f"{model}|{kind}|{rid}"
                if key in seen:
                    continue
                run = res[y].get(rid)
                if run is None or not isinstance(run["payoff"].iloc[0], str):
                    skip("結果がまだ")
                    complete = False
                    continue
                recent = (_date.fromisoformat(today) - _date.fromisoformat(date)).days <= RETRY_DAYS
                pay = parse_payoffs(run["payoff"].iloc[0], res[y].fix.get(rid))
                if pay is None:
                    skip("払戻が読めない")
                    complete = complete and not recent           # 直近は、表の作り直し・補正を待って読み直す
                    continue
                try:
                    post = datetime.fromisoformat(f"{date}T{str(run['start_time'].iloc[0])[:5]}:00+09:00")
                except (TypeError, ValueError):
                    post = None
                rec, why = race_record(rid, date, pred, run[["horse_id", "horse_number", "odds", "popularity", "going"]], pay, pick_pre(pre_all, rid, post))
                if rec is None:
                    skip(why)
                    complete = complete and not recent
                    continue
                if kind == "forward" and rec["mkt_pre"] is None and (_date.fromisoformat(today) - _date.fromisoformat(date)).days <= PRE_WAIT_DAYS:
                    skip("発走前オッズ待ち")
                    complete = False
                    continue
                rec.update({"key": key, "model": model, "source": kind})
                rec["buy"] = {}
                for v in VARIANTS:
                    cond = conditions(rec, fr, v)
                    rec["buy"][v] = {r["id"]: decide(r, cond) for r in fr["rules"] if date >= r.get("since", "0000-00-00")}   # since より前の日は、その決まりのキーを書かない
                new.append(rec)
                seen.add(key)
            if complete:
                done.setdefault(sk, []).append(day)
    if new:
        with ledger.open("a", encoding="utf-8") as fh:
            for r in new:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    done_f.write_text(json.dumps({k: sorted(set(v)) for k, v in done.items()}, ensure_ascii=False), encoding="utf-8")
    summary = {"added": len(new), "skipped": skipped, "days": len(days)}
    with (out_dir / "runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "range": [min(days), max(days)] if days else [], **summary}, ensure_ascii=False) + "\n")
    if verbose:
        print(json.dumps(summary, ensure_ascii=False))
    return summary


# ───────────────────────── 集計 ─────────────────────────
def summarize(ret: np.ndarray, mret: np.ndarray | None, dates: np.ndarray, cost: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """買ったレースの回収率・的中率・区間（レース単位のブートストラップ）、同じレースを市場で買った回収率との差、1 日の振れ。"""
    n = len(ret)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rb = ret[idx].sum(1) / (n * cost)
    out = {"n": int(n), "hit_rate": float((ret > 0).mean()), "avg_pay": float(ret[ret > 0].mean()) if (ret > 0).any() else 0.0,
           "roi": float(ret.sum() / (n * cost)), "roi_lo": float(np.quantile(rb, 0.025)), "roi_hi": float(np.quantile(rb, 0.975))}
    if mret is not None:
        mb = mret[idx].sum(1) / (n * cost)
        out.update({"mkt_roi": float(mret.sum() / (n * cost)), "vs_mkt": out["roi"] - float(mret.sum() / (n * cost)),
                    "vs_mkt_lo": float(np.quantile(rb - mb, 0.025)), "vs_mkt_hi": float(np.quantile(rb - mb, 0.975))})
    d = pd.DataFrame({"date": dates, "ret": ret}).groupby("date").agg(ret=("ret", "sum"), n=("ret", "size"))
    stake = d["n"] * cost
    pl = (d["ret"] - stake).cumsum().to_numpy()
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], pl]))[1:] - pl))
    out.update({"days": int(len(d)), "day_roi_sd": float((d["ret"] / stake).std(ddof=1)) if len(d) > 1 else None,
                "max_drawdown_yen": dd, "drawdown_days": dd / float(stake.mean())})
    return out


def scorecard(led: pd.DataFrame, cfg: dict) -> dict:
    fr = cfg["frozen"]
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "track_from": cfg["frozen"]["track_from"],
           "min_races_to_judge": fr["min_races_to_judge"], "cards": []}
    if led.empty:
        return out
    excl = ft.excluded_dates({"ft": json.loads(ft.CFG_FT.read_text(encoding="utf-8")) if ft.CFG_FT.exists() else {}})
    drop = led["date"].astype(str).isin(excl)                     # 除外日は台帳に焼き込まず、集計のたびに日付で外す（後から足した日にも効く）
    out["excluded_dates"] = {d: {"reason": excl[d], "rows": int((led["date"].astype(str) == d).sum())} for d in sorted(excl) if (led["date"].astype(str) == d).any()}
    out["excluded_rows"] = int(drop.sum())
    out["excluded_dates_all"] = sorted(excl)
    led = led[~drop].copy()
    led["era"] = [era_of(cfg, m, d) for m, d in zip(led["model"], led["date"].astype(str))]          # 区分も集計のたびに付け直す
    led["period"] = np.where((led["source"] == "forward") & (led["date"].astype(str) >= cfg["frozen"]["track_from"]), "forward", "reference")
    rules = {r["id"]: r for r in fr["rules"]}
    for (period, model, source, era), g in led.groupby(["period", "model", "source", "era"], dropna=False):
        for v in VARIANTS:
            mcol = f"mkt_{v}"
            gv = g if v == "final" else g[g[mcol].map(lambda x: isinstance(x, dict))] if mcol in g.columns else g.iloc[0:0]   # pre は発走前オッズのあるレースだけ
            for rid_, rule in rules.items():
                bt = rule["bet"]
                active = gv["buy"].map(lambda b: rid_ in (b.get(v) or {}))       # since より前の行にはキーが無い（対象外）
                buy = gv["buy"].map(lambda b: (b.get(v) or {}).get(rid_))
                sel = gv[active & (buy == True)]  # noqa: E712
                undecided = int((active & buy.isna()).sum())               # 判定に要るオッズが無かった
                ret = np.array([r[bt] for r in sel["ret"]], dtype=float)
                mk = sel[mcol] if mcol in sel.columns else pd.Series([None] * len(sel))
                have_m = mk.map(lambda x: isinstance(x, dict)).to_numpy(bool) if len(sel) else np.array([], bool)
                mret = np.array([x["ret"][bt] for x in mk[have_m]], dtype=float) if have_m.all() and len(sel) else None
                s = summarize(ret, mret, sel["date"].to_numpy(), COST[bt])
                out["cards"].append({"period": period, "model": model, "source": source, "era": era, "variant": v, "rule": rid_, "bet": bt,
                                     "label": rule["label"], "races": int(active.sum()), "undecided": undecided,
                                     "progress": (s["n"] / fr["min_races_to_judge"]) if s.get("n") else 0.0, **s})
    return out


def render_md(card: dict) -> str:
    L = ["# 買い目の決まりの前向き追跡（rule_tracking）", "",
         f"事前登録: config/rule_tracking.json（前向きの記録は {card['track_from']} から）。生成 {card['generated_at']}。",
         f"**約 {card['min_races_to_judge']:,} レースたまるまでは途中の値で判断しない**（区間が広い）。決まりは選び直さない。",
         "回収率の括弧はレース単位のブートストラップの 95% 区間。市場 = 同じレースを市場の 1 番人気（→ 2・3 番人気）で買った回収率。",
         "final = 確定オッズの人気で判定・市場も確定の人気。pre = 発走前の最後のスナップショットで判定（単勝の決まりと市場だけが変わる）。", ""]
    rc = card.get("registration_changed") or {}
    if rc.get("changed"):
        L += [f"**停止中: 事前登録の決まり・閾値が変わっている（{rc.get('items')}）。追記していない。**", ""]
    for r in card.get("excluded_dates_changes") or []:
        L += [f"**除外日の変更（{r['at']}）: 追加 {r['added']}・削除 {r['removed']}**（事前登録の後に変えた。結果を見てから外していないか確かめる）"]
    for d, v in (card.get("excluded_dates") or {}).items():
        L += [f"集計から外した日: {d}（{v['rows']} 行）— {v['reason']}"]
    if card.get("excluded_dates"):
        L.append("")
    cards = pd.DataFrame(card["cards"])
    if cards.empty:
        return "\n".join(L + ["まだ確定したレースがありません。"]) + "\n"
    for period, title in (("forward", "前向き（固定後・日々の予測）"), ("reference", "固定前の参考（track_from より前・後から作った予測を含む）")):
        c = cards[cards["period"] == period]
        L += [f"## {title}", ""]
        if c.empty:
            L += ["まだありません。", ""]
            continue
        for (model, source, era, v), g in c.groupby(["model", "source", "era", "variant"], dropna=False):
            if v == "pre" and int(g["races"].max()) == 0:
                continue
            L += [f"### {model}（{source}{'・' + era if era else ''}）・オッズ {v}・確定したレース {int(g['races'].max()):,}", "",
                  "| 決まり | 買ったレース | 判定できず | 的中率 | 平均払戻 | 回収率（区間） | 市場（同じレース） | 市場との差（区間） | 日数 | 1日の回収率の標準偏差 | 落ち込み（日数分） | 判断までの進み |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in g.iterrows():
                if not r.get("n"):
                    L.append(f"| {r['rule']} {r['label']} | 0 | {r['undecided']} | | | | | | | | | |")
                    continue
                mk = "" if pd.isna(r.get("mkt_roi")) else f"{r['mkt_roi']:.3f}"
                vs = "" if pd.isna(r.get("vs_mkt")) else f"{r['vs_mkt']:+.3f}（{r['vs_mkt_lo']:+.3f}〜{r['vs_mkt_hi']:+.3f}）"
                sd = "" if r.get("day_roi_sd") is None or pd.isna(r.get("day_roi_sd")) else f"{r['day_roi_sd']:.2f}"
                L.append(f"| {r['rule']} {r['label']} | {int(r['n']):,} | {r['undecided']} | {r['hit_rate']:.3f} | {r['avg_pay']:.0f} | "
                         f"{r['roi']:.3f}（{r['roi_lo']:.3f}〜{r['roi_hi']:.3f}） | {mk} | {vs} | {r['days']} | {sd} | {r['drawdown_days']:.1f} | {min(r['progress'], 9.99):.0%} |")
            L.append("")
    return "\n".join(L) + "\n"


def render_html(md: str) -> str:
    body = []
    for line in md.splitlines():
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if body and body[-1].startswith("<tr>") and not line.startswith("|"):
            body.append("</table>")
        if line.startswith("#"):
            lv = len(line) - len(line.lstrip("#"))
            body.append(f"<h{lv}>{html.escape(line.lstrip('#').strip())}</h{lv}>")
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
        if line.startswith("### "):
            body.append("<table border=1 cellpadding=3 style='border-collapse:collapse;font-size:13px'>")
    return "<!doctype html><meta charset=utf-8><title>買い目の決まりの追跡</title>" + "\n".join(body)


def log_excluded(out: Path, excl: list[str]) -> list[dict]:
    """除外日の一覧が変わったら excluded_log.jsonl に追記し、記録の全体を返す（後から日を足したことを見えるようにする）。"""
    f = out / "excluded_log.jsonl"
    recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.exists() else []
    if not recs or recs[-1]["dates"] != excl:
        rec = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dates": excl,
               "added": sorted(set(excl) - set(recs[-1]["dates"])) if recs else [], "removed": sorted(set(recs[-1]["dates"]) - set(excl)) if recs else []}
        out.mkdir(parents=True, exist_ok=True)
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        recs.append(rec)
    return recs


def cmd_report(a) -> None:
    out = Path(a.out_dir)
    cfg = load_cfg()
    led = ft.load_ledger(out / "ledger.jsonl")
    card = scorecard(led, cfg)
    same, bad = check_registration(cfg, out, write=False)
    card["registration_changed"] = {"changed": not same, "items": bad}
    excl = sorted(ft.excluded_dates({"ft": json.loads(ft.CFG_FT.read_text(encoding="utf-8")) if ft.CFG_FT.exists() else {}}))
    log = log_excluded(out, excl)
    card["excluded_dates_changes"] = [r for r in log[1:]]          # 最初の記録より後に足した・外した除外日
    out.mkdir(parents=True, exist_ok=True)
    md = render_md(card)
    (out / "latest.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "latest.md").write_text(md, encoding="utf-8")
    (out / "latest.html").write_text(render_html(md), encoding="utf-8")
    print(json.dumps({"cards": len(card["cards"]), "registration_changed": card["registration_changed"]}, ensure_ascii=False))


def cmd_track(a) -> None:
    out = Path(a.out_dir)
    cfg = load_cfg()
    same, bad = check_registration(cfg, out)
    if not same:
        cmd_report(argparse.Namespace(out_dir=str(out)))          # スコアカードにも停止を出す（registration_changed）
        print(f"REGISTRATION_MISMATCH: 事前登録の決まり・閾値が変わっている（{bad}）。追記を拒否する。決まりを足すなら since を付けて足す。"
              "変えるなら RULE_TRACKING_DIR を新しくして別の履歴にする", file=sys.stderr)
        sys.exit(3)
    to = a.to or datetime.now().strftime("%Y%m%d")
    if a.days:
        days = a.days.split(",")
    else:
        found: set[str] = set()
        for src in cfg["sources"]:
            base = Path(src["dir"]) if Path(src["dir"]).is_absolute() else ROOT / src["dir"]
            if base.exists():
                found |= {p.name if p.is_dir() else p.stem for p in base.iterdir() if (p.name if p.is_dir() else p.stem).isdigit()}
        days = sorted(d for d in found if getattr(a, "from") <= d <= to)
    run_track(days, cfg, out)
    cmd_report(argparse.Namespace(out_dir=str(out)))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("track")
    t.add_argument("--from", dest="from", default="20260101")
    t.add_argument("--to", default="")
    t.add_argument("--days", default="", help="YYYYMMDD,YYYYMMDD（指定時は期間より優先）")
    t.add_argument("--out-dir", default=str(OUT_DIR))
    r = sub.add_parser("report")
    r.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    {"track": cmd_track, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
