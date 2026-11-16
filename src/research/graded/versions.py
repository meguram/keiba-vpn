"""重賞の条件の版（条件の更新のバッチ。docs/ideas/graded_rule_based_handicapper.md §27）。

- 窓: 重賞ごとに、結果のある直近 2 回を「検証」、それより前（2018 年以降）を「発見」にする。発見の回だけで B の作法（20 サイクルと同じ
  候補、cycles.space）で ◎ と消しの規則を選び、検証の回で「その規則で買ったらどうなるか」を 1 回だけ数える。検証は採否に使わず表示だけ。
  直近の回が 1 回しかなければ、その 1 回を検証にする。発見の回が無ければ規則を作らない（前日版の ◎ だけ）。
- 予想には、発見だけで決めた規則をそのまま使う（検証の回を足して作り直さない。表示する検証の成績が、予想に使う規則そのものの成績になるように）。
- 版: 重賞ごとに data/meta/graded/rules_versions.json に足していく（書き換えない）。最初の版は、20 サイクルの B（prereg_cycles.json、
  発見 2018〜2024・検証 2025〜2026）。古い = 版の窓の後に、その重賞の新しい回の結果がある。前日の計算（upcoming）が、古ければ先に作り直す。
- 材料（使う型・縮める強さ・前日版の点の型。prereg.json）は年 1 回、前々年の末より前で決め直す（手で流す。§27）。材料が古いときは
  計算を止めずに警告する。
候補の作り方（cycles.space）はデータを見ない（型の一覧と乱数だけ）ので、窓を変えても候補は同じ。点の元の集計は各レースの前日まで（as-of）。
版は前日の予想に使うので、当日の値を使う決め方（picks.needs_dayof）は候補から外す（最初の版＝20 サイクルの B には、当日の値の規則が残る）。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
JST = timezone(timedelta(hours=9))
SEED = 20261115            # 20 サイクルと同じ乱数の種（候補を同じにする）
DEV_FROM = 2018            # 発見に使う最初の年（表が 2018 年から）
N_VAL = 2                  # 検証の回数（直近 2 回）
V0_ID = "v0-cycles-2026-11-15"
V0_RULES_END = "2024-12-31"   # 最初の版の規則を選んだデータの終わり（これより後の回にだけ使える）
V0_DATA_END = "2026-11-08"    # 最初の版（20 サイクル）の表の結果の終わり（検証の値を数えた最後の日。凍結の日ではない）


def ledger_path() -> Path:
    return Path(os.environ.get("GRADED_RULES_VERSIONS") or REPO / "data/meta/graded/rules_versions.json")


def load_ledger() -> dict:
    try:
        return json.loads(ledger_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"note": "重賞ごとの条件の版（versions.py）。足していくだけで、書き換えない。最初の版は 20 サイクルの B（prereg_cycles.json）", "rkeys": {}}


def save_ledger(L: dict) -> None:
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(L, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def append_version(rkey: str, v: dict) -> dict:
    """台帳に 1 つ足す（ロックを取り、書く直前に読み直して足す。root の裏の計算と手で流す計算が重なっても、相手の版を消さない）。"""
    import fcntl
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix(".lock"), "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            L = load_ledger()
            vs = L.setdefault("rkeys", {}).setdefault(rkey, [])
            if not any(x.get("sha") == v.get("sha") and x.get("before") == v.get("before") for x in vs):
                vs.append(v)
            save_ledger(L)
            return L
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _sha(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def v0(prereg: dict, rkey: str) -> dict | None:
    """最初の版（20 サイクルの B）。規則は 2018〜2024 年で選び、検証は 2025 年〜表の結果の終わり（V0_DATA_END）。"""
    RB = prereg["rules"]["B"]
    h = RB["honmei_by_rkey"].get(rkey)
    if h is None:
        return None
    return {"id": V0_ID, "made_at": prereg["frozen_at"], "before": "2025-01-01", "discovery": {"from": f"{DEV_FROM}-01-01", "to": V0_RULES_END},
            "validation": {"from": "2025-01-01", "to": V0_DATA_END}, "honmei": h, "kill": RB["kill_by_rkey"].get(rkey),
            "sha": prereg["sha256_rules"], "from": "prereg_cycles.json"}


def current(rkey: str, prereg: dict, ledger: dict | None = None, date: str | None = None) -> dict | None:
    """その重賞で、date の回に使える版のうち最新。使える = 版を作ったときの before が date 以前（その版の規則は date より前の結果だけで
    選んだ）。台帳に無ければ最初の版（2025 年以降の回だけ）。date を渡さなければ台帳の最新（表示用）。"""
    L = ledger if ledger is not None else load_ledger()
    vs = [v for v in (L.get("rkeys") or {}).get(rkey) or [] if date is None or str(v.get("before", "9999")) <= date]
    if vs:
        return vs[-1]
    v = v0(prereg, rkey)
    return v if v is not None and (date is None or date > V0_RULES_END) else None


def is_stale(ver: dict | None, edition_dates: list[str], before: str) -> bool:
    """古いか: 版の窓（検証の終わり）より後で、before（予想する回の日）より前に、その重賞の結果のある回がある。"""
    if ver is None:
        return bool([d for d in edition_dates if d < before])
    end = ver["validation"]["to"]
    return any(end < d < before for d in edition_dates)


def windows(dates: list[str], before: str) -> tuple[list[str], list[str]]:
    """（発見の回, 検証の回）。before より前の結果のある回のうち、直近 N_VAL 回が検証、それより前（DEV_FROM 年以降）が発見。"""
    past = sorted(d for d in dates if d < before)
    val = past[-N_VAL:]
    disc = [d for d in past[: len(past) - len(val)] if int(d[:4]) >= DEV_FROM]
    return disc, val


def _race_cols(t):
    import numpy as np
    dt = t.r.groupby(t.race_of)["dt"].first().astype(str).str[:10].to_numpy()
    has = ~t.r.groupby(t.race_of)["fin"].apply(lambda s: s.isna().all()).to_numpy()   # 出馬表の段階の回（結果なし）は除く
    return dt, t.race["rkey"].to_numpy(), np.asarray(has, bool)


def build(t, rkeys: list[str], before: str, cycles: int = 20) -> dict[str, dict]:
    """表 t（cycles.Table。before より前の回は結果つき）で、rkeys の版を作る。戻り値は {重賞: 版}。発見の回が無い重賞は版を作らない。"""
    import numpy as np

    from src.research.graded import cycles as CY
    from src.research.graded import playbook as PB

    dt, rk, has = _race_cols(t)
    win = {}
    for k in rkeys:
        disc, val = windows(sorted(set(dt[(rk == k) & has])), before)
        if disc:
            win[k] = (disc, val)
    if not win:
        return {}
    keys = sorted(win)
    # 対象の重賞の回だけの表にする（1 重賞なら数十秒。値は重賞ごとに数えるので、ほかの重賞の行は要らない）
    m = np.isin(rk[t.race_of], keys)
    if not m.all():
        t = CY.Table(t.r[m].reset_index(drop=True), t.F[m], t.names)
        dt, rk, has = _race_cols(t)
    dv = np.zeros(len(rk), bool)
    ev = np.zeros(len(rk), bool)
    for k in keys:
        disc, val = win[k]
        dv |= (rk == k) & np.isin(dt, disc) & has
        ev |= (rk == k) & np.isin(dt, val) & has
    B = CY.Best(t, dv=dv, ev=ev, rkeys=keys)
    rng = np.random.default_rng(SEED)
    terms: dict[str, dict] = {}
    from src.research.graded.picks import needs_dayof
    for c in range(1, cycles + 1):
        _, cand = CY.space(c, t, rng)
        # 版は前日の予想に使うので、当日の値（馬体重・馬場・当日版の点など）を使う決め方は選ばない（乱数の流れは 20 サイクルと同じ）
        cand = [(d, n) for d, n in cand if not needs_dayof(d)]
        if not cand:
            continue
        W, lab = CY.matrix(t, cand)
        terms.update({n: d for d, n in cand})
        B.update(W, lab)
    rep = B.report()["B"]
    hp = {x["rkey"]: x for x in rep["honmei_by_rkey"]["per_rkey"]}
    kp = {x["rkey"]: x for x in rep["kill_by_rkey"]["per_rkey"]}
    out = {}
    now = datetime.now(JST).isoformat(timespec="seconds")
    for k in keys:
        disc, val = win[k]
        h = CY.rule_of(hp[k]["rule"], terms) if hp[k]["rule"] else None
        kl = CY.rule_of(kp[k]["rule"], terms) if kp[k]["rule"] else None
        vi = np.flatnonzero(ev & (rk == k))
        vh = PB.honmei_stats(t, CY.honmei(t, CY._vec(t, h["terms"])[:, None])[:, 0], vi) if h else None
        vk = PB.kill_stats(t, CY._vec(t, kl["terms"]), kl["k"], vi) if kl else None
        body = {"discovery": {"from": disc[0], "to": disc[-1], "n": len(disc)},
                "validation": {"from": val[0], "to": val[-1], "n": len(val)} if val else {"from": None, "to": disc[-1], "n": 0},
                "honmei": h, "kill": kl}
        out[k] = body | {"id": f"{before}-{k}", "made_at": now, "before": before, "val_stats": {"honmei": vh, "kill": vk},
                         "flag": "val_weak" if vh and vh.get("roi_t") is not None and vh["roi_t"] < 1.0 else None, "sha": _sha(body)}
    return out


def material_cutoff(path: Path | None = None) -> str:
    """材料（prereg.json）の区切り（periods の確認の終わりの年の末）。"""
    try:
        P = json.loads((path or REPO / "src/research/graded/prereg.json").read_text(encoding="utf-8"))
        return f"{int(P['periods']['confirm'][1])}-12-31"
    except (OSError, ValueError, KeyError, TypeError):
        return V0_RULES_END


def material_warning(race_date: str, cutoff: str | None = None) -> str | None:
    """材料（prereg.json・prereg_kill.json・prereg_roi.json）の区切りが、その年の区切り（前々年の末）より古ければ警告の文。
    材料は年 1 回、手で決め直す（§27）。"""
    material = cutoff or material_cutoff()
    want = f"{int(race_date[:4]) - 2}-12-31"
    if material < want:
        return f"材料（使う型・縮める強さ）が {material[:4]} 年までで決めたままです（{want[:4]} 年までで決め直す時期。§27 の手順で流してください）"
    return None
