"""``config/gcp_job_map.json``（ジョブ対応表）の読み込み・検証と、cron 式の小さな評価器。

対応表は『ホストの cron → GCP（Cloud Run Jobs）/ VPS 実行役 / ホストに残す / 廃止』の正本。
Scheduler 定義の生成・ディスパッチャの許可リスト・並走検証・切替判定がすべてここを読む。
cron 式は 5 フィールド（分 時 日 月 曜）。``*``・``a-b``・``a,b``・``*/n``・``a-b/n`` に対応する（croniter は使わない）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "config" / "gcp_job_map.json"

DESTINATIONS = ("gcp", "vps", "host", "retire")
SIZES = ("s", "m", "l")
PARITY_KINDS = ("none", "parquet_frame", "json_tree", "file_hash")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


# ───────────────────────── cron 式 ─────────────────────────

_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]


def _parse_field(field: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for part in field.split(","):
        step = None
        if "/" in part:
            part, s = part.split("/", 1)
            step = int(s)
            if step < 1:
                raise ValueError(f"step が不正: {field}")
        if part in ("*", ""):
            a, b = lo, hi
        elif "-" in part:
            x, y = part.split("-", 1)
            a, b = int(x), int(y)
        else:
            a = int(part)
            b = hi if step is not None else a
        if a < lo or b > hi or a > b:
            raise ValueError(f"範囲外: {field}")
        out.update(range(a, b + 1, step or 1))
    return out


def parse_cron(expr: str) -> tuple[set[int], set[int], set[int], set[int], set[int], bool, bool]:
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"cron は 5 フィールド: {expr!r}")
    sets = [_parse_field(p, lo, hi) for p, (lo, hi) in zip(parts, _RANGES)]
    sets[4] = {d % 7 for d in sets[4]}      # 7 = 日曜
    return (*sets, parts[2] != "*", parts[4] != "*")  # type: ignore[return-value]


def cron_matches(expr: str, dt: datetime) -> bool:
    """``dt``（分単位）が cron 式に一致するか。日と曜の両方が指定されたときは標準の cron どおり OR。"""
    mi, ho, dom, mon, dow, dom_r, dow_r = parse_cron(expr)
    if dt.minute not in mi or dt.hour not in ho or dt.month not in mon:
        return False
    d_ok, w_ok = dt.day in dom, ((dt.weekday() + 1) % 7) in dow
    if dom_r and dow_r:
        return d_ok or w_ok
    return d_ok and w_ok


def occurrences(expr: str, start: datetime, end: datetime) -> list[datetime]:
    """``start`` 以上 ``end`` 未満で cron 式に一致する分（分単位に切り捨て）。"""
    t = start.replace(second=0, microsecond=0)
    if t < start:
        t += timedelta(minutes=1)
    out: list[datetime] = []
    while t < end:
        if cron_matches(expr, t):
            out.append(t)
        t += timedelta(minutes=1)
    return out


def prev_occurrence(expr: str, now: datetime, *, max_days: int = 8) -> datetime | None:
    """``now`` 以前で最も近い一致（分単位）。見つからなければ None。"""
    t = now.replace(second=0, microsecond=0)
    for _ in range(max_days * 24 * 60):
        if cron_matches(expr, t):
            return t
        t -= timedelta(minutes=1)
    return None


def jst_to_utc_cron(expr: str) -> str:
    """JST の cron 式を UTC の式に直す（時刻が固定の式のみ。表示・突き合わせ用）。

    時を -9 して、日をまたぐときは曜日を 1 つ前にずらす。時が ``*`` や ``*/n`` の式（毎分・N 分おき）はそのまま返す。
    日（1〜31）の指定が付いた式は対象外。
    """
    mi, ho, dom, mon, dow, *_ = expr.split()
    if dom != "*":
        raise ValueError("日付指定つきの式は変換しない")
    if not re.fullmatch(r"[0-9,\-]+", ho):
        return expr if ho in ("*",) else _shift_hour_range(expr)
    hours = sorted(_parse_field(ho, 0, 23))
    new_hours, wrap = [], []
    for h in hours:
        nh = h - 9
        wrap.append(nh < 0)
        new_hours.append(nh % 24)
    if any(wrap) and not all(wrap):
        raise ValueError("日をまたぐ時刻と同じ日の時刻が混在する式は変換しない")
    new_dow = dow
    if all(wrap) and dow != "*":
        days = sorted({(d - 1) % 7 for d in _parse_field(dow, 0, 7)})
        new_dow = ",".join(str(d) for d in days)
    return f"{mi} {_compress(new_hours)} * {mon} {new_dow}"


def _shift_hour_range(expr: str) -> str:
    raise ValueError(f"時の範囲を含む式は変換しない: {expr!r}")


def _compress(values: list[int]) -> str:
    vals = sorted(values)
    if len(vals) > 2 and vals == list(range(vals[0], vals[-1] + 1)):
        return f"{vals[0]}-{vals[-1]}"
    return ",".join(str(v) for v in vals)


# ───────────────────────── 対応表 ─────────────────────────

@lru_cache(maxsize=4)
def _load_cached(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path: str | Path | None = None) -> dict[str, Any]:
    return _load_cached(str(path or DEFAULT_PATH))


def jobs(doc: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return list((doc or load())["jobs"])


def job_by_id(job_id: str, doc: dict[str, Any] | None = None) -> dict[str, Any] | None:
    for j in jobs(doc):
        if j["id"] == job_id:
            return j
    return None


def by_destination(dest: str, doc: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [j for j in jobs(doc) if j["destination"] == dest]


def group_of(group_id: str, doc: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return ((doc or load()).get("scheduler_groups") or {}).get(group_id)


def schedule_of(job: dict[str, Any], doc: dict[str, Any] | None = None) -> str | None:
    """ジョブの JST の実行式。GCP はグループの式、VPS は ``schedule_jst``。"""
    if job["destination"] == "gcp" and job.get("group"):
        g = group_of(job["group"], doc)
        return g["cron_jst"] if g else None
    return job.get("schedule_jst")


def cutover_policy(job: dict[str, Any], doc: dict[str, Any] | None = None) -> dict[str, Any]:
    base = dict((doc or load()).get("cutover_defaults") or {})
    base.update(job.get("cutover") or {})
    return base


def validate(doc: dict[str, Any], *, known_fetch_tasks: set[str] | None = None, agent_tasks: set[str] | None = None) -> list[str]:
    """対応表の整合性を検査して、問題の一覧を返す（空なら合格）。"""
    errs: list[str] = []
    rt = doc.get("runtime") or {}
    run_jobs = rt.get("run_jobs") or {}
    if set(run_jobs) != set(SIZES):
        errs.append(f"runtime.run_jobs は {SIZES} を持つこと")
    env = rt.get("env") or {}
    if env.get("KEIBA_ENV") != "prod" or env.get("KEIBA_RUN_LOCATION") != "gcp":
        errs.append("runtime.env に KEIBA_ENV=prod と KEIBA_RUN_LOCATION=gcp が必要（通知で環境・実行場所を見分けるため）")
    groups = doc.get("scheduler_groups") or {}
    bud = rt.get("scheduler_budget") or {}
    if not (bud.get("min", 0) <= len(groups) <= bud.get("max", 99)):
        errs.append(f"Scheduler のグループ数 {len(groups)} が予算 {bud} の範囲外")
    seen: set[str] = set()
    ids = [j.get("id") for j in doc.get("jobs") or []]
    for jid in ids:
        if not jid or not _ID.match(jid):
            errs.append(f"job id が不正: {jid!r}")
        if jid in seen:
            errs.append(f"job id が重複: {jid}")
        seen.add(jid)
    for gid, g in groups.items():
        try:
            parse_cron(g["cron_jst"])
        except Exception as e:  # noqa: BLE001
            errs.append(f"{gid}: cron_jst が不正: {e}")
        if g.get("size") not in SIZES:
            errs.append(f"{gid}: size は {SIZES}")
        if not g.get("jobs"):
            errs.append(f"{gid}: ジョブが空")
    models = doc.get("models") or {}
    for j in doc.get("jobs") or []:
        jid = j.get("id", "?")
        dest = j.get("destination")
        if dest not in DESTINATIONS:
            errs.append(f"{jid}: destination が不正: {dest!r}")
            continue
        if j.get("blockers") and j.get("ready"):
            errs.append(f"{jid}: blockers があるのに ready=true")
        if not j.get("blockers") and not j.get("ready"):
            errs.append(f"{jid}: ready=false なのに blockers が空（理由を書く）")
        if dest in ("host", "retire") and not j.get("reason"):
            errs.append(f"{jid}: host/retire には reason が必要")
        org = j.get("origin") or {}
        if org.get("cron_utc"):
            try:
                parse_cron(org["cron_utc"])
            except Exception as e:  # noqa: BLE001
                errs.append(f"{jid}: origin.cron_utc が不正: {e}")
        if j.get("when_jst"):
            try:
                parse_cron(j["when_jst"])
            except Exception as e:  # noqa: BLE001
                errs.append(f"{jid}: when_jst が不正: {e}")
        pk = (j.get("parity") or {}).get("kind", "none")
        if dest in ("gcp", "vps") and pk not in PARITY_KINDS:
            errs.append(f"{jid}: parity.kind が不正: {pk}")
        if dest == "gcp":
            g = j.get("group")
            if not g and j.get("trigger") not in ("cloud_tasks", "manual"):
                errs.append(f"{jid}: group か trigger（cloud_tasks / manual）が必要")
            if not g:
                sz = j.get("size")
                if sz not in SIZES:
                    errs.append(f"{jid}: group の無い GCP ジョブには size（{SIZES}）が必要")
                elif run_jobs.get(sz, {}).get("memory_mib", 0) < j.get("memory_mib", 0):
                    errs.append(f"{jid}: メモリ {j.get('memory_mib')}MiB が size {sz} を超える")
            if g:
                grp = groups.get(g)
                if not grp:
                    errs.append(f"{jid}: 未定義のグループ {g}")
                else:
                    if jid not in grp["jobs"]:
                        errs.append(f"{jid}: group {g} の jobs に載っていない")
                    if grp.get("size") in SIZES and run_jobs.get(grp["size"], {}).get("memory_mib", 0) < j.get("memory_mib", 0):
                        errs.append(f"{jid}: メモリ {j.get('memory_mib')}MiB がグループ {g} のサイズ {grp['size']} を超える")
                    if j.get("when_jst") and j["enabled"]:
                        # グループの tick のどこかで when_jst が成立しうること
                        if not _can_match(grp["cron_jst"], j["when_jst"]):
                            errs.append(f"{jid}: when_jst {j['when_jst']} がグループ {g} の tick で一度も成立しない")
            if not j.get("steps"):
                errs.append(f"{jid}: steps が空")
            for st in j.get("steps") or []:
                if not st.get("argv") or st["argv"][0] not in ("-m",):
                    errs.append(f"{jid}: step の argv は ['-m', module, ...] の形")
            for m in j.get("models") or []:
                if m not in models:
                    errs.append(f"{jid}: 未定義のモデル {m}")
            from src.ops import gcs_mirror as _gm

            outs = [o["path"] for o in j.get("state_out") or []] + list(j.get("mirror_out") or [])
            for pat in j.get("require_out") or []:
                if not any(pat == o or pat.startswith(o.rstrip("/") + "/") for o in outs):
                    errs.append(f"{jid}: require_out の {pat} が state_out / mirror_out の配下ではない（作っても上げない・確認できない）")
            for t in j.get("mirror_out") or []:
                if not _gm.tree_known(t):
                    errs.append(f"{jid}: mirror_out の {t} は gcs_mirror の PUSH_TREES の配下ではない（画面が読めない）")
            tot = sum(int(s.get("max_mb", 0)) for s in j.get("state_in") or [])
            if tot > int(rt.get("state_total_budget_mb", 1500)):
                errs.append(f"{jid}: state_in の合計 {tot}MB が予算を超える")
            if j.get("memory_mib", 0) > run_jobs.get("l", {}).get("memory_mib", 4096):
                errs.append(f"{jid}: メモリが最大サイズを超える")
            if j.get("memory_mib", 0) and tot and tot + 600 > j["memory_mib"] and j.get("enabled"):
                errs.append(f"{jid}: state_in {tot}MB + 実行メモリ 600MB が memory_mib {j['memory_mib']} を超える（コンテナのディスクはメモリ上）")
        if dest == "vps":
            ft = j.get("fetch_task") or {}
            if not ft.get("task"):
                errs.append(f"{jid}: vps には fetch_task.task が必要")
            elif known_fetch_tasks is not None:
                for t in [ft["task"], *(j.get("extra_tasks") or [])]:
                    if t not in known_fetch_tasks and not j.get("blockers"):
                        errs.append(f"{jid}: タスク {t} が VPS 実行役の許可リストに無い")
            if agent_tasks is not None and j.get("ready"):
                for t in [ft.get("task"), *(j.get("extra_tasks") or [])]:
                    if t and t not in agent_tasks:
                        errs.append(f"{jid}: ready=true なのにタスク {t} が実行役の既定の担当（DEFAULT_AGENT_TASKS）に無い")
            if not j.get("schedule_jst") and not j.get("reserved_by"):
                errs.append(f"{jid}: schedule_jst か reserved_by が必要")
            if j.get("schedule_jst"):
                try:
                    parse_cron(j["schedule_jst"])
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{jid}: schedule_jst が不正: {e}")
    for gid, g in groups.items():
        for jid in g["jobs"]:
            j = next((x for x in doc["jobs"] if x["id"] == jid), None)
            if j is None or j.get("group") != gid:
                errs.append(f"{gid}: jobs の {jid} が定義と食い違う")
    return errs


def _can_match(group_cron: str, when_cron: str) -> bool:
    """グループの tick のうち when_cron に一致するものが、1 週間のどこかにあるか。"""
    start = datetime(2026, 1, 5, 0, 0, tzinfo=JST)    # 月曜
    for t in occurrences(group_cron, start, start + timedelta(days=7)):
        if cron_matches(when_cron, t):
            return True
    return False


# ───────────────────────── CLI（準備状況の更新） ─────────────────────────

def _save(doc: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _load_cached.cache_clear()


def main(argv: list[str] | None = None) -> int:
    """``validate`` / ``list`` / ``show <id>`` / ``ready <id> --note …`` / ``block <id> --reason …``。

    ``ready`` は『blockers を確認して解消した』ときに使う（ready=true・blockers=[]・確認日とメモを残す）。
    Scheduler への登録（scheduler_gen）とディスパッチャの実行は、ready=true のジョブだけが対象になる。
    """
    import argparse
    import sys
    from datetime import date

    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("job_id")
    r = sub.add_parser("ready")
    r.add_argument("job_id")
    r.add_argument("--note", required=True, help="何を確認したか（例: ホストで state_sync plan を実行し 380MB で上限内）")
    b = sub.add_parser("block")
    b.add_argument("job_id")
    b.add_argument("--reason", required=True)
    ns = ap.parse_args(argv)
    path = DEFAULT_PATH
    doc = json.loads(path.read_text(encoding="utf-8"))
    if ns.cmd == "validate":
        from src.ops import fetch_task as FT

        errs = validate(doc, known_fetch_tasks=set(FT.ALLOWED_TASKS), agent_tasks=set(FT.DEFAULT_AGENT_TASKS))
        print("\n".join(errs) if errs else "OK")
        return 1 if errs else 0
    if ns.cmd == "list":
        for j in doc["jobs"]:
            print(f"{j['destination']:6s} {'ready' if j['ready'] else 'WAIT ':5s} {j['id']}")
        return 0
    j = next((x for x in doc["jobs"] if x["id"] == ns.job_id), None)
    if j is None:
        print(f"未知のジョブ: {ns.job_id}", file=sys.stderr)
        return 2
    if ns.cmd == "show":
        print(json.dumps(j, ensure_ascii=False, indent=1))
        return 0
    if ns.cmd == "ready":
        j["blockers"], j["ready"] = [], True
        j["ready_note"], j["ready_confirmed_at"] = ns.note, date.today().isoformat()
    else:
        j["blockers"], j["ready"] = list(j.get("blockers") or []) + [ns.reason], False
    _save(doc, path)
    errs = validate(doc)
    print("\n".join(errs) if errs else f"{ns.job_id}: ready={j['ready']}")
    return 1 if errs else 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
