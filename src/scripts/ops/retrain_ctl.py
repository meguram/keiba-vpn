"""再学習の進み具合を記録し、次にやることを返す小さな管理 CLI（標準ライブラリだけ。ホストにそのまま置いて動く）。

手順の定義は config/retrain_plan.json、状態は RETRAIN_STATE_DIR/state.json（既定: /home/jovyan/work/retrain_state、無ければ data/local/retrain_state）。
このツールは処理を**実行しない**。次の段・コマンド・完了の確認を返し、段の開始・完了・失敗・止まりを記録するだけ。

    python3 retrain_ctl.py status            # 全体の表
    python3 retrain_ctl.py next              # 次にやる段（コマンド・完了の確認つき）
    python3 retrain_ctl.py decide scope models --confirm
    python3 retrain_ctl.py approve s12 --by "ユーザーの発言の要約"
    python3 retrain_ctl.py start s9 / done s9 --note "..." / fail s9 --note "..." / block s9 --reason "..." / skip s5 --reason "..." / reset s9
    python3 retrain_ctl.py log "メモ"
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DONE_LIKE = ("done", "skipped")
MARK = {"pending": "・", "running": "▶", "done": "✔", "failed": "✘", "blocked": "■", "skipped": "－"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def state_dir() -> Path:
    env = os.environ.get("RETRAIN_STATE_DIR")
    if env:
        return Path(env)
    host = Path("/home/jovyan/work/retrain_state")
    if host.parent.exists():
        return host
    return HERE.parents[2] / "data" / "local" / "retrain_state"


def plan_path() -> Path:
    env = os.environ.get("RETRAIN_PLAN")
    if env:
        return Path(env)
    for c in (HERE / "retrain_plan.json", state_dir() / "retrain_plan.json", HERE.parents[2] / "config" / "retrain_plan.json"):
        if c.exists():
            return c
    sys.exit("手順の定義 retrain_plan.json が見つからない")


def load_plan() -> dict:
    return json.loads(plan_path().read_text(encoding="utf-8"))


class Store:
    def __init__(self) -> None:
        self.dir = state_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "state.json"
        self.lock = open(self.dir / ".lock", "w")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.plan = load_plan()
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"version": 1, "created": now(), "decisions": {}, "steps": {}, "history": []}
        for s in self.plan["steps"]:
            self.state["steps"].setdefault(s["id"], {"status": "pending", "attempts": 0})
        for k, v in self.plan["decisions"].items():
            self.state["decisions"].setdefault(k, {"value": v["default"], "confirmed": False})
        self.apply_scope()

    def step(self, sid: str) -> dict:
        for s in self.plan["steps"]:
            if s["id"] == sid:
                return s
        sys.exit(f"段 {sid} は無い")

    def decision(self, key: str) -> str:
        return self.state["decisions"][key]["value"]

    def active(self, s: dict) -> bool:
        if self.decision("scope") not in s["scopes"]:
            return False
        return not (s.get("optional") and self.decision("with_optional") != "yes")

    def apply_scope(self) -> None:
        for s in self.plan["steps"]:
            st = self.state["steps"][s["id"]]
            if not self.active(s) and st["status"] == "pending":
                st.update(status="skipped", reason="scope", note=f"scope={self.decision('scope')} / with_optional={self.decision('with_optional')} の対象外")
            elif self.active(s) and st["status"] == "skipped" and st.get("reason") == "scope":
                st.update(status="pending")
                st.pop("reason", None)

    def event(self, sid: str, kind: str, note: str = "") -> None:
        self.state["history"].append({"at": now(), "step": sid, "event": kind, "note": note})

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def blockers(self, s: dict) -> list[str]:
        return [d for d in s["deps"] if self.state["steps"][d]["status"] not in DONE_LIKE]


def cmd_status(a, st: Store) -> None:
    dec = ", ".join(f"{k}={v['value']}{'' if v['confirmed'] else '(既定・未確認)'}" for k, v in st.state["decisions"].items())
    print(f"決定: {dec}")
    print(f"状態ファイル: {st.path}")
    for s in st.plan["steps"]:
        x = st.state["steps"][s["id"]]
        extra = f"  [{x['status']}]" + (f" {x.get('note', '')}" if x.get("note") else "")
        gate = " ★要OK" if s.get("gate") == "user_ok" and not x.get("approved") and x["status"] not in DONE_LIKE else ""
        print(f"{MARK[x['status']]} {s['id']:>3} {s['title']}{gate}{extra if x['status'] != 'pending' else ''}")
    n = sum(1 for s in st.plan["steps"] if st.state["steps"][s["id"]]["status"] in DONE_LIKE)
    print(f"進み: {n}/{len(st.plan['steps'])}")


def cmd_next(a, st: Store) -> None:
    unconfirmed = [k for k, v in st.state["decisions"].items() if not v["confirmed"] and k in ("scope", "train_years")]
    if unconfirmed:
        print("決定が未確認:", ", ".join(f"{k}={st.decision(k)}" for k in unconfirmed), "→ 既定のまま進めてよいか利用者に一言伝え、`decide <key> <値> --confirm` で確定する。")
        if st.decision("train_years") != "2020-2025":
            print("train_years が 2020-2025 以外: コード固定の TRAINING_YEARS・YEARS・FIT_END の修正と位置取り予想の代替の一致確認が要る。ここで止まって利用者に確認する。")
            return
    running = [s for s in st.plan["steps"] if st.state["steps"][s["id"]]["status"] == "running"]
    if running:
        print("実行中の段あり:", ", ".join(s["id"] for s in running), "→ ログを見て、終わっていれば done / 失敗なら fail を記録する。")
        return
    failed = [s for s in st.plan["steps"] if st.state["steps"][s["id"]]["status"] in ("failed", "blocked")]
    if failed:
        print("失敗・止まりの段あり:", ", ".join(f"{s['id']}({st.state['steps'][s['id']]['status']})" for s in failed), "→ 原因を調べて reset して再実行するか、利用者に判断を仰ぐ。解消するまで次へ進まない。")
        return
    for s in st.plan["steps"]:
        x = st.state["steps"][s["id"]]
        if x["status"] != "pending":
            continue
        bl = st.blockers(s)
        if bl:
            continue
        print(f"次: {s['id']}（段 {s['no']}）{s['title']}")
        print("コマンド:")
        for c in s["commands"]:
            print("  -", c)
        print("完了の確認:", s["done_check"])
        if s.get("heavy"):
            print("重い処理: /tmp/heavy_run.sh <ラベル> <コマンド> 経由・直列・6 コア空け・load<コア数。ログは state_dir/logs/<段>.log")
        for k in ("notes", "once_note"):
            if s.get(k):
                print("注意:", s[k])
        if s.get("gate") == "user_ok" and not x.get("approved"):
            print("★この段は実行前に利用者の OK が要る。OK を得たら `approve " + s["id"] + " --by \"発言の要約\"` を記録してから start。まだなら、ここまでを報告して待つ。")
        else:
            print("→ `start " + s["id"] + "` を記録してから実行する。")
        return
    if all(st.state["steps"][s["id"]]["status"] in DONE_LIKE for s in st.plan["steps"]):
        print("全段が完了。docs/html/operations/model_versions.html の採用一覧と履歴を更新し、利用者に報告する。")
    else:
        print("進められる段が無い（依存の段が未完了）。")


def need(st: Store, sid: str) -> tuple[dict, dict]:
    return st.step(sid), st.state["steps"][sid]


def cmd_start(a, st: Store) -> None:
    s, x = need(st, a.step)
    bl = st.blockers(s)
    if bl:
        sys.exit(f"依存の段が未完了: {', '.join(bl)}")
    if any(v["status"] == "running" for k, v in st.state["steps"].items() if k != a.step):
        sys.exit("別の段が実行中（直列の決まり）。終わってから始める")
    if s.get("gate") == "user_ok" and not x.get("approved"):
        sys.exit("この段は利用者の OK が要る。approve を先に記録する")
    x.update(status="running", started=now(), attempts=x.get("attempts", 0) + 1)
    st.event(a.step, "start", a.note)
    st.save()
    print(f"{a.step} 開始を記録")


def cmd_mark(kind: str, status: str):
    def run(a, st: Store) -> None:
        s, x = need(st, a.step)
        note = getattr(a, "note", None) or getattr(a, "reason", None) or ""
        if kind == "done" and x["status"] != "running":
            sys.exit(f"{a.step} は実行中ではない（status={x['status']}）")
        x.update(status=status, finished=now(), note=note)
        x.pop("reason", None) if kind != "skip" else x.update(reason="manual")
        st.event(a.step, kind, note)
        st.save()
        print(f"{a.step} → {status}")
    return run


def cmd_reset(a, st: Store) -> None:
    s, x = need(st, a.step)
    x.update(status="pending", note="")
    st.event(a.step, "reset", a.note or "")
    st.save()
    print(f"{a.step} を pending に戻した")


def cmd_approve(a, st: Store) -> None:
    s, x = need(st, a.step)
    x.update(approved=a.by, approved_at=now())
    st.event(a.step, "approve", a.by)
    st.save()
    print(f"{a.step} の実行の OK を記録: {a.by}")


def cmd_decide(a, st: Store) -> None:
    spec = st.plan["decisions"].get(a.key)
    if not spec:
        sys.exit(f"決定項目 {a.key} は無い。{list(st.plan['decisions'])}")
    if a.value not in spec["choices"]:
        sys.exit(f"{a.key} の選択肢: {spec['choices']}")
    st.state["decisions"][a.key] = {"value": a.value, "confirmed": bool(a.confirm), "at": now()}
    st.apply_scope()
    st.event("-", "decide", f"{a.key}={a.value} confirmed={bool(a.confirm)}")
    st.save()
    print(f"{a.key}={a.value}（{'確定' if a.confirm else '仮'}）")


def cmd_log(a, st: Store) -> None:
    st.event("-", "log", a.text)
    st.save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(f=cmd_status)
    sub.add_parser("next").set_defaults(f=cmd_next)
    for name in ("start",):
        p = sub.add_parser(name); p.add_argument("step"); p.add_argument("--note", default=""); p.set_defaults(f=cmd_start)
    for name, status in (("done", "done"), ("fail", "failed")):
        p = sub.add_parser(name); p.add_argument("step"); p.add_argument("--note", default=""); p.set_defaults(f=cmd_mark(name, status))
    for name, status in (("block", "blocked"), ("skip", "skipped")):
        p = sub.add_parser(name); p.add_argument("step"); p.add_argument("--reason", default=""); p.set_defaults(f=cmd_mark(name, status))
    p = sub.add_parser("reset"); p.add_argument("step"); p.add_argument("--note", default=""); p.set_defaults(f=cmd_reset)
    p = sub.add_parser("approve"); p.add_argument("step"); p.add_argument("--by", required=True); p.set_defaults(f=cmd_approve)
    p = sub.add_parser("decide"); p.add_argument("key"); p.add_argument("value"); p.add_argument("--confirm", action="store_true"); p.set_defaults(f=cmd_decide)
    p = sub.add_parser("log"); p.add_argument("text"); p.set_defaults(f=cmd_log)
    a = ap.parse_args()
    st = Store()
    a.f(a, st)
    if a.cmd in ("status", "next"):
        st.save()


if __name__ == "__main__":
    main()
