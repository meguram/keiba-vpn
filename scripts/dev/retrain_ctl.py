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


# ── retrain_order.html のチャートに状態の印を付ける ─────────────────────────
# チャート（Mermaid）の箱 → その箱を作る段。印は箱の文字の末尾に足し、実行中は橙・失敗/止めは赤の枠にする。
# 書き換えるのは目印（RETRAIN-STATUS / %% STATUS）で囲んだ範囲だけ。手で書いた部分には触らない。
NODE_STEPS = {"DATA": ["s0"], "POS": ["s3"], "EASE": ["s3"], "LAP": ["s4"], "S3": ["s7"], "MEGU": ["s8"],
              "PARENT": ["s9", "s10", "s11", "s12"], "POST": ["s13", "s14", "s15", "s16", "s17"], "DISP": ["s2", "s5", "s6"]}
# §2.6 のチャート（区分 A/B/C）の箱。箱ごとの記録（state["nodes"]、`node` 命令）があればそれを、無ければ対応する段の状態を使う
NODE_STEPS_26 = {"RAW": ["s0"], "POS": ["s3"], "PACE": ["s4"], "EASE": ["s3"], "S3": ["s7"], "AS": ["s8"],
                 "PARENT": ["s9", "s10", "s11", "s12"], "EXO": ["s16"], "POST": ["s13", "s14", "s15", "s16"], "FWD": ["s17"]}
PLANNED_26: set[str] = set()  # 親モデルへの上乗せを評価する予定の箱
_BADGE = {"running": ("▶ 実行中", "#fb923c"), "failed": ("✘ 失敗", "#f87171"), "blocked": ("■ 判断待ち", "#f87171"),
          "done": ("✔ 完了", "#4ade80"), "skipped": ("－ 対象外", "#9aa7c7"), "pending": ("・ 未着手", "#9aa7c7"), "partial": ("◐ 途中", "#fbbf24")}


def order_html_path() -> Path:
    env = os.environ.get("RETRAIN_ORDER_HTML")
    if env:
        return Path(env)
    repo = Path(os.environ.get("KEIBA_REPO") or "/home/jovyan/work/keiba-vpn")
    for c in (repo / "docs/html/operations/retrain_order.html", HERE.parents[1] / "docs/html/operations/retrain_order.html"):
        if c.exists():
            return c
    return repo / "docs/html/operations/retrain_order.html"


def node_status(steps: list[str], state: dict) -> tuple[str, list[str]]:
    """箱の状態（running / failed / blocked / done / skipped / partial / pending）と、実行中・失敗の段。"""
    sts = {sid: state["steps"].get(sid, {}).get("status", "pending") for sid in steps}
    for k in ("running", "failed", "blocked"):
        hit = [sid for sid, v in sts.items() if v == k]
        if hit:
            return k, hit
    if all(v == "skipped" for v in sts.values()):
        return "skipped", []
    if all(v in DONE_LIKE for v in sts.values()):
        return "done", []
    if any(v in DONE_LIKE for v in sts.values()):
        return "partial", [sid for sid, v in sts.items() if v not in DONE_LIKE]
    return "pending", []


def _short(title: str, n: int = 16) -> str:
    """段の名前を図に収まる長さに（[I] などの記号と括弧の補足を外す）。"""
    import re

    t = re.sub(r"\[[A-Z]\]", "", title).split("（")[0].strip()
    return t if len(t) <= n else t[:n] + "…"


def _chart_block(chart: str, nodes: dict, status_of) -> str:
    """Mermaid のチャートの末尾に、箱ごとの状態の印（箱の文字の末尾に足す）と枠の色の行を足す。前回の分は消して書き直す。"""
    import re

    chart = re.sub(r"\n\s*%% STATUS:BEGIN.*?%% STATUS:END", "", chart, flags=re.S)
    lines = ["", "  %% STATUS:BEGIN（scripts/dev/retrain_ctl.py が書く。手で直さない）",
             "  classDef st_running stroke:#fb923c,stroke-width:5px;", "  classDef st_bad stroke:#f87171,stroke-width:5px;"]
    for node in nodes:
        lab = re.search(rf'^\s*{node}\["(.*?)"\]', chart, re.M)
        if not lab:
            continue
        k, text = status_of(node)
        lines.append(f'  {node}["{lab.group(1)}<br/><span>{text}</span>"]')
        if k == "running":
            lines.append(f"  class {node} st_running;")
        elif k in ("failed", "blocked"):
            lines.append(f"  class {node} st_bad;")
    lines.append("  %% STATUS:END")
    return chart.rstrip() + "\n" + "\n".join(lines) + "\n"


def _replace_chart_after(html: str, heading: str, fn) -> str:
    """見出し heading の後ろで最初の <div class="mermaid"> の中身を fn(中身) に置き換える。"""
    import re

    h = html.find(heading)
    if h < 0:
        return html
    m = re.compile(r'<div class="mermaid">(.*?)</div>', re.S).search(html, h)
    if not m:
        return html
    return html[:m.start(1)] + fn(m.group(1)) + html[m.end(1):]


def render_status(plan: dict, state: dict, html: str) -> str:
    import re

    titles = {s["id"]: s["title"] for s in plan["steps"]}

    def from_steps(steps: list[str]) -> tuple[str, str]:
        k, which = node_status(steps, state)
        text = _BADGE[k][0]
        if which and k in ("running", "failed", "blocked"):
            text += "：" + "・".join(f"{sid} {_short(titles.get(sid, ''))}" for sid in which)
        elif which and k == "partial":
            text += f"（残り {'・'.join(which)}）"
        return k, text

    def status_21(node: str) -> tuple[str, str]:
        return from_steps(NODE_STEPS[node])

    def status_26(node: str) -> tuple[str, str]:
        rec = (state.get("nodes") or {}).get(node)
        if rec:
            k = rec["status"]
            text = _BADGE.get(k, ("?", ""))[0] + (f"：{_short(rec.get('note', ''), 22)}" if rec.get("note") else "")
            return k, text
        if node in NODE_STEPS_26:
            return from_steps(NODE_STEPS_26[node])
        if node in PLANNED_26:
            return "pending", "評価予定"
        return "pending", "（対象外の案・未採用）"

    html = _replace_chart_after(html, "<h3>2.1 チャート", lambda c: _chart_block(c, NODE_STEPS, status_21))
    all26 = ["RAW", "POS", "PACE", "ODDS", "EASE", "AS", "STAT", "S3", "PARENT", "EXO", "POST", "FWD"]
    html = _replace_chart_after(html, "2.6 作るものの区分", lambda c: _chart_block(c, all26, status_26))
    # チャートの上の 1 行（いま何をしているか・進み・更新時刻）
    done = sum(1 for v in state["steps"].values() if v["status"] in DONE_LIKE)
    run = [sid for sid, v in state["steps"].items() if v["status"] == "running"]
    runn = [f"{n}（{r.get('note', '')}）" for n, r in (state.get("nodes") or {}).items() if r.get("status") == "running"]
    bad = [sid for sid, v in state["steps"].items() if v["status"] in ("failed", "blocked")]
    nxt = next((s["id"] for s in plan["steps"] if state["steps"][s["id"]]["status"] == "pending"), None)
    parts = []
    if runn:
        parts.append("▶ 作業中（§2.6 の箱）: <b>" + "・".join(runn) + "</b>")
    if run:
        parts.append("▶ 学習中: " + "・".join(f"<b>{sid} {titles.get(sid, '')}</b>（{state['steps'][sid].get('started', '')[:16].replace('T', ' ')} 開始）" for sid in run))
    if bad:
        parts.append("<span class='warn'>✘/■ 要対応: " + "・".join(f"{sid} {titles.get(sid, '')}" for sid in bad) + "</span>")
    if not parts:
        parts.append(f"・ 次: <b>{nxt} {titles.get(nxt, '')}</b>" if nxt else "<span class='ok'>✔ すべて完了</span>")
    yrs = state["decisions"].get("train_years", {}).get("value", "")
    banner = (f'<!-- RETRAIN-STATUS:BEGIN --><div class="box"><b>再学習の状態</b>（{time.strftime("%Y-%m-%d %H:%M")} 更新・学習年 {yrs}・段 {done}/{len(state["steps"])}）: '
              + "<br/>".join(parts) +
              f'<br/><span class="muted">§2.1 と §2.6 のチャートの箱の下に状態（✔ 完了・▶ 実行中・・ 未着手・✘ 失敗・■ 判断待ち・－ 対象外）。'
              f'実行中の箱は橙の太枠。<code>retrain_ctl.py</code> が記録のたびに書き直す（手動は <code>retrain_ctl.py html</code>、§2.6 の箱は <code>retrain_ctl.py node 箱 状態 --note …</code>）。</span></div><!-- RETRAIN-STATUS:END -->')
    html = re.sub(r"<!-- RETRAIN-STATUS:BEGIN -->.*?<!-- RETRAIN-STATUS:END -->\n?", "", html, flags=re.S)
    for head in ("<h3>2.1 チャート", "2.6 作るものの区分"):
        h3 = html.find(head)
        if h3 >= 0:
            end = html.find("</h3>", h3) + len("</h3>")
            html = html[:end] + "\n" + banner + html[end:]
    return html


def update_order_html(plan: dict, state: dict) -> None:
    p = order_html_path()
    if not p.exists():
        return
    old = p.read_text(encoding="utf-8")
    new = render_status(plan, state, old)
    if new != old:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(new, encoding="utf-8")
        os.replace(tmp, p)


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
        try:                                   # 読み物（retrain_order.html）のチャートに状態の印を付ける。失敗しても記録は止めない
            update_order_html(self.plan, self.state)
        except Exception as exc:  # noqa: BLE001
            print(f"（retrain_order.html の状態の印を更新できませんでした: {exc}）", file=sys.stderr)

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


def cmd_node(a, st: Store) -> None:
    nodes = st.state.setdefault("nodes", {})
    nodes[a.node] = {"status": a.status, "note": a.note, "at": now()}
    st.event(f"node:{a.node}", a.status, a.note)
    st.save()
    print(f"{a.node} → {a.status}{('（' + a.note + '）') if a.note else ''}")


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
    p = sub.add_parser("node", help="§2.6 のチャートの箱の状態を記録する（例: node POS running --note '馬柱の代替を作成中'）")
    p.add_argument("node"); p.add_argument("status", choices=["pending", "running", "done", "failed", "blocked", "skipped", "partial"]); p.add_argument("--note", default="")
    p.set_defaults(f=cmd_node)
    sub.add_parser("html", help="retrain_order.html のチャートの状態の印だけを書き直す").set_defaults(f=lambda a, st: print(f"更新: {order_html_path()}"))
    a = ap.parse_args()
    st = Store()
    a.f(a, st)
    if a.cmd in ("status", "next", "html"):
        st.save()


if __name__ == "__main__":
    main()
