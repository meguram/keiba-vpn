#!/usr/bin/env python3
"""定期実行ジョブの一覧のページ（docs/html/operations/final/scheduled_jobs.html）を作る。

正本は ``config/scheduled_jobs.json``（GCP・ConoHa VPS・ホストの定期実行の一覧）。GCP の Scheduler のグループの中身
（ジョブごとの説明・ステップ・GCS との受け渡し）は ``config/gcp_job_map.json`` の ``scheduler_groups`` と ``jobs`` から引く。
ホストの cron の一覧は別のページ（``docs/html/operations/cron_schedule.html``、``build_cron_schedule_html.py``）。

生成結果に現在時刻は入れない（日付は JSON の ``updated``）。同じ入力からは毎回同じ HTML になるので、
``--check`` で置いてあるファイルとの食い違い（手で直した・JSON を直して作り直していない）を見つけられる。

  python -m src.scripts.docs.build_scheduled_jobs_html
  python -m src.scripts.docs.build_scheduled_jobs_html --check
  python -m src.scripts.docs.build_scheduled_jobs_html --out /tmp/scheduled_jobs.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
JOBS_PATH = REPO_ROOT / "config" / "scheduled_jobs.json"
JOB_MAP_PATH = REPO_ROOT / "config" / "gcp_job_map.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "html" / "operations" / "final" / "scheduled_jobs.html"
TITLE = "定期実行ジョブの一覧（GCP・ConoHa VPS）"

# 曜日（cron は 0=日〜6=土、7 も日）。並べる順は土日月火…（開催日の土日月が先に来る）。
DOW = ["日", "月", "火", "水", "木", "金", "土"]
WHERE_LABEL = {"gcp": "GCP", "vps": "ConoHa VPS", "host": "ホスト"}
KIND_LABEL = {
    "timer": "systemd タイマー",
    "daemon": "常駐（systemd サービス）",
    "scheduler": "Cloud Scheduler → Cloud Run Jobs",
    "run_job": "Cloud Run Job（手動）",
    "cron": "ホストの cron",
}
# 状態 → (表示名, 色のクラス)
STATUS_LABEL = {
    "active": ("稼働中", "ok"),
    "paused": ("停止中", "warn"),
    "planned": ("今後", "mu"),
    "retired": ("削除済み", "mu"),
}
WEIGHT_LABEL = {"high": "重い（止まると困る）", "med": "中", "low": "軽い"}

CSS = """
:root {
  --bg:#f7f8fb; --panel:#ffffff; --panel2:#eef1f7; --border:#d5dbe7; --text:#1b2333; --muted:#5b6578;
  --accent:#2f5fc4; --ok:#1d7f45; --warn:#9a6200; --bad:#b4232c; --code:#eef1f7;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60; --text:#e6ecff; --muted:#9aa7c7;
    --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --code:#0b1428;
  }
}
:root[data-theme="dark"] {
  --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60; --text:#e6ecff; --muted:#9aa7c7;
  --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --code:#0b1428;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.7 "Noto Sans JP",system-ui,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:28px 16px 80px; }
a { color:var(--accent); }
h1 { font-size:24px; margin:0 0 6px; line-height:1.4; }
h2 { font-size:19px; margin:40px 0 6px; padding-top:16px; border-top:1px solid var(--border); }
h3 { font-size:16px; margin:22px 0 4px; }
p, li { overflow-wrap:anywhere; }
.muted { color:var(--muted); font-size:13px; }
.sub { color:var(--muted); font-size:13.5px; margin:0 0 8px; }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; margin:8px 0 14px; border:1px solid var(--border); border-radius:8px; }
table { border-collapse:collapse; width:100%; min-width:760px; font-size:13.5px; background:var(--panel); }
th, td { border-bottom:1px solid var(--border); border-right:1px solid var(--border); padding:6px 10px; text-align:left; vertical-align:top; }
th { background:var(--panel2); color:var(--muted); font-weight:600; white-space:nowrap; }
tr:last-child td { border-bottom:none; }
code { font-family:"JetBrains Mono",ui-monospace,Menlo,monospace; background:var(--code); padding:1px 5px; border-radius:5px; font-size:12px; overflow-wrap:anywhere; }
pre { background:var(--code); border:1px solid var(--border); border-radius:8px; padding:10px 12px; overflow-x:auto; font-size:12px; }
pre code { padding:0; background:none; overflow-wrap:normal; }
.box { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px 16px; margin:12px 0; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:14px 0; }
.kpi { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px 12px; }
.kpi b { display:block; font-size:24px; line-height:1.2; }
.kpi span { color:var(--muted); font-size:12.5px; }
.when { white-space:nowrap; font-weight:600; }
.b { display:inline-block; font-size:11.5px; padding:0 8px; border-radius:10px; border:1px solid currentColor; white-space:nowrap; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .bad { color:var(--bad); } .mu { color:var(--muted); }
.ev { color:var(--muted); font-size:12.5px; }
details { background:var(--panel); border:1px solid var(--border); border-radius:10px; margin:8px 0; }
summary { cursor:pointer; padding:8px 12px; line-height:1.5; overflow-wrap:anywhere; }
summary .sm { color:var(--muted); font-size:13px; }
details[open] > summary { border-bottom:1px solid var(--border); }
.dbody { padding:8px 14px 12px; font-size:13.5px; }
.dbody > div, .kv > div { margin:4px 0; overflow-wrap:anywhere; }
ul.jobs { padding-left:18px; margin:6px 0; } ul.jobs > li { margin:10px 0; }
.kv { margin:4px 0 0; padding-left:10px; border-left:3px solid var(--border); }
.kv ol, .dbody ol, .dbody ul { margin:4px 0; padding-left:20px; }
.dbody .scroll table { min-width:720px; }
""".strip("\n")


# ───────────────────────── 小道具 ─────────────────────────

def esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def code(s: Any) -> str:
    return f"<code>{esc(s)}</code>"


def text_br(s: Any) -> str:
    """改行を <br> にして文を出す（JSON の文は素のテキストなので、すべてエスケープする）。"""
    return "<br>".join(esc(x) for x in str(s or "").split("\n"))


def or_dash(s: Any) -> str:
    return text_br(s) if s not in (None, "", []) else "—"


def badge(status: str) -> str:
    label, cls = STATUS_LABEL.get(status, (status, "mu"))
    return f'<span class="b {cls}">{esc(label)}</span>'


def basis_html(basis: Any) -> str:
    items = basis if isinstance(basis, list) else ([basis] if basis else [])
    return "、".join(code(b) for b in items) if items else "—"


def table(head: list[str], rows: list[str]) -> str:
    """表は必ず横スクロールの div で包む（スマホで横にはみ出さない）。"""
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    return '<div class="scroll"><table>\n<tr>' + th + "</tr>\n" + "\n".join(rows) + "\n</table></div>"


# ───────────────────────── スケジュールの言葉 ─────────────────────────

def _ints(spec: str, lo: int, hi: int) -> list[int] | None:
    """cron の 1 項目を数の並びにする（*・a-b・a,b・*/n・a-b/n）。読めなければ None。"""
    vals: set[int] = set()
    try:
        for part in spec.split(","):
            step = 1
            if "/" in part:
                part, s = part.split("/", 1)
                step = int(s)
            if part == "*":
                a, b = lo, hi
            elif "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
            else:
                a = b = int(part)
            if step <= 0 or a < lo or b > hi or a > b:
                return None
            vals.update(range(a, b + 1, step))
    except ValueError:
        return None
    return sorted(vals)


def _dow_text(spec: str) -> str | None:
    if spec == "*":
        return ""
    days = _ints(spec, 0, 7)
    if days is None:
        return None
    uniq = sorted({d % 7 for d in days}, key=lambda d: (d + 1) % 7)  # 土日月火…の順
    return "".join(DOW[d] for d in uniq)


def cron_text(expr: str) -> str:
    """JST の cron 式を「毎日 07:30」「土日月 08:00〜16:59 毎分」のような言葉にする。読めない形はそのまま返す。"""
    f = expr.split()
    if len(f) != 5 or f[2] != "*" or f[3] != "*":
        return expr
    mi_s, hr_s, _, _, dw_s = f
    dow = _dow_text(dw_s)
    mins = _ints(mi_s, 0, 59)
    hours = _ints(hr_s, 0, 23)
    if dow is None or mins is None or hours is None:
        return expr
    prefix = dow if dow else "毎日"
    step_min = None
    if mi_s == "*":
        step_min = 1
    elif mi_s.startswith("*/"):
        step_min = int(mi_s[2:])
    if hr_s == "*":
        if step_min is not None:
            body = "毎分" if step_min == 1 else f"{step_min} 分おき"
            return body if not dow else f"{dow} {body}"
        body = "毎時 " + "・".join(f"{m:02d}" for m in mins) + " 分"
        return body if not dow else f"{dow} {body}"
    if step_min is not None:
        span = f"{hours[0]:02d}:{mins[0]:02d}〜{hours[-1]:02d}:{mins[-1]:02d}"
        body = "毎分" if step_min == 1 else f"{step_min} 分おき"
        if hours != list(range(hours[0], hours[-1] + 1)):
            return expr
        return f"{prefix} {span} {body}"
    times = [f"{h:02d}:{m:02d}" for h in hours for m in mins]
    return f"{prefix} " + "・".join(times)


def schedule_text(sch: dict[str, Any] | None, kind: str = "") -> tuple[str, str]:
    """(言葉, cron 式) を返す。cron 式が無いものは空文字。"""
    if not sch:
        return ("常駐" if kind == "daemon" else "手動だけ（定期の起動なし）"), ""
    if "cron" in sch:
        return cron_text(sch["cron"]), sch["cron"]
    n = sch.get("every_min")
    if n == 1:
        return "60 秒おき", ""
    if isinstance(n, (int, float)):
        return f"{n:g} 分おき", ""
    return json.dumps(sch, ensure_ascii=False), ""


def schedule_cell(j: dict[str, Any]) -> str:
    words, cron = schedule_text(j.get("schedule"), j.get("kind", ""))
    return f'<td class="when">{esc(words)}' + (f"<br>{code(cron)}" if cron and cron != words else "") + "</td>"


# ───────────────────────── 名前 ─────────────────────────

NAME_KEYS = ("scheduler", "timer", "service", "run_job", "cron_tag", "job_id")


def unit_name(j: dict[str, Any]) -> str:
    """表の見出しに使う名前（Scheduler 名・タイマー名・サービス名など）。"""
    ref = j.get("ref") or {}
    for k in ("scheduler", "timer", "service", "run_job", "cron_tag"):
        if ref.get(k):
            return ref[k]
    return j["id"]


def name_cell(j: dict[str, Any], *, with_badge: bool = False) -> str:
    ref = j.get("ref") or {}
    parts = [code(unit_name(j))]
    if ref.get("timer") and ref.get("service"):
        parts.append("→ " + code(ref["service"]))
    if ref.get("cron_tag") and ref.get("job_id"):
        parts.append("job_id " + code(ref["job_id"]))
    if with_badge:
        parts.append(badge(j.get("status", "")))
    parts.append(f'<span class="ev">id: {esc(j["id"])}</span>')
    return "<td>" + "<br>".join(parts) + "</td>"


# ───────────────────────── GCP のグループ ─────────────────────────

class JobMap:
    def __init__(self, m: dict[str, Any]):
        self.groups: dict[str, Any] = m.get("scheduler_groups") or {}
        self.jobs: dict[str, Any] = {x["id"]: x for x in m.get("jobs") or [] if x.get("id")}
        rt = m.get("runtime") or {}
        self.run_jobs: dict[str, Any] = rt.get("run_jobs") or {}
        self.state_prefix: str = rt.get("state_prefix") or "state"

    def size_text(self, gid: str, *, with_name: bool = True) -> str:
        g = self.groups.get(gid) or {}
        rj = self.run_jobs.get(g.get("size") or "") or {}
        if not rj:
            return ""
        t = (f"{rj.get('name')}" if with_name else "") + f"（{rj.get('memory')}・{rj.get('cpu')} vCPU"
        if g.get("timeout_sec"):
            t += f"、上限 {g['timeout_sec']} 秒"
        return t + "）"

    def job_names(self, gid: str) -> list[str]:
        return list((self.groups.get(gid) or {}).get("jobs") or [])


def _paths(items: list[Any], prefix: str) -> str:
    out = []
    for it in items or []:
        p = it.get("path") if isinstance(it, dict) else it
        if p:
            out.append(code(f"{prefix}/{p}" if prefix else p))
    return ", ".join(out) if out else "—"


def gcp_group_jobs_html(jm: JobMap, gid: str) -> str:
    """グループの中のジョブ（説明・ステップ・GCS との受け渡し・enabled/ready）を並べる。"""
    names = jm.job_names(gid)
    if not names:
        return f"<div>グループ {code(gid)} は <code>config/gcp_job_map.json</code> に無い。</div>"
    lis = []
    for name in names:
        x = jm.jobs.get(name)
        if x is None:
            lis.append(f"<li><b>{code(name)}</b> — <span class=\"warn\">gcp_job_map.json の jobs に無い</span></li>")
            continue
        flags = []
        if x.get("enabled") is False:
            flags.append('<span class="b mu">enabled=false</span>')
        if x.get("ready") is False:
            flags.append('<span class="b warn">ready=false</span>')
        head = f"<li><b>{code(name)}</b> {' '.join(flags)} — {esc(x.get('desc'))}"
        kv = ['<div class="kv">']
        steps = x.get("steps") or []
        if steps:
            kv.append("ステップ:<ol>")
            for st in steps:
                argv = " ".join(str(a) for a in (st.get("argv") or []))
                tail = '（失敗しても続ける）' if st.get("allow_fail") else ""
                kv.append(f"<li>{code('python ' + argv)}{esc(tail)}</li>")
            kv.append("</ol>")
        else:
            kv.append("<div>ステップ: —</div>")
        read = _paths(x.get("state_in") or [], jm.state_prefix)
        if x.get("mirror_in"):
            read += "、ミラー " + _paths(x["mirror_in"], "")
        write = _paths(x.get("state_out") or [], jm.state_prefix)
        if x.get("mirror_out"):
            write += "、画面用のミラー " + _paths(x["mirror_out"], "")
        kv.append(f"<div>読む（GCS → コンテナ）: {read}</div>")
        kv.append(f"<div>書く（コンテナ → GCS）: {write}</div>")
        kv.append(f"<div class=\"ev\">enabled={esc(str(x.get('enabled')).lower())}・ready={esc(str(x.get('ready')).lower())}</div>")
        kv.append("</div>")
        lis.append(head + "".join(kv) + "</li>")
    return '<ul class="jobs">' + "".join(lis) + "</ul>"


def gcp_group_summary(jm: JobMap, gid: str) -> str:
    """表の「中身」の列: グループの各ジョブを ① ② で短く並べる。"""
    nums = "①②③④⑤⑥⑦⑧⑨⑩"
    out = []
    for i, name in enumerate(jm.job_names(gid)):
        x = jm.jobs.get(name) or {}
        mark = nums[i] if i < len(nums) else f"({i + 1})"
        off = "（無効）" if x.get("enabled") is False else ""
        out.append(f"{mark} {code(name)}{esc(off)}: {esc(x.get('desc') or '')}")
    return "<br>".join(out)


# ───────────────────────── 節 ─────────────────────────

def sel(jobs: list[dict[str, Any]], *, where: str, status: set[str] | None = None, kind: set[str] | None = None,
        not_kind: set[str] | None = None) -> list[dict[str, Any]]:
    return [j for j in jobs if j.get("where") == where
            and (status is None or j.get("status") in status)
            and (kind is None or j.get("kind") in kind)
            and (not_kind is None or j.get("kind") not in not_kind)]


def section_summary(d: dict[str, Any], jobs: list[dict[str, Any]]) -> str:
    gcp_live = sel(jobs, where="gcp", status={"active"})
    vps_live = sel(jobs, where="vps", status={"active"})
    host_live = sel(jobs, where="host", status={"active"})
    idle = [j for j in jobs if j.get("status") != "active"]
    retired = d.get("retired") or []

    def names(js: list[dict[str, Any]]) -> str:
        return "・".join(code(unit_name(j)) for j in js) or "なし"

    def breakdown(js: list[dict[str, Any]]) -> str:
        parts = []
        for w in ("gcp", "vps", "host"):
            for st in ("retired", "paused", "planned"):
                n = sum(1 for j in js if j.get("where") == w and j.get("status") == st)
                if n:
                    parts.append(f"{WHERE_LABEL[w]} {STATUS_LABEL[st][0]} {n}")
        return "、".join(parts)

    kpis = [
        (f"{len(gcp_live)} 件", f"GCP で動いている定期実行（{names(gcp_live)}）"),
        (f"{len(vps_live)} 件", f"ConoHa VPS で動いている systemd タイマー（{names(vps_live)}）"),
        (f"{len(idle)} 件", f"今は動いていない・今後起動しうるもの（{esc(breakdown(idle))}）"),
        (f"{len(retired)} 種", "廃止（" + ("・".join(esc(r.get("name") or r.get("id")) for r in retired) or "なし") + "）"),
    ]
    if host_live:
        kpis.insert(2, (f"{len(host_live)} 件", f"ホストで動いている（この一覧に載せたもの: {names(host_live)}）"))
    out = ['<h2 id="summary">まとめ</h2>', '<div class="kpis">']
    out += [f'  <div class="kpi"><b>{esc(n)}</b><span>{t}</span></div>' for n, t in kpis]
    out.append("</div>")
    if d.get("about"):
        out.append(f'<div class="box"><p><b>この一覧について</b>: {text_br(d["about"])}</p></div>')
    out.append('<p class="sub">ジョブごとの中身（ステップ・入口のコマンド・読む元と書き出し先・取得先の URL の例）は '
               '<a href="#detail">ジョブごとの中身</a> の節で、行を開くと出る。</p>')
    srcs = d.get("sources") or []
    if srcs:
        out.append("<h3>データの出どころ</h3>")
        out.append("<ul>" + "".join(f"<li>{text_br(s)}</li>" for s in srcs) + "</ul>")
    return "\n".join(out)


def section_gcp_live(jobs: list[dict[str, Any]], jm: JobMap) -> str:
    out = ['<h2 id="gcp-live">表 1. GCP: 稼働中の定期実行</h2>',
           '<p class="sub">Cloud Scheduler → Cloud Run Jobs（<code>keiba-dispatch-*</code>）。場所 asia-northeast1。'
           '中身は <code>config/gcp_job_map.json</code> の <code>scheduler_groups</code> から。</p>']
    rows = []
    for j in sel(jobs, where="gcp", status={"active"}, kind={"scheduler"}):
        ref = j.get("ref") or {}
        gid = ref.get("group") or ""
        target = f"Cloud Run Job {code(ref.get('run_job'))}"
        target += esc(jm.size_text(gid, with_name=False))
        if gid:
            target += f"<br>グループ {code(gid)}"
        body = text_br(j.get("what"))
        inner = gcp_group_summary(jm, gid) if gid else ""
        if inner:
            body += "<br>" + inner
        if j.get("note"):
            body += f"<br><span class=\"ev\">{text_br(j['note'])}</span>"
        rows.append(f"<tr>{name_cell(j, with_badge=True)}{schedule_cell(j)}<td>{target}</td><td>{body}</td>"
                    f"<td>{or_dash(j.get('cost'))}</td><td>{or_dash(j.get('host_equivalent'))}</td>"
                    f"<td class=\"ev\">{basis_html(j.get('basis'))}</td></tr>")
    if rows:
        out.append(table(["名前", "スケジュール（JST）", "対象", "中身", "費用の目安（月）", "ホストでも同じ処理？", "根拠"], rows))
    else:
        out.append('<p class="muted">なし。</p>')

    others = [j for j in sel(jobs, where="gcp") if j.get("kind") == "run_job" or j.get("status") == "paused"]
    if others:
        out.append("<h3>定期ではなく、呼ばれたときだけ動くもの（GCP）</h3>")
        out.append('<p class="sub">この一覧（<code>config/scheduled_jobs.json</code>）に載せたものだけ。Cloud Run のサービス（画面の API・監視ポータル）は定期実行ではないので載せていない。</p>')
        rows = [f"<tr>{name_cell(j, with_badge=True)}<td>{esc(KIND_LABEL.get(j.get('kind'), j.get('kind')))}</td>"
                f"<td>{text_br(j.get('what'))}<br><span class=\"ev\">{or_dash(j.get('note'))}</span></td>"
                f"<td>{or_dash(j.get('cost'))}</td><td class=\"ev\">{basis_html(j.get('basis'))}</td></tr>" for j in others]
        out.append(table(["名前", "種類", "中身・今の呼ばれ方", "費用の目安", "根拠"], rows))
    return "\n".join(out)


def _vps_row(j: dict[str, Any], *, with_note_col: bool) -> str:
    body = text_br(j.get("what"))
    if not with_note_col and j.get("note"):
        body += f"<br><span class=\"ev\">{text_br(j['note'])}</span>"
    cells = [name_cell(j, with_badge=True), schedule_cell(j), f"<td>{body}</td>",
             f"<td>{or_dash(j.get('memory'))}</td>"]
    if with_note_col:
        cells.append(f"<td>{or_dash(j.get('note'))}</td>")
    cells += [f"<td>{or_dash(j.get('cost'))}</td>", f"<td>{or_dash(j.get('host_equivalent'))}</td>",
              f"<td class=\"ev\">{basis_html(j.get('basis'))}</td>"]
    return "<tr>" + "".join(cells) + "</tr>"


def section_vps_live(jobs: list[dict[str, Any]]) -> str:
    out = ['<h2 id="vps-live">表 2. ConoHa VPS: 稼働中の定期実行</h2>',
           '<p class="sub">稼働中の systemd タイマーと、取得の実行役などの常駐（止めているものも状態を付けてここに載せる）。'
           'OS 標準のタイマー（apt-daily・logrotate など）は数えていない。タイマーの時刻は JST。</p>']
    js = sel(jobs, where="vps", status={"active"}) + [j for j in sel(jobs, where="vps", kind={"daemon"}) if j.get("status") != "active"]
    rows = [_vps_row(j, with_note_col=False) for j in js]
    if rows:
        out.append(table(["タイマー（起動するもの）", "スケジュール（JST）", "中身", "メモリの枠", "費用の目安（月）", "ホストでも同じ処理？", "根拠"], rows))
    else:
        out.append('<p class="muted">なし。</p>')
    return "\n".join(out)


def section_future(d: dict[str, Any], jobs: list[dict[str, Any]], jm: JobMap) -> str:
    idle = {"planned", "paused", "retired"}
    out = ['<h2 id="future">表 3. 今は動いていない・今後起動する</h2>']

    gcp = sel(jobs, where="gcp", status=idle, kind={"scheduler"})
    if gcp:
        out.append(f"<h3>GCP: 動いていない Scheduler（{len(gcp)} 件）</h3>")
        out.append('<p class="sub">削除済みの Scheduler を作り直すと、1 件 約 15 円/月と Jobs の実行の費用が戻る。作り直しの設定とコマンドは根拠の docs。</p>')
        rows = []
        for j in gcp:
            ref = j.get("ref") or {}
            gid = ref.get("group") or ""
            target = code(ref.get("run_job")) + (f"<br>グループ {code(gid)}" if gid else "")
            inner = gcp_group_summary(jm, gid) if gid else ""
            body = text_br(j.get("what")) + (f"<br>{inner}" if inner else "")
            rows.append(f"<tr>{name_cell(j, with_badge=True)}{schedule_cell(j)}<td>{target}</td><td>{body}</td>"
                        f"<td>{or_dash(j.get('host_equivalent'))}</td><td>{or_dash(j.get('note'))}</td>"
                        f"<td class=\"ev\">{basis_html(j.get('basis'))}</td></tr>")
        out.append(table(["名前", "スケジュール（JST）", "対象", "中身（グループの処理）", "ホストで代わりに動いているか", "メモ", "根拠"], rows))

    vps = sel(jobs, where="vps", status=idle, not_kind={"daemon"})
    if vps:
        out.append("<h3>ConoHa VPS: 今後起動するもの・止めているもの</h3>")
        rows = [_vps_row(j, with_note_col=True) for j in vps]
        out.append(table(["タイマー", "スケジュール（JST）", "中身", "メモリの枠", "いつ有効になるか・メモ", "費用の目安（月）", "ホストでも同じ処理？", "根拠"], rows))

    host = sel(jobs, where="host", status=idle)
    if host:
        out.append("<h3>ホスト: この一覧で見張る予定のもの</h3>")
        out.append('<p class="sub">ホストの cron の全体は <a href="../cron_schedule.html">cron の一覧</a>（<code>docs/html/operations/cron_schedule.html</code>）。ここには、この一覧で見張るものだけを載せる。</p>')
        rows = [f"<tr>{name_cell(j, with_badge=True)}{schedule_cell(j)}<td>{text_br(j.get('what'))}</td>"
                f"<td>{or_dash(j.get('note'))}</td><td>{or_dash(j.get('cost'))}</td>"
                f"<td class=\"ev\">{basis_html(j.get('basis'))}</td></tr>" for j in host]
        out.append(table(["名前", "スケジュール（JST）", "中身", "いつ有効になるか・メモ", "費用の目安（月）", "根拠"], rows))

    retired = d.get("retired") or []
    out.append("<h3>廃止したもの</h3>")
    if retired:
        rows = [f"<tr><td>{esc(r.get('name') or r.get('id'))}<br><span class=\"ev\">id: {esc(r.get('id'))}</span></td>"
                f"<td>{esc(WHERE_LABEL.get(r.get('where'), r.get('where')))}</td><td>{or_dash(r.get('was'))}</td>"
                f"<td>{esc(r.get('retired_on') or '')}。{text_br(r.get('why'))}</td>"
                f"<td class=\"ev\">{basis_html(r.get('basis'))}</td></tr>" for r in retired]
        out.append(table(["名前", "場所", "以前のスケジュール", "廃止の日・理由", "根拠"], rows))
    else:
        out.append('<p class="muted">なし。</p>')
    return "\n".join(out)


def _list_html(items: Any) -> str:
    items = items if isinstance(items, list) else ([items] if items else [])
    if not items:
        return "—"
    if len(items) == 1:
        return text_br(items[0])
    return "<ul>" + "".join(f"<li>{text_br(x)}</li>" for x in items) + "</ul>"


def detail_item(j: dict[str, Any], jm: JobMap) -> str:
    words, cron = schedule_text(j.get("schedule"), j.get("kind", ""))
    meta = [cron + "（JST）" if cron else words]
    if j.get("memory"):
        meta.append(str(j["memory"]))
    ref = j.get("ref") or {}
    gid = ref.get("group") or ""
    if gid and jm.size_text(gid):
        meta = [meta[0], jm.size_text(gid)]
    summary = (f'<summary>{badge(j.get("status", ""))} <b>{esc(unit_name(j))}</b> '
               f'<span class="mu">— {esc("・".join(meta))}</span>'
               f'<br><span class="sm">{text_br(j.get("what"))}</span></summary>')
    rows = [
        ("id", code(j["id"]) + f'（{esc(WHERE_LABEL.get(j.get("where"), j.get("where")))}・'
                               f'{esc(KIND_LABEL.get(j.get("kind"), j.get("kind")))}・重さ {esc(WEIGHT_LABEL.get(j.get("weight"), j.get("weight")))}）'),
        ("名前", "・".join(f"{esc(k)} {code(ref[k])}" for k in NAME_KEYS + ("group",) if ref.get(k)) or "—"),
        ("スケジュール", esc(words) + (f"（{code(cron)}）" if cron and cron != words else "")),
        ("入口", code(j["command"]) if j.get("command") else "—"),
        ("取得先の URL の例", code(j["sample_url"]) if j.get("sample_url") else "なし"),
        ("読む", _list_html(j.get("inputs"))),
        ("書く", _list_html(j.get("outputs"))),
        ("メモリの枠", or_dash(j.get("memory"))),
        ("費用の目安（月）", or_dash(j.get("cost"))),
        ("ホストでも同じ処理？", or_dash(j.get("host_equivalent"))),
        ("根拠", f'<span class="ev">{basis_html(j.get("basis"))}</span>'),
        ("メモ", or_dash(j.get("note"))),
    ]
    body = ['<div class="dbody">'] + [f"<div><b>{esc(k)}</b>: {v}</div>" for k, v in rows]
    if gid:
        body.append(f"<div><b>グループの中のジョブ</b>（<code>config/gcp_job_map.json</code> の "
                    f"<code>scheduler_groups[{esc(gid)}].jobs</code>）:</div>")
        body.append(gcp_group_jobs_html(jm, gid))
    body.append("</div>")
    return f'<details id="job-{esc(j["id"])}">{summary}\n' + "\n".join(body) + "</details>"


def _detail_order(js: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {"active": 0, "paused": 1, "planned": 2, "retired": 3}
    return sorted(js, key=lambda j: rank.get(j.get("status"), 9))  # 安定ソート（同じ状態の中は JSON の順）


def section_detail(jobs: list[dict[str, Any]], jm: JobMap) -> str:
    out = ['<h2 id="detail">ジョブごとの中身（行を開くと詳細）</h2>',
           '<p class="sub">各行を開くと、入口のコマンド・読む元と書き出し先・取得先の URL の例・費用・根拠が出る。'
           'URL は<b>形の例</b>（レース <code>202605050811</code> など）で、実際には叩いていない。</p>',
           '<p class="sub">GCP のグループの共通: Scheduler が <code>keiba-dispatch-&lt;大きさ&gt;</code> の Job を '
           '<code>args=[-m, src.ops.job_dispatch, group, &lt;グループ&gt;]</code> で起動し、コンテナの中で '
           '<code>python -m src.ops.job_dispatch group &lt;グループ&gt;</code> がジョブを順に動かす（<code>src/ops/job_dispatch.py</code>）。'
           '「読む・書く」の <code>state/…</code> は GCS の <code>state/</code> の下の写しで、実行のたびにコンテナへ取り込み・書き戻す'
           '（<code>src/ops/state_sync.py</code>）。</p>']
    titles = [("gcp", "GCP（Cloud Scheduler → Cloud Run Jobs）"), ("vps", "ConoHa VPS（systemd のタイマー・常駐）"),
              ("host", "ホスト（cron）")]
    for w, title in titles:
        js = _detail_order(sel(jobs, where=w))
        if not js:
            continue
        out.append(f"<h3>{esc(title)}</h3>")
        out += [detail_item(j, jm) for j in js]
    return "\n".join(out)


def _sev_class(sev: str) -> str:
    if "済" in sev:
        return "ok"
    if "高" in sev:
        return "bad"
    if "中" in sev:
        return "warn"
    return "mu"


def section_audit(d: dict[str, Any]) -> str:
    audit = d.get("audit") or []
    out = ['<h2 id="audit">設計上の点検</h2>']
    if d.get("audit_note"):
        out.append(f'<p class="sub">{text_br(d["audit_note"])}重さは 高・中・低。</p>')
    has_status = any(a.get("status") for a in audit)
    head = ["#", "重さ", "種類", "指摘", "根拠", "推奨", "費用への影響"] + (["状態"] if has_status else [])
    rows = []
    for a in audit:
        sev = str(a.get("severity") or "")
        cells = [f"<td>{esc(a.get('no'))}</td>", f'<td class="{_sev_class(sev)}">{esc(sev)}</td>',
                 f"<td>{esc(a.get('kind'))}</td>", f"<td>{text_br(a.get('finding'))}</td>",
                 f"<td class=\"ev\">{text_br(a.get('basis'))}</td>", f"<td>{text_br(a.get('advice'))}</td>",
                 f"<td>{or_dash(a.get('cost'))}</td>"]
        if has_status:
            cells.append(f"<td>{or_dash(a.get('status'))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    out.append(table(head, rows) if rows else '<p class="muted">なし。</p>')
    if d.get("audit_ok"):
        out.append(f'<p class="muted">{text_br(d["audit_ok"])}</p>')
    return "\n".join(out)


# ───────────────────────── 全体 ─────────────────────────

def render(d: dict[str, Any], job_map: dict[str, Any]) -> str:
    jobs = list(d.get("jobs") or [])
    jm = JobMap(job_map)
    updated = str(d.get("updated") or "")
    tz = d.get("tz") or "Asia/Tokyo"
    head = [
        "<!doctype html>",
        '<html lang="ja">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(TITLE)}</title>",
        "<style>",
        CSS,
        "</style>",
        "</head>",
        "<body><main>",
        f"<h1>{esc(TITLE)}</h1>",
        f'<p class="muted">更新: {esc(updated)}（一覧の <code>updated</code>） / 時刻はすべて <b>JST</b>（{esc(tz)}） / '
        'このページは <code>config/scheduled_jobs.json</code>（＋<code>config/gcp_job_map.json</code>）から '
        '<code>python -m src.scripts.docs.build_scheduled_jobs_html</code> で作る（手で直さない）。'
        'ホストの cron の一覧は <a href="../cron_schedule.html">cron の一覧</a>（<code>docs/html/operations/cron_schedule.html</code>）。</p>',
    ]
    body = [
        section_summary(d, jobs),
        section_gcp_live(jobs, jm),
        section_vps_live(jobs),
        section_future(d, jobs, jm),
        section_detail(jobs, jm),
        section_audit(d),
    ]
    return "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n</main></body>\n</html>\n"


def build(jobs_path: Path = JOBS_PATH, job_map_path: Path = JOB_MAP_PATH) -> str:
    d = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
    m = json.loads(Path(job_map_path).read_text(encoding="utf-8"))
    return render(d, m)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="config/scheduled_jobs.json から定期実行ジョブの一覧のページを作る")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="書き出し先（--check のときは比べる相手）")
    ap.add_argument("--check", action="store_true", help="書かずに比べるだけ。一致しなければ終了コード 1")
    ap.add_argument("--jobs", type=Path, default=JOBS_PATH, help=argparse.SUPPRESS)
    ap.add_argument("--job-map", type=Path, default=JOB_MAP_PATH, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    text = build(args.jobs, args.job_map)
    if args.check:
        try:
            cur = args.out.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"{args.out}: ファイルが無い（python -m src.scripts.docs.build_scheduled_jobs_html で作る）")
            return 1
        if cur != text:
            print(f"{args.out}: 生成結果と一致しない（config/scheduled_jobs.json から作り直す。手で直さない）")
            return 1
        print(f"{args.out}: 一致")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"{args.out}: {len(text.encode('utf-8')):,} バイト")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
