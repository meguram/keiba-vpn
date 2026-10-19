"""誕生月日を 1 頭ずつ手入力するローカルツール（127.0.0.1 だけで待ち受け。外部サイトへは一切アクセスしない）。

    python -m src.research.pedigree.manual_entry_server --ancestors /tmp/ancestor_birthdays_for_subplan.csv \
        --zone data/meta/dormello/zone_review.csv --nk /tmp/nk_birth_results_local.jsonl --dir data/meta/dormello --port 8765

ブラウザで http://127.0.0.1:8765/ を開く。「ページを開く」で調べ先を別タブに開き、日付を入力して「保存して次へ」を押すと、
その場で birth_manual.csv と birth_search_log.csv に書き込まれる（取り消しも可能）。載っていないときは「記載なし」を押すと、再度は出てこない。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import secrets
import subprocess
import threading
import urllib.parse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.research.pedigree import manual_checklist as mc
from src.research.pedigree.birth_subplans import LOG_COLUMNS, MANUAL_COLUMNS, upsert_log

MON = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
NONE_NOTE = "ユーザー確認: 記載なし"
OK_NOTE = "ユーザーがブラウザで目視確認"


def parse_date_input(s: str) -> tuple[int, int, int] | None:
    s = (s or "").strip()
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s) or re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日?", s)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    else:
        m = re.fullmatch(r"(\d{1,2})[-\s]([A-Za-z]{3})[a-z]*[-\s,]*(\d{4})", s)
        if m and m[2].lower() in MON:
            y, mo, d = int(m[3]), MON[m[2].lower()], int(m[1])
        else:
            m = re.fullmatch(r"([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s*(\d{4})", s)
            if not (m and m[1].lower() in MON):
                return None
            y, mo, d = int(m[3]), MON[m[1].lower()], int(m[2])
    try:
        date(y, mo, d)
    except ValueError:
        return None
    return y, mo, d


class Store:
    def __init__(self, ancestors: Path, zone: Path, nk: Path | None, d: Path, base_url: str, limit: int = 3000, commit_every: int = 0):
        self.dir = d
        self.commit_every = commit_every          # 0 なら自動コミットしない
        self.pending_msgs: list[str] = []
        self.git_status = ""
        self.manual_p, self.log_p = d / "birth_manual.csv", d / "birth_search_log.csv"
        self.urls_p = d / "sporthorse_urls.csv"      # 実際のページ名が分かった馬の URL（horse_id,url）
        self.url_overrides = {r[0]: r[1] for r in self._read_csv(self.urls_p)[1:] if len(r) > 1} if self.urls_p.exists() else {}
        anc = {r["horse_id"]: r for r in csv.DictReader(open(ancestors, encoding="utf-8"))}
        zone_rows = list(csv.DictReader(open(zone, encoding="utf-8")))
        nk_map: dict[str, dict] = {}
        if nk and nk.exists():
            for line in open(nk, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    nk_map[r["horse_id"]] = r
        self.rows = mc.build_rows(anc, zone_rows, nk_map, set(), limit, base_url)
        self.lock = threading.Lock()
        self.skipped: list[str] = []
        self.undo_stack: list[dict] = []
        self.session_saved = self.session_none = 0
        if not self.manual_p.exists():
            self._write_csv(self.manual_p, [MANUAL_COLUMNS])
        if commit_every:  # 前回の未コミット分が残っていれば、次のコミットに含める
            try:
                dirty = self._git("status", "--porcelain", "--", str(self.manual_p.resolve()), str(self.log_p.resolve())).stdout.strip()
            except Exception:  # noqa: BLE001
                dirty = ""
            if dirty:
                self.pending_msgs.append("前回セッションの未コミット分")

    # --- csv helpers
    @staticmethod
    def _read_csv(p: Path) -> list[list[str]]:
        return list(csv.reader(open(p, encoding="utf-8"))) if p.exists() else []

    @staticmethod
    def _write_csv(p: Path, rows: list[list]) -> None:
        with open(p, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)

    def _resolved(self) -> set[str]:
        done = {r[0] for r in self._read_csv(self.manual_p)[1:] if r}
        done |= {r[0] for r in self._read_csv(self.log_p)[1:] if len(r) > 8 and r[8].startswith(NONE_NOTE)}
        return done

    def _row(self, hid: str) -> dict | None:
        return next((r for r in self.rows if r["horse_id"] == hid), None)

    # --- git（永久保存）
    def _git(self, *args: str, timeout: int = 90) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=self.dir, capture_output=True, text=True, timeout=timeout)

    def _commit_and_push(self) -> str:
        files = [str(self.manual_p.resolve()), str(self.log_p.resolve())] + ([str(self.urls_p.resolve())] if self.urls_p.exists() else [])
        try:
            top = self._git("rev-parse", "--show-toplevel").stdout.strip()
            if not top:
                return "git リポジトリではないためコミットできません"
            self._git("pull", "-q", "--ff-only", "origin", "HEAD")  # 失敗しても続ける（他の変更と衝突しない限り先に進む）
            self._git("add", "--", *files)
            if not self._git("diff", "--cached", "--quiet", "--", *files).returncode:
                return "コミットする変更はありません"
            lines = self.pending_msgs[-10:]
            msg = ("feat(research): 手入力ツールで誕生月日を " + f"{len(self.pending_msgs)} 件記録する\n\n" + "\n".join(f"- {m}" for m in lines)
                   + "\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\n")
            c = self._git("commit", "-q", "-m", msg, "--", *files)
            if c.returncode:
                return "コミット失敗: " + (c.stderr or c.stdout).strip()[:160]
            sha = self._git("rev-parse", "--short", "HEAD").stdout.strip()
            pu = self._git("push", "-q", "origin", "HEAD")
            if pu.returncode:  # リモートが進んでいたら rebase して再試行
                self._git("pull", "-q", "--rebase", "--autostash", "origin", "HEAD")
                pu = self._git("push", "-q", "origin", "HEAD")
            if pu.returncode:
                return f"コミット {sha} は作成済み。プッシュに失敗: " + (pu.stderr or pu.stdout).strip()[:160]
            self.pending_msgs = []
            return f"コミット {sha} を GitHub へプッシュしました"
        except Exception as e:  # noqa: BLE001
            return f"自動コミットでエラー: {e!r}"[:200]

    def _after_change(self, msg: str | None, force: bool = False) -> None:
        if msg:
            self.pending_msgs.append(msg)
        if self.commit_every and (force or len(self.pending_msgs) >= self.commit_every):
            self.git_status = self._commit_and_push()

    def commit_now(self) -> dict:
        with self.lock:
            if not self.commit_every:
                return {"ok": False, "error": "自動コミットが無効です（--commit-every を指定してください）"}
            self.git_status = self._commit_and_push()
            return {"ok": True, "status": self.git_status}

    # --- operations
    def pending(self) -> list[dict]:
        done = self._resolved()
        todo = [r for r in self.rows if r["horse_id"] not in done]
        first = [r for r in todo if r["horse_id"] not in self.skipped]
        later = [r for r in todo if r["horse_id"] in self.skipped]
        return first + later

    def _with_override(self, item: dict | None) -> dict | None:
        if item and item["horse_id"] in self.url_overrides:
            u = self.url_overrides[item["horse_id"]]
            return {**item, "url": u, "candidates": [u] + [c for c in item.get("candidates", []) if c != u]}
        return item

    def remember_url(self, horse_id: str, url: str) -> None:
        if not url or self.url_overrides.get(horse_id) == url:
            return
        self.url_overrides[horse_id] = url
        self._write_csv(self.urls_p, [["horse_id", "url"], *[[h, u] for h, u in self.url_overrides.items()]])

    def next(self) -> dict:
        with self.lock:
            p = self.pending()
            return {"item": self._with_override(p[0]) if p else None, "remaining": len(p), "saved": self.session_saved, "none": self.session_none,
                    "unsaved": len(self.pending_msgs), "git": self.git_status, "auto_commit": self.commit_every,
                    "recent": [{"horse_id": u["horse_id"], "name": u["name"], "text": u["text"]} for u in self.undo_stack[-5:]][::-1]}

    def save(self, horse_id: str, text: str, confirm_jan1: bool = False, url: str | None = None) -> dict:
        with self.lock:
            r = self._row(horse_id)
            if not r:
                return {"ok": False, "error": "対象の馬が見つかりません"}
            d = parse_date_input(text)
            if not d:
                return {"ok": False, "error": "日付の形式が読めません（例: 1970-01-30 / 30-Jan-1970 / Jan 30, 1970）"}
            if d[0] != r["year"]:
                return {"ok": False, "error": f"生年が違います（{r['year']} のはず）。違う馬のページかもしれません"}
            if d[1] == 1 and d[2] == 1 and not confirm_jan1:
                return {"ok": False, "need_confirm": True, "error": "1 月 1 日は日付不明の仮置きの可能性があります。実際に 1 月 1 日なら確認して保存してください"}
            src = url or self._with_override(r)["url"]
            if url:
                self.remember_url(horse_id, url)
            note = OK_NOTE + ("（1月1日を確認のうえ保存）" if d[1] == 1 and d[2] == 1 else "")
            prev_log = next((x for x in self._read_csv(self.log_p)[1:] if x and x[0] == horse_id), None)
            with open(self.manual_p, "a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([horse_id, r["name"], d[0], d[1], d[2], src, note])
            upsert_log(self.log_p, [[horse_id, r["name"], "adopted", d[0], d[1], d[2], src, date.today().isoformat(), note]])
            self.undo_stack.append({"kind": "save", "horse_id": horse_id, "name": r["name"], "text": f"{d[0]}-{d[1]:02d}-{d[2]:02d}", "prev_log": prev_log})
            self.session_saved += 1
            self._after_change(f"{r['name']}: {d[0]}-{d[1]:02d}-{d[2]:02d}")
            return {"ok": True}

    def _current_matches(self, url: str, title: str) -> dict | None:
        """ユーザースクリプトが送ってきたページが、いま表示中の馬のものか（URL の末尾の名前か、タイトルの馬名で判定）。"""
        p = self.pending()
        if not p:
            return None
        cur = p[0]
        seg = urllib.parse.urlparse(url).path.rstrip("/").rsplit("/", 1)[-1].lower()
        slugs = [c.rsplit("/", 1)[-1].lower() for c in cur.get("candidates", [cur["url"]])]
        norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())  # noqa: E731
        if seg in slugs or (cur["name_en"] and norm(cur["name_en"]) and norm(cur["name_en"]) in norm(title)):
            return cur
        return None

    def capture(self, url: str, title: str, text: str) -> dict:
        cur = None
        with self.lock:
            cur = self._current_matches(url, title)
        if not cur:
            return {"ok": False, "error": "いま表示中の馬と違うページです。ツールの画面の馬のページを開いてください"}
        return self.save(cur["horse_id"], text, confirm_jan1=False, url=url)

    def capture_none(self, url: str, title: str) -> dict:
        with self.lock:
            cur = self._current_matches(url, title)
        if not cur:
            return {"ok": False, "error": "いま表示中の馬と違うページです"}
        return self.none(cur["horse_id"])

    def none(self, horse_id: str) -> dict:
        with self.lock:
            r = self._row(horse_id)
            if not r:
                return {"ok": False, "error": "対象の馬が見つかりません"}
            prev_log = next((x for x in self._read_csv(self.log_p)[1:] if x and x[0] == horse_id), None)
            upsert_log(self.log_p, [[horse_id, r["name"], "no_data", r["year"], "", "", r["url"], date.today().isoformat(), NONE_NOTE]])
            self.undo_stack.append({"kind": "none", "horse_id": horse_id, "name": r["name"], "text": "記載なし", "prev_log": prev_log})
            self.session_none += 1
            self._after_change(f"{r['name']}: 記載なし")
            return {"ok": True}

    def skip(self, horse_id: str) -> dict:
        with self.lock:
            if horse_id in self.skipped:
                self.skipped.remove(horse_id)
            self.skipped.append(horse_id)
            return {"ok": True}

    def undo(self) -> dict:
        with self.lock:
            if not self.undo_stack:
                return {"ok": False, "error": "取り消せる操作がありません"}
            u = self.undo_stack.pop()
            hid = u["horse_id"]
            if u["kind"] == "save":
                rows = self._read_csv(self.manual_p)
                idx = max(i for i, r in enumerate(rows) if r and r[0] == hid)
                del rows[idx]
                self._write_csv(self.manual_p, rows)
                self.session_saved -= 1
            else:
                self.session_none -= 1
            log = [r for r in self._read_csv(self.log_p) if not (r and r[0] == hid)]
            if u["prev_log"]:
                log.append(u["prev_log"])
            self._write_csv(self.log_p, log if log else [LOG_COLUMNS])
            if self.pending_msgs:
                self.pending_msgs.pop()
            return {"ok": True, "name": u["name"]}


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>誕生日 手入力</title>
<style>body{font-family:sans-serif;max-width:760px;margin:24px auto;padding:0 12px}.big{font-size:1.6em;font-weight:bold}.muted{color:#666}
button,a.btn{display:inline-block;margin:4px 6px 4px 0;padding:8px 14px;font-size:1em;border:1px solid #888;border-radius:6px;background:#f4f4f4;text-decoration:none;color:#000;cursor:pointer}
button.primary{background:#2d7ff9;color:#fff;border-color:#2d7ff9}input{font-size:1.2em;padding:6px;width:220px}.err{color:#b00020}.ok{color:#0a7d2c}
.card{border:1px solid #ccc;border-radius:8px;padding:16px;margin:12px 0}</style></head><body>
<h1>誕生日 手入力</h1><p class="muted" id="stat"></p><div id="main" class="card"></div>
<h3>直近の操作</h3><ul id="recent"></ul>
<script>
const TOKEN="__TOKEN__";let cur=null;
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json','X-Token':TOKEN},body:body?JSON.stringify(body):undefined});return r.json()}
function q(s){return encodeURIComponent(s)}
async function load(){const d=await api('/api/next');document.getElementById('stat').innerHTML=`残り ${d.remaining} 頭 ／ このセッションで保存 ${d.saved} 頭・記載なし ${d.none} 頭<br>`+(d.auto_commit?`永久保存（コミット）まで あと ${Math.max(d.auto_commit-d.unsaved,0)} 件 ／ 未コミット ${d.unsaved} 件 <button onclick="commitNow()">今すぐコミット</button> <span class="muted">${d.git||''}</span>`:'自動コミット: 無効');
 document.getElementById('recent').innerHTML=d.recent.map(r=>`<li>${r.name}: ${r.text}</li>`).join('')||'<li class="muted">なし</li>';
 const m=document.getElementById('main');cur=d.item;
 if(!cur){m.innerHTML='<p class="ok big">すべて処理しました。</p>';return}
 const en=cur.name_en||cur.name;
 m.innerHTML=`<div class="big">${cur.name}</div><div class="muted">${cur.name_en||''} ／ 生年 ${cur.year} ／ 影響 ${cur.impact}</div>
 <p><a class="btn" href="${cur.url}" target="_blank" rel="noopener" onclick="document.getElementById('d').focus()">① ページを開く（SporthorseData）</a>
 <a class="btn" href="https://www.jbis.jp/navi_search/?keyword=${q(en)}" target="_blank" rel="noopener">JBIS で探す</a>
 <a class="btn" href="https://en.wikipedia.org/w/index.php?search=${q(en+' horse')}" target="_blank" rel="noopener">Wikipedia で探す</a></p>
 <p class="muted">開けないときの別候補: ${(cur.candidates||[]).slice(1,9).map((u,i)=>`<a href="${u}" target="_blank" rel="noopener">${u.split('/').pop()}</a>`).join(' ／ ')}</p>
 <p>② 日付を入力 <input id="d" placeholder="例 30-Jan-1970" autofocus> <span id="msg"></span></p>
 <p class="muted">実際のページの URL が違っていたら貼ってください（次回からそのリンクを使います）: <input id="u" style="width:380px" placeholder="https://sporthorse-data.com/pedigree/..."></p>
 <p><button class="primary" onclick="save(false)">③ 保存して次へ</button><button onclick="none_()">日付が載っていない（記載なし）</button><button onclick="skip_()">あとで</button><button onclick="undo_()">直前を取り消す</button></p>`;
 document.getElementById('d').addEventListener('keydown',e=>{if(e.key==='Enter')save(false)})}
async function save(c){const v=document.getElementById('d').value;const r=await api('/api/save',{horse_id:cur.horse_id,text:v,confirm_jan1:c,url:document.getElementById('u').value.trim()});
 if(r.ok){load()}else if(r.need_confirm){if(confirm(r.error))save(true)}else{const m=document.getElementById('msg');m.className='err';m.textContent=r.error}}
async function none_(){await api('/api/none',{horse_id:cur.horse_id});load()}
async function skip_(){await api('/api/skip',{horse_id:cur.horse_id});load()}
async function commitNow(){const r=await api('/api/commit',{});if(!r.ok)alert(r.error);load()}
async function undo_(){const r=await api('/api/undo',{});if(!r.ok)alert(r.error);load()}
load();
setInterval(async()=>{if(!cur)return;const d=await api('/api/next');if(d.item&&d.item.horse_id!==cur.horse_id&&!document.getElementById('d').value){load()}else if(!d.item){load()}},3000);
</script></body></html>"""


def make_handler(store: Store, token: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 静かに
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _authorized(self) -> bool:
            return self.headers.get("X-Token") == token

        def do_GET(self):  # noqa: N802
            path = urllib.parse.urlparse(self.path).path
            if path == "/":
                b = PAGE.replace("__TOKEN__", token).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif path == "/api/next" and self._authorized():
                self._json(store.next())
            elif path == "/api/token":  # ユーザースクリプト用（ブラウザの他サイトからは読めない）
                self._json({"token": token})
            else:
                self._json({"ok": False, "error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            if not self._authorized():
                return self._json({"ok": False, "error": "unauthorized"}, 403)
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/save":
                self._json(store.save(body.get("horse_id", ""), body.get("text", ""), bool(body.get("confirm_jan1")), (body.get("url") or "").strip() or None))
            elif path == "/api/none":
                self._json(store.none(body.get("horse_id", "")))
            elif path == "/api/commit":
                self._json(store.commit_now())
            elif path == "/api/skip":
                self._json(store.skip(body.get("horse_id", "")))
            elif path == "/api/undo":
                self._json(store.undo())
            elif path == "/api/ext_capture":
                self._json(store.capture(body.get("url", ""), body.get("title", ""), body.get("text", "")))
            elif path == "/api/ext_none":
                self._json(store.capture_none(body.get("url", ""), body.get("title", "")))
            else:
                self._json({"ok": False, "error": "not found"}, 404)

    return H


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ancestors", required=True, type=Path)
    ap.add_argument("--zone", required=True, type=Path)
    ap.add_argument("--nk", type=Path)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--base-url", default="https://sporthorse-data.com/pedigree")
    ap.add_argument("--commit-every", type=int, default=5, help="この件数ごとに birth_manual.csv と birth_search_log.csv だけをコミットして GitHub へプッシュする（0 で無効）")
    a = ap.parse_args(argv)
    store = Store(a.ancestors, a.zone, a.nk, a.dir, a.base_url, commit_every=a.commit_every)
    token = secrets.token_urlsafe(16)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(store, token))
    print(f"対象 {len(store.pending())} 頭。ブラウザで http://127.0.0.1:{a.port}/ を開いてください（Ctrl+C で終了）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if a.commit_every and store.pending_msgs:
            print("終了前に未コミットの分を保存します:", store.commit_now().get("status"), flush=True)


if __name__ == "__main__":
    main()
