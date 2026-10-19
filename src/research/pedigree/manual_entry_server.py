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
        self.detected: dict[str, dict] = {}       # horse_id -> ユーザースクリプトが読み取った日付（保存はまだ）
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

    def batch(self, n: int = 5) -> dict:
        """先頭 n 頭（まとめ表示用）。各頭の検出済みの日付つき。"""
        with self.lock:
            p = self.pending()
            items = [self._with_override(x) for x in p[: max(1, min(int(n), 12))]]
            return {"items": items, "detected": {i["horse_id"]: self.detected[i["horse_id"]] for i in items if i["horse_id"] in self.detected},
                    "remaining": len(p), "saved": self.session_saved, "none": self.session_none, "unsaved": len(self.pending_msgs),
                    "git": self.git_status, "auto_commit": self.commit_every,
                    "recent": [{"horse_id": u["horse_id"], "name": u["name"], "text": u["text"]} for u in self.undo_stack[-5:]][::-1]}

    def set_url(self, horse_id: str, url: str) -> dict:
        """「実際の URL」欄に貼られた URL をその場で記憶する（保存や記載なしを押さなくても残す）。"""
        with self.lock:
            url = (url or "").strip()
            if not self._row(horse_id):
                return {"ok": False, "error": "対象の馬が見つかりません"}
            if not url.startswith("https://sporthorse-data.com/"):
                return {"ok": False, "error": "SporthorseData の URL（https://sporthorse-data.com/...）を入れてください"}
            self.remember_url(horse_id, url)
            return {"ok": True}

    def batch_save(self, rows: list[dict]) -> dict:
        """行ごとに 保存 / 記載なし / あとで を処理する。失敗した行は未処理のまま残り、理由を返す。"""
        results = []
        for r in rows:
            hid = r.get("horse_id", "")
            if r.get("url") and (r.get("none") or not r.get("text")):   # 保存しない行でも、貼られた URL は残す
                self.set_url(hid, r["url"])
            if r.get("none"):
                out = self.none(hid)
            elif r.get("text"):
                out = self.save(hid, r["text"], bool(r.get("confirm_jan1")), (r.get("url") or "").strip() or None)
            else:
                out = self.skip(hid)
            results.append({"horse_id": hid, **out})
        return {"results": results}

    def next(self) -> dict:
        with self.lock:
            p = self.pending()
            det = self.detected.get(p[0]["horse_id"]) if p else None
            return {"item": self._with_override(p[0]) if p else None, "detected": det, "remaining": len(p), "saved": self.session_saved, "none": self.session_none,
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

    WINDOW = 12   # 先頭からこの頭数のうちのどの馬のページかを判定する（まとめ表示で複数同時に開くため）

    def _current_matches(self, url: str, title: str) -> dict | None:
        """ユーザースクリプトが送ってきたページが、表示中の馬（先頭 WINDOW 頭）のどれのものか。URL の末尾の名前か、タイトルの馬名で判定する。"""
        seg = urllib.parse.urlparse(url).path.rstrip("/").rsplit("/", 1)[-1].lower()
        norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())  # noqa: E731
        win = self.pending()[: self.WINDOW]
        for cur in win:
            slugs = [c.rsplit("/", 1)[-1].lower() for c in cur.get("candidates", [cur["url"]])]
            if cur["horse_id"] in self.url_overrides:
                slugs.append(self.url_overrides[cur["horse_id"]].rsplit("/", 1)[-1].lower())
            if seg in slugs:
                return cur
        for cur in win:
            if cur["name_en"] and norm(cur["name_en"]) and norm(cur["name_en"]) in norm(title):
                return cur
        return None

    def detect(self, url: str, title: str, text: str | None, year_only: str | None = None) -> dict:
        """埋め込み表示や別タブのページから読み取った日付を一時的に覚える（保存は画面のボタンで行う）。"""
        with self.lock:
            cur = self._current_matches(url, title)
            if not cur:
                return {"ok": False, "error": "表示中の馬と違うページです"}
            self.detected[cur["horse_id"]] = {"text": text, "year_only": year_only, "url": url}
            return {"ok": True}

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


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>誕生日 手入力（まとめ）</title>
<style>*{box-sizing:border-box}body{font-family:sans-serif;margin:0 auto;max-width:860px;padding:10px 14px}
.muted{color:#666;font-size:.88em}.err{color:#b00020}.ok{color:#0a7d2c}.row{border:1px solid #ccc;border-radius:8px;padding:8px 10px;margin:7px 0}.row.bad{border-color:#b00020;background:#fff5f5}
button,a.btn{display:inline-block;margin:2px 4px 2px 0;padding:6px 10px;font-size:.92em;border:1px solid #888;border-radius:6px;background:#f4f4f4;color:#000;text-decoration:none;cursor:pointer}
button.primary{background:#2d7ff9;color:#fff;border-color:#2d7ff9;font-size:1.05em;padding:9px 16px}input[type=text]{font-size:1em;padding:4px;width:170px}.det{margin:3px 0}
.steps{background:#f7f7ff;border:1px solid #ccd;border-radius:8px;padding:8px 12px}</style></head><body>
<h2 style="margin:2px 0">誕生日 手入力（まとめ表示）</h2><div class="muted" id="stat"></div>
<div class="steps"><b>使い方</b> ① 「この頭数のページを開く」を押す（別タブが開き、次回からは同じタブを使い回します） → ② 各行の「検出」を確認 → ③ 「保存して次へ」
<br><span class="muted">ユーザースクリプト（バージョン 1.2）を入れていれば、ポップアップに邪魔されず背景タブで開きます。サイトが埋め込みを禁止しているため、画面内には表示できません。</span></div>
<div style="margin:8px 0">1 回に扱う頭数 <select id="n" onchange="N=+this.value;poll(true)"><option>1</option><option>3</option><option selected>5</option><option>8</option><option>10</option></select>
<button class="primary" onclick="openAll()">① この頭数のページを開く</button><button onclick="undo_()">直前を取り消す</button> <span id="popmsg" class="err"></span></div>
<div id="rows"></div><div style="margin:10px 0"><button class="primary" onclick="saveBatch()">③ この頭数を保存して次へ</button></div>
<div class="muted">直近の操作</div><ul id="recent" style="margin:4px 0;padding-left:18px"></ul>
<script>
const TOKEN="__TOKEN__";let N=5,batch=[],sig="",rows={};
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json','X-Token':TOKEN},body:body?JSON.stringify(body):undefined});return r.json()}
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const q=s=>encodeURIComponent(s);const $=id=>document.getElementById(id);
function url(it){const c=it.candidates||[it.url];return c[rows[it.horse_id].ci%c.length]}
function openOne(i){const it=batch[i];if(!it)return null;return window.open(url(it),'sh_'+i)}      // 名前つきタブ: 次の頭数でも同じタブを使い回す（noopener は付けない）
function openAll(){
 if(document.documentElement.dataset.dormelloUserscript){window.postMessage({type:'dormello-open-tabs',urls:batch.map(url)},location.origin);$('popmsg').className='ok';$('popmsg').textContent='ユーザースクリプトで '+batch.length+' 頭分を背景タブに開きました（前回のタブは閉じます）';return}
 let blocked=0;batch.forEach((it,i)=>{if(!openOne(i))blocked++});$('popmsg').className='err';
 $('popmsg').textContent=blocked?`${blocked} 件がブロックされました。ユーザースクリプトを最新版（1.2）にすると、ブロックされずに開けます（各行の「開く」でも個別に開けます）`:''}
function cand(i){const it=batch[i];rows[it.horse_id].ci++;$('cu_'+it.horse_id).textContent=url(it).split('/').pop();openOne(i)}
function initBatch(){const L=$('rows');L.innerHTML='';
 if(!batch.length){L.innerHTML='<p class="ok"><b>すべて処理しました。</b></p>';return}
 batch.forEach((it,i)=>{const h=it.horse_id;const st=rows[h]||(rows[h]={ci:0,edited:false,err:''});
  const div=document.createElement('div');div.className='row'+(st.err?' bad':'');
  div.innerHTML=`<div><b>${i+1}. ${esc(it.name)}</b> <span class="muted">${esc(it.name_en||'')} ／ 生年 ${it.year} ／ 影響 ${it.impact}</span></div>
  <div class="det muted" id="det_${h}">検出待ち…</div>
  <div><input type="text" id="in_${h}" placeholder="例 30-Jan-1970" oninput="rows['${h}'].edited=true"> <label><input type="checkbox" id="nn_${h}"> 記載なし</label> <label><input type="checkbox" id="j1_${h}"> 1月1日で確認</label></div>
  <div class="muted"><button onclick="openOne(${i})">開く</button><button onclick="cand(${i})">次の候補で開く</button>候補: <span id="cu_${h}">${esc(url(it).split('/').pop())}</span>
   <a class="btn" href="https://www.jbis.jp/navi_search/?keyword=${q(it.name_en||it.name)}" target="sh_alt" >JBIS</a><a class="btn" href="https://en.wikipedia.org/w/index.php?search=${q((it.name_en||it.name)+' horse')}" target="sh_alt">Wikipedia</a>
   <input type="text" id="ur_${h}" placeholder="実際のURL（違うとき）" style="width:34%" onchange="setUrl('${h}')"></div><div class="err" id="er_${h}">${esc(st.err)}</div>`;
  L.appendChild(div)})}
function updateDet(m){batch.forEach(it=>{const h=it.horse_id,d=m[h],st=rows[h];if(!$('det_'+h))return;
 if(d&&d.text){$('det_'+h).innerHTML=`<span class="ok">検出: <b>${esc(d.text)}</b></span> <span class="muted">${esc((d.url||'').split('/').pop())}</span>`;if(!st.edited&&!$('in_'+h).value)$('in_'+h).value=d.text;$('ur_'+h).dataset.det=d.url||''}
 else if(d&&d.year_only){$('det_'+h).innerHTML=`<span class="err">年のみ（${esc(d.year_only)}）のページ</span>`;if(!st.edited&&!st.nnAuto){$('nn_'+h).checked=true;st.nnAuto=true}}
 else{$('det_'+h).textContent='検出待ち…（ページを開いて、ユーザースクリプトで読み取ります）'}})}
async function poll(force){const d=await api('/api/batch?n='+N);
 $('stat').innerHTML=`残り ${d.remaining} 頭 ／ 保存 ${d.saved}・記載なし ${d.none} ／ `+(d.auto_commit?`コミットまで あと ${Math.max(d.auto_commit-d.unsaved,0)} 件 <button onclick="commitNow()">今すぐコミット</button> <span class="muted">${esc(d.git||'')}</span>`:'自動コミット: 無効');
 $('recent').innerHTML=d.recent.map(r=>`<li>${esc(r.name)}: ${esc(r.text)}</li>`).join('')||'<li class="muted">なし</li>';
 const s=d.items.map(i=>i.horse_id).join(',');batch=d.items;if(s!==sig||force){sig=s;initBatch()}updateDet(d.detected)}
async function saveBatch(){const out=[];batch.forEach(it=>{const h=it.horse_id,text=$('in_'+h).value.trim(),none=$('nn_'+h).checked,j1=$('j1_'+h).checked;
  const u=$('ur_'+h).value.trim()||$('ur_'+h).dataset.det||'';
  if(none)out.push({horse_id:h,none:true});else if(text)out.push({horse_id:h,text,confirm_jan1:j1,url:u});else out.push({horse_id:h,skip:true})});
 const r=await api('/api/batch_save',{rows:out});let bad=0;
 r.results.forEach(x=>{if(!x.ok){bad++;rows[x.horse_id].err=x.error||'保存できません'}else{delete rows[x.horse_id]}});
 await poll(true);if(bad)alert(bad+' 頭は保存できませんでした。赤い行の理由を確認してください')}
async function commitNow(){const r=await api('/api/commit',{});if(!r.ok)alert(r.error);poll(true)}
async function undo_(){const r=await api('/api/undo',{});if(!r.ok)alert(r.error);poll(true)}
poll(true);setInterval(()=>poll(false),1500);
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
            elif path == "/api/batch" and self._authorized():
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                self._json(store.batch(int((q.get("n") or ["5"])[0])))
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
            elif path == "/api/set_url":
                self._json(store.set_url(body.get("horse_id", ""), body.get("url", "")))
            elif path == "/api/batch_save":
                self._json(store.batch_save(body.get("rows", [])))
            elif path == "/api/ext_detect":
                self._json(store.detect(body.get("url", ""), body.get("title", ""), body.get("text"), body.get("year_only")))
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
