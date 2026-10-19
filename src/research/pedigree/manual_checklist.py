"""ブラウザで人が見た誕生日を入力するための、ローカルだけで動く HTML チェックリストを作る（サイトへは一切アクセスしない）。

    python -m src.research.pedigree.manual_checklist --ancestors ancestor_birthdays.csv --zone data/meta/dormello/zone_review.csv \
        --nk nk_birth_results.jsonl --manual data/meta/dormello/birth_manual.csv --out data/meta/dormello/sporthorse_checklist.html --limit 100

HTML の入力欄に日付（2025-03-04 / 04-Mar-2025 / Mar 4, 2025 のいずれか）を入れると、生年が合うかを確かめ、
birth_manual.csv と同じ列（horse_id,name,birth_year,birth_month,birth_day,source_url,note）の CSV を書き出せる。
入力は ブラウザの localStorage に保存される。1 月 1 日は仮置きの疑いがあるため警告する。
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path

TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>誕生日 入力チェックリスト</title>
<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:4px 8px}
th{background:#eee;position:sticky;top:0}input{width:130px}.ok{background:#e6f7e6}.warn{background:#fff3cd}.bad{background:#f8d7da}.muted{color:#666}
button{margin:4px 8px 8px 0;padding:6px 12px}</style></head><body>
<h1>誕生日 入力チェックリスト</h1>
<p class="muted">各行のリンクをブラウザで開き、ページの「Date of Birth」（例: 30-Jan-1970）を右の欄に入力します。馬名のページ名はサイトによって <code>-1</code> などが付くため、開けないときは馬名でサイト内を探してください。入力はこのブラウザに自動保存されます。</p>
<p>確認済み: <b id="cnt">0</b> / <span id="tot"></span>
<button onclick="exportCsv()">CSV を書き出す（クリップボードへコピー）</button><button onclick="downloadCsv()">CSV をダウンロード</button></p>
<table><thead><tr><th>#</th><th>馬名</th><th>生年</th><th>影響</th><th>ページ</th><th>誕生日（入力）</th><th>判定</th></tr></thead><tbody id="rows"></tbody></table>
<script>
const DATA = __DATA__;
const MON = {jan:1,feb:2,mar:3,apr:4,may:5,jun:6,jul:7,aug:8,sep:9,oct:10,nov:11,dec:12};
function parse(s){s=s.trim();let m;
 if(m=s.match(/^(\\d{4})-(\\d{1,2})-(\\d{1,2})$/))return [+m[1],+m[2],+m[3]];
 if(m=s.match(/^(\\d{1,2})[-\\s]([A-Za-z]{3})[a-z]*[-\\s,]*(\\d{4})$/)){const k=MON[m[2].toLowerCase()];return k?[+m[3],k,+m[1]]:null}
 if(m=s.match(/^([A-Za-z]{3})[a-z]*\\s+(\\d{1,2}),?\\s*(\\d{4})$/)){const k=MON[m[1].toLowerCase()];return k?[+m[3],k,+m[2]]:null}
 return null}
function judge(d,row){if(!d)return ['bad','日付の形式が読めません'];
 if(d[0]!==row.year)return ['bad','生年が違います（'+row.year+' のはず）'];
 if(d[1]===1&&d[2]===1)return ['warn','1月1日は仮置きの疑い（そのまま使うなら備考に注記）'];
 return ['ok','OK']}
const store=JSON.parse(localStorage.getItem('dormello_checklist')||'{}');
const tb=document.getElementById('rows');
DATA.forEach((r,i)=>{const tr=document.createElement('tr');
 tr.innerHTML=`<td>${i+1}</td><td>${r.name}<br><span class="muted">${r.name_en||''}</span></td><td>${r.year}</td><td>${r.impact}</td>
 <td><a href="${r.url}" target="_blank" rel="noopener">開く</a></td><td><input data-i="${i}" value="${store[r.horse_id]||''}" placeholder="例 30-Jan-1970"></td><td class="j"></td>`;
 tb.appendChild(tr)});
document.getElementById('tot').textContent=DATA.length;
function refresh(){let n=0;document.querySelectorAll('input').forEach(inp=>{const r=DATA[+inp.dataset.i];const v=inp.value.trim();
 const td=inp.closest('tr').querySelector('.j');td.className='j';
 if(!v){td.textContent='';return}
 const [c,msg]=judge(parse(v),r);td.className='j '+c;td.textContent=msg;if(c==='ok'||c==='warn')n++;
 store[r.horse_id]=v});
 localStorage.setItem('dormello_checklist',JSON.stringify(store));document.getElementById('cnt').textContent=n}
document.querySelectorAll('input').forEach(i=>i.addEventListener('input',refresh));refresh();
function csv(){const lines=['horse_id,name,birth_year,birth_month,birth_day,source_url,note'];
 DATA.forEach(r=>{const v=(store[r.horse_id]||'').trim();if(!v)return;const d=parse(v);const [c]=judge(d,r);if(c==='bad'||!d)return;
  const note=c==='warn'?'1月1日の仮置きの疑いあり。ユーザーがブラウザで確認':'ユーザーがブラウザで目視確認';
  lines.push([r.horse_id,'"'+r.name.replace(/"/g,'""')+'"',d[0],d[1],d[2],r.url,'"'+note+'"'].join(','))});
 return lines.join('\\n')+'\\n'}
function exportCsv(){navigator.clipboard.writeText(csv()).then(()=>alert('CSV をコピーしました。会話に貼ってください。'))}
function downloadCsv(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv()],{type:'text/csv'}));a.download='birth_manual_input.csv';a.click()}
</script></body></html>
"""


def slug(name: str) -> str:
    s = re.sub(r"[\(（].*?[\)）]", "", name.replace("\xa0", " ")).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def build_rows(ancestors: dict[str, dict], zone: list[dict], nk: dict[str, dict], manual: set[str], limit: int, base_url: str):
    weight: dict[str, int] = defaultdict(int)
    for r in zone:
        if r["min_gen"] and int(r["min_gen"]) <= 3:
            weight[r["stallion_id"]] += int(r["n_horses"])
    cand = [(h, n) for h, n in weight.items() if h in ancestors and ancestors[h]["precision"] == "year" and h not in manual
            and int(ancestors[h]["birthday"][:4]) >= 1900]
    cand.sort(key=lambda x: -x[1])
    rows = []
    for h, n in cand[:limit]:
        a = ancestors[h]
        en = (nk.get(h, {}).get("name") or "").strip() or re.sub(r"[ァ-ヶー・]+", "", a["name"]).strip()
        rows.append({"horse_id": h, "name": a["name"].replace("\xa0", " "), "name_en": en, "year": int(a["birthday"][:4]),
                     "impact": n, "url": f"{base_url.rstrip('/')}/{slug(en) or slug(a['name'])}"})
    return rows


def render(rows: list[dict]) -> str:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", data)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ancestors", required=True, type=Path)
    ap.add_argument("--zone", required=True, type=Path)
    ap.add_argument("--nk", type=Path, help="nk_birth_results.jsonl（英語名を得る）")
    ap.add_argument("--manual", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--base-url", default="https://sporthorse-data.com/pedigree")
    a = ap.parse_args(argv)
    anc = {r["horse_id"]: r for r in csv.DictReader(open(a.ancestors, encoding="utf-8"))}
    zone = list(csv.DictReader(open(a.zone, encoding="utf-8")))
    nk = {}
    if a.nk and a.nk.exists():
        for line in open(a.nk, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                nk[r["horse_id"]] = r
    manual = {r["horse_id"] for r in csv.DictReader(open(a.manual, encoding="utf-8"))}
    rows = build_rows(anc, zone, nk, manual, a.limit, a.base_url)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(rows), encoding="utf-8")
    print(f"{len(rows)} 頭 -> {a.out}")


if __name__ == "__main__":
    main()
