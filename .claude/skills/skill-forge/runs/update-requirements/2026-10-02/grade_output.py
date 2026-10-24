#!/usr/bin/env python3
"""update-requirements の成果物HTMLを機械採点して grading.json を書く。
usage: grade_output.py OUT.html STATE.json EVAL_ID [PREV_FIXTURE.html] > grading.json
  content:* = スキル無しでも満たせるべき内容, format:* = スキルの出力契約"""
import json, re, subprocess, sys
from pathlib import Path
ROOT = Path('/home/hirokiakataoka/project/myproject/keiba-vpn')
out, state_p, eid = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
prev = Path(sys.argv[4]) if len(sys.argv) > 4 else None
res = []
def add(text, ok, ev): res.append({"text": text, "passed": bool(ok), "evidence": ev})
if not out.exists():
    add("output exists", False, "no file"); print(json.dumps({"expectations": res, "summary": {"passed":0,"failed":1,"total":1,"pass_rate":0.0}}, ensure_ascii=False)); sys.exit()
raw = out.read_text(encoding='utf-8', errors='ignore')
text = re.sub(r'<!--.*?-->', '', raw, flags=re.S)
plain = re.sub(r'<[^>]+>', ' ', text)
st = json.load(open(state_p))
def section(sid):
    m = re.search(rf'<section id="{sid}".*?(?=<section id=|</main>|$)', text, re.S)
    return m.group(0) if m else ''
def h2s(): return re.findall(r'<h2[^>]*>(.*?)</h2>', text, re.S)
# ---- content (baseline でも測る)
pages = [p['route'] for p in st['frontend_pages']]
miss = [r for r in pages if (r.split('[')[0].rstrip('/') or '/') not in plain]
add("content: フロント全ページ(frontend_pages)が文書内に言及されている", len(miss) <= 1 and len(pages) > 0, f"{len(pages)-len(miss)}/{len(pages)} missing={miss[:6]}")
keys = [m['key'] for m in st['mlflow_models']['catalog']]
miss = [k for k in keys if k not in plain]
add("content: MLflow catalog の全モデルキーが言及されている", not miss, f"{len(keys)-len(miss)}/{len(keys)} missing={miss}")
add("content: 3層APIのポート(8000/5000/9090)がすべて記載", all(p in plain for p in ('8000','5000','9090')), "ports")
decs = sorted({re.search(r'(DEC-\d+)', d['file']).group(1) for d in st['decisions'] if 'DEC-' in d['file']})
miss = [d for d in decs if d not in plain]
add("content: DEC-013〜026 の全IDが言及されている", not miss, f"{len(decs)-len(miss)}/{len(decs)} missing={miss}")
add("content: DEC-024 の精度目標 40% がKPIとして記載", bool(re.search(r'40\s*%', plain)), "40% grep")
add("content: stg_mock(モック)が文書内で明示されている", 'stg_mock' in plain or 'モック' in plain, "stg_mock/モック grep")
add("content: 最後の h2 がスケジューリング(TODO)である", bool(h2s()) and bool(re.search(r'スケジュ|TODO', h2s()[-1])), f"last h2={re.sub('<[^>]+>','',h2s()[-1]).strip() if h2s() else None}")
docs = [d['file'] for d in st['docs_index'] if d['file'].startswith('docs/html/') and d['file'].endswith('.html')]
miss = [d for d in docs if d.split('docs/html/')[1] not in text]
add("content: docs/html 配下の全ドキュメントへリンクされている", len(miss) <= 2, f"{len(docs)-len(miss)}/{len(docs)} missing={miss[:5]}")
# ---- format contract
r = subprocess.run([sys.executable, str(ROOT/'.claude/skills/update-requirements/scripts/validate_requirements.py'), str(out)], capture_output=True, text=True)
add("format: validate_requirements.py が exit 0", r.returncode == 0, (r.stdout.strip().splitlines() or [''])[-1] + (' | ' + ' / '.join(l for l in r.stdout.splitlines() if l.startswith('[error]'))[:300] if r.returncode else ''))
head = subprocess.check_output(['git','rev-parse','--short','HEAD'], cwd=ROOT, text=True).strip()
m = re.search(r'name="requirements-commit" content="([^"]*)"', text)
add("format: meta requirements-commit が現HEAD", m and m.group(1) == head, f"{m.group(1) if m else None} vs {head}")
m = re.search(r'name="requirements-updated" content="([^"]*)"', text)
add("format: meta requirements-updated が実行日(2026-10-02)", m and m.group(1) == '2026-10-02', m.group(1) if m else None)
rows = re.findall(r'<tr\b[^>]*data-todo-id="[^"]+"[^>]*>', text)
kinds = [re.search(r'data-kind="(\w+)"', x).group(1) for x in rows if 'data-kind' in x]
add("format: §14 に MOCK 種別の TODO が1件以上ある(モックの明示)", 'MOCK' in kinds, f"{len(rows)} rows kinds={ {k:kinds.count(k) for k in set(kinds)} }")
tags = {k: len(re.findall(rf'class="tag {k}"', text)) for k in ('done','partial','mock','todo','dep')}
add("format: 状況タグ(done/partial/mock/todo)が本文で全種使われている", all(tags[k] > 2 for k in ('done','partial','mock','todo')), str(tags))
add("format: §7 に frontend 全ページ表(各ページにタグ)", section('delivery').count('class="tag') >= len(pages)//2, f"tags in delivery={section('delivery').count('class=\"tag')}")
tm = re.sub(r'\s+',' ', section('ml-flow'))
add("format: §6 に MLflow 全キーが載る", all(k in tm for k in keys), f"missing={[k for k in keys if k not in tm]}")
# ---- scenario specific
decs_html = [d['file'].split('docs/')[1] for d in st['docs_index'] if d['file'].startswith('docs/decisions/')]
miss = [d for d in decs_html if '../' + d not in text]
add("format(v2): §13 に docs/decisions 配下の全DEC/AREAが相対リンクで載る", len(miss) == 0, f"{len(decs_html)-len(miss)}/{len(decs_html)} missing={miss[:4]}")
if eid in (2, 3, 4) and prev:
    ptext = prev.read_text()
    pids = re.findall(r'data-todo-id="([^"]+)"', ptext)
    ids = re.findall(r'data-todo-id="([^"]+)"', text)
    if eid == 4:
        add("scenario: 前回の全TODO ID がそのまま残っている(再採番/削除なし)", all(i in ids for i in pids), f"prev={len(pids)} kept={sum(i in ids for i in pids)}")
        add("scenario: 前回 done の行は done のまま", all(re.search(rf'data-todo-id="{i}"[^>]*data-status="(?:done)"', text) for i in re.findall(r'data-todo-id="([^"]+)"[^>]*data-status="done"', ptext)), "done rows preserved")
    if eid == 2:
        add("scenario: 前回の T-001..T-004 がID維持で残っている(再採番/削除なし)", all(i in ids for i in pids), f"prev={pids} now_has={[i for i in pids if i in ids]}")
        news = [int(i[2:]) for i in ids if i not in pids and re.fullmatch(r'T-\d+', i)]
        add("scenario: 新規TODOは T-005 以降で採番", bool(news) and min(news) >= 5, f"new min={min(news) if news else None} count={len(news)}")
        m = re.search(r'data-todo-id="T-004"[^>]*data-status="(\w+)"', text)
        add("scenario: 前回 done の T-004 は done のまま", m and m.group(1) == 'done', m.group(1) if m else None)
        add("scenario: 本文中に『前回』との差分(since_prev)への言及がある", bool(re.search(r'前回|差分|since', plain)) , "grep")
    else:
        add("scenario: 存在しない依存 T-099 が残っていない", 'T-099' not in re.findall(r'data-depends="([^"]*)"', text).__str__(), "T-099 in depends")
        deps = {re.search(r'data-todo-id="([^"]+)"',x).group(1): re.search(r'data-depends="([^"]*)"',x).group(1) for x in rows if 'data-depends' in x}
        add("scenario: T-002/T-003 の循環依存が解消されている", not (deps.get('T-002')=='T-003' and deps.get('T-003')=='T-002'), str({k:v for k,v in deps.items() if k in ('T-001','T-002','T-003')}))
        mm = re.search(r'name="requirements-commit" content="([^"]*)"', text)
        add("scenario: meta requirements-commit に壊れた前回 commit(deadbeef) を引き継がない", mm and mm.group(1) != 'deadbeef', mm.group(1) if mm else None)
p = sum(x['passed'] for x in res)
print(json.dumps({"expectations": res, "summary": {"passed": p, "failed": len(res)-p, "total": len(res), "pass_rate": round(p/len(res), 3)}}, ensure_ascii=False, indent=2))
