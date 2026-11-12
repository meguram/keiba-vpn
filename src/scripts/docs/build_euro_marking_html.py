"""欧州型に割り振られている種牡馬（smartrc の国系統）の一覧に、「欧州瞬発力だと考えられる（○）／考えられない（×）」を付けるための HTML を作る。

使い方: python src/scripts/docs/build_euro_marking_html.py [OUT.html]
入力:
  data/meta/sire_lines/sire_smartrc_colors.csv  … 種牡馬ごとの smartrc の区分（python -m src.scraper.smartrc_sire_lines colors）
  data/page_reference/pedigree_race_index/horse_sire_map.parquet … 手元の産駒数の数え上げ
  data/local/horse_lap/horse_lap_entries.parquet … 代表産駒の名前（実測ラップの取れた馬だけ。無くても動く）
出力: data/meta/sire_lines/euro_kick_marking.html（git 管理。docs/ 配下ではないので本には載せない）

画面の操作: ○・×・未判定のボタン。マークはブラウザの localStorage に自動保存される。「JSON を保存」でダウンロード、「読み込み」で復元できる。
判定を偏らせないため、こちらの測定値（素点・順位）は表示しない。手がかりは、系統・産駒数・代表産駒の名前だけ。
保存した JSON の形: {"version": 1, "marks": {"<種牡馬ID>": {"name": 種牡馬名, "mark": "o" | "x"}}, "saved_at": 日時}
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
COLORS = REPO / "data/meta/sire_lines/sire_smartrc_colors.csv"
SIRE_MAP = REPO / "data/page_reference/pedigree_race_index/horse_sire_map.parquet"
ENTRIES = REPO / "data/local/horse_lap/horse_lap_entries.parquet"
OUT = REPO / "data/meta/sire_lines/euro_kick_marking.html"
JST = timezone(timedelta(hours=9))
COUNTRY = "欧州型"


def load() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """欧州型の種牡馬の表（産駒数つき）と、種牡馬 ID → 代表産駒の名前（最大 3 頭）。"""
    t = pd.read_csv(COLORS, dtype=str)
    t = t[t["country"] == COUNTRY].copy()
    m = pd.read_parquet(SIRE_MAP, columns=["horse_id", "sire_id"]).astype(str)
    t["n"] = t["sire_id"].map(m.groupby("sire_id").size()).fillna(0).astype(int)
    prog: dict[str, list[str]] = {}
    try:
        e = pd.read_parquet(ENTRIES, columns=["horse_id", "horse_name"]).astype(str)
        runs = e.groupby("horse_id").size().rename("runs")
        names = e.drop_duplicates("horse_id").set_index("horse_id")["horse_name"]
        j = m.merge(runs, left_on="horse_id", right_index=True).merge(names, left_on="horse_id", right_index=True)
        for sid, g in j.sort_values("runs", ascending=False).groupby("sire_id"):
            prog[sid] = list(g["horse_name"].head(3))
    except FileNotFoundError:
        prog = {}      # 実測ラップの表が無い環境では、代表産駒を空にする（画面の動作には影響しない）
    return t, prog


def build(t: pd.DataFrame, prog: dict[str, list[str]]) -> str:
    """表から HTML を作る（t は sire_id・name・small_name・type_name・n を持つ）。"""
    t = t.sort_values(["small_name", "n", "name"], ascending=[True, False, True])
    smalls = list(t.groupby("small_name").size().sort_values(ascending=False).index)
    rows = []
    for _, r in t.iterrows():
        sid = html.escape(str(r["sire_id"]), quote=True)
        name = html.escape(str(r["name"]))
        pg = html.escape("、".join(prog.get(str(r["sire_id"]), [])))
        rows.append(
            f'<tr data-id="{sid}" data-name="{html.escape(str(r["name"]).lower(), quote=True)}" data-small="{html.escape(str(r["small_name"]), quote=True)}" data-n="{int(r["n"])}">'
            f'<td class="nm">{name}</td><td>{html.escape(str(r["small_name"]))}</td><td class="ty">{html.escape(str(r["type_name"]))}</td>'
            f'<td class="num">{int(r["n"])}</td><td class="pg">{pg}</td>'
            f'<td class="btns"><button type="button" class="b o" data-m="o" title="欧州瞬発力だと考えられる">○</button>'
            f'<button type="button" class="b x" data-m="x" title="欧州瞬発力だとは考えられない">×</button>'
            f'<button type="button" class="b c" data-m="" title="未判定に戻す">－</button></td></tr>'
        )
    opts = "".join(f'<option value="{html.escape(s, quote=True)}">{html.escape(s)}（{int((t["small_name"] == s).sum())}）</option>' for s in smalls)
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>欧州瞬発力 マーキング（欧州型の種牡馬）</title><style>
:root {{ --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e5e9f5; --muted:#9aa7c7; --accent:#7aa9ff; --ok:#34d399; --ng:#f87171; }}
@media (prefers-color-scheme: light) {{ :root {{ --bg:#f7f9fd; --panel:#fff; --border:#d3dcee; --text:#1c2740; --muted:#5b6b8f; --accent:#2a5bd7; --ok:#0f9d6a; --ng:#d6453d; }} }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14.5px/1.6 "Noto Sans JP",system-ui,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px 20px 80px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} .muted {{ color:var(--muted); font-size:13px; }}
.bar {{ position:sticky; top:0; z-index:5; background:var(--bg); padding:10px 0; border-bottom:1px solid var(--border); display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
.bar input,.bar select,.bar button {{ background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:5px 10px; font-size:13.5px; }}
.bar button {{ cursor:pointer; }} .bar button:hover {{ border-color:var(--accent); }}
.cnt b {{ font-variant-numeric:tabular-nums; }} .cnt .o {{ color:var(--ok); }} .cnt .x {{ color:var(--ng); }}
table {{ border-collapse:collapse; width:100%; margin-top:8px; }} th,td {{ border-bottom:1px solid var(--border); padding:5px 8px; text-align:left; vertical-align:middle; }}
th {{ color:var(--muted); font-weight:600; font-size:12.5px; position:sticky; top:52px; background:var(--bg); }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.nm {{ font-weight:600; }} td.ty,td.pg {{ color:var(--muted); font-size:12.5px; }} td.btns {{ white-space:nowrap; }}
.b {{ width:36px; height:30px; border:1px solid var(--border); background:var(--panel); color:var(--muted); border-radius:6px; cursor:pointer; font-size:15px; margin-right:4px; }}
tr.mo {{ background:rgba(52,211,153,.12); }} tr.mx {{ background:rgba(248,113,113,.12); }}
tr.mo .b.o {{ background:var(--ok); color:#fff; border-color:var(--ok); }} tr.mx .b.x {{ background:var(--ng); color:#fff; border-color:var(--ng); }}
.hidden {{ display:none; }} .note.warn {{ border-color:var(--ng); color:var(--ng); font-weight:600; }} .note {{ margin:10px 0; padding:8px 12px; border:1px solid var(--border); border-radius:8px; background:var(--panel); font-size:13px; }}
</style></head><body><main>
<h1>欧州瞬発力 マーキング（欧州型の種牡馬）</h1>
<p class="muted">作成: {now}。smartrc で欧州型に割り振られている種牡馬 {len(t)} 頭。各馬に、<b>欧州瞬発力だと考えられる（○）</b>／<b>考えられない（×）</b>を付けてください（付けない馬は未判定）。
マークはこのブラウザに自動保存されます。終わったら「JSON を保存」でファイルにして、リポジトリの <code>data/meta/sire_lines/euro_kick_marks.json</code> に置いてください。
こちらの測定値（素点・順位）は、判定を偏らせないよう表示していません。系統・産駒数（手元の馬の頭数）・代表産駒（実測ラップのある産駒の走数順）が手がかりです。</p>
<div class="bar">
<input id="q" type="search" placeholder="種牡馬名で絞り込み" size="22">
<select id="s"><option value="">小系統: すべて</option>{opts}</select>
<select id="f"><option value="">表示: すべて</option><option value="u">未判定のみ</option><option value="o">○のみ</option><option value="x">×のみ</option></select>
<select id="o"><option value="n">産駒数の多い順</option><option value="name">名前順</option><option value="small">小系統順</option></select>
<span class="cnt">○ <b class="o" id="co">0</b> ／ × <b class="x" id="cx">0</b> ／ 未判定 <b id="cu">0</b></span>
<button type="button" id="save">JSON を保存</button><button type="button" id="copy">JSON をコピー</button>
<label class="muted">読み込み <input type="file" id="load" accept=".json,application/json"></label><button type="button" id="reset">全部クリア</button>
</div>
<div class="note warn hidden" id="warn" role="alert"></div>
<div class="note" id="msg" aria-live="polite">操作: ボタンを押すとすぐ保存されます（このブラウザの中）。別の PC で続ける場合は「JSON を保存」→「読み込み」。</div>
<table id="t"><thead><tr><th>種牡馬</th><th>小系統</th><th>国別タイプ</th><th>手元の産駒数</th><th>代表産駒</th><th>判定</th></tr></thead><tbody>
{"".join(rows)}
</tbody></table>
<script>
(function(){{
  var KEY='euro_kick_marks_v1', marks=Object.create(null), rows=[].slice.call(document.querySelectorAll('#t tbody tr')), storeOk=true;
  var known=Object.create(null), nameOf=Object.create(null);
  rows.forEach(function(r){{ known[r.dataset.id]=true; nameOf[r.dataset.id]=r.querySelector('.nm').textContent; }});
  var q=document.getElementById('q'), s=document.getElementById('s'), f=document.getElementById('f'), o=document.getElementById('o'), msg=document.getElementById('msg'), warn=document.getElementById('warn');
  function say(t){{ msg.textContent=t; }}
  function isMark(m){{ return m==='o'||m==='x'; }}
  function plain(d){{ return d&&typeof d==='object'&&!Array.isArray(d); }}
  // 保存先(localStorage)が使えるか、起動時に書いて読んで確かめる。だめなら、閉じると消えることを常に見える場所に出す
  try {{ localStorage.setItem(KEY+'_probe','1'); storeOk=(localStorage.getItem(KEY+'_probe')==='1'); localStorage.removeItem(KEY+'_probe'); }} catch(e) {{ storeOk=false; }}
  if(!storeOk) {{ warn.textContent='このブラウザでは自動保存が使えません。マークはこのページを閉じると消えます。こまめに「JSON を保存」してください。'; warn.classList.remove('hidden'); }}
  try {{
    var saved=JSON.parse(localStorage.getItem(KEY)||'{{}}');
    if(plain(saved)) Object.keys(saved).forEach(function(id){{ if(isMark(saved[id])) marks[id]=saved[id]; }});
  }} catch(e) {{}}
  function persist(){{
    var o2={{}}; Object.keys(marks).forEach(function(id){{ o2[id]=marks[id]; }});
    try {{ localStorage.setItem(KEY, JSON.stringify(o2)); }} catch(e) {{ storeOk=false; warn.textContent='自動保存に失敗しました。マークはこのページを閉じると消えます。「JSON を保存」で手元に残してください。'; warn.classList.remove('hidden'); }}
  }}
  function paint(r){{ var m=marks[r.dataset.id]; r.classList.toggle('mo', m==='o'); r.classList.toggle('mx', m==='x'); }}
  function count(){{ var a=0,b=0; rows.forEach(function(r){{ var m=marks[r.dataset.id]; if(m==='o')a++; else if(m==='x')b++; }});
    document.getElementById('co').textContent=a; document.getElementById('cx').textContent=b; document.getElementById('cu').textContent=rows.length-a-b; }}
  function view(){{
    var t=q.value.trim().toLowerCase(), sm=s.value, fl=f.value;
    rows.forEach(function(r){{ var m=marks[r.dataset.id]||'';
      var ok=(!t||r.dataset.name.indexOf(t)>=0)&&(!sm||r.dataset.small===sm)&&(!fl||(fl==='u'?m==='':m===fl));
      r.classList.toggle('hidden', !ok); }});
  }}
  function order(){{
    var tb=document.querySelector('#t tbody'), a=rows.slice();
    a.sort(function(x,y){{ if(o.value==='n') return (y.dataset.n-x.dataset.n)||x.dataset.name.localeCompare(y.dataset.name);
      if(o.value==='name') return x.dataset.name.localeCompare(y.dataset.name);
      return x.dataset.small.localeCompare(y.dataset.small)||(y.dataset.n-x.dataset.n); }});
    a.forEach(function(r){{ tb.appendChild(r); }});
  }}
  document.getElementById('t').addEventListener('click', function(e){{
    var b=e.target.closest('button.b'); if(!b) return; var r=b.closest('tr'), id=r.dataset.id, m=b.dataset.m;
    if(isMark(m)) marks[id]=m; else delete marks[id];
    persist(); paint(r); count(); view();
  }});
  // 書き出し: 表にある種牡馬のマークに加えて、表から外れた種牡馬のマーク(表を作り直した場合など)も落とさない
  function out(){{
    var mk={{}}; Object.keys(marks).forEach(function(id){{ mk[id]={{name:nameOf[id]||'(この表に無い種牡馬)', mark:marks[id]}}; }});
    return JSON.stringify({{version:1, marks:mk, saved_at:new Date().toISOString()}}, null, 1);
  }}
  document.getElementById('save').addEventListener('click', function(){{
    var blob=new Blob([out()],{{type:'application/json'}}), a=document.createElement('a'), u=URL.createObjectURL(blob);
    a.href=u; a.download='euro_kick_marks.json'; document.body.appendChild(a); a.click(); a.remove(); setTimeout(function(){{ URL.revokeObjectURL(u); }}, 4000);
    say('euro_kick_marks.json のダウンロードを始めました。保存先のフォルダに入っているか確認してください（マーク '+Object.keys(marks).length+' 件）。');
  }});
  document.getElementById('copy').addEventListener('click', function(){{
    var t=out(); if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(function(){{ say('JSON をコピーしました（マーク '+Object.keys(marks).length+' 件）。'); }}, function(){{ say('コピーに失敗しました。「JSON を保存」を使ってください。'); }});
    else say('この環境ではコピーできません。「JSON を保存」を使ってください。');
  }});
  // 読み込み: 全体を先に検証し、通ったときだけ今のマークを置き換える（途中で失敗しても画面は変わらない）
  document.getElementById('load').addEventListener('change', function(e){{
    var inp=e.target, fl=inp.files[0]; if(!fl) return; var rd=new FileReader();
    rd.onload=function(){{
      try {{
        var d=JSON.parse(rd.result);
        if(!plain(d)||d.version!==1||!plain(d.marks)) throw new Error('形式が違います');
        var next=Object.create(null), ok=0, unknown=0, bad=0;
        Object.keys(d.marks).forEach(function(id){{
          var v=d.marks[id], m=plain(v)?v.mark:null;
          if(!isMark(m)) {{ bad++; return; }}
          if(!known[id]) {{ unknown++; }}
          next[id]=m; ok++;
        }});
        var cur=Object.keys(marks).length;
        if(cur>0&&!confirm('いまのマーク '+cur+' 件を、ファイルの '+ok+' 件に置き換えます。よろしいですか？')) {{ say('読み込みを取りやめました。'); return; }}
        marks=next; persist(); rows.forEach(paint); count(); view();
        say(ok+' 件を読み込みました'+(unknown?'（うち、この表に無い種牡馬 '+unknown+' 件は保持）':'')+(bad?'。無効な値 '+bad+' 件は無視しました':'')+'。');
      }} catch(err) {{ say('読み込めませんでした（'+err.message+'）。いまのマークは変わっていません。'); }}
    }};
    rd.onerror=function(){{ say('ファイルを読めませんでした。'); }};
    rd.readAsText(fl); inp.value='';                     // 同じファイルを選び直しても change が発火するように戻す
  }});
  document.getElementById('reset').addEventListener('click', function(){{ if(confirm('全部のマークを消しますか？（先に「JSON を保存」しておくことをおすすめします）')){{ marks=Object.create(null); persist(); rows.forEach(paint); count(); view(); say('クリアしました。'); }} }});
  q.addEventListener('input', view); s.addEventListener('change', view); f.addEventListener('change', view); o.addEventListener('change', order);
  rows.forEach(paint); count(); order(); view();
}})();
</script>
</main></body></html>
"""


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    t, prog = load()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(t, prog), encoding="utf-8")
    print(json.dumps({"sires": len(t), "with_progeny_names": len(prog), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
