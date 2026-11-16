"""重賞の HTML（ルール一覧・早見表・予想家）で共通の、会場 → コースのプルダウン。

- コースの候補は表にある値から作る。並びは会場の順（札幌・函館・福島・新潟・東京・中山・中京・京都・阪神・小倉）→ 芝・ダ → 距離 → 内外。
- 件数（その会場・コースの重賞の数）を併記する。会場を選ぶと、コースの候補がその会場だけに絞られる。
- 文字の検索とは別に、組み合わせて絞れる（各ページの apply が両方を見る）。
"""

from __future__ import annotations

import json
import re

VENUE_ORDER = ("札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉")
_COURSE = re.compile(r"^(?P<v>.+?)(?P<s>芝|ダ|障)(?P<d>\d+)(?P<io>.*)$")


def split_course(course: str) -> tuple[str, str, int, str]:
    """「東京芝2400」「阪神芝1600外」→（会場, 芝ダ, 距離, 内外）。読めなければ（全体, "", 0, ""）。"""
    m = _COURSE.match(str(course or ""))
    if not m:
        return str(course or ""), "", 0, ""
    return m["v"], m["s"], int(m["d"]), m["io"]


def sort_key(course: str) -> tuple:
    v, s, d, io = split_course(course)
    vi = VENUE_ORDER.index(v) if v in VENUE_ORDER else len(VENUE_ORDER)
    return (vi, v, {"芝": 0, "ダ": 1, "障": 2}.get(s, 3), d, io)


def options(courses_by_key: dict[str, str]) -> list[dict]:
    """{重賞の鍵: コース} → [{"v": 会場, "c": コース, "n": 重賞の数}]（並べた順）。"""
    n: dict[str, int] = {}
    for c in courses_by_key.values():
        if c:
            n[c] = n.get(c, 0) + 1
    return [{"v": split_course(c)[0], "c": c, "n": k} for c, k in sorted(n.items(), key=lambda kv: sort_key(kv[0]))]


def options_json(courses_by_key: dict[str, str]) -> str:
    return json.dumps(options(courses_by_key), ensure_ascii=False).replace("</", "<\\/")


# setupCourse(会場の select, コースの select, 候補, 変わったときに呼ぶ関数, 「すべて」の文言の頭, 数の単位（既定「重賞」）)。
# 会場の候補にも重賞の数を付ける。会場を変えると、コースの候補をその会場だけにし、選んでいたコースが外れたら「すべて」に戻す
COURSE_JS = r"""
function setupCourse(vs,cs,L,cb,head,unit){unit=unit||'重賞';const e=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const hv=head?'会場: ':'',hc=head?'コース: ':'';const nv={};L.forEach(x=>{nv[x.v]=(nv[x.v]||0)+x.n});
 vs.innerHTML=`<option value="">${hv}すべて</option>`+Object.keys(nv).map(v=>`<option value="${e(v)}">${e(v)}（${nv[v]} ${unit}）</option>`).join('');
 const fill=()=>{const v=vs.value,cur=cs.value;cs.innerHTML=`<option value="">${hc}すべて</option>`+L.filter(x=>!v||x.v===v).map(x=>`<option value="${e(x.c)}">${e(x.c)}（${x.n} ${unit}）</option>`).join('');
  cs.value=[...cs.options].some(o=>o.value===cur)?cur:''};
 vs.addEventListener('change',()=>{fill();cb()});cs.addEventListener('change',cb);fill()}
"""
