"""手元にある全種牡馬を、smartrc の系統の区分（国系統 → 大系統 → 小系統）ごとに並べた 1 枚の HTML を作る。

使い方: python src/scripts/docs/build_sire_lines_html.py [OUT.html]
入力:
  data/meta/sire_lines/sire_smartrc_colors.csv  … 種牡馬ごとの smartrc の区分と色（python -m src.scraper.smartrc_sire_lines colors で作る。取得なし）
  data/page_reference/pedigree_race_index/horse_sire_map.parquet … 手元の馬 → 父の対応（産駒数の数え上げと、系統が空欄の種牡馬の netkeiba の系統）
出力: docs/html/pedigree/sire_lines_all.html（本の章 sire_lines_all）
色は smartrc の画面と同じ（国系統: 日本型 #ddffdd・米国型 #BA945B・欧州型 #00CCFF、大系統は大系統ごとの色）。
smartrc で系統が空欄の種牡馬（比較的新しい輸入種牡馬）は「系統なし」にまとめ、netkeiba の系統で並べる（運用の決まり 2026-11-06）。
"""

from __future__ import annotations

import html
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
COLORS = REPO / "data/meta/sire_lines/sire_smartrc_colors.csv"
SIRE_MAP = REPO / "data/page_reference/pedigree_race_index/horse_sire_map.parquet"
OUT = REPO / "docs/html/pedigree/sire_lines_all.html"
JST = timezone(timedelta(hours=9))
COUNTRIES = ["欧州型", "米国型", "日本型"]
NO_LINE = "系統なし（smartrc で空欄）"
CSS_COLOR = {"Aqua": "#00ffff", "Moccasin": "#ffe4b5", "LimeGreen": "#32cd32", "DeepPink": "#ff1493", "Sienna": "#a0522d", "Orange": "#ffa500"}


def _col(c: str | None) -> str:
    c = CSS_COLOR.get(c or "", c or "#888")
    return c if all(ch.isalnum() or ch == "#" for ch in c) else "#888"   # style 属性に入れるので、色名・#RGB 以外は使わない


def load() -> pd.DataFrame:
    t = pd.read_csv(COLORS)
    m = pd.read_parquet(SIRE_MAP, columns=["sire_id", "sire_name", "sire_line_name"])
    g = m.groupby("sire_id").agg(n=("sire_id", "size"), nk_line=("sire_line_name", lambda s: s.dropna().mode().iat[0] if s.notna().any() else None))
    t = t.merge(g, left_on="sire_id", right_index=True, how="left")
    t["n"] = t["n"].fillna(0).astype(int)
    t["name"] = t["name"].fillna(t["sire_id"])
    t["country"] = t["country"].where(t["status"] == "ok", NO_LINE)
    t["large_name"] = t["large_name"].where(t["status"] == "ok", "netkeiba の系統")
    t["small_name"] = t["small_name"].where(t["status"] == "ok", t["nk_line"].fillna("不明"))
    return t


def _chip(r: pd.Series) -> str:
    return (f'<span class="chip" data-n="{r.n}" data-name="{html.escape(str(r["name"]).lower())}" title="{html.escape(str(r["name"]))} ／ 手元の産駒 {r.n} 頭 ／ ID {html.escape(str(r.sire_id))}">'
            f'{html.escape(str(r["name"]))}<i>{r.n}</i></span>')


def build(t: pd.DataFrame) -> str:
    total = len(t)
    n_large, n_small = t.loc[t["status"] == "ok", "large_name"].nunique(), t.loc[t["status"] == "ok", "small_name"].nunique()
    parts = []
    cnt = t.groupby("country").size()
    order = [c for c in COUNTRIES if c in cnt.index] + sorted(c for c in cnt.index if c not in COUNTRIES and c != NO_LINE) + ([NO_LINE] if NO_LINE in cnt.index else [])
    # 全体のマップ: 国系統 × 大系統の頭数
    mat = t[t["country"] != NO_LINE].pivot_table(index="large_name", columns="country", values="sire_id", aggfunc="count", fill_value=0)
    mat = mat.reindex(columns=[c for c in order if c in mat.columns])
    mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in mat.columns)
    rows = "".join(f"<tr><th>{html.escape(str(i))}</th>" + "".join(f'<td class="num">{v if v else ""}</td>' for v in r) + f'<td class="num"><b>{int(r.sum())}</b></td></tr>' for i, r in mat.iterrows())
    parts.append(f'<div class="scroll"><table class="mat"><tr><th>大系統 ＼ 国系統</th>{head}<th>計</th></tr>{rows}'
                 f'<tr><th>計（系統あり {int((t["country"] != NO_LINE).sum())}）</th>' + "".join(f'<td class="num"><b>{int(v)}</b></td>' for v in mat.sum(axis=0)) + f'<td class="num"><b>{int(mat.values.sum())}</b></td></tr></table></div>')
    summary = "".join(f'<a class="pill" href="#c-{i}" style="border-color:{_col(t[t["country"] == c]["country_color"].dropna().iat[0] if (t["country"] == c).any() and t[t["country"] == c]["country_color"].notna().any() else "#888")}">{html.escape(str(c))} <b>{cnt.get(c, 0)}</b></a>' for i, c in enumerate(order))
    body = []
    for i, c in enumerate(order):
        sub = t[t["country"] == c]
        if sub.empty:
            continue
        color = _col(sub["country_color"].dropna().iat[0]) if sub["country_color"].notna().any() else "#888"
        body.append(f'<section class="country" id="c-{i}"><h2 style="border-left:8px solid {color}">{html.escape(c)} <span class="n">{len(sub)} 頭</span></h2>')
        larges = sub.groupby("large_name").size().sort_values(ascending=False)
        for lg in larges.index:
            ls = sub[sub["large_name"] == lg]
            lc = _col(ls["large_color"].dropna().iat[0]) if ls["large_color"].notna().any() else "#888"
            body.append(f'<div class="large"><h3><span class="dot" style="background:{lc}"></span>{html.escape(str(lg))} <span class="n">{len(ls)} 頭</span></h3>')
            for sm in ls.groupby("small_name").size().sort_values(ascending=False).index:
                ss = ls[ls["small_name"] == sm].assign(_k=lambda d: d["name"].astype(str).str.lower()).sort_values(["n", "_k"], ascending=[False, True])
                chips = "".join(_chip(r) for _, r in ss.iterrows())
                body.append(f'<div class="small"><h4>{html.escape(str(sm))} <span class="n">{len(ss)} 頭・手元の産駒 {int(ss.n.sum()):,} 頭</span></h4><div class="chips">{chips}</div></div>')
            body.append("</div>")
        body.append("</section>")
    now = datetime.now(JST).strftime("%Y-%m-%d")
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>全種牡馬の系統（smartrc の区分）— keiba-vpn docs</title><style>
:root {{ --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e5e9f5; --muted:#9aa7c7; --accent:#7aa9ff; --code:#1c2a4d; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.7 "Noto Sans JP",system-ui,sans-serif; }}
main {{ max-width:1280px; margin:0 auto; padding:36px 28px 80px; }}
h1 {{ font-size:26px; margin:0 0 6px; }} h2 {{ font-size:20px; margin:38px 0 8px; padding:4px 0 4px 12px; background:var(--panel); border-radius:0 8px 8px 0; }}
h3 {{ font-size:16px; margin:18px 0 4px; }} h4 {{ font-size:14px; margin:12px 0 4px; color:var(--text); }}
.muted {{ color:var(--muted); font-size:13px; }} .n {{ color:var(--muted); font-size:12.5px; font-weight:400; }}
code {{ font-family:ui-monospace,Menlo,monospace; background:var(--code); padding:1px 6px; border-radius:5px; font-size:12px; }}
.scroll {{ overflow-x:auto; }} table.mat {{ border-collapse:collapse; margin:8px 0 14px; font-size:13px; }}
.mat th,.mat td {{ border:1px solid var(--border); padding:4px 10px; text-align:left; }} .mat th {{ background:var(--panel); color:var(--muted); font-weight:600; white-space:nowrap; }}
.mat td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.pill {{ display:inline-block; margin:2px 8px 2px 0; padding:2px 12px; border:2px solid; border-radius:999px; color:var(--text); text-decoration:none; font-size:13.5px; }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:8px; vertical-align:-1px; border:1px solid #0006; }}
.chips {{ display:flex; flex-wrap:wrap; gap:5px; }} .chip {{ background:var(--code); border:1px solid var(--border); border-radius:6px; padding:1px 8px; font-size:12.5px; white-space:nowrap; }}
.chip i {{ font-style:normal; color:var(--muted); margin-left:6px; font-size:11.5px; font-variant-numeric:tabular-nums; }} .chip.hit {{ border-color:var(--accent); background:#223a6e; }} .chip.dim {{ opacity:.25; }}
.bar {{ position:sticky; top:0; z-index:5; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--border); display:flex; gap:12px; flex-wrap:wrap; align-items:center; }}
.bar input,.bar select {{ background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:5px 9px; font-size:13.5px; }}
.hidden {{ display:none; }}
</style></head><body><main>
<h1>全種牡馬の系統（smartrc の区分）</h1>
<p class="muted">作成: {now}。手元にある種牡馬 {total:,} 頭（手元の馬 → 父の対応表に出てくる全頭）を、smartrc の出馬表の区分で並べた。
国系統（欧州型・米国型・日本型。色は smartrc の画面と同じ）→ 大系統（{n_large} 種）→ 小系統（{n_small} 種）の順で、頭数の多い順。各小系統の中は手元の産駒数の多い順。
名前の横の数字は、手元の産駒の頭数。smartrc で系統が空欄の種牡馬（新しい輸入種牡馬など）は「系統なし」にして、netkeiba の系統で並べた。
元データ: <code>data/meta/sire_lines/sire_smartrc_colors.csv</code>（<code>python -m src.scraper.smartrc_sire_lines colors</code>）。作成: <code>src/scripts/docs/build_sire_lines_html.py</code>。</p>
<p>{summary}</p>
<h3>国系統 × 大系統の頭数</h3>
{parts[0]}
<div id="sl-root"><div class="bar"><input id="q" type="search" placeholder="種牡馬名で絞り込み（例: Galileo、ドゥラメンテ）" size="34">
<label class="muted">並び順 <select id="o"><option value="n">手元の産駒数の多い順</option><option value="name">名前順</option></select></label>
<span class="muted" id="hits"></span></div>
{"".join(body)}
</div>
<script>
(function(){{
  var root=document.getElementById('sl-root'), q=document.getElementById('q'), o=document.getElementById('o'), hits=document.getElementById('hits');
  if(!root||!q||!o) return;
  function sortChips(){{
    root.querySelectorAll('.chips').forEach(function(box){{
      var a=[].slice.call(box.children);
      a.sort(function(x,y){{ return o.value==='n' ? (y.dataset.n-x.dataset.n)||x.dataset.name.localeCompare(y.dataset.name) : x.dataset.name.localeCompare(y.dataset.name); }});
      a.forEach(function(c){{ box.appendChild(c); }});
    }});
  }}
  function filter(){{
    var s=q.value.trim().toLowerCase(), n=0;
    root.querySelectorAll('.chip').forEach(function(c){{
      var m = !s || c.dataset.name.indexOf(s)>=0;
      c.classList.toggle('dim', !m); c.classList.toggle('hit', !!s && m); if(m && s) n++;
    }});
    hits.textContent = s ? n+' 頭が一致' : '';
  }}
  q.addEventListener('input', filter); o.addEventListener('change', sortChips);
}})();
</script>
</main></body></html>
"""


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    t = load()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(t), encoding="utf-8")
    print(f"{len(t)} 頭 → {out}")


if __name__ == "__main__":
    main()
