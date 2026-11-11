"""調教師 × 最終追い切りパターンごとの成績表（着順内訳・勝率・単勝回収率・複勝率・複勝回収率）。ターゲットエンコーディングの元データ。

python -m src.scripts.data.trainer_pattern_table [--out DIR] [--y0 2020] [--y1 2025]

出力:
  trainer_pattern_summary_{y0}_{y1}.csv   期間まるごとの集計（見る・調べる用。特徴量には使わない: 未来の結果を含む）
  trainer_pattern_asof.parquet            レースごとに『その日より前の』調教師 × パターンの累計（特徴量用。同日のレースは含めない）

集計の単位は馬（その調教師のそのパターンで出走した馬 1 頭 = 1 行）。回収率 = 払戻 / 賭け金（100 円ずつ買ったとき。1.00 = 100%）。
払戻が結果表から読めない出走は、率・回収率の両方から除く（分母を揃える）。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import html as _html
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import final_week_pattern_eval as P
from src.scripts.data import final_week_roi_eval as RO
from src.scripts.data.final_week_check import md

KEY = ["trainer_id", "pat"]
SUMS = ["n", "c1", "c2", "c3", "win_ret_sum", "place_ret_sum"]


def _indicators(d: pd.DataFrame) -> pd.DataFrame:
    """パターンが付き、払戻が読める行に、集計用の 0/1 と払戻を付ける（読めない行は全て 0）。"""
    x = d[d["pat"].notna() & ~d["trainer_id"].isin(["nan", "None", ""])].copy()   # 調教師が不明の行は、1 人の調教師として累計しない
    ok = (x["win_ret"].notna() & x["place_ret"].notna()).to_numpy()
    fp = x["finish_position"].to_numpy()
    x["n"] = ok.astype("int64")
    for k in (1, 2, 3):
        x[f"c{k}"] = (ok & (fp == k)).astype("int64")
    x["win_ret_sum"] = np.where(ok, x["win_ret"].fillna(0.0), 0.0)
    x["place_ret_sum"] = np.where(ok, x["place_ret"].fillna(0.0), 0.0)
    return x


def derive(g: pd.DataFrame, pre: str = "") -> pd.DataFrame:
    """累計（n, c1, c2, c3, 払戻の合計）から、4着以下の頭数と率・回収率を求める。n=0 の行は率が NaN。"""
    n = g["n"].astype("float64").where(g["n"] > 0)
    out = pd.DataFrame(index=g.index)
    out[f"{pre}c4p"] = g["n"] - g["c1"] - g["c2"] - g["c3"]
    out[f"{pre}win_rate"] = g["c1"] / n
    out[f"{pre}win_roi"] = g["win_ret_sum"] / n
    out[f"{pre}top3_rate"] = (g["c1"] + g["c2"] + g["c3"]) / n
    out[f"{pre}place_roi"] = g["place_ret_sum"] / n
    return out


def summary(d: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """調教師 × パターンごとの期間集計。着順内訳は『1着-2着-3着-4着以下』の頭数。"""
    x = _indicators(d[d["year"].between(y0, y1)])
    S = x.groupby(KEY)[SUMS].sum()
    S = S[S["n"] > 0]
    S = S.join(derive(S))
    names = d.sort_values("race_date").drop_duplicates("trainer_id", keep="last").set_index("trainer_id")["trainer_name"]
    S = S.reset_index()
    S["trainer_name"] = S["trainer_id"].map(names)
    S["着順内訳"] = S["c1"].astype(str) + "-" + S["c2"].astype(str) + "-" + S["c3"].astype(str) + "-" + S["c4p"].astype(str)
    S = S.rename(columns={"trainer_id": "調教師ID", "trainer_name": "調教師", "pat": "調教パターン", "n": "出走数", "c1": "1着", "c2": "2着", "c3": "3着", "c4p": "4着以下",
                          "win_rate": "勝率", "win_roi": "単勝回収率", "top3_rate": "複勝率", "place_roi": "複勝回収率"})
    cols = ["調教師ID", "調教師", "調教パターン", "出走数", "着順内訳", "1着", "2着", "3着", "4着以下", "勝率", "単勝回収率", "複勝率", "複勝回収率"]
    return S[cols].sort_values(["調教師ID", "出走数"], ascending=[True, False]).reset_index(drop=True)


def asof_table(d: pd.DataFrame) -> pd.DataFrame:
    """各出走（パターンが付いたもの）に、その調教師 × パターンの『出走日より前の』累計を付ける。同じ日の結果は含めない。
    列: trpat_n, trpat_c1, trpat_c2, trpat_c3, trpat_c4p, trpat_win_rate, trpat_win_roi, trpat_top3_rate, trpat_place_roi（と払戻の合計 2 列）。"""
    x = _indicators(d)
    day = x.groupby(KEY + ["race_date"], sort=True)[SUMS].sum().reset_index()
    cum = day.groupby(KEY)[SUMS].cumsum()
    prior = day[KEY + ["race_date"]].join(cum - day[SUMS])   # 累計から同日分を引く = 厳密に前の日までの合計
    prior = prior.join(derive(prior[SUMS]))
    prior = prior.rename(columns={c: f"trpat_{c}" for c in prior.columns if c not in KEY + ["race_date"]})
    keep = [c for c in ("race_id", "horse_id", "horse_number", "race_date") if c in x.columns] + KEY
    out = x[keep].merge(prior, on=KEY + ["race_date"], how="left", validate="many_to_one")
    assert len(out) == len(x)
    return out

START, END = "<!-- tpt:start -->", "<!-- tpt:end -->"
HI_N = 30   # 回収率が 100% を超え、かつ出走数がこれ以上のセルに色を付ける
Z_HI = 2.5   # 勝率・複勝率が全体の水準より、二項分布の標準誤差の何倍以上高いか（出走 HI_N 頭以上のセルに青を付ける）
MIN_N = 10   # 出走数がこれ未満の行は、表で薄く表示する（率・回収率が大きく振れるため）

_CSS = """<style>
#tptw{max-height:720px;overflow:auto;border:1px solid var(--border)} #tpt{margin:0;width:100%} #tpt th{position:sticky;top:0;z-index:1;cursor:pointer;white-space:nowrap}
#tpt tbody{border-top:2px solid var(--border)} #tpt tr.t td{background:#1b2a4d;font-weight:bold;cursor:pointer} #tpt tr.p td{background:#172544;cursor:pointer}
#tpt tr.p td.l{padding-left:26px} #tpt tr.q td.l{padding-left:46px} #tpt td.id{color:var(--muted);font-family:monospace} #tpt tr.sm td{opacity:.5}
#tpt tr td.gr{background:hsl(142 55% calc(var(--l)*1%))} #tpt tr td.bl{background:hsl(215 70% calc(var(--l)*1%))} #tpt tr td.lt{color:#06121f} #tpt tr td.hot{font-weight:bold;box-shadow:inset 0 0 0 1px rgba(255,255,255,.6)}
#tpt tr.ac td.id,#tpt tr.ac td.l{background:#78350f;color:#fde68a} #tpt tr.ac td.id{border-left:5px solid #fbbf24} #tpt tr.ac td.l::before{content:"★ ";color:#fbbf24}
#tpt tbody.c tr.p,#tpt tbody.c tr.q,#tpt tr.q.hid,#tpt tbody.f{display:none} .tptb button{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:3px 10px;margin-right:6px;cursor:pointer}
.tptb input{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:3px 8px;margin-left:8px;width:160px}
</style>"""

_JS = """<script>(function(){var T=document.getElementById("tpt");if(!T)return;var B=function(){return Array.prototype.slice.call(T.tBodies)};
function lv(n){B().forEach(function(b){b.classList.toggle("c",n===0);Array.prototype.forEach.call(b.querySelectorAll("tr.q"),function(r){r.classList.toggle("hid",n<2)})})}
document.getElementById("tpt0").onclick=function(){lv(0)};document.getElementById("tpt1").onclick=function(){lv(1)};document.getElementById("tpt2").onclick=function(){lv(2)};
T.addEventListener("click",function(e){var r=e.target.closest("tr");if(!r||r.parentNode.tagName!=="TBODY")return;
if(r.classList.contains("t"))r.parentNode.classList.toggle("c");
else if(r.classList.contains("p")){var x=r.nextElementSibling;while(x&&x.classList.contains("q")){x.classList.toggle("hid");x=x.nextElementSibling}}});
document.getElementById("tptf").oninput=function(){var v=this.value.trim();B().forEach(function(b){b.classList.toggle("f",v!==""&&b.getAttribute("data-s").indexOf(v)<0)})};
var H=Array.prototype.slice.call(T.tHead.rows[0].cells);
function key(c,i){var t=c.textContent;if(i===3){var a=t.split("-");return a[0]*1e6+a[1]*1e3+ +a[2]}return i<2?t:parseFloat(t)}
function cmp(d){return function(x,y){return(x[1]<y[1]?-1:x[1]>y[1]?1:0)*d}}
H.forEach(function(th,i){th.title="クリックで並べ替え（調教師・場所・パターンの各段の中で並べ替える）";th.onclick=function(){
var d=th._d===undefined?(i<2?1:-1):-th._d;H.forEach(function(h){h._d=undefined;h.textContent=h.textContent.replace(/ [▲▼]$/,"")});th._d=d;th.textContent+=d>0?" ▲":" ▼";
var bs=B().map(function(b){return[b,key(b.rows[0].cells[i],i)]}).sort(cmp(d));
bs.forEach(function(x){var b=x[0],ps=[];Array.prototype.slice.call(b.rows,1).forEach(function(r){var k=[r,key(r.cells[i],i)];
if(r.classList.contains("p")){k.push([]);ps.push(k)}else ps[ps.length-1][2].push(k)});
ps.sort(cmp(d));ps.forEach(function(p){b.appendChild(p[0]);p[2].sort(cmp(d)).forEach(function(q){b.appendChild(q[0])})});T.appendChild(b)})}});})();</script>"""


def _row(cls: str, idv: str, label: str, n, c, win_ret, place_ret, base: tuple[float, float]) -> str:
    """表の 1 行。c = (1着, 2着, 3着, 4着以下)。回収率は払戻の合計 / 出走数。"""
    n = int(n)
    rate = lambda v: v / n   # noqa: E731
    def shade(t: float) -> str:   # 0〜1 → 明るさ（%）。成績が良いほど明るい。明るい側は文字を暗くして読めるようにする
        light = 24 + 40 * min(1.0, max(0.0, t))
        return (f'" style="--l:{light:.0f}' if light < 46 else f' lt" style="--l:{light:.0f}')

    def gr(v: float) -> str:   # 回収率: 1.00 以下は色なし。1.00 超で緑、1.8 以上で最も明るい
        return " gr" + shade((v - 1.0) / 0.8) if n >= HI_N and round(v * 100, 1) > 100.0 else ""   # 表示（小数 1 位の %）が 100.0% 以下なら色なし

    def zs(k: int, p0: float) -> float:
        return (k / n - p0) / (p0 * (1 - p0) / n) ** 0.5 if 0 < p0 < 1 else 0.0

    def bl(k: int, p0: float) -> str:   # 勝率・複勝率: 全体の水準 p0 以下は色なし。水準を超えると青、標準誤差の 6 倍以上高いと最も明るい。Z_HI 倍以上は枠と太字（hot）も付ける
        z = zs(k, p0)
        return f' bl{" hot" if z >= Z_HI else ""}' + shade(z / 6) if n >= HI_N and z > 0 else ""
    shown = [(n, str(n), ""), (None, "-".join(str(int(k)) for k in c), ""), (rate(c[0]), f"{rate(c[0]):.3f}", bl(c[0], base[0])),
             (rate(win_ret), f"{rate(win_ret) * 100:.1f}%", gr(rate(win_ret))), (rate(c[0] + c[1] + c[2]), f"{rate(c[0] + c[1] + c[2]):.3f}", bl(c[0] + c[1] + c[2], base[1])),
             (rate(place_ret), f"{rate(place_ret) * 100:.1f}%", gr(rate(place_ret)))]
    cells = [f'<td class=id>{_html.escape(idv)}</td>', f'<td class=l>{_html.escape(label)}</td>']
    cells += [f'<td class="num{h}">{txt}</td>' for _, txt, h in shown]
    k3 = c[0] + c[1] + c[2]
    won = cls == "q" and n >= HI_N and zs(c[0], base[0]) >= Z_HI and rate(win_ret) > 1.0     # 行のアクセントはパターン行だけ。勝率が水準より高く、単勝回収率も 100% 超
    placed = cls == "q" and n >= HI_N and zs(k3, base[1]) >= Z_HI and rate(place_ret) > 1.0   # 複勝率が水準より高く、複勝回収率も 100% 超
    ac = (" ac", ' title="勝率と単勝回収率の両方が基準超え"' if won and not placed else ' title="複勝率と複勝回収率の両方が基準超え"' if placed and not won else ' title="勝率・単勝回収率と、複勝率・複勝回収率の両方が基準超え"') if won or placed else ("", "")
    return f'<tr class="{cls}{" sm" if n < MIN_N else ""}{ac[0]}"{ac[1]}>' + "".join(cells) + "</tr>"


def html_block(S: pd.DataFrame) -> str:
    """summary() の結果を、調教師 > 場所（坂路・ウッド・ポリ）> パターン の 3 段で畳める表（章に埋め込む HTML）にする。
    調教師の行と場所の行は、下の行の合計。行のクリックで開閉、見出しのクリックで各段の中の並べ替え。"""
    S = S.copy()
    S["_w"], S["_p"] = S["単勝回収率"] * S["出走数"], S["複勝回収率"] * S["出走数"]
    S[["場所", "型"]] = S["調教パターン"].str.split("|", n=1, expand=True)
    cols = ["1着", "2着", "3着", "4着以下"]
    tot = S[cols].sum()
    base = (float(tot["1着"] / tot.sum()), float((tot["1着"] + tot["2着"] + tot["3着"]) / tot.sum()))   # 表の全組をまとめた勝率・複勝率
    out = []
    for tid, t in sorted(S.groupby("調教師ID"), key=lambda kv: -kv[1]["出走数"].sum()):
        name = str(t["調教師"].iloc[0])
        out.append(f'<tbody data-s="{_html.escape(tid)} {_html.escape(name)}">')
        out.append(_row("t", tid, name, t["出走数"].sum(), t[cols].sum().tolist(), t["_w"].sum(), t["_p"].sum(), base))
        for place, g in sorted(t.groupby("場所"), key=lambda kv: -kv[1]["出走数"].sum()):
            out.append(_row("p", tid, place, g["出走数"].sum(), g[cols].sum().tolist(), g["_w"].sum(), g["_p"].sum(), base))
            for _, r in g.sort_values("出走数", ascending=False).iterrows():
                out.append(_row("q", tid, str(r["型"]), r["出走数"], [r[k] for k in cols], r["_w"], r["_p"], base))
        out.append("</tbody>")
    heads = ["調教師ID", "調教師 / 場所 / パターン", "出走数", "着順内訳", "勝率", "単勝回収率", "複勝率", "複勝回収率"]
    head = "".join("<th>" + h + "</th>" for h in heads)
    ctl = ('<div class="tptb"><button id="tpt0">調教師のみ</button><button id="tpt1">場所まで</button><button id="tpt2">すべて開く</button>'
           '調教師ID・名前で絞り込み<input id="tptf" placeholder="例: 清水 / 01075"></div>')
    return (f"{START}\n{_CSS}\n{ctl}\n"
            f'<div id="tptw"><table id="tpt"><thead><tr>{head}</tr></thead>\n' + "\n".join(out) + f"\n</table></div>\n{_JS}\n{END}")


def inject(chapter: Path, block: str) -> None:
    """章の HTML の START〜END の間を block で置き換える（マーカーが無ければ失敗させる）。"""
    s = chapter.read_text(encoding="utf-8")
    a = s.index(START)
    b = s.index(END, a) + len(END)
    chapter.write_text(s[:a] + block + s[b:], encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/retrain_out/base_2016/article_check/final_week/trainer_pattern")
    ap.add_argument("--y0", type=int, default=2020)
    ap.add_argument("--y1", type=int, default=2025)
    ap.add_argument("--chapter", default=None, help="指定すると、この章の HTML の表（tpt:start〜tpt:end）を作り直す")
    ap.add_argument("--laps", default=str(P.LAPS))
    ap.add_argument("--table", default=str(P.TABLE))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = RO.add_returns(P.load(Path(a.laps), Path(a.table)))

    S = summary(d, a.y0, a.y1)
    S.to_csv(out / f"trainer_pattern_summary_{a.y0}_{a.y1}.csv", index=False, encoding="utf-8-sig")
    if a.chapter:
        inject(Path(a.chapter), html_block(S))
    A = asof_table(d)
    A.to_parquet(out / "trainer_pattern_asof.parquet", index=False)

    print(f"# 調教師 × 調教パターンの成績表（{a.y0}〜{a.y1}）\n")
    print(f"- 組（調教師 × パターン）: {len(S):,}（調教師 {S['調教師ID'].nunique():,} 人、パターン {S['調教パターン'].nunique()} 種）")
    for k in (5, 10, 30, 100):
        print(f"- 出走 {k} 頭以上の組: {int((S['出走数'] >= k).sum()):,}（出走の {S.loc[S['出走数'] >= k, '出走数'].sum() / S['出走数'].sum():.1%} をカバー）")
    tot = S["出走数"].sum()
    print(f"- 全体: 出走 {int(tot):,} 頭、単勝回収率 {(S['単勝回収率'] * S['出走数']).sum() / tot:.3f}、複勝回収率 {(S['複勝回収率'] * S['出走数']).sum() / tot:.3f}")
    ex = S.sort_values("出走数", ascending=False).head(15)
    print("\n## 出走数の多い組（先頭 15）\n")
    print(md(P.fmt(ex, nd=3)))

    pr = A.dropna(subset=["trpat_n"])
    print("\n## 日付つきの累計（特徴量用）\n")
    print(f"- 行数 {len(A):,}（パターンが付いた出走。{d['year'].min()}〜{d['year'].max()}）")
    print(f"- 直前までの出走が 1 頭以上ある行: {(pr['trpat_n'] > 0).mean():.1%}、10 頭以上: {(pr['trpat_n'] >= 10).mean():.1%}、30 頭以上: {(pr['trpat_n'] >= 30).mean():.1%}")
    by = pr.assign(year=pr["race_date"].str[:4]).groupby("year")["trpat_n"].agg(中央値="median", ゼロの割合=lambda s: (s == 0).mean()).round(3)
    print(md(by.reset_index()))
    print("\n注: 2016〜2019 年は調教の記録の被覆が低く（10〜66%）、累計も薄い。2020 年以降を使う場合も、各年の頭の累計は 2016 年からの積み上げ。")


if __name__ == "__main__":
    main()
