"""走法軸・脚質・展開予想の「一緒に来やすさ」の説明力の調査（pair_cohesion の結果）を、1 ファイルの HTML にする。

python -m src.scripts.docs.build_running_style_pairs_html

元: /home/jovyan/work/retrain_out/pair_cohesion/result.json（src/scripts/data/pair_cohesion.py が作る）と、
docs/operations/retrain_trials/running_style_pairs_review.json（別のエージェントのレビューの指摘と対応。あれば）。
数字はすべて元から引く（手で写さない）。結論の文は、表の値と差の区間から決まりで作る（行は群の名前で引く）。
"""
from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = Path("/home/jovyan/work/retrain_out/pair_cohesion/result.json")
REVIEW = ROOT / "docs/operations/retrain_trials/running_style_pairs_review.json"
OUT = ROOT / "docs/html/operations/running_style_pairs.html"

CSS = """:root{color-scheme:light dark;--bg:#f7f8fa;--panel:#fff;--text:#1d2330;--muted:#5b6578;--border:#d8dde6;--accent:#2f6fde;--good:#1d8a4e;--bad:#b3261e;--head:#eef1f6;--bar:#2f6fde}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;--bad:#ff8a80;--head:#1d2638;--bar:#7aa7ff}}
:root[data-theme="dark"]{--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;--bad:#ff8a80;--head:#1d2638;--bar:#7aa7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.7 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:920px;margin:0 auto;padding:16px}h1{font-size:1.4rem;margin:.2em 0}
h2{font-size:1.15rem;margin:1.8em 0 .5em;border-bottom:1px solid var(--border);padding-bottom:.2em}
.muted{color:var(--muted);font-size:.88rem}.lead{background:var(--panel);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:6px;padding:10px 14px}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.5em 0}
th,td{border:1px solid var(--border);padding:4px 6px;text-align:right;white-space:nowrap}th{background:var(--head);font-weight:600}td.l,th.l{text-align:left;white-space:normal}
code{font-size:.82rem;background:var(--head);padding:0 .3em;border-radius:3px;word-break:break-all}
svg{display:block;width:100%;height:auto}svg text{fill:var(--muted);font-size:11px}
details{margin:.5em 0}summary{cursor:pointer;color:var(--accent)}ul{padding-left:1.2em}"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def f3(x) -> str:
    return "－" if x is None else f"{x:.3f}"


def table(head, rows, left=1) -> str:
    th = "".join(f'<th class="l">{esc(h)}</th>' if i < left else f"<th>{esc(h)}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="l">{esc(c)}</td>' if i < left else f"<td>{esc(c)}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _ok(v) -> bool:
    return v is not None and v == v


def lift_rows(t: list[dict], label=None) -> list[list]:
    return [[label(x["群"]) if label else x["群"], f"{x['組の数']:,}", f"{x['両方3着内（実）'] * 100:.2f}%", f"{x['見込み（層をそろえた）'] * 100:.2f}%",
             f"{x['比']:.3f}（{x['比_lo']:.3f}〜{x['比_hi']:.3f}）" if _ok(x["比"]) else "－", f"{x['着順の差（頭数で割った平均）']:.3f}",
             f"{x['市場を除いた相関']:+.4f}" if _ok(x["市場を除いた相関"]) else "－"] for x in t]


HEAD = ["群", "組の数", "両方 3 着内（実際）", "見込み", "比（95% 区間）", "着順の差（参考）", "市場を除いた相関"]


def chart(t: list[dict], label) -> str:
    """比と区間の横棒（1.0 の線つき）。"""
    w, rh, x0 = 640, 26, 150
    lo_ = min(0.85, min(x["比_lo"] for x in t) - 0.02)
    hi_ = max(1.15, max(x["比_hi"] for x in t) + 0.02)
    sx = lambda v: x0 + (v - lo_) / (hi_ - lo_) * (w - x0 - 20)          # noqa: E731
    h = rh * len(t) + 30
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="比と区間">',
         f'<line x1="{sx(1):.1f}" y1="4" x2="{sx(1):.1f}" y2="{h - 22}" stroke="currentColor" stroke-dasharray="3 3" opacity=".5"/>',
         f'<text x="{sx(1):.1f}" y="{h - 8}" text-anchor="middle">1.0（見込みどおり）</text>']
    for k, x in enumerate(t):
        y = 8 + k * rh
        s.append(f'<text x="{x0 - 8}" y="{y + 13}" text-anchor="end">{esc(label(x["群"]))}</text>')
        s.append(f'<line x1="{sx(x["比_lo"]):.1f}" y1="{y + 9}" x2="{sx(x["比_hi"]):.1f}" y2="{y + 9}" stroke="var(--bar)" stroke-width="3"/>')
        s.append(f'<circle cx="{sx(x["比"]):.1f}" cy="{y + 9}" r="5" fill="var(--bar)"/>')
        s.append(f'<text x="{sx(x["比_hi"]) + 6:.1f}" y="{y + 13}">{x["比"]:.3f}</text>')
    s.append("</svg>")
    return "".join(s)


Q5 = lambda v: ["最も近い", "近い", "中ほど", "遠い", "最も遠い"][int(float(v))]          # noqa: E731


def row(t: list[dict], name) -> dict:
    return next(x for x in t if str(x["群"]) == str(name))


def judge(lo: float, hi: float, ref: float = 0.0) -> str:
    return "上" if lo > ref else ("下" if hi < ref else "偶然の範囲")


def render(r: dict, review: dict | None) -> str:
    T, D = r["tables"], r.get("diffs") or {}
    yrs = r["years"]
    st, st0 = T["脚質の関係|e_dec"], T["脚質の関係|較正なし（前の版）"]
    pl = T["4角の位置の差の5分位|e_dec"]
    ax = T.get("走法軸の距離の5分位（2024・TS系3軸）|e_dec") or []
    fsh = T["展開予想の向く側の頭数（スロー・ハイ予想のレースだけ）|e_dec"]
    f0 = T["展開予想の向く側の頭数|較正なし（前の版）"]
    rnd = T.get("無作為の5群（対照）|e_dec") or []
    base_corr = sum(x["市場を除いた相関"] for x in rnd) / max(len(rnd), 1)
    dsf, dpl, dfa = D["脚質 同じ−離れた"], D["4角の差 最も近い−最も遠い"], D["展開 2頭−0頭（スロー・ハイ）"]
    far, same = row(st, "離れた"), row(st, "同じ")
    # 結論（表の値と差の区間から決まりで作る）
    JW = {"上": "見込みより多い", "下": "見込みより少ない", "偶然の範囲": "見込みとの差は偶然の範囲"}
    DW = {"上": "差がある", "下": "逆向きの差がある", "偶然の範囲": "差は偶然の範囲"}
    nb = row(st, "隣")
    dnf = D["脚質 隣−離れた"]
    c1 = [f"脚質（前走までの 4 角の位置）が離れた群どうしの 2 頭は、一緒に 3 着以内に来る割合が見込みの {far['比']:.3f} 倍で、{JW[judge(far['比_lo'], far['比_hi'], 1.0)]}。",
          f"同じ群どうしは {same['比']:.3f} 倍で{JW[judge(same['比_lo'], same['比_hi'], 1.0)]}、隣の群どうしは {nb['比']:.3f} 倍で{JW[judge(nb['比_lo'], nb['比_hi'], 1.0)]}。",
          f"同じ群 − 離れた群は {dsf['diff']:+.3f}（95% 区間 {dsf['lo']:+.3f}〜{dsf['hi']:+.3f}）で{DW[judge(dsf['lo'], dsf['hi'])]}。"
          f"隣の群 − 離れた群は {dnf['diff']:+.3f}（{dnf['lo']:+.3f}〜{dnf['hi']:+.3f}）。"]
    if judge(far["比_lo"], far["比_hi"], 1.0) == "下" and judge(same["比_lo"], same["比_hi"], 1.0) != "上":
        c1.append("説明力はあるが小さく、言えるのは「脚質が大きく離れた組は来にくい」まで。「近いほど来やすい」とは言えない。")
    elif judge(dsf["lo"], dsf["hi"]) == "上":
        c1.append("説明力はあるが小さい。")
    else:
        c1.append("説明力は確かめられない。")
    fr = D["展開 2頭−0頭（スロー・ハイ・較正なし）"]
    c2 = (f"展開予想が向く側の 2 頭（スロー・ハイ予想のレースだけ）と 0 頭の差は {dfa['diff']:+.3f}（{dfa['lo']:+.3f}〜{dfa['hi']:+.3f}）で、{DW[judge(dfa['lo'], dfa['hi'])]}。"
          f"1 頭ずつの市場のずれを掛ける前は {fr['diff']:+.3f}（{fr['lo']:+.3f}〜{fr['hi']:+.3f}）で、{DW[judge(fr['lo'], fr['hi'])]}。")
    if judge(fr["lo"], fr["hi"]) == "上" and judge(dfa["lo"], dfa["hi"]) != "上":
        c2 += "掛ける前の差は、2 頭が一緒に来る分ではなく、市場が向く側の馬を 1 頭ずつ安く見ている分だった。"
    if ax:
        da = D.get("2024 走法軸 最も近い−最も遠い") or {}
        dp24 = D.get("2024 4角の差 最も近い−最も遠い") or {}
        c3 = (f"走法軸（2024 年の TS・ギアチェンジ・TS持続力の 3 軸）の近さは、最も近い組 − 最も遠い組が {da['diff']:+.3f}（{da['lo']:+.3f}〜{da['hi']:+.3f}）で、{DW[judge(da['lo'], da['hi'])]}。"
              f"同じ組の 4 角の位置の差は {dp24['diff']:+.3f}（{dp24['lo']:+.3f}〜{dp24['hi']:+.3f}）で、{DW[judge(dp24['lo'], dp24['hi'])]}。")
        if judge(da["lo"], da["hi"]) == "偶然の範囲" and da["hi"] >= dp24["diff"]:
            c3 += "走法軸の区間は広く、4 角の位置の差と同じ大きさの差も否定できない。"
        c3 += "本番の走法マップ（8 方面の地図）の近さは、2025 年以降にしか値が無く、測っていない。"
    else:
        c3 = "走法軸は、データが足りず測れない。"
    concl = [" ".join(c1), c2, c3]
    nbt = r.get("n_boot", "")
    pl_far, pl_rest = row(pl, "4.0"), [row(pl, f"{k}.0") for k in range(4)]
    pl_note = ("下がるのは最も遠い 5 分位だけで、近いほど来やすい形ではない。"
               if judge(pl_far["比_lo"], pl_far["比_hi"], 1.0) == "下" and all(judge(x["比_lo"], x["比_hi"], 1.0) != "下" for x in pl_rest) else "")
    rc_ = r.get("axis_rank_corr_with_pl")
    rc_note = (f"4 角の位置の差との順位相関は {rc_:.2f}" + ("で、脚質とは別のものを測っている。" if _ok(rc_) and abs(rc_) < 0.2 else "。")) if _ok(rc_) else ""
    o = [f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>走法の近さと一緒に来る馬</title><style>{CSS}</style></head><body><main>',
         "<h1>脚質・展開予想・走法軸（2024 年の 3 軸）が近い馬は、一緒に 3 着以内に来やすいか</h1>",
         f'<p class="muted">最終更新: {r.get("generated_at") or date.today().isoformat()}。期間 {yrs[0]}〜{yrs[-1]} 年の芝・ダート。この調査で読んだ着順は {yrs[0]}〜{yrs[-1]} 年だけ。ただし型の元のうち、展開予想は 2025 年まで、走法軸の換算の基準は 2026 年までの結果で作られている（1 節）。{r["n_races"]:,} レース・{r["n_pairs"]:,} 組。'
         "生成: <code>python -m src.scripts.docs.build_running_style_pairs_html</code>（元は <code>src/scripts/data/pair_cohesion.py</code> の結果）。"
         '関係: <a href="bet_optimization_story.html">推奨馬券の最適化の過程</a>・前の検証 <code>docs/ideas/running_style_pairs.md</code>（2025〜26 年）。</p>',
         '<div class="box lead"><b>結論</b><ol>' + "".join(f"<li>{esc(x)}</li>" for x in concl) + "</ol></div>",
         "<p>比は「2 頭とも 3 着以内」の実際の割合 ÷ 見込み。1 より大きいほど、市場が織り込んでいない分だけ一緒に来る。</p>"]
    o += ["<h2>1. 型と見込みの作り</h2><ul>",
          "<li>脚質: 位置取りモデルの特徴量 <code>h_pl_3</code>（前 3 走の 4 角の位置の平均。0 = 先頭、1 = 最後方）。<code>src/scripts/data/pos_model.py</code> が前走の行を shift して作るので、そのレースの結果は入らない。"
          "群は 逃げ（0.15 以下）・先行（0.35 以下）・差し（0.65 以下）・追込。初角の位置 <code>h_pf_w5</code>（前 5 走の加重平均）も同じ作り。2016 年は前走が少なく、前 1〜2 走だけの値も混ざる。</li>",
          "<li>展開予想: ラップ予想の確率（スロー・ミドル・ハイ。<code>retrain_out/tp55_2016/groups/LAP.parquet</code>）。年を 1 つ外して学習した予測で、その年の着順は入らない。学習に使う年は 2025 年までなので、どの年の予測にも 2025 年の結果が入る。"
          "向く側は、スロー予想なら逃げ・先行、ハイ予想なら差し・追込。ミドル予想のレースには向く側を決めないので、比べるのはスロー・ハイ予想のレースだけ。</li>",
          "<li>走法軸: <code>data/local/running_profile/axis_scores10.parquet</code>（各レースの直前の時点の 10 点。<code>build_axis_scores10.py</code>）。元の実測ラップが 2023 年の終わりからしか無く、2023 年まではほぼ空。"
          "2024 年も位置系の軸は 1 割ほどしか無いので、TS・ギアチェンジ・TS持続力が 2 頭ともそろう組だけで測った。先読みの注意: 父・全馬の平均への引き寄せが全期間の値で、10 点への換算の基準も全期間の好走から作る（軸ごとの尺度が変わるので、3 軸の距離の重みに結果が入る）。</li>",
          "<li>見込み: 市場の単勝オッズ（確定）の確率から、Harville（各馬が順に入る形）で「2 頭とも 3 着以内」の確率を正確に計算した。"
          "次に、1 頭ずつの市場のずれ（3 着以内の実際 ÷ 見込みを、ペースの予想 × 脚質 × 人気帯ごとに、その年を外した年で作った値）を 2 頭分掛けた。"
          "最後に、その確率の 10 分位ごとの全体の比で割った（全体の比は 1.000 になる）。</li>",
          "<li>1 頭のずれを掛けるのは、脚質ごとに市場が 1 頭単位でずれる分（例: 追込は市場の見込みより多く 3 着以内に来る）を、2 頭が一緒に来る分と混ぜないため。掛ける前の値も並べた。</li>",
          f"<li>区間はレースで束ねたブートストラップ（{nbt} 回）。層の比も回ごとに作り直す。差の区間も同じ重みで作った。市場を除いた相関は 2 頭の「3 着以内か − 見込み」の相関で、無作為の群でも平均 {base_corr:+.4f} ある（これが基準）。</li></ul>"]
    o += [f"<h2>2. 脚質（{yrs[0]}〜{yrs[-1]} 年）</h2>", chart(st, str), table(HEAD, lift_rows(st)),
          "<p>1 頭のずれを掛ける前（前の版）</p>", table(HEAD, lift_rows(st0)),
          "<p>4 角の位置の差の 5 分位</p>", chart(pl, Q5), table(HEAD, lift_rows(pl, Q5)),
          f'<p class="muted">最も近い − 最も遠い: {dpl["diff"]:+.3f}（{dpl["lo"]:+.3f}〜{dpl["hi"]:+.3f}）。{pl_note}</p>',
          "<details><summary>初角の位置の差の 5 分位</summary>" + table(HEAD, lift_rows(T["初角の位置の差の5分位|e_dec"], Q5)) + "</details>"]
    fsh0 = T.get("展開予想の向く側の頭数（スロー・ハイ予想のレースだけ）|較正なし（前の版）") or []
    o += ["<h2>3. 展開予想</h2>", "<p>向く側の頭数（スロー・ハイ予想のレースだけ。脚質の分かる馬の組）</p>", table(HEAD, lift_rows(fsh, lambda v: f"{v} 頭")),
          "<p>1 頭のずれを掛ける前（同じレース）</p>", table(HEAD, lift_rows(fsh0, lambda v: f"{v} 頭")),
          "<details><summary>1 頭のずれを掛ける前（ミドル予想を含む全レース）</summary>" + table(HEAD, lift_rows(f0, lambda v: f"{v} 頭")) + "</details>"]
    o += ["<h2>4. 走法軸（2024 年・TS 系 3 軸）と、同じ組での脚質</h2>"]
    if ax:
        n = r.get("axis_note") or {}
        o += [chart(ax, Q5), table(HEAD, lift_rows(ax, Q5)),
              f'<p class="muted">{n.get("n_races", 0):,} レース・{n.get("n_pairs", 0):,} 組。近さは 3 軸の 10 点の差の距離。{rc_note}</p>',
              "<p>同じ組（2024 年）の脚質の関係と、4 角の位置の差</p>",
              table(HEAD, lift_rows(T.get("同じ組（2024）での比べ: 脚質の関係") or [])),
              table(HEAD, lift_rows(T.get("同じ組（2024）での比べ: 4角の位置の差の5分位") or [], Q5))]
        xs = T.get("走法軸の距離の5分位×脚質の関係（2024）") or []
        if xs:
            o.append("<details><summary>脚質の関係ごとの走法軸の近さ（脚質を超える上乗せ）</summary>"
                     + table(["脚質の関係"] + HEAD, [[x["脚質の関係"]] + lift_rows([x], Q5)[0] for x in xs], left=2) + "</details>")
    o += ["<h2>5. 対照</h2>",
          "<p>無作為の 5 群（組をレースと関係なく 5 つに分けた。どれも 1.0 前後なら、比の作りに偏りが無い）</p>",
          table(HEAD, lift_rows(rnd, lambda v: f"群 {v}")),
          "<details><summary>人気帯の関係・人気帯の組でそろえた脚質の関係</summary>"
          "<p>2 頭の人気帯の関係。差は Harville の残りの偏り（確率の水準でそろえても、組の中身で残る分）で、人気が近いことで一緒に来る分ではない。</p>"
          + table(HEAD, lift_rows(T.get("人気帯の関係（対照）|e_dec") or []))
          + table(HEAD, lift_rows(T["脚質の関係|pbpair"]))
          + table(["人気帯の関係"] + HEAD, [[x["人気帯の関係"]] + lift_rows([x])[0] for x in (T.get("脚質の関係×人気帯の関係|e_dec") or [])], left=2) + "</details>"]
    o.append("<h2>6. 条件別・年ごと（脚質の関係）</h2><p class=\"muted\">多くの層を並べている。見るのは、離れた群が下がる向きが層をまたいでそろうかだけ。1 つの層の差では読まない。</p>")
    names = {"yband": "年", "fband": "頭数", "surface": "芝・ダート", "dband": "距離", "paceL": "ペースの予想", "goingL": "馬場"}
    det = []
    for k, nm in names.items():
        rows = T.get(f"条件別の脚質の関係|{k}") or []
        det.append(f"<p>{esc(nm)}</p>" + table([nm, "脚質の関係", "組の数", "比（95% 区間）", "市場を除いた相関"],
                                              [[x["条件"], x["群"], f"{x['組の数']:,}", f"{x['比']:.3f}（{x['比_lo']:.3f}〜{x['比_hi']:.3f}）" if _ok(x["比"]) else "－",
                                                f"{x['市場を除いた相関']:+.4f}" if _ok(x["市場を除いた相関"]) else "－"] for x in rows], left=2))
    o.append("<details><summary>条件別の表</summary>" + "".join(det) + "</details>")
    o += ["<h2>7. 限界</h2><ul>",
          "<li>本番の走法マップ（8 方面の地図の近さ）は測っていない。値が 2025 年以降にしか無いため。</li>",
          "<li>本番の買い目の補正 <code>STYLE_PAIR_BOOST</code> の元は、前の検証（2025〜26 年）。その比は 1 頭のずれを掛けない作りなので、大きめに出ているおそれがある。測り直すなら、1 頭のずれを掛けた見込みで行う。</li>",
          "<li>走法軸は 2024 年の TS 系 3 軸だけ。位置系の軸を含めた近さは測れていない。</li>",
          "<li>見込みは市場の確定オッズから作った。比は、市場とこの見込みの作りが織り込んでいない分を表す。組の券種の払戻はそれを織り込んでいることがあり、回収率は測っていない。</li>",
          "<li>見たのは「2 頭とも 3 着以内か」。着順の差は市場で補正していない参考の値。</li>",
          "<li>2016 年は前走が少ない。同着・競走中止の扱いは近似（影響は小さい）。</li></ul>"]
    o.append("<h2>8. 別のエージェントのレビューの指摘と対応</h2>")
    if review and review.get("items"):
        o.append(table(["指摘", "重さ", "対応"], [[x.get("finding"), x.get("severity"), x.get("response")] for x in review["items"]], left=3))
    o.append("</main></body></html>")
    return "\n".join(o)


def main() -> None:
    r = json.loads(SRC.read_text(encoding="utf-8"))
    rv = json.loads(REVIEW.read_text(encoding="utf-8")) if REVIEW.exists() else None
    OUT.write_text(render(r, rv), encoding="utf-8")
    print("書いた", OUT)


if __name__ == "__main__":
    main()
