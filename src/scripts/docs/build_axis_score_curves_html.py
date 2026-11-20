"""走法軸の点数（1〜10）ごとの勝率・複勝率を、対象（クラス・馬場・距離・コース）× 走法軸のマトリクスで見る専用 HTML を作る（読むだけ・HTML を書く）。

入力: src/scripts/data/axis_score_curves.py の出力 JSON。
使い方: python src/scripts/docs/build_axis_score_curves_html.py CURVES.json [OUT.html|- [TS.json]]
  TS.json = src/scripts/data/ts_course_resolution.py の出力（6 節の TS の分布。無ければその部分は出さない）。軸の定義の最終更新は手元の git から作る。
"""
from __future__ import annotations

import html
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.scripts.docs.build_course_axis_demand_html import CSS, SORT_JS  # noqa: E402

OUT = REPO / "docs/html/operations/running_axis_curves.html"
MIN_N = 30            # 実測で線を引く最低の走数（これ未満の区間は推定で埋める）（JS の MINN と同じ）
MIN_SIDE = 100        # 高い側・低い側の走数がこれ未満なら差を出さない
Z_MARK = 2.6          # 印を付ける z（軸 × 対象で数百の組を見るので 1.96 より厳しめ）
SKEW_MARK = 1.0       # TS の歪みの絶対値がこれ以上のコースを強調する
VENUE_ORDER = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
SURF_KEY = {"芝": "芝", "ダ": "ダ", "ダート": "ダ"}
ONLY_SURFACE = {"wett": "芝", "wetd": "ダ"}          # 道悪適性は片方の馬場だけ（axis_score_curves.py の ONLY_SURFACE と同じ）
COURSE_RE = re.compile(r"^(.+?)(芝|ダ)(\d+)m$")
RUN_AXES = "src/research/race/run_axes.py"
# 軸 → 定義のコード（ファイル, 関数）。モジュールの定数や 10 点換算（axis_score10.py）の変更は拾えない。endure・front・cr・makuri・slope は 1 つの関数 axes_from_frame の中で作られる（関数単位でも軸ごとには分けられない）
AXIS_DEF = {
    "ts": ("src/research/race/horse_ts.py", "course_ts"), "sustain": (RUN_AXES, "_sustain_one"), "debut": (RUN_AXES, "_debut"), "gear": (RUN_AXES, "_gear_one"),
    "endure": (RUN_AXES, "axes_from_frame"), "front": (RUN_AXES, "axes_from_frame"), "cr": (RUN_AXES, "axes_from_frame"),
    "makuri": (RUN_AXES, "axes_from_frame"), "slope": (RUN_AXES, "axes_from_frame"),
    "basic": ("src/research/race/basic_speed.py", "compute"),
    "wett": ("src/research/race/wet_going.py", "compute"), "wetd": ("src/research/race/wet_going.py", "compute"),
}
GIT_FMT = "%cs %h %s"
LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) ([0-9a-f]{4,40}) (.*)$")
EXTRA_CSS = """
.cgw { overflow:auto; border:1px solid var(--border); border-radius:10px; margin:8px 0; }
table.cg { border-collapse:separate; border-spacing:0; margin:0; width:auto; font-size:12px; }
table.cg th, table.cg td { border:0; border-right:1px solid var(--border); border-bottom:1px solid var(--border); padding:4px 6px; }
table.cg thead th { background:var(--panel); text-align:center; position:sticky; top:0; z-index:2; }
table.cg th.rh { position:sticky; left:0; z-index:1; background:var(--panel); min-width:54px; font-weight:600; color:var(--text); }
table.cg thead th.rh { z-index:3; }
table.cg td.hm { min-width:74px; line-height:1.3; } table.cg td.hm b { font-size:14px; } table.cg td.hm small { display:block; font-size:10.5px; color:var(--muted); font-weight:400; }
table.cg td.hm[data-a], table.cg td.hm[data-t] { cursor:pointer; } table.cg td.hm[data-a]:hover, table.cg td.hm[data-t]:hover { outline:2px solid var(--accent); outline-offset:-2px; }
table.cg td.hm.sel { outline:2px solid var(--warn); outline-offset:-2px; }
table.cg td.hm.sk { background:rgba(248,113,113,.30); }
.defn { font-size:12px; font-weight:400; color:var(--muted); } .defn .strong { color:var(--warn); }
.cgn { color:var(--muted); font-size:13px; margin:2px 0 10px; }
details.fl { margin:8px 0; } details.fl summary { cursor:pointer; color:var(--accent); font-size:13px; }
.ctl { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin:10px 0; } .ctl label { font-size:13px; color:var(--muted); }
.ctl button { background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:5px 12px; font-size:13px; cursor:pointer; }
.ctl button.on { background:var(--accent); color:#0b1428; font-weight:700; }
.mxwrap { overflow:auto; max-height:82vh; border:1px solid var(--border); border-radius:10px; }
table.mx { border-collapse:separate; border-spacing:0; margin:0; width:auto; font-size:12px; }
table.mx th, table.mx td { border:0; border-right:1px solid var(--border); border-bottom:1px solid var(--border); padding:4px 5px; vertical-align:top; background:var(--bg); }
table.mx thead th { position:sticky; top:0; z-index:3; background:var(--panel); white-space:normal; min-width:178px; text-align:center; }
table.mx th.rh { position:sticky; left:0; z-index:2; background:var(--panel); min-width:132px; max-width:150px; white-space:normal; text-align:left; font-weight:600; color:var(--text); }
table.mx thead th.rh { z-index:4; }
table.mx td { cursor:pointer; } table.mx td:hover { outline:2px solid var(--accent); outline-offset:-2px; }
table.mx td.sel { outline:2px solid var(--warn); outline-offset:-2px; }
.rh small, th small { display:block; color:var(--muted); font-weight:400; font-size:11px; line-height:1.35; }
.cap { font-size:10.5px; color:var(--muted); line-height:1.35; text-align:center; } .cap b { color:var(--text); }
.big { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:10px 14px; margin:12px 0; }
.big svg { width:100%; max-width:880px; height:auto; display:block; } .big table { width:auto; }
.lg span { margin-right:16px; font-size:12px; color:var(--muted); } .lg i { display:inline-block; width:18px; height:0; border-top:3px solid; vertical-align:middle; margin-right:5px; } .lg i.d { border-top-style:dashed; }
td.hm { text-align:center; font-variant-numeric:tabular-nums; min-width:62px; } td.hm.k { font-weight:700; outline:1px solid var(--text); outline-offset:-1px; }
"""


def side_stats(arr: list, lo: tuple[int, int] = (0, 4), hi: tuple[int, int] = (5, 10)) -> dict | None:
    """arr = [n, w, p, ew, ep]（各 10 区間）。低（4.5 点以下）と高（6.0 点以上）の複勝率・人気補正後の差。"""
    def agg(a: tuple[int, int]) -> tuple[float, float, float]:
        n = sum(arr[0][a[0]:a[1]])
        return n, sum(arr[2][a[0]:a[1]]), sum(arr[4][a[0]:a[1]])
    nl, pl, el = agg(lo)
    nh, ph, eh = agg(hi)
    if nl < MIN_SIDE or nh < MIN_SIDE:
        return None
    rl, rh = pl / nl, ph / nh
    se = math.sqrt(rl * (1 - rl) / nl + rh * (1 - rh) / nh)
    d_adj = (ph - eh) / nh - (pl - el) / nl
    return {"nl": nl, "nh": nh, "raw": rh - rl, "adj": d_adj, "z": d_adj / se if se > 0 else 0.0}


def parse_course_key(key: str) -> tuple[str, str, int] | None:
    """「京都芝1600m」→ (京都, 芝, 1600)。ダートは「ダ」。読めなければ None。"""
    m = COURSE_RE.match(key)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def build_course_grid(entries: list[tuple[str, str, int, int]]) -> dict:
    """(場, 芝/ダ, 距離, 添字) の並びから、馬場ごとの 縦 = 競馬場・横 = 距離 の格子を作る。
    競馬場は VENUE_ORDER の順（知らない場は末尾）、距離はその馬場に出る距離の和集合を昇順。全軸で同じ列順になる。idx['場|距離'] = 添字。"""
    out = {"芝": {"venues": [], "dists": [], "idx": {}}, "ダ": {"venues": [], "dists": [], "idx": {}}}
    for venue, surf, dist, i in entries:
        g = out[SURF_KEY[surf]]
        g["idx"][f"{venue}|{dist}"] = i
        if venue not in g["venues"]:
            g["venues"].append(venue)
        if dist not in g["dists"]:
            g["dists"].append(dist)
    for g in out.values():
        g["venues"].sort(key=lambda v: (VENUE_ORDER.index(v) if v in VENUE_ORDER else len(VENUE_ORDER), v))
        g["dists"].sort()
    return out


def axis_course_grid(d: dict) -> dict:
    ent = []
    for i, g in enumerate(d["dims"].get("course", {}).get("groups", [])):
        p = parse_course_key(g["key"])
        if p:
            ent.append((p[0], p[1], p[2], i))
    return build_course_grid(ent)


def ts_course_grid(ts: dict) -> dict:
    return build_course_grid([(c["venue"], c["surface"], int(c["distance"]), i) for i, c in enumerate(ts["courses"])])


def run_git(args: list[str]) -> str | None:
    """git を読むだけで呼ぶ。使えない・失敗したときは None（生成は止めない）。"""
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def parse_log_line(out: str | None) -> dict | None:
    """「2026-11-12 abc1234 件名」の最初の 1 行を読む。"""
    for line in (out or "").splitlines():
        m = LOG_LINE.match(line.strip())
        if m:
            return {"date": m.group(1), "hash": m.group(2), "subject": m.group(3)}
    return None


def last_change(file: str, func: str | None, runner=run_git) -> dict | None:
    """定義のコードの最終コミット（日付・短縮ハッシュ・件名）。関数単位（git log -L）が取れなければファイル単位。どちらも取れなければ None。"""
    if func:
        r = parse_log_line(runner(["log", "-1", f"--format={GIT_FMT}", "-s", "-L", f":{func}:{file}"]))
        if r:
            return {**r, "grain": "関数", "file": file, "func": func}
    r = parse_log_line(runner(["log", "-1", f"--format={GIT_FMT}", "--", file]))
    return {**r, "grain": "ファイル", "file": file, "func": None} if r else None


def axis_defs(keys: list[str], runner=run_git) -> dict[str, dict | None]:
    """軸ごとの 定義の最終更新。浅いクローンは日付が当てにならないので全部「不明」にする。"""
    if (runner(["rev-parse", "--is-shallow-repository"]) or "").strip() == "true":
        return {k: None for k in keys}
    return {k: (last_change(*AXIS_DEF[k], runner=runner) if k in AXIS_DEF else None) for k in keys}


def def_note_html(info: dict | None, score_date: str) -> str:
    """見出し・一覧に出す「定義の最終更新 日付（ハッシュ: 件名）」。定義の更新が点の作成日より後なら「点が未更新の可能性」の印。"""
    if not info:
        return "定義の最終更新 不明（git なし）"
    subj = info["subject"] if len(info["subject"]) <= 44 else info["subject"][:43] + "…"
    where = f"{info['file'].rsplit('/', 1)[-1]}::{info['func']}" if info["func"] else info["file"].rsplit("/", 1)[-1]
    stale = ' <span class="strong">点が未更新の可能性</span>' if score_date and info["date"] > score_date else ""
    return (f"定義の最終更新 {info['date']}（<span title=\"{html.escape(info['subject'])}\">{info['hash']}: {html.escape(subj)}</span>）"
            f" <span title=\"根拠の粒度\">[{info['grain']}単位 {html.escape(where)}]</span>{stale}")


def _num(v: float | None, nd: int = 2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def ts_section(ts: dict, note: str = "") -> str:
    """コース別の TS 分布の節の静的な部分（説明・距離帯の表・コース一覧）。格子とヒストグラムは JS が描く。"""
    meta = ts["meta"]
    bands, courses = ts["bands"], ts["courses"]
    pk = meta.get("hist", {}).get("peak_outlier", 20.0)
    n_out_bands = sum(b.get("peak20", 0) for b in bands)
    out_courses = [c for c in courses if c.get("peak20", 0) > 0]
    out_txt = "、".join(f"{c['venue']}{'芝' if SURF_KEY[c['surface']] == '芝' else 'ダ'}{c['distance']}m（{c['peak20']} 走・歪み {c['ts_skew']:+.1f}）" for c in sorted(out_courses, key=lambda c: -c["peak20"]))

    def band_row(b: dict) -> str:
        r = b.get("rel") or {}
        rho = f"{r['rho_split']:.2f} [{r['rho_split_ci'][0]:.2f}, {r['rho_split_ci'][1]:.2f}]" if r else "—"
        rv = r["rho_split"] if r else -1
        sk = ' class="num strong"' if abs(b["ts_skew"]) >= SKEW_MARK else ' class="num"'
        return (f"<tr><td>{html.escape(b['surface'])}</td><td data-v=\"{DIST_ORDER_IDX.get(b['band'], 9)}\">{html.escape(b['band'])}</td><td class=\"num\">{b['n']:,}</td><td class=\"num\">{b['races']:,}</td>"
                f"<td class=\"num\">{b['ts_sd']:.2f}</td><td{sk}>{b['ts_skew']:+.2f}</td><td class=\"num\">{b['ts_p05']:.1f}</td><td class=\"num\">{b['ts_p50']:.1f}</td><td class=\"num\">{b['ts_p95']:.1f}</td>"
                f"<td class=\"num\" data-v=\"{rv:.4f}\">{rho}</td><td class=\"num\">{b['peak_mean']:.2f}</td><td class=\"num\">{b['front115'] * 100:.1f}%</td><td class=\"num\">{b['peak20']}</td></tr>")

    band_rows = "".join(band_row(b) for b in bands)
    srt = sorted(courses, key=lambda c: (0 if SURF_KEY[c["surface"]] == "芝" else 1, VENUE_ORDER.index(c["venue"]) if c["venue"] in VENUE_ORDER else 99, c["distance"]))
    course_rows = "".join(
        f"<tr><td>{html.escape(c['venue'])}</td><td>{html.escape(c['surface'])}</td><td class=\"num\">{c['distance']}</td><td class=\"num\">{c['n']:,}</td>"
        f"<td class=\"num\">{c['ts_sd']:.2f}</td><td class=\"num{' strong' if abs(c['ts_skew']) >= SKEW_MARK else ''}\">{c['ts_skew']:+.2f}</td>"
        f"<td class=\"num\">{c['ts_p05']:.1f}</td><td class=\"num\">{c['ts_p50']:.1f}</td><td class=\"num\">{c['ts_p95']:.1f}</td>"
        f"<td class=\"num\">{c['ts_min']:.1f}</td><td class=\"num\">{c['ts_max']:.1f}</td><td class=\"num\">{c['peak20']}</td></tr>"
        for c in srt)
    rel = ts.get("all_rel") or {}
    rel_txt = f"全体の 1 走の再現性 ρ = {rel['rho_split']:.2f}（区間 {rel['rho_split_ci'][0]:.2f}〜{rel['rho_split_ci'][1]:.2f}）。" if rel else ""
    return f"""<h3 id="ts-dist">TS（ラスト 3F の最速ラップの偏差値）の分布{' <span class="defn">— ' + note + '</span>' if note else ''}</h3>
<p class="sub">ここの「TS」は、走法軸の点数（上の「TS」の軸）ではなく、<b>各走の偏差値そのもの</b>。定義: ラスト 3F（最後の 3 区間）のうち最速のラップを、レース水準のずれで補正した値 x に直し、<b>コース（場 × 芝ダ × 距離）ごとの平均・標準偏差</b>（走数が少ないコースは芝ダ × 距離帯の値へ縮める）で <code>ts = 50 − 10 × (x_w − 平均) ÷ 標準偏差</code> にしたもの（速いほど高い。x_w は x をコースの 0.5%・99.5% 点で打ち切った値で、平均・標準偏差も x_w から出す）。
集計期間 {meta['date_from']}〜{meta['date_to']}（{meta['runs']:,} 走・{meta['races']:,} レース。最速ラップが取れた平地の走）。{rel_txt}作成: {meta['built']}。</p>
<p class="sub"><b>2026-11-20 から、TS の x はコースの 0.5%・99.5% 点で打ち切ってから偏差値にしている</b>（走数 300 未満のコースは、同じ芝ダ・距離帯の標準化した値の分位点をそのコースの平均・標準偏差に戻して使う）。以前は、最速ラップが {pk:.0f} 秒以上の異常走（中止・歩様の乱れなど）が、そのコースの標準偏差の推定を膨らませ、ほかの走の TS の広がりを狭めていた。打ち切り後は、コースの標準偏差の最小が 8.5 前後まで戻り、歪みの絶対値が 1 以上のコースは大きく減った。異常走そのものの TS は打ち切らずに残るので、最小値と {pk:.0f} 秒以上の走数にはまだ出る（距離帯の表に入る走のうち {n_out_bands} 走{(": " + html.escape(out_txt)) if out_txt else ""}）。この表・図は打ち切り後の TS。</p>
<div class="scroll"><table class="sortable"><thead><tr><th>芝ダ</th><th>距離帯</th><th>走数</th><th>レース数</th><th>TS の標準偏差</th><th>歪み</th><th>5%点</th><th>50%点</th><th>95%点</th><th>1 走の再現性 ρ [95%区間]</th><th>最速ラップ平均 (秒)</th><th>11.5 秒以内</th><th>{pk:.0f} 秒以上の走数</th></tr></thead><tbody>{band_rows}</tbody></table></div>
<p class="sub">標準偏差が 10 に近いほど TS は設計どおりの広がり。歪み（絶対値 {SKEW_MARK:g} 以上を強調）が負に大きいのは、遅い側に異常走が長く伸びているため。再現性 ρ = 同じ芝ダ・距離帯を走った馬の古い順 4 走を (1,3) と (2,4) に分けた平均の相関 → Spearman-Brown → 1 走に換算（馬単位のブートストラップ）。区間が空欄（—）は標本が 150 頭未満。</p>
<div id="cg-ts"></div>
<details class="fl"><summary>コースごとの TS の分布（一覧・列で並べ替え。{len(courses)} コース）</summary>
<div class="scroll"><table class="sortable"><thead><tr><th>場</th><th>芝ダ</th><th>距離</th><th>走数</th><th>標準偏差</th><th>歪み</th><th>5%点</th><th>50%点</th><th>95%点</th><th>最小</th><th>最大</th><th>{pk:.0f} 秒以上の走数</th></tr></thead><tbody>{course_rows}</tbody></table></div>
</details>"""


def fmt_pt(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.1f}"


DIST_ORDER_IDX = {b: i for i, b in enumerate(["〜1200m", "1300〜1400m", "1500〜1700m", "1800〜2000m", "2100〜2400m", "2500m〜"])}


def build(d: dict, ts: dict | None = None, defs: dict | None = None) -> str:
    """defs = axis_defs() の結果（軸 → 定義の最終更新）。None なら日付は出さない。"""
    meta = d["meta"]
    score_date = meta["history_mtime"][:10]
    notes = {a["key"]: def_note_html((defs or {}).get(a["key"]), score_date) for a in d["axes"]} if defs is not None else {}
    cg = {"axg": axis_course_grid(d), "only": ONLY_SURFACE, "defn": notes, "skm": SKEW_MARK,
          "ts": ({"grid": ts_course_grid(ts), "courses": [
              {"v": c["venue"], "s": SURF_KEY[c["surface"]], "d": c["distance"], "n": c["n"], "sd": c["ts_sd"], "sk": c["ts_skew"], "q": [c["ts_p05"], c["ts_p25"], c["ts_p50"], c["ts_p75"], c["ts_p95"]],
               "mn": c["ts_min"], "mx": c["ts_max"], "h": c["ts_hist"], "o": c["peak20"]} for c in ts["courses"]],
                  "hist": ts["meta"]["hist"]} if ts else None)}
    cg_json = json.dumps(cg, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    g_all = d["dims"]["all"]["groups"][0]["data"]
    tr = []
    for ax in d["axes"]:
        c = g_all.get(ax["key"])
        if not c:
            continue
        a, b = side_stats(c["A"]), side_stats(c["B"])
        mark = ' class="strong"' if a and abs(a["z"]) >= Z_MARK else ""
        tr.append(
            f"<tr><td>{html.escape(ax['label'])}</td><td class=\"num\">{sum(c['A'][0]):,}</td><td class=\"num\">{sum(c['B'][0]):,}</td>"
            f"<td class=\"num\">{fmt_pt(a['raw'] if a else None)}</td><td class=\"num\"{mark}>{fmt_pt(a['adj'] if a else None)}</td><td class=\"num\">{('%+.1f' % a['z']) if a else '—'}</td>"
            f"<td class=\"num\">{fmt_pt(b['raw'] if b else None)}</td><td class=\"num\">{fmt_pt(b['adj'] if b else None)}</td></tr>")
    axis_rows = "".join(
        f"<tr><td>{html.escape(a['label'])}</td><td>{html.escape(a['high'])}</td><td>{html.escape(a['low'])}</td>"
        f"<td class=\"num\">{meta['axis_runs'][a['key']]['A']:,}</td><td class=\"num\">{meta['axis_runs'][a['key']]['B']:,}</td>"
        + (f"<td>{notes[a['key']]}</td>" if defs is not None else "") + "</tr>"
        for a in d["axes"])
    def_th = "<th>定義の最終更新</th>" if defs is not None else ""
    def_p = (f'<p class="sub"><b>点の作成日 {html.escape(score_date)}</b>（<code>axis_scores10.parquet</code> の更新日）。「定義の最終更新」は、軸を作るコードの<b>最終コミット日</b>で、意味の変更とは限らない（コメント・整形・テストの修正でも更新される）。'
             "根拠の粒度: 「関数単位」= その軸を作る関数の最終コミット（<code>git log -L</code>）。Hペース耐性・先行力・Cr性能・まくり特性・急坂適性は 1 つの関数（<code>axes_from_frame</code>）の中で作られるので同じ日付になる。"
             "<b>定義の最終更新が点の作成日より後の軸には「点が未更新の可能性」の印</b>（その後に点が作り直されていなければ、ここの点は古い定義のまま）。</p>") if defs is not None else ""
    n_marked = 0
    n_all = 0
    for dim in d["dims"].values():
        for g in dim["groups"]:
            for ax in d["axes"]:
                c = g["data"].get(ax["key"])
                s = side_stats(c["A"]) if c else None
                if s:
                    n_all += 1
                    n_marked += abs(s["z"]) >= Z_MARK
    chance = n_all * math.erfc(Z_MARK / math.sqrt(2))      # |z|>=Z_MARK の両側確率 × 組数
    bonf = NormalDist().inv_cdf(1 - 0.025 / max(n_all, 1))   # 全組で 5% に抑える線（Bonferroni）
    ts_part = ts_section(ts, notes.get("ts", "")) if ts else ""
    data_json = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>走法軸の点数別 勝率・複勝率（軸 × 対象のマトリクス） — keiba-vpn docs</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>走法軸の点数別 勝率・複勝率（軸 × 対象のマトリクス）</h1>
<p class="muted">作成: {meta['built']}。走法マップの 10 点満点の走法軸（出走前の点）ごとに、横軸 = 点数、縦軸 = その点数の馬の<b>勝率（左軸）</b>と<b>複勝率（右軸）</b>を、
対象（クラス・馬場・距離・コース）ごとに並べたチャート集。集計は <code>src/scripts/data/axis_score_curves.py</code>（ホストのローカルの表だけ。GCS・API は使わない）、この HTML は <code>src/scripts/docs/build_axis_score_curves_html.py</code> が作る。
費用: 集計は一度きりのホスト内の計算で、周期処理・GCS 読み出しは増えない（GCP・Cloudflare の課金に影響なし）。</p>

<h2>1. 読み方</h2>
<ul>
<li><b>点数</b>: 各走の<b>直前までの走</b>で作った走法軸の点（1〜10、0.5 刻み。画面の走法マップと同じ）。点の整数部で 10 区間に分けた（「5」の区間 = 5.0〜5.5 点）。測れない軸は欠測で、その走は数えない。</li>
<li><b>勝率</b> = 1 着の割合。<b>複勝率</b> = 3 着以内（8 頭以上）／2 着以内（5〜7 頭）の割合。5 頭未満・障害・取消除外などは除いた。</li>
<li>チャート: 青の実線・丸 = 勝率（左軸）、橙の実線・四角 = 複勝率（右軸）。<b>右軸は左軸の 3 倍</b>に固定してあるので、全体の平均（点線）が近い高さに並ぶ。灰色の棒 = 走数。走数 {MIN_N} 未満の区間は、<b>推定値</b>で埋めて 1〜10 点の全ての点を描く（走数 {MIN_N} 以上の区間が 3 つ未満の対象は、推定できないので従来どおり欠ける）（色枠で中が暗い印 + 点線でつなぐ。実測の区間は塗りつぶしの印 + 実線）。推定は、走数 {MIN_N} 以上の区間（3 区間以上あるとき）に、点数に対する 1 次のロジスティック曲線を走数で重み付けして当てはめ、その曲線に実測を {MIN_N} 走ぶんの重みで混ぜた値（走数 0 の区間は曲線そのもの。曲線は実測の区間の最小〜最大の範囲に収めた。人気から期待される率の破線は、平均なので走数が少なくても実測のままで、走数 0 の区間だけ補う）。<b>外側への延長なので、実際の値を保証しない</b>。信頼区間の帯は実測の区間だけ。</li>
<li><b>薄い破線 = 人気補正の目安</b>: 同じ頭数・同じ人気順の馬が、{meta['exp_years'][0]}〜{meta['exp_years'][1]} 年に平均して取っていた勝率・複勝率の、その区間での平均。
走法軸の点が高い馬ほど人気も高い（Hペース耐性は実力の代理でもある）ので、<b>実線が破線を上回る分</b>が「人気（市場が見ている実力）では説明できない上乗せ」の目安。実線だけを見ると実力の差が混ざる。</li>
<li>期間: 実線 = {meta['periods']['A'][0]}〜{meta['periods']['A'][1]}（走数 {meta['runs']['A']:,}）。「2026 年を重ねる」を押すと、{meta['periods']['B'][0]}〜（走数 {meta['runs']['B']:,}。テストセット②。<b>確認だけで、選択には使わない</b>）を白抜きの点で重ねる。</li>
<li>各マスの下の数字 = <b>高 − 低</b>（6.0 点以上の複勝率 − 4.5 点以下の複勝率、ポイント。5.0・5.5 点は除く）と、<b>補正後</b>（同じ差を、人気から期待される複勝率を引いてから取ったもの）。マスをクリックすると、下の拡大図に信頼区間（Wilson 95%）と数表が出る。</li>
<li><b>コース別</b>（場 × 芝ダ × 距離。{meta['course_min_runs']} 走以上のコース）は、70 列ほどになって読みにくいので、3・5 節のボタンからは外し、<b>6 節に「走法軸ごと：縦 = 競馬場 × 横 = 距離」の表</b>として分けた。</li>
</ul>

<h2>2. 走法軸の一覧と走数</h2>
<div class="scroll"><table><thead><tr><th>走法軸</th><th>点が高い</th><th>点が低い</th><th>走数 2024-25</th><th>走数 2026</th>{def_th}</tr></thead><tbody>{axis_rows}</tbody></table></div>
<p class="sub">走数 = その軸の点が測れた走の数。位置取り由来の軸（Hペース耐性・先行力・Cr性能・まくり特性）は 2024 年の位置データが薄いので、2025 年中心。
道悪適性（芝）は芝の走だけ、道悪適性（ダート）はダートの走だけ。急坂適性は、中山・阪神での過去走から作った点（どのコースの走にも付く）。<b>基礎スピード・道悪適性は、走法マップの良馬場／道悪の地図の軸で、馬場状態が点の計算に入る</b>。</p>

<h2>3. マトリクス</h2>
<div class="ctl" id="ctl"></div>
<div class="big" id="big"><span class="muted">下のマスをクリックすると、ここに拡大図と数表が出る。</span></div>
<div class="mxwrap"><table class="mx" id="mx"></table></div>
<p class="sub">対象のボタンは、全体・クラス・馬場・距離。コース別は 6 節（場 × 距離の表）。</p>
<p class="sub">左軸（勝率）の目盛は、この対象の中で走数 200 以上の区間の最大値（複勝率の 1/3 も含む）で共通（マス同士の高さを比べられる）。上限を超える区間は上端で切る（走数の少ない区間で起きる）。</p>

<h2>4. 軸ごとの要約（全体・複勝率の高 − 低）</h2>
<div class="scroll"><table><thead><tr><th>走法軸</th><th>走数 2024-25</th><th>走数 2026</th><th>高−低 (pt) 2024-25</th><th>補正後 (pt) 2024-25</th><th>z</th><th>高−低 (pt) 2026</th><th>補正後 (pt) 2026</th></tr></thead><tbody>{''.join(tr)}</tbody></table></div>
<p class="sub">「高−低」= 6.0 点以上の複勝率 − 4.5 点以下の複勝率（5.0・5.5 点は除く。区間は整数部なので、低い側は 1〜4 の区間、高い側は 6〜10 の区間）。「補正後」は人気から期待される複勝率を引いた差（人気では説明できない分）。z = 補正後の差 ÷ 標準誤差（期待値は固定として扱う）。</p>
<p class="sub">軸 × 対象の全 {n_all} 組（2024-25 年）のうち |z| ≥ {Z_MARK} は {n_marked} 組。偶然だけでも約 {chance:.0f} 組は出る（全 {n_all} 組で偶然を 5% に抑える Bonferroni の線は |z| ≥ {bonf:.1f}）。対象（全体・クラス・馬場・距離・コース）は同じ走を重ねて切っているので、印は独立な発見の数ではない。z は、同じレースの馬どうしの相関と同じ馬の重複を無視した標準誤差なので、実際より大きめに出る。<b>印は「見る候補」で、個別のコース・軸の優位とは言い切れない</b>。再現するかは 2026 年の欄（同符号か）で確認する。</p>

<h2>5. 一覧（補正後の差・対象 × 軸）</h2>
<p class="sub">複勝率の「高 − 低」から人気分を引いた差（pt）。色 = 向き（緑 = 高い点ほど人気以上に来る、赤 = 逆）、濃さ = |z|。太枠 = |z| ≥ {Z_MARK}。上の対象のボタン（全体・クラス・馬場・距離）で切り替わる。コース別は 6 節。</p>
<div class="scroll"><table class="mx2" id="hm"></table></div>

<h2>6. コース別（走法軸ごと：場 × 距離）</h2>
{def_p}
<p class="sub">走法軸ごとに見出しを分け、<b>縦 = 競馬場、横 = 距離</b>（同じ列は全ての軸・競馬場で同じ距離）。芝とダートはボタンで切り替わる（全ての表が同時に切り替わる）。
セル: 大きい数字 = <b>補正後の差</b>（人気補正後の複勝率の 高 − 低、pt）、小さい数字 = 高−低（補正なし）と走数（2024-25）。緑 = 高い点ほど人気以上に来る、赤 = 逆、濃さ = |z|、太枠 = |z| ≥ {Z_MARK}。
「—」= そのコースが無い、または走数が {meta['course_min_runs']} 走未満（コース別に出す下限）／高い側・低い側が {MIN_SIDE} 走未満。セルをクリックすると、その表の下に拡大図と数表が出る。</p>
<div class="ctl" id="cgctl"></div>
{ts_part}
<div id="cg-axes"></div>

<h2>7. 限界</h2>
<ul>
<li><b>実力が混ざる</b>: 実線（補正なし）は、点の高い馬ほど強い馬（人気の馬）が多いことを含む。人気で補正した破線との差を見ること。人気は市場が見ている実力のほぼ全てで、補正後の差は小さい（多くは 0 近辺）のが普通。</li>
<li><b>点は「過去走」の値だが、基準は全期間</b>: 各走の点はその走より前の走から作るが、走歴が浅い馬の点を寄せる先（父の平均・全馬の平均）と、10 点への換算の基準は全期間（2024〜25 年・2026 年を含む）の値で作ってある（<code>build_axis_scores10.py</code>）。走歴の浅い馬では、点と着順の関係が実際より強く出うる。ここでは走歴で絞っていない。点の換算は全馬に同じ単調な変換なので、点と着順の関係そのものはほぼ変えない。</li>
<li><b>頭数と中止の扱い</b>: 複勝の頭数の区切り（8 頭以上は 3 着以内）には、結果表の頭数（取消・除外を含みうる）を使った。着順のない走（競走中止など）は除いたので、勝率・複勝率はわずかに高めに出る。人気から期待される率の表には障害のレースも混ざる。いずれも影響は小さい見込みだが、数値は確かめていない。</li>
<li><b>軸どうしは独立でない</b>: Hペース耐性は TS・TS持続力・前後半比と、先行力は出脚と相関がある。軸ごとの曲線は互いの影響を含む（1 軸ずつの周辺の図）。</li>
<li>期間は 2024 年以降だけ（位置・ラップ由来の軸が使えるのが 2024 年以降のため）。2020〜2023 年は含めていない。2026 年は確認用で、決定には使わない。</li>
<li>コース別は走数が少なく（{meta['course_min_runs']} 走以上のコースだけ）、区間ごとの走数は数十〜数百。点線の推定の区間は走数不足で、「効かない」ではなく「見えない」。</li>
<li>点が高い・低いに寄る走は少なく、1〜2 点・9〜10 点の区間は走数が少ない（棒を見ること）。その区間の印は推定で、中央の区間の傾きを延ばし、実測の最小〜最大で頭打ちにしただけ（端での急な上昇・下降は描けない）。</li>
</ul>
<p class="muted">再現: ホストで <code>cd /home/jovyan/work/keiba-vpn &amp;&amp; nice -n 19 /opt/conda/bin/python -I src/scripts/data/axis_score_curves.py OUT.json</code> → 手元で <code>python src/scripts/docs/build_axis_score_curves_html.py OUT.json [OUT.html [TS.json]]</code>
（OUT.html を省くときは <code>-</code>。TS.json = 6 節の「TS の分布」。ホストで <code>cd /home/jovyan/work/keiba-vpn &amp;&amp; nice -n 19 /opt/conda/bin/python -I src/scripts/data/ts_course_resolution.py TS.json /home/jovyan/acq/horse_lap/horse_lap_*.jsonl</code> を実行し、手元へ持ってくる。
定義の最終更新は手元の git から作る。git が使えない・浅いクローンのときは「不明（git なし）」）。
元の点の表: <code>data/local/running_profile/axis_scores10.parquet</code>（{html.escape(meta['history_mtime'][:10])} 時点）。</p>

<script>
var D={data_json};
var CG={cg_json};
{JS.replace("__MINN__", str(MIN_N)).replace("__SIDE__", str(MIN_SIDE)).replace("__ZM__", str(Z_MARK))}
</script>
{SORT_JS if ts else ""}
</main></body></html>"""


JS = r"""
(function(){
var MINN=__MINN__,MINB=20,SIDE=__SIDE__,ZM=__ZM__,NS="http://www.w3.org/2000/svg";
var DIMS=Object.keys(D.dims).filter(function(k){return k!=="course";}),st={dim:"class",ov:false,exp:true,sel:null,surf:"芝"};
var AX=D.axes,axIdx={};AX.forEach(function(a){axIdx[a.key]=a;});
function wilson(k,n){if(n<=0)return [0,0];var z=1.96,p=k/n,d=1+z*z/n,c=p+z*z/(2*n),h=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n));return [(c-h)/d,(c+h)/d];}
function fitLogit(k,n){ // 走数 MINN 以上の区間だけで、点数（区間の番号）に対する 1 次のロジスティック回帰（走数で重み付け）。3 区間未満・当てはめ不能なら null
  var ix=[],i;for(i=0;i<10;i++){if(n[i]>=MINN)ix.push(i);}
  if(ix.length<3)return null;
  var K=0,N=0;ix.forEach(function(j){K+=k[j];N+=n[j];});
  var p0=Math.min(Math.max(K/N,1e-4),1-1e-4),a=Math.log(p0/(1-p0)),b=0;
  for(var it=0;it<40;it++){
    var g0=0,g1=0,h00=0,h01=0,h11=0;
    ix.forEach(function(j){var x=j-4.5,p=1/(1+Math.exp(-(a+b*x))),w=n[j]*p*(1-p);g0+=k[j]-n[j]*p;g1+=(k[j]-n[j]*p)*x;h00+=w;h01+=w*x;h11+=w*x*x;});
    var det=h00*h11-h01*h01;if(!(det>1e-12))return null;
    var da=(h11*g0-h01*g1)/det,db=(h00*g1-h01*g0)/det;
    if(!isFinite(da)||!isFinite(db))return null;
    a+=da;b+=db;if(Math.abs(da)+Math.abs(db)<1e-9)break;
  }
  return function(j){return 1/(1+Math.exp(-(a+b*(j-4.5))));};
}
function est(k,n,exact){ // exact = 人気から期待される率（平均なので走数が少なくても実測のまま。走数 0 だけ曲線で補う）
   // 各区間の {v: 率, e: 推定か}。走数 MINN 以上は実測。未満は、当てはめ曲線に実測を（MINN 走ぶんの重みで）混ぜる。走数 0 は曲線そのもの
  var f=fitLogit(k,n),out=[],lo=1,hi=0,i;
  for(i=0;i<10;i++){if(n[i]>=MINN){lo=Math.min(lo,k[i]/n[i]);hi=Math.max(hi,k[i]/n[i]);}}
  for(i=0;i<10;i++){ // 曲線は、実測の区間の最小〜最大の範囲に収める（端で勝率が跳ね上がる延長を防ぐ）
    if(n[i]>=MINN||(exact&&n[i]>0))out.push({v:k[i]/n[i],e:false});
    else if(f)out.push({v:(k[i]+MINN*Math.min(Math.max(f(i),lo),hi))/(n[i]+MINN),e:true});
    else out.push(null);
  }
  return out;
}
function sum(a,i,j){var s=0;for(var k=i;k<j;k++)s+=a[k];return s;}
function side(c){ // c = [n,w,p,ew,ep]
  var nl=sum(c[0],0,4),pl=sum(c[2],0,4),el=sum(c[4],0,4),nh=sum(c[0],5,10),ph=sum(c[2],5,10),eh=sum(c[4],5,10);
  if(nl<SIDE||nh<SIDE)return null;
  var rl=pl/nl,rh=ph/nh,se=Math.sqrt(rl*(1-rl)/nl+rh*(1-rh)/nh),adj=(ph-eh)/nh-(pl-el)/nl;
  return {raw:rh-rl,adj:adj,z:se>0?adj/se:0};
}
function f1(x){return (x>=0?"+":"")+(x*100).toFixed(1);}
function groups(){return D.dims[st.dim].groups;}
function wmaxOf(gs){
  var m=0;gs.forEach(function(g){Object.keys(g.data).forEach(function(k){var c=g.data[k].A;for(var i=0;i<10;i++){if(c[0][i]>=200){var r=c[1][i]/c[0][i];if(r>m)m=r;var q=c[2][i]/c[0][i]/3;if(q>m)m=q;}}});});
  m=Math.min(Math.max(m*1.08,0.10),0.35);return Math.ceil(m*100/5)/20; // 0.05 刻みに切り上げ
}
function chart(c,W,H,wmax,big){
  var pl=big?46:26,pr=big?50:26,pt=big?14:8,pb=big?34:18,iw=W-pl-pr,ih=H-pt-pb;
  var A=c.A,B=c.B,s=[];
  function X(i){return pl+(i+0.5)*iw/10;}
  function YW(v){return pt+ih*(1-Math.min(v/wmax,1));}
  function YP(v){return pt+ih*(1-Math.min(v/(3*wmax),1));}
  s.push('<svg xmlns="'+NS+'" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="走法軸の点数別 勝率・複勝率">');
  s.push('<rect x="'+pl+'" y="'+pt+'" width="'+iw+'" height="'+ih+'" fill="#0b1428" stroke="#243358"/>');
  var nmax=Math.max.apply(null,A[0].concat([1]));
  for(var i=0;i<10;i++){var bh=ih*0.18*A[0][i]/nmax;s.push('<rect x="'+(X(i)-iw/26)+'" y="'+(pt+ih-bh)+'" width="'+(iw/13)+'" height="'+bh+'" fill="#3b4a73" opacity="0.8"/>');}
  var nt=sum(A[0],0,10),wm=sum(A[1],0,10)/Math.max(nt,1),pm=sum(A[2],0,10)/Math.max(nt,1);
  if(nt>0){s.push('<line x1="'+pl+'" x2="'+(pl+iw)+'" y1="'+YW(wm)+'" y2="'+YW(wm)+'" stroke="#7aa9ff" stroke-dasharray="1 3" opacity="0.7"/>');
    s.push('<line x1="'+pl+'" x2="'+(pl+iw)+'" y1="'+YP(pm)+'" y2="'+YP(pm)+'" stroke="#fbbf24" stroke-dasharray="1 3" opacity="0.7"/>');}
  function band(idx,col,Y){ if(!big)return; // Wilson の帯
    var up=[],lo=[];for(var i=0;i<10;i++){if(A[0][i]>=MINN){var w=wilson(A[idx][i],A[0][i]);up.push([X(i),Y(w[1])]);lo.push([X(i),Y(w[0])]);}else{flush();}}
    flush();
    function flush(){if(up.length>1){var d="M"+up.map(function(p){return p[0]+" "+p[1];}).join("L")+"L"+lo.slice().reverse().map(function(p){return p[0]+" "+p[1];}).join("L")+"Z";s.push('<path d="'+d+'" fill="'+col+'" opacity="0.16"/>');}up=[];lo=[];}
  }
  band(1,"#7aa9ff",YW);band(2,"#fbbf24",YP);
  function line(S,Y,col,dash,w,marker){ // S = est() の結果。実測の区間は実線、推定の区間は点線でつなぐ
    var i,seg=[];
    for(i=0;i<9;i++){if(S[i]&&S[i+1]&&(S[i].e||S[i+1].e))s.push('<line x1="'+X(i)+'" y1="'+Y(S[i].v)+'" x2="'+X(i+1)+'" y2="'+Y(S[i+1].v)+'" stroke="'+col+'" stroke-width="'+(w*0.8)+'" stroke-dasharray="1.5 3"/>');}
    function flush(){if(seg.length>1)s.push('<polyline points="'+seg.map(function(p){return p[0]+","+p[1];}).join(" ")+'" fill="none" stroke="'+col+'" stroke-width="'+w+'"'+(dash?' stroke-dasharray="4 3"':'')+'/>');seg=[];}
    for(i=0;i<10;i++){if(S[i]&&!S[i].e)seg.push([X(i),Y(S[i].v)]);else flush();}flush();
    if(marker){var r=big?4:2.4;for(i=0;i<10;i++){if(!S[i])continue;var x=X(i),y=Y(S[i].v),f=S[i].e?'#0b1428':col,sw=S[i].e?' stroke="'+col+'" stroke-width="1.4"':'';
      s.push(marker==="c"?'<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+f+'"'+sw+'/>':'<rect x="'+(x-r)+'" y="'+(y-r)+'" width="'+2*r+'" height="'+2*r+'" fill="'+f+'"'+sw+'/>');}}
  }
  var SW=est(A[1],A[0]),SP=est(A[2],A[0]);
  if(st.exp){line(est(A[3],A[0],true),YW,"#7aa9ff",true,big?1.6:1,null);line(est(A[4],A[0],true),YP,"#fbbf24",true,big?1.6:1,null);}
  line(SW,YW,"#7aa9ff",false,big?2.4:1.6,"c");line(SP,YP,"#fbbf24",false,big?2.4:1.6,"s");
  if(st.ov){for(var i=0;i<10;i++){if(B[0][i]>=MINB){var r=big?4.5:2.8;
    s.push('<circle cx="'+X(i)+'" cy="'+YW(B[1][i]/B[0][i])+'" r="'+r+'" fill="none" stroke="#e6ecff" stroke-width="1.3"/>');
    s.push('<rect x="'+(X(i)-r)+'" y="'+(YP(B[2][i]/B[0][i])-r)+'" width="'+2*r+'" height="'+2*r+'" fill="none" stroke="#e6ecff" stroke-width="1.3"/>');}}}
  // 軸の目盛
  var tf=big?11:8.5;
  s.push('<text x="'+(pl-4)+'" y="'+(pt+4)+'" fill="#7aa9ff" font-size="'+tf+'" text-anchor="end">'+(wmax*100).toFixed(0)+'%</text><text x="'+(pl-4)+'" y="'+(pt+ih)+'" fill="#7aa9ff" font-size="'+tf+'" text-anchor="end">0</text>');
  s.push('<text x="'+(pl+iw+4)+'" y="'+(pt+4)+'" fill="#fbbf24" font-size="'+tf+'">'+(3*wmax*100).toFixed(0)+'%</text><text x="'+(pl+iw+4)+'" y="'+(pt+ih)+'" fill="#fbbf24" font-size="'+tf+'">0</text>');
  for(var i=0;i<10;i++){if(big||i===0||i===4||i===9)s.push('<text x="'+X(i)+'" y="'+(pt+ih+(big?16:11))+'" fill="#9aa7c7" font-size="'+tf+'" text-anchor="middle">'+(i+1)+'</text>');}
  if(big){s.push('<text x="'+(pl+iw/2)+'" y="'+(H-4)+'" fill="#9aa7c7" font-size="12" text-anchor="middle">走法軸の点数（整数部。10 = 10 点）</text>');
    s.push('<text transform="translate(12,'+(pt+ih/2)+') rotate(-90)" fill="#7aa9ff" font-size="12" text-anchor="middle">勝率</text><text transform="translate('+(W-8)+','+(pt+ih/2)+') rotate(90)" fill="#fbbf24" font-size="12" text-anchor="middle">複勝率</text>');}
  s.push('</svg>');return s.join("");
}
function ctl(){
  var h=[],el=document.getElementById("ctl");
  h.push('<label>対象</label>');
  DIMS.forEach(function(k){h.push('<button data-d="'+k+'" class="'+(k===st.dim?'on':'')+'">'+D.dims[k].label+'</button>');});
  h.push('<button data-t="ov" class="'+(st.ov?'on':'')+'">2026 年を重ねる</button><button data-t="exp" class="'+(st.exp?'on':'')+'">人気補正の破線</button>');
  el.innerHTML=h.join("");
  Array.prototype.forEach.call(el.querySelectorAll("button"),function(b){b.onclick=function(){
    if(b.dataset.d){st.dim=b.dataset.d;st.sel=null;}else if(b.dataset.t==="ov")st.ov=!st.ov;else st.exp=!st.exp;
    ctl();render();big();};});
}
function render(){
  var gs=groups(),wmax=wmaxOf(gs),h=[];
  h.push('<thead><tr><th class="rh">走法軸</th>'+gs.map(function(g){var n=0;Object.keys(g.data).forEach(function(k){n=Math.max(n,sum(g.data[k].A[0],0,10));});return '<th>'+g.label+'<small>最大走数 '+n.toLocaleString()+'</small></th>';}).join("")+'</tr></thead><tbody>');
  AX.forEach(function(a){
    h.push('<tr><th class="rh">'+a.label+'<small>高: '+a.high+'<br>低: '+a.low+'</small></th>');
    gs.forEach(function(g,gi){
      var c=g.data[a.key];
      if(!c||sum(c.A[0],0,10)<MINN){h.push('<td><div class="cap">—</div></td>');return;}
      var s=side(c.A),cap=s?'高−低 <b>'+f1(s.raw)+'</b><br>補正後 <b'+(Math.abs(s.z)>=ZM?' style="color:var(--warn)"':'')+'>'+f1(s.adj)+'</b>':'走数不足';
      h.push('<td data-a="'+a.key+'" data-g="'+gi+'" class="'+(st.sel&&st.sel[0]===a.key&&st.sel[1]===gi?'sel':'')+'">'+chart(c,178,104,wmax,false)+'<div class="cap">'+cap+'</div></td>');
    });
    h.push('</tr>');
  });
  h.push('</tbody>');
  var mx=document.getElementById("mx");mx.innerHTML=h.join("");
  Array.prototype.forEach.call(mx.querySelectorAll("td[data-a]"),function(td){td.onclick=function(){st.sel=[td.dataset.a,+td.dataset.g];var o=mx.querySelector("td.sel");if(o)o.classList.remove("sel");td.classList.add("sel");big();document.getElementById("big").scrollIntoView({block:"nearest"});};});
  heat();
}
function big(){
  var el=document.getElementById("big");
  if(!st.sel){el.innerHTML='<span class="muted">下のマスをクリックすると、ここに拡大図と数表が出る。</span>';return;}
  bigInto(el,axIdx[st.sel[0]],groups()[st.sel[1]]);
}
function bigInto(el,a,g){
  var c=g.data[a.key];
  var wmax=wmaxOf([g]),h=[];
  var s=side(c.A),sb=side(c.B);
  h.push('<b>'+a.label+' × '+g.label+'</b> <span class="muted">（高: '+a.high+' ／ 低: '+a.low+'）</span>');
  h.push('<div class="lg"><span><i style="border-color:#7aa9ff"></i>勝率（左軸）</span><span><i style="border-color:#fbbf24"></i>複勝率（右軸 = 左の 3 倍）</span><span><i class="d" style="border-color:#7aa9ff"></i><i class="d" style="border-color:#fbbf24"></i>人気から期待される率（同じ色の破線）</span><span>帯 = Wilson 95%（実測の区間だけ）</span><span>白抜き（白枠）= 2026 年</span><span>色枠で中が暗い印 + 点線 = 推定（走数 '+MINN+' 未満）</span></div>');
  h.push(chart(c,880,380,wmax,true));
  h.push('<p class="sub">高−低（6.0 点以上 − 4.5 点以下の複勝率）: 2024-25 '+(s?f1(s.raw)+' pt（補正後 '+f1(s.adj)+'、z '+s.z.toFixed(1)+'）':'走数不足')+' ／ 2026 '+(sb?f1(sb.raw)+' pt（補正後 '+f1(sb.adj)+'、z '+sb.z.toFixed(1)+'）':'走数不足')+'</p>');
  h.push('<div class="scroll"><table><thead><tr><th>点</th><th>走数</th><th>勝率</th><th>勝率の期待</th><th>複勝率</th><th>複勝率の期待</th><th>2026 走数</th><th>2026 勝率</th><th>2026 複勝率</th></tr></thead><tbody>');
  function r(x,n){return n>0?(x/n*100).toFixed(1)+"%":"—";}
  var E=[est(c.A[1],c.A[0]),est(c.A[3],c.A[0],true),est(c.A[2],c.A[0]),est(c.A[4],c.A[0],true)];
  function cell(S,i){var o=S[i];if(!o)return '<td class="num">—</td>';return o.e?'<td class="num" style="color:var(--muted);font-style:italic">'+(o.v*100).toFixed(1)+'%（推）</td>':'<td class="num">'+(o.v*100).toFixed(1)+'%</td>';}
  for(var i=0;i<10;i++){var n=c.A[0][i],m=c.B[0][i];
    h.push('<tr><td>'+(i+1)+'</td><td class="num">'+n+'</td>'+cell(E[0],i)+cell(E[1],i)+cell(E[2],i)+cell(E[3],i)+'<td class="num">'+m+'</td><td class="num">'+r(c.B[1][i],m)+'</td><td class="num">'+r(c.B[2][i],m)+'</td></tr>');}
  h.push('</tbody></table></div>');
  el.innerHTML=h.join("");
}
function heat(){
  var gs=groups(),h=['<thead><tr><th>走法軸</th>'+gs.map(function(g){return '<th>'+g.label+'</th>';}).join("")+'</tr></thead><tbody>'];
  AX.forEach(function(a){
    h.push('<tr><td>'+a.label+'</td>');
    gs.forEach(function(g){var c=g.data[a.key],s=c?side(c.A):null;
      if(!s){h.push('<td class="hm">—</td>');return;}
      var al=Math.min(Math.abs(s.z)/4,1)*0.75,col=s.adj>=0?'74,222,128':'248,113,113';
      h.push('<td class="hm'+(Math.abs(s.z)>=ZM?' k':'')+'" style="background:rgba('+col+','+al.toFixed(2)+')" title="z '+s.z.toFixed(1)+'">'+f1(s.adj)+'</td>');});
    h.push('</tr>');});
  h.push('</tbody>');document.getElementById("hm").innerHTML=h.join("");
}
function heatBg(s){var al=Math.min(Math.abs(s.z)/4,1)*0.75,col=s.adj>=0?'74,222,128':'248,113,113';return 'background:rgba('+col+','+al.toFixed(2)+')';}
function gridHead(g){return '<thead><tr><th class="rh">競馬場＼距離</th>'+g.dists.map(function(d){return '<th>'+d+'m</th>';}).join("")+'</tr></thead>';}
function cgCtl(){
  var el=document.getElementById("cgctl");if(!el)return;
  el.innerHTML='<label>馬場</label>'+[["芝","芝"],["ダ","ダート"]].map(function(x){return '<button data-s="'+x[0]+'" class="'+(x[0]===st.surf?'on':'')+'">'+x[1]+'</button>';}).join("");
  Array.prototype.forEach.call(el.querySelectorAll("button"),function(b){b.onclick=function(){st.surf=b.dataset.s;cgCtl();cgAxes();cgTs();};});
}
function cgAxes(){
  var el=document.getElementById("cg-axes");if(!el)return;
  var g=CG.axg[st.surf],gs=D.dims.course?D.dims.course.groups:[],h=[];
  AX.forEach(function(a){
    h.push('<h3>'+a.label+' <span class="defn">'+(CG.defn[a.key]?'— '+CG.defn[a.key]:'')+'</span></h3><p class="sub">高: '+a.high+' ／ 低: '+a.low+'</p>');
    if(CG.only[a.key]&&CG.only[a.key]!==st.surf){h.push('<p class="cgn">この馬場（'+(st.surf==="芝"?"芝":"ダート")+'）では対象外（'+(CG.only[a.key]==="芝"?"芝":"ダート")+'の走だけで測る軸）。</p>');return;}
    var rows=['<div class="cgw"><table class="cg" data-ax="'+a.key+'">'+gridHead(g)+'<tbody>'];
    g.venues.forEach(function(v){
      rows.push('<tr><th class="rh">'+v+'</th>');
      g.dists.forEach(function(d){
        var gi=g.idx[v+"|"+d],c=gi===undefined?null:gs[gi].data[a.key];
        if(!c){rows.push('<td class="hm">—</td>');return;}
        var n=sum(c.A[0],0,10),sd=side(c.A);
        if(!sd){rows.push('<td class="hm" data-a="'+a.key+'" data-g="'+gi+'" title="高い側・低い側の走数が足りない"><small>走数不足</small><small>'+n.toLocaleString()+' 走</small></td>');return;}
        rows.push('<td class="hm'+(Math.abs(sd.z)>=ZM?' k':'')+'" data-a="'+a.key+'" data-g="'+gi+'" style="'+heatBg(sd)+'" title="z '+sd.z.toFixed(1)+'"><b>'+f1(sd.adj)+'</b><small>高−低 '+f1(sd.raw)+'</small><small>'+n.toLocaleString()+' 走</small></td>');
      });
      rows.push('</tr>');
    });
    rows.push('</tbody></table></div><div class="big" id="cgbig-'+a.key+'" style="display:none"></div>');
    h.push(rows.join(""));
  });
  el.innerHTML=h.join("");
  Array.prototype.forEach.call(el.querySelectorAll("td[data-a]"),function(td){td.onclick=function(){
    var k=td.dataset.a,tb=td.closest("table"),o=tb.querySelector("td.sel");if(o)o.classList.remove("sel");td.classList.add("sel");
    var b=document.getElementById("cgbig-"+k);b.style.display="block";bigInto(b,axIdx[k],gs[+td.dataset.g]);
  };});
}
function ncdf(x){var t=1/(1+0.2316419*Math.abs(x)),d=0.3989423*Math.exp(-x*x/2),p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));return x>0?1-p:p;}
function tsHist(c,hm){
  var NB=c.h.length,W=880,H=300,pl=52,pr=16,pt=14,pb=40,iw=W-pl-pr,ih=H-pt-pb,bw=iw/NB,mx=Math.max.apply(null,c.h.concat([1])),s=[];
  s.push('<svg xmlns="'+NS+'" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="TS のヒストグラム">');
  s.push('<rect x="'+pl+'" y="'+pt+'" width="'+iw+'" height="'+ih+'" fill="#0b1428" stroke="#243358"/>');
  var exp=[];
  for(var i=0;i<NB;i++){var bh=ih*c.h[i]/mx,out=(i===0||i===NB-1);
    s.push('<rect x="'+(pl+i*bw+1)+'" y="'+(pt+ih-bh)+'" width="'+(bw-2)+'" height="'+bh+'" fill="'+(out?"#f87171":"#7aa9ff")+'" opacity="0.85"/>');
    if(c.h[i]>0)s.push('<text x="'+(pl+(i+0.5)*bw)+'" y="'+(pt+ih-bh-3)+'" fill="#9aa7c7" font-size="9.5" text-anchor="middle">'+c.h[i]+'</text>');
    if(i>0&&i<NB-1){var lo=hm.lo+(i-1)*hm.step,hi=lo+hm.step;exp.push([pl+(i+0.5)*bw,pt+ih*(1-c.n*(ncdf((hi-50)/10)-ncdf((lo-50)/10))/mx)]);}}
  s.push('<polyline points="'+exp.map(function(p){return p[0]+","+Math.max(p[1],pt);}).join(" ")+'" fill="none" stroke="#fbbf24" stroke-width="1.6" stroke-dasharray="4 3"/>');
  for(var t=hm.lo;t<=hm.hi+1e-9;t+=20){var x=pl+(1+(t-hm.lo)/hm.step)*bw;s.push('<line x1="'+x+'" x2="'+x+'" y1="'+pt+'" y2="'+(pt+ih)+'" stroke="#243358"/><text x="'+x+'" y="'+(pt+ih+14)+'" fill="#9aa7c7" font-size="11" text-anchor="middle">'+t+'</text>');}
  var x50=pl+(1+(50-hm.lo)/hm.step)*bw;
  s.push('<line x1="'+x50+'" x2="'+x50+'" y1="'+pt+'" y2="'+(pt+ih)+'" stroke="#e6ecff" stroke-dasharray="2 3"/>');
  s.push('<text x="'+(pl+iw/2)+'" y="'+(H-6)+'" fill="#9aa7c7" font-size="12" text-anchor="middle">TS（4 点刻み。赤 = 10 未満・90 以上）</text>');
  s.push('<text x="'+(pl-6)+'" y="'+(pt+4)+'" fill="#9aa7c7" font-size="11" text-anchor="end">'+mx+'</text><text x="'+(pl-6)+'" y="'+(pt+ih)+'" fill="#9aa7c7" font-size="11" text-anchor="end">0</text>');
  s.push('</svg>');return s.join("");
}
function tsBig(c,hm,el){
  var q=c.q,h=[];
  h.push('<b>'+c.v+(c.s==="芝"?"芝":"ダート")+c.d+'m の TS の分布</b> <span class="muted">（'+c.n.toLocaleString()+' 走）</span>');
  h.push('<div class="lg"><span><i style="border-color:#7aa9ff"></i>走数（棒）</span><span><i class="d" style="border-color:#fbbf24"></i>TS が平均 50・標準偏差 10 の正規分布だったときの走数</span></div>');
  h.push(tsHist(c,hm));
  h.push('<div class="scroll"><table><thead><tr><th>標準偏差</th><th>歪み</th><th>5%点</th><th>25%点</th><th>50%点</th><th>75%点</th><th>95%点</th><th>最小</th><th>最大</th><th>'+hm.peak_outlier.toFixed(0)+' 秒以上の走数</th></tr></thead><tbody><tr><td class="num">'+c.sd.toFixed(2)+'</td><td class="num">'+(c.sk>=0?"+":"")+c.sk.toFixed(2)+'</td>'+q.map(function(x){return '<td class="num">'+x.toFixed(1)+'</td>';}).join("")+'<td class="num">'+c.mn.toFixed(1)+'</td><td class="num">'+c.mx.toFixed(1)+'</td><td class="num">'+c.o+'</td></tr></tbody></table></div>');
  el.innerHTML=h.join("");
}
function cgTs(){
  var el=document.getElementById("cg-ts");if(!el||!CG.ts)return;
  var T=CG.ts,g=T.grid[st.surf],h=['<h4>コース別（縦 = 競馬場、横 = 距離）: TS の標準偏差</h4><p class="sub">大きい数字 = TS の標準偏差（設計値は 10）、下 = 歪み・走数・'+T.hist.peak_outlier.toFixed(0)+' 秒以上の走数。<b>赤 = 歪みの絶対値が '+CG.skm+' 以上</b>。クリックでヒストグラムと分位点。</p><div class="cgw"><table class="cg">'+gridHead(g)+'<tbody>'];
  g.venues.forEach(function(v){
    h.push('<tr><th class="rh">'+v+'</th>');
    g.dists.forEach(function(d){
      var i=g.idx[v+"|"+d];if(i===undefined){h.push('<td class="hm">—</td>');return;}
      var c=T.courses[i];
      h.push('<td class="hm'+(Math.abs(c.sk)>=CG.skm?' sk':'')+'" data-t="'+i+'"><b>'+c.sd.toFixed(1)+'</b><small>歪み '+(c.sk>=0?"+":"")+c.sk.toFixed(1)+'</small><small>'+c.n.toLocaleString()+' 走'+(c.o>0?'・外れ値 '+c.o:'')+'</small></td>');
    });
    h.push('</tr>');
  });
  h.push('</tbody></table></div><div class="big" id="cgbig-tsdist" style="display:none"></div>');
  el.innerHTML=h.join("");
  Array.prototype.forEach.call(el.querySelectorAll("td[data-t]"),function(td){td.onclick=function(){
    var tb=td.closest("table"),o=tb.querySelector("td.sel");if(o)o.classList.remove("sel");td.classList.add("sel");
    var b=document.getElementById("cgbig-tsdist");b.style.display="block";tsBig(T.courses[+td.dataset.t],T.hist,b);
  };});
}
ctl();render();cgCtl();cgAxes();cgTs();
})();
"""


def main() -> None:
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else OUT
    ts = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")) if len(sys.argv) > 3 else None
    out.write_text(build(d, ts, axis_defs([a["key"] for a in d["axes"]])), encoding="utf-8")
    print("wrote", out, f"{out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
