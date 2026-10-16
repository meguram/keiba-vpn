#!/usr/bin/env python3
"""データの「収集 → 保存 → 加工 → 統合（特徴量・推論）」の流れを、cron の発生時刻つきのフローチャートにする。

出力: docs/html/modeling/chapters/data_flow_schedule.html

時刻は cron の登録内容（正本 scripts/cron/setup_all_cron.sh）から読む（JST に換算済み）ため、cron を変えて再生成すれば追従する。
どのデータがどのジョブで集まり、どの加工を経て、どこで統合されるか（誰が・いつ動かすか）は、下の ROWS に事実だけを書く。

  python -m src.scripts.docs.build_data_flow_html
  python -m src.scripts.docs.build_data_flow_html --crontab-file /tmp/crontab.txt --out /tmp/flow.html
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.scripts.docs.build_cron_schedule_html import JST, parse

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "docs" / "html" / "modeling" / "chapters" / "data_flow_schedule.html"

# 統合先（右端の列）
TARGETS = {
    "train": ("学習（手動・ノートブック）", "tp-01 母表 → tp-02 履歴特徴量 127 列 → tp-03 〜 tp-07。2020〜2025 年のみ。cron では動かない", "#4f8bff"),
    "accum": ("履歴特徴量の蓄積", "当年分の 127 列を確定結果から作り直し、推論モードとのずれを記録", "#34d399"),
    "infer": ("当日の推論（特徴量づくり）", "馬体重の検知 → 推論モードで特徴量 → （既定では予測は起動しない）", "#fb923c"),
    "bet": ("期待値・推奨馬券（LayerB）", "T-15 のオッズと合成。オッズは LayerA に入れない", "#c084fc"),
}


# 統合先ごとの cron ジョブ（箱に時刻を出す）。train は cron では動かない
TARGET_JOB = {"accum": "feature-accumulate", "infer": "raceday-weight-runner", "bet": "raceday-runner"}


@dataclass
class Row:
    label: str                      # データ名
    src: str                        # データ元
    collect: list[tuple[str, str]]  # (cron ジョブキー or "", 説明)
    store: str                      # 保存先
    process: list[tuple[str, str]]  # (cron ジョブキー or "", 説明)
    targets: list[str]              # TARGETS のキー
    note: str = ""                  # 補足（表に出す）


ROWS: list[Row] = [
    Row("速報結果", "race.netkeiba（result.html）",
        [("raceday-result-runner", "各R 発走15分後に取得"), ("raceday-evening", "開催日夕方にまとめて取得")],
        "GCS: race_result_on_time",
        [("export_tables_daily", "flat 表へ増分（race_result_on_time ほか）")], ["accum", "train"],
        "着順の通過順・上り・馬体重の補正（on_time_repair）に使う。"),
    Row("確定結果・ラップ・タイム指数", "db.netkeiba / race.netkeiba",
        [("recent-retry", "直近21日の未取得を取り直し"), ("weekly-update", "直近10日を更新"), ("backfill", "年度別の取りこぼしを補填")],
        "GCS: race_result・race_result_lap・race_index",
        [("export_tables_daily", "flat 表へ増分（race_result・race_result_lap）。race_index の flat は cron なし（手動）")], ["accum", "train"],
        "結果は翌日以降に確定。上り・通過順は会員限定ページ。"),
    Row("出馬表・馬柱・追い切り", "race.netkeiba（shutuba / 馬柱 / 追い切り）",
        [("raceday-eve", "前日夕方に取得"), ("raceday-runner", "各R 発走15分前に再取得（T-15）")],
        "GCS: race_shutuba・race_shutuba_past・race_oikiri・smartrc",
        [("export_tables_daily", "flat 表へ増分（race_shutuba のみ）。馬柱・追い切りの flat は cron なし（手動）")], ["train", "infer"],
        "枠順・斤量は木〜金に出る。馬柱は火曜から取得可。"),
    Row("馬体重（出馬表の再取得）", "race.netkeiba（shutuba）",
        [("raceday-weight-runner", "各R 発走65〜30分前に90秒間隔で再取得し、馬体重が揃うのを検知")],
        "GCS: race_shutuba（更新）／ローカル: data/local/inference",
        [("raceday-weight-runner", "揃ったら推論モードで特徴量を作る（予測は既定で起動しない）")], ["infer"],
        "馬体重の発表は基本的に発走の約1時間前。発表時刻の実測は未了。"),
    Row("オッズ（単複・馬連・ワイド・3連複）", "race.netkeiba（API）",
        [("raceday-runner", "各R 発走60分前・15分前"), ("raceday-evening", "確定オッズ")],
        "GCS: race_odds・race_pair_odds（T-60/T-15 の履歴を同じJSONに保持）",
        [], ["bet"], "LayerA（学習の母表）には入れない。期待値の計算（LayerB）でだけ使う。"),
    Row("厩舎コメント（レース前）", "race.netkeiba",
        [("raceday-comment", "前日と当日朝。出ているレースだけ保存")],
        "GCS: race_trainer_comment",
        [], [], "特徴量ストアには登録されているが、現行の母表・train_table（127列）には使っていない。"),
    Row("バロメーター（走行データ）", "race.netkeiba（API）",
        [("weekly-update", "金曜に直近10日"), ("barometer-retry", "公開済みで未取得のものだけ")],
        "GCS: race_barometer",
        [], [], "公開はレース翌週の金曜18時ごろ。現行の train_table には使っていない。"),
    Row("JRA 馬場（クッション値・含水率）", "JRA 公式",
        [("jra-baba-morning", "開催日の朝に10分ごと")],
        "ローカル: data/page_reference/cushion（GCS は手動アップロード）",
        [], ["infer"], "当日の馬場差（P7）の補助。"),
    Row("レース一覧・発走時刻", "netkeiba / JRA",
        [("daily-race-lists", "朝・夕の2回。14日先まで毎回、それより先は月曜だけ")],
        "ローカル: data/page_reference/race_lists・race_day_schedule",
        [], ["infer"], "各ランナーが発走時刻表として参照する。"),
    Row("馬情報・血統", "db.netkeiba（馬ページ・血統表）",
        [("weekly-update", "出走馬の馬情報（金曜）"), ("new-horse-pedigree", "新馬の5世代血統（毎日）"), ("backfill", "馬情報の補填")],
        "GCS: horse_result・horse_pedigree_5gen／ローカル: ped_tbl",
        [("new-horse-pedigree", "10世代・ped_tbl の生成")], ["train"],
        "父・母父の列は出馬表から。"),
    Row("騎手・調教師の成績", "race_result の集計（取得ではない）",
        [],
        "ローカル: data/features/jockey_trainer_stats",
        [("update_jockey_trainer_stats", "統計表を再生成")], [], "騎手・調教師の通算成績（jockey_prior_*）は母表の構築時に race_result から別に計算する。"),
]

# 各工程の処理時間（ダウンタイム）。**未計測なので空文字**。計測したら文字列（例 "約 4 分"）を入れて再生成する。
# キー: "collect:<データ名>"（取得）/ "process:<データ名>"（加工）/ "target:<統合先>"（統合・利用）。値が空の間は空欄の記入欄を出す。
DURATIONS: dict[str, str] = (
    {f"collect:{r.label}": "" for r in ROWS if r.collect}
    | {f"process:{r.label}": "" for r in ROWS if r.process}
    | {f"target:{k}": "" for k in TARGETS}
)
DUR_MARK = "\x00DUR:"


def dur_line(key: str) -> tuple[str, str]:
    return (DUR_MARK + DURATIONS.get(key, ""), "#9aa7c7")


# 1 日のタイムラインに載せるジョブ（レーン, ジョブキー, 短い名前）。時刻は cron から読む。
LANES = [("取得", ["jra-baba-morning", "recent-retry", "daily-race-lists", "raceday-comment", "raceday-runner", "raceday-result-runner", "raceday-weight-runner",
                  "barometer-retry", "raceday-evening", "raceday-eve", "weekly-update", "new-horse-pedigree"]),
         ("加工・統合", ["sync_pg_from_gcs", "update_jockey_trainer_stats", "export_tables_daily", "feature-accumulate", "horse-name-index"])]
SHORT = {
    "jra-baba-morning": "JRA馬場", "recent-retry": "取り直し", "daily-race-lists": "レース一覧", "raceday-comment": "厩舎コメント",
    "raceday-runner": "オッズ/出馬表", "raceday-result-runner": "速報結果", "raceday-weight-runner": "馬体重検知", "barometer-retry": "バロメーター",
    "raceday-evening": "夕方まとめ", "raceday-eve": "前日出馬表", "weekly-update": "週次更新", "new-horse-pedigree": "新馬血統",
    "sync_pg_from_gcs": "DB同期", "update_jockey_trainer_stats": "騎手調教師", "export_tables_daily": "flat表更新", "feature-accumulate": "特徴量蓄積",
    "horse-name-index": "馬名索引",
}


def job_times(jobs) -> dict[str, list[str]]:
    """ジョブキー → JST の時刻表記（重複排除。『毎日 07:30』『金 18:00』形式）。"""
    out: dict[str, list[str]] = {}
    for j in jobs:
        if j.when == "システム起動時" or j.schedule == "@reboot":
            continue
        out.setdefault(j.key, [])
        if j.when not in out[j.key]:
            out[j.key].append(j.when)
    return out


def compact(whens: list[str]) -> str:
    """['毎日 06:10', '毎日 19:40'] → '毎日 06:10・19:40'。"""
    by_day: dict[str, list[str]] = {}
    other: list[str] = []
    for w in whens:
        m = re.match(r"^(\S+) (\d{2}:\d{2}(?:・\d{2}:\d{2})*)$", w)
        if m:
            by_day.setdefault(m.group(1), []).extend(m.group(2).split("・"))
        else:
            other.append(w)
    parts = [f"{d} {'・'.join(sorted(set(ts)))}" for d, ts in by_day.items()] + other
    return " ／ ".join(parts)


def tstr(times: dict[str, list[str]], key: str) -> str:
    if not key:
        return ""
    if key == "backfill":
        return "毎日 深夜 00:00〜04:00・06:00・07:50〜09:00"
    return compact(times.get(key, [])) or "（cron 未登録）"


# ── SVG ────────────────────────────────────────────────────────────────────────

def _w(ch: str) -> float:
    return 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.55


def wrap(text: str, px: float, size: float) -> list[str]:
    """px 幅に収まるように折り返す（全角=size、半角=0.55*size 程度で概算）。区切り（・_/ 空白 句読点）の後ろで折る。"""
    breaks = set("・_/ 　、。）)")
    lines, cur, w = [], "", 0.0
    for ch in text:
        cw = _w(ch) * size
        if w + cw > px and cur:
            cut = max((i for i, c in enumerate(cur) if c in breaks), default=-1)
            if cut >= len(cur) * 0.4 and cut + 1 < len(cur):
                lines.append(cur[:cut + 1])
                cur = cur[cut + 1:]
            else:
                lines.append(cur)
                cur = ""
            w = sum(_w(c) * size for c in cur)
        cur += ch
        w += cw
    if cur:
        lines.append(cur)
    return lines


def esc(s: str) -> str:
    return html.escape(s, quote=True)


COLS = {"src": (10, 150), "collect": (180, 330), "store": (535, 200), "proc": (760, 235), "tgt": (1020, 250)}
FS = 11.5
LH = 15.5


def box(x: float, y: float, w: float, lines: list[tuple[str, str]], fill: str, stroke: str, h: float | None = None) -> tuple[str, float]:
    """lines: [(テキスト, 色)] を縦に並べたボックスの SVG と高さ。DUR_MARK で始まる行は『処理時間』の記入欄として描く。"""
    h = h or (len(lines) * LH + 12)
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>']
    for i, (t, c) in enumerate(lines):
        ty = y + 16 + i * LH
        if t.startswith(DUR_MARK):
            val = t[len(DUR_MARK):]
            out.append(f'<text x="{x + 8}" y="{ty:.1f}" font-size="{FS - 1}" fill="#9aa7c7">処理時間</text>')
            out.append(f'<rect x="{x + 64}" y="{ty - 11:.1f}" width="{min(w - 108, 130)}" height="14" rx="3" fill="#2a2410" fill-opacity="0.55" stroke="#fbbf24" stroke-width="1" stroke-dasharray="3 2"/>')
            out.append(f'<text x="{x + 70}" y="{ty:.1f}" font-size="{FS - 1}" fill="#ffd479">{esc(val)}</text>')
            out.append(f'<text x="{x + 64 + min(w - 108, 130) + 6}" y="{ty:.1f}" font-size="9.5" fill="#fbbf24">{"" if val else "未計測"}</text>')
        else:
            out.append(f'<text x="{x + 8}" y="{ty:.1f}" font-size="{FS}" fill="{c}">{esc(t)}</text>')
    return "".join(out), h


def main_svg(times: dict[str, list[str]]) -> str:
    TXT, MUT, TIME = "#e6ecff", "#9aa7c7", "#ffd479"
    parts: list[str] = []
    W = COLS["tgt"][0] + COLS["tgt"][1] + 10
    heads = [("src", "データ"), ("collect", "取得（cron・JST）"), ("store", "保存"), ("proc", "加工（cron・JST）"), ("tgt", "統合・利用")]
    for k, t in heads:
        x, w = COLS[k]
        parts.append(f'<text x="{x + 4}" y="22" font-size="13" font-weight="700" fill="#7aa9ff">{esc(t)}</text>')
    y = 36.0
    row_pos: list[tuple[Row, float, float]] = []
    for r in ROWS:
        c0 = [(l, TXT if i == 0 else MUT) for i, l in enumerate(wrap(r.label, COLS["src"][1] - 14, FS))] + \
             [(l, MUT) for l in wrap(r.src, COLS["src"][1] - 14, FS - 1.5)]
        c1: list[tuple[str, str]] = []
        for key, desc in r.collect:
            for i, l in enumerate(wrap(desc, COLS["collect"][1] - 14, FS)):
                c1.append((l, TXT))
            c1.append(("時刻 " + tstr(times, key), TIME))
        if r.collect:
            c1.append(dur_line(f"collect:{r.label}"))
        else:
            c1 = [("（取得ではなく集計）", MUT)]
        c2 = [(l, TXT) for l in wrap(r.store, COLS["store"][1] - 14, FS)]
        c3: list[tuple[str, str]] = []
        for key, desc in r.process:
            for l in wrap(desc, COLS["proc"][1] - 14, FS):
                c3.append((l, TXT))
            if key:
                c3.append(("時刻 " + tstr(times, key), TIME))
        if r.process:
            c3.append(dur_line(f"process:{r.label}"))
        else:
            c3 = [("（加工ジョブなし）", MUT)]
        h = max(len(c) for c in (c0, c1, c2, c3)) * LH + 12
        for k, lines in (("src", c0), ("collect", c1), ("store", c2), ("proc", c3)):
            x, w = COLS[k]
            fill = "#15213c" if k != "src" else "#101a33"
            s, _ = box(x, y, w, lines, fill, "#243358", h)
            parts.append(s)
        # 矢印: src→collect→store→proc
        cy = y + h / 2
        for a, b in (("src", "collect"), ("collect", "store"), ("store", "proc")):
            x1 = COLS[a][0] + COLS[a][1]
            x2 = COLS[b][0]
            parts.append(f'<line x1="{x1}" y1="{cy}" x2="{x2 - 1}" y2="{cy}" stroke="#7aa9ff" stroke-width="1.4" marker-end="url(#ah)"/>')
        row_pos.append((r, y, h))
        y += h + 9
    total_h = y
    # 統合先の箱: 行の中央付近に等間隔で配置
    tx, tw = COLS["tgt"]
    n = len(TARGETS)
    tgt_h = [(len(wrap(TARGETS[k][1], tw - 16, FS - 1)) + len(wrap(TARGETS[k][0], tw - 16, FS)) + 2) * LH + 14 for k in TARGETS]
    gap = (total_h - 36 - sum(tgt_h)) / (n + 1)
    ty = 36 + gap
    tpos: dict[str, tuple[float, float]] = {}
    for (k, (title, desc, color)), th in zip(TARGETS.items(), tgt_h):
        lines = [(l, TXT) for l in wrap(title, tw - 16, FS)] + [(l, MUT) for l in wrap(desc, tw - 16, FS - 1)]
        lines.append(("時刻 " + tstr(times, TARGET_JOB[k]) if k in TARGET_JOB else "cron なし（手動で実行）", TIME if k in TARGET_JOB else MUT))
        lines.append(dur_line(f"target:{k}"))
        s, _ = box(tx, ty, tw, lines, "#17233f", color, th)
        parts.append(s)
        tpos[k] = (ty, th)
        ty += th + gap
    # 加工 → 統合 の線（統合先ごとの色。同じ箱に入る線は箱の左辺に束ねる）
    used: dict[str, int] = {}
    for r, ry, rh in row_pos:
        for t in r.targets:
            color = TARGETS[t][2]
            by, bh = tpos[t]
            k = used.get(t, 0)
            used[t] = k + 1
            ty_in = by + 10 + (k * 7) % max(bh - 20, 10)
            x1 = COLS["proc"][0] + COLS["proc"][1]
            ym = ry + rh / 2
            xm = x1 + 8 + (list(TARGETS).index(t)) * 5
            parts.append(f'<polyline points="{x1},{ym} {xm},{ym} {xm},{ty_in} {tx - 1},{ty_in}" fill="none" stroke="{color}" stroke-width="1.3" stroke-opacity="0.85" marker-end="url(#ah)"/>')
    return (f'<svg viewBox="0 0 {W} {total_h + 6:.0f}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans JP, system-ui, sans-serif" role="img" aria-label="データ収集から統合までのフロー">'
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#7aa9ff"/></marker></defs>'
            + "".join(parts) + "</svg>")


def _minutes(when: str) -> list[int]:
    """'毎日 07:30' / '金 18:00' / '毎日 05:00〜08:50（10分ごと）' → 当日中の発生分（代表点）。"""
    out = []
    for m in re.finditer(r"(\d{2}):(\d{2})", when):
        out.append(int(m.group(1)) * 60 + int(m.group(2)))
    return out


def timeline_svg(times: dict[str, list[str]]) -> str:
    X0, PXH = 90, 44.0
    W = X0 + 24 * PXH + 20
    parts = []
    for h in range(0, 25):
        x = X0 + h * PXH
        parts.append(f'<line x1="{x}" y1="30" x2="{x}" y2="{{BOT}}" stroke="#243358" stroke-width="{1.2 if h % 6 == 0 else 0.6}"/>')
        if h % 2 == 0:
            parts.append(f'<text x="{x}" y="22" font-size="11" text-anchor="middle" fill="#9aa7c7">{h:02d}</text>')
    y = 40.0
    for lane, keys in LANES:
        items = []
        for key in keys:
            for w in times.get(key, []):
                ms = _minutes(w)
                if not ms:
                    continue
                weekly = not w.startswith("毎日")
                items.append((ms[0], (ms[-1] if "〜" in w else None), SHORT.get(key, key), weekly, w))
        items.sort()
        tiers: list[float] = []   # 各段の最後のラベル右端 x
        placed = []
        for start, end, name, weekly, w in items:
            x = X0 + start / 60 * PXH
            label = f"{name}{'（金）' if weekly else ''} {start // 60:02d}:{start % 60:02d}"
            lw = sum(_w(c) * 10.5 for c in label) + 8
            tier = next((i for i, right in enumerate(tiers) if x - 4 > right), None)
            if tier is None:
                tiers.append(0.0)
                tier = len(tiers) - 1
            x_end = X0 + end / 60 * PXH if end else x
            tiers[tier] = max(x + lw, x_end)
            placed.append((x, end, tier, label, weekly))
        lane_h = max(len(tiers), 1) * 19 + 14
        parts.append(f'<rect x="4" y="{y}" width="{W - 8}" height="{lane_h}" rx="6" fill="#101a33" fill-opacity="0.55"/>')
        parts.append(f'<text x="10" y="{y + 16}" font-size="12" font-weight="700" fill="#7aa9ff">{esc(lane)}</text>')
        for x, end, tier, label, weekly in placed:
            yy = y + 8 + tier * 19
            col = "#fbbf24" if weekly else "#34d399"
            if end:
                parts.append(f'<rect x="{x:.1f}" y="{yy + 4}" width="{(end - 0) / 60 * PXH + X0 - x:.1f}" height="8" rx="3" fill="{col}" fill-opacity="0.35"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{yy + 8}" r="3.5" fill="{col}"/>')
            parts.append(f'<text x="{x + 6:.1f}" y="{yy + 12}" font-size="10.5" fill="#e6ecff">{esc(label)}</text>')
        y += lane_h + 8
    bot = y
    body = "".join(parts).replace("{BOT}", f"{bot:.0f}")
    return (f'<svg viewBox="0 0 {W:.0f} {bot + 6:.0f}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans JP, system-ui, sans-serif" role="img" aria-label="1日のタイムライン">'
            + body + "</svg>")


# ── HTML ───────────────────────────────────────────────────────────────────────

CSS = """
:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#7aa9ff; --warn:#fbbf24; --ok:#34d399; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.7 "Noto Sans JP",system-ui,sans-serif; }
main { max-width:1280px; margin:0 auto; padding:36px 28px 80px; }
h1 { font-size:26px; margin:0 0 6px; } h2 { font-size:19px; margin:38px 0 6px; padding-top:16px; border-top:1px solid var(--border); }
p, li { color:var(--text); } .muted { color:var(--muted); font-size:13px; }
table { border-collapse:collapse; width:100%; margin:10px 0 20px; font-size:13.5px; }
th, td { border:1px solid var(--border); padding:6px 9px; text-align:left; vertical-align:top; } th { background:var(--panel); color:var(--accent); }
td.t { color:#ffd479; white-space:nowrap; } code { background:#0b1428; padding:1px 5px; border-radius:4px; font-size:12.5px; }
.box { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:12px 16px; margin:12px 0; }
.legend span { display:inline-block; margin-right:16px; } a { color:var(--accent); }
td.tbd { background:#2a2410; border:1px dashed var(--warn); min-width:64px; } td.na { color:var(--muted); text-align:center; }
.tbd-note { border-color:var(--warn); background:#241f0e; }
"""


def render(jobs, source: str) -> str:
    times = job_times(jobs)
    o = ['<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
         "<title>データの収集〜加工〜統合フロー（cron 時刻つき） — keiba-vpn docs</title>", f"<style>{CSS}</style></head><body><main>",
         "<h1>データの収集〜加工〜統合フロー（cron 時刻つき）</h1>",
         f'<p class="muted">時刻はすべて <b>JST</b>。cron の登録内容（{esc(source)}）から生成: {datetime.now(JST):%Y-%m-%d %H:%M}。'
         "再生成: <code>python -m src.scripts.docs.build_data_flow_html</code>。cron の一覧は <a href=\"../../operations/cron_schedule.html\">cron 登録内容</a>、"
         "特徴量とモデルの設計は <a href=\"feature_strategy_tp03_tp04.html\">特徴量戦略書</a>。</p>",
         '<div class="box legend"><b>読み方</b>　左から「データ → 取得するジョブと時刻 → 保存先 → 加工するジョブと時刻 → 統合・利用」。'
         '<span style="color:#ffd479">時刻 黄色</span> が cron の発生時刻。右端の線の色は統合先（'
         + "".join(f'<span style="color:{c}">■ {esc(t)}</span>' for t, _, c in TARGETS.values()) + ")。"
         "<br>cron で動くのは取得と、一部の加工（flat 表・履歴特徴量の蓄積）まで。<b>学習（tp-01〜tp-07）は手動</b>で、cron には載っていない。</div>",
         '<div class="box tbd-note"><b>処理時間は未計測（空欄）</b>　各工程の処理時間（ダウンタイム）は<b>まだ計測しておらず、空欄にしてある</b>。図の黄色い破線の枠と、一覧表の「処理時間」列が記入欄。'
         "計測したら <code>DURATIONS</code>（<code>src/scripts/docs/build_data_flow_html.py</code>）に書いて再生成する。"
         "<b>工程の分け方・時刻・処理時間は今後変更する可能性がある</b>（この図は現時点の暫定版）。</div>",
         "<h2>1. データごとの流れ</h2>", main_svg(times),
         "<h2>2. 1日のタイムライン</h2>",
         '<p class="muted">緑 = 毎日、黄 = 曜日指定（金曜など）。帯は連続して動く時間帯（10分ごとのポーリング）。開催日でない日は、ランナー系のジョブは数秒で終わる。</p>',
         timeline_svg(times),
         "<h2>3. 一覧（データ別）</h2>",
         "<table><tr><th>データ</th><th>取得ジョブと時刻（JST）</th><th>取得の処理時間<br><span class='muted'>未計測</span></th><th>保存先</th><th>加工ジョブと時刻（JST）</th><th>加工の処理時間<br><span class='muted'>未計測</span></th><th>統合・利用</th><th>補足</th></tr>"]
    for r in ROWS:
        c = "<br>".join(f"{esc(d)}<br><span style='color:#ffd479'>{esc(tstr(times, k))}</span>" for k, d in r.collect) or "—"
        p = "<br>".join(f"{esc(d)}" + (f"<br><span style='color:#ffd479'>{esc(tstr(times, k))}</span>" if k else "") for k, d in r.process) or "—"
        tg = "、".join(f"<span style='color:{TARGETS[t][2]}'>{esc(TARGETS[t][0])}</span>" for t in r.targets) or "（現行の学習・推論では未使用）"
        dc = f"<td class='tbd'>{esc(DURATIONS.get('collect:' + r.label, ''))}</td>" if r.collect else "<td class='na'>—</td>"
        dp = f"<td class='tbd'>{esc(DURATIONS.get('process:' + r.label, ''))}</td>" if r.process else "<td class='na'>—</td>"
        o.append(f"<tr><td><b>{esc(r.label)}</b><br><span class='muted'>{esc(r.src)}</span></td><td>{c}</td>{dc}<td>{esc(r.store)}</td><td>{p}</td>{dp}<td>{tg}</td><td>{esc(r.note)}</td></tr>")
    o.append("</table>")
    o.append("<h3>統合・利用の処理時間（未計測）</h3><table><tr><th>統合先</th><th>起動</th><th>処理時間<br><span class='muted'>未計測</span></th></tr>"
             + "".join(f"<tr><td style='color:{c}'><b>{esc(n)}</b></td><td>{esc('時刻 ' + tstr(times, TARGET_JOB[k]) if k in TARGET_JOB else 'cron なし（手動で実行）')}</td>"
                       f"<td class='tbd'>{esc(DURATIONS.get('target:' + k, ''))}</td></tr>" for k, (n, _, c) in TARGETS.items())
             + "</table>")
    o += ["<h2>4. 毎日の流れ（結果が特徴量になるまで）</h2><ol>"
          f"<li>前日の結果: 夕方 {esc(tstr(times, 'raceday-evening'))} に速報結果、翌朝 {esc(tstr(times, 'recent-retry').split(' ／ ')[0])} の取り直しで確定結果・ラップ・指数が揃う。</li>"
          f"<li>flat 表: {esc(tstr(times, 'export_tables_daily'))} に当年分を増分更新（race_result・race_shutuba・race_result_lap・race_result_on_time）。</li>"
          f"<li>履歴特徴量: {esc(tstr(times, 'feature-accumulate'))} に当年分の127列を作り直し、推論モードの特徴量とのずれを記録（flat 表の更新後に動く順序）。</li>"
          f"<li>当日: {esc(tstr(times, 'raceday-weight-runner'))} に検知ランナーが起動し、各R 発走65〜30分前に出馬表を取り直す。馬体重が揃ったら推論モードで特徴量を作る。</li>"
          f"<li>オッズ: {esc(tstr(times, 'raceday-runner'))} に起動したランナーが、各R 発走60分前・15分前に取得する（T-15 は出馬表・馬柱なども）。</li></ol>",
          "<h2>5. cron に載っていない工程（手動）</h2><ul>"
          "<li>学習: tp-01（母表・分割）→ tp-02（履歴特徴量）→ tp-03（ablation）→ tp-04（前段モデル）→ tp-05（最終モデル）→ tp-06（較正）→ tp-07（最終再学習・登録）。ノートブックを手動で実行する。</li>"
          "<li>flat 表の更新は4カテゴリのみ（race_result・race_shutuba・race_result_lap・race_result_on_time）。race_index・race_shutuba_past・race_oikiri・race_odds・race_paddock・race_barometer・race_trainer_comment の flat 表は手動で export する。</li>"
          "<li>予測の起動点は2系統（T-45 の疑似特徴量ビルダー、T-15 のモック）が残っている。馬体重の検知ランナーは既定で予測を起動せず、特徴量の作成までを行う。</li>"
          "<li>優先処理（2020〜2025 年の取り直し）が動いている間は、同じ netkeiba アカウントを使う定期ジョブ（レース一覧・週次更新・バロメーター・recent-retry・backfill）を自動で見送る。</li></ul>",
          "<h2>6. 整合性の注意</h2><ul>"
          "<li>2026 年のデータは学習に使わない（2020〜2025 年のみ）。</li>"
          "<li>オッズ・人気は母表（LayerA）に入れない。入るのは LayerB（期待値の計算）だけ。</li>"
          "<li>この図の時刻は cron の起動時刻。実際にデータが公開される時刻（馬体重・バロメーターなど）は、データの可用時点の表（特徴量戦略書の補足）を参照。</li></ul>",
          "</main></body></html>"]
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crontab-file", type=Path, help="`crontab -l` の保存先。省略時は setup_all_cron.sh show の出力")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.crontab_file:
        text, source = args.crontab_file.read_text(encoding="utf-8"), str(args.crontab_file)
    else:
        text = subprocess.run(["bash", str(REPO_ROOT / "scripts/cron/setup_all_cron.sh"), "show"], capture_output=True, text=True, check=True).stdout
        source = "scripts/cron/setup_all_cron.sh show"
    jobs, _ = parse(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(jobs, source), encoding="utf-8")
    print(f"{args.out}: {len(jobs)} ジョブから生成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
