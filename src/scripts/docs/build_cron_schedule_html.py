#!/usr/bin/env python3
"""cron の登録内容を、役割ごとにグループ分けした HTML にする（docs/html/operations/cron_schedule.html）。

正本は ``scripts/cron/setup_all_cron.sh``。既定ではその ``show`` の出力を読む（cron の無い PC でも作れる）。
ホストの実際の登録内容から作るなら ``--crontab-file`` に ``crontab -l`` の保存先を渡す。

  python -m src.scripts.docs.build_cron_schedule_html
  python -m src.scripts.docs.build_cron_schedule_html --crontab-file /tmp/root_cron.txt --out /tmp/cron.html
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "docs" / "html" / "operations" / "cron_schedule.html"
JST = timezone(timedelta(hours=9))
DOW = ["日", "月", "火", "水", "木", "金", "土"]

# ジョブのキー → (グループ, 名前, 説明)。キーは run_auto_scrape_logged.sh のタスク名、または下の _job_key が返す名前。
GROUPS = [
    ("live", "開催日のリアルタイム取得", "開催日の前日夕方〜当日に、出走表・オッズ・結果・馬場を取りに行く。"),
    ("refresh", "定期更新・取りこぼしの取り直し", "レース一覧・確定結果・バロメーターなど、公開が遅れるデータの更新と再取得。"),
    ("backfill", "過去データの補填（backfill）", "年度別に、取りこぼした結果・出走表・馬情報を深夜に補う。"),
    ("prep", "推論・学習用のデータ整備", "特徴量や推論が読むテーブル・統計・DB を更新する。"),
    ("ops", "保守・監視", "サーバの死活監視、ログの整理、ページ構造の品質チェック。"),
]
GCS_NOTE = "GCS（正本）"
JOBS: dict[str, tuple[str, str, str, str, str]] = {
    # key: (グループ, 名前, 内容, 加工, 保存先)
    "raceday-eve": ("live", "前日夕方の出馬表・馬柱・追い切り", "翌日が開催日のときだけ動く。枠順・馬柱・追い切り・SmartRC を取得",
                    "構造化 + 推定（追走難度・想定オッズ。モデルによる計算結果を保存）",
                    "GCS: 出馬表・馬柱・追い切り・SmartRC は構造化JSON、追走難度・想定オッズは推定結果のキャッシュ。ローカル: 取得状況 data/local/meta/date_coverage、発走時刻表 data/page_reference/race_day_schedule（ローカルのみ）。DB 同期を起動（ホストでは失敗中）"),
    "raceday-runner": ("live", "開催日ランナー（オッズ・出馬表）", "開催日のみ常駐。各R 発走60分前にオッズ、15分前に出馬表・オッズ・馬柱などをまとめて取得",
                       "構造化のみ（T-15 の予測トリガはモックで保存なし）",
                       "GCS: 出馬表・オッズ（T-60/T-15 の履歴を同じJSONに保持）・馬柱・追い切り・SmartRC。JRA 馬場はローカル data/page_reference/cushion"),
    "raceday-result-runner": ("live", "速報結果ランナー", "開催日のみ常駐。各R 発走15分後に速報結果を取得",
                              "構造化のみ。全R後に結果テーブルの増分追記と馬場速度指数の計算",
                              "GCS: 速報結果。ローカル: data/page_reference/tables（race_result_flat の増分）、data/analysis/track_speed（馬場速度指数）"),
    "raceday-weight-runner": ("live", "馬体重の検知ランナー", "各R 発走65〜30分前に出馬表を取り直し、馬体重が揃ったら推論モードで特徴量を作る（予測は起動しない）",
                              "構造化（出馬表）+ 計算（127特徴量）",
                              "GCS: 出馬表。ローカルのみ: data/local/inference（特徴量 parquet・予測時点の出馬表スナップショット・イベント記録。GCS には上げない）"),
    "raceday-evening": ("live", "開催日夕方の結果・確定オッズ", "速報結果・確定オッズ・SmartRC のまとめ取得",
                        "構造化のみ", "GCS: 速報結果・確定オッズ・SmartRC。ローカル: 取得状況 date_coverage。DB 同期を起動（ホストでは失敗中）"),
    "jra-baba-morning": ("live", "JRA 馬場（クッション値・含水率）", "開催日の朝に10分ごとにポーリング",
                         "構造化（JRA 公式ページを整形したJSON）",
                         "ローカルのみ: data/page_reference/cushion。GCS へは自動では上げない（手動アップロード）"),
    "daily-race-lists": ("refresh", "レース一覧", "今日以降の全開催日のレース一覧を更新（朝・夕の2回）",
                         "構造化のみ", "ローカルのみ: data/page_reference/race_lists。キューにレース単位のジョブが溜まっていれば、その分は GCS へ"),
    "weekly-update": ("refresh", "週次更新", "直近10日の確定結果・タイム指数・バロメーター・馬情報",
                      "構造化 + 計算（めぐ指数）+ DB 取り込み",
                      "GCS: 確定結果・指数・バロメーター・馬情報。PostgreSQL: ETL（ホストでは DB 未起動・psycopg2 未導入のため失敗中）"),
    "horse-name-index": ("refresh", "馬名インデックス", "馬名リストと成長曲線（週次更新の後）",
                         "計算（馬名リスト・成長曲線の集計）", "ローカルのみ: data/calculated_data"),
    "barometer-retry": ("refresh", "バロメーターの取り直し", "公開済みで未取得のものだけを取得",
                        "構造化のみ", "GCS: race_barometer"),
    "recent-retry": ("refresh", "直近の未取得データの取り直し", "直近21日の確定結果・結果ラップ・タイム指数・厩舎コメントのうち未取得のものを、間隔と回数を制限して取得",
                     "構造化のみ", "GCS: race_result・race_result_lap・race_index・race_trainer_comment。ローカル: 試行状態 data/local/meta/recent_retry_state.json"),
    "backfill": ("backfill", "backfill", "",
                 "構造化のみ", "GCS: 同じカテゴリ（馬情報は horse_result）。ローカル: 進捗台帳 data/local/meta/backfill_progress.json、取得状況 date_coverage"),
    "export_tables_daily": ("prep", "履歴テーブルの増分更新", "推論が読む race_result_flat などを当年分だけ更新",
                            "加工（JSON を表に平坦化。1行=1馬の _flat と 1行=1レースの _race の Parquet）", "ローカルのみ: data/page_reference/tables/<年>/"),
    "feature-accumulate": ("prep", "履歴特徴量の蓄積", "当年分の特徴量（127列）を確定結果から作り直し、推論モードで保存した特徴量とのずれを記録",
                           "計算（as-of の履歴特徴量。学習と同じ計算）",
                           "ローカルのみ: data/local/modeling/feature_store/horse_history（features_<年>.parquet、manifest、skew/<日付>.json）。2026年は学習に使わない"),
    "update_jockey_trainer_stats": ("prep", "騎手・調教師統計", "統計テーブルを再生成",
                                    "計算（成績の集計）", "ローカルのみ: data/features/jockey_trainer_stats ほか（Parquet）"),
    "sync_pg_from_gcs": ("prep", "GCS → PostgreSQL 同期", "直近7日の増分",
                         "加工（ETL で正規化テーブルへ）", "PostgreSQL（ホストでは毎晩失敗: DB 未起動・psycopg2 未導入）"),
    "server_watchdog": ("ops", "サーバ死活監視", "FastAPI・Flask・Next.js・MLflow を監視し、落ちていれば再起動", "データなし", "—"),
    "rotate_logs": ("ops", "ログローテーション", "ログの整理", "データなし", "—"),
    "page_quality_check": ("ops", "ページ品質チェック", "スクレイパーが各ページの期待スキーマを取得できるかを確認",
                           "検査のみ", "ローカル: data/local/page_quality/latest.json（過去30件は history/）"),
}
# 優先処理（2020-2025年の取り直し）が動いている間は見送るジョブ
YIELDS = {"daily-race-lists", "weekly-update", "barometer-retry", "recent-retry", "backfill"}


@dataclass
class Job:
    key: str
    group: str
    name: str
    desc: str
    schedule: str          # cron の時刻部分（UTC）
    when: str              # 人が読む形（JST）
    sort_minutes: int      # 1日の流れの並べ替え用（JST の 0:00 からの分）
    yields: bool
    command: str
    process: str = ""
    dest: str = ""


def _expand(field: str, lo: int, hi: int) -> list[int]:
    out: list[int] = []
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part in ("*", ""):
            a, b = lo, hi
        elif "-" in part:
            a, b = (int(x) for x in part.split("-"))
        else:
            a = b = int(part)
        out.extend(range(a, b + 1, step))
    return sorted(set(out))


def to_jst(minute: str, hour: str, dow: str) -> tuple[str, int]:
    """cron（UTC）の分・時・曜日を、JST の「毎日 07:30」のような文字列と並べ替え用の分にする。"""
    mins = _expand(minute, 0, 59)
    hours = _expand(hour, 0, 23)
    days = _expand(dow, 0, 6) if dow != "*" else None
    shifted = []
    for h in hours:
        jh = (h + 9) % 24
        carry = (h + 9) // 24
        shifted.append((jh, carry))
    jdays = None
    if days is not None:
        carries = {c for _, c in shifted}
        jdays = sorted({(d + c) % 7 for d in days for c in carries}) if len(carries) == 1 else sorted({(d + shifted[0][1]) % 7 for d in days})
    day_txt = "毎日" if jdays is None else "・".join(DOW[d] for d in jdays)
    if minute.startswith("*/") and len(hours) == 24:
        return f"{day_txt} 終日 {mins[1] - mins[0]}分ごと", 0
    if len(mins) > 1 and len(hours) > 1:
        a, b = sorted(shifted)[0][0], sorted(shifted)[-1][0]
        step = mins[1] - mins[0]
        if minute.startswith("*/"):
            jhs = sorted(h for h, _ in shifted)
            return f"{day_txt} {jhs[0]:02d}:00〜{jhs[-1]:02d}:50（{step}分ごと）", jhs[0] * 60
        return f"{day_txt} " + "・".join(f"{h:02d}:{m:02d}" for h, _ in sorted(shifted) for m in mins), sorted(shifted)[0][0] * 60 + mins[0]
    if len(mins) > 1:
        return f"{day_txt} 毎時 " + "・".join(f":{m:02d}" for m in mins), mins[0]
    hs = sorted(h for h, _ in shifted)
    if len(hs) == 24:
        return f"{day_txt} {mins[0]}分ごと（毎時）" if minute != "*" else f"{day_txt} 1分ごと", 0
    return f"{day_txt} " + "・".join(f"{h:02d}:{mins[0]:02d}" for h in hs), hs[0] * 60 + mins[0]


_TASK = re.compile(r"run_auto_scrape_logged\.sh\s+\S+\s+(\S+)")
_BACKFILL = re.compile(r"src\.scraper\.backfill\s+(.*?)(?:>>|$)")


def _job_key(cmd: str) -> tuple[str, str]:
    """コマンド文字列から (キー, 補足) を返す。"""
    if "src.scraper.backfill" in cmd:
        m = _BACKFILL.search(cmd)
        args = (m.group(1) if m else "").strip()
        y = re.search(r"--year (\d{4})", args)
        ph = re.search(r"--phase (\w+)", args)
        label = (f"{y.group(1)}年 " if y else "") + {"fast": "結果+出馬表", "full": "補助データ", "horse": "馬情報"}.get(ph.group(1) if ph else "", "")
        return "backfill", label.strip()
    m = _TASK.search(cmd)
    if m:
        return m.group(1), ""
    for k in ("server_watchdog", "rotate_logs", "update_jockey_trainer_stats", "sync_pg_from_gcs", "export_tables_daily", "page_quality_check"):
        if k in cmd:
            return k, ""
    return "", ""


def parse(crontab_text: str) -> tuple[list[Job], list[str]]:
    jobs: list[Job] = []
    others: list[str] = []
    for raw in crontab_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("PATH=") or line.startswith("CRON_TZ="):
            continue
        if line.startswith("@reboot"):
            key, extra = _job_key(line)
            if key in JOBS:
                g, n, d, pr, ds = JOBS[key]
                jobs.append(Job(key, g, n + "（起動時）", d, "@reboot", "システム起動時", -1, False, line, pr, ds))
            else:
                others.append(line)
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            others.append(line)
            continue
        minute, hour, _dom, _mon, dow, cmd = parts
        key, extra = _job_key(cmd)
        if key not in JOBS:
            others.append(line)
            continue
        g, n, d, pr, ds = JOBS[key]
        if key == "backfill":
            n, d = f"backfill: {extra}", f"{extra}を補填"
        when, sm = to_jst(minute, hour, dow)
        jobs.append(Job(key, g, n, d, f"{minute} {hour} * * {dow}", when, sm, key in YIELDS or "yield_to_rescrape.sh" in cmd, cmd, pr, ds))
    return jobs, others


CSS = """
:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --code:#0b1428; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.7 "Noto Sans JP",system-ui,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:36px 28px 80px; }
h1 { font-size:26px; margin:0 0 6px; } h2 { font-size:19px; margin:40px 0 4px; padding-top:16px; border-top:1px solid var(--border); }
.muted { color:var(--muted); font-size:13px; } .sub { color:var(--muted); font-size:13.5px; margin:0 0 8px; }
table { border-collapse:collapse; width:100%; margin:8px 0 14px; font-size:13.5px; }
th,td { border:1px solid var(--border); padding:6px 10px; text-align:left; vertical-align:top; }
th { background:var(--panel); color:var(--muted); font-weight:600; white-space:nowrap; }
code { font-family:"JetBrains Mono",ui-monospace,Menlo,monospace; background:var(--code); padding:1px 6px; border-radius:5px; font-size:12px; }
.badge { display:inline-block; font-size:11.5px; padding:0 8px; border-radius:10px; border:1px solid var(--warn); color:var(--warn); white-space:nowrap; }
.box { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px 18px; margin:12px 0; }
.when { white-space:nowrap; font-weight:600; }
"""


COMMON_FLOW = (
    '<h2>取得データの扱い（共通）</h2>'
    '<p class="sub">netkeiba を取得する全ジョブ（出馬表・オッズ・結果・指数など）は、同じ流れで保存する。</p>'
    '<table><tr><th>段階</th><th>内容</th></tr>'
    '<tr><td>1. 生のHTML</td><td><b>原則、保存しない。</b>ページ構造の変更を検出するためのサンプルだけを GCS に gzip で置く（カテゴリごとに最大 10 件）。ローカルには残さない。</td></tr>'
    '<tr><td>2. 構造化</td><td>パーサーが HTML から項目を取り出し、<b>JSON にする</b>（出走馬の行、オッズ、着順・時計など）。値は画面に出ているまま。集計や派生量は作らない。取得時刻とスキーマ検証の結果（<code>_meta</code>）を付ける。</td></tr>'
    '<tr><td>3. 検証</td><td>スキーマに合わない場合は GCS に<b>保存せず</b>、ローカルの <code>data/local/quarantine/</code> に隔離し、違反の記録を <code>data/local/meta/schema_violations/</code> に残す（既定）。</td></tr>'
    '<tr><td>4. 保存先（正本）</td><td><b>GCS</b>（<code>chuou/data/preprocessed/netkeiba/pc/&lt;カテゴリ&gt;/&lt;年 または馬IDの先頭4桁&gt;/&lt;ID&gt;.json</code>）。'
    'ローカルには、メモリキャッシュと、週に 2 回以上読まれたキーだけの <code>data/cache/</code> を補助として置く。GCS と同じ形のミラー（<code>data/local/mirror/</code>）は設定で無効（ホストでは未設定）。</td></tr>'
    '<tr><td>例外（ローカルのみ）</td><td>レース一覧（<code>data/page_reference/race_lists</code>）と発走時刻表（<code>data/page_reference/race_day_schedule</code>）は GCS に置かない。JRA 馬場（<code>data/page_reference/cushion</code>）もローカルで、GCS へは手動でアップロードする。</td></tr>'
    '<tr><td>加工して別の場所に置くもの</td><td>取得とは別の段階で作る。履歴テーブル（Parquet、<code>data/page_reference/tables</code>）、騎手・調教師統計（<code>data/features</code>）、馬場速度指数（<code>data/analysis/track_speed</code>）、推論用の特徴量（<code>data/local/inference</code>）、DB への取り込み（PostgreSQL）。いずれも GCS の構造化JSONを元に作る。</td></tr>'
    '</table>'
)

def render(jobs: list[Job], others: list[str], source: str) -> str:
    esc = html.escape
    out = ['<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           "<title>cron 登録内容（グループ別） — keiba-vpn docs</title>", f"<style>{CSS}</style></head><body><main>",
           "<h1>cron 登録内容（グループ別）</h1>",
           f'<p class="muted">時刻はすべて <b>JST</b>（crontab 本体は UTC で記述）。生成: {datetime.now(JST):%Y-%m-%d %H:%M} / 元データ: {esc(source)} / '
           "再生成: <code>python -m src.scripts.docs.build_cron_schedule_html</code>。定義の正は <code>scripts/cron/setup_all_cron.sh</code>。</p>",
           '<div class="box"><span class="badge">取り直し中は見送り</span> のジョブは、優先処理（2020-2025年の取り直し。<code>run_priority_rescrape.sh</code>）が動いている間は自動で見送り、'
           "終われば自動で通常どおり動く（<code>scripts/cron/yield_to_rescrape.sh</code>。同じ netkeiba アカウントの別セッションが取り直し側のセッションを無効にするため）。"
           "見送りの記録は <code>logs/yield_skips.log</code>。止めるには <code>data/local/meta/rescrape/YIELD_DISABLED</code> を置く。</div>"]

    out.append(COMMON_FLOW)
    out.append("<h2>1日の流れ（JST）</h2><p class=\"sub\">毎日動くジョブを時刻順に並べたもの。曜日指定のあるものは右端に曜日を付けた。</p>")
    out.append("<table><tr><th>時刻</th><th>ジョブ</th><th>グループ</th></tr>")
    gname = {k: n for k, n, _ in GROUPS}
    for j in sorted((j for j in jobs if j.sort_minutes >= 0), key=lambda j: (j.sort_minutes, j.name)):
        out.append(f'<tr><td class="when">{esc(j.when)}</td><td>{esc(j.name)}</td><td>{esc(gname[j.group])}</td></tr>')
    out.append("</table>")

    for gid, gtitle, gdesc in GROUPS:
        gj = sorted((j for j in jobs if j.group == gid), key=lambda j: (j.sort_minutes, j.name))
        if not gj:
            continue
        out.append(f"<h2>{esc(gtitle)}（{len(gj)}件）</h2><p class=\"sub\">{esc(gdesc)}</p>")
        out.append("<table><tr><th>時刻（JST）</th><th>ジョブ</th><th>内容</th><th>データの加工</th><th>保存先</th><th>cron（UTC）</th><th></th></tr>")
        for j in gj:
            badge = '<span class="badge">取り直し中は見送り</span>' if j.yields else ""
            out.append(f'<tr><td class="when">{esc(j.when)}</td><td>{esc(j.name)}</td><td>{esc(j.desc)}</td><td>{esc(j.process)}</td><td>{esc(j.dest)}</td><td><code>{esc(j.schedule)}</code></td><td>{badge}</td></tr>')
        out.append("</table>")
    if others:
        out.append("<h2>その他（グループ外）</h2><p class=\"sub\">このスクリプトの対象外のジョブ（別ツールが登録したもの）。</p><table><tr><th>登録行</th></tr>")
        out += [f"<tr><td><code>{esc(o)}</code></td></tr>" for o in others]
        out.append("</table>")
    out.append("</main></body></html>")
    return "\n".join(out)


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
    jobs, others = parse(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(jobs, others, source), encoding="utf-8")
    print(f"{args.out}: {len(jobs)} ジョブ / グループ外 {len(others)} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
