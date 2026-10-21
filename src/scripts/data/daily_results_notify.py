"""その日のレース結果が揃ったときに、予測成績の集計・推奨馬券の回収率と的中率を Slack に通知する。

入力は ``data/local/prediction_results/<YYYYMMDD>.json``（日次の集計。``src.pipeline.inference.prediction_results``）。
結果が揃った（status=final かつ未取得レースが無い）ときに 1 日 1 回だけ送る（送信済みフラグ ``data/local/meta/slack_sent/``）。

  python -m src.scripts.data.daily_results_notify --date 20261018              # dry-run（既定。送らずに JSON とプレビューを出力）
  python -m src.scripts.data.daily_results_notify --date 20261018 --send       # 実送信（KEIBA_SLACK_NOTIFY=1 と Webhook の環境変数が必要）
  python -m src.scripts.data.daily_results_notify --send-test                  # 接続テスト（最小メッセージ。実行はユーザーの確認後）

終了コード: 0=送信/dry-run 完了, 10=まだ結果が揃っていない（cron が再試行）, 11=送信済み, 12=開催なし/集計なし, 13=無効（KEIBA_SLACK_NOTIFY 未設定）,
            14=送信失敗, 1=その他のエラー。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.pipeline.inference import prediction_results as PR
from src.utils import slack_notify as SN

REPO_ROOT = Path(__file__).resolve().parents[3]
JST = timezone(timedelta(hours=9))
BET_ORDER = ["tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku"]
MARK_ROWS = [("honmei", "◎"), ("pair", "○"), ("star", "★"), ("kesi", "消")]
EXIT_OK, EXIT_NOT_READY, EXIT_SENT, EXIT_NONE, EXIT_DISABLED, EXIT_FAILED = 0, 10, 11, 12, 13, 14


def enabled(env: dict[str, str] | None = None) -> bool:
    return str((os.environ if env is None else env).get("KEIBA_SLACK_NOTIFY", "")).strip() == "1"


# ───────────────────────── 結果が揃ったか・送信済みか ─────────────────────────

def check_ready(doc: dict | None, *, allow_provisional: bool = False) -> tuple[bool, str]:
    """(揃ったか, 理由)。理由: ok / no_doc / no_races / has_no_result / not_final。"""
    if not doc:
        return False, "no_doc"
    s = doc.get("summary") or {}
    if not s.get("races_settled"):
        return False, "no_races"
    if s.get("races_no_result"):
        return False, "has_no_result"
    if doc.get("status") != "final" and not allow_provisional:
        return False, "not_final"
    return True, "ok"


def sent_path(date: str, sent_dir: Path | None = None) -> Path:
    return (sent_dir or REPO_ROOT / "data" / "local" / "meta" / "slack_sent") / f"results_{date}.json"


def already_sent(date: str, sent_dir: Path | None = None) -> dict | None:
    try:
        return json.loads(sent_path(date, sent_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def mark_sent(date: str, info: dict, sent_dir: Path | None = None) -> Path:
    p = sent_path(date, sent_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p


# ───────────────────────── 集計の読み取り ─────────────────────────

def _rate(n: int, d: int) -> float | None:
    return n / d if d else None


def roi_of(summary: dict) -> float | None:
    b = summary.get("bets") or {}
    return _rate(b.get("returned", 0), b.get("invested", 0))


def collect_periods(date: str, base: Path | None = None) -> dict[str, dict]:
    """週累計・先週・月累計・年累計の合算（日次の集計から）。"""
    d = datetime.strptime(date, "%Y%m%d").date()
    prev_week_day = (d - timedelta(days=7)).strftime("%Y%m%d")
    out: dict[str, dict] = {}
    for key, scope, dt in (("week", "week", date), ("prev_week", "week", prev_week_day), ("month", "month", date), ("year", "year", date)):
        out[key] = PR.period_summary(scope, dt, base)["summary"]
    return out


def _kesi_target(root: Path | None = None) -> float | None:
    try:
        return float(json.loads(((root or REPO_ROOT) / "config" / "kesi_mark.json").read_text(encoding="utf-8")).get("target_upper"))
    except (OSError, ValueError, TypeError):
        return None


# ───────────────────────── メッセージ ─────────────────────────

def _bet_lines(bets: dict) -> list[list[str]]:
    rows = []
    bt = bets.get("by_type") or {}
    for k in BET_ORDER:
        v = bt.get(k)
        if not v or not v.get("invested"):
            continue
        rows.append([PR.BET_LABELS.get(k, k), f"{v['invested']:,}", f"{v['returned']:,}",
                     SN.fmt_pct(_rate(v["returned"], v["invested"]), 0), f"{v.get('hit', 0)}/{v.get('n', 0)}"])
    return rows


def _mark_rows(marks: dict) -> list[list[str]]:
    rows = []
    for key, sym in MARK_ROWS:
        c = (marks or {}).get(key) or {}
        n = c.get("n", 0)
        if not n:
            continue
        top2 = c.get("f1", 0) + c.get("f2", 0)
        top3 = top2 + c.get("f3", 0)
        rows.append([sym, str(n), SN.fmt_pct(_rate(c.get("f1", 0), n), 0), SN.fmt_pct(_rate(top2, n), 0), SN.fmt_pct(_rate(top3, n), 0)])
    return rows


def _period_line(label: str, s: dict, ref: dict | None = None) -> str | None:
    b = s.get("bets") or {}
    if not b.get("invested"):
        return None
    roi = roi_of(s)
    delta = None
    if ref is not None and roi_of(ref) is not None and roi is not None:
        delta = roi - roi_of(ref)
    tail = f"（先週比 {SN.fmt_delta_pt(delta)}）" if delta is not None else ""
    return f"{label}: 回収率 *{SN.fmt_pct(roi, 1)}*　損益 {SN.fmt_yen(b['returned'] - b['invested'], signed=True)}　的中 {b.get('races_hit', 0)}/{b.get('races_bought', 0)}R{tail}"


def build_payload(doc: dict, periods: dict[str, dict] | None = None, *, link_base: str | None = None,
                  kesi_target: float | None = None) -> dict:
    """日次の集計 → Slack のペイロード。1 画面で『回収率・損益・的中』が見えるようにし、詳細は下に並べる。"""
    date = doc["date"]
    s = doc["summary"]
    b = s.get("bets") or {}
    final = doc.get("status") == "final"
    inv, ret = b.get("invested", 0), b.get("returned", 0)
    roi = roi_of(s)
    n_set = s.get("races_settled", 0)
    top1 = s.get("top1") or {}
    hon = (s.get("marks") or {}).get("honmei") or {}
    hn = hon.get("n", 0)
    ll = s.get("logloss") or {}
    ll_mean = (ll["sum"] / ll["n"]) if ll.get("n") else None

    d = datetime.strptime(date, "%Y%m%d")
    status_word = "確定" if final else "暫定"
    title = f"結果 {d.year}/{d.month}/{d.day}（{SN.jp_weekday(date)}）［{status_word}］"
    if inv:
        headline = f"*{SN.esc(title)}*\n推奨馬券 回収率 *{SN.fmt_pct(roi, 1)}*　損益 *{SN.fmt_yen(ret - inv, signed=True)}*　的中 {b.get('races_hit', 0)}/{b.get('races_bought', 0)}R"
    else:
        headline = f"*{SN.esc(title)}*\n推奨馬券の購入対象なし"
    blocks: list[dict] = [SN.section(headline)]

    kpis: list[tuple[str, str]] = []
    if inv:
        kpis += [("購入 → 払戻", f"{SN.fmt_yen(inv)} → {SN.fmt_yen(ret)}"),
                 ("的中レース", SN.fmt_ratio(b.get("races_hit", 0), b.get("races_bought", 0)))]
    if hn:
        kpis += [("◎ 勝率", SN.fmt_ratio(hon.get("f1", 0), hn)),
                 ("◎ 3着内", SN.fmt_ratio(hon.get("f1", 0) + hon.get("f2", 0) + hon.get("f3", 0), hn))]
    if top1.get("n"):
        kpis.append(("勝率1位の的中", SN.fmt_ratio(top1.get("f1", 0), top1["n"])))
    kpis.append(("対象レース", f"{n_set}R（予測なし {s.get('races_no_prediction', 0)}R / 結果なし {s.get('races_no_result', 0)}R）"))
    blocks.append(SN.fields(kpis))

    rows = _bet_lines(b)
    if rows:
        blocks.append(SN.section("*券種別*\n" + SN.code_table(["券種", "購入", "払戻", "回収率", "的中/点"], rows)))
    mrows = _mark_rows(s.get("marks") or {})
    if mrows:
        blocks.append(SN.section("*印別の着順*\n" + SN.code_table(["印", "頭数", "1着", "連対", "3着内"], mrows)))
    kc = (s.get("marks") or {}).get("kesi") or {}
    if kc.get("n"):
        k3 = kc.get("f1", 0) + kc.get("f2", 0) + kc.get("f3", 0)
        tgt = f"（目標 {SN.fmt_pct(kesi_target, 0)}以下）" if kesi_target else ""
        blocks.append(SN.section(f"*消印*　{kc['n']}頭のうち3着内 {k3}頭（{SN.fmt_pct(_rate(k3, kc['n']), 1)}）{tgt}"))

    if periods:
        lines = [x for x in (_period_line("週累計", periods["week"], periods.get("prev_week")),
                             _period_line("月累計", periods["month"]), _period_line("年累計", periods["year"])) if x]
        if lines:
            blocks.append(SN.divider())
            blocks.append(SN.section("*累計*\n" + "\n".join(lines)))

    hl = doc.get("highlights") or []
    if hl:
        blocks.append(SN.section("*注目*\n" + "\n".join("• " + SN.esc(h["text"]) for h in hl[:5])))

    ctx = []
    if ll.get("n"):
        ctx.append(f"勝ち馬の LogLoss 平均 {ll_mean:.3f}（{ll['n']}R）")
    models = doc.get("models") or []
    if models:
        ctx.append("モデル: " + SN.esc(", ".join(models)))
    srcs = doc.get("prediction_sources") or {}
    if srcs.get("backfill") or srcs.get("shadow"):
        ctx.append("バックフィル/shadow の予測を含む")
    if not final:
        ctx.append("暫定: 確定払戻が揃っていないレースを含む")
    if b.get("estimated_races"):
        ctx.append(f"払戻が概算のレース {b['estimated_races']}R")
    if link_base:
        ctx.append(f"<{link_base.rstrip('/')}/prediction-results?scope=day&date={date}|詳細を開く>")
    if ctx:
        blocks.append(SN.context(ctx))

    color = SN.SEVERITY_COLOR["good"] if (roi is not None and roi >= 1.0) else SN.SEVERITY_COLOR["normal"]
    fallback = f"{title} 回収率 {SN.fmt_pct(roi, 1)} 損益 {SN.fmt_yen(ret - inv, signed=True)}" if inv else f"{title}（購入対象なし）"
    return SN.build_message(title, blocks, severity="normal", fallback=fallback, color=color)


# ───────────────────────── 実行 ─────────────────────────

def run(date: str, *, send: bool = False, refresh: bool = False, force: bool = False, allow_provisional: bool = False,
        results_dir: Path | None = None, sent_dir: Path | None = None, out_dir: Path | None = None,
        poster: Callable[..., Any] | None = None, link_base: str | None = None, env: dict[str, str] | None = None,
        storage: Any = None) -> dict:
    """1 日分の通知を判定して、dry-run なら出力、send なら送る。戻り値: {"code", "reason", ...}。"""
    e = os.environ if env is None else env
    if send and not enabled(dict(e)):
        return {"code": EXIT_DISABLED, "reason": "KEIBA_SLACK_NOTIFY!=1"}
    if send and not force and (prev := already_sent(date, sent_dir)):
        return {"code": EXIT_SENT, "reason": "already_sent", "sent": prev}
    if refresh:
        if storage is None:
            from src.scraper.storage import HybridStorage

            storage = HybridStorage()
        PR.aggregate_day(date, storage, base=results_dir)
    doc = PR.read_day(date, results_dir)
    ready, why = check_ready(doc, allow_provisional=allow_provisional)
    if not ready:
        code = EXIT_NONE if why in ("no_doc", "no_races") else EXIT_NOT_READY
        return {"code": code, "reason": why}
    periods = collect_periods(date, results_dir)
    payload = build_payload(doc, periods, link_base=link_base or (e.get("KEIBA_PUBLIC_BASE_URL") or None), kesi_target=_kesi_target())
    res: dict[str, Any] = {"code": EXIT_OK, "reason": "ok", "status": doc.get("status"), "payload": payload}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"results_{date}.json").write_text(SN.dumps(payload), encoding="utf-8")
        (out_dir / f"results_{date}.html").write_text(SN.render_preview_html(payload, title=f"結果通知 {date}"), encoding="utf-8")
    if send:
        kw = {"post": poster, "env": dict(e)} if poster else {"env": dict(e)}
        r = SN.send_payload(payload, channel="results", **kw)
        res["send"] = r
        if not r["ok"]:
            res["code"], res["reason"] = EXIT_FAILED, r.get("error") or "send_failed"
            return res
        mark_sent(date, {"sent_at": datetime.now(JST).isoformat(timespec="seconds"), "status": doc.get("status"),
                         "digest": hashlib.sha256(SN.dumps(payload).encode()).hexdigest()[:16]}, sent_dir)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", help="YYYYMMDD（既定: 今日 JST）")
    ap.add_argument("--send", action="store_true", help="実際に送る（既定は dry-run）")
    ap.add_argument("--send-test", action="store_true", help="接続テストの最小メッセージだけ送る")
    ap.add_argument("--refresh", action="store_true", help="先に日次の集計を取り直す（HybridStorage を使う）")
    ap.add_argument("--force", action="store_true", help="送信済みでも送る")
    ap.add_argument("--allow-provisional", action="store_true", help="確定でなくても（暫定のまま）送る")
    ap.add_argument("--out", default="data/local/meta/slack_preview", help="dry-run の出力先（JSON と HTML プレビュー）")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--link-base", default=None, help="詳細リンクの基点 URL（既定: 環境変数 KEIBA_PUBLIC_BASE_URL）")
    a = ap.parse_args(argv)
    if a.send_test:
        r = SN.send_test()
        print(json.dumps(r, ensure_ascii=False))
        return EXIT_OK if r["ok"] else EXIT_FAILED
    date = (a.date or datetime.now(JST).strftime("%Y%m%d")).replace("-", "")
    res = run(date, send=a.send, refresh=a.refresh, force=a.force, allow_provisional=a.allow_provisional,
              results_dir=Path(a.results_dir) if a.results_dir else None, out_dir=Path(a.out), link_base=a.link_base)
    summary = {k: v for k, v in res.items() if k != "payload"}
    print(json.dumps(summary, ensure_ascii=False, default=str))
    if res["code"] == EXIT_OK and not a.send:
        print(f"dry-run: {a.out}/results_{date}.json と .html を出力しました（送信していません）")
    return res["code"]


if __name__ == "__main__":
    sys.exit(main())
