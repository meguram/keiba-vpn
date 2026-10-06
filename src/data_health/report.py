"""レポート出力: コンソール要約 / JSON / 自己完結 HTML（期間×カテゴリのヒートマップ付き）。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

E = html.escape
SYMBOL = {"ok": "OK", "warn": "WARN", "fail": "FAIL", "info": "INFO", "skip": "SKIP"}
HISTORY_KEEP = 20


# ── ファイル出力 ──────────────────────────────────────────────────────────

def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["generated_at"].replace(":", "").replace("-", "")[:15]
    paths = {
        "json": out_dir / "latest.json",
        "html": out_dir / "latest.html",
        "plan": out_dir / "scrape_plan.json",
        "history": out_dir / f"report_{stamp}.json",
    }
    side = report.pop("_side", None) or {}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    paths["json"].write_text(text, encoding="utf-8")
    paths.update(write_race_ids(report, out_dir, side))
    paths["history"].write_text(text, encoding="utf-8")
    paths["plan"].write_text(json.dumps(report["plan"], ensure_ascii=False, indent=1), encoding="utf-8")
    paths["html"].write_text(render_html(report), encoding="utf-8")
    old = sorted(out_dir.glob("report_*.json"))[:-HISTORY_KEEP]
    for p in old:
        p.unlink(missing_ok=True)
    return paths


def write_race_ids(report: dict[str, Any], out_dir: Path, side: dict[str, Any]) -> dict[str, Path]:
    """race_id の配列をファイルに出す（1 行 1 race_id）。何が揃っていないかを他ツール・スクリプトに渡しやすくする。

    race_ids/missing_<cat>.txt / invalid_<cat>.txt / unvalidated_<cat>.txt、race_ids/by_race.json（race_id → 不健全なカテゴリ）、
    race_keys.csv（race_id ↔ 開催日・場・R・レース名 ＋ カテゴリ別の状態）。
    """
    d = out_dir / "race_ids"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.txt"):
        old.unlink()
    by_race: dict[str, dict[str, Any]] = {}
    for cat, groups in (report.get("race_ids") or {}).items():
        for kind in ("missing", "invalid", "unvalidated"):
            ids = sorted(groups.get(kind, {}))
            if ids:
                (d / f"{kind}_{cat}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
            for rid in ids:
                by_race.setdefault(rid, {})[cat] = kind if kind != "invalid" else {"invalid": groups["invalid"][rid]}
    detail = {cat: g["invalid_detail"] for cat, g in (report.get("race_ids") or {}).items() if g.get("invalid_detail")}
    (d / "invalid_detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "by_race.json").write_text(json.dumps(dict(sorted(by_race.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    out = {"race_ids": d}
    if side.get("race_keys"):
        from src.data_health.keytable import write_csv

        out["race_keys"] = out_dir / "race_keys.csv"
        write_csv(out["race_keys"], side["race_keys"]["columns"], side["race_keys"]["rows"])
    return out


# ── コンソール ────────────────────────────────────────────────────────────

def render_text(r: dict[str, Any], *, limit: int = 25) -> str:
    s = r["summary"]
    lines = [f"データヘルス [{r['env']}] {SYMBOL[s['overall']] if s['overall'] in SYMBOL else s['overall'].upper()}"
             f"  fail={s['counts']['fail']} warn={s['counts']['warn']} info={s['counts']['info']}"
             f"  計画ジョブ={s['planned_jobs']}  （{r['range']['since']} 〜 {r['range']['until']}）", ""]
    comp = r.get("completeness")
    if comp:
        lines.insert(1, "  完全性(" + comp["scope"] + "): " + ("OK" if comp["complete"] else f"NG {len(comp['reasons'])}項目"))
    meta = r.get("meta") or {}
    if meta:
        lines.insert(2 if comp else 1, f"  実行環境={meta.get('actual_env')} / host={meta.get('host')} / commit={meta.get('commit') or '-'}"
                        f" / 保存先キー={meta.get('key')}")
    cfg = r.get("settings") or {}
    for w in cfg.get("warnings", []):
        lines.append(f"  ! 設定: {w}")
    if cfg.get("levels"):
        lines.append("  設定で上書き: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg["levels"].items())))
    lines.append("")
    lines.append("■ インフラ")
    for c in r["checks"]:
        lines.append(f"  [{SYMBOL.get(c['status'], c['status'])}] {c['label']}: {c['detail']}")
    lines.append("")
    lines.append("■ 派生データ")
    for a in r["artifacts"]:
        if a["status"] != "skip":
            lines.append(f"  [{SYMBOL[a['status']]}] {a['id']} {a['label']}: {a['detail']}")
    rc = r["race_coverage"]
    lines += ["", f"■ レース存在カバレッジ（対象 {rc['universe']['races']} レース: {rc['universe']['by_confidence']}）"]
    for y, cats in rc["by_year"].items():
        row = []
        for cat in rc["categories"]:
            v = cats.get(cat["name"])
            if v:
                ok, tot = _ok_total(v)
                extra = [f"{lbl}{v[k]}" for k, lbl in (("invalid", "不適合"), ("unvalidated", "未検証"), ("pending", "待機")) if v.get(k)]
                row.append(f"{cat['name']}={ok}/{tot}" + (f"({' '.join(extra)})" if extra else ""))
        lines.append(f"  {y}: " + "  ".join(row))
    sc = r.get("schema")
    if sc:
        lines += ["", f"■ スキーマ（定義 {sc['defined']} カテゴリ / fingerprint={sc['fingerprint']} / v{sc['version']}）"]
        for cat, v in sc["categories"].items():
            lines.append(f"  {cat}: {v['passed']}/{v['sampled']} 適合" + ("（advisory）" if v["advisory"] else ""))
        und = [u["category"] for u in sc["undefined"] if u["status"] == "pending"]
        if und:
            lines.append(f"  観測待ち（未定義）: {', '.join(und)}")
    log = r.get("schema_log") or {}
    if log.get("groups"):
        lines += ["", f"■ 保存時のスキーマ違反（直近30日: {log['records']} 回 / {log['decisions']} / 隔離 {log.get('quarantined') or 'なし'}）"]
        for g in log["groups"][:5]:
            vals = " ; ".join(f"{v} ×{n}" for v, n in g["values"][:2])
            lines.append(f"  {g['category']} {g['field']} [{g['rule']}] {g['count']}回: {vals}")
    hc = r["horse_coverage"]
    if hc.get("categories"):
        lines += ["", f"■ 馬（{hc['window'].get('from')}〜{hc['window'].get('to')} の出走馬 {hc['horses']} 頭）"]
        for c in hc["categories"]:
            lines.append(f"  {c['name']}: 不足 {c['missing']}/{c['expected']}")
    lines += ["", "■ 要対応（重大順）"]
    if not r["findings"]:
        lines.append("  なし")
    for f in r["findings"][:limit]:
        lines.append(f"  [{SYMBOL.get(f['severity'])}] ({f['area']}) {f['message']}")
    if len(r["findings"]) > limit:
        lines.append(f"  … 他 {len(r['findings']) - limit} 件（latest.json / latest.html 参照）")
    d = r.get("diff")
    if d:
        delta = " ".join(f"{k}{v:+d}" for k, v in d["counts_delta"].items())
        lines += ["", f"■ 前回比（前回 {d['previous_at']} は {d['overall_before']}）: {delta}"]
        lines += [f"  + 新規: {m}" for m in d["new"][:10]] + [f"  - 解消: {m}" for m in d["resolved"][:10]]
        lines += [f"  ~ 変化: {c['before']} → {c['after']}" for c in d["changed"][:10]]
    pc = r["plan"]["counts"]
    lines += ["", f"■ スクレイピング計画: {pc['jobs']} ジョブ " + " / ".join(f"{k}={v}" for k, v in pc["by_runner"].items())]
    return "\n".join(lines)


# ── HTML ─────────────────────────────────────────────────────────────────

CSS = """
:root{--bg:#0f172a;--panel:#15213c;--border:#243358;--text:#e6ecff;--muted:#9aa7c7;--acc:#7aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.7 "Noto Sans JP",system-ui,sans-serif;padding:28px 36px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--border)}
.sub{color:var(--muted);font-size:12px}code{background:#1a2748;border:1px solid var(--border);border-radius:5px;padding:0 5px;font-size:12px}
.wrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px;margin:8px 0}
table{border-collapse:collapse;width:100%;background:var(--panel);font-size:12.5px}
th,td{padding:6px 10px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}th{background:#1a2748;color:var(--muted);font-size:11px;white-space:nowrap}
.b{display:inline-block;min-width:44px;text-align:center;border-radius:6px;padding:0 6px;font-weight:700;font-size:11px}
.b-ok{background:#12452f;color:#5ff0a2}.b-warn{background:#55400f;color:#ffd166}.b-fail{background:#5a1a24;color:#ff8a98}
.b-info{background:#163a6b;color:#8ec1ff}.b-skip{background:#2a3350;color:#9aa7c7}
.hm td.c{text-align:center;min-width:92px;font-variant-numeric:tabular-nums}
.g0{background:#12452f;color:#bff5d8}.g1{background:#55400f;color:#ffe29a}.g2{background:#5a1a24;color:#ffc2cb}.gn{background:#1d2742;color:#6f7ba0}
.pend{color:#8ec1ff;font-size:10.5px}.sum{display:flex;gap:10px;margin:12px 0}.sum div{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:8px 16px}
.sum b{font-size:20px;display:block}
"""


def _badge(st: str) -> str:
    return f'<span class="b b-{st}">{SYMBOL.get(st, st)}</span>'


def _ok_total(v: dict) -> tuple[int, int]:
    """(健全または存在, 期限到来分の合計)。待機・取得不可は分母に入れない。"""
    ok = v.get("present", 0) + v.get("healthy", 0)
    return ok, ok + v.get("invalid", 0) + v.get("unvalidated", 0) + v.get("missing", 0)


def _cell(v: dict | None) -> str:
    if not v:
        return '<td class="c gn">–</td>'
    ok, tot = _ok_total(v)
    notes = "".join(f'<div class="pend">{lbl} {v[k]}</div>' for k, lbl in (("invalid", "不適合"), ("unvalidated", "未検証"), ("pending", "待機"))
                    if v.get(k))
    if tot == 0:
        return f'<td class="c gn">–{notes}</td>'
    pct = ok / tot * 100
    cls = "g0" if ok == tot else "g1" if pct >= 90 else "g2"
    return f'<td class="c {cls}">{ok}/{tot}<div class="pend">{pct:.0f}%</div>{notes}</td>'


def _heatmap(title: str, rows: dict[str, dict], cats: list[dict]) -> str:
    if not rows:
        return f"<h3>{E(title)}</h3><p class='sub'>対象データなし</p>"
    head = "".join(f'<th title="{E(c["label"])}">{E(c["name"])}<div class="sub">{E(c["level"])}</div></th>' for c in cats)
    body = "".join(f"<tr><td><b>{E(k)}</b></td>" + "".join(_cell(v.get(c["name"])) for c in cats) + "</tr>"
                   for k, v in rows.items())
    return f'<h3>{E(title)}</h3><div class="wrap"><table class="hm"><tr><th>期間</th>{head}</tr>{body}</table></div>'


def _table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{E(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="wrap"><table><tr>{th}</tr>{tr}</table></div>'


def render_html(r: dict[str, Any]) -> str:
    s, rc, hc, pl = r["summary"], r["race_coverage"], r["horse_coverage"], r["plan"]
    parts = [f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>データヘルス [{E(r["env"])}]</title>'
             f'<style>{CSS}</style></head><body>',
             f'<h1>データ存在チェック / ヘルスチェック <span class="sub">env={E(r["env"])}</span></h1>',
             f'<div class="sub">生成 {E(r["generated_at"])} ／ 対象期間 {E(r["range"]["since"])} 〜 {E(r["range"]["until"])}'
             f' ／ 総合 {_badge(s["overall"])}</div>',
             '<div class="sum">' + "".join(f'<div><b>{s["counts"][k]}</b>{SYMBOL[k]}</div>' for k in ("fail", "warn", "info"))
             + f'<div><b>{s["planned_jobs"]}</b>計画ジョブ</div><div><b>{rc["universe"]["races"]}</b>対象レース</div></div>']

    comp = r.get("completeness")
    if comp:
        if comp["complete"]:
            parts.append(f'<div class="callout"><b>完全性 OK</b>（{E(comp["scope"])}）: 対象範囲（{E("/".join(comp["levels"]))}）のすべてが揃い、スキーマに適合しています。</div>')
        else:
            items = "".join(f'<li>{E(x["message"])}</li>' for x in comp["reasons"][:40])
            parts.append(f'<div class="callout warn"><b>完全性 NG</b>（{E(comp["scope"])}）: 次の項目が未達です（{len(comp["reasons"])} 項目）。<ul>{items}</ul></div>')
    meta, cfg, d = r.get("meta") or {}, r.get("settings") or {}, r.get("diff")
    if meta:
        parts.append(f'<div class="sub">実行環境 <b>{E(str(meta.get("actual_env")))}</b> ／ host {E(str(meta.get("host")))} ／ '
                     f'commit <code>{E(meta.get("commit") or "-")}</code> ／ 保存先キー <code>{E(str(meta.get("key")))}</code></div>')
    if cfg.get("levels") or cfg.get("warnings"):
        ov = ", ".join(f"<code>{E(k)}={E(v)}</code>" for k, v in sorted(cfg.get("levels", {}).items()))
        parts.append(f'<div class="sub">設定で上書き（DATA_HEALTH_LEVELS / SKIP）: {ov or "なし"}'
                     + "".join(f' ／ <span style="color:#ffd166">{E(w)}</span>' for w in cfg.get("warnings", [])) + "</div>")
    if d:
        delta = " ".join(f"{k} {v:+d}" for k, v in d["counts_delta"].items())
        rows = ([[_badge("fail"), "新規", E(m)] for m in d["new"]] + [[_badge("ok"), "解消", E(m)] for m in d["resolved"]]
                + [[_badge("info"), "変化", E(c["before"] + " → " + c["after"])] for c in d["changed"]])
        parts.append(f'<h2>前回比</h2><p class="sub">前回 {E(str(d["previous_at"]))}（{E(d["overall_before"])}）から: {E(delta)}</p>')
        parts.append(_table(["", "区分", "内容"], rows) if rows else "<p class='sub'>変化なし</p>")
    parts.append("<h2>要対応（重大順）</h2>")
    parts.append(_table(["重大度", "領域", "内容", "対処"], [[_badge(f["severity"]), E(f["area"]), E(f["message"]), E(f["hint"])]
                                                        for f in r["findings"][:200]]) if r["findings"] else "<p>なし</p>")

    parts.append("<h2>ヘルスチェック（インフラ）</h2>")
    parts.append(_table(["状態", "項目", "詳細", "対処"], [[_badge(c["status"]), E(c["label"]), E(c["detail"]), E(c["hint"])]
                                                     for c in r["checks"]]))
    parts.append("<h2>派生データ（特徴量・血統・モデル・設定）</h2>")
    parts.append(_table(["状態", "ID", "グループ", "名称", "重要度", "詳細", "生成コマンド"],
                        [[_badge(a["status"]), E(a["id"]), E(a["group"]), E(a["label"]), E(a["level"]), E(a["detail"]),
                          f'<code>{E(a["hint"])}</code>' if a["hint"] else ""] for a in r["artifacts"]]))

    parts.append("<h2>レース存在カバレッジ</h2>")
    u = rc["universe"]["by_confidence"]
    parts.append(f'<p class="sub">対象 {rc["universe"]["races"]} レース（race_lists で確認 {u["confirmed"]} / データのみ {u["present"]} / '
                 f'連番から推定 {u["inferred"]}）。セルは 存在/期限到来分（%）、青字は期限前（SLA）で「待機」。'
                 f'日付が判明しないレースは「YYYY-??」行に集計。</p>')
    parts.append(_heatmap("年別", rc["by_year"], rc["categories"]))
    parts.append(_heatmap("月別（race_lists で日付が分かるレース）", {k: v for k, v in rc["by_period"].items() if "??" not in k},
                          rc["categories"]))
    fresh = rc.get("freshness", {})
    if fresh:
        parts.append("<h3>鮮度（カテゴリ別の最新保存）</h3>")
        parts.append(_table(["カテゴリ", "件数", "最新保存", "経過(h)"], [[E(k), str(v["count"]), E(v["latest"] or "–"),
                                                                   str(v["age_hours"] if v["age_hours"] is not None else "–")]
                                                                  for k, v in sorted(fresh.items())]))
    kt = r.get("key_table") or {}
    parts.append('<h3>健全性と race_id 一覧</h3><p class="sub">セルの数字は <b>健全（スキーマに適合）または存在 / 期限到来分</b>。'
                 '「不適合」は存在するがスキーマに合わないもの、「未検証」は存在するがまだ download して検証していないもの（full モードは複数回の実行で進む）。'
                 + (f'キーテーブル: {kt.get("races", 0)} レース（開催日が分かるもの {kt.get("with_date", 0)}）。' if kt else "")
                 + '<code>race_keys.csv</code>（race_id ↔ 開催日・場・R・レース名 ＋ カテゴリ別の状態）と <code>race_ids/</code>（1 行 1 race_id のテキスト）に出力。</p>')
    for cat, groups in (r.get("race_ids") or {}).items():
        n_m, n_i, n_u = len(groups.get("missing", [])), len(groups.get("invalid", {})), len(groups.get("unvalidated", []))
        if not (n_m or n_i or n_u):
            continue
        body = []
        for lbl, ids in (("不足", groups.get("missing", [])), ("不適合", sorted(groups.get("invalid", {}))), ("未検証", groups.get("unvalidated", []))):
            if ids:
                shown = ", ".join(ids[:150]) + (f" …他 {len(ids) - 150} 件" if len(ids) > 150 else "")
                body.append(f"<p><b>{lbl} {len(ids)} 件</b><br><code style='word-break:break-all'>{E(shown)}</code></p>")
        inv = groups.get("invalid", {})
        det = groups.get("invalid_detail", {})
        if inv:
            body.append("<p class='sub'>不適合の例（どの項目がどの値か）:<br>" + "<br>".join(
                E(f"{k}: " + (" / ".join(det.get(k, [])) or ",".join(v[:3]))) for k, v in list(inv.items())[:8]) + "</p>")
        parts.append(f"<details><summary><b>{E(cat)}</b> — 不足 {n_m} / 不適合 {n_i} / 未検証 {n_u}</summary>{''.join(body)}</details>")
    gaps = rc["gaps"]
    parts.append(f"<h3>不足レース一覧（{sum(rc['gap_total'].values()) + sum(rc.get('invalid_total', {}).values())} 件中 {min(len(gaps), 300)} 件表示）</h3>")
    parts.append(_table(["状態", "カテゴリ", "重要度", "日付", "race_id", "場・R", "判定根拠"],
                        [[E({"invalid": "不適合", "missing": "不足"}.get(g.get("status", "missing"), "不足")), E(g["category"]), E(g["level"]),
                          E(g["date"] or "不明"), f'<code>{E(g["race_id"])}</code>', f'{E(g["venue"])} {g["round"]}R', E(g["confidence"])]
                         for g in gaps[:300]]))

    sc = r.get("schema")
    if sc:
        parts.append(f'<h2>スキーマ</h2><p class="sub">定義 {sc["defined"]} カテゴリ ／ fingerprint <code>{E(sc["fingerprint"])}</code> ／ v{sc["version"]}'
                     f'（<code>src/scraper/schema_defs.json</code>、git 管理。環境間で fingerprint が同じなら同一定義）。'
                     f'適合率はカテゴリごとの最新 {sc["sample"]} 件を検証。</p>')
        rows = [[E(c), f'{v["passed"]}/{v["sampled"]}', "advisory" if v["advisory"] else "厳格",
                 E("; ".join(f"{m}×{n}" for m, n in sorted(v["issues"].items(), key=lambda x: -x[1])[:3])) or "–",
                 E(" / ".join(v.get("examples", []))) or "–",
                 E(", ".join(v["failed_keys"])) or "–"] for c, v in sc["categories"].items()]
        parts.append(_table(["カテゴリ", "適合", "モード", "主な不適合", "どの値で引っかかったか（例）", "race_id の例"], rows)
                     if rows else "<p class='sub'>サンプル検証なし</p>")
        log = r.get("schema_log") or {}
        if log.get("groups"):
            parts.append(f'<h3>保存時に引っかかった記録（直近30日）</h3><p class="sub">記録 {log["records"]} 回 ／ 判断 {E(str(log["decisions"]))} ／ '
                         f'隔離中 {E(str(log.get("quarantined") or "なし"))}。スクレイピングして保存しようとしたときの不適合で、'
                         '<code>data/local/meta/schema_violations/</code>（違反の中身）と <code>data/local/quarantine/</code>（拒否したデータ本体）に残ります。'
                         '詳細は <code>python -m src.scraper.schema_violations summary|show</code>。</p>')
            parts.append(_table(["カテゴリ", "項目", "規則（期待）", "回数 / key数", "どの値で（上位）", "最後"],
                                [[E(g["category"]), f'<code>{E(g["field"])}</code>', E(f'{g["rule"]}（{g["expected"]}）'),
                                  f'{g["count"]} / {g["keys"]}', E(" ; ".join(f"{val} ×{n}" for val, n in g["values"][:3])), E(g["last"])]
                                 for g in log["groups"]]))
        und = [[E(u["category"]), E(u["status"]), E(u["note"])] for u in sc["undefined"]]
        parts.append("<h3>スキーマ未定義のカテゴリ</h3>" + _table(["カテゴリ", "状態", "理由"], und))
    parts.append("<h2>馬（直近〜出走予定の出走馬）</h2>")
    if hc.get("categories"):
        parts.append(f'<p class="sub">{E(str(hc["window"].get("from")))}〜{E(str(hc["window"].get("to")))} の出走馬 {hc["horses"]} 頭'
                     f'（出馬表 {hc["races_examined"]} レース' + ("・上限で打ち切り" if hc.get("races_truncated") else "") + '）</p>')
        parts.append(_table(["カテゴリ", "重要度", "対象馬", "不足", "再取得タスク"],
                            [[E(c["name"]), E(c["level"]), str(c["expected"]), str(c["missing"]), f'<code>{E(c["task"])}</code>']
                             for c in hc["categories"]]))
    else:
        parts.append("<p class='sub'>この環境では評価しない</p>")

    parts.append("<h2>スクレイピング計画</h2>")
    counts = pl["counts"]
    parts.append(f'<p class="sub">{counts["jobs"]} ジョブ: ' + " / ".join(f"{E(k)}={v}" for k, v in counts["by_runner"].items())
                 + '。全件は <code>scrape_plan.json</code>（<code>ScrapeJobQueue.bulk_add_jobs</code> / <code>POST /api/scrape/enqueue</code> と同形式、'
                 'smart_skip=true・overwrite=false で既存は上書きしない）。</p>')
    parts.append(_table(["実行場所", "種別", "対象", "タスク", "理由"],
                        [[E(sp["runner"]), E(sp["job_kind"]), f'<code>{E(sp["target_id"])}</code>', E(", ".join(sp["tasks"])),
                          E(sp["reason"])] for sp in pl["specs"][:300]]))
    parts.append("</body></html>")
    return "\n".join(parts)


def render_index(entries: list[dict], now: "datetime | None" = None) -> str:
    """全環境の最新結果の一覧（環境別管理の入口）。"""
    from datetime import datetime, timezone

    now = now or datetime.now(timezone.utc)
    rows = []
    for e in entries:
        rep, key = e["report"], e["key"]
        s, meta = rep["summary"], rep.get("meta") or {}
        try:
            age = (now - datetime.fromisoformat(rep["generated_at"])).total_seconds() / 3600
        except ValueError:
            age = None
        stale = age is not None and age > 48
        age_txt = "–" if age is None else (f"{age:.0f} 時間前" + ("（古い）" if stale else ""))
        fp = meta.get("schema_fingerprint") or "–"
        rows.append([f"<b>{E(key)}</b>", _badge(s["overall"]),
                     f'{s["counts"]["fail"]} / {s["counts"]["warn"]} / {s["counts"]["info"]}', str(s["planned_jobs"]),
                     E(rep["generated_at"]), E(age_txt), E(str(meta.get("host", "–"))), f'<code>{E(meta.get("commit") or "-")}</code>', f"<code>{E(fp)}</code>",
                     f'<a href="{E(key)}/latest.html">レポート</a> / <a href="{E(key)}/latest.json">JSON</a>'])
    parts = [f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>データヘルス一覧</title><style>{CSS}</style></head><body>',
             '<h1>データ存在チェック — 環境別の最新結果</h1>',
             '<p class="sub">各環境で <code>python -m src.data_health</code> を実行すると更新されます。他 PC の結果は '
             '<code>--import &lt;latest.json&gt;</code> で取り込めます。キー <code>stg@dev</code> は「dev PC で stg の要件を評価した」結果です。</p>']
    fps = {(e["report"].get("meta") or {}).get("schema_fingerprint") for e in entries} - {None, ""}
    if len(fps) > 1:
        parts.append('<p style="color:#ffd166"><b>スキーマ定義が環境間で一致していません</b>（fingerprint が複数）。'
                     '全環境で <code>src/scraper/schema_defs.json</code> を同じ commit にしてください。</p>')
    parts.append(_table(["環境キー", "総合", "FAIL / WARN / INFO", "計画ジョブ", "生成時刻", "経過", "実行ホスト", "commit", "スキーマ", "リンク"], rows)
                 if rows else "<p>結果がまだありません。</p>")
    for e in entries:
        hist = e["history"]
        if len(hist) < 2:
            continue
        parts.append(f'<h2>{E(e["key"])} の推移（直近 {len(hist)} 回）</h2>')
        parts.append(_table(["生成時刻", "総合", "FAIL", "WARN", "不足レース", "計画ジョブ", "ホスト"],
                            [[E(h["generated_at"]), _badge(h["overall"]), str(h["counts"].get("fail", 0)), str(h["counts"].get("warn", 0)),
                              str(h.get("missing_races", 0)), str(h.get("planned_jobs", 0)), E(h.get("host", ""))] for h in reversed(hist)]))
    parts.append("</body></html>")
    return "\n".join(parts)
