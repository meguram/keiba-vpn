"""stg と prod の実効リリース・マニフェストを取得して差分を出す（特徴量・モデル・config・フラグ・フロントが揃っているか）。

  # 照合（URL か、ssh などで取ったJSONファイル）。意図した差（環境名・ホスト名・STG バッジ等）は除く。差があれば終了コード 1
  python -m src.scripts.ops.env_parity_check compare --stg-api https://stg.megu-keiba.com --prod-api https://app.megu-keiba.com \\
      --stg-ui https://stg.megu-keiba.com/version.json --prod-ui https://app.megu-keiba.com/version.json [--slack]
  python -m src.scripts.ops.env_parity_check compare --stg-file stg.json --prod-file prod.json
  # 昇格の前検査（promote.sh から）: この環境（stg 側のソース）の UI が、昇格するコミットの frontend と同じか、フラグ表のずれが無いか
  python -m src.scripts.ops.env_parity_check promote-precheck --commit <rev>

トークンは環境変数 KEIBA_MANIFEST_TOKEN（個別なら _STG / _PROD）。Cloudflare Access のサービストークンは CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET。
値は表示しない。終了コード: 0=一致 / 1=差あり / 2=取得できない。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.ops import release_manifest as RM

REPO_ROOT = Path(__file__).resolve().parents[3]


def _get_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 20) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310  運用者が指定した URL のみ
        return json.loads(r.read().decode("utf-8"))


def _headers(which: str) -> dict[str, str]:
    h: dict[str, str] = {}
    tok = os.environ.get(f"KEIBA_MANIFEST_TOKEN_{which.upper()}") or os.environ.get("KEIBA_MANIFEST_TOKEN")
    if tok:
        h["X-Keiba-Manifest-Token"] = tok.strip()
    cid, sec = os.environ.get("CF_ACCESS_CLIENT_ID"), os.environ.get("CF_ACCESS_CLIENT_SECRET")
    if cid and sec:
        h["CF-Access-Client-Id"], h["CF-Access-Client-Secret"] = cid.strip(), sec.strip()
    return h


def fetch_side(which: str, api: str | None, ui: str | None, file: str | None) -> dict[str, Any]:
    """1 環境ぶんの比較用ドキュメント（API のマニフェスト + UI の版）を作る。"""
    if file:
        doc = json.loads(Path(file).read_text(encoding="utf-8"))
    elif api:
        doc = _get_json(api.rstrip("/") + "/api/ops/release-manifest", headers=_headers(which))
    else:
        raise ValueError(f"{which}: --{which}-api か --{which}-file が必要です")
    if ui:
        v = _get_json(ui, headers=_headers(which))
        doc["frontend"] = {"source_digest": v.get("source_digest"), "public_env": v.get("public_env") or {}, "env": v.get("env"), "release": v.get("release"),
                           "commit": v.get("commit"), "built_at": v.get("built_at"), "build_id": v.get("build_id")}
    return doc


def summarize(diffs: list[dict[str, Any]], docs: tuple[dict, dict]) -> tuple[list[dict], list[str]]:
    bad = RM.blocking(diffs)
    notes: list[str] = []
    a, b = docs
    for label, d in (("stg", a), ("prod", b)):
        if d.get("flag_drift"):
            bad.append({"path": f"{label}.flag_drift", "a": d["flag_drift"], "b": "（config/behavior_flags.json と .env が不一致）", "intentional": False})
    if a.get("host") and a.get("host") == b.get("host"):
        notes.append("stg と prod は同じ API（同一ホスト）を見ています。バックエンドの挙動は構造上そろっています。")
    if "frontend" not in a or not (a.get("frontend") or {}).get("source_digest"):
        notes.append("stg の UI ソースの版を取得できていません（frontend.source_digest なし）。")
    if "frontend" not in b or not (b.get("frontend") or {}).get("source_digest"):
        notes.append("prod の UI の版（version.json の source_digest）がありません。次の昇格で付きます。")
        bad = [d for d in bad if d["path"] != "frontend.source_digest"]
    return bad, notes


def render(bad: list[dict], diffs: list[dict], notes: list[str]) -> str:
    lines: list[str] = []
    for n in notes:
        lines.append(f"注: {n}")
    if not bad:
        lines.append("一致: 意図しない差はありません（意図した差 %d 件は除外）" % sum(1 for d in diffs if d["intentional"]))
        return "\n".join(lines)
    lines.append(f"差あり: {len(bad)} 件")
    for d in bad:
        lines.append(f"  - {d['path']}: stg={_short(d['a'])} / prod={_short(d['b'])}")
    return "\n".join(lines)


def _short(v: Any) -> str:
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= 60 else s[:57] + "..."


def cmd_compare(a: argparse.Namespace) -> int:
    try:
        stg = fetch_side("stg", a.stg_api, a.stg_ui, a.stg_file)
        prod = fetch_side("prod", a.prod_api, a.prod_ui, a.prod_file)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        msg = f"マニフェストを取得できませんでした: {type(exc).__name__}"
        print(msg, file=sys.stderr)
        if a.slack:
            _slack("stg/prod の照合ができません", [msg], "warning")
        return 2
    diffs = RM.compare(stg, prod, allow=RM.overridable_paths() | set(a.allow or []))
    bad, notes = summarize(diffs, (stg, prod))
    print(render(bad, diffs, notes))
    if bad and a.slack:
        _slack("stg と prod の挙動がそろっていません", [f"{d['path']}（stg={_short(d['a'])} / prod={_short(d['b'])}）" for d in bad[:8]], "warning")
    return 1 if bad else 0


def commit_frontend_digest(rev: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        arc = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", rev, "frontend"], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", td], input=arc.stdout, check=True)
        return RM.frontend_digest(Path(td) / "frontend")


def cmd_precheck(a: argparse.Namespace) -> int:
    m = RM.build_manifest()
    problems: list[str] = []
    if m.get("flag_drift"):
        problems.append("挙動フラグが config/behavior_flags.json と .env で食い違っています: " + ", ".join(m["flag_drift"]))
    here = (m.get("frontend") or {}).get("source_digest")
    want = commit_frontend_digest(a.commit)
    if here and here != want:
        problems.append(f"この環境（stg）の UI ソース({here}) が昇格するコミットの frontend({want}) と違います。stg で確認した内容と別物を prod に出すことになります")
    if problems:
        print("昇格の前検査: 差があります", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        print("  → stg にそのコミットを反映して確認してから昇格するか、承知の上なら --skip-parity を付けてください。", file=sys.stderr)
        return 1
    print(f"昇格の前検査: OK（frontend {want}、フラグずれなし）")
    return 0


def _slack(title: str, lines: list[str], severity: str) -> None:
    try:
        from src.utils.slack_notify import notify_event

        notify_event(title, lines, severity=severity, channel="alert")
    except Exception as exc:  # noqa: BLE001  通知の失敗で検査自体は落とさない
        print(f"Slack 通知に失敗: {type(exc).__name__}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    for w in ("stg", "prod"):
        c.add_argument(f"--{w}-api")
        c.add_argument(f"--{w}-ui")
        c.add_argument(f"--{w}-file")
    c.add_argument("--allow", action="append", help="追加で除外する項目（例 flags.KEIBA_PACE_TYPE）")
    c.add_argument("--slack", action="store_true")
    c.set_defaults(fn=cmd_compare)
    p = sub.add_parser("promote-precheck")
    p.add_argument("--commit", default="HEAD")
    p.set_defaults(fn=cmd_precheck)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
