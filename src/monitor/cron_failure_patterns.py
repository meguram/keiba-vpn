"""cron ジョブの失敗・警告・見送りを、出力の末尾と終了コードから『原因』に変える辞書。

ホストの実ログ（logs/*.log、2026-10-16 時点）で実際に起きたものを元にしている。新しい失敗が見つかったら PATTERNS に足す
（原因が分からないまま『不明』と出た失敗が、足す候補）。

種別（kind）:
  failed   終了コードが 0 以外のときだけ原因として使う（Traceback、メモリ不足、タイムアウト など）
  warning  終了コード 0 でも出る問題の兆候（ジョブは完了したが一部が取れていない、など）。終了コードが 0 以外のときは、
           failed のパターンに当たらなかった場合の原因候補としても使う
  skipped  見送り（失敗ではない）。終了コード 0 のときだけ

classify(exit_code, output_text) は次の dict を返す。当たらず、成功なら None:
  {"pattern_id", "kind": "failed"|"warning"|"skipped", "label", "hint", "evidence"}
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    id: str
    kind: str                    # failed | warning | skipped
    label: str                   # 原因（日本語）。(?P<名前>…) のキャプチャを {名前} で差し込める
    hint: str                    # 直し方
    regex: str = ""              # 出力の末尾に対する正規表現（空なら終了コードだけで判別）
    exit_codes: tuple[int, ...] = ()   # これらの終了コードでも当たる


_NK = r"(?:race|db|regist|www|sp|nar)\.netkeiba\.com"

# 先に書いたものが優先される。種別ごとに（failed → warning、成功時は skipped → warning）この順で評価する。
PATTERNS: list[Pattern] = [
    # ── 見送り（失敗ではない） ───────────────────────────────────────────────
    Pattern("yield_to_rescrape", "skipped", "優先処理（2020〜2025 年の取り直し）が動いているため見送り",
            "取り直しが終われば自動で通常どおり動く。止めたいときは data/local/meta/rescrape/YIELD_DISABLED を置く。",
            r"SKIP \S+: 優先処理"),
    Pattern("lock_held", "skipped", "別のプロセスが実行中のため見送り（排他ロック）",
            "前回の実行が長引いている。実行中のプロセスが無いのに続くなら、ロックファイルが残っていないか確認する。",
            r"already running|lock held|ロック(?:を取得できない|中)"),

    # ── 終了コードや例外で判別できる失敗 ──────────────────────────────────────
    Pattern("dev_env", "failed", "KEIBA_ENV が dev に解決され、取得せずに中止した（データが GCS に保存されない設定のため）",
            ".env の KEIBA_ENV を stg に直す（ホストの .env を確認）。開発 PC でわざと動かすときだけ KEIBA_ALLOW_DEV_CRON=1。",
            r"KEIBA_ENV が dev に解決", (3,)),
    Pattern("access_restriction", "failed", "netkeiba からアクセス制限を受けた（取得を中断）",
            "しばらく（数時間）取得を止める。別セッションが同じアカウントで同時にログインしていないか、並列数が多すぎないか確認。"
            "解除は python -m src.scraper.scrape_from_health_plan の制限ガード（終了コード 5）の手順を参照。",
            r"AccessRestrictionDetected|アクセス制限を検知", (5,)),
    Pattern("smartrc_restricted", "failed", "SmartRC からアクセス制限を受けた（3 時間は取得しない）",
            "data/local/meta/smartrc_restriction.json の期限が過ぎるまで待つ。頻度・並列を下げる。",
            r"SmartRCRestricted|SmartRC はアクセス制限"),
    Pattern("schema_validation", "failed", "取得したデータがスキーマに合わず、保存を拒否した",
            "python -m src.scraper.schema_violations summary で項目と値を確認。サイトの構造変更ならパーサーを直す。",
            r"SchemaValidationError"),
    Pattern("gcp_forbidden", "failed", "dev 環境から GCP に接続しようとした（禁止されている）",
            "このジョブが KEIBA_ENV=dev で動いている。ホストの .env を確認する。",
            r"GcpAccessForbidden"),
    Pattern("gcs_auth", "failed", "GCS の認証に失敗した",
            ".env の GCS_* 認証情報（サービスアカウントの鍵）の期限・値を確認する。",
            r"DefaultCredentialsError|invalid_grant|RefreshError|google\.auth\.exceptions"),
    Pattern("missing_module", "failed", "Python パッケージが足りない: {mod}",
            "ホストの conda 環境に入れる（pip install ...）。cron は /opt/conda/bin/python3 で動くので、そちらに入れること。",
            r"ModuleNotFoundError: No module named '(?P<mod>[\w.\-]+)'"),
    Pattern("disk_full", "failed", "ディスクの空きが無い",
            "df -h で使用量を確認し、不要なログ・キャッシュを整理する（rotate_logs.sh、data/cache）。",
            r"No space left on device"),
    Pattern("oom_killed", "failed", "メモリ不足で強制終了された（OOM kill）",
            "他の重い処理（学習・特徴量の再構築）と同時に動いていないか確認する。WSL/コンテナのメモリ上限を見直す。",
            r"(?m)^Killed$|MemoryError|Cannot allocate memory", (137,)),
    Pattern("timeout", "failed", "時間切れで中断された（timeout）",
            "対象の件数が多すぎるか、接続先の応答が遅い。分割して実行するか、timeout を延ばす。", "", (124,)),
    Pattern("sigterm", "failed", "外から停止された（SIGTERM）",
            "ホストの再起動・デプロイ・手動 kill で起きる。その時刻に作業が無ければ、メモリ不足でコンテナが再起動していないか確認する。次の実行で自動的に再開される。",
            "", (143,)),
    Pattern("command_not_found", "failed", "コマンドが見つからない",
            "cron の PATH に無いコマンドを使っている。crontab 先頭の PATH と、スクリプト内のコマンドのパスを確認する。", "", (127,)),
    Pattern("not_executable", "failed", "スクリプトを実行できない（権限またはパス）",
            "chmod +x を確認する（setup_all_cron.sh install が付ける）。", "", (126,)),
    Pattern("postgres_down", "failed", "PostgreSQL に接続できない",
            "ensure_postgres.sh が 5 分おきに起動を試みる。止まったままなら bash scripts/server/ensure_postgres.sh を手で実行して確認する。",
            r"connection to server at .* port 5432 failed|psycopg\.OperationalError|psycopg2\.OperationalError"),

    # ── 警告（完了はしたが、問題の兆候） ──────────────────────────────────────
    Pattern("playwright_missing", "warning", "ブラウザ（Playwright）が入っていない／見つからない: 含水率が取れない",
            "cron は root で動くので /root/.cache/ms-playwright が必要。jovyan 側の ~/.cache/ms-playwright へのシンボリックリンクを張るか、"
            "playwright install chromium を root で実行する。",
            r"Executable doesn't exist at (?P<path>\S+ms-playwright\S*)"),
    Pattern("parse_audit_fail", "warning", "パースの監査で必須項目の欠損が見つかった（PARSE_AUDIT_FAIL）",
            "上の行にどの項目が欠けたかが出ている。サイト側の構造変更ならパーサーを直す。docs/operations/stg_cron_schedule_20261014.md 参照。",
            r"PARSE_AUDIT_FAIL"),
    Pattern("parse_drift_fail", "warning", "flat テーブルの列で空・文字化けの急増が見つかった（PARSE_DRIFT_FAIL）",
            "上の行に列名が出ている。直前の取得で壊れたデータが入っていないか確認する。", r"PARSE_DRIFT_FAIL"),
    Pattern("missing_module_warn", "warning", "Python パッケージが足りず、一部の処理を飛ばした: {mod}",
            "PostgreSQL 取り込み（ETL）などが飛ばされている。ホストの conda 環境に入れる（psycopg2 なら psycopg2-binary）。",
            r"No module named '(?P<mod>[\w.\-]+)'"),
    Pattern("postgres_down_warn", "warning", "PostgreSQL に接続できず、DB への取り込みを飛ばした",
            "ensure_postgres.sh が 5 分おきに起動を試みる。止まったままなら手で起動して確認する。",
            r"DB 未接続|port 5432.*(?:failed|refused)|connection refused.*5432"),
    Pattern("jra_unreachable", "warning", "JRA 公式サイト（www.jra.go.jp）に接続できない",
            "ホストの外向き通信か JRA 側の障害。続く場合は curl -sI https://www.jra.go.jp/ で確認する。開催日の朝は 10 分おきに再試行される。",
            r"HTTPSConnectionPool\(host='www\.jra\.go\.jp'"),
    Pattern("netkeiba_unreachable", "warning", "netkeiba に接続できない（ネットワーク断または相手側の障害）",
            "ホストの外向き通信を確認する（curl -sI https://db.netkeiba.com/）。一時的なら次回以降の取り直し（recent-retry・backfill）で埋まる。",
            r"HTTPSConnectionPool\(host='" + _NK),
    Pattern("netkeiba_login_failed", "warning", "netkeiba へのログインに失敗した",
            "通信断のとき、または別セッションが同じアカウントでログインして無効化されたとき。ID/パスワード（.env の netkeiba_id/netkeiba_pw）も確認する。",
            r"ログイン失敗"),
    Pattern("netkeiba_400", "warning", "netkeiba が 400 Bad Request を返した（ログインセッション切れ、または制限の前兆）",
            "ログインできているか確認する。別セッションの取得が同時に動いていないか、アクセス間隔・並列を見直す。",
            r"400 Client Error: Bad Request for url: https://" + _NK),
    Pattern("gcs_not_saved", "warning", "GCS に保存できなかった（未接続・認証・通信）",
            "GCS の認証と通信を確認する。保存されなかった分は取り直しが必要（data_health の不足一覧に出る）。",
            r"GCS 未接続のため save スキップ|GCS save 失敗|GCS バックオフ中"),
    Pattern("gcs_list_timeout", "warning", "GCS の一覧取得がタイムアウトした",
            "一時的な通信の遅れ。続くなら GCS への通信とバケットの状態を確認する。", r"batch_list_blobs 失敗|Timeout of [\d.]+s exceeded"),
    Pattern("permission_denied", "warning", "書き込み権限が無い: {path}",
            "ファイルの所有者を確認する（cron は root、手で作ったファイルは jovyan のことがある）。chown するか権限を直す。",
            r"Permission denied: '(?P<path>[^']+)'|Permission denied.*?'(?P<path2>[^']+)'"),
    Pattern("fetch_errors", "warning", "取得に失敗したページがある",
            "どのページかは出力の ERROR 行を確認する。数が多いなら接続・ログインの問題の可能性がある。",
            r"\[ERROR\] scraper\.\w+: (?:取得失敗|馬柱取得失敗|馬情報取得エラー)"),

    # ── 最後の受け皿（失敗のときだけ） ─────────────────────────────────────────
    Pattern("python_exception", "failed", "Python の例外で異常終了した",
            "出力の末尾の Traceback を確認する。", r"Traceback \(most recent call last\)"),
]

_COMPILED = [(p, re.compile(p.regex) if p.regex else None) for p in PATTERNS]
_ERRORISH = re.compile(r"ERROR|Error|エラー|Exception|FAILED|失敗|Traceback|Killed")


def _last_match_line(text: str, rx: re.Pattern) -> tuple[str, re.Match | None]:
    """正規表現に当たった最後の行と、その match を返す。"""
    found, hit = "", None
    for line in text.splitlines():
        m = rx.search(line)
        if m:
            found, hit = line.strip(), m
    if hit is None:                                   # 複数行にまたがる正規表現
        m = None
        for m in rx.finditer(text):
            pass
        if m:
            hit, found = m, text[max(0, m.start() - 80): m.end() + 80].strip().splitlines()[-1] if text else ""
    return found[:240], hit


def _label(p: Pattern, m: re.Match | None) -> str:
    if m is None or "{" not in p.label:
        return p.label
    groups = {k: v for k, v in m.groupdict().items() if v}
    path = groups.get("path") or groups.get("path2") or ""
    return p.label.format(mod=groups.get("mod", ""), path=path or groups.get("path", ""))


def _result(p: Pattern, rx: re.Pattern | None, kind: str, text: str, exit_code: int | None) -> dict:
    line, m = _last_match_line(text, rx) if rx else ("", None)
    if not line and exit_code not in (None, 0):
        line = f"終了コード {exit_code}"
    return {"pattern_id": p.id, "kind": kind, "label": _label(p, m), "hint": p.hint, "evidence": line}


def _matches(p: Pattern, rx: re.Pattern | None, exit_code: int | None, text: str) -> bool:
    if exit_code is not None and exit_code in p.exit_codes:
        return True
    return bool(rx and rx.search(text))


def classify(exit_code: int | None, output_text: str) -> dict | None:
    """終了コードと出力の末尾から、原因を返す。成功で何も当たらなければ None。

    失敗（exit_code が 0 以外）: failed パターン → warning パターンの順に探し、当たらなければ『原因を自動判別できません』。
    成功（0 または不明）    : skipped パターン → warning パターンの順に探す。"""
    text = output_text or ""
    failed = exit_code not in (None, 0)
    if failed:
        for kind in ("failed", "warning"):
            for p, rx in _COMPILED:
                if p.kind == kind and _matches(p, rx, exit_code, text):
                    return _result(p, rx, "failed", text, exit_code)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        errs = [ln for ln in lines if _ERRORISH.search(ln)]
        evidence = (errs or lines or [f"終了コード {exit_code}"])[-1][:240]
        return {"pattern_id": "unknown", "kind": "failed", "label": "原因を自動判別できません",
                "hint": "出力の末尾を確認し、繰り返し起きるなら cron_failure_patterns.py にパターンを足す。", "evidence": evidence}
    for kind in ("skipped", "warning"):
        for p, rx in _COMPILED:
            if p.kind == kind and rx and rx.search(text):
                return _result(p, rx, kind, text, exit_code)
    return None
