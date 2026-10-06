"""データ存在チェック／ヘルスチェック CLI。

  python -m src.data_health                    # 現在の KEIBA_ENV で実行
  KEIBA_ENV=stg python -m src.data_health      # 学習PC（GCS は list のみ。download は出馬表の評価窓だけ）
  python -m src.data_health --env stg          # dev PC で stg の要件を評価（データの存在のみ。結果は stg@dev に保存）
  python -m src.data_health --import stg_latest.json   # 学習PC/VPS の結果を取り込む
  python -m src.data_health --index-only       # 全環境の一覧を作り直す

設定は環境変数 ``DATA_HEALTH_*``（``config.py`` 参照）または CLI 引数。CLI > 環境変数 > 既定値。
結果は ``<保存ルート>/<環境キー>/`` に保存され、``<保存ルート>/index.html`` に全環境の一覧が出る。
dev は GCP に接続しない（KEIBA_ENV=dev が必須。未設定は prod 扱い）。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from src.data_health import store
from src.data_health.config import FAIL_ON_VALUES, load_settings
from src.data_health.report import render_text
from src.data_health.runner import JST, run_health
from src.data_health.spec import ENVS


def _d(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="データ存在チェック／ヘルスチェック")
    ap.add_argument("--env", choices=ENVS, help="評価する要件プロファイル（既定: DATA_HEALTH_ENV → KEIBA_ENV）")
    ap.add_argument("--since", type=_d, help="評価開始日（既定: stg/prod=2020-01-01, dev=90日前）")
    ap.add_argument("--until", type=_d, help="評価終了日（既定: 実行日の前日）")
    ap.add_argument("--out-dir", help="保存ルート（既定: data/local/meta/data_health）。直下に環境キーのディレクトリを作る")
    ap.add_argument("--no-infra", action="store_true", help="DB/Redis/GCS 接続チェックを省く")
    ap.add_argument("--no-horses", action="store_true", help="馬カバレッジ（出馬表の download を伴う）を省く")
    ap.add_argument("--include-optional", action="store_true", help="optional カテゴリもスクレイピング計画に含める")
    ap.add_argument("--fail-on", choices=FAIL_ON_VALUES, help="終了コード 2 にする重大度（既定: fail）")
    ap.add_argument("--schema-sample", type=int, help="スキーマ適合率を確認する件数（カテゴリごとの最新 N 件。0 で無効）")
    ap.add_argument("--require-complete", action="store_true",
                    help="2020 年以降が全件スキーマ適合で揃っていなければ終了コード 3（stg の受け入れ判定用）")
    ap.add_argument("--quiet", action="store_true", help="コンソール要約を出さない")
    ap.add_argument("--now", help="現在時刻を固定（YYYY-MM-DDTHH:MM, JST。テスト用）")
    ap.add_argument("--import", dest="import_path", metavar="LATEST_JSON", help="他 PC の latest.json を環境別ディレクトリへ取り込む")
    ap.add_argument("--index-only", action="store_true", help="全環境の一覧（index.html）とダッシュボード（dashboard.html）だけ作り直す")
    ap.add_argument("--dashboard", action="store_true", help="ダッシュボード(dashboard.html)を作り直して場所を表示（チェックは実行しない）")
    args = ap.parse_args(argv)

    from src.config.deployment import keiba_env
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()   # KEIBA_ENV / DATA_HEALTH_* を .env（＋.env.<env>）から読んでから判定する
    try:
        cfg = load_settings(profile=args.env)
    except ValueError as e:
        ap.error(str(e))
    for name in ("since", "until", "out_dir", "fail_on", "schema_sample"):
        val = getattr(args, name)
        if val is not None:
            setattr(cfg, name, val)
            cfg.sources[name] = "cli"

    if args.import_path:
        res = store.import_report(args.import_path, cfg.out_dir)
        print(f"取り込みました: {res['key']}（{'最新として反映' if res['latest_updated'] else '既存の方が新しいため履歴のみ'}）")
        print(f"一覧: {store.base_dir(cfg.out_dir) / 'index.html'}")
        return 0
    if args.index_only or args.dashboard:
        idx = store.rebuild_index(store.base_dir(cfg.out_dir))
        print(f"一覧を更新しました: {idx['html']}\nダッシュボード（ブラウザで開く。サーバ不要・自動更新）: {idx['dashboard']}")
        return 0

    now = datetime.fromisoformat(args.now).replace(tzinfo=JST) if args.now else None
    from src.data_health import dashboard

    base, key = store.base_dir(cfg.out_dir), store.env_key(cfg.env, keiba_env())

    def progress(phase, done=None, total=None, detail=""):      # ダッシュボードに「実行中」を出す
        dashboard.write_run_status(base, key, phase, done, total, detail)

    progress("データチェックを開始")
    try:
        report = run_health(settings=cfg, now=now, infra=not args.no_infra, horses=not args.no_horses,
                            include_optional_in_plan=args.include_optional, actual_env=keiba_env(), progress=progress)
        paths = store.save(report, cfg.out_dir)
    except BaseException:
        dashboard.finish_run_status(base, key, "失敗（中断）")
        raise
    dashboard.finish_run_status(base, key, "完了", detail=f"総合判定 {report['summary']['overall']}")
    if not args.quiet:
        print(render_text(report))
        print(f"\nレポート: {paths['html']}\n計画  : {paths['plan']}\n一覧  : {paths['index']}\n"
              f"ダッシュボード（ブラウザで開く。サーバ不要・約30秒ごとに自動更新）: {paths['dashboard']}")
    comp = report["completeness"]
    if args.require_complete or not args.quiet:
        print(f"\n完全性（{comp['scope']}）: " + ("OK ─ 対象範囲のすべてが揃い、スキーマに適合" if comp["complete"] else
                                               f"NG ─ {len(comp['reasons'])} 項目が未達"))
        for rs in comp["reasons"][:30]:
            print(f"  - {rs['message']}")
    if args.require_complete and not comp["complete"]:
        return 3
    overall = report["summary"]["overall"]
    if cfg.fail_on == "fail" and overall == "fail":
        return 2
    if cfg.fail_on == "warn" and overall in ("fail", "warn"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
