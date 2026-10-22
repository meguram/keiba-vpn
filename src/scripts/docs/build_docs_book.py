#!/usr/bin/env python3
"""docs/ 配下の全 HTML（docs/html/ の各ページ、決定記録 docs/decisions/、運用 docs/operations/、外部予想家の分析 docs/other_predictor/）を、
部・章・節で目次をつけた 1 冊の本（docs/html/book.html）に統合する。

元の HTML は変えない（個別ページの URL・他からのリンク・他の人の編集はそのまま）。この本は元の HTML から**毎回作り直す**ので、
元の HTML を直したら再生成する。docs/ に HTML を足したら、下の PARTS にも載せる（載せ忘れはテストが検出する）。

  python -m src.scripts.docs.build_docs_book
  python -m src.scripts.docs.build_docs_book --out /tmp/book.html

やっていること:
  - 各 HTML から本文だけを取り出す（旧ページの左サイドバー・前後リンク・フッターは外す）
  - 壊れた属性（”_1” のような全角の引用符）を直す
  - id と # リンクを章ごとに名前空間化する（章の間で id が衝突しない）。他のページへのリンク（ディレクトリをまたぐものも）は本の中の位置へ張り替える。
    本の外のファイル（コード・md）へのリンクは、本の位置から見た相対パスに直す
  - 各 HTML の CSS を、章の中だけに効くように範囲を限定して取り込む
  - 左に目次（部 → 章 → 節）。章ごとに「何について書かれているか」「更新日」「状態（現行／旧設計／時点の記録）」「関連する章」を付ける
  - 専門用語は押すとその場に解説が開く（modeling_glossary.py）。図（mermaid）は assets/ の同梱版で、見える位置に来たときに描く
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, unquote

from src.scripts.docs import modeling_glossary as GL

REPO_ROOT = Path(os.environ.get("KEIBA_REPO_ROOT") or Path(__file__).resolve().parents[3])
DOCS_DIR = REPO_ROOT / "docs"
HTML_DIR = DOCS_DIR / "html"
MODELING_DIR = HTML_DIR / "modeling"
CHAPTERS_DIR = MODELING_DIR / "chapters"                      # モデリング各章の HTML（子ファイル）
DEFAULT_OUT = HTML_DIR / "book.html"                          # 本はここ（docs/html/ の直下）
BOOK_NAME = "book.html"
JST = timezone(timedelta(hours=9))


@dataclass
class Chapter:
    slug: str                      # 章の識別子（既定は元のファイル名・拡張子なし）
    blurb: str = ""                # 何について書かれているか（1〜2 文）。空なら元の HTML の最初の段落から作る
    kind: str = "設計"             # 要件 / 設計 / 仕様 / 運用 / 結果 / ログ / メモ / 決定 / 用語
    see: list[tuple[str, str]] = field(default_factory=list)   # (関連する章の slug, 関係の説明)
    src: str = ""                  # docs/ からの元ファイルの相対パス。空なら html/modeling/chapters/<slug>.html
    status: str = ""               # "" = 現行 / "legacy" = 旧設計（一部が現行と異なる）/ "snapshot" = ある時点の記録 / "history" = 決定の履歴
    note: str = ""                 # 状態の説明（現行と異なる点）
    now: list[tuple[str, str]] = field(default_factory=list)   # 現行はこちら (slug, 説明)
    keep_scripts: bool = False     # 元ページの画面操作用スクリプトを残す

    @property
    def path(self) -> Path:
        return DOCS_DIR / (self.src or f"html/modeling/chapters/{self.slug}.html")


@dataclass
class Part:
    title: str
    lead: str
    chapters: list[Chapter]
    label: str = ""                # 「第I部」「付録A」


C = Chapter


def _decision_chapters() -> list[Chapter]:
    """docs/decisions/*.html を、MASTER → AREA → DEC の順に章にする（説明は元のページの最初の段落から）。"""
    files = sorted(p.stem for p in (DOCS_DIR / "decisions").glob("*.html"))
    order = [s for s in files if s == "MASTER"] + [s for s in files if s.startswith("AREA")] + [s for s in files if s.startswith("DEC")]
    rest = [s for s in files if s not in order]
    return [C(s, "", "決定", src=f"decisions/{s}.html", status="history") for s in order + rest]


PARTS: list[Part] = [
    Part("要件と全体像", "何を作るのか、どういう構成か。", [
        C("requirements", "keiba-vpn の要件定義書。目的、要件、KPI、構成、ワークフロー、ユーザーへの提供フロー、実装状況（実装済み／部分／モック／未実装）、関連文書の索引。`/update-requirements` で現状から再生成される。",
          "要件", src="html/index.html"),
        C("ARCHITECTURE", "システム仕様書。コンポーネント、データフロー、モデル、運用の全体像（2026-03〜05 の設計を土台に、現状の差分を冒頭に追記）。", "設計", src="html/design/ARCHITECTURE.html", status="snapshot",
          note="設計の土台は 2026-05。現在の構成（VPS＋GCP＋学習PC）との差分は冒頭の節を参照。",
          now=[("requirements", "要件定義書（最新状態）"), ("server_architecture", "いま動いているサーバー一覧")]),
        C("FULLSTACK_ARCHITECTURE", "フロントエンド（Next.js）・バックエンド（FastAPI / Flask）・DB（PostgreSQL / Redis）・GCS を横断した現状構成図と、各コンポーネントの状態。", "設計", src="html/design/FULLSTACK_ARCHITECTURE.html"),
        C("LAYER_DESIGN", "レイヤー構成・設計仕様。スクレイピング・ストレージ・ML 予測実行・Web API の 5 つの責務層、データフロー、主要な設計パターン。", "設計", src="html/design/LAYER_DESIGN.html"),
        C("server_architecture", "ローカル開発〜stg 運用で動いている全サーバーコンポーネント（FastAPI / Flask / Next.js / MLflow / PostgreSQL / Redis / GCS）の役割・ポート・起動方法。", "運用", src="operations/server_architecture.html"),
    ]),
    Part("データ", "どのデータが、どこから取得され、どこに保存され、どう検証されるか。", [
        C("DATASET_CATALOG", "AI パイプライン／分析に必要なデータセット一覧と、dev / stg / prod ごとのアクセス状況。サンプルの折りたたみつき。", "参照", src="html/data/DATASET_CATALOG.html", keep_scripts=True,
          see=[("DATASOURCES", "取得元の仕様"), ("database_schemas", "保存先の構造")]),
        C("DATASOURCES", "netkeiba・SmartRC 等の取得元ごとの URL パターン、認証、パーサー、ストレージカテゴリ、取得フィールドの仕様。", "仕様", src="html/data/DATASOURCES.html"),
        C("data_scraping_storage_reference", "自動スクレイピングのタスクとキュー、保存先（GCS・ローカル）、カテゴリ別の取得・保存の手順のリファレンス。", "仕様", src="html/data/data_scraping_storage_reference.html"),
        C("database_schemas", "全データストア（GCS の JSON、PostgreSQL、Parquet、ローカルのファイル型ストア、キュー、MLflow、Redis）の場所・形式・キー・列・書き手と読み手。", "仕様", src="html/data/database_schemas.html"),
        C("data_features_reference", "特徴量ストア（data/local/features）のディレクトリ構成、キー設計、各列の意味。", "仕様", src="html/data/data_features_reference.html"),
        C("DATA_HEALTH", "データ存在チェック／ヘルスチェック。どの時期・どのデータが足りないか、どこをスクレイピングすべきか、基盤は疎通しているかを 1 コマンドで可視化する。", "運用", src="html/data/DATA_HEALTH.html"),
        C("data_quality_issues", "2026-10-13〜20 に見つかったデータ・実装の問題（払戻の連結、2025 年のオッズ破損、出馬表フラットの潰れ、追い切りの空保存など）を、解決済みと未解決に分けてまとめたもの。再発を防ぐ検査の仕組みも。", "参照", src="html/data/data_quality_issues.html",
          see=[("DATA_HEALTH", "データの存在・スキーマ適合の検査"), ("training_pipeline_decisions", "調査への影響（§33 B）")]),
        C("data_flow_schedule", "データの収集 → 保存 → 加工 → 統合の流れを、cron の発生時刻つきで示す。1 日のタイムライン、データ別の一覧、手動の工程、処理時間の記入欄。", "仕様",
          see=[("feature_strategy_tp03_tp04", "T-X で実行できるか（成立性）の調査はこちら"), ("cron_schedule", "cron の登録内容")]),
    ]),
    Part("取得と運用", "スクレイピングの設計、cron、SLA、API の入口。", [
        C("scraper_architecture", "JRA 開催スケジュールに連動した自動スクレイピングの設計。タスク種別・実行タイミング・取得対象・ストレージ構成。", "設計", src="html/operations/scraper_architecture.html"),
        C("raceday_prediction_workflow", "毎レースの予測出力までのワークフロー（cron 設計）。前日夕方〜発走前の時系列、馬体重検知から予測の保存まで、影モードと切り替え手順、結果確定側、確認項目と落とし穴。", "運用", src="html/operations/raceday_prediction_workflow.html",
          see=[("cron_schedule", "cron の登録内容"), ("data_flow_schedule", "データごとの流れ")]),
        C("cron_schedule", "cron の登録内容を役割ごとにグループ分けしたもの（時刻は JST）。ホスト実測の稼働状況と、確認済みの問題点。", "運用", src="html/operations/cron_schedule.html",
          see=[("data_flow_schedule", "データごとの流れ"), ("raceday_prediction_workflow", "開催日の予測までの流れ")]),
        C("scraping_sla", "運用 SLA（品質保証基準）と今後の目標（レース前の AI 予測）に向けた要件・現状ギャップ。スクレイピング自動化と予測の処理の流れ。", "運用", src="html/operations/scraping_sla.html"),
        C("stg_frontend_prod_build", "stg（ホスト）の画面配信を next dev から本番ビルド（next build + next start）にする運用。無停止の切替、ホストの 9p が遅い問題への対処（/tmp の作業コピーでビルド）、負荷ガード、反映手順とロールバック。", "運用", src="operations/stg_frontend_prod_build.md"),
        C("prediction_results", "予測成績（予測結果の集計）の運用。日次の集計（暫定→確定）、指標の定義（◎○★・仮想購入）、自動実行の契機と環境変数、API、有効化とロールバック。", "運用", src="operations/prediction_results.md",
          see=[("prediction_backfill", "過去のレースの予測の算出")]),
        C("prediction_backfill", "先週までのレースで予測が無いものを、推論経路で後から算出して画面に出す仕組み（バックフィル）。API の優先順位、算出の手順、as-of の確認結果。", "運用", src="operations/prediction_backfill.md",
          see=[("prediction_results", "予測成績の集計")]),
        C("tracking_v2_precompute", "追走難度 v2 を毎週の開催前夜に算出して保存し、当日は固定値として返す仕組み。保存先・取消時の計算し直しの方針・API・週次の突き合わせ（予測 vs 実測）・バックフィル・cron・反映とロールバック。", "運用", src="operations/tracking_v2_precompute.md",
          see=[("tracking_difficulty_v2", "追走難度 v2 の作り直し（tp-54）の章")]),
        C("megu_v3a_rollout", "めぐ指数 v3a の本番反映の手順書（DB・GCS・日次 cron・退避とロールバック）。", "運用", src="operations/megu_v3a_rollout.md",
          see=[("megu_index_v3a", "めぐ指数 v3a の章")]),
        C("vps_load_and_cache", "VPS（prod の画面）の段階的な負荷テストの結果（同時 100 接続でエラー 0、Next.js 単体は約 400 回/秒で頭打ち）と、VPS 内のキャッシュの現状・改善の候補。", "運用", src="html/operations/vps_load_and_cache.html"),
        C("data_terms_and_public_policy", "取得元（netkeiba・JRA など）の規約の要点と、一般公開・有料化の方針（公開の 3 段階、公開許可リスト、有料ゾーンの推奨度、取得の姿勢）。", "運用", src="html/operations/data_terms_and_public_policy.html"),
        C("local_dataset_endpoints", "src/api/app.py のエンドポイントのうち、ローカルのデータセットを参照するものの一覧。", "参照", src="html/operations/local_dataset_endpoints.html"),
    ]),
    Part("モデリングの方針と分割", "モデルをどう作るかの考え方と、全体の設計、データの分割。", [
        C("modeling_coding_strategy", "モデリングを実装するときのコーディング方針。推奨ライブラリ、ディレクトリ構成、ファイル命名、メモリ節約の原則など。", "設計", status="snapshot",
          note="2026-05 時点の方針。ディレクトリ構成は src/pipeline のサブパッケージ化（features/・models/・inference/）の前のもの。"),
        C("staged_probability_pipeline", "勝率・連対率・複勝率を、位置取り → ペース → 当日馬場 → 血統適性 … と多段階で推定して最終確率につなぐパイプライン（Stage0〜7）の設計。", "設計",
          [("modeling_strategy_tp04_tp07", "実装の方針はこちら"), ("feature_strategy_tp03_tp04", "前段予測量（P1〜P7）の具体化はこちら")], status="legacy",
          note="初期の Stage0〜7 の設計。現行は P1〜P8 の体系で、最終は NN 主体の複数モデル。Stage2〜5 相当のうち不採用になったものがある。",
          now=[("training_pipeline_decisions", "確定した事項"), ("feature_strategy_tp03_tp04", "現行の特徴量・前段予測量")]),
        C("modeling_strategy_tp04_tp07", "tp-04〜tp-07 のモデリング方針。既存ドキュメントの整理（踏襲・更新・矛盾の解消）、検証カタログ、採否基準、各 notebook の戦略。", "設計",
          [("feature_strategy_tp03_tp04", "使う特徴量の設計"), ("training_pipeline_decisions", "確定した結果")], status="legacy",
          note="v2 までの記述が中心。アンサンブル v3・tp-21〜24 は反映していない。",
          now=[("research_overview", "調査の全体像"), ("training_pipeline_decisions", "v3 は §29〜32")]),
        C("advanced_modeling_techniques", "レース直前確率モデルに使う実戦的な ML テクニック。カテゴリ変数のエンコーディング（target encoding など）を中心に。", "メモ", status="snapshot", note="2026-05 時点の調査メモ。"),
        C("reference_insights", "参考資料から取り込んだ設計メモ。ベースラインを先に固定する、確率の信頼性を ROI より先に評価する、など。", "メモ", status="snapshot", note="2026-05 時点の調査メモ。"),
        C("ml_pipeline", "レース直前確率モデルの機械学習パイプライン全体図。Stage0〜7 の流れ、データ定義、テクニック、運用ルール。", "設計", src="html/design/ml_pipeline.html", status="legacy",
          note="Stage0〜7 の初期設計をもとにした全体図。分割・使うモデルなどは学習パイプラインで確定した内容に直してあるが、Stage の体系は初期のまま。",
          now=[("training_pipeline_decisions", "確定した事項"), ("train_valid_test_split_strategy", "分割")]),
        C("horse_pre_race_dataset_spec", "horse_pre_race_dataset（学習の母表）の仕様。目的、設計思想、列の定義、カテゴリ変数の扱い。", "仕様", status="legacy",
          note="発走 10 分前を前提にした設計。現行は T-60（馬体重の発表後）で、母表は LayerA（layer_a.parquet）。horse_pre_race_dataset 自体は未実装。",
          now=[("training_pipeline_decisions", "確定した事項"), ("feature_strategy_tp03_tp04", "現行の特徴量")]),
        C("pre_race_feature_whitelist", "レース直前に使ってよい特徴量のホワイトリスト。テーブル別に、使う列と使わない列（オッズ・人気を入れない原則）を定める。", "仕様", status="legacy",
          note="発走 10 分前を前提にした設計。現行は T-60 と tier（0=前日 / 1=当日朝 / 2=馬体重の発表後）。「オッズ・人気を入れない」原則は現行も有効。",
          now=[("feature_strategy_tp03_tp04", "tier の定義")]),
        C("train_valid_test_split_strategy", "Train / Valid / Test の分割戦略。なぜランダム分割が通用しないか、時系列 CV、補助評価セット、分布の把握。", "設計",
          [("training_pipeline_decisions", "実際に確定した分割と分布差は §2")], status="legacy",
          note="valid を 2024 年の 1 本にする初期案。現行は 1 暦年ずつの expanding window（valid 2022〜2024、test 2025 は最後まで見ない）。",
          now=[("training_pipeline_decisions", "確定した分割")]),
    ]),
    Part("特徴量と血統", "特徴量の設計・算出方法、血統・コース・馬場の知識、当日に作るための仕組み。", [
        C("feature_strategy_tp03_tp04", "tp-02〜tp-04 の特徴量戦略書。ファミリー A〜H（127 列）、前段予測量、ablation の計画、予測フローの図、T-X で実行できるか、特徴量と生データの依存関係。", "設計",
          [("training_pipeline_decisions", "確定した結果はこちら"), ("data_flow_schedule", "各データが集まる時刻"), ("feature_serving_low_memory", "当日の特徴量の作り方")], status="legacy",
          note="tp-02〜tp-04 時点の設計（履歴 127 列・前段 P1〜P7）。現行の 301 列（LayerA・追い切りラップ・血統/追加を含む）には追い付いていない。",
          now=[("adopted_features_v3", "現行の特徴量（グループ別）"), ("training_pipeline_decisions", "採否の確定")]),
        C("feature_engineering_exploration", "特徴量の探索メモ。出走メンバー依存の特徴、仕上げのリズム、コース・馬場、ペース・位置取りの合成、欠損自体を特徴にする、など。", "メモ", status="snapshot", note="2026-05 時点の探索メモ。採否は学習パイプラインの確定事項を参照。",
          now=[("training_pipeline_decisions", "採否の確定")]),
        C("race_performance_rating_system", "レースパフォーマンス（TimeFigure、着差補正、斤量補正など）を算出するシステムの設計と実装。", "仕様"),
        C("track_bias_pedigree_pipeline", "血統 × 馬場傾向（当日・前日の勝ち馬の上りと、父・母父の適性）の特徴量パイプライン。", "仕様", status="snapshot",
          note="2026-10-15 に「同日の後のレースの結果が混ざる」先読みの不具合を修正した（この章の元の記述は修正前の説明を含む）。",
          now=[("training_pipeline_decisions", "修正後の扱い")]),
        C("horse_pedigree_10gen_merge_design", "5 世代血統表を接ぎ木して、10 世代相当の血統を作る設計（アンカー選定、一意キー、アルゴリズム）。", "設計"),
        C("pedigree_stage_research_notes", "血統・馬場傾向の研究方針メモ。ゴール、現状のパイプライン、GCS 負荷を抑える運用原則、未解決の論点。", "メモ", status="snapshot", note="2026-05 時点の研究メモ。"),
        C("NOTE_APTITUDE_5GEN", "適性 3D × 5 世代血統。父系・母系の経路をどう区別して重み付けするか。", "仕様", src="html/pedigree/NOTE_APTITUDE_5GEN.html",
          see=[("PEDIGREE_NOTE_APTITUDE", "血統適性ナレッジ"), ("horse_pedigree_10gen_merge_design", "10 世代への拡張")]),
        C("PEDIGREE_NOTE_APTITUDE", "血統適性ナレッジ（note まとめ → 特徴量 → モデル）。父・母父・牝系のブレンド比率（1.0 / 0.55 / 0.35）など。", "仕様", src="html/pedigree/PEDIGREE_NOTE_APTITUDE.html"),
        C("PERFORMANCE_GENES", "競走馬パフォーマンス遺伝子ナレッジベース。運動能力に関係する遺伝子マーカーと推論システム。", "仕様", src="html/pedigree/PERFORMANCE_GENES.html"),
        C("course_geometry_atlas", "JRA 全 10 競馬場のコースの概形を、競馬場 → コース（芝・ダート、内外回り）→ 距離の順に切り替えて見られる図。", "参照",
          [("course_master_verification", "検証の記録（3 ラウンド）"), ("feature_strategy_tp03_tp04", "特徴量の定義")]),
        C("course_master_verification", "JRA コース形状マスタ（108 コース）の検証記録。3 つの独立収集の突き合わせ、一次資料での再確認と訂正。", "ログ",
          [("course_geometry_atlas", "コースの概形を図で見る")]),
        C("course_performance_similarity", "馬の過去成績から作る『成績が連動するコース』の対応づけ（tp-26）。転移相関・置換検定・FDR、コース概形との一致、類似マップ、勝率予測への増分。", "結果",
          [("course_geometry_atlas", "コースの概形を図で見る"), ("training_pipeline_decisions", "tp-15（形状だけの類似度）の結果")]),
        C("course_aptitude_groups", "期待外れ・期待以上の馬のグループ差分から作るコース適性（tp-29）。グループの定義と件数、有意な属性、コース固有の超過、人気を制御した予測力、勝率予測への増分。", "結果",
          [("course_performance_similarity", "成績が連動するコース（tp-26）"), ("training_pipeline_decisions", "決定ログ")]),
        C("course_specific_effects", "コース特注の要因（tp-31）。V4 の予測残差がコース×属性でずれているかの監査、前の年からの補正の検証、コース特注の特徴量（厳密コース×属性の as-of lift）の増分。", "結果",
          [("course_aptitude_groups", "コース適性（tp-29）"), ("training_pipeline_decisions", "決定ログ")]),
        C("equipment_and_change", "馬具（ブリンカー）と『初めて・替わり』系の状況特徴（tp-32）。馬具データの所在と過去分の使えるか、距離の延長・短縮の段階、芝↔ダートの替わり、初経験、昇級・降級、転厩、乗り替わり、去勢明け、馬体重・斤量、年齢の節目と組み合わせの増分、今後の取得（cron）と前向きの検証。", "結果",
          [("course_specific_effects", "コース特注の要因（tp-31）"), ("training_pipeline_decisions", "決定ログ")]),
        C("geld_after_races", "去勢明け 2 戦目以降の成績（tp-32 GELD の拡張）。去勢明け 1・2・3・4–5・6+ 戦目ごとの勝率・連対率・複勝率・回収率、同条件のセ馬一般・現行モデルの予測とのずれ、去勢前の同じ馬との対応あり比較、層別、特徴量化の増分の簡易評価。", "結果",
          [("equipment_and_change", "馬具と『初めて・替わり』系の状況特徴（tp-32）"), ("training_pipeline_decisions", "決定ログ")]),
        C("transport_and_stay", "輸送のありなし（厩舎の拠点×開催場）と滞在（追い切りの場所）で好走率が変わるか（tp-33）。判定できるデータの範囲、V4 の予測に対する残差、年別の再現性、特徴量の増分。", "結果",
          [("course_specific_effects", "コース特注の要因（tp-31）"), ("training_pipeline_decisions", "決定ログ")]),
        C("growth_curve_weight", "成長曲線（馬体重の変化とパフォーマンス）の特徴量化（tp-39）。馬ごとの『体重が増えると上がる／減ると上がる／無関係』の判別、母集団の成長曲線（性別×年齢×季節×馬格）と履歴の浅い馬への適用、補正ポイントの V4 への増分、megu v3a の版の記録。", "結果",
          [("lap_groups_and_trainer_stats", "追い切りのラップ種別（tp-35）"), ("training_pipeline_decisions", "決定ログ")]),
        C("lap_groups_and_trainer_stats", "追い切りのラップ種別（A1〜B3）と、調教師別・コース別・前週土日の坂路時計の成績（tp-35）。1F ごとのラップに直せる割合と分布、傾向の再現性、V4 への増分、TL_A 再生成の効果。", "結果",
          [("rest_and_interval", "休養明け・レース間隔（tp-34）"), ("training_pipeline_decisions", "決定ログ")]),
        C("training_condition_growth", "調教の時計の出方から読む調子・成長（tp-40）。今回の追い切りサイクルの中の推移・自分の過去との比較・サイクル間の推移・休み明けの曲線・母集団曲線・馬ごとの傾きの 7 群、馬ごとの傾きの再現性（前半/後半）、V4 への増分、履歴の浅い馬（新馬・初出走）での効き、先読みの完全一致検査、推論側の設計メモ。", "結果",
          [("lap_groups_and_trainer_stats", "追い切りのラップ種別（tp-35）"), ("growth_curve_weight", "成長曲線（馬体重、tp-39）"), ("training_pipeline_decisions", "決定ログ")]),
        C("slow_start_real_label", "出遅れ率（実ラベル版）。戦績ページの備考『出遅』を実ラベルにした馬・騎手・調教師・枠・血統の出遅れ率と出遅れ確率（p_slow）。ラベルの被覆の偏り（ラベルの有無が将来の情報になる）とその対処、V4 全体への増分（第 2 段で再現せず不採用）、先読み検査。", "結果",
          [("lap_groups_and_trainer_stats", "追い切りのラップ種別（tp-35）"), ("training_pipeline_decisions", "決定ログ")]),
        C("pedigree_position_cross", "血統の位置・存在割合・クロス・類似種牡馬（tp-38）。祖先 × 位置の区分 × 条件の階層縮約、位置の重みを推定するリッジ回帰、近交係数、血統空間の座標と構成が近い種牡馬による縮約、位置ごとの寄与の割合、V4 への増分（第 1 段・第 2 段）と推論側の設計メモ。", "結果",
          [("lap_groups_and_trainer_stats", "追い切りのラップ種別（tp-35）"), ("training_pipeline_decisions", "決定ログ")]),
        C("similar_horses_pairs", "似た馬は同じレースで着順が近くなるか（tp-41）。力・適性・位置取り・血統・陣営の類似度ごとの『ともに上位に来る』観測/期待、調整後の回帰、組合せ券（3連複・馬連）の確率への補正、買い目の選び方、特徴量の増分。", "結果",
          [("exotic_bet_probabilities", "馬連・ワイド・3連複の確率"), ("training_pipeline_decisions", "決定ログ")]),
        C("workout_remaining_ideas", "追い切りの残りの余地（tp-37）。調教師ごとの時計の標準化・時計の出し方のクラスタ、追い切り×条件の交互作用、本数・間隔・前走比、追い切り騎手・短評、記者コメント、V4 の予測と追い切りの条件を組み合わせた馬券（レース単位・reality check・2026 年の 1 回評価）。", "結果",
          [("lap_groups_and_trainer_stats", "ラップ種別と調教師別の成績（tp-35）"), ("training_pipeline_decisions", "決定ログ")]),
        C("rest_and_interval", "休養明け・レース間隔の分析（tp-34）。間隔の帯・叩き何戦目別の好走率と V4 に対する残差、馬・調教師ごとの再現性、特徴量 8 群の増分、見かけの大きな改善が浮動小数点の端数のリークだった経緯と修正。", "結果",
          [("race_volatility", "荒れやすいレースの分析（tp-30）"), ("training_pipeline_decisions", "決定ログ")]),
        C("race_volatility", "荒れやすいレースの分析（tp-30）。荒れの事後指標、荒れやすい条件・コース、事前予測（オッズなし/あり）、予想の特徴量・温度調整への寄与、3連複の買い目の幅（フォーメーション・流し・点数）への寄与。", "結果",
          [("course_aptitude_groups", "コース適性（tp-29）"), ("training_pipeline_decisions", "決定ログ")]),
        C("megu_index_v3a", "めぐ指数（ユーザー独自のレーティング指数。1点=0.1秒）の改善と本番反映（2026-10-19）。旧版の問題（水準の段差・基準外・補正の過大）、改善の工程と合格基準 10 項目、DB・GCS・日次 cron への反映、使い方（成長曲線・特徴量・想定めぐ指数）、限界。", "結果",
          [("research_overview", "調査の全体像"), ("feature_investigation_catalog", "特徴量調査の総覧"), ("adopted_features_v3", "v3 の特徴量"), ("megu_v3a_rollout", "本番反映の手順書")]),
        C("class_benchmark", "クラス別の基準めぐ指数（突破ライン）。『○歳XXX』のクラスごとに、2着馬の指数の中央値と下位25%（突破ライン）を出す。基準以上の馬は未満の馬より勝ちやすい（2025年の検証）。馬柱と出馬表に重ねて表示する。", "結果",
          [("megu_index_v3a", "めぐ指数 v3a"), ("training_pipeline_decisions", "決定ログ")]),
        C("feature_serving_low_memory", "当日の特徴量を小さなメモリで作る設計。出走馬だけで計算できる理由、サービング用バンドル、検証、見つかった不具合。", "設計",
          [("feature_strategy_tp03_tp04", "特徴量の定義はこちら")]),
    ]),
    Part("学習パイプラインと実験結果", "tp-00〜tp-08 以降の確定事項、チューニング工程、実験結果。", [
        C("research_overview", "特徴量・モデリング・馬券の調査を、結果と状態つきで 1 か所にまとめた一覧。現在地、調査の一覧（採用・不採用・保留）、これからの調査、本書内の食い違い。調査の全体像はここから。", "結果",
          [("training_pipeline_decisions", "各調査の数値と根拠（§N）"), ("adopted_features_v3", "v3 の特徴量"), ("feature_investigation_catalog", "特徴量調査の総覧"), ("megu_index_v3a", "めぐ指数 v3a"), ("data_quality_issues", "データ品質の既知の問題")]),
        C("feature_investigation_catalog", "これまでの特徴量・前段モデル・モデル構造の調査を、概要・結果（勝率 LogLoss の差）・採否・根拠の節つきで網羅した総覧。現行 v4（1,135 列）の構成、枠順×コースの検証、データ・実装の問題、今後の調査対象の一覧。特徴量の採否はここから。", "結果",
          [("adopted_features_v3", "v3 の 301 列のグループ別一覧"), ("research_overview", "調査の全体像"), ("training_pipeline_decisions", "数値の正（§N）")]),
        C("model_accuracy_summary", "予測モデル（アンサンブル v1〜v4）の精度を、train（学習に使ったデータ）・valid（調整用）・test 2025（最後に一度だけ）で並べた総括。過学習の読み方に加え、予測確率の帯ごとの実際の発生率（較正表と較正プロット）と、単勝・複勝の回収率（参考値）。", "結果",
          [("research_overview", "調査の全体像"), ("feature_investigation_catalog", "特徴量調査の総覧"), ("training_pipeline_decisions", "各数値の根拠（§N）")]),
        C("adopted_features_v3", "アンサンブル v3 が使う 301 列を、LayerA・履歴・前段/相対化/追い切りラップ・血統/追加のグループごとに整理。代表列、意味、採用の根拠、注意点、不採用・保留の候補、列の正本の場所。", "結果",
          [("research_overview", "調査の全体像"), ("training_pipeline_decisions", "採否の根拠")], status="snapshot",
          note="v3（301 列）時点の一覧。現行の v4 は 1,135 列（v3 + 追加 834 列）で、追加分の内訳は特徴量調査の総覧と決定ログ §33・§34 にある。",
          now=[("feature_investigation_catalog", "特徴量調査の総覧（v4 を含む）"), ("training_pipeline_decisions", "v4 は §33・§34")]),
        C("training_pipeline_decisions", "学習パイプラインで確定した事項のログ。分割、特徴量、データ完全化の経緯と結果、ablation、前段モデル。**確定事項の正**。", "ログ",
          [("feature_strategy_tp03_tp04", "設計の根拠はこちら")]),
        C("hyperparameter_tuning_workflow", "ハイパーパラメータ・モデル選定の工程。リーク防止の絶対ルール、二層 CV、凍結する順、目的関数、アンチパターン。", "設計", status="legacy",
          note="§10 の「Optuna 未接続」は古い（tp-05d で接続済み。分割は time_split.py）。工程の考え方は現行も有効。",
          now=[("training_pipeline_decisions", "tp-05d の結果")]),
        C("tp05a_experiment_report", "tp-05a の実験結果。勝ち馬確率の目的関数の比較（3 fold × 3 seed）。", "結果",
          [("modeling_experiment_report", "tp-05a〜tp-08 の総合はこちら")]),
        C("modeling_experiment_report", "tp-05a〜tp-08 の総合実験結果。現在の予測パイプラインの精度、段階別の結果、馬券最適化の強化学習（tp-08）、MLflow。", "結果",
          [("training_pipeline_decisions", "確定事項")], status="snapshot",
          note="tp-05a〜tp-08 時点（アンサンブル v1〜v2）の総合結果。v3（test 2025: 勝率 0.21039）と tp-21〜24 は反映していない。",
          now=[("research_overview", "調査の全体像"), ("training_pipeline_decisions", "v3 は §29〜32")]),
    ]),
    Part("オッズと馬券", "オッズの予測、複合馬券の確率、バックテストと馬券推奨の設計。", [
        C("final_odds_model", "最終オッズ予測モデル。発走前のデータから単勝・複勝オッズを推定する LightGBM 3 ヘッドモデルの設計・学習・推論フロー。", "設計"),
        C("exotic_bet_probabilities", "馬連・ワイド・馬単・3 連複の的中確率を、勝率・連対率・複勝率から一貫して計算するモデル（割引 Harville）。", "設計"),
        C("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）。構造の全数調査、モデルと市場のずれ（Benter 混合）、組合せ券、強化学習（リーグ戦・批評役）、多重比較の補正、WIN5 のデータ有無、2026 年での最終評価。", "結果",
          [("exotic_bet_probabilities", "組合せ券の確率の作り方"), ("modeling_experiment_report", "強化学習（tp-08）の結果")]),
        C("race_type_marks", "レースの性質（堅い/荒れやすい）で印（◎○▲）の付け方と買い方を変える方式の評価（tp-43）。荒れ度の定義 6 種、印 10 通り、買い方 20 通り、設計年での選択と検証年・2026 年での確認、reality check、破産確率。切り替えの価値は見つからず、印の付け方（期待値＋複勝確率の閾値）は組合せ券で良い傾向だが 100% 超の確かな根拠は無い。", "結果",
          [("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("similar_horses_pairs", "似た馬は同じレースで着順が近くなるか（tp-41）"), ("race_volatility", "荒れやすいレースの分析（tp-30）"), ("training_pipeline_decisions", "決定ログ")]),
        C("mark_rules", "印（◎○▲★各1頭＋△複数）の付け方をレースの状況（突出・荒れ・拮抗・通常）で切り替えるかの評価（tp-46）。現行の基準と切替を同じ尺度（◎の勝率・連対カバー・3着内の拾い方・回収率・安定性）で比較し、切替は印の質を下げたので採用しない。印の数の実態（旧ルールは平均11.7個）と新ルールの内容も記す。", "結果",
          [("race_type_marks", "レースの性質で変える印と買い方（tp-43）"), ("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("training_pipeline_decisions", "決定ログ")]),
        C("kesi_mark", "消印＝3着内に来る確率が低い馬（人気区分別の目標: 1〜5人気≤10%・6〜9人気なし・10人気以下≤5%）と、推奨列を印（◎○▲★△消）で出す修正（tp-67）。Clopper-Pearson 上側限界での期間別判定、V4 複勝確率の低確率域の較正、10人気以下は複勝確率8%以下で達成、1〜5人気は消し要因の分類器でも10%以下と言える集合がなく付けない、他の印との重なり、初版（上位5人気の最小1頭）の検証。", "結果",
          [("mark_rules", "印の付け方（tp-46）"), ("race_type_marks", "レースの性質で変える印と買い方（tp-43）"), ("training_pipeline_decisions", "決定ログ")]),
        C("ev_calibration_audit", "期待値（的中確率×オッズ）の水準の監査。オッズ側・確率側の恒等式（控除率どおり）、全組合せ買いの回収率、連対 EV の『単勝オッズ×1.65』と複勝 EV の中央値オッズが高く出ていた誤りの修正、組合せ券・買い目提案・レース価値の色分けの較正曲線、縮約係数 k（既定は無効）。", "結果",
          [("exotic_bet_probabilities", "組合せ券の確率の作り方"), ("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("mark_rules", "印の付け方"), ("training_pipeline_decisions", "決定ログ")]),
        C("ensemble_v5", "アンサンブル v5 の構成選択と最終学習（tp-52）。V4 に tl_ 再取得版・CY（tp-40）・ALL_BEST（tp-38）を累積で足した 6 構成の比較（valid 2022〜24）、採用構成、LOYO アンサンブル、最終学習の test 2025 の 1 回評価、推論側の CY の一致検証。", "結果",
          [("training_condition_growth", "調教の時計の出方から読む調子・成長（tp-40）"), ("pedigree_position_cross", "血統の位置・存在割合・クロス（tp-38）"), ("training_pipeline_decisions", "決定ログ")]),
        C("ev_calibration_adopt", "期待値の較正（表示する期待値を実現回収率に収束させる、tp-49）。縮約・線形・単調回帰・定数を walk-forward（2023〜2025 年）と未使用の 2026 年で比べ、馬ごとの期待値（勝・複勝・連対）と買い推奨度は収束するので有効化。推奨ポートフォリオの期待回収率は 2026 年に基準を満たさず（年によって実現 78〜105%）、目安として年別の範囲を併記。買い推奨度・レース価値の閾値を較正後の値で再設計。", "結果",
          [("ev_calibration_audit", "期待値の水準の監査（tp-47）"), ("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("mark_rules", "印の付け方"), ("training_pipeline_decisions", "決定ログ")]),
        C("proven_methods_roi", "回収率 100% 超の実績者の手法と検証（tp-44）。公開情報から集めた手法 12 件（種類の印・検証可能性・原文の場所）、プール間の不整合（Hausch–Ziemba・Dr. Z 型）の 67 戦略、公開ルール（函館の3番人気）の再現、WIN5 のキャリーオーバー、発走直前のオッズ低下。検証できる形で 100% 超を示した公開記録は無く、自前の検証でも区間下限が 100% を超える手法は無かった。", "結果",
          [("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("race_type_marks", "レースの性質で変える印と買い方（tp-43）"), ("training_pipeline_decisions", "決定ログ")]),
        C("bet_optimization", "馬券最適化（強化学習ほか）を、最新モデル（V4）と修正後の期待値で大規模に回す（tp-50）。レース×買い目の表（2022〜2026 年）に、固定ルール・LightGBM・MLP アンサンブル・方策勾配・PPO・リーグ戦・進化戦略・決定木の約 900 方策を walk-forward で比較。強化学習系は『賭けない』に近い方策に収束し、最良はワイドの LightGBM 系。多重比較の補正（Reality Check）と、2026 年の事前固定・1 回評価、購入時点オッズの感度。", "結果",
          [("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("trifecta_true_odds", "3連単の実オッズでの再評価（tp-48）"), ("ev_calibration_adopt", "期待値の較正（tp-49）"), ("training_pipeline_decisions", "決定ログ")]),
        C("trifecta_true_odds", "3連単の実オッズ（2024〜2026 年、9,711 レース）で、tp-36 が保留した買い方を再評価（tp-48）。データ品質（Σ(1/オッズ)・払戻との一致）、3連単の EV 上位 k 点・しきい値・縮約後しきい値・プール間の不整合、3連複（3連単プール由来の価格）、印の組合せ、的中率が高い券種（複勝・ワイド・馬連）の『期待値が高いものだけ』、購入時点オッズの感度、Reality Check、2026 年の最終評価。区間下限が 100% を超える方策は無く、100% 超は高配当に依存した。", "結果",
          [("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("ev_calibration_audit", "期待値の水準の監査（tp-47）"), ("proven_methods_roi", "回収率 100% 超の実績者の手法と検証（tp-44）"), ("training_pipeline_decisions", "決定ログ")]),
        C("ensemble_structure", "アンサンブルの構造（多層化・バギング・特徴量グルーピング、tp-51）。V4 の 7 メンバーの OOF での多層スタッキング・後段、特徴量 16 グループによる専門家・leave-one-group-out・ランダム部分空間・階層スタッキング、別系統メンバー（CatBoost・extra_trees・全列 XGBoost）と seed バギングの追加。勝率・連対だけを『現行 + クロス対象 LR』にする合成が基準の境界で、他の構造は V4 に足しても効果が無い。", "結果",
          [("training_pipeline_decisions", "決定ログ"), ("feature_investigation_catalog", "特徴量調査の総覧")]),
        C("ensemble_diversity", "アンサンブルの幅出し（ランダムフォレスト・決定木の袋詰め・別の目的関数・見せ方違い）と、メンバー別・条件別のハイパーパラメータ（tp-63）。V5 の 7 メンバーの多様性の棚卸し（GBM 4 本は実質同じ、多様性は NN・線形側）、sklearn のランダムフォレスト・浅い決定木の袋詰め・単一の決定木、rank_xendcg、芝ダ別・レース質（ペース段階）別の専門モデル、直近 3 年・半減期の重み、ロジスティック回帰を V5 に足した増分。現行の合成では重み 0、L3 型の上で木 3 種が勝率 −0.00013（効果量 0.0002 に届かない）。", "結果",
          [("ensemble_structure", "アンサンブルの構造（tp-51）"), ("feature_cross_search", "特徴量の掛け合わせ探索（tp-58）"), ("training_pipeline_decisions", "決定ログ")]),
        C("pace_consistency", "想定位置取り・追走難度・想定上がりの連動の検証と整合層（tp-53）。3 つは別系統で共通のペース想定が無く、追走側のペースは 91% がミドルで実測と無相関、想定上がりは位置取りにほぼ反応しない（実測は後ろの馬ほど速く、ペースが速いほど差が大きい）。位置取り側の先行勢からペースを推定し、位置取り×ペースと自己の過去の上がり偏差で上がりを決める整合層（既定オフ）を、2023〜2025 年の walk-forward で検証（上がり偏差の相関 0.07〜0.16 → 0.35〜0.36）。", "結果",
          [("training_pipeline_decisions", "決定ログ")]),
        C("lap_forecast", "ハロンごとのラップ予想（ラップ傾向別のタイム＋各傾向の発生確率、tp-55）。レース前の特徴量（脚質の分布・枠並び・SmartRC の前走までのテン1F・コース・同日の先のレースのラップ）から、スロー/ミドル/ハイ（par との差の前半後半の分位）の発生確率と、傾向ごとのハロン別タイムを予測する混合モデル。2022〜2025 年の walk-forward、80% 帯のカバー率、特徴量の抜き取り、2026 年の 1 回評価、勝率予測への増分、予測 API のフィールドとフラグ。", "結果",
          [("pace_consistency", "想定位置取り・追走難度・想定上がりの連動（tp-53）"), ("training_pipeline_decisions", "決定ログ")]),
        C("tracking_difficulty_v2", "追走難度 v2（tp-54）。『想定位置取りを取るために必要な通過ラップが、その馬にとって楽か大変か』を、想定位置取り×想定ペース×コース（要求）とその馬の過去 5 走（供給）から、実測の終いの落ち（過去の力のベースラインを引いた上がり偏差）を予測して表す。現行の追走難度は実測の終いの落ちをほぼ予測せず（相関 −0.03〜−0.06）、新しい指標は 0.22〜0.23（十分位で −0.3〜+0.5 秒）。ペース想定の較正し直し（逃げ0頭・交互作用で相関 0.40→0.45）、前走比の副次ビュー、V5 への増分の一次選別（足す根拠なし）、フラグ KEIBA_TRACKING_V2 / KEIBA_PACE_COHERENCE_V2（既定オフ）。", "結果",
          [("pace_consistency", "想定位置取り・追走難度・想定上がりの連動の検証（tp-53）"), ("training_pipeline_decisions", "決定ログ")]),
        C("weight_growth_performance", "馬体重の増減と指数の結びつき・成長時期の血統・成長指数（tp-57）。前走比の体重変化はめぐ指数の変化と直線では結びつかず（大きく動くほど悪化の逆 U 字）、自分の過去平均より重い状態が指数の高さと結びつく。馬ごとの結びつきは先の走を予測するほど安定せず（個体の指標は出さない）、種牡馬の体重の伸びは産駒で再現するが指数の伸び（晩成度）は父でのみ弱く再現。T-60 時点の成長指数は追い切り・厩舎コメントと成長期が主役で体重・血統の上乗せはほぼ無く、5 段階で実績が単調に分かれる。", "結果",
          [("megu_index_v3a", "めぐ指数 v3a"), ("tracking_difficulty_v2", "追走難度 v2（tp-54）"), ("training_pipeline_decisions", "決定ログ")]),
        C("yosou_procedure", "ユーザー自身の予想手順の再現と評価（tp-56）。レースの傾向（直前の同競馬場の流れ）→ 馬のタイプ分け → 適合グループ内で実力上位に印 → 穴/人気別の買い目・レース回収率 500% の設計、を先読みなしの特徴量・印・買い目にして、各段階の単独の予測力、V5 への増分の一次選別（27 群）、統合した印の質、72 方策の馬券バックテスト（Reality Check）、2026 年の 1 回評価を示す。V5 への上乗せ・馬券の 100% 超はともに見つからず、手順を文字どおり（適合内から印）にすると悪化。", "結果",
          [("pace_consistency", "想定位置取り・追走難度・想定上がりの連動の検証（tp-53）"), ("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("training_pipeline_decisions", "決定ログ")]),
        C("deployment_readiness", "各モデルの導入可否（精度が導入可能なレベルか）の確認。基準を 2026 年の結果を見る前に固定し、v4（現在画面に出ている予測）・v5・前段/補助ヘッド・予測最終オッズ・追走難度 v2/ペース v2/ラップ予想・期待値の較正と印・組合せ券の確率を、独立データの 2026 年で判定する。v4 は条件付き導入可（連対の悪化が基準を 0.00015 超過、被覆 96.7%、モデルと市場の差が月とともに拡大）、v5 は抜き取り 122 レースで v4 と同等以上だが優劣は示せない、複勝の期待値は最上位で実現を 0.20 上回る、予測最終オッズの単勝は不可。v5 を predict に切り替えてよいかの所見と、必要な改善の優先順。", "結果",
          [("ensemble_v5", "アンサンブル v5 の構成選択と最終学習（tp-52）"), ("ev_calibration_adopt", "期待値の較正（tp-49）"), ("lap_forecast", "ハロンごとのラップ予想（tp-55）"), ("tracking_difficulty_v2", "追走難度 v2（tp-54）"), ("exotic_bet_probabilities", "組合せ券の確率の作り方"), ("training_pipeline_decisions", "決定ログ")]),
        C("v6_features", "v6 への追加候補の評価。V5 の構成 d（1,142 列）に累積して、新しい知見（追走難度 v2・位置取り予想・ラップ予想・展開タイプ・枠順バイアス・クラス基準）の別の使い方（独立列・レース内相対・シナリオ条件付き・前段 S3 の作り直し）と、他の調査の候補（めぐ指数/netkeiba 指数の履歴 RB・非 JRA 過去走・rs59・展開利・馬主名の正規化）を同じ手順で評価。独立列は全て不採用、RB が単独で基準を満たし、前段 S3 の作り直し（S3B）は RB の上でも増分があり、RB + S3B − 騎手/調教師の通算 10 列 − fe_year_frac（v6-lite、1,147 列）が勝率 −0.00059・連対 −0.00074・複勝 −0.00064。再現（新 seed）の結果と v6 最終学習に必要な作業。", "結果",
          [("ensemble_v5", "アンサンブル v5 の構成選択と最終学習（tp-52）"), ("index_variance", "めぐ指数と netkeiba 指数の分散・外れ値の扱い（tp-65）"), ("pace_type", "展開タイプ（tp-61）"), ("lap_forecast", "ハロンごとのラップ予想（tp-55）"), ("tracking_difficulty_v2", "追走難度 v2（tp-54）"), ("training_pipeline_decisions", "決定ログ")]),
        C("drift_diagnosis", "ドリフト診断：2026 年（テストセット②）にモデルと市場の勝率 LogLoss の差が開いた原因の切り分けと、週次監視の設計。差の拡大は 6〜10 月（同季節の基準に対して +0.0080、z=5.6）で、市場側・構成・季節・較正では説明できず、予測の尖り（順位付け）が落ちた形。調教ページ（horse_training）の被覆が 4 月から崩れ（4 月 79%・5 月 18%・6〜9 月 5〜6%）、調教系の特徴量 87 列が約 95% の行で欠損のまま予測されていた。欠損の再現実験、固定モデルの経年、寄与の概算。週次監視（直近 8 週の窓＋累積、EWMA/CUSUM、ラベルなしの先行指標、4 段階の警報、層別の FDR、v4/v5 shadow）の設計と、2025〜2026 年の各週へのバックテスト（2025 年の誤警報 3.8%）。", "結果",
          [("deployment_readiness", "導入可否の確認"), ("ensemble_v5", "アンサンブル v5（tp-52）"), ("feature_gap_audit", "特徴量のやり残しの棚卸し"), ("training_pipeline_decisions", "決定ログ")]),
        C("feature_gap_audit", "モデルを作り直す前の特徴量のやり残しの棚卸し。試した群の採否・増分の総括、V5 の列の由来、schema_defs.json 42 カテゴリの未使用データ（非 JRA の過去走・馬ごとの区間ラップ・母側きょうだい・併せ馬）、実行中の調査（v6・交互作用・予想家）と重なるものの除外、やり残し候補 13 個（見込み・必要データ・先読みリスク・工数・2 段階の評価計画）と、再学習の前にやるもの・後でよいもの。特徴量 1 群の見込み（≤0.0006）より 2026 年を含めた再学習の設計のほうが効く可能性が高い、という整理。", "結果",
          [("feature_investigation_catalog", "特徴量調査の総覧"), ("ensemble_v5", "アンサンブル v5（tp-52）"), ("deployment_readiness", "導入可否の確認"), ("training_pipeline_decisions", "決定ログ")]),
        C("handicapper_viewpoints", "予想家の観点の調査。15 のレース名で note の公開記事を検索して 249 本（X の本文は取得できず 0 件）を集め、予想の観点の頻度表を作り、V5 の特徴量との対応を 4 分類（入っている・試して不採用・データはあるが未実施・データなし）で整理。未実施でデータがあるのは、当日・前日のレースから見た馬場の実測の偏り（新規）、進路のロス・包まれ（棚卸しの候補 2・6 を補強）、前走ステップの傾向、騎手の積極性。予想家の主張は未検証で、観点の一覧であって効果の証拠ではない。", "結果",
          [("feature_gap_audit", "特徴量のやり残しの棚卸し"), ("feature_investigation_catalog", "特徴量調査の総覧"), ("yosou_procedure", "ユーザーの予想手順の再現（tp-56）"), ("training_pipeline_decisions", "決定ログ（§41）")]),
        C("data_acquisition", "新規データの取得（セリ取引価格・馬主 ID・非 JRA 過去走・馬ごとの区間ラップ・併せ馬・母側のきょうだい）。セリ価格・馬主 ID・近親馬は既に取っている馬ページのプロフィール表にあり（読み捨てていた）、馬体測定は取れない（JBIS は Crawl-delay 600 秒）。パイロット 635 頭で価格あり 28%・馬主 ID 100%、基準の OOF への簡易検定では優位は示せず（検出限界は勝 LL ±0.0005）、本取得は約 39,000 頭・約 100 時間で承認待ち。非 JRA 過去走は保存済み（全走の 51%）で勝 −0.00015 だが基準未満、母側のきょうだいは複勝 +0.00106 で不採用。馬ごとの区間ラップは取得先が 2024 年以降しか提供せず、併せ馬メモは保存済みが 5.6% で年に偏る。", "結果",
          [("feature_gap_audit", "特徴量のやり残しの棚卸し"), ("handicapper_viewpoints", "予想家の観点の調査"), ("ensemble_v5", "アンサンブル v5（tp-52）"), ("training_pipeline_decisions", "決定ログ")]),
        C("feature_cross_search", "特徴量の掛け合わせ（交互作用・組合せ）の探索（tp-58）。V5 の OOF 確率へのスコア検定で、重要度上位 210 列の積・差・対数比 44,631 候補を再学習なしで総当たり（発見 2022–23・確認 2024・多重検定補正）、手設計のカテゴリ交差 12 種（as-of 階層縮約）と数値交差 30 種、交互作用を表現できる NN（MLP/DCN-v2/FM/DeepFM）、残差ブースティングで『加法と交互作用』の取り分を分離。結果は掛け合わせの上乗せなし（現行の合成に対する残差ブースティングは深さ 3 でも勝率 −0.00002）、NN は構造でなく別系統として勝率 −0.0002 だが L3 と重なる。", "結果",
          [("feature_gap_audit", "特徴量のやり残しの棚卸し"), ("ensemble_structure", "アンサンブルの構造（tp-51）"), ("feature_investigation_catalog", "特徴量調査の総覧"), ("training_pipeline_decisions", "決定ログ")]),
        C("jockey_trainer_roi", "騎手・調教師の分析と、単勝・複勝の回収率として稼げるか（tp-59）。勝率・複勝率は前年→翌年で強く再現（順位相関 0.7〜0.9）するが回収率は再現せず（0〜0.3、本物の差の割合は単勝で最大 0.22）、100% を超えると言える騎手・調教師・組み合わせは全数調査で 0（延べ約 4,600 中、偶然の期待約 230 に対し 1）。市場とモデルは強さ別に ±0.5pt で較正済みで、人気の騎手・調教師は 5〜7pt 買われすぎ。回収率で買う規則 528 通りは Reality Check p=0.9995。唯一の手がかりは『単勝 EV≥1.5 かつ騎手・調教師の勝ち数がともに市場の見込みを上回る』（5 年とも 122〜147%）だが、事後に絞った候補で穴馬依存のため凍結して前向きに追跡する。特徴量は既存列が gain の約 1%、V5 の OOF に対する関係者の残差が一次選別を通る。", "結果",
          [("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("proven_methods_roi", "回収率 100% 超の実績者の手法と検証（tp-44）"), ("feature_gap_audit", "特徴量のやり残しの棚卸し"), ("training_pipeline_decisions", "決定ログ（§42）")]),
        C("jockey_switch_ev", "騎手の乗り替わり（有力↔非有力）に期待値はあるか。有力＝その日より前の直近 365 日の平地勝利数の上位 N 人（回収率は使わない）、型は継続／非有力→有力／有力→非有力／有力→有力／非有力→非有力。事前に固定した 38 セル × 単勝・複勝 = 76 本を選択 2021〜2024 で検定し（Holm と Reality Check）、test 2025 は最良の数セルの評価 1 回、2026 は記述のみ。結果は期待値の取れる型なし（平均利益が正のセル 0/76、Reality Check p=1.00）。主 4 型の回収率は 65〜87% で継続騎乗や全体と区別できず、モデルの勝率予測に対する実績の比も年ごとに 1.0 の周りを上下するだけ。『有力→非有力で前走好走』は選択期間で買われすぎだったが 2025・2026 は戻ったため避ける規則にもしない。凍結する候補なし。", "結果",
          [("jockey_trainer_roi", "騎手・調教師の分析と回収率（tp-59）"), ("equipment_and_change", "馬具・条件替わり（tp-32）"), ("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("training_pipeline_decisions", "決定ログ")]),
        C("edge_factor_mining", "買う要素の網羅探索（tp-72）。モデル（V4 の OOF）と市場の両方に過小評価され、かつ回収率に妙味のある状況を、事前登録した 881 型（性別×季節・夏の牝馬、クッション値・含水率×脚質×枠、逃げはさみ、ブリンカー×脚質、枠×脚質のコース別、体重・追い切り・間隔の同時発生、血統×騎手ほか）で検定（検定 1,762 本、発見 2022–24・確認 2025 を 1 回・2026 は記述のみ）。穴馬の好走の共通要素（1,110 本）と、記述子 43 のペア走査（903 ペア×3 目的変数、学習を伴わない集計版）も行った。結果は期待値の取れる要素なし（回収率の区間が 100% を超える型・ペアは 0）。再現したのは『夏の牝馬』が勝ち +0.55pt・複勝 +0.85pt 過小評価される点だが、回収率は 74〜77% で買い目にならない。クッション値はモデルが ±0.5pt で較正済み、逃げはさみは理論と逆に初角が後ろへ下がる、ブリンカー初着用は初角が前へ出るが市場が織り込み済み。", "結果",
          [("jockey_switch_ev", "騎手の乗り替わりに期待値はあるか"), ("geld_after_races", "去勢明け 2 戦目以降の成績"), ("jockey_trainer_roi", "騎手・調教師の分析と回収率（tp-59）"), ("feature_cross_search", "特徴量の掛け合わせ探索（tp-58）"), ("training_pipeline_decisions", "決定ログ（§62）")]),
        C("pedigree_roi", "血統の単勝・複勝回収率の分析と、血統ベースのレース質分析（tp-60）。【中心課題】血統マップのグループ（構成埋め込みのクラスタ K=16/40・父の系統・産駒数上位 30 の父）× レースの質（展開タイプ 9・ラップ型 6・上がり水準・馬場・勝ち時計・荒れ度）で、人気以上の好走の差分を測り、事前の質の確率で未知のレースに適用（オラクル上限と実運用の下限）。効果は弱く翌年の再現も低い（順位相関 −0.05〜+0.37、BH 通過 6/1,986 セル、説明力 R² ≤ 0.0007）。レース質ラベルの共有表（data/local/modeling/shared_cache/race_quality/）を新設。血統の回収率は、区間の下限が 100% を超えるセル 0・規則 198 本の reality check p = 1.00、勝率・複勝率は翌年も続くが回収率は続かず、EV 補正も騎手・調教師規則への重ね掛けも再現しない。10 世代の血量（被覆 99.9999%、出現率で候補 169 頭 → 代表 40 頭）、親の ID を持ち込まない構成埋め込み（全兄弟 = 距離 0、サドラーズウェルズ→ヌレイエフ 3 位、Node2Vec は不適）、近さ ≠ 適性（全兄弟でも能力の相関 0.13）。", "結果",
          [("jockey_trainer_roi", "騎手・調教師の分析と回収率（tp-59）"), ("pedigree_position_cross", "血統の位置・存在割合・クロス（tp-38）"), ("pace_type", "展開タイプ（tp-61）"), ("lap_forecast", "ハロンごとのラップ予想（tp-55）"), ("similar_horses_pairs", "似た馬どうしのペア"), ("training_pipeline_decisions", "決定ログ")]),
        C("pace_type", "展開タイプ（ペース × 隊列の凝縮度）と、展開シナリオごとの各馬の確率（tp-61）。文献（note の解説記事約 380 本）のラップ・ペースの分類を軸に分解して MECE の規則（6 型）にし、実測 1.9 万レースで脚質別の成績を検証。成績を分けるのは『ペース 3 段階 × 隊列 3 段階』で、中緩み・加速の細分は上乗せなし（スロー・縦長は逃げ 1.34／追込 0.71、スロー・凝縮は逃げ 1.17／追込 0.94）。ハナ主張馬の数と前半3F、事前予測の精度（隊列は出せる、ペースは UI の 3 段階より良い）、展開シナリオ別の各馬の確率（表示・説明用、精度は変えない）、予測 API の pace_type / scenario_forecast（KEIBA_PACE_TYPE=1、既定オフ）。", "結果",
          [("lap_forecast", "ハロンごとのラップ予想（tp-55）"), ("pace_consistency", "想定位置取り・追走難度・想定上がりの連動（tp-53）"), ("tracking_difficulty_v2", "追走難度 v2（tp-54）"), ("handicapper_viewpoints", "予想家の観点の調査"), ("training_pipeline_decisions", "決定ログ（§43）")]),
        C("prob_calibration_gap", "確率の較正（予測と実績の差）の改善（tp-70）。モデルの複勝確率は人気馬の『来ない』を過小に見積もり（3〜5 人気で予測 3〜10% の馬が実績 13〜29%）、10 人気以下は過大。原因は『モデルが市場から離れるほど自信過剰』で、実績はモデルと市場の間にある。単勝オッズから作る市場の確率との対数線形ブレンドを、レース内で合計 1/2/3 に合わせる較正で、生に対し LogLoss が勝率 −0.0049・連対 −0.0073・複勝 −0.0076（2026 年 −0.0077/−0.0114/−0.0124）、低確率の人気馬のずれ 7.4pt→1.6〜3.0pt、EV の |予測−実現| 0.31→0.05〜0.08、組合せ券 −2〜3%、◎の勝率 +1.6pt。ただし個々の馬の見分けは直らず（人気馬の低確率層は約 16% 来る）、2026 年は勝率・連対でモデルが市場に追加情報を持たない。頑健な 100% 超は無し。導入は実装済み（KEIBA_PROB_CALIBRATION=1、既定オフ・本番未反映）で、配線・EV 較正の作り直し（二重補正を避ける）・★と消印のしきい値の再設計・画面表示・ドリフト監視を含む。", "結果",
          [("kesi_mark", "消印（tp-67）"), ("ev_calibration_adopt", "期待値の較正（tp-49）"), ("ev_calibration_audit", "期待値の水準の監査"), ("exotic_bet_probabilities", "馬連・ワイド・3連複の確率"), ("drift_diagnosis", "ドリフトの診断"), ("training_pipeline_decisions", "決定ログ（§51）")]),
        C("pace_benefit", "過去走の展開利（展開に助けられたか）と、次走の期待展開とのずれ（tp-64）。実測の展開 9 通り（tp-61）× 脚質ごとに『平均より着順がどれだけ良かったか』を、その年より前の年だけで推定して各走に当てはめ（標本外の相関 0.12〜0.14・傾き約 1）、直近の展開利が大きい馬ほど V5 の複勝確率が約 1pt 過大（上位 20% − 下位 20% = −1.05〜−1.27pt、区間は 0 を跨がない）だが、今走の期待展開利との交互作用は検出できず、V5 への増分は再学習なしの先行指標で基準 0.0002 に届かない見込み（本評価はスロット待ちで未実施）。", "結果",
          [("pace_type", "展開タイプ（tp-61）"), ("tracking_difficulty_v2", "追走難度 v2（tp-54）"), ("lap_forecast", "ハロンごとのラップ予想（tp-55）"), ("training_pipeline_decisions", "決定ログ")]),
        C("index_variance", "めぐ指数と netkeiba タイム指数の、分散と外れ値の扱いの比較（tp-65）。絶対どうしで SD は同程度（17.3 対 18.7）、1 走のノイズ（馬内 SD 11.7 対 11.9 点）も同じで、netkeiba の ICC が高いのは馬間の差が大きいため。netkeiba に下限・固定の上限は無く、大差負けを緩やかに縮める程度（8 秒超で約 9 点／秒、30 秒超で 6.8）で、値は一度出ると改訂されない。同じレースの勝ち馬−40 点の下限をどちらの指数にも入れると次走予測が改善（+0.006、netkeiba は減衰 0.5 で +0.013）、2 指標を併用すると片方だけより +0.03〜0.05。V5 への増分（LightGBM）と、運用での出走前ページの取得可否は別途。", "結果",
          [("megu_index_v3a", "めぐ指数 v3a"), ("race_performance_rating_system", "レーティング指数の設計"), ("training_pipeline_decisions", "決定ログ")]),
        C("megu_excusable", "めぐ指数 v3b の集約と、無視してよい低下（excusable）の判別（tp-69）。v3b（勝ち馬−40 点の下限＋低信頼の走の重み 0.5＋減衰 0.65 の加重 Huber）は次走予測を +0.0036 改善（採用）。不利・状態の異常で一過性に低い走かを結果表の事実・レース水準・備考・馬体重・間隔・履歴の外れ度から判定する器は AUC 0.57〜0.60（ベースライン 0.53〜0.55）、判定した走の反発率 52〜60% 対 41〜46%（lift 1.2〜1.4、誤除外 40〜48%）と弱く、判定した走を集約から除くと次走予測が −0.003〜−0.008 悪化するため、除かず表示用の参考にとどめる。V5 への増分は実行待ち。", "結果",
          [("index_variance", "指数の分散と外れ値の扱い（tp-65）"), ("megu_index_v3a", "めぐ指数 v3a"), ("pace_type", "展開タイプ（tp-61）"), ("training_pipeline_decisions", "決定ログ")]),
        C("megu_marking", "めぐ指数の印（好走に値する馬）と、過去走の集約方式の比較（tp-66）。直近 5 走を新しい走ほど重い加重 Huber（低信頼は半分）で集約する v3b が、レース内の判別力（AUC 0.67）、上位 1 頭の倍率とも最良で、最高値や 5 走の平均より良い。印は『レース内 1 位かつ突破ライン以上＝めぐ◎（3 着内率 約 48%・倍率 2.1）、2〜3 位＝めぐ○』。V5 の確率（AUC 0.79）には及ばず、V5 に足せる情報は小さい（2022〜24 年 −0.00018、2025 年 +0.00007）ので、人気以上の好走を拾う印ではなく表示用の補助。API は megu_mark、画面はめぐ指数の専用テーブルに印の列。", "結果",
          [("class_benchmark", "クラス別の基準めぐ指数"), ("megu_excusable", "v3b と excusable（tp-69）"), ("index_variance", "指数の分散と外れ値の扱い（tp-65）"), ("training_pipeline_decisions", "決定ログ（§55）")]),
        C("monthly_roi_search", "月次過半・年間 100% 超の馬券戦略の探索（tp-62）。『その年より前だけで選んで翌年に使う』ウォークフォワード（検証 2023〜2026）で、期待値の族（騎手・調教師の条件つき、戦略 5〜7 万通り）と、的中率の下限×レース回収率 500%・的中率 20% を課す族の両方で、年間 100% 超かつ月の過半が 100% 超を複数年続けて満たす方策は無し（的中率下限の族が 2023 年だけ通過 = 運の確率 3% 台、2024・2025 年は 49% 前後）。帰無（市場が正しい世界で選び方ごとやり直す）で運の確率を出し、事後に基準を満たす戦略は帰無でも数百〜数千ある。絞り込みなしで水準だけを課した馬連が 97〜122% で最も近いが、確証には 26 年（真の 110%）要り、月次過半は真の 120% でも約 7%（ワイド約 50%）。馬連の候補は凍結して前向きに追跡。", "結果",
          [("betting_edge_search", "馬券で回収率 100% を超える余地の探索（tp-36）"), ("jockey_trainer_roi", "騎手・調教師の分析と回収率（tp-59）"), ("ev_calibration_adopt", "期待値の較正（tp-49）"), ("training_pipeline_decisions", "決定ログ（§46）")]),
        C("betting_backtest_dataset_spec", "betting_backtest_dataset の仕様。確率テーブル・オッズ・結果を結んだバックテスト用データの列と粒度。", "仕様", status="legacy",
          note="safe_ev を前提にした設計。tp-08 の実装は Benter ブレンドで、safe_ev は使っていない。",
          now=[("modeling_experiment_report", "tp-08 の結果")]),
        C("betting_policy_reinforcement_learning", "馬券推奨レイヤと強化学習の設計議論。なぜ機械学習の延長と切り分けるか、行動サイクル、バックテスト環境。", "設計",
          [("modeling_experiment_report", "強化学習（tp-08）の結果")], status="legacy",
          note="強化学習を第一候補とした設計議論。tp-08 の結果は「確定オッズでは優位なし（回収率は最良でも約 88%）」。",
          now=[("modeling_experiment_report", "tp-08 の結果")]),
    ]),
    Part("推論・MLflow・コスト", "予測を出す仕組み、モデルの管理、費用。", [
        C("inference_services", "推論層の設計。src/pipeline/inference/ のサービスと MLflow ランタイム、キャッシュのフロー、API エンドポイント。", "設計", src="html/design/inference_services.html"),
        C("mlflow_platform", "MLflow によるモデリング・予測プラットフォーム。モデルキー単位の学習・Registry・実行サービス・API キャッシュ。モデルの正は catalog.py。", "設計", src="html/design/mlflow_platform.html"),
        C("cost", "GCS のコスト分析レポート（GCS の操作コストのみ。2026-04 時点）。GCP 全体の試算は docs/operations/cost-estimate.md。", "結果", src="html/design/cost.html", status="snapshot",
          note="数値の基準日は 2026-04。"),
    ]),
    Part("決定記録", "なぜそう決めたかの記録。当時の判断の履歴で、書き換えず、古くなったものは注記を足す。", _decision_chapters(), label="付録A"),
]
for _i, _p in enumerate(PARTS):
    if not _p.label:
        _p.label = "第" + ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][_i] + "部"

CHAPTERS: dict[str, Chapter] = {c.slug: c for p in PARTS for c in p.chapters}
GLOSSARY_PART = Part("用語集", "本文に出てくるデータサイエンスの専門用語の解説。", [C(GL.GLOSSARY_SLUG, GL.GLOSSARY_BLURB, "用語")], label="付録C")
ALL_PARTS: list[Part] = PARTS + [GLOSSARY_PART]
KINDS = {"要件": "#f87171", "設計": "#7aa9ff", "仕様": "#34d399", "運用": "#38bdf8", "参照": "#fb923c", "結果": "#fbbf24", "ログ": "#c084fc", "メモ": "#9aa7c7", "決定": "#a3e635", "用語": "#f472b6"}
STATUS = {"legacy": ("旧設計", "#fb923c"), "snapshot": ("時点の記録", "#7aa9ff"), "history": ("決定の履歴", "#a3e635")}


def all_source_files() -> set[Path]:
    """本に載せるべき HTML（docs/ 配下の全 HTML。本自身を除く）。"""
    out = {p for p in DOCS_DIR.rglob("*.html") if p.name != BOOK_NAME and "node_modules" not in p.parts}
    return {p.resolve() for p in out}


# ── 読み込みと整形 ─────────────────────────────────────────────────────────────

_TAG = re.compile(r"<[^<>]*>")


def fix_curly_quotes(s: str) -> str:
    """タグの中の全角の引用符（class=”x” など）を半角に直す（旧ページの生成時の不具合）。"""
    return _TAG.sub(lambda m: m.group(0).replace("“", '"').replace("”", '"'), s)


def read_chapter_source(path: Path) -> str:
    """章の元ファイルを HTML として返す。運用メモなどの Markdown（.md）は、表とコードブロックを HTML にして <article> で包む。"""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".md":
        return text
    import markdown  # 本の生成時だけ使う（requirements.txt）

    return f'<article class="html-body">{markdown.markdown(text, extensions=["tables", "fenced_code"])}</article>'


def extract(raw: str) -> tuple[str, str, str]:
    """(タイトル, CSS, 本文) を返す。本文は h1・サイドバー・前後リンク・フッター・script を除いたもの。"""
    raw = fix_curly_quotes(raw)
    css = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", raw, re.S))
    body = raw
    m = re.search(r"<article\b[^>]*>(.*?)</article>", raw, re.S) or re.search(r"<main\b[^>]*>(.*?)</main>", raw, re.S) \
        or re.search(r"<body\b[^>]*>(.*?)</body>", raw, re.S)
    if m:
        body = m.group(1)
    body = re.sub(r"<script\b.*?</script>", "", body, flags=re.S)
    body = re.sub(r"<style\b.*?</style>", "", body, flags=re.S)
    body = re.sub(r"<aside\b.*?</aside>", "", body, flags=re.S)
    body = re.sub(r"<nav class=\"doc-nav\">.*?</nav>", "", body, flags=re.S)
    body = re.sub(r"<footer\b.*?</footer>", "", body, flags=re.S)
    body = re.sub(r"<header class=\"hero\">.*?</header>", "", body, flags=re.S)
    h1 = re.search(r"<h1\b[^>]*>(.*?)</h1>", body, re.S)
    title = html.unescape(_TAG.sub("", h1.group(1))).strip() if h1 else ""
    if h1:
        body = body.replace(h1.group(0), "", 1)
    if not title:
        t = re.search(r"<title>(.*?)</title>", raw, re.S)
        title = html.unescape(re.sub(r"\s+—.*$", "", t.group(1))).strip() if t else ""
    return title, css, body.strip()


_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_REL = re.compile(r'(\s(?:href|src))="([^"#?]*)((?:[?#][^"]*)?)"')


def namespace(body: str, slug: str, slugs: set[str], src_path: str | None = None,
              path_to_slug: dict[str, str] | None = None, book_dir: str | None = None) -> str:
    """id・# リンク・url(#...) を章ごとに名前空間化し、他のページへの相対リンクを本の中の位置に張り替える。

    src_path（この章の元ファイル）と path_to_slug（元ファイルの絶対パス → 章の slug）を渡すと、ディレクトリをまたぐリンクも
    パスで解決する。本に載っていないファイル（コード・md など）へのリンクは book_dir から見た相対パスに直す。
    渡さない場合は、同じディレクトリのファイル名（slug）だけで解決する。"""
    seen: dict[str, int] = {}

    def new_id(m: re.Match) -> str:
        seen[m.group(2)] = seen.get(m.group(2), 0) + 1
        dup = f"__{seen[m.group(2)]}" if seen[m.group(2)] > 1 else ""      # 元のページ自身が同じ id を複数持つ場合は 2 つ目以降を別名にする
        return f'{m.group(1)}{slug}--{m.group(2)}{dup}{m.group(3)}'

    body = re.sub(r'(\sid=")([^"]+)(")', new_id, body)
    body = re.sub(r'(\s(?:xlink:)?href="#)([^"]+)(")', lambda m: f'{m.group(1)}{slug}--{m.group(2)}{m.group(3)}', body)
    body = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{slug}--{m.group(1)})", body)

    if src_path is None:
        body = re.sub(r'((?:href|src)=")\.\./(\.\./)', r"\1\2", body)

        def link(m: re.Match) -> str:
            pre, target, frag = m.group(1), m.group(2), m.group(3)
            stem = target[:-5]
            if stem not in slugs:
                return m.group(0)
            return f'{pre}"#{stem}--{frag[1:]}"' if frag else f'{pre}"#ch-{stem}"'

        return re.sub(r'(\shref=)"(?:\./)?([A-Za-z0-9_\-]+\.html)(#[^"]*)?"', link, body)

    base = os.path.dirname(src_path)

    def rel(m: re.Match) -> str:
        attr, target, tail = m.group(1), m.group(2), m.group(3)
        if not target or _SCHEME.match(target) or target.startswith("/"):
            return m.group(0)
        res = os.path.normpath(os.path.join(base, unquote(target)))
        frag = tail if tail.startswith("#") else ""
        to = (path_to_slug or {}).get(res)
        if to and attr.strip() == "href":
            return f'{attr}="#{to}--{frag[1:]}"' if frag else f'{attr}="#ch-{to}"'
        if book_dir is not None:
            return f'{attr}="{quote(os.path.relpath(res, book_dir))}{tail}"'
        return m.group(0)

    return _REL.sub(rel, body)


def sections(body: str, slug: str) -> list[tuple[str, str]]:
    """本文中の h2 から (id, 見出し) を返す。id が無い h2 には付ける（本文も書き換えて返せるよう別関数で付与）。"""
    out = []
    for m in re.finditer(r"<h2\b([^>]*)>(.*?)</h2>", body, re.S):
        idm = re.search(r'id="([^"]+)"', m.group(1))
        if idm:
            out.append((idm.group(1), html.unescape(_TAG.sub("", m.group(2))).strip()))
    return out


def ensure_h2_ids(body: str, slug: str) -> str:
    n = [0]

    def f(m: re.Match) -> str:
        n[0] += 1
        if re.search(r'\sid="', m.group(1)):
            return m.group(0)
        return f'<h2 id="{slug}--s{n[0]}"{m.group(1)}>{m.group(2)}</h2>'

    return re.sub(r"<h2\b([^>]*)>(.*?)</h2>", f, body, flags=re.S)


# ── CSS を章の中だけに効かせる ────────────────────────────────────────────────────

def _split_blocks(css: str) -> list[tuple[str, str]]:
    """トップレベルの (セレクタ or @規則の頭, 中身) を返す。"""
    out, i, n = [], 0, len(css)
    while i < n:
        j = css.find("{", i)
        if j < 0:
            break
        head = css[i:j].strip()
        depth, k = 1, j + 1
        while k < n and depth:
            depth += {"{": 1, "}": -1}.get(css[k], 0)
            k += 1
        out.append((head, css[j + 1:k - 1]))
        i = k
    return out


def scope_css(css: str, scope: str) -> str:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = []
    for head, inner in _split_blocks(css):
        if head.startswith("@media") or head.startswith("@supports"):
            out.append(f"{head}{{{scope_css(inner, scope)}}}")
        elif head.startswith("@"):
            out.append(f"{head}{{{inner}}}")
        else:
            sels = []
            for s in (x.strip() for x in head.split(",")):
                if s in (":root", "html", "body", "html,body", "*"):
                    sels.append(scope if s != "*" else f"{scope}, {scope} *")
                elif s.startswith(("body ", "html ")):
                    sels.append(scope + " " + s.split(" ", 1)[1])
                else:
                    sels.append(f"{scope} {s}")
            out.append(f"{', '.join(sels)}{{{inner}}}")
    return "\n".join(out)


# ── 更新日 ───────────────────────────────────────────────────────────────────

def last_updated(path: Path) -> str:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d", "--", str(path)], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
        if out:
            return out
    except OSError:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime, JST).strftime("%Y-%m-%d")


# ── 組み立て ─────────────────────────────────────────────────────────────────

SHELL_CSS = """
:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --code-bg:#0b1428; }
* { box-sizing: border-box; } html { scroll-behavior: auto; }
body.bk { margin:0; background:#0b1226; color:var(--text); font:15px/1.8 "Noto Sans JP",system-ui,sans-serif; }
.bk-layout { display:grid; grid-template-columns: 340px 1fr; min-height:100vh; }
.bk-side { position:sticky; top:0; height:100vh; overflow-y:auto; background:#0a1124; border-right:1px solid var(--border); padding:16px 12px 48px; font-size:13px; }
.bk-side h1 { font-size:16px; margin:0 0 2px; } .bk-side h1 a { color:var(--text); text-decoration:none; }
.bk-side .sub { color:var(--muted); font-size:12px; margin-bottom:8px; } .bk-side .sub a { color:var(--accent-2); text-decoration:none; }
.bk-side input { width:100%; padding:7px 10px; border-radius:6px; border:1px solid var(--border); background:#0f1a36; color:var(--text); margin-bottom:6px; }
.bk-side .tools { display:flex; gap:6px; margin-bottom:6px; } .bk-side .tools button { flex:1; background:#0f1a36; color:var(--muted); border:1px solid var(--border); border-radius:5px; font-size:11.5px; padding:3px 0; cursor:pointer; }
.bk-side .tools button:hover { color:var(--text); border-color:var(--accent); }
/* 部（いちばん上の階層）: 帯 */
.bk-side .part { margin:18px -4px 6px; padding:5px 10px; background:linear-gradient(90deg,#1b2d5c,#0f1a36); border-left:4px solid var(--accent); border-radius:0 6px 6px 0; color:#fff; font-weight:700; font-size:13px; letter-spacing:.02em; }
.bk-side .part small { font-weight:400; color:#b9c6e8; margin-left:6px; }
/* 章（2 階層目）: 番号つきの行。左に開閉ボタン */
.bk-side .chap { margin:1px 0; }
.bk-side .chap .row { display:flex; align-items:flex-start; gap:2px; }
.bk-side .chap .tg { flex:0 0 22px; height:24px; background:none; border:0; color:var(--muted); cursor:pointer; font-size:11px; padding:0; }
.bk-side .chap .tg:hover { color:#fff; } .bk-side .chap .tg::before { content:"▶"; display:inline-block; transition:transform .12s; } .bk-side .chap.open .tg::before { transform:rotate(90deg); }
.bk-side .chap .row > a { flex:1; display:flex; gap:7px; align-items:flex-start; color:var(--text); text-decoration:none; padding:3px 6px; border-radius:5px; font-weight:600; line-height:1.45; }
.bk-side .chap .row > a:hover { background:#16254a; }
.bk-side .chap .num { flex:0 0 24px; text-align:center; background:#1a2a52; color:var(--accent-2); border-radius:4px; font-size:11.5px; font-weight:700; line-height:20px; margin-top:1px; }
.bk-side .chap.cur > .row > a { background:#1c3566; color:#fff; } .bk-side .chap.cur .num { background:var(--accent); color:#fff; }
/* 節（3 階層目）: 字下げ + 縦線 + 小さく淡い色 */
.bk-side .secs { display:none; margin:2px 0 6px 31px; padding-left:8px; border-left:2px solid #2b3d6e; }
.bk-side .chap.open .secs { display:block; }
.bk-side .secs a { display:block; color:#9fb0d6; text-decoration:none; font-size:12px; font-weight:400; line-height:1.4; padding:2px 6px; border-radius:4px; }
.bk-side .secs a:hover { background:#16254a; color:#fff; } .bk-side .secs a.cur { color:#fff; background:#223a6e; box-shadow:inset 3px 0 0 var(--accent-2); }
.bk-main { min-width:0; padding:0 0 120px; }
.bk-front, .bk-chapter { max-width:1040px; margin:0 auto; padding:30px 40px 10px; }
.bk-front h1 { font-size:30px; margin:20px 0 6px; } .bk-front h2 { font-size:19px; margin:30px 0 8px; border-top:1px solid var(--border); padding-top:14px; }
.bk-front table, .bk-chapter table.bk-map { border-collapse:collapse; width:100%; font-size:13.5px; margin:10px 0; }
.bk-front th, .bk-front td { border:1px solid var(--border); padding:6px 10px; text-align:left; vertical-align:top; } .bk-front th { background:var(--panel); color:var(--accent-2); }
.bk-front a, .bk-head a, .bk-pn a { color:var(--accent-2); }
.bk-front td:nth-child(1), .bk-front td:nth-child(3), .bk-front td:nth-child(5) { white-space:nowrap; }
.bk-head { border-top:3px solid var(--accent); margin-top:36px; padding-top:14px; } .bk-head .no { color:var(--muted); font-size:13px; letter-spacing:.06em; }
.bk-head .partname { color:var(--accent-2); font-size:12.5px; margin-bottom:2px; }
.bk-head h2.bk-title { font-size:26px; margin:2px 0 8px; border:0; padding:0; } .bk-head .meta { color:var(--muted); font-size:12.5px; margin:0 0 6px; }
.bk-head .blurb { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:10px 14px; margin:8px 0 6px; font-size:14px; }
.bk-tag { display:inline-block; font-size:11.5px; border:1px solid currentColor; border-radius:4px; padding:0 6px; margin-right:6px; }
.bk-see { font-size:13px; color:var(--muted); margin:4px 0 0; } .bk-see a { color:var(--accent-2); }
.bk-pn { display:flex; justify-content:space-between; gap:12px; margin:28px 0 0; padding-top:12px; border-top:1px solid var(--border); font-size:13.5px; }
.bk-pn a { text-decoration:none; max-width:48%; }
.bk-body { padding-top:6px; } .bk-body img, .bk-body svg { max-width:100%; height:auto; }
.bk-chapter { background:transparent !important; max-width:1040px !important; margin:0 auto !important; padding:30px 40px 10px !important; }
.bk-body table { max-width:100%; } .bk-body pre { max-width:100%; overflow-x:auto; }
/* 用語の解説（本文の用語ボタン・その場に開くカード・用語集の章） */
.bk-main button.gl-term { font:inherit; color:inherit; background:none; border:0; border-bottom:1.5px dotted #7aa9ff; padding:0; margin:0; cursor:help; }
.bk-main button.gl-term:hover, .bk-main button.gl-term.open { background:#1c3566; color:#fff; border-radius:3px; }
.bk-main aside.gl-card { background:#10234a; border:1px solid #3a63b8; border-left:4px solid var(--accent); border-radius:8px; padding:8px 14px 6px; margin:8px 0 14px; font-size:14px; line-height:1.75; color:var(--text); text-align:left; }
.gl-card .gl-ch { display:flex; justify-content:space-between; align-items:center; color:var(--accent-2); font-size:13px; }
.gl-card .gl-x { background:none; border:0; color:var(--muted); font-size:20px; line-height:1; cursor:pointer; padding:0 4px; }
.gl-card .gl-x:hover { color:#fff; } .gl-card .gl-more { font-size:12.5px; margin:4px 0 2px; }
.bk-chapter .gl-one { font-weight:700; color:#fff; margin:4px 0 6px; }
.bk-chapter .gl-fig { background:#0b1428; border:1px solid var(--border); border-radius:6px; padding:8px; margin:8px 0; text-align:center; overflow-x:auto; }
.bk-chapter .gl-fig svg { max-width:100%; height:auto; }
.bk-chapter .gl-links { font-size:13px; color:var(--muted); margin:6px 0; } .bk-chapter .gl-links a { margin-right:12px; color:var(--accent-2); }
.bk-chapter details.gl-item { border:1px solid var(--border); border-radius:8px; background:var(--panel); margin:6px 0; }
.bk-chapter details.gl-item > summary { cursor:pointer; padding:8px 12px; list-style:none; display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }
.bk-chapter details.gl-item > summary::-webkit-details-marker { display:none; }
.bk-chapter details.gl-item > summary::before { content:"▶"; color:var(--muted); font-size:11px; } .bk-chapter details.gl-item[open] > summary::before { content:"▼"; }
.bk-chapter .gl-short { color:var(--muted); font-size:13px; } .bk-chapter details.gl-item .gl-body { padding:2px 14px 10px; }
.bk-chapter button.gl-all { background:#0f1a36; color:var(--muted); border:1px solid var(--border); border-radius:5px; font-size:12.5px; padding:3px 10px; cursor:pointer; }
.bk-chapter input.gl-q { padding:5px 10px; border-radius:6px; border:1px solid var(--border); background:#0f1a36; color:var(--text); margin-left:8px; width:220px; }
.bk-status { --c:#7aa9ff; border:1px solid var(--c); border-left-width:5px; background:#14213f; border-radius:6px; padding:8px 12px; margin:8px 0 6px; font-size:13.5px; line-height:1.7; }
.bk-status b { color:var(--c); margin-right:6px; } .bk-status a { color:var(--accent-2); }
.bk-front td.lead { background:#101c38; color:var(--muted); font-size:12.5px; } .bk-front tr.partrow td { background:#16254a; color:#fff; font-weight:700; }
@media (max-width: 900px) { .bk-layout { grid-template-columns: 1fr; } .bk-side { position:relative; height:auto; } }
"""

SHELL_JS = """
(function(){
  const side = document.querySelector('.bk-side');
  const chaps = [...side.querySelectorAll('.chap')];
  const input = document.getElementById('bk-q');
  const OFFSET = 12;
  if ('scrollRestoration' in history) history.scrollRestoration = 'manual';   // 戻る/進むの位置はこちらで管理する

  /* 移動: 目的の要素の位置へ即座に移動する（ページが長く、図の描画で高さが変わるため、滑らかなスクロールにはしない） */
  function jump(id, push) {
    const el = document.getElementById(id);
    if (!el) return;
    const dt = el.tagName === 'DETAILS' ? el : el.closest('details'); if (dt) dt.open = true;   // 用語集の項目は開いてから移動
    const y0 = window.scrollY;
    const ch0 = el.closest('.bk-chapter'); if (ch0) renderIn(ch0).then(() => { if (location.hash === '#' + id || !push) window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - OFFSET); });
    window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - OFFSET);
    if (push) { history.replaceState({y: y0}, '', location.href); history.pushState({y: window.scrollY}, '', '#' + id); }
    markCurrent(id);
  }
  side.addEventListener('click', e => {
    const tg = e.target.closest('.tg');
    if (tg) { tg.closest('.chap').classList.toggle('open'); return; }
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    e.preventDefault();
    jump(a.getAttribute('href').slice(1), true);
  });
  document.querySelector('.bk-main').addEventListener('click', e => {
    const a = e.target.closest('a[href^="#"]');
    if (!a || a.getAttribute('href') === '#') return;
    const id = decodeURIComponent(a.getAttribute('href').slice(1));
    if (!document.getElementById(id)) return;
    e.preventDefault(); jump(id, true);
  });
  window.addEventListener('popstate', e => {
    if (e.state && typeof e.state.y === 'number') window.scrollTo(0, e.state.y);
    else if (location.hash) jump(decodeURIComponent(location.hash.slice(1)), false);
    else window.scrollTo(0, 0);
  });

  /* 用語の解説: 本文の用語ボタンを押すと、その段落の下に用語集の説明をその場で開く */
  document.querySelector('.bk-main').addEventListener('click', e => {
    const x = e.target.closest('.gl-x');
    if (x) { const c = x.closest('.gl-card'); const b = c && c._btn; if (b) b.classList.remove('open'); if (c) c.remove(); return; }
    const btn = e.target.closest('button.gl-term');
    if (!btn) return;
    if (btn._card && btn._card.isConnected) { btn._card.remove(); btn.classList.remove('open'); return; }
    const src = document.getElementById('gl-' + btn.dataset.gl);
    if (!src) return;
    const card = document.createElement('aside');
    card.className = 'gl-card'; card.dataset.gl = btn.dataset.gl; card._btn = btn;
    const body = src.querySelector('.gl-body').cloneNode(true);
    body.querySelectorAll('.gl-links').forEach(n => { if (!n.textContent.startsWith('詳しくは')) n.remove(); });
    card.innerHTML = '<div class="gl-ch"><b>' + src.querySelector('summary b').textContent + '</b><button type="button" class="gl-x" aria-label="閉じる">×</button></div>';
    card.append(...body.childNodes);
    const more = document.createElement('p'); more.className = 'gl-more';
    more.innerHTML = '<a href="#gl-' + btn.dataset.gl + '">用語集の項目を開く →</a>'; card.append(more);
    const host = btn.closest('table') || btn.closest('p, li, blockquote, dd, figure') || btn.parentElement;
    if (host.tagName === 'LI') host.appendChild(card); else host.after(card);
    btn._card = card; btn.classList.add('open');
  });
  document.querySelector('.bk-main').addEventListener('click', e => {
    const b = e.target.closest('button.gl-all'); if (!b) return;
    document.querySelectorAll('details.gl-item').forEach(d => { if (d.style.display !== 'none') d.open = b.dataset.open === '1'; });
  });
  document.querySelector('.bk-main').addEventListener('input', e => {
    if (!e.target.classList.contains('gl-q')) return;
    const q = e.target.value.trim().toLowerCase();
    document.querySelectorAll('details.gl-item').forEach(d => { d.style.display = !q || (d.dataset.name + ' ' + d.textContent).toLowerCase().includes(q) ? '' : 'none'; });
  });

  /* 現在位置の表示: サイドバーの章・節を強調し、その章だけ開く */
  function revealInSide(a) {
    const r = a.getBoundingClientRect(), s = side.getBoundingClientRect();
    if (r.top < s.top + 80) side.scrollTop -= (s.top + 80 - r.top);
    else if (r.bottom > s.bottom - 24) side.scrollTop += (r.bottom - s.bottom + 24);
  }
  function markCurrent(id) {
    const el = document.getElementById(id); if (!el) return;
    const ch = el.closest('.bk-chapter');
    side.querySelectorAll('a.cur').forEach(a => a.classList.remove('cur'));
    chaps.forEach(c => {
      const mine = ch && c.dataset.slug === ch.dataset.slug;
      c.classList.toggle('cur', !!mine);
      if (!input.value && !c.dataset.pinned) c.classList.toggle('open', !!mine);
    });
    const a = side.querySelector('a[href="#' + CSS.escape(id) + '"]') || (ch && side.querySelector('a[href="#ch-' + ch.dataset.slug + '"]'));
    if (a) { a.classList.add('cur'); revealInSide(a); }
  }
  const targets = [...document.querySelectorAll('.bk-chapter, .bk-body h2')].filter(t => t.id);
  let ticking = false;
  function spy() {
    ticking = false;
    let cur = null;
    for (const t of targets) { if (t.getBoundingClientRect().top <= 90) cur = t; else break; }
    if (cur) markCurrent(cur.id);
  }
  window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(spy); } }, {passive: true});

  /* しぼり込み・全部開く・全部閉じる */
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    chaps.forEach(c => {
      const hit = !q || c.textContent.toLowerCase().includes(q);
      c.style.display = hit ? '' : 'none';
      if (q) c.classList.toggle('open', hit);
    });
    side.querySelectorAll('.part').forEach(p => {
      let n = p.nextElementSibling, any = false;
      while (n && !n.classList.contains('part')) { if (n.style.display !== 'none') any = true; n = n.nextElementSibling; }
      p.style.display = any ? '' : 'none';
    });
  });
  document.getElementById('bk-open').addEventListener('click', () => chaps.forEach(c => { c.classList.add('open'); c.dataset.pinned = '1'; }));
  document.getElementById('bk-close').addEventListener('click', () => chaps.forEach(c => { c.classList.remove('open'); delete c.dataset.pinned; }));

  /* 読み込み直後・図の描画後に、URL の # の位置へ合わせ直す（図の描画でページの高さが変わるため） */
  function settle() { if (location.hash) jump(decodeURIComponent(location.hash.slice(1)), false); else spy(); }
  window.addEventListener('load', settle);
  /* 図（mermaid）は、その章が見える位置に来たときに描く（全部を最初に描くと重いため） */
  function renderIn(root) {
    if (!window.mermaid) return Promise.resolve();
    const nodes = [...root.querySelectorAll('.mermaid')].filter(n => !n.dataset.processed);
    return nodes.length ? mermaid.run({nodes}).catch(() => {}) : Promise.resolve();
  }
  if (window.mermaid) {
    mermaid.initialize({startOnLoad: false, theme: 'dark'});
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.unobserve(e.target); renderIn(e.target); } }), {rootMargin: '1500px 0px'});
    document.querySelectorAll('.bk-chapter').forEach(c => io.observe(c));
  }
})();
"""


def extract_scripts(raw: str) -> str:
    """元ページのインライン script（src の無いもの）。画面操作用のものだけを残す章で使う。"""
    return "\n".join(s for s in re.findall(r"<script\b(?![^>]*\bsrc=)[^>]*>(.*?)</script>", raw, re.S) if s.strip())


def derive_blurb(body: str, skip_meta: bool = False) -> str:
    """元ページの最初の段落から、章の説明（130 字まで）を作る。"""
    for m in re.finditer(r"<p\b[^>]*>(.*?)</p>", body, re.S):
        t = re.sub(r"\s+", " ", html.unescape(_TAG.sub("", m.group(1)))).strip()
        if len(t) < 30 or (skip_meta and re.search(r"日付:|最終更新:|担当:", t[:80])):
            continue
        return t[:130] + ("…" if len(t) > 130 else "")
    return ""


def build(out: Path = DEFAULT_OUT) -> tuple[str, dict]:
    book_dir = str(out.parent)
    chapters = [c for p in PARTS for c in p.chapters]
    slugs = set(CHAPTERS)
    path_to_slug = {os.path.normpath(str(c.path)): c.slug for c in chapters}
    built: dict[str, dict] = {}
    css_groups: dict[str, str] = {}
    used: dict[str, set[str]] = {}
    for ch in chapters:
        raw = read_chapter_source(ch.path)
        title, css, body = extract(raw)
        blurb = ch.blurb or derive_blurb(body, skip_meta=(ch.kind == "決定"))
        script = extract_scripts(fix_curly_quotes(raw)) if ch.keep_scripts else ""
        ids = re.findall(r'\sid="([^"]+)"', body)
        body = ensure_h2_ids(body, ch.slug)
        body = namespace(body, ch.slug, slugs, str(ch.path), path_to_slug, book_dir)
        body = GL.annotate(body, used, ch.slug)
        if script:
            for i in sorted(ids, key=len, reverse=True):
                script = re.sub(r"(['\"])" + re.escape(i) + r"\1", lambda m, i=i: f"{m.group(1)}{ch.slug}--{i}{m.group(1)}", script)
            body += f"<script>{script}</script>"
        gid = hashlib.md5(re.sub(r"\s+", "", css).encode()).hexdigest()[:6]
        css_groups.setdefault(gid, css)
        built[ch.slug] = {"title": title or ch.slug, "body": body, "gid": gid, "secs": sections(body, ch.slug), "updated": last_updated(ch.path), "blurb": blurb}

    # 用語集（付録）。本文で押せるようにした用語の説明と、それが使われている章を載せる
    label, n = {}, 0
    for part in PARTS:
        for ch in part.chapters:
            n += 1
            label[ch.slug] = f"第{n}章 {built[ch.slug]['title']}"
    gl_body, gl_secs = GL.render_chapter(used, label)
    allb = {**built, GL.GLOSSARY_SLUG: {"title": GL.GLOSSARY_TITLE, "body": gl_body, "gid": "gl", "secs": gl_secs, "updated": datetime.now(JST).strftime("%Y-%m-%d"), "blurb": GL.GLOSSARY_BLURB}}
    parts_all = ALL_PARTS
    order = [c for p in parts_all for c in p.chapters]
    num = {c.slug: i + 1 for i, c in enumerate(order)}
    part_of = {c.slug: f"{p.label}　{p.title}" for p in parts_all for c in p.chapters}

    # 目次（部 → 章 → 節 の 3 階層）
    side = []
    for part in parts_all:
        first = num[part.chapters[0].slug]
        span = f"第{first}章" if len(part.chapters) == 1 else f"第{first}〜{first + len(part.chapters) - 1}章"
        side.append(f'<div class="part">{html.escape(part.label)}　{html.escape(part.title)}<small>{span}</small></div>')
        for ch in part.chapters:
            b = allb[ch.slug]
            secs = "".join(f'<a href="#{html.escape(i)}" title="{html.escape(t)}">{html.escape(t[:46])}{"…" if len(t) > 46 else ""}</a>' for i, t in b["secs"])
            mark = f' <span class="mk" style="color:{STATUS[ch.status][1]}" title="{STATUS[ch.status][0]}">●</span>' if ch.status in ("legacy",) else ""
            side.append(f'<div class="chap" data-slug="{ch.slug}"><div class="row"><button class="tg" type="button" aria-label="節を開閉"></button>'
                        f'<a href="#ch-{ch.slug}"><span class="num">{num[ch.slug]}</span><span>{html.escape(b["title"])}{mark}</span></a></div><div class="secs">{secs}</div></div>')

    # 前付け: 全体の地図
    rows = []
    for part in parts_all:
        rows.append(f'<tr class="partrow"><td colspan="5">{html.escape(part.label)}　{html.escape(part.title)}</td></tr><tr><td class="lead" colspan="5">{html.escape(part.lead)}</td></tr>')
        for ch in part.chapters:
            b = allb[ch.slug]
            st = f' <span class="bk-tag" style="color:{STATUS[ch.status][1]}">{STATUS[ch.status][0]}</span>' if ch.status in ("legacy", "snapshot") else ""
            rows.append(f'<tr><td>第{num[ch.slug]}章</td><td><a href="#ch-{ch.slug}">{html.escape(b["title"])}</a>{st}</td>'
                        f'<td><span class="bk-tag" style="color:{KINDS[ch.kind]}">{ch.kind}</span></td><td>{html.escape(b["blurb"])}</td><td>{b["updated"]}</td></tr>')

    def link(slug: str) -> str:
        return f'<a href="#ch-{slug}">{html.escape(allb[slug]["title"])}</a>'

    front = (
        '<section class="bk-front" id="front"><h1>keiba-vpn ドキュメント総覧</h1>'
        f'<p class="meta" style="color:#9aa7c7">docs/ 配下の {len(CHAPTERS)} 本の HTML（docs/html/・docs/decisions/・docs/operations/・docs/other_predictor/）を、1 冊の本にまとめたもの。生成: {datetime.now(JST):%Y-%m-%d %H:%M} JST。'
        "元の各ページはそのまま残っている（各章の見出しの「単独ページ」から開ける）。この本は元のページから自動で作るので、<b>直すときは元のページを直して</b> "
        "<code>python -m src.scripts.docs.build_docs_book</code> で作り直す。</p>"
        "<h2>どれを読めばよいか</h2><ul>"
        f'<li>プロダクトの目的・要件・現状 → {link("requirements")}</li>'
        f'<li>いまの精度・最新の結論 → {link("modeling_experiment_report")}</li>'
        f'<li>学習で何が確定したか → {link("training_pipeline_decisions")}</li>'
        f'<li>データがいつ集まり、どこで特徴量になるか → {link("data_flow_schedule")}</li>'
        f'<li>どのデータがどこにあるか → {link("database_schemas")}</li>'
        f'<li>特徴量の定義と設計の根拠 → {link("feature_strategy_tp03_tp04")}</li>'
        f'<li>馬券の確率・期待値 → {link("exotic_bet_probabilities")}</li>'
        f'<li>cron・運用 → {link("cron_schedule")}</li>'
        f'<li>なぜそう決めたか → {link("MASTER")}（決定記録は付録A）</li>'
        f'<li>専門用語がわからない → 本文の<button type="button" class="gl-term" data-gl="logloss">点線の下線がついた語</button>をクリックすると、その場に説明が開く。まとめて読むなら <a href="#ch-{GL.GLOSSARY_SLUG}">用語集</a>（{len(GL.TERMS)} 語・図つき）</li></ul>'
        "<h2>章の種類と状態</h2><p>" + "".join(f'<span class="bk-tag" style="color:{c}">{k}</span>' for k, c in KINDS.items()) +
        "　要件 = 要件定義 ／ 設計 = 方針や仕組みの設計 ／ 仕様 = 列・手順などの取り決め ／ 運用 = cron・サーバーなどの運用 ／ 参照 = 一覧・カタログ ／ 結果 = 実験の結果 ／ ログ = 確定事項の記録 ／ メモ = 調査・探索のメモ ／ 決定 = 決定記録。</p>"
        f'<p><span class="bk-tag" style="color:{STATUS["legacy"][1]}">旧設計</span> = 一部が現行と異なる（各章の冒頭に、異なる点と現行の章へのリンクがある）。'
        f'<span class="bk-tag" style="color:{STATUS["snapshot"][1]}">時点の記録</span> = ある時点の設計・調査メモで、その後に変わった可能性がある。'
        f'<span class="bk-tag" style="color:{STATUS["history"][1]}">決定の履歴</span> = 当時の判断の記録（書き換えない）。「更新日」は元のページの最終コミット日。</p>'
        '<h2>全体の地図</h2><table><tr><th>章</th><th>タイトル</th><th>種類</th><th>何について書かれているか</th><th>更新日</th></tr>' + "".join(rows) + "</table></section>"
    )

    # 章
    chapters_html = []
    for idx, ch in enumerate(order):
        b = allb[ch.slug]
        see = "".join(f'<a href="#ch-{s}">第{num[s]}章 {html.escape(allb[s]["title"])}</a>（{html.escape(why)}）　' for s, why in ch.see if s in allb)
        now = "".join(f'<a href="#ch-{s}">第{num[s]}章 {html.escape(allb[s]["title"])}</a>（{html.escape(why)}）　' for s, why in ch.now if s in allb)
        status = ""
        if ch.status in STATUS:
            name, color = STATUS[ch.status]
            msg = ch.note or ("当時の判断の記録。書き換えず、古くなったものは注記を足す。" if ch.status == "history" else "")
            status = f'<div class="bk-status" style="--c:{color}"><b>{name}</b>{html.escape(msg)}' + (f'　現行: {now}' if now else "") + "</div>"
        prev_a = f'<a href="#ch-{order[idx - 1].slug}">← 第{idx}章 {html.escape(allb[order[idx - 1].slug]["title"])}</a>' if idx else "<span></span>"
        next_a = f'<a href="#ch-{order[idx + 1].slug}">第{idx + 2}章 {html.escape(allb[order[idx + 1].slug]["title"])} →</a>' if idx + 1 < len(order) else "<span></span>"
        if ch.slug in built:
            origin = os.path.relpath(ch.path, book_dir)
            meta = f'<p class="meta">更新日 {b["updated"]} ／ 元のファイル <a href="{quote(origin)}">{html.escape(origin)}</a>（単独ページ）</p>'
        else:
            meta = f'<p class="meta">生成日 {b["updated"]} ／ 定義: src/scripts/docs/modeling_glossary.py（本の生成時に作られる。単独ページは無い）</p>'
        chapters_html.append(
            f'<section class="bk-chapter css-{b["gid"]}" id="ch-{ch.slug}" data-slug="{ch.slug}">'
            f'<div class="bk-head"><div class="partname">{html.escape(part_of[ch.slug])}</div><div class="no">第{num[ch.slug]}章　<span class="bk-tag" style="color:{KINDS[ch.kind]}">{ch.kind}</span></div>'
            f'<h2 class="bk-title">{html.escape(b["title"])}</h2>' + meta + status +
            f'<div class="blurb">{html.escape(b["blurb"])}</div>' + (f'<p class="bk-see">関連する章: {see}</p>' if see else "") + "</div>"
            f'<div class="bk-body html-body">{b["body"]}</div><div class="bk-pn">{prev_a}{next_a}</div></section>'
        )

    css = SHELL_CSS + "\n" + "\n".join(f"/* {gid} */\n" + scope_css(c, f".css-{gid}") for gid, c in css_groups.items()) + "\n.bk-chapter{background:transparent !important}"
    mermaid_src = quote(os.path.relpath(HTML_DIR / "assets" / "mermaid.min.js", book_dir))
    out_html = (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>keiba-vpn ドキュメント総覧</title>"
        f"<style>{css}</style></head><body class=\"bk\"><div class=\"bk-layout\"><nav class=\"bk-side\"><h1><a href=\"#front\">ドキュメント総覧</a></h1>"
        '<div class="sub">docs/ の全 HTML を 1 冊に統合</div><input id="bk-q" type="search" placeholder="章・節をしぼり込む"><div class="tools"><button id="bk-open" type="button">すべて開く</button><button id="bk-close" type="button">すべて閉じる</button></div>'
        + "".join(side) + '</nav><main class="bk-main">' + front + "".join(chapters_html) +
        f'</main></div><script src="{mermaid_src}"></script><script>' + SHELL_JS + "</script></body></html>"
    )
    return out_html, built


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out_html, built = build(args.out)
    args.out.write_text(out_html, encoding="utf-8")
    print(f"{args.out}: {len(built)} 章, {len(out_html) // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
