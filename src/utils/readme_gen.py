"""
README.md 自動生成ユーティリティ

Orchestrator の実際の階層構造を動的にイントロスペクトして
README.md を生成・更新する。

Orchestrator の初期化時と handle() 実行後に自動呼び出しされる。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator


def generate_readme(orch: Orchestrator, base_dir: str = ".") -> str:
    """Orchestrator の実際の状態からREADME.mdの内容を生成する。"""

    hierarchy = _build_hierarchy(orch)
    data_summary = _build_data_summary(base_dir)
    commands_table = _build_commands_table()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""# ML-AutoPilot Keiba

マルチエージェント競馬AI予測システム

> 最終更新: {now}

## アーキテクチャ

将軍（Orchestrator）が4体の家老（中間管理層）を統率し、
各家老が配下のサブエージェントを連携させてタスクを遂行する階層型マルチエージェントシステム。

```
{hierarchy}
```

## エージェント一覧

{_build_agent_table(orch)}

## 対応コマンド

{commands_table}

## ディレクトリ構成

```
keiba/
├── agents/                 エージェント群
│   ├── base.py             BaseAgent 基底クラス
│   ├── manager_base.py     MiddleManager 家老基底クラス
│   ├── orchestrator.py     将軍 (Orchestrator)
│   ├── config_manager.py   軍師 (ConfigManager)
│   ├── pipeline_manager.py パイプライン奉行 (PipelineManager)
│   ├── data_manager.py     データ奉行 (DataManager)
│   ├── info_manager.py     目付 (InfoManager)
│   ├── requirements_agent.py   RDA  - 要件定義
│   ├── scraping_agent.py       SMA  - データ取得
│   ├── data_organization_agent.py DOA - データ整理
│   ├── feature_engineering_agent.py FEA - 特徴量生成
│   ├── model_training_agent.py MTEA - モデル学習
│   └── prediction_agent.py     POA  - 予測出力
├── scraper/                スクレイピングモジュール
│   ├── client.py           HTTP クライアント (認証対応)
│   ├── selectors.py        耐久セレクタ (SelectorChain)
│   ├── parsers.py          各ページパーサー
│   ├── storage.py          JSON ストレージ
│   └── run.py              CLI ランナー
├── messaging/              エージェント間通信
│   ├── bus.py              メッセージバス
│   └── status.py           ステータスボード
├── config/
│   └── settings.yaml       プロジェクト設定
├── data/
│   ├── mock/               モックデータ (CSV)
│   └── scraped/            スクレイピング済データ (JSON)
├── commander.py            対話 CLI
├── main.py                 エントリーポイント
├── README.md               概要 (自動生成)
└── ARCHITECTURE.md         詳細仕様書 (自動生成)
```

## データ状況

{data_summary}

## 使い方

```bash
# 対話モード
python main.py

# コマンドを1回実行
python main.py --command "予測して"

# Webダッシュボード
python main.py --server
```

## 詳細仕様書

エージェントの詳細な責務・入出力・通信パターン・API仕様は **[ARCHITECTURE.md](ARCHITECTURE.md)** を参照。

## 家老間の連携パターン

| パターン | フロー |
|---------|--------|
| 予測パイプライン | 将軍 → 軍師(MLP仕様) → パイプライン奉行(SMA→DOA→FEA→MTEA→POA) |
| 設定変更 | 将軍 → 軍師(要件蓄積) → 次の「予測して」で自動反映 |
| データ取得 | 将軍 → データ奉行 → ScraperRunner |
| 情報照会 | 将軍 → 目付 → 軍師/パイプライン奉行/データ奉行から収集 |
| 不明コマンド | 将軍 → 選択肢を提示 (予測を自動実行しない) |

---
*このファイルはシステム起動時・コマンド実行時に自動更新されます。*
"""


def _build_hierarchy(orch: Orchestrator) -> str:
    """Orchestrator の実際のインスタンスから階層図を構築する。"""
    lines = ["将軍 (Orchestrator) ← ユーザーの命令"]

    managers = [
        (orch._config_mgr, "設定・要件管理"),
        (orch._pipeline_mgr, "予測パイプライン統率"),
        (orch._data_mgr, "データ取得・確認"),
        (orch._info_mgr, "情報照会・報告"),
    ]

    for i, (mgr, desc) in enumerate(managers):
        is_last = i == len(managers) - 1
        prefix = "└──" if is_last else "├──"
        child_prefix = "    " if is_last else "│   "
        status = mgr.status.value
        lines.append(f"  {prefix} {mgr.role} ({mgr.name}) [{status}]  {desc}")

        sub_agents = list(mgr.sub_agents.items()) if hasattr(mgr, 'sub_agents') else []
        for j, (key, agent) in enumerate(sub_agents):
            sub_is_last = j == len(sub_agents) - 1
            sub_prefix = "└──" if sub_is_last else "├──"
            lines.append(f"  {child_prefix}  {sub_prefix} {key} ({agent.role})")

    return "\n".join(lines)


def _build_agent_table(orch: Orchestrator) -> str:
    """全エージェントの一覧テーブルを生成する。"""
    rows = [
        "| 階層 | 名前 | 役割 | 状態 | 説明 |",
        "|------|------|------|------|------|",
        f"| 将軍 | Orchestrator | 将軍 | {orch.status.value} | コマンド解析・家老への委譲 |",
    ]

    manager_descs = {
        "ConfigMgr": "要件蓄積・MLP仕様生成・設定管理",
        "PipelineMgr": "ML予測パイプラインの全6フェーズ統率",
        "DataMgr": "スクレイピング・データ確認・エクスポート",
        "InfoMgr": "他家老から情報収集・ユーザーへ報告",
    }
    agent_descs = {
        "RDA": "自然言語要件→MLP仕様の構造化",
        "SMA": "netkeiba.comからのデータ取得・監視",
        "DOA": "CSV/JSONデータの整理・結合・分割",
        "FEA": "特徴量生成・エンコーディング・パイプライン管理",
        "MTEA": "LightGBM + Optuna による学習・評価",
        "POA": "予測結果の生成・フォーマット出力",
    }

    managers = [orch._config_mgr, orch._pipeline_mgr, orch._data_mgr, orch._info_mgr]
    for mgr in managers:
        desc = manager_descs.get(mgr.name, "")
        rows.append(f"| 家老 | {mgr.name} | {mgr.role} | {mgr.status.value} | {desc} |")
        for key, agent in mgr.sub_agents.items():
            ad = agent_descs.get(key, "")
            rows.append(f"| 足軽 | {key} | {agent.role} | {agent.status.value} | {ad} |")

    return "\n".join(rows)


def _build_commands_table() -> str:
    """対応コマンドのテーブルを生成する。"""
    return """| カテゴリ | コマンド例 | 担当家老 | 説明 |
|---------|-----------|---------|------|
| 予測 | 「予測して」「再学習して」 | パイプライン奉行 | ML予測パイプライン全体を実行 |
| 設定変更 | 「ターゲットを1着に変更」 | 軍師 | 予測対象・特徴量・フィルタの変更 |
| 設定変更 | 「オッズを除外して」 | 軍師 | 特徴量の追加・除外 |
| 設定変更 | 「芝のみにして」 | 軍師 | フィルタ条件の設定 |
| 設定変更 | 「設定確認」 | 軍師 | 現在の設定を表示 |
| データ | 「データを見せて」 | データ奉行 | データ概要を表示 |
| データ | 「統計を見せて」 | データ奉行 | 統計情報を表示 |
| データ | 「カラム一覧」 | データ奉行 | 列の一覧を表示 |
| スクレイピング | 「レース結果を取得 {race_id}」 | データ奉行 | 過去レース結果を取得 |
| スクレイピング | 「出馬表を取得 {race_id}」 | データ奉行 | 出馬表を取得 |
| スクレイピング | 「馬情報を取得 {horse_id}」 | データ奉行 | 馬の詳細情報を取得 |
| スクレイピング | 「タイム指数を取得 {race_id}」 | データ奉行 | タイム指数を取得 [要ログイン] |
| 情報 | 「状態は？」 | 目付 | エージェント状態を表示 |
| 情報 | 「精度は？」 | 目付 | モデル評価結果を表示 |
| 情報 | 「特徴量重要度」 | 目付 | 重要度ランキングを表示 |
| 情報 | 「概要」 | 目付 | システム全体の概要 |
| システム | 「リセット」 | 軍師 | 蓄積要件をクリア |
| システム | 「エクスポート」 | データ奉行 | 予測結果をJSON出力 |
| システム | 「ヘルプ」 | 将軍 | コマンド一覧を表示 |"""


def _build_data_summary(base_dir: str) -> str:
    """data/ ディレクトリの現状を反映する。"""
    lines = []

    mock_dir = os.path.join(base_dir, "data", "mock")
    if os.path.isdir(mock_dir):
        csv_files = [f for f in sorted(os.listdir(mock_dir)) if f.endswith(".csv")]
        if csv_files:
            lines.append("### モックデータ")
            lines.append("")
            lines.append("| ファイル | 行数 |")
            lines.append("|---------|------|")
            for fname in csv_files:
                fpath = os.path.join(mock_dir, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        n = sum(1 for _ in f) - 1
                    lines.append(f"| {fname} | {n:,} |")
                except Exception:
                    lines.append(f"| {fname} | - |")
            lines.append("")

    scraped_dir = os.path.join(base_dir, "data", "scraped")
    if os.path.isdir(scraped_dir):
        subdirs = sorted(
            d for d in os.listdir(scraped_dir)
            if os.path.isdir(os.path.join(scraped_dir, d))
        )
        has_data = False
        for d in subdirs:
            count = len([
                f for f in os.listdir(os.path.join(scraped_dir, d))
                if f.endswith(".json")
            ])
            if count > 0:
                if not has_data:
                    lines.append("### スクレイピング済データ")
                    lines.append("")
                    lines.append("| カテゴリ | ファイル数 |")
                    lines.append("|---------|----------|")
                    has_data = True
                lines.append(f"| {d} | {count} |")
        if has_data:
            lines.append("")

    if not lines:
        lines.append("データはまだありません。")

    return "\n".join(lines)


def update_readme(orch: Orchestrator, base_dir: str = "."):
    """README.md を生成してファイルに書き出す。"""
    content = generate_readme(orch, base_dir)
    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
