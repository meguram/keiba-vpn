"""
Jupyter / 単発スクリプト向け: リポジトリルートの .env を確実に読み込む。

カレントディレクトリが notebooks/feature_engineering/ などの配下でも、ルートの .env を探して load_dotenv する。

環境別オーバーレイ:
  KEIBA_ENV=stg  → .env を読んだ後 .env.stg を上書きマージ（存在する場合）
  KEIBA_ENV=prod → .env を読んだ後 .env.prod を上書きマージ（存在する場合）

これにより stg/prod それぞれの DATABASE_URL 等を独立して管理できる。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def find_project_root(start: Path | None = None) -> Path:
    """`.env.example` と `requirements.txt` があるディレクトリを keiba-vpn ルートとみなす。"""
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".env.example").is_file() and (parent / "requirements.txt").is_file():
            return parent
    raise FileNotFoundError(
        "keiba-vpn プロジェクトルートが見つかりません。"
        "カレントディレクトリがリポジトリ内か、.env.example / requirements.txt が存在するか確認してください。"
    )


def load_project_dotenv(
    *,
    dotenv_path: Path | str | None = None,
    override: bool = False,
) -> Path | None:
    """
    ルートの `.env` を読み込み、環境別オーバーレイ (.env.stg / .env.prod) を上書きマージする。

    読み込み順序:
      1. `.env`（共通設定・シークレット）
      2. `.env.<KEIBA_ENV>`（stg/prod の DB URL 等を上書き）

    - `dotenv_path` 省略時: 環境変数 `KEIBA_DOTENV_PATH` があればそれを使用、なければ `<ルート>/.env`
    - ファイルが無い場合は読み込まず `None` を返す
    """
    if dotenv_path is not None:
        path = Path(dotenv_path).expanduser().resolve()
    else:
        raw = (os.environ.get("KEIBA_DOTENV_PATH") or "").strip()
        if raw:
            path = Path(raw).expanduser().resolve()
        else:
            path = find_project_root() / ".env"

    loaded: Path | None = None
    if path.is_file():
        load_dotenv(path, override=override)
        loaded = path

    # 環境別オーバーレイ: .env.stg / .env.prod を上書きマージ
    env_name = (os.environ.get("KEIBA_ENV") or "").strip().lower()
    if env_name in ("stg", "prod"):
        root = path.parent if path.is_file() else find_project_root()
        overlay = root / f".env.{env_name}"
        if overlay.is_file():
            load_dotenv(overlay, override=True)  # 常に override=True で上書き
            loaded = overlay

    return loaded
