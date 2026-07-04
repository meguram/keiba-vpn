"""
Jupyter / 単発スクリプト向け: リポジトリルートの .env を確実に読み込む。

カレントディレクトリが notebooks/feature_engineering/ などの配下でも、ルートの .env を探して load_dotenv する。
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
    ルートの `.env` を `os.environ` に読み込む。

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
    if not path.is_file():
        return None
    load_dotenv(path, override=override)
    return path
