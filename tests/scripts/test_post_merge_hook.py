"""scripts/dev/git-hooks/post-merge: マージした作業ツリーの frontend/.next* だけを消す（共有の作業ツリーは消さない）。

2026-11-16 まではフックの場所（共有の .git/hooks）から ROOT を作っていたため、worktree でマージしても
共有の作業ツリーの .next-dev などを消しに行っていた。
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "scripts" / "dev" / "git-hooks" / "post-merge"
INSTALL = REPO / "scripts" / "dev" / "install_git_hooks.sh"

pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None, reason="git / bash が無い環境")


def _env() -> dict[str, str]:
    """一時リポジトリ用の環境。外の GIT_*（フックの中から pytest を動かしたときなど）と、利用者のグローバル設定
    （core.hooksPath・commit.gpgsign など）を切り離す。本物の .git/hooks を書き換えないため。"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
                "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
    return env


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, env=_env(), check=True, capture_output=True, text=True).stdout


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    (main / "frontend").mkdir(parents=True)
    _git(main, "init", "-q", "-b", "develop")
    (main / "frontend" / "a.ts").write_text("1\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "init")
    # リポジトリ管理のフックを、入れるスクリプトで共有の .git/hooks に入れる
    (main / "scripts" / "dev" / "git-hooks").mkdir(parents=True)
    shutil.copy2(HOOK, main / "scripts" / "dev" / "git-hooks" / "post-merge")
    shutil.copy2(INSTALL, main / "scripts" / "dev" / "install_git_hooks.sh")
    subprocess.run(["bash", "scripts/dev/install_git_hooks.sh"], cwd=main, env=_env(), check=True, capture_output=True)
    assert (main / ".git" / "hooks" / "post-merge").exists()

    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", "-b", "side", str(wt))
    _git(main, "checkout", "-q", "-b", "feature")
    (main / "frontend" / "a.ts").write_text("2\n")
    _git(main, "commit", "-q", "-am", "change frontend")
    _git(main, "checkout", "-q", "develop")
    for d in (main, wt):
        (d / "frontend" / ".next-dev").mkdir()
        (d / "frontend" / ".next-dev" / "x").write_text("cache")
    return main, wt


def test_merge_in_worktree_clears_only_that_worktree(tmp_path: Path) -> None:
    main, wt = _setup(tmp_path)
    _git(wt, "merge", "-q", "--no-ff", "-m", "merge", "feature")
    assert not (wt / "frontend" / ".next-dev").exists(), "マージした worktree のキャッシュは消す"
    assert (main / "frontend" / ".next-dev" / "x").exists(), "共有の作業ツリーのキャッシュは消さない"


def test_merge_in_main_clears_main(tmp_path: Path) -> None:
    main, wt = _setup(tmp_path)
    _git(main, "merge", "-q", "--no-ff", "-m", "merge", "feature")
    assert not (main / "frontend" / ".next-dev").exists()
    assert (wt / "frontend" / ".next-dev" / "x").exists()
