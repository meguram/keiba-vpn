"""最終オッズ学習のプログレス表示・ステータスファイル。"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

STATUS_PATH = Path("data/local/meta/final_odds_train_status.json")


@dataclass
class FinalOddsProgress:
    """stderr バー + JSON ステータス（API/別ターミナルから参照可）。"""

    enabled: bool = True
    phase: str = "init"
    message: str = ""
    current: int = 0
    total: int = 0
    started_at: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)
    _last_draw: float = 0.0
    _min_interval_sec: float = 0.25

    def start(self, *, train_years: list[str], eval_years: list[str]) -> None:
        self.phase = "init"
        self.message = "開始"
        self.current = 0
        self.total = 0
        self.started_at = time.time()
        self.extra = {
            "train_years": train_years,
            "eval_years": eval_years,
        }
        self._persist(running=True)
        self._draw(force=True)

    def set_phase(self, phase: str, total: int = 0, message: str = "") -> None:
        self.phase = phase
        self.total = max(0, int(total))
        self.current = 0
        if message:
            self.message = message
        self._persist(running=True)
        self._draw(force=True)

    def update(
        self,
        current: int | None = None,
        *,
        delta: int = 0,
        message: str | None = None,
        total: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if current is not None:
            self.current = int(current)
        else:
            self.current += int(delta)
        if message is not None:
            self.message = message
        if total is not None:
            self.total = int(total)
        if meta:
            self.extra.update(meta)
        self._persist(running=True)
        self._draw()

    def finish(self, *, result: dict | None = None, error: str | None = None) -> None:
        self.phase = "error" if error else "done"
        self.message = error or "完了"
        if self.total > 0:
            self.current = self.total
        self._persist(running=False, result=result, error=error)
        self._draw(force=True)
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    def _persist(
        self,
        *,
        running: bool,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "running": running,
            "phase": self.phase,
            "message": self.message,
            "current": self.current,
            "total": self.total,
            "pct": round(100.0 * self.current / self.total, 1) if self.total else 0.0,
            "elapsed_sec": round(self.elapsed_sec(), 1),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "extra": dict(self.extra),
        }
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error
        try:
            STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATUS_PATH.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _draw(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and (now - self._last_draw) < self._min_interval_sec:
            return
        self._last_draw = now

        if self.total > 0:
            pct = min(100, int(100 * self.current / self.total))
            width = 32
            filled = int(width * pct / 100)
            bar = "=" * filled + "-" * (width - filled)
            line = (
                f"\r[final_odds] {self.phase} [{bar}] {pct:3d}% "
                f"({self.current}/{self.total}) {self.message} "
                f"{self.elapsed_sec():.0f}s"
            )
        else:
            line = (
                f"\r[final_odds] {self.phase} … {self.message} "
                f"({self.current}) {self.elapsed_sec():.0f}s"
            )
        sys.stderr.write(line[:200].ljust(200))
        sys.stderr.flush()


def load_status_file() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {"running": False, "phase": "idle"}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"running": False, "phase": "idle", "error": "status read failed"}
