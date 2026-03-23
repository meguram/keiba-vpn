"""
スクレイピングジョブキューシステム

複数のスクレイピングリクエストを管理し、同時実行を防ぐ。
ファイルベースのキュー + ロックファイルで排他制御。

ジョブは job_kind（race / horse / date）+ target_id + tasks[] で任意の取得単位を指定できる。
期間内の出走馬への馬タスクは API enqueue-period-horses が race_lists→各レースの出馬表/結果から馬IDを展開してからキューへ載せる。
期間内の JRA レースへのレースタスクは enqueue-period-races が race_lists を走査して race_id 単位で bulk_add_jobs する。
"""

import json
import logging
import os
import secrets
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore
    _HAS_FCNTL = False

logger = logging.getLogger(__name__)

# キューディレクトリ
QUEUE_DIR = Path(__file__).parent.parent / "data" / "queue"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

LOCK_FILE = QUEUE_DIR / ".scrape.lock"
QUEUE_FILE = QUEUE_DIR / "scrape_queue.json"
# 複数 Uvicorn ワーカーが同時に JSON を読み書きすると破損するため、ファイル単位で排他する
QUEUE_JSON_FLOCK = QUEUE_DIR / ".scrape_queue.flock"


@contextmanager
def _exclusive_queue_json_lock():
    """scrape_queue.json の load/save をプロセス間で直列化（Linux/WSL のみ有効）。"""
    if not _HAS_FCNTL:
        yield
        return
    QUEUE_JSON_FLOCK.parent.mkdir(parents=True, exist_ok=True)
    fh = open(QUEUE_JSON_FLOCK, "a+b")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            fh.close()
        except OSError:
            pass

# キュー優先度（大きいほど先）。適性3D用 5 世代血統は最上位。
PRIORITY_DEFAULT = 0
PRIORITY_URGENT_PEDIGREE_5GEN = 9_000_000


def _legacy_dedupe_key(job: dict) -> str:
    rid = job.get("race_id")
    if rid:
        return f"race:{rid}:race_all"
    return job.get("job_id") or ""


def _effective_dedupe_key(job: dict) -> str:
    dk = job.get("dedupe_key")
    if dk:
        return dk
    return _legacy_dedupe_key(job)


class ScrapeJobQueue:
    """スクレイピングジョブキューマネージャー"""

    def __init__(self):
        self.queue_file = QUEUE_FILE
        self.lock_file = LOCK_FILE

    def _lock_json_pid_dead_or_invalid(self) -> bool:
        """
        ロック JSON の pid が死んでいる・不正なら True（ファイルは呼び出し側で削除）。
        権限でシグナル不能なときは False（ロック有効のまま）。
        """
        try:
            raw = self.lock_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            pid = int(data.get("pid") or 0)
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            return True
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError:
            return False
        return False

    def is_locked(self) -> bool:
        """現在スクレイピング処理が実行中かチェック"""
        if not self.lock_file.exists():
            return False

        # ロックファイルのタイムスタンプをチェック（10分以上古い場合は無効化）
        try:
            mtime = self.lock_file.stat().st_mtime
            if time.time() - mtime > 600:  # 10分
                logger.warning("古いロックファイルを削除: %s", self.lock_file)
                self.lock_file.unlink()
                return False
        except Exception as e:
            logger.error("ロックファイルチェックエラー: %s", e)
            return False

        if self._lock_json_pid_dead_or_invalid():
            logger.warning(
                "ロック保持プロセスが不在のため削除: %s",
                self.lock_file,
            )
            try:
                self.lock_file.unlink()
            except OSError:
                pass
            return False

        return True

    def acquire_lock(self) -> bool:
        """ロックを取得（既にロックされている場合はFalse）"""
        if self.is_locked():
            return False

        try:
            with open(self.lock_file, "w") as f:
                f.write(json.dumps({
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                    "datetime": datetime.now().isoformat(),
                }))
            return True
        except Exception as e:
            logger.error("ロック取得エラー: %s", e)
            return False

    def release_lock(self):
        """ロックを解放"""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            logger.error("ロック解放エラー: %s", e)

    def load_queue(self) -> list[dict]:
        """キューを読み込み"""
        if not self.queue_file.exists():
            return []

        try:
            with _exclusive_queue_json_lock():
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("jobs", [])
        except Exception as e:
            logger.error("キュー読み込みエラー: %s", e)
            return []

    def save_queue(self, jobs: list[dict]):
        """キューを保存（同一ディレクトリへ temp → replace で原子化）"""
        try:
            with _exclusive_queue_json_lock():
                payload = {
                    "jobs": jobs,
                    "updated_at": datetime.now().isoformat(),
                }
                d = self.queue_file.parent
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    delete=False,
                    dir=d,
                    prefix="scrape_queue_",
                    suffix=".json",
                ) as tf:
                    json.dump(payload, tf, ensure_ascii=False, indent=2)
                    tmp_path = tf.name
                try:
                    os.replace(tmp_path, self.queue_file)
                except OSError:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
        except Exception as e:
            logger.error("キュー保存エラー: %s", e)

    def _normalize_incoming_job(self, job: dict) -> dict[str, Any]:
        from scraper.queue_tasks import (
            build_job_label,
            normalize_tasks,
            validate_tasks_for_kind,
        )

        if job.get("job_kind"):
            kind = str(job["job_kind"]).strip()
            tid = str(job.get("target_id") or "").strip()
            if kind not in ("race", "horse", "date"):
                raise ValueError("job_kind は race / horse / date のいずれかです")
            if not tid:
                raise ValueError("target_id が必要です")
            tasks = normalize_tasks(job.get("tasks"))
            if not tasks:
                raise ValueError("tasks に1つ以上のタスクIDを指定してください")
            err = validate_tasks_for_kind(kind, tasks)
            if err:
                raise ValueError(err)
            dedupe = f"{kind}:{tid}:{':'.join(sorted(set(tasks)))}"
            label = job.get("job_label") or build_job_label(kind, tid, tasks)
            ss = job.get("smart_skip")
            smart_skip = True if ss is None else bool(ss)
            try:
                priority = int(job.get("priority", PRIORITY_DEFAULT))
            except (TypeError, ValueError):
                priority = PRIORITY_DEFAULT
            return {
                "job_kind": kind,
                "target_id": tid,
                "tasks": tasks,
                "dedupe_key": dedupe,
                "job_label": label,
                "date": job.get("date") or "",
                "venue": job.get("venue") or "",
                "round": job.get("round", 0),
                "race_name": job.get("race_name") or "",
                "smart_skip": smart_skip,
                "priority": priority,
            }

        race_id = str(job.get("race_id") or "").strip()
        if not race_id:
            raise ValueError("race_id または job_kind+target_id+tasks が必要です")
        tasks = ["race_all"]
        dedupe = f"race:{race_id}:race_all"
        label = build_job_label("race", race_id, tasks)
        try:
            priority = int(job.get("priority", PRIORITY_DEFAULT))
        except (TypeError, ValueError):
            priority = PRIORITY_DEFAULT
        return {
            "job_kind": "race",
            "target_id": race_id,
            "tasks": tasks,
            "dedupe_key": dedupe,
            "job_label": label,
            "date": job.get("date") or "",
            "venue": job.get("venue") or "",
            "round": job.get("round", 0),
            "race_name": job.get("race_name") or "",
            "priority": priority,
        }

    def _ingest_normalized_job(
        self, jobs: list[dict], normalized: dict[str, Any], original_job: dict
    ) -> tuple[str, str | None]:
        """
        正規化済み1件を jobs に反映（インプレース）。
        戻り値: (action, job_id) — action は created | requeued | duplicate
        """
        dedupe_key = normalized["dedupe_key"]

        new_pri = int(normalized.get("priority", PRIORITY_DEFAULT))

        for existing_job in jobs:
            if _effective_dedupe_key(existing_job) != dedupe_key:
                continue
            if existing_job.get("status") == "completed":
                existing_job["status"] = "pending"
                existing_job["queued_at"] = datetime.now().isoformat()
                if "smart_skip" in normalized:
                    existing_job["smart_skip"] = normalized["smart_skip"]
                existing_job["priority"] = max(
                    int(existing_job.get("priority") or PRIORITY_DEFAULT), new_pri
                )
                return "requeued", existing_job.get("job_id")
            if existing_job.get("status") == "pending":
                old_pri = int(existing_job.get("priority") or PRIORITY_DEFAULT)
                if new_pri > old_pri:
                    existing_job["priority"] = new_pri
                if "smart_skip" in normalized:
                    existing_job["smart_skip"] = normalized["smart_skip"]
            return "duplicate", existing_job.get("job_id")

        job_id = f"q_{int(time.time())}_{secrets.token_hex(4)}"
        tid = normalized["target_id"]
        new_job = {
            "job_id": job_id,
            "dedupe_key": dedupe_key,
            "job_kind": normalized["job_kind"],
            "target_id": tid,
            "tasks": normalized["tasks"],
            "job_label": normalized["job_label"],
            "race_id": tid if normalized["job_kind"] == "race" else original_job.get("race_id", ""),
            "date": normalized["date"],
            "venue": normalized["venue"],
            "round": normalized["round"],
            "race_name": normalized["race_name"],
            "types": list(normalized["tasks"]),
            "status": "pending",
            "queued_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "smart_skip": normalized.get("smart_skip", True),
            "priority": int(normalized.get("priority", PRIORITY_DEFAULT)),
        }
        jobs.append(new_job)
        return "created", job_id

    def add_job(self, job: dict) -> dict:
        """
        ジョブをキューに追加

        新形式: job_kind, target_id, tasks[]
        旧形式: race_id（race_all 相当）
        """
        jobs = self.load_queue()
        normalized = self._normalize_incoming_job(job)
        dedupe_key = normalized["dedupe_key"]
        action, job_id = self._ingest_normalized_job(jobs, normalized, job)
        self.save_queue(jobs)

        if action == "requeued":
            return {
                "status": "queued",
                "position": self._get_position(jobs, dedupe_key),
                "job_id": job_id,
                "action": "requeued",
            }
        if action == "duplicate":
            st = "pending"
            for ej in jobs:
                if _effective_dedupe_key(ej) == dedupe_key:
                    st = ej.get("status", "pending")
                    break
            return {
                "status": st,
                "position": self._get_position(jobs, dedupe_key),
                "job_id": job_id,
                "action": "duplicate",
            }

        if self.is_locked():
            return {
                "status": "queued",
                "position": self._get_position(jobs, dedupe_key),
                "job_id": job_id,
                "action": "created",
            }
        return {
            "status": "pending",
            "position": 1,
            "job_id": job_id,
            "action": "created",
        }

    def add_horse_jobs_bulk(self, horse_ids: list[str], tasks: list[str]) -> dict[str, int]:
        """
        同一 tasks の馬ジョブをまとめて追加（キューは1回だけ load/save）。
        horse_ids の順で処理し、dedupe は add_job と同じ。
        """
        from scraper.queue_tasks import normalize_tasks, validate_tasks_for_kind

        tasks_norm = normalize_tasks(tasks)
        if not tasks_norm:
            raise ValueError("tasks に1つ以上のタスクIDを指定してください")
        err = validate_tasks_for_kind("horse", tasks_norm)
        if err:
            raise ValueError(err)

        jobs = self.load_queue()
        created = requeued = duplicate = 0
        base = {"job_kind": "horse", "tasks": tasks_norm}

        for hid in horse_ids:
            hid = str(hid).strip()
            if not hid:
                continue
            spec = dict(base, target_id=hid)
            try:
                normalized = self._normalize_incoming_job(spec)
            except ValueError:
                continue
            action, _ = self._ingest_normalized_job(jobs, normalized, spec)
            if action == "created":
                created += 1
            elif action == "requeued":
                requeued += 1
            else:
                duplicate += 1

        self.save_queue(jobs)
        return {
            "created": created,
            "requeued": requeued,
            "duplicate": duplicate,
            "processed_horses": created + requeued + duplicate,
        }

    def bulk_add_jobs(self, specs: list[dict]) -> dict[str, int]:
        """
        複数ジョブを1回の load/save で投入。add_job と同じ正規化・重複判定。
        """
        jobs = self.load_queue()
        created = requeued = duplicate = 0
        skipped = 0
        for spec in specs:
            try:
                normalized = self._normalize_incoming_job(spec)
            except ValueError as e:
                logger.warning("bulk_add_jobs スキップ: %s — %s", spec, e)
                skipped += 1
                continue
            action, _ = self._ingest_normalized_job(jobs, normalized, spec)
            if action == "created":
                created += 1
            elif action == "requeued":
                requeued += 1
            else:
                duplicate += 1
        self.save_queue(jobs)
        return {
            "created": created,
            "requeued": requeued,
            "duplicate": duplicate,
            "skipped": skipped,
            "total_specs": len(specs),
        }

    def _sorted_pending(self, jobs: list[dict]) -> list[dict]:
        """優先度降順、同順位は queued_at 昇順（早い依頼を先）。"""
        pending = [j for j in jobs if j.get("status") == "pending"]
        pending.sort(
            key=lambda j: (
                -int(j.get("priority") or PRIORITY_DEFAULT),
                j.get("queued_at") or "",
            )
        )
        return pending

    def _get_position(self, jobs: list[dict], dedupe_key: str) -> int:
        """キュー内の位置（実行中があれば先頭扱い、続けて優先度ソート済み待ち）"""
        run = [j for j in jobs if j.get("status") == "running"]
        for j in run:
            if _effective_dedupe_key(j) == dedupe_key:
                return 1
        for i, job in enumerate(self._sorted_pending(jobs), 1):
            if _effective_dedupe_key(job) == dedupe_key:
                return i + len(run)
        return 0

    def requeue_stale_running_jobs(self, *, assume_lock_holder: bool = False) -> int:
        """
        status=running が JSON にだけ残っている孤児を pending に戻す。

        - assume_lock_holder=False（既定）: ロックファイルが無いときだけ実行。
          実ワーカーがいないのに running のみが残り、キューが進まない典型パターン向け。
        - assume_lock_holder=True: process_queue がロック取得直後に呼ぶ。前回クラッシュの running を掃除。
        """
        if not assume_lock_holder and self.is_locked():
            return 0
        jobs = self.load_queue()
        n = 0
        for job in jobs:
            if job.get("status") == "running":
                job["status"] = "pending"
                job["started_at"] = None
                n += 1
        if n:
            self.save_queue(jobs)
            logger.warning("孤児ジョブ %d 件を pending に戻しました（スクレイパ中断・プロセス強制終了など）", n)
        return n

    def get_next_job(self) -> dict | None:
        """次に実行すべきジョブを取得（優先度最大 → 依頼時刻が早い順）"""
        from scraper.scrape_access_pause import read_access_pause

        if read_access_pause().get("active"):
            return None
        jobs = self.load_queue()
        pend = self._sorted_pending(jobs)
        return pend[0] if pend else None

    def update_job_status(self, job_id: str, status: str, error: str | None = None):
        """ジョブのステータスを更新"""
        jobs = self.load_queue()
        for job in jobs:
            if job.get("job_id") == job_id:
                job["status"] = status
                if status == "running":
                    job["started_at"] = datetime.now().isoformat()
                elif status in ("completed", "failed"):
                    job["completed_at"] = datetime.now().isoformat()
                if error:
                    job["error"] = error
                break
        self.save_queue(jobs)

    def pause_queue_for_access_error(self, reason: str) -> int:
        """
        実行中ジョブをすべて pending に戻し、アクセス系一時停止フラグを立てる。
        フラグが立っている間は get_next_job は常に None。
        """
        from scraper.scrape_access_pause import write_access_pause

        jobs = self.load_queue()
        n = 0
        for job in jobs:
            if job.get("status") == "running":
                job["status"] = "pending"
                job["started_at"] = None
                job["error"] = None
                n += 1
        self.save_queue(jobs)
        write_access_pause(reason=reason)
        logger.warning(
            "アクセス系エラーによりキューを一時停止しました（実行中→待機: %d 件）: %s",
            n,
            (reason or "")[:500],
        )
        return n

    def get_status(self) -> dict:
        """キュー全体のステータスを取得"""
        from scraper.scrape_access_pause import read_access_pause

        jobs = self.load_queue()
        is_running = self.is_locked()

        pending = [j for j in jobs if j.get("status") == "pending"]
        running = [j for j in jobs if j.get("status") == "running"]
        completed = [j for j in jobs if j.get("status") == "completed"]
        failed = [j for j in jobs if j.get("status") == "failed"]

        current_job = running[0] if running else None
        pending_ordered = self._sorted_pending(jobs)
        active_jobs = running + pending_ordered
        processing_queue = running + pending_ordered

        failed_sorted = sorted(
            [j for j in jobs if j.get("status") == "failed"],
            key=lambda j: (j.get("completed_at") or "", j.get("job_id") or ""),
            reverse=True,
        )[:200]

        pause = read_access_pause()
        return {
            "is_running": is_running,
            "current_job": current_job,
            "pending_queue": processing_queue[:200],
            "active_jobs": active_jobs,
            "transport_pause": {
                "active": bool(pause.get("active")),
                "reason": pause.get("reason"),
                "paused_at": pause.get("paused_at"),
            },
            "queue": {
                "pending": len(pending),
                "running": len(running),
                "completed": len(completed),
                "failed": len(failed),
                "total": len(jobs),
            },
            "jobs": jobs[-80:],
            "failed_jobs": failed_sorted,
            "job_by_id": {
                str(j["job_id"]): j for j in jobs if j.get("job_id")
            },
        }

    def requeue_failed_jobs(
        self,
        *,
        job_ids: list[str] | None = None,
        all_failed: bool = False,
    ) -> tuple[int, str | None]:
        """
        status=failed を pending に戻す。
        all_failed=True なら全 failed、否则は job_ids で指定（1件以上必須）。
        戻り値: (件数, エラーメッセージ or None)
        """
        ids = {str(x).strip() for x in (job_ids or []) if str(x).strip()}
        if not all_failed and not ids:
            return 0, "job_ids を指定するか all_failed を true にしてください"
        jobs = self.load_queue()
        n = 0
        for job in jobs:
            if job.get("status") != "failed":
                continue
            jid = str(job.get("job_id") or "")
            if not all_failed and jid not in ids:
                continue
            job["status"] = "pending"
            job["started_at"] = None
            job["completed_at"] = None
            job["error"] = None
            job["queued_at"] = datetime.now().isoformat()
            n += 1
        if n:
            self.save_queue(jobs)
        elif not all_failed and ids:
            return 0, "該当する失敗ジョブがありません（ID を確認してください）"
        return n, None

    def remove_failed_jobs(
        self,
        *,
        job_ids: list[str] | None = None,
        all_failed: bool = False,
    ) -> tuple[int, str | None]:
        """failed のみキューから削除。all_failed または job_ids が必要。"""
        ids = {str(x).strip() for x in (job_ids or []) if str(x).strip()}
        if not all_failed and not ids:
            return 0, "job_ids を指定するか all_failed を true にしてください"
        jobs = self.load_queue()
        new_jobs: list[dict] = []
        removed = 0
        for job in jobs:
            if job.get("status") != "failed":
                new_jobs.append(job)
                continue
            jid = str(job.get("job_id") or "")
            if all_failed or jid in ids:
                removed += 1
            else:
                new_jobs.append(job)
        if removed:
            self.save_queue(new_jobs)
        elif not all_failed and ids:
            return 0, "該当する失敗ジョブがありません（ID を確認してください）"
        return removed, None

    def clear_completed(self):
        """完了・失敗したジョブをクリア"""
        jobs = self.load_queue()
        jobs = [j for j in jobs if j.get("status") not in ("completed", "failed")]
        self.save_queue(jobs)

    def clear_all_jobs(self) -> int:
        """待機・実行中・完了・失敗を含むジョブをすべて削除する。戻り値: 削除前件数。"""
        jobs = self.load_queue()
        n = len(jobs)
        self.save_queue([])
        return n

    def process_queue(self):
        """
        キューを処理する（バックグラウンドワーカーから呼ばれる）
        """
        from scraper.queue_worker_log import ensure_queue_worker_log_handler, mark_queue_worker_active

        ensure_queue_worker_log_handler()

        if not self.acquire_lock():
            logger.info("既にスクレイピング処理が実行中です")
            return

        mark_queue_worker_active(True)
        try:
            self.requeue_stale_running_jobs(assume_lock_holder=True)

            from scraper.scrape_access_pause import is_access_or_transport_error, read_access_pause

            while True:
                if read_access_pause().get("active"):
                    logger.info("アクセス一時停止中のためキュー処理を中断します")
                    break

                job = self.get_next_job()
                if not job:
                    logger.info("キューが空です")
                    break

                job_id = job["job_id"]
                label = job.get("job_label") or job.get("target_id") or job.get("race_id")

                logger.info("ジョブ開始: %s (%s)", job_id, label)
                self.update_job_status(job_id, "running")

                try:
                    self._execute_scraping(job)
                    self.update_job_status(job_id, "completed")
                    logger.info("ジョブ完了: %s", job_id)

                except Exception as e:
                    if is_access_or_transport_error(e):
                        logger.error(
                            "アクセス／ネットワーク系エラー — キューを一時停止し全実行中ジョブを待機に戻します: %s",
                            job_id,
                            exc_info=True,
                        )
                        self.pause_queue_for_access_error(str(e))
                        break
                    logger.error("ジョブ失敗: %s - %s", job_id, e, exc_info=True)
                    self.update_job_status(job_id, "failed", str(e))

        finally:
            mark_queue_worker_active(False)
            self.release_lock()

    def _execute_scraping(self, job: dict):
        """実際のスクレイピングを実行"""
        from scraper.queue_tasks import execute_job
        from scraper.run import ScraperRunner

        runner = ScraperRunner()
        execute_job(runner, job)
        time.sleep(1)  # レート制限
