"""
スクレイピングジョブキューシステム

複数のスクレイピングリクエストを管理する。
ファイルベースのキュー + ロックファイルで「キュー処理セッション」の排他（別プロセスが同時に process_queue しない）。
process_queue 内は SCRAPE_QUEUE_PARALLEL 本（既定 6）まで同時にジョブ実行する。

ジョブは job_kind（race / horse / date）+ target_id + tasks[] で任意の取得単位を指定できる。
期間内の出走馬への馬タスクは API enqueue-period-horses が race_lists→各レースの出馬表/結果から馬IDを展開してからキューへ載せる。
期間内の JRA レースへのレースタスクは enqueue-period-races が race_lists を走査して race_id 単位で bulk_add_jobs する。

**上書きなし**かつ **smart_skip**（スキップあり）の新規行は、タスク内容に依らず待機 (pending) の前に
status ``precheck`` を経由する。``process_queue`` 冒頭の ``_run_storage_precheck`` で
ストレージ上の充足を一括判定し、満足分を ``completed``、要取得を ``pending`` へ遷移する
（``horse_training`` は併せて GCS 一括キー照会の最適化あり。判定基準は verify 系の
``is_*_task_satisfied`` と揃える）。

定期メンテ: API リーダー process が一定間隔（SCRAPE_QUEUE_HOURLY_MAINTENANCE_SEC、既定1時間）で
失敗ジョブを全件待機に戻し、完了（completed）レコードだけをキュー JSON から除去。終了目安は get_status 内の queue_eta が毎回再計算される。
"""

import copy
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore
    _HAS_FCNTL = False

logger = logging.getLogger(__name__)


def _horse_pedigree_5gen_complete_in_index(
    hid: str,
    pedigree_index: dict[str, dict] | None,
) -> bool:
    """
    ローカルスナップショット索引だけで 5 世代揃いを判定（I/O なし）。
    索引に無いが GCS にだけある馬は False（ワーカ側 smart_skip で実取得を省略）。
    """
    if not pedigree_index:
        return False
    rec = pedigree_index.get(hid)
    return rec is not None and len((rec.get("ancestors") or [])) >= 5


# キューディレクトリ
QUEUE_DIR = Path(__file__).parents[2] / "data" / "queue"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_HOURLY_MAINTAIN_STATE = QUEUE_DIR / "queue_hourly_maintain_state.json"

LOCK_FILE = QUEUE_DIR / ".scrape.lock"
LOCK_FILE_URGENT = QUEUE_DIR / ".scrape_urgent.lock"
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

# process_queue 内の同時実行数（1〜32）。
# 既定は queue_load_settings.json または環境変数 SCRAPE_QUEUE_PARALLEL（既定6）。
# 多並列＋各ジョブ独立 Session だと netkeiba 側のブロックが出やすい
def _queue_parallel_workers() -> int:
    from src.scraper.queue_load_settings import get_effective_parallel

    return int(get_effective_parallel())


def _queue_stagger_delay_sec() -> float:
    """一括起動で同秒スタートを避けるジョブ開始遅延（秒・ジョブごとに idx 倍）。"""
    from src.scraper.queue_load_settings import get_effective_stagger_sec

    return get_effective_stagger_sec()


def _task_list_normalized(job: dict) -> list[str]:
    raw = job.get("tasks") or job.get("types")
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    return [s]


def _job_is_likely_fast_horse_training_skip(job: dict) -> bool:
    """
    ワーカでネット再取得に入らない可能性が高い horse_training（ETA 用ヒューリスティック）。

    ScrapeRunner.scrape_horse_training: ローカル+GCS 既存のとき即返（overwrite や smart_skip 除外）。
    """
    st = str(job.get("status") or "")
    if st not in ("pending", "running", "precheck"):
        return False
    if str(job.get("job_kind") or "") != "horse":
        return False
    if job.get("overwrite") is True:
        return False
    if job.get("smart_skip") is False:
        return False
    tnorm = _task_list_normalized(job)
    if tnorm != ["horse_training"]:
        return False
    return True


def _job_eligible_for_storage_precheck(normalized: dict[str, Any]) -> bool:
    """
    投入ジョブが precheck 列に乗るか: 上書きなし・スキップあり（smart_skip）のとき全タスク種で統一。
    """
    if bool(normalized.get("overwrite", False)) is True:
        return False
    if bool(normalized.get("smart_skip", True)) is False:
        return False
    kind = str(normalized.get("job_kind") or "").strip()
    if kind not in ("race", "horse", "date"):
        return False
    t = normalized.get("tasks")
    if not t:
        return False
    if isinstance(t, (list, tuple)):
        x = [str(s).strip() for s in t if str(s).strip()]
    else:
        x = [str(t).strip()]
    return bool(x)


def _initial_status_for_new_job(
    normalized: dict[str, Any], *_extra: Any,
) -> str:
    if _job_eligible_for_storage_precheck(normalized):
        return "precheck"
    return "pending"


def _job_row_eligible_for_storage_precheck(job: dict) -> bool:
    """ディスク上の1ジョブ行（tasks/types 等）を正規化して precheck 列へ戻せるか。"""
    n: dict[str, Any] = {
        "job_kind": job.get("job_kind"),
        "tasks": job.get("tasks") or job.get("types"),
        "smart_skip": job.get("smart_skip", True),
        "overwrite": job.get("overwrite", False),
    }
    return _job_eligible_for_storage_precheck(n)


def _eta_sec_for_queue_jobs(
    sec_per_config: float, rem: int, active_jobs: list[dict] | None
) -> tuple[float, int, int]:
    """
    補正後の 1 ジョブ秒・smart_skip 扱い件数・件数。バルク horse_training で過大表示を抑える。

    環境変数 SCRAPE_QUEUE_ETA_HORSE_TRAINING_SKIP_SEC（既定 0.55）を重み付けブレンドに使用。
    """
    sec_per = max(0.5, float(sec_per_config))
    if not active_jobs or rem <= 0:
        return sec_per, 0, rem
    n_fast = sum(1 for j in active_jobs if _job_is_likely_fast_horse_training_skip(j))
    if n_fast < rem * 0.12 or n_fast < 5:
        return sec_per, n_fast, rem
    try:
        v = float(os.environ.get("SCRAPE_QUEUE_ETA_HORSE_TRAINING_SKIP_SEC", "0.55"))
    except (TypeError, ValueError):
        v = 0.55
    sec_fast = max(0.12, min(8.0, v))
    r = n_fast / float(rem)
    blended = (1.0 - r) * sec_per + r * sec_fast
    return max(0.2, blended), n_fast, rem


def _build_queue_eta(
    *,
    n_pending: int,
    n_running: int,
    transport_paused: bool,
    active_jobs: list[dict] | None = None,
) -> dict[str, Any]:
    """
    キュー消化の粗い目安（queue_status の「終了の目安」用）。

    基本: ceil(残り件数 / 並列数) × 1 ジョブ秒（SCRAPE_QUEUE_ETA_SEC_PER_JOB、既定 10）。
    主に ``horse`` × ``horse_training`` × ``smart_skip`` の列では 1 ジョブ秒を下げ、過大表示を抑える。
    """
    import math

    rem = int(n_pending) + int(n_running)
    if rem <= 0:
        return {
            "available": True,
            "remaining_jobs": 0,
            "eta_seconds": 0,
            "eta_finish_at_jst": None,
            "note": "",
            "eta_sec_per_job_effective": 0.0,
            "horse_training_skip_optimism_jobs": 0,
        }
    if transport_paused:
        return {
            "available": False,
            "remaining_jobs": rem,
            "eta_seconds": None,
            "eta_finish_at_jst": None,
            "note": "アクセス一時停止中のため終了時刻は算出していません",
            "eta_sec_per_job_effective": None,
            "horse_training_skip_optimism_jobs": 0,
        }
    from src.scraper.queue_load_settings import get_effective_eta_sec_per_job

    workers = max(1, _queue_parallel_workers())
    sec_per = get_effective_eta_sec_per_job()
    sec_per = max(0.5, float(sec_per))
    act = active_jobs
    if act is not None and len(act) != rem:
        act = list(act)[:rem]
    sec_effective, n_opt, _ = _eta_sec_for_queue_jobs(sec_per, rem, act)
    batches = max(1, math.ceil(rem / workers))
    eta_seconds = int(max(1, math.ceil(batches * sec_effective)))
    jst = timezone(timedelta(hours=9))
    finish = datetime.now(tz=jst) + timedelta(seconds=eta_seconds)
    eta_finish_at_jst = finish.strftime("%Y-%m-%d %H:%M")
    use_blend = n_opt > 0 and (sec_effective < sec_per - 1e-6)
    note = (
        f"並列 {workers} 本・設定上 1 ジョブ {sec_per:g} 秒"
        f"（.env または queue_load_settings.json / SCRAPE_QUEUE_ETA_SEC_PER_JOB）· 約 {batches} ウェーブ"
    )
    if use_blend:
        try:
            skip_sec = float(os.environ.get("SCRAPE_QUEUE_ETA_HORSE_TRAINING_SKIP_SEC", "0.55"))
        except (TypeError, ValueError):
            skip_sec = 0.55
        note += (
            f" · 表示補正: smart_skip+horse_training 想定 {n_opt} 件"
            f" により 1 ジョブ ≈{sec_effective:.2f}s 相当"
            f"（SKIP ベース {skip_sec:g}s: SCRAPE_QUEUE_ETA_HORSE_TRAINING_SKIP_SEC）"
        )
    return {
        "available": True,
        "remaining_jobs": rem,
        "eta_seconds": eta_seconds,
        "eta_finish_at_jst": eta_finish_at_jst,
        "note": note,
        "eta_sec_per_job_effective": round(sec_effective, 4),
        "horse_training_skip_optimism_jobs": n_opt,
    }


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

    _LOCK_TIMEOUT = 1800  # 30分

    def is_locked(self) -> bool:
        """現在スクレイピング処理が実行中かチェック"""
        if not self.lock_file.exists():
            return False

        try:
            mtime = self.lock_file.stat().st_mtime
            if time.time() - mtime > self._LOCK_TIMEOUT:
                logger.warning("古いロックファイルを削除 (%.0f分超): %s",
                               self._LOCK_TIMEOUT / 60, self.lock_file)
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

    def touch_lock(self):
        """ロックファイルのmtimeを更新（ハートビート）。長時間ジョブ中にタイムアウト回避。"""
        try:
            if self.lock_file.exists():
                self.lock_file.touch()
            else:
                with open(self.lock_file, "w") as f:
                    f.write(json.dumps({
                        "pid": os.getpid(),
                        "timestamp": time.time(),
                        "datetime": datetime.now().isoformat(),
                    }))
        except OSError:
            pass

    def release_lock(self):
        """ロックを解放"""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            logger.error("ロック解放エラー: %s", e)

    def _load_queue_nolock(self) -> list[dict]:
        """QUEUE_JSON の flock 取得済みのときのみ。"""
        if not self.queue_file.exists():
            return []
        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("jobs", [])
        except Exception as e:
            logger.error("キュー読み込みエラー: %s", e)
            return []

    def load_queue(self) -> list[dict]:
        """キューを読み込み"""
        try:
            with _exclusive_queue_json_lock():
                return self._load_queue_nolock()
        except Exception as e:
            logger.error("キュー読み込みエラー: %s", e)
            return []

    def _save_queue_nolock(self, jobs: list[dict]) -> None:
        """QUEUE_JSON の flock 取得済みのときのみ。"""
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

    def save_queue(self, jobs: list[dict]):
        """キューを保存（同一ディレクトリへ temp → replace で原子化）"""
        try:
            with _exclusive_queue_json_lock():
                self._save_queue_nolock(jobs)
        except Exception as e:
            logger.error("キュー保存エラー: %s", e)

    def _normalize_incoming_job(self, job: dict) -> dict[str, Any]:
        from src.scraper.queue_tasks import (
            build_job_label,
            normalize_tasks,
            validate_tasks_for_kind,
        )
        from src.scraper.scrape_policy import coerce_bool, resolve_enqueue_overwrite_smart_skip

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
            overwrite, smart_skip = resolve_enqueue_overwrite_smart_skip(job)
            try:
                priority = int(job.get("priority", PRIORITY_DEFAULT))
            except (TypeError, ValueError):
                priority = PRIORITY_DEFAULT
            skip_lm = bool(coerce_bool(job.get("skip_local_mirror"), default=False))
            skip_ped = bool(coerce_bool(job.get("skip_pedigree"), default=False))
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
                "overwrite": overwrite,
                "priority": priority,
                "skip_local_mirror": skip_lm,
                "skip_pedigree": skip_ped,
            }

        race_id = str(job.get("race_id") or "").strip()
        if not race_id:
            raise ValueError("race_id または job_kind+target_id+tasks が必要です")
        tasks = ["race_all"]
        dedupe = f"race:{race_id}:race_all"
        label = build_job_label("race", race_id, tasks)
        overwrite, smart_skip = resolve_enqueue_overwrite_smart_skip(job)
        try:
            priority = int(job.get("priority", PRIORITY_DEFAULT))
        except (TypeError, ValueError):
            priority = PRIORITY_DEFAULT
        skip_lm = bool(coerce_bool(job.get("skip_local_mirror"), default=False))
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
            "smart_skip": smart_skip,
            "overwrite": overwrite,
            "priority": priority,
            "skip_local_mirror": skip_lm,
        }

    def _ingest_normalized_job(
        self,
        jobs: list[dict],
        normalized: dict[str, Any],
        original_job: dict,
        *,
        dedupe_index: dict[str, dict] | None = None,
    ) -> tuple[str, str | None]:
        """
        正規化済み1件を jobs に反映（インプレース）。
        戻り値: (action, job_id) — action は created | requeued | duplicate

        dedupe_index を渡すと dedupe_key 検索が O(1)（大量 bulk 投入用）。
        """
        dedupe_key = normalized["dedupe_key"]

        new_pri = int(normalized.get("priority", PRIORITY_DEFAULT))

        if dedupe_index is not None:
            existing_job = dedupe_index.get(dedupe_key)
        else:
            existing_job = None
            for ej in jobs:
                if _effective_dedupe_key(ej) == dedupe_key:
                    existing_job = ej
                    break

        if existing_job is not None:
            if existing_job.get("status") == "completed":
                existing_job["status"] = _initial_status_for_new_job(
                    normalized, original_job
                )
                existing_job["queued_at"] = datetime.now().isoformat()
                existing_job["smart_skip"] = normalized["smart_skip"]
                existing_job["overwrite"] = normalized["overwrite"]
                existing_job["skip_local_mirror"] = bool(
                    normalized.get("skip_local_mirror", False)
                )
                if "skip_pedigree" in normalized:
                    existing_job["skip_pedigree"] = bool(normalized["skip_pedigree"])
                existing_job["priority"] = max(
                    int(existing_job.get("priority") or PRIORITY_DEFAULT), new_pri
                )
                return "requeued", existing_job.get("job_id")
            if existing_job.get("status") in ("pending", "precheck"):
                old_pri = int(existing_job.get("priority") or PRIORITY_DEFAULT)
                if new_pri > old_pri:
                    existing_job["priority"] = new_pri
                existing_job["smart_skip"] = normalized["smart_skip"]
                existing_job["overwrite"] = normalized["overwrite"]
                existing_job["skip_local_mirror"] = bool(
                    normalized.get("skip_local_mirror", False)
                )
                if "skip_pedigree" in normalized:
                    existing_job["skip_pedigree"] = bool(normalized["skip_pedigree"])
            return "duplicate", existing_job.get("job_id")

        job_id = f"q_{int(time.time())}_{secrets.token_hex(4)}"
        tid = normalized["target_id"]
        init_st = _initial_status_for_new_job(normalized, original_job)
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
            "status": init_st,
            "queued_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "smart_skip": normalized.get("smart_skip", True),
            "overwrite": bool(normalized.get("overwrite", False)),
            "priority": int(normalized.get("priority", PRIORITY_DEFAULT)),
            "skip_local_mirror": bool(normalized.get("skip_local_mirror", False)),
            "skip_pedigree": bool(normalized.get("skip_pedigree", False)),
        }
        jobs.append(new_job)
        if dedupe_index is not None:
            dedupe_index[dedupe_key] = new_job
        return "created", job_id

    def add_job(self, job: dict) -> dict:
        """
        ジョブをキューに追加

        新形式: job_kind, target_id, tasks[]
        旧形式: race_id（race_all 相当）
        """
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            normalized = self._normalize_incoming_job(job)
            dedupe_key = normalized["dedupe_key"]
            action, job_id = self._ingest_normalized_job(jobs, normalized, job)
            self._save_queue_nolock(jobs)

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

        st_new = "pending"
        for ej in jobs:
            if str(ej.get("job_id")) == str(job_id):
                st_new = str(ej.get("status") or "pending")
                break
        if self.is_locked():
            return {
                "status": "queued",
                "position": self._get_position(jobs, dedupe_key),
                "job_id": job_id,
                "action": "created",
                "job_status": st_new,
            }
        return {
            "status": st_new,
            "position": 1,
            "job_id": job_id,
            "action": "created",
            "job_status": st_new,
        }

    def add_horse_jobs_bulk(
        self,
        horse_ids: list[str],
        tasks: list[str],
        *,
        priority: int | None = None,
        smart_skip: bool | None = None,
        overwrite: bool | None = None,
        skip_local_mirror: bool | None = None,
        pedigree_index: dict[str, dict] | None = None,
        skip_pedigree_5gen_if_complete: bool = False,
    ) -> dict[str, int]:
        """
        同一 tasks の馬ジョブをまとめて追加（キューは1回だけ load/save）。
        horse_ids の順で処理し、dedupe は add_job と同じ。

        skip_pedigree_5gen_if_complete=True かつ tasks が horse_pedigree_5gen のみのとき、
        pedigree_index 上で既に 5 世代分の ancestors がある馬はキューに載せない（索引 I/O のみ）。
        索引に無い馬はキューに載せ、ワーカ側 smart_skip が GCS 既存を省略する。
        """
        from src.scraper.queue_tasks import normalize_tasks, validate_tasks_for_kind

        tasks_norm = normalize_tasks(tasks)
        if not tasks_norm:
            raise ValueError("tasks に1つ以上のタスクIDを指定してください")
        err = validate_tasks_for_kind("horse", tasks_norm)
        if err:
            raise ValueError(err)

        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            dedupe_index: dict[str, dict] = {}
            for j in jobs:
                dk = _effective_dedupe_key(j)
                if dk and dk not in dedupe_index:
                    dedupe_index[dk] = j

            created = requeued = duplicate = skipped_complete = 0
            base: dict[str, Any] = {"job_kind": "horse", "tasks": tasks_norm}
            if priority is not None:
                base["priority"] = int(priority)
            if smart_skip is not None:
                base["smart_skip"] = bool(smart_skip)
            if overwrite is not None:
                base["overwrite"] = bool(overwrite)
            if skip_local_mirror is not None:
                base["skip_local_mirror"] = bool(skip_local_mirror)

            do_skip_complete = (
                skip_pedigree_5gen_if_complete
                and tasks_norm == ["horse_pedigree_5gen"]
            )

            for hid in horse_ids:
                hid = str(hid).strip()
                if not hid:
                    continue
                if do_skip_complete and _horse_pedigree_5gen_complete_in_index(
                    hid, pedigree_index
                ):
                    skipped_complete += 1
                    continue
                spec = dict(base, target_id=hid)
                try:
                    normalized = self._normalize_incoming_job(spec)
                except ValueError:
                    continue
                action, _ = self._ingest_normalized_job(
                    jobs, normalized, spec, dedupe_index=dedupe_index
                )
                if action == "created":
                    created += 1
                elif action == "requeued":
                    requeued += 1
                else:
                    duplicate += 1

            self._save_queue_nolock(jobs)
        return {
            "created": created,
            "requeued": requeued,
            "duplicate": duplicate,
            "skipped_already_complete": skipped_complete,
            "processed_horses": created + requeued + duplicate,
        }

    def bulk_add_jobs(self, specs: list[dict]) -> dict[str, int]:
        """
        複数ジョブを1回の load/save で投入。add_job と同じ正規化・重複判定。
        """
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            dedupe_index: dict[str, dict] = {}
            for j in jobs:
                dk = _effective_dedupe_key(j)
                if dk and dk not in dedupe_index:
                    dedupe_index[dk] = j

            created = requeued = duplicate = 0
            skipped = 0
            for spec in specs:
                try:
                    normalized = self._normalize_incoming_job(spec)
                except ValueError as e:
                    logger.warning("bulk_add_jobs スキップ: %s — %s", spec, e)
                    skipped += 1
                    continue
                action, _ = self._ingest_normalized_job(
                    jobs, normalized, spec, dedupe_index=dedupe_index
                )
                if action == "created":
                    created += 1
                elif action == "requeued":
                    requeued += 1
                else:
                    duplicate += 1
            self._save_queue_nolock(jobs)
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

    def _sorted_precheck(self, jobs: list[dict]) -> list[dict]:
        pre = [j for j in jobs if j.get("status") == "precheck"]
        pre.sort(
            key=lambda j: (
                -int(j.get("priority") or PRIORITY_DEFAULT),
                j.get("queued_at") or "",
            )
        )
        return pre

    def count_precheck_jobs(self) -> int:
        """現在の precheck ジョブ数を返す（ロック不要・軽量）。"""
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
        return sum(1 for j in jobs if j.get("status") == "precheck")

    def _work_queue_list(self, jobs: list[dict]) -> list[dict]:
        """存在確認 (precheck) 済み待機 (pending) の可視用一覧（実行順イメージ）。"""
        return self._sorted_precheck(jobs) + self._sorted_pending(jobs)

    def _get_position(self, jobs: list[dict], dedupe_key: str) -> int:
        """キュー内の位置（実行中があれば先頭扱い、続けて優先度ソート済み待ち）"""
        run = [j for j in jobs if j.get("status") == "running"]
        for j in run:
            if _effective_dedupe_key(j) == dedupe_key:
                return 1
        for i, job in enumerate(self._work_queue_list(jobs), 1):
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
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            n = 0
            for job in jobs:
                if job.get("status") == "running":
                    job["status"] = "pending"
                    job["started_at"] = None
                    n += 1
            if n:
                self._save_queue_nolock(jobs)
        if n:
            logger.warning(
                "孤児ジョブ %d 件を pending に戻しました（スクレイパ中断・プロセス強制終了など）",
                n,
            )
        return n

    def _run_storage_precheck(self) -> dict[str, Any]:
        """
        status=precheck（上書きなし・スキップあり）の行をストレージで充足判定し、
        完了 or pending へ。process_queue / process_queue_urgent 冒頭で1回。

        判定は ``queue_storage_precheck`` の統一フロー（``horse_training`` には
        従来どおり batch キー照会を併用）。
        """
        from src.scraper.queue_storage_precheck import (
            apply_unified_storage_precheck_patches,
            plan_unified_storage_precheck,
        )
        from src.scraper.run import REPO_ROOT
        from src.scraper.storage import HybridStorage

        st = HybridStorage(str(REPO_ROOT))
        with _exclusive_queue_json_lock():
            j0 = self._load_queue_nolock()
        if not any(
            j.get("status") == "precheck"
            and j.get("overwrite") is not True
            and j.get("smart_skip", True) is not False
            for j in j0
        ):
            return {
                "ok": True,
                "from_precheck": 0,
                "to_pending": 0,
                "to_completed": 0,
                "skipped": 0,
            }
        patches = plan_unified_storage_precheck(j0, st)
        if not patches:
            return {
                "ok": True,
                "from_precheck": 0,
                "to_pending": 0,
                "to_completed": 0,
                "skipped": 0,
            }
        with _exclusive_queue_json_lock():
            j1 = self._load_queue_nolock()
            pstats = apply_unified_storage_precheck_patches(j1, patches)
            self._save_queue_nolock(j1)
        if pstats.to_completed or pstats.to_pending or pstats.skipped:
            logger.info(
                "storage precheck(unified): 対象=%d 直完=%d 待機送り=%d 行取り違い=%d",
                pstats.from_precheck,
                pstats.to_completed,
                pstats.to_pending,
                pstats.skipped,
            )
        return {
            "ok": True,
            "from_precheck": pstats.from_precheck,
            "to_pending": pstats.to_pending,
            "to_completed": pstats.to_completed,
            "skipped": pstats.skipped,
        }

    def _run_storage_precheck_horse(self) -> dict[str, Any]:
        """下位互換名。``_run_storage_precheck`` と同一。"""
        return self._run_storage_precheck()

    def run_storage_precheck_horse_now(self) -> dict[str, Any]:
        """
        ワーカを起動せずに precheck 充足判定だけ走らせる（投入直後の中身確定用）。
        ``process_queue`` 冒頭の ``_run_storage_precheck`` と同じ挙止。
        """
        return self._run_storage_precheck()

    def run_storage_precheck_now(self) -> dict[str, Any]:
        """``run_storage_precheck_horse_now`` と同一（名称が対象全ジョブを指す）。"""
        return self._run_storage_precheck()

    def migrate_pending_to_storage_precheck(self, *, dry_run: bool = False) -> dict[str, Any]:
        """
        既存 ``pending`` のうち、上書きなし・スキップあり（smart_skip）の行を ``precheck`` へ戻し、
        次回 ``_run_storage_precheck`` の対象にする。running は触らない。dry_run は件数のみ。
        """
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            n = 0
            for j in jobs:
                if j.get("status") != "pending":
                    continue
                if not _job_row_eligible_for_storage_precheck(j):
                    continue
                n += 1
            if n == 0:
                return {"ok": True, "migrated": 0, "would_migrate": 0}
            if dry_run:
                return {"ok": True, "migrated": 0, "would_migrate": n}
            for j in jobs:
                if j.get("status") != "pending":
                    continue
                if not _job_row_eligible_for_storage_precheck(j):
                    continue
                j["status"] = "precheck"
            self._save_queue_nolock(jobs)
        logger.info("migrate: pending→precheck(unified) %d 件", n)
        return {"ok": True, "migrated": n, "would_migrate": n}

    def migrate_pending_horse_training_to_precheck(
        self, *, dry_run: bool = False
    ) -> dict[str, Any]:
        """下位互換: 以前は horse_training 専用だったが、全ジョブで migrate_pending_to_storage_precheck と同じ。"""
        return self.migrate_pending_to_storage_precheck(dry_run=dry_run)

    def get_next_job(self) -> dict | None:
        """次に実行すべきジョブを取得（優先度最大 → 依頼時刻が早い順）"""
        from src.scraper.scrape_access_pause import read_access_pause

        if read_access_pause().get("active"):
            return None
        jobs = self.load_queue()
        pend = self._sorted_pending(jobs)
        return pend[0] if pend else None

    def claim_pending_jobs_batch(self, max_n: int) -> list[dict]:
        """
        待機中を先頭から最大 max_n 件、1 回の flock で running にし、各コピーを返す。
        連続 claim による JSON ロック争奪を減らし、並列枠を素早く埋める。
        """
        from src.scraper.scrape_access_pause import read_access_pause

        if max_n <= 0:
            return []
        if read_access_pause().get("active"):
            return []
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            pend = self._sorted_pending(jobs)
            if not pend:
                return []
            out: list[dict] = []
            for job in pend[:max_n]:
                jid = job.get("job_id")
                if not jid:
                    continue
                for j in jobs:
                    if j.get("job_id") == jid:
                        j["status"] = "running"
                        j["started_at"] = datetime.now().isoformat()
                        break
                out.append(copy.deepcopy(job))
            if not out:
                return []
            self._save_queue_nolock(jobs)
            return out

    def claim_next_pending_job(self) -> dict | None:
        """
        待機中の先頭1件を原子的に running にし、そのコピーを返す。
        並列ワーカーが同時に同じジョブを取らないために使う。
        """
        batch = self.claim_pending_jobs_batch(1)
        return batch[0] if batch else None

    def update_job_status(self, job_id: str, status: str, error: str | None = None):
        """ジョブのステータスを更新"""
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            for job in jobs:
                if job.get("job_id") == job_id:
                    prev = job.get("status")
                    if status in ("completed", "failed") and prev != "running":
                        # 並列実行中に pause_queue が pending に戻したあとに完了しないよう無視
                        # （precheck→完了はキュー行を直接更新する。本メソッドは主に running 完了用）
                        logger.debug(
                            "ジョブステータス更新スキップ (%s→%s): %s",
                            prev,
                            status,
                            job_id,
                        )
                        return
                    job["status"] = status
                    if status == "running":
                        job["started_at"] = datetime.now().isoformat()
                    elif status in ("completed", "failed"):
                        job["completed_at"] = datetime.now().isoformat()
                    if error:
                        job["error"] = error
                    break
            self._save_queue_nolock(jobs)

    def _requeue_with_retry(self, job_id: str, retry_count: int) -> None:
        """ジョブを pending に戻し、リトライカウントを記録。"""
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            for job in jobs:
                if job.get("job_id") == job_id:
                    job["status"] = "pending"
                    job["started_at"] = None
                    job["retry_count"] = retry_count
                    job["error"] = None
                    break
            self._save_queue_nolock(jobs)

    def pause_queue_for_access_error(self, reason: str) -> int:
        """
        実行中ジョブをすべて pending に戻し、アクセス系一時停止フラグを立てる。
        フラグが立っている間は get_next_job は常に None。
        """
        from src.scraper.scrape_access_pause import write_access_pause

        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            n = 0
            for job in jobs:
                if job.get("status") == "running":
                    job["status"] = "pending"
                    job["started_at"] = None
                    job["error"] = None
                    n += 1
            self._save_queue_nolock(jobs)
        write_access_pause(reason=reason)
        logger.warning(
            "アクセス系エラーによりキューを一時停止しました（実行中→待機: %d 件）: %s",
            n,
            (reason or "")[:500],
        )
        return n

    def get_status(self) -> dict:
        """キュー全体のステータスを取得"""
        from src.scraper.scrape_access_pause import read_access_pause

        jobs = self.load_queue()
        is_running = self.is_locked()
        is_running_urgent = self.is_locked_urgent()

        pending = [j for j in jobs if j.get("status") == "pending"]
        precheck = [j for j in jobs if j.get("status") == "precheck"]
        running = [j for j in jobs if j.get("status") == "running"]
        completed = [j for j in jobs if j.get("status") == "completed"]
        failed = [j for j in jobs if j.get("status") == "failed"]
        urgent_pending = [
            j
            for j in jobs
            if j.get("status") in ("pending", "precheck")
            and int(j.get("priority") or 0) >= PRIORITY_URGENT_PEDIGREE_5GEN
        ]

        current_job = running[0] if running else None
        # 並列 process_queue では running が複数。UI は current_jobs を参照する。
        current_jobs = running[:32]
        work_list = self._work_queue_list(jobs)
        active_jobs = running + work_list
        processing_queue = running + work_list
        pending_ordered = self._sorted_pending(jobs)  # 下位互換: pending のみソート

        failed_sorted = sorted(
            [j for j in jobs if j.get("status") == "failed"],
            key=lambda j: (j.get("completed_at") or "", j.get("job_id") or ""),
            reverse=True,
        )[:200]

        pause = read_access_pause()
        transport_paused = bool(pause.get("active"))
        proc_total = len(pending) + len(precheck) + len(running)
        queue_eta = _build_queue_eta(
            n_pending=len(pending) + len(precheck),
            n_running=len(running),
            transport_paused=transport_paused,
            active_jobs=active_jobs,
        )
        return {
            "queue_hourly_maintenance": read_queue_hourly_maintain_state(),
            "is_running": is_running,
            "is_running_urgent": is_running_urgent,
            # キュー process_queue が同時に走らせるジョブ数（環境変数 SCRAPE_QUEUE_PARALLEL）
            "scrape_queue_parallel_workers": _queue_parallel_workers(),
            "current_job": current_job,
            "current_jobs": current_jobs,
            "pending_queue": processing_queue,
            "active_jobs": active_jobs,
            "queue_eta": queue_eta,
            "transport_pause": pause,
            "queue": {
                "pending": len(pending),
                "precheck": len(precheck),
                "running": len(running),
                "completed": len(completed),
                "failed": len(failed),
                "total": len(jobs),
                "urgent_pending": len(urgent_pending),
                "processing_total": proc_total,
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
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
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
                self._save_queue_nolock(jobs)
        if not n and not all_failed and ids:
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
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
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
                self._save_queue_nolock(new_jobs)
        if not removed and not all_failed and ids:
            return 0, "該当する失敗ジョブがありません（ID を確認してください）"
        return removed, None

    def clear_completed(self):
        """完了・失敗したジョブをクリア"""
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            jobs = [j for j in jobs if j.get("status") not in ("completed", "failed")]
            self._save_queue_nolock(jobs)

    def clear_completed_only(self) -> int:
        """status=completed のレコードだけをキューから取り除く。戻り値: 削除件数。"""
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            n = sum(1 for j in jobs if j.get("status") == "completed")
            if n:
                jobs = [j for j in jobs if j.get("status") != "completed"]
                self._save_queue_nolock(jobs)
        return n

    def clear_all_jobs(self) -> int:
        """待機・実行中・完了・失敗を含むジョブをすべて削除する。戻り値: 削除前件数。

        load / save を同一 flock 内で行い、別プロセスのキュー書き込みと競合して
        空にならない事故を防ぐ。
        """
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            n = len(jobs)
            self._save_queue_nolock([])
        return n

    def fail_all_pending_and_running(self, reason: str) -> int:
        """
        HTTP 400 ブロック疑い発生時に呼ぶ。
        pending・running・precheck のジョブをすべて failed に移動し、
        アクセス一時停止フラグを立てる。completed ジョブはそのまま。
        戻り値: 失敗に移動した件数。
        """
        from src.scraper.scrape_access_pause import write_access_pause

        now = datetime.now().isoformat()
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            n = 0
            for job in jobs:
                if job.get("status") in ("pending", "running", "precheck"):
                    job["status"] = "failed"
                    job["completed_at"] = now
                    job["error"] = (reason or "")[:2000]
                    n += 1
            self._save_queue_nolock(jobs)
        write_access_pause(reason=reason)
        logger.warning(
            "HTTP 400 ブロック疑い — pending/running/precheck %d 件を失敗に移動・キューを停止: %s",
            n,
            (reason or "")[:300],
        )
        return n

    def clear_all_jobs_with_transport_pause(
        self, *, reason: str | None = None
    ) -> int:
        """
        短い transport 一時停止のうえで全ジョブを削除し、最後に解除する。
        API ``POST /api/scrape-queue/stop-and-clear`` の本体と同序（ワーカのキックは呼び出し側）。
        """
        from src.scraper.scrape_access_pause import clear_access_pause, write_access_pause

        write_access_pause(
            reason=reason
            or "キュー全消去処理中（完了後に自動解除）"
        )
        try:
            n = self.clear_all_jobs()
            if self.lock_file.exists() and not self.is_locked():
                try:
                    self.lock_file.unlink()
                except OSError:
                    pass
            return n
        finally:
            clear_access_pause()

    def _start_lock_heartbeat(self) -> threading.Event:
        """ロックファイルのmtimeを定期的に更新するハートビートスレッドを起動。"""
        stop_evt = threading.Event()

        def _heartbeat():
            while not stop_evt.wait(timeout=60):
                self.touch_lock()

        t = threading.Thread(target=_heartbeat, daemon=True, name="lock-heartbeat")
        t.start()
        return stop_evt

    def _process_claimed_job(
        self,
        job: dict,
        stop_event: threading.Event,
        *,
        start_delay_sec: float = 0.0,
    ) -> None:
        """claim_next_pending_job 済みのジョブを1件実行（並列ワーカーから呼ばれる）。"""
        if start_delay_sec and start_delay_sec > 0:
            time.sleep(start_delay_sec)

        job_id = job["job_id"]
        label = job.get("job_label") or job.get("target_id") or job.get("race_id")
        max_retries = int(job.get("max_retries", 2))
        retry_count = int(job.get("retry_count", 0))

        logger.info("ジョブ開始: %s (%s)", job_id, label)

        try:
            self._execute_scraping(job)
            self.update_job_status(job_id, "completed")
            logger.info("ジョブ完了: %s", job_id)
            from src.scraper.scrape_access_pause import on_queue_job_completed_successfully

            on_queue_job_completed_successfully()

        except Exception as e:
            from src.scraper.scrape_access_pause import (
                handle_queue_transport_error,
                is_access_or_transport_error,
            )

            if is_access_or_transport_error(e):
                cleared = handle_queue_transport_error(self, e)
                logger.error(
                    "アクセス／ネットワーク系エラー — %s: %s",
                    "キューを自動全消去しました" if cleared else "キューを一時停止しました",
                    job_id,
                    exc_info=True,
                )
                stop_event.set()
                return

            if retry_count < max_retries:
                logger.warning(
                    "ジョブ失敗 (リトライ %d/%d): %s - %s",
                    retry_count + 1, max_retries, job_id, e,
                )
                self._requeue_with_retry(job_id, retry_count + 1)
                time.sleep(2)
            else:
                logger.error("ジョブ失敗 (リトライ上限): %s - %s", job_id, e, exc_info=True)
                self.update_job_status(job_id, "failed", str(e))

        finally:
            self.touch_lock()

    def process_queue(self):
        """
        キューを処理する（バックグラウンドワーカーから呼ばれる）
        SCRAPE_QUEUE_PARALLEL 本まで同時並列（既定 6）。
        """
        from src.scraper.queue_worker_log import ensure_queue_worker_log_handler, mark_queue_worker_active

        ensure_queue_worker_log_handler()

        if not self.acquire_lock():
            logger.info("既にスクレイピング処理が実行中です")
            return

        heartbeat_stop = self._start_lock_heartbeat()
        mark_queue_worker_active(True)
        try:
            self.requeue_stale_running_jobs(assume_lock_holder=True)
            self._run_storage_precheck()

            from src.scraper.scrape_access_pause import read_access_pause

            stop_event = threading.Event()
            workers = _queue_parallel_workers()

            if workers <= 1:
                while True:
                    if stop_event.is_set():
                        break
                    if read_access_pause().get("active"):
                        logger.info("アクセス一時停止中のためキュー処理を中断します")
                        break

                    job = self.claim_next_pending_job()
                    if not job:
                        logger.info("キューが空です")
                        break

                    self._process_claimed_job(job, stop_event)

            else:
                logger.info("キュー並列処理: SCRAPE_QUEUE_PARALLEL=%d", workers)
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="queue-worker",
                ) as executor:
                    futures: list = []
                    try:
                        while True:
                            if stop_event.is_set():
                                break
                            if read_access_pause().get("active"):
                                logger.info(
                                    "アクセス一時停止中のためキュー処理を中断します",
                                )
                                break

                            while (
                                len(futures) < workers
                                and not stop_event.is_set()
                            ):
                                need = workers - len(futures)
                                batch = self.claim_pending_jobs_batch(need)
                                if not batch:
                                    break
                                stagger = _queue_stagger_delay_sec()
                                for i, j in enumerate(batch):
                                    futures.append(
                                        executor.submit(
                                            self._process_claimed_job,
                                            j,
                                            stop_event,
                                            start_delay_sec=(i * stagger),
                                        ),
                                    )

                            if not futures:
                                break

                            done, pending = wait(
                                futures,
                                return_when=FIRST_COMPLETED,
                            )
                            futures = list(pending)
                            for f in done:
                                f.result()
                    finally:
                        for f in futures:
                            try:
                                f.result()
                            except Exception:
                                pass

        finally:
            heartbeat_stop.set()
            mark_queue_worker_active(False)
            self.release_lock()

    # ── ファストレーン（最優先ジョブ専用ワーカー） ──

    def is_locked_urgent(self) -> bool:
        lf = LOCK_FILE_URGENT
        if not lf.exists():
            return False
        try:
            mtime = lf.stat().st_mtime
            if time.time() - mtime > self._LOCK_TIMEOUT:
                logger.warning("古い urgent ロックファイルを削除: %s", lf)
                lf.unlink()
                return False
        except Exception:
            return False
        try:
            raw = lf.read_text(encoding="utf-8")
            data = json.loads(raw)
            pid = int(data.get("pid") or 0)
        except Exception:
            return True
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                lf.unlink()
            except OSError:
                pass
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def acquire_lock_urgent(self) -> bool:
        if self.is_locked_urgent():
            return False
        try:
            with open(LOCK_FILE_URGENT, "w") as f:
                f.write(json.dumps({
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                    "datetime": datetime.now().isoformat(),
                }))
            return True
        except Exception as e:
            logger.error("urgent ロック取得エラー: %s", e)
            return False

    def release_lock_urgent(self):
        try:
            if LOCK_FILE_URGENT.exists():
                LOCK_FILE_URGENT.unlink()
        except Exception as e:
            logger.error("urgent ロック解放エラー: %s", e)

    def get_next_urgent_job(self) -> dict | None:
        """priority >= PRIORITY_URGENT_PEDIGREE_5GEN かつ通常ワーカーが実行していないジョブを返す。"""
        from src.scraper.scrape_access_pause import read_access_pause

        if read_access_pause().get("active"):
            return None
        jobs = self.load_queue()
        pend = [
            j for j in jobs
            if j.get("status") == "pending"
            and int(j.get("priority") or 0) >= PRIORITY_URGENT_PEDIGREE_5GEN
        ]
        pend.sort(key=lambda j: (
            -int(j.get("priority") or 0),
            j.get("queued_at") or "",
        ))
        return pend[0] if pend else None

    def claim_next_urgent_job(self) -> dict | None:
        """最優先待機ジョブを原子的に running にし、そのコピーを返す。"""
        from src.scraper.scrape_access_pause import read_access_pause

        if read_access_pause().get("active"):
            return None
        with _exclusive_queue_json_lock():
            jobs = self._load_queue_nolock()
            pend = [
                j for j in jobs
                if j.get("status") == "pending"
                and int(j.get("priority") or 0) >= PRIORITY_URGENT_PEDIGREE_5GEN
            ]
            pend.sort(key=lambda j: (
                -int(j.get("priority") or 0),
                j.get("queued_at") or "",
            ))
            if not pend:
                return None
            job = pend[0]
            jid = job.get("job_id")
            if not jid:
                return None
            for j in jobs:
                if j.get("job_id") == jid:
                    j["status"] = "running"
                    j["started_at"] = datetime.now().isoformat()
                    break
            self._save_queue_nolock(jobs)
            return copy.deepcopy(job)

    def has_urgent_pending(self) -> bool:
        jobs = self.load_queue()
        return any(
            j.get("status") in ("pending", "precheck")
            and int(j.get("priority") or 0) >= PRIORITY_URGENT_PEDIGREE_5GEN
            for j in jobs
        )

    def process_queue_urgent(self):
        """最優先ジョブだけを処理するファストレーンワーカー。通常ワーカーと並走可能。並列数は SCRAPE_QUEUE_PARALLEL。"""
        from src.scraper.queue_worker_log import ensure_queue_worker_log_handler

        ensure_queue_worker_log_handler()

        if not self.acquire_lock_urgent():
            logger.info("[fast-lane] 既に urgent ワーカーが実行中です")
            return

        try:
            self._run_storage_precheck()
            from src.scraper.scrape_access_pause import read_access_pause

            stop_event = threading.Event()
            workers = _queue_parallel_workers()

            if workers <= 1:
                while True:
                    if stop_event.is_set():
                        break
                    if read_access_pause().get("active"):
                        logger.info("[fast-lane] アクセス一時停止中のため中断します")
                        break

                    job = self.claim_next_urgent_job()
                    if not job:
                        logger.info("[fast-lane] 最優先キューが空です")
                        break

                    self._process_claimed_job(job, stop_event)

            else:
                logger.info(
                    "[fast-lane] 並列処理: SCRAPE_QUEUE_PARALLEL=%d",
                    workers,
                )
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="queue-worker-urgent",
                ) as executor:
                    futures: list = []
                    try:
                        while True:
                            if stop_event.is_set():
                                break
                            if read_access_pause().get("active"):
                                logger.info(
                                    "[fast-lane] アクセス一時停止中のため中断します",
                                )
                                break

                            while (
                                len(futures) < workers
                                and not stop_event.is_set()
                            ):
                                job = self.claim_next_urgent_job()
                                if not job:
                                    break
                                futures.append(
                                    executor.submit(
                                        self._process_claimed_job,
                                        job,
                                        stop_event,
                                    ),
                                )

                            if not futures:
                                break

                            done, pending = wait(
                                futures,
                                return_when=FIRST_COMPLETED,
                            )
                            futures = list(pending)
                            for f in done:
                                f.result()
                    finally:
                        for f in futures:
                            try:
                                f.result()
                            except Exception:
                                pass

        finally:
            self.release_lock_urgent()

    def _execute_scraping(self, job: dict):
        """実際のスクレイピングを実行"""
        from src.scraper.queue_tasks import execute_job
        from src.scraper.run import ScraperRunner

        runner = ScraperRunner()
        execute_job(runner, job)
        # ネット/HTML 未取得（スマート即返等）のジョブは待機0。実取得のときだけ間隔（負荷緩和）
        try:
            th = float(os.environ.get("SCRAPE_QUEUE_THROTTLE_SEC", "0.2"))
        except (TypeError, ValueError):
            th = 0.2
        th = max(0.0, th)
        if (
            th > 0.0
            and getattr(runner, "_queue_mute_scrape_throttle", None) is not True
        ):
            time.sleep(th)


# ---------------------------------------------------------------------------
# 定期メンテ（失敗→待機、完了レコード削除）・ワーカ起動
# ---------------------------------------------------------------------------


def read_queue_hourly_maintain_state() -> dict[str, Any]:
    """直近の定期メンテ結果（queue_hourly_maintain_state.json）。"""
    if not QUEUE_HOURLY_MAINTAIN_STATE.exists():
        return {"available": False}
    try:
        with open(QUEUE_HOURLY_MAINTAIN_STATE, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return {"available": False}
        out = dict(d)
        out["available"] = True
        return out
    except (OSError, json.JSONDecodeError, TypeError):
        return {"available": False}


def _write_queue_hourly_maintain_state(payload: dict[str, Any]) -> None:
    tmp = QUEUE_HOURLY_MAINTAIN_STATE.with_suffix(".json.tmp")
    to_save = {**payload, "written_at": datetime.now().isoformat()}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)
    tmp.replace(QUEUE_HOURLY_MAINTAIN_STATE)


def kick_process_queue_background() -> None:
    """非 asyncio 文脈から: 未処理中なら process_queue を別スレッドで起動。"""

    def _run() -> None:
        try:
            q = ScrapeJobQueue()
            if not q.is_locked():
                q.process_queue()
        except Exception as e:
            logger.warning("kick_process_queue_background: %s", e, exc_info=True)

    threading.Thread(target=_run, daemon=True, name="queue-kick-pq").start()


def kick_urgent_process_queue_background() -> None:
    """非 asyncio 文脈から: 最優先待ちがあれば process_queue_urgent を起動。"""

    def _run() -> None:
        try:
            q = ScrapeJobQueue()
            if q.has_urgent_pending() and not q.is_locked_urgent():
                q.process_queue_urgent()
        except Exception as e:
            logger.warning("kick_urgent_process_queue_background: %s", e, exc_info=True)

    threading.Thread(target=_run, daemon=True, name="queue-kick-urgent").start()


def run_hourly_queue_maintenance() -> dict[str, Any]:
    """
    ストール中（running のまま放置）のジョブを回収し、完了レコードをキューから除去する。
    HTTP 400 ブロックによる一時停止中は failed ジョブを pending に戻さない（ユーザーが
    「再開」ボタンを押すまで failed のまま保持）。
    終了目安 (queue_eta) は get_status 取得時に pending+running から都度再計算される。
    """
    from src.scraper.scrape_access_pause import read_access_pause

    q = ScrapeJobQueue()
    stale_recovered = q.requeue_stale_running_jobs(assume_lock_holder=False)

    # アクセス一時停止中は failed→pending の自動復元をスキップ（ユーザー手動再開待ち）
    pause = read_access_pause()
    if pause.get("active"):
        out: dict[str, Any] = {
            "ok": True,
            "requeued": 0,
            "completed_removed": 0,
            "stale_recovered": stale_recovered,
            "skipped_requeue": True,
            "error": None,
        }
        _write_queue_hourly_maintain_state(out)
        return out

    requeued, err = q.requeue_failed_jobs(all_failed=True)
    if err:
        out = {
            "ok": False,
            "error": err,
            "requeued": 0,
            "completed_removed": 0,
            "stale_recovered": stale_recovered,
        }
        _write_queue_hourly_maintain_state(out)
        return out
    removed = q.clear_completed_only()
    out = {
        "ok": True,
        "requeued": requeued,
        "completed_removed": removed,
        "stale_recovered": stale_recovered,
        "skipped_requeue": False,
        "error": None,
    }
    _write_queue_hourly_maintain_state(out)
    kick_process_queue_background()
    kick_urgent_process_queue_background()
    return out
