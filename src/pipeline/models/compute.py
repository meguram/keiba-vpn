"""学習に使う計算資源（CPU スレッド・GPU・RAM）の検出と、各ライブラリ向けパラメータの組み立て。"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Compute:
    threads: int
    gpu: bool
    ram_gb: float


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _available_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def _has_gpu() -> bool:
    if os.environ.get("KEIBA_USE_GPU") == "0":
        return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and "GPU" in r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def detect_compute(reserve_cores: int = 4) -> Compute:
    """reserve_cores 分は他のジョブ（スクレイピング・特徴量生成など）のために空けておく。"""
    return Compute(threads=max(1, _cpu_count() - reserve_cores), gpu=_has_gpu(), ram_gb=_available_ram_gb())


def lgb_params(c: Compute) -> dict:
    # pip 版 LightGBM は GPU 非対応。CPU の全スレッドを使う。
    return {"n_jobs": c.threads}


def xgb_params(c: Compute) -> dict:
    return {"tree_method": "hist", "device": "cuda" if c.gpu else "cpu", "n_jobs": c.threads}


def catboost_params(c: Compute) -> dict:
    # border_count を固定し、CPU/GPU で量子化の既定差が結果に出ないようにする。
    p = {"task_type": "GPU" if c.gpu else "CPU", "thread_count": c.threads, "border_count": 254}
    if c.gpu:
        p["gpu_ram_part"] = 0.8
    return p


def optuna_workers(c: Compute, threads_per_trial: int) -> int:
    """試行を並列に回す本数。1 試行あたりのスレッド数で CPU を使い切るように割る。"""
    return max(1, c.threads // max(1, threads_per_trial))
