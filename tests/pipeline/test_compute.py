from src.pipeline.models import compute
from src.pipeline.models.compute import Compute


def test_threads_reserve_cores(monkeypatch):
    monkeypatch.setattr(compute, "_cpu_count", lambda: 24)
    monkeypatch.setattr(compute, "_has_gpu", lambda: False)
    assert compute.detect_compute(reserve_cores=4).threads == 20
    monkeypatch.setattr(compute, "_cpu_count", lambda: 2)
    assert compute.detect_compute(reserve_cores=4).threads == 1


def test_gpu_env_override(monkeypatch):
    monkeypatch.setenv("KEIBA_USE_GPU", "0")
    assert compute._has_gpu() is False


def test_params_follow_gpu_flag():
    gpu = Compute(threads=20, gpu=True, ram_gb=80)
    cpu = Compute(threads=20, gpu=False, ram_gb=80)
    assert compute.xgb_params(gpu)["device"] == "cuda"
    assert compute.xgb_params(cpu)["device"] == "cpu"
    assert compute.catboost_params(gpu)["task_type"] == "GPU"
    assert compute.catboost_params(cpu)["task_type"] == "CPU"
    assert compute.catboost_params(gpu)["border_count"] == compute.catboost_params(cpu)["border_count"]
    assert compute.lgb_params(gpu) == {"n_jobs": 20}


def test_optuna_workers():
    c = Compute(threads=20, gpu=False, ram_gb=80)
    assert compute.optuna_workers(c, 4) == 5
    assert compute.optuna_workers(c, 64) == 1
