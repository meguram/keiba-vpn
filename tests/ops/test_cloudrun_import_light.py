"""Cloud Run の API イメージには mlflow / torch を入れない（deploy/docker/requirements-cloudrun.txt）。
API のハンドラが使うサービスのモジュールが、それらが無い環境でも import でき、事前計算の読み出しが動くこと。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_CODE = r'''
import importlib, sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("mlflow", "torch"):
            raise ModuleNotFoundError(f"No module named '{name}'")
sys.meta_path.insert(0, Block())
for m in ("src.pipeline.inference.tracking_difficulty_service", "src.pipeline.mlflow", "src.utils.mlflow_client"):
    importlib.import_module(m)
from src.pipeline.inference.tracking_difficulty_service import get_or_compute
r = get_or_compute(None, "NO_SUCH_RACE_FOR_TEST")
assert r["status"] == "not_precomputed", r
print("ok")
'''


def test_tracking_difficulty_reads_precomputed_without_mlflow_or_torch():
    r = subprocess.run([sys.executable, "-c", _CODE], cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "KEIBA_ENV": "prod", "PATH": "/usr/bin:/bin", "HOME": "/tmp"})
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-1500:]
