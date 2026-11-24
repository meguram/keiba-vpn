"""急ぎの最大台数の変更（scripts/gcp/api_revision_clone.py）: 通信 100% の版の設定をそのまま写し、最大台数だけ変え、通信は動かさない。"""
import importlib.util
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[2] / "scripts" / "gcp" / "api_revision_clone.py"
spec = importlib.util.spec_from_file_location("api_revision_clone", P)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def _svc(traffic):
    return {"apiVersion": "serving.knative.dev/v1", "kind": "Service",
            "metadata": {"name": "keiba-api", "namespace": "123", "uid": "x", "resourceVersion": "y",
                         "annotations": {"run.googleapis.com/ingress": "all", "serving.knative.dev/creator": "a"}},
            "spec": {"template": {"metadata": {"name": "keiba-api-00250-kaf"}, "spec": {"containers": [{"image": "stg-image"}]}},
                     "traffic": [{"revisionName": r, "percent": p, "tag": t} for r, p, t in traffic]},
            "status": {"traffic": [{"revisionName": r, "percent": p, "tag": t} for r, p, t in traffic]}}


REV = {"metadata": {"name": "keiba-api-00251-lor",
                    "annotations": {"autoscaling.knative.dev/maxScale": "1", "run.googleapis.com/startup-cpu-boost": "true",
                                    "run.googleapis.com/operation-id": "op", "serving.knative.dev/creator": "a"},
                    "labels": {"serving.knative.dev/service": "keiba-api", "commit": "abc"}},
       "spec": {"containerConcurrency": 30, "timeoutSeconds": 60, "serviceAccountName": "keiba-api@x",
                "containers": [{"image": "asia/keiba:prod-tag", "resources": {"limits": {"cpu": "1", "memory": "4Gi"}},
                                "env": [{"name": "KEIBA_ENV", "value": "prod"}], "startupProbe": {"httpGet": {"path": "/ops/ready"}}}]},
       "status": {"imageDigest": "asia/keiba@sha256:abc"}}


def test_copies_serving_revision_and_changes_only_max_scale():
    svc = _svc([("keiba-api-00250-kaf", 0, "rstg"), ("keiba-api-00251-lor", 100, "rc62a6a0eb")])
    assert C.serving_revision(svc) == "keiba-api-00251-lor"
    out = C.build(svc, REV, 3, "keiba-api-max3-1124")
    t = out["spec"]["template"]
    assert t["metadata"]["name"] == "keiba-api-max3-1124"
    assert t["metadata"]["annotations"]["autoscaling.knative.dev/maxScale"] == "3"
    assert t["metadata"]["annotations"]["run.googleapis.com/startup-cpu-boost"] == "true"
    assert not any(k.startswith(("serving.knative.dev/", "run.googleapis.com/operation-id")) for k in t["metadata"]["annotations"])
    c = t["spec"]["containers"][0]
    assert c["image"] == "asia/keiba@sha256:abc"                        # 本番の版の digest（stg のひな形の image ではない）
    assert c["resources"]["limits"]["memory"] == "4Gi" and c["startupProbe"]["httpGet"]["path"] == "/ops/ready"
    assert out["spec"]["traffic"] == svc["spec"]["traffic"]              # 通信は今のまま（新しい版は 0%）
    assert "uid" not in out["metadata"] and "status" not in out
    assert REV["metadata"]["annotations"]["autoscaling.knative.dev/maxScale"] == "1"     # 元の版の情報は書き換えない


def test_refuses_split_traffic_latest_and_bad_count():
    with pytest.raises(SystemExit):
        C.serving_revision(_svc([("a", 50, None), ("b", 50, None)]))
    svc = _svc([("keiba-api-00251-lor", 100, None)])
    svc["spec"]["traffic"] = [{"latestRevision": True, "percent": 100}]
    with pytest.raises(SystemExit):
        C.build(svc, REV, 3, "n")
    with pytest.raises(SystemExit):
        C.build(_svc([("keiba-api-00251-lor", 100, None)]), REV, 0, "n")
