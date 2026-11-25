#!/usr/bin/env python3
"""本番で通信 100% の版と同じ設定（イメージの digest・環境・秘密・資源・起動プローブ・同時数）で、最大台数だけを変えた版の
Knative の manifest（gcloud run services replace に渡す JSON）を作る。通信の割り当ては今のまま（新しい版は 0%）。

``gcloud run services update`` は「最後に作られた版」のひな形から新しい版を作る。stg（rstg）の配備の後だと、それは stg の版
（テストユーザー・2Gi・未昇格のイメージ）になり、本番に載せてしまう（2026-11-24 のレビュー。2026-11-05 の事故と同じ形）。そのため使わない。

使い方（scripts/gcp/set_api_max_instances.sh から呼ぶ）:
  python3 api_revision_clone.py <service.json> <revision.json> <最大台数> <新しい版の名前>  → 標準出力に manifest（JSON）
"""
from __future__ import annotations

import copy
import json
import sys

# 版のひな形に写さない注記・ラベル（Cloud Run が付けるもの・作った人の記録）
_DROP_PREFIXES = ("serving.knative.dev/", "run.googleapis.com/operation-id", "run.googleapis.com/client-", "client.knative.dev/")


def _keep(d: dict | None) -> dict:
    return {k: v for k, v in (d or {}).items() if not k.startswith(_DROP_PREFIXES)}


def serving_revision(service: dict) -> str:
    """通信 100% の版の名前。100% の版が無い（分けている）ときは止める。"""
    full = [t["revisionName"] for t in service.get("status", {}).get("traffic", []) if int(t.get("percent") or 0) == 100]
    if len(full) != 1:
        raise SystemExit(f"通信 100% の版が 1 つではありません: {full}（分けて流している間は使わない）")
    return full[0]


def build(service: dict, revision: dict, max_instances: int, new_name: str) -> dict:
    if not (1 <= max_instances <= 10):
        raise SystemExit("最大台数は 1〜10")
    rev_spec = copy.deepcopy(revision["spec"])
    digest = revision.get("status", {}).get("imageDigest")
    if not digest:
        raise SystemExit("版のイメージの digest が読めません")
    rev_spec["containers"][0]["image"] = digest          # 同じイメージを digest で指す（タグが動いても変わらない）
    ann = _keep(revision["metadata"].get("annotations"))
    ann["autoscaling.knative.dev/maxScale"] = str(max_instances)
    template = {"metadata": {"name": new_name, "labels": _keep(revision["metadata"].get("labels")), "annotations": ann}, "spec": rev_spec}
    meta = service["metadata"]
    out = {
        "apiVersion": service.get("apiVersion", "serving.knative.dev/v1"),
        "kind": "Service",
        "metadata": {"name": meta["name"], "namespace": meta.get("namespace"), "labels": _keep(meta.get("labels")),
                     "annotations": _keep(meta.get("annotations"))},
        "spec": {"template": template, "traffic": copy.deepcopy(service["spec"]["traffic"])},     # 通信は今のまま
    }
    if any(t.get("latestRevision") for t in out["spec"]["traffic"]):
        raise SystemExit("通信が「最新の版」に向いています。新しい版を作ると、その場で通信が移るので止めます")
    return out


def main(argv: list[str]) -> int:
    svc = json.load(open(argv[1], encoding="utf-8"))
    rev = json.load(open(argv[2], encoding="utf-8"))
    if rev["metadata"]["name"] != serving_revision(svc):
        raise SystemExit("渡された版が、通信 100% の版ではありません")
    json.dump(build(svc, rev, int(argv[3]), argv[4]), sys.stdout, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
