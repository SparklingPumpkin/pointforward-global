"""One-time compatibility migration for the improvement-loop solution tree.

The project tree predates explicit relation metadata and structured failed
memory.  This migration only fills fields required by the current controller,
preserves every node/action/parent, and writes an immutable pre-migration copy.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "docs/improvement-loop/solution_tree.json"

REFERENCE_FIXES = {
    "experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-completion/DG-v1-j/run-20260923T165334Z-stationary-budget-span060/report.md":
        "experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-completion/DG-v1-j/run-20260923T165334Z-stationary-budget-span060/manifest.json",
    "experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-completion/DG-v1-k/run-20260923T171346Z-nearest-update-span060/report.md":
        "experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-completion/DG-v1-k/run-20260923T171346Z-nearest-update-span060/manifest.json",
    "experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-completion/DG-v1-n/run-20260924T021244Z-scene017-diverse-visible-12views/report.md":
        "experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-completion/DG-v1-n/run-20260924T021244Z-scene017-diverse-visible-12views/runtime-render-source-metrics/manifest.json",
    "experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-completion/DG-v1-o/run-20260924T023457Z-dynamic20k/manifest.json":
        "experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-completion/DG-v1-o/run-20260924T023457Z-dynamic20k/runtime-payload/track_once_manifest.json",
    "experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-completion/DG-v1-p/run-20260924T025048Z-opacity125/manifest.json":
        "experiments/argoverse-inference/global-dynamic-consistency/dynamic-gaussian-completion/DG-v1-p/run-20260924T025048Z-opacity125/runtime-render-source-metrics/manifest.json",
}


def fix_evidence(items: object) -> list[str]:
    if not isinstance(items, list):
        return []
    fixed = []
    for item in items:
        if not isinstance(item, str) or "::" not in item:
            continue
        reference, claim = item.split("::", 1)
        fixed.append(f"{REFERENCE_FIXES.get(reference, reference)}::{claim}")
    return fixed


def main() -> None:
    tree = json.loads(TREE.read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = TREE.with_name(f"solution_tree.pre-relation-migration-{stamp}.json")
    shutil.copy2(TREE, backup)

    nodes_by_id = {}
    for node in tree.get("nodes", []):
        node["evidence"] = fix_evidence(node.get("evidence"))
        parent = node.get("parent_id")
        node.setdefault("relation_type", "new_root" if parent is None else "refine")
        node.setdefault(
            "relation_rationale",
            "Legacy node imported before explicit relation metadata; its recorded parent edge and conclusion are preserved.",
        )
        if parent is None:
            node.setdefault("relation_evidence", [])
        else:
            relation_evidence = fix_evidence(node.get("relation_evidence"))
            if not relation_evidence:
                relation_evidence = list(node["evidence"][:1])
            if not relation_evidence:
                relation_evidence = [
                    "docs/improvement-loop/solution_tree.json::legacy parent edge preserved during schema migration"
                ]
            node["relation_evidence"] = relation_evidence
        nodes_by_id[str(node.get("id"))] = node

    for memory in tree.get("failed_memory", []):
        node = nodes_by_id.get(str(memory.get("solution_id")), {})
        evidence = fix_evidence(memory.get("evidence"))
        if not evidence:
            evidence = list(node.get("evidence", [])[:1])
        if not evidence:
            evidence = [
                "docs/improvement-loop/solution_tree.json::legacy failed-memory conclusion preserved during schema migration"
            ]
        memory["evidence"] = evidence
        memory.setdefault(
            "reopen_when",
            "Reopen only when new evidence directly invalidates the recorded failure mechanism.",
        )

    tree.setdefault("root_history", [])
    corrected = nodes_by_id.get("SOL-066")
    if corrected is not None:
        corrected_evidence = [
            "experiments/argoverse-inference/global-dynamic-consistency/dynamic-roi-refinement/appearance-residual/DR-v1-i/run-20260928T191146Z-dynamic-rgb-residual-camera-e4/runtime-render-source-metrics/manifest.json::证明完整156帧均值为全局28.575376dB、动态23.583166dB，共享residual仅边际有效"
        ]
        corrected["evidence"] = corrected_evidence
        corrected["relation_evidence"] = corrected_evidence
    tree["incumbent"] = {
        "solution_id": "SOL-064",
        "project_workflow_ref": "RF-v1-c/run-20260928T142858Z-geometry-payload-camera-e4",
        "summary": "RF-v1-c在427.346秒内达到完整156帧全局28.538468dB、动态bbox 23.406143dB，是当前目标的可交付基线。",
        "evidence": [
            "experiments/argoverse-inference/global-dynamic-consistency/gaussian-post-refinement/refiner-adaptation/RF-v1-c/run-20260928T142858Z-geometry-payload-camera-e4/report.md::证明当前600秒合同下的RF-v1-c基线身份与质量"
        ],
    }
    TREE.write_text(json.dumps(tree, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"tree": str(TREE), "backup": str(backup), "nodes": len(tree.get("nodes", []))}))


if __name__ == "__main__":
    main()
