#!/usr/bin/env python3
"""Summarize exact-composition matrix artifacts without merging unlike scales."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def workload_name(result: dict) -> str:
    if "scenario" in result:
        return result["scenario"]
    return Path(result["dataset_dir"]).name


def flatten(path: Path, artifact: dict) -> list[dict]:
    rows = []
    for row in artifact["rows"]:
        result = row["result"]
        exhaustive_ns = result["exhaustive_latency"]["mean_ns"]
        isolated = row.get("isolated_execution") or {}
        dynamic_probe = isolated.get("dynamic") or {}
        exhaustive_probe = isolated.get("exhaustive") or {}
        rows.append(
            {
                "artifact": path.name,
                "kind": artifact["kind"],
                "workload": workload_name(result),
                "topology": result["topology"],
                "budget": result["exhaustive_after_pulls_per_input"],
                "mismatches": result["ordered_top_k_mismatches"],
                "leaf_work_ratio": result["leaf_work_ratio"],
                "intermediate_work_ratio": result["intermediate_work_ratio"],
                "latency_ratio": (
                    result["dynamic_latency"]["mean_ns"] / exhaustive_ns
                    if exhaustive_ns
                    else None
                ),
                "dynamic_mean_ns": result["dynamic_latency"]["mean_ns"],
                "exhaustive_mean_ns": exhaustive_ns,
                "peak_replay_identities": result["peak_replay_identities"],
                "process_max_rss_bytes": row["process_max_rss_bytes"],
                "isolated_dynamic_max_rss_bytes": dynamic_probe.get(
                    "process_max_rss_bytes"
                ),
                "isolated_exhaustive_max_rss_bytes": exhaustive_probe.get(
                    "process_max_rss_bytes"
                ),
                "flat_order_drift_queries": result["flat_order_drift_queries"],
                "mean_flat_overlap_at_k": result["mean_flat_overlap_at_k"],
                "dynamic_leaf_pulls": result["dynamic_leaf_pulls"],
                "physical_leaf_nodes": result["physical_leaf_nodes"],
                "quality": result.get("quality"),
            }
        )
    return rows


def artifact_summary(name: str, rows: list[dict]) -> dict:
    latency = [row["latency_ratio"] for row in rows if row["latency_ratio"] is not None]
    leaf = [row["leaf_work_ratio"] for row in rows]
    rss = [row["process_max_rss_bytes"] for row in rows if row["process_max_rss_bytes"]]
    dynamic_rss = [
        row["isolated_dynamic_max_rss_bytes"]
        for row in rows
        if row["isolated_dynamic_max_rss_bytes"]
    ]
    exhaustive_rss = [
        row["isolated_exhaustive_max_rss_bytes"]
        for row in rows
        if row["isolated_exhaustive_max_rss_bytes"]
    ]
    isolated_rss_ratios = [
        row["isolated_dynamic_max_rss_bytes"]
        / row["isolated_exhaustive_max_rss_bytes"]
        for row in rows
        if row["isolated_dynamic_max_rss_bytes"]
        and row["isolated_exhaustive_max_rss_bytes"]
    ]
    return {
        "artifact": name,
        "configurations": len(rows),
        "ordered_top_k_mismatches": sum(row["mismatches"] for row in rows),
        "leaf_work_ratio_min": min(leaf),
        "leaf_work_ratio_median": statistics.median(leaf),
        "leaf_work_ratio_max": max(leaf),
        "latency_ratio_min": min(latency),
        "latency_ratio_median": statistics.median(latency),
        "latency_ratio_max": max(latency),
        "peak_replay_identities_max": max(row["peak_replay_identities"] for row in rows),
        "process_max_rss_bytes_max": max(rss) if rss else None,
        "isolated_dynamic_max_rss_bytes_max": max(dynamic_rss) if dynamic_rss else None,
        "isolated_exhaustive_max_rss_bytes_max": (
            max(exhaustive_rss) if exhaustive_rss else None
        ),
        "isolated_dynamic_to_exhaustive_rss_ratio_min": (
            min(isolated_rss_ratios) if isolated_rss_ratios else None
        ),
        "isolated_dynamic_to_exhaustive_rss_ratio_median": (
            statistics.median(isolated_rss_ratios) if isolated_rss_ratios else None
        ),
        "isolated_dynamic_to_exhaustive_rss_ratio_max": (
            max(isolated_rss_ratios) if isolated_rss_ratios else None
        ),
    }


def shared_comparisons(rows: list[dict]) -> list[dict]:
    indexed = {
        (row["artifact"], row["workload"], row["budget"], row["topology"]): row
        for row in rows
    }
    comparisons = []
    for key, shared in indexed.items():
        artifact, workload, budget, topology = key
        if topology != "diamond_shared":
            continue
        unfolded = indexed.get((artifact, workload, budget, "diamond_unfolded"))
        if unfolded is None:
            continue
        comparisons.append(
            {
                "artifact": artifact,
                "workload": workload,
                "budget": budget,
                "shared_to_unfolded_leaf_pulls": (
                    shared["dynamic_leaf_pulls"] / unfolded["dynamic_leaf_pulls"]
                    if unfolded["dynamic_leaf_pulls"]
                    else None
                ),
                "shared_to_unfolded_latency": (
                    shared["dynamic_mean_ns"] / unfolded["dynamic_mean_ns"]
                    if unfolded["dynamic_mean_ns"]
                    else None
                ),
                "shared_replay_identities": shared["peak_replay_identities"],
                "unfolded_replay_identities": unfolded["peak_replay_identities"],
            }
        )
    return sorted(
        comparisons,
        key=lambda row: (row["artifact"], row["workload"], row["budget"] or 0),
    )


def snapshot_quality_cost(rows: list[dict]) -> list[dict]:
    quality_rows = []
    for row in rows:
        quality = row["quality"]
        if row["kind"] != "snapshot" or not quality or not quality["judged_queries"]:
            continue
        quality_rows.append(
            {
                "artifact": row["artifact"],
                "workload": row["workload"],
                "topology": row["topology"],
                "budget": row["budget"],
                "judged_queries": quality["judged_queries"],
                "mean_recall_at_k": quality["mean_recall_at_k"],
                "mean_ndcg_at_k": quality["mean_ndcg_at_k"],
                "dynamic_mean_ns": row["dynamic_mean_ns"],
                "leaf_work_ratio": row["leaf_work_ratio"],
            }
        )
    for row in quality_rows:
        peers = [
            peer
            for peer in quality_rows
            if peer["artifact"] == row["artifact"] and peer["budget"] == row["budget"]
        ]
        row["pareto"] = not any(
            peer["mean_recall_at_k"] >= row["mean_recall_at_k"]
            and peer["mean_ndcg_at_k"] >= row["mean_ndcg_at_k"]
            and peer["dynamic_mean_ns"] <= row["dynamic_mean_ns"]
            and (
                peer["mean_recall_at_k"] > row["mean_recall_at_k"]
                or peer["mean_ndcg_at_k"] > row["mean_ndcg_at_k"]
                or peer["dynamic_mean_ns"] < row["dynamic_mean_ns"]
            )
            for peer in peers
        )
    return sorted(
        quality_rows,
        key=lambda row: (row["artifact"], row["budget"] or 0, row["topology"]),
    )


def markdown(summary: dict) -> str:
    lines = [
        "# Exact composition matrix summary",
        "",
        "Every correctness comparison uses full materialization of the same logical network. "
        "Topology drift is reported separately and is not an execution mismatch.",
        "",
        "## Artifact overview",
        "",
        "| Artifact | Configs | Mismatch | Leaf ratio min/median/max | Dynamic/full latency min/median/max | Max replay | Paired RSS MiB | Isolated RSS D/E MiB | D/E RSS ratio min/median/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["artifacts"]:
        rss = row["process_max_rss_bytes_max"]
        dynamic_rss = row["isolated_dynamic_max_rss_bytes_max"]
        exhaustive_rss = row["isolated_exhaustive_max_rss_bytes_max"]
        isolated_rss = (
            f"{dynamic_rss / 1048576:.1f}/{exhaustive_rss / 1048576:.1f}"
            if dynamic_rss is not None and exhaustive_rss is not None
            else "n/a"
        )
        rss_ratio = (
            f"{row['isolated_dynamic_to_exhaustive_rss_ratio_min']:.2f}/"
            f"{row['isolated_dynamic_to_exhaustive_rss_ratio_median']:.2f}/"
            f"{row['isolated_dynamic_to_exhaustive_rss_ratio_max']:.2f}"
            if row["isolated_dynamic_to_exhaustive_rss_ratio_min"] is not None
            else "n/a"
        )
        lines.append(
            f"| {row['artifact']} | {row['configurations']} | "
            f"{row['ordered_top_k_mismatches']} | "
            f"{row['leaf_work_ratio_min']:.3f}/{row['leaf_work_ratio_median']:.3f}/{row['leaf_work_ratio_max']:.3f} | "
            f"{row['latency_ratio_min']:.2f}/{row['latency_ratio_median']:.2f}/{row['latency_ratio_max']:.2f} | "
            f"{row['peak_replay_identities_max']} | "
            f"{rss / 1048576:.1f} | {isolated_rss} | {rss_ratio} |"
        )

    lines.extend(
        [
            "",
            "## Shared versus unfolded diamond",
            "",
            "Ratios below 1 favor sharing for that metric. Replay is the exact simultaneous identity count.",
            "",
            "| Artifact | Workload | Budget | Shared/unfolded leaf pulls | Shared/unfolded latency | Replay shared/unfolded |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["shared_comparisons"]:
        budget = "none" if row["budget"] is None else row["budget"]
        lines.append(
            f"| {row['artifact']} | {row['workload']} | {budget} | "
            f"{row['shared_to_unfolded_leaf_pulls']:.3f} | "
            f"{row['shared_to_unfolded_latency']:.3f} | "
            f"{row['shared_replay_identities']}/{row['unfolded_replay_identities']} |"
        )

    lines.extend(
        [
            "",
            "## Snapshot topology quality and cost",
            "",
            "Pareto means no topology in the same artifact and execution budget has at least as much Recall and nDCG with no more dynamic latency.",
            "",
            "| Artifact | Topology | Budget | Judged | Recall@K | nDCG@K | Dynamic mean ms | Leaf ratio | Pareto |",
            "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in summary["snapshot_quality_cost"]:
        budget = "none" if row["budget"] is None else row["budget"]
        lines.append(
            f"| {row['artifact']} | {row['topology']} | {budget} | "
            f"{row['judged_queries']} | {row['mean_recall_at_k']:.4f} | "
            f"{row['mean_ndcg_at_k']:.4f} | {row['dynamic_mean_ns'] / 1_000_000:.3f} | "
            f"{row['leaf_work_ratio']:.3f} | {'yes' if row['pareto'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- A lower leaf-work ratio is not a latency guarantee; certification, replay, and iterator overhead remain visible.",
            "- Shared execution is selected only when saved physical work exceeds replay and coordination cost.",
            "- Different topologies are different ranking semantics unless separately proved equivalent.",
            "- Snapshot quality uses the deterministic next-query rewrite stress construction; it is not a production rewrite quality claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    all_rows = []
    summaries = []
    for path in args.inputs:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        rows = flatten(path, artifact)
        all_rows.extend(rows)
        summaries.append(artifact_summary(path.name, rows))
    result = {
        "schema_version": 1,
        "experiment": "spectra-exact-composition-analysis-v1",
        "ordered_top_k_mismatches": sum(row["mismatches"] for row in all_rows),
        "artifacts": summaries,
        "shared_comparisons": shared_comparisons(all_rows),
        "snapshot_quality_cost": snapshot_quality_cost(all_rows),
        "rows": all_rows,
    }
    atomic_json(args.json_output, result)
    atomic_text(args.markdown_output, markdown(result))


if __name__ == "__main__":
    main()
