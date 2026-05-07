#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "diffusion_policy"))

from BACInfer.analysis.run_policy import run_policy


def _l1_distance_per_step(trace_a, trace_b):
    n = min(len(trace_a), len(trace_b))
    xs = []
    ys = []
    for i in range(n):
        a = trace_a[i].float()
        b = trace_b[i].float()
        ys.append(float((a - b).abs().mean().item()))
        xs.append(i)
    return np.array(xs), np.array(ys, dtype=np.float32)


def _proxy_curve_from_block_trace(block_trace):
    xs = []
    ys = []
    for step in block_trace:
        blocks = step.get("blocks", {})
        scores = [
            float(info["score"]) for info in blocks.values()
            if info.get("score") is not None
        ]
        if not scores:
            continue
        xs.append(int(step["step_index"]))
        ys.append(float(np.mean(scores)))
    return np.array(xs), np.array(ys, dtype=np.float32)


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--demo_idx", default=0, type=int)
@click.option("--skip_score_threshold", default=0.03, type=float)
@click.option("--skip_max_stale", default=4, type=int)
@click.option("--skip_min_progress", default=0.0, type=float)
@click.option("--online_block_gate_config", default=None)
@click.option("--random_seed", default=11, type=int)
def main(
    checkpoint,
    output_dir,
    device,
    demo_idx,
    skip_score_threshold,
    skip_max_stale,
    skip_min_progress,
    online_block_gate_config,
    random_seed,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_policy, _, _ = run_policy(
        checkpoint=checkpoint,
        output_dir=str(output_dir / "original_run"),
        device=device,
        demo_idx=demo_idx,
        cache_mode="original",
        skip_score_threshold=skip_score_threshold,
        skip_max_stale=skip_max_stale,
        skip_min_progress=skip_min_progress,
        return_obs_action=True,
        random_seed=random_seed,
    )
    cached_policy, _, _ = run_policy(
        checkpoint=checkpoint,
        output_dir=str(output_dir / "cached_run"),
        device=device,
        demo_idx=demo_idx,
        cache_mode="hybrid",
        skip_score_threshold=skip_score_threshold,
        skip_max_stale=skip_max_stale,
        skip_min_progress=skip_min_progress,
        online_block_gate_config=online_block_gate_config,
        return_obs_action=True,
        random_seed=random_seed,
    )

    gt_trace = list(getattr(original_policy, "_model_output_trace", []))
    cached_trace = list(getattr(cached_policy, "_model_output_trace", []))
    xs, l1_curve = _l1_distance_per_step(gt_trace, cached_trace)

    block_trace = list(getattr(cached_policy, "_cache", {}).get("block_trace", []))
    proxy_xs, proxy_curve = _proxy_curve_from_block_trace(block_trace)

    fig, ax1 = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    ax1.plot(xs, l1_curve, color="#2b6cb0", marker="o", linewidth=2.0, label="Ground-truth vs Cached L1 Distance")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("L1 Distance")
    ax1.grid(True, alpha=0.35)

    if len(proxy_xs) > 0:
        ax2 = ax1.twinx()
        ax2.plot(proxy_xs, proxy_curve, color="#dd6b20", marker="o", linewidth=1.8, alpha=0.9, label="Mean Block Cosine Similarity")
        ax2.set_ylabel("Mean Block Cosine Similarity")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    else:
        ax1.legend(loc="upper center")

    fig.savefig(output_dir / "error_curve.png", dpi=220)
    plt.close(fig)

    payload = {
        "checkpoint": checkpoint,
        "demo_idx": demo_idx,
        "skip_score_threshold": skip_score_threshold,
        "skip_max_stale": skip_max_stale,
        "skip_min_progress": skip_min_progress,
        "online_block_gate_config": online_block_gate_config,
        "l1_curve": l1_curve.tolist(),
        "proxy_curve": proxy_curve.tolist(),
        "step_indices": xs.tolist(),
        "proxy_step_indices": proxy_xs.tolist(),
    }
    with open(output_dir / "error_curve.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "output_dir": str(output_dir),
        "num_steps": int(len(xs)),
        "mean_l1": float(np.mean(l1_curve)) if len(l1_curve) else None,
        "max_l1": float(np.max(l1_curve)) if len(l1_curve) else None,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
