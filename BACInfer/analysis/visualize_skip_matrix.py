#!/usr/bin/env python3

import json
import os
import re
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "diffusion_policy"))

from BACInfer.analysis.run_policy import run_policy


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TMP_DIR = REPO_ROOT / "tmp"


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown"


def _resolve_output_dir(output_dir, checkpoint, cache_mode, demo_idx):
    if output_dir:
        return Path(output_dir)

    checkpoint_path = Path(checkpoint)
    task_name = checkpoint_path.parents[3].name if len(checkpoint_path.parents) >= 4 else checkpoint_path.stem
    mode_name = _safe_name(cache_mode or "original")
    return DEFAULT_TMP_DIR / _safe_name(task_name) / f"skip_matrix_{mode_name}_demo_{demo_idx}"


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _build_step_matrix(skip_trace):
    if not skip_trace:
        return np.zeros((1, 0), dtype=np.float32)
    return np.array(
        [[1.0 if step.get("used_cache", False) else 0.0 for step in skip_trace]],
        dtype=np.float32,
    )


def _build_block_reuse_matrix(block_trace, block_keys, num_steps):
    if not block_keys:
        return np.zeros((0, num_steps), dtype=np.float32)

    matrix = np.full((len(block_keys), num_steps), np.nan, dtype=np.float32)
    block_index = {block_key: i for i, block_key in enumerate(block_keys)}
    for step_entry in block_trace:
        step_idx = int(step_entry.get("step_index", -1))
        if step_idx < 0 or step_idx >= num_steps:
            continue
        for block_key, info in step_entry.get("blocks", {}).items():
            row = block_index[block_key]
            matrix[row, step_idx] = 1.0 if info.get("reused", False) else 0.0
    return matrix


def _block_type_from_key(block_key):
    if block_key.endswith("_sa_block"):
        return "sa_block"
    if block_key.endswith("_mha_block"):
        return "mha_block"
    if block_key.endswith("_ff_block"):
        return "ff_block"
    return "unknown"


def _slice_block_matrix_by_type(block_reuse_matrix, block_keys):
    result = {}
    for block_type in ("sa_block", "mha_block", "ff_block"):
        indices = [i for i, key in enumerate(block_keys) if _block_type_from_key(key) == block_type]
        keys = [block_keys[i] for i in indices]
        if not indices:
            result[block_type] = {
                "matrix": np.zeros((0, block_reuse_matrix.shape[1]), dtype=np.float32),
                "keys": [],
            }
        else:
            result[block_type] = {
                "matrix": block_reuse_matrix[indices, :],
                "keys": keys,
            }
    return result


def _save_heatmap(data, output_path, title, xlabel, ylabel, yticklabels=None, nan_color="#d9d9d9"):
    fig_height = 2.4 if data.shape[0] <= 2 else min(18.0, max(4.0, 0.32 * data.shape[0]))
    fig_width = max(8.0, min(24.0, 0.45 * max(1, data.shape[1])))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color=nan_color)

    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([str(i) for i in range(data.shape[1])], rotation=90, fontsize=8)

    if yticklabels is None:
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels([str(i) for i in range(data.shape[0])], fontsize=8)
    else:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("1=reuse/skip, 0=compute/update")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_combined_figure(step_matrix, block_matrix, block_keys, output_path):
    block_rows = max(1, block_matrix.shape[0])
    fig_height = 2.5 + min(16.0, 0.28 * block_rows)
    fig_width = max(9.0, min(26.0, 0.45 * max(1, step_matrix.shape[1])))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": [1, max(1, block_rows)]},
        constrained_layout=True,
    )

    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9")

    step_im = axes[0].imshow(step_matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].set_title("Step Skip Matrix")
    axes[0].set_ylabel("step")
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(["skip"])
    axes[0].set_xticks(np.arange(step_matrix.shape[1]))
    axes[0].set_xticklabels([])
    fig.colorbar(step_im, ax=axes[0], fraction=0.03, pad=0.02)

    if block_matrix.size == 0:
        axes[1].text(0.5, 0.5, "No block trace", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        block_im = axes[1].imshow(block_matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        axes[1].set_title("Block Reuse Matrix")
        axes[1].set_xlabel("diffusion step index")
        axes[1].set_ylabel("block")
        axes[1].set_xticks(np.arange(block_matrix.shape[1]))
        axes[1].set_xticklabels([str(i) for i in range(block_matrix.shape[1])], rotation=90, fontsize=8)
        axes[1].set_yticks(np.arange(len(block_keys)))
        axes[1].set_yticklabels(block_keys, fontsize=7)
        fig.colorbar(block_im, ax=axes[1], fraction=0.03, pad=0.02)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", default=None)
@click.option("-d", "--device", default="cuda:0")
@click.option("--demo_idx", default=0, type=int)
@click.option("--cache_mode", default="hybrid", type=click.Choice(["original", "skip", "adaptive", "hybrid"]))
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
    cache_mode,
    skip_score_threshold,
    skip_max_stale,
    skip_min_progress,
    online_block_gate_config,
    random_seed,
):
    output_dir = _resolve_output_dir(output_dir, checkpoint, cache_mode, demo_idx)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, obs_dict, action_dict = run_policy(
        checkpoint=checkpoint,
        output_dir=str(output_dir),
        device=device,
        demo_idx=demo_idx,
        cache_mode=cache_mode,
        skip_score_threshold=skip_score_threshold,
        skip_max_stale=skip_max_stale,
        skip_min_progress=skip_min_progress,
        online_block_gate_config=online_block_gate_config,
        return_obs_action=True,
        random_seed=random_seed,
    )
    del obs_dict
    del action_dict

    skip_trace = list(getattr(policy, "_skip_trace", []))
    skip_stats = dict(getattr(policy, "_skip_stats", {}))
    cache = getattr(policy, "_cache", {})
    block_trace = list(cache.get("block_trace", []))
    block_keys = list(cache.get("block_keys", []))

    num_steps = len(skip_trace)
    step_matrix = _build_step_matrix(skip_trace)
    block_reuse_matrix = _build_block_reuse_matrix(block_trace, block_keys, num_steps)
    block_update_matrix = np.where(np.isnan(block_reuse_matrix), np.nan, 1.0 - block_reuse_matrix)

    np.save(output_dir / "step_skip_matrix.npy", step_matrix)
    np.save(output_dir / "block_reuse_matrix.npy", block_reuse_matrix)
    np.save(output_dir / "block_update_matrix.npy", block_update_matrix)

    raw_trace = {
        "cache_mode": cache_mode,
        "checkpoint": checkpoint,
        "device": device,
        "demo_idx": demo_idx,
        "skip_score_threshold": skip_score_threshold,
        "skip_max_stale": skip_max_stale,
        "skip_min_progress": skip_min_progress,
        "online_block_gate_config": online_block_gate_config,
        "skip_stats": skip_stats,
        "skip_trace": skip_trace,
        "block_keys": block_keys,
        "block_trace": block_trace,
    }
    with open(output_dir / "skip_traces.json", "w", encoding="utf-8") as f:
        json.dump(_to_serializable(raw_trace), f, indent=2, ensure_ascii=False)

    _save_heatmap(
        step_matrix,
        output_dir / "step_skip_matrix.png",
        title="Step Skip Matrix",
        xlabel="diffusion step index",
        ylabel="row",
        yticklabels=["skip"],
    )
    _save_heatmap(
        block_reuse_matrix,
        output_dir / "block_reuse_matrix.png",
        title="Block Reuse Matrix",
        xlabel="diffusion step index",
        ylabel="block",
        yticklabels=block_keys,
    )
    by_type = _slice_block_matrix_by_type(block_reuse_matrix, block_keys)
    for block_type, payload in by_type.items():
        np.save(output_dir / f"{block_type}_reuse_matrix.npy", payload["matrix"])
        _save_heatmap(
            payload["matrix"],
            output_dir / f"{block_type}_reuse_matrix.png",
            title=f"{block_type} Reuse Matrix",
            xlabel="diffusion step index",
            ylabel="block",
            yticklabels=payload["keys"],
        )
    _save_combined_figure(
        step_matrix,
        block_reuse_matrix,
        block_keys,
        output_dir / "skip_matrix_combined.png",
    )

    summary = {
        "num_diffusion_steps": num_steps,
        "num_step_skips": int(sum(1 for step in skip_trace if step.get("used_cache", False))),
        "num_step_computes": int(sum(1 for step in skip_trace if step.get("computed", False))),
        "num_blocks": len(block_keys),
        "num_block_trace_steps": len(block_trace),
        "step_skip_ratio": 0.0 if num_steps == 0 else float(np.nanmean(step_matrix)),
        "block_reuse_ratio": None if block_reuse_matrix.size == 0 else float(np.nanmean(block_reuse_matrix)),
        "block_reuse_ratio_by_type": {
            block_type: (
                None if payload["matrix"].size == 0 else float(np.nanmean(payload["matrix"]))
            )
            for block_type, payload in by_type.items()
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(_to_serializable({
        "output_dir": str(output_dir),
        "summary": summary,
    }), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
