#!/usr/bin/env python3

import json
import math
from collections import defaultdict
from pathlib import Path

import click


FORMULAS = {
    "delta_last_update_rms_over_input_rms": lambda r: _safe_div(r.get("delta_last_update_rms"), r.get("input_rms")),
    "delta_prev_rms_over_input_rms": lambda r: _safe_div(r.get("delta_prev_rms"), r.get("input_rms")),
    "delta_last_update_l2_over_input_l2": lambda r: _safe_div(r.get("delta_last_update_l2_norm"), r.get("input_l2_norm")),
    "delta_prev_l2_over_input_l2": lambda r: _safe_div(r.get("delta_prev_l2_norm"), r.get("input_l2_norm")),
    "delta_last_update_rms": lambda r: _to_float(r.get("delta_last_update_rms")),
    "delta_prev_rms": lambda r: _to_float(r.get("delta_prev_rms")),
}


def _to_float(x):
    if x is None:
        return None
    return float(x)


def _safe_div(a, b, eps=1e-12):
    if a is None or b is None:
        return None
    b = float(b)
    if abs(b) <= eps:
        return None
    return float(a) / b


def _load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _balanced_accuracy(records, formula_name, threshold):
    tp = tn = fp = fn = 0
    for r in records:
        score = FORMULAS[formula_name](r)
        if score is None:
            continue
        pred_update = score >= threshold
        label_update = bool(r["label_update"])
        if pred_update and label_update:
            tp += 1
        elif pred_update and not label_update:
            fp += 1
        elif (not pred_update) and label_update:
            fn += 1
        else:
            tn += 1
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr), {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "tpr": tpr, "tnr": tnr}


def _candidate_thresholds(records, formula_name):
    values = []
    for r in records:
        score = FORMULAS[formula_name](r)
        if score is not None and math.isfinite(score):
            values.append(float(score))
    values = sorted(set(values))
    return values


def _fit_best_threshold(records, formula_name):
    best = None
    for thr in _candidate_thresholds(records, formula_name):
        ba, cm = _balanced_accuracy(records, formula_name, thr)
        item = {
            "formula": formula_name,
            "threshold": float(thr),
            "balanced_accuracy": float(ba),
            "confusion": cm,
        }
        if best is None or item["balanced_accuracy"] > best["balanced_accuracy"]:
            best = item
    return best


def _bucket_cache_age(age):
    if age is None:
        return "none"
    age = int(age)
    if age <= 1:
        return "1"
    if age <= 3:
        return "2-3"
    if age <= 6:
        return "4-6"
    return "7+"


def _summarize_by_group(records, formula_name, threshold, group_key):
    groups = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)
    summary = {}
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ba, cm = _balanced_accuracy(items, formula_name, threshold)
        summary[str(key)] = {
            "num_records": len(items),
            "balanced_accuracy": float(ba),
            **cm,
        }
    return summary


@click.command()
@click.option("--trace_dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", default=None, type=click.Path(path_type=Path))
def main(trace_dir: Path, output: Path | None):
    gate_path = trace_dir / "all_gate_feature_records.jsonl"
    records = list(_load_jsonl(gate_path))

    by_block_type = defaultdict(list)
    for r in records:
        if r.get("had_cache_before") is False:
            continue
        if r.get("label_update") is None:
            continue
        by_block_type[r["block_type"]].append(r)

    report = {
        "trace_dir": str(trace_dir),
        "num_records_total": len(records),
        "num_records_effective": int(sum(len(v) for v in by_block_type.values())),
        "results_by_block_type": {},
        "recommended_gate": {
            "mode": "feature_distance",
            "default_by_type": {},
            "blocks": {},
        },
    }

    for block_type, items in sorted(by_block_type.items()):
        fits = []
        for formula_name in FORMULAS:
            best = _fit_best_threshold(items, formula_name)
            if best is not None:
                fits.append(best)
        fits.sort(key=lambda x: x["balanced_accuracy"], reverse=True)
        best = fits[0]
        report["results_by_block_type"][block_type] = {
            "num_records": len(items),
            "num_update": int(sum(int(r["label_update"]) for r in items)),
            "num_skip": int(sum(int(r["label_skip"]) for r in items)),
            "best_formula": best,
            "all_formulas": fits,
            "by_layer": _summarize_by_group(
                items, best["formula"], best["threshold"], lambda r: r["layer_name"]
            ),
            "by_cache_age_bucket": _summarize_by_group(
                items, best["formula"], best["threshold"], lambda r: _bucket_cache_age(r.get("cache_age"))
            ),
        }

        report["recommended_gate"]["default_by_type"][block_type] = {
            "score_formula": best["formula"].replace("_rms_", "_l2_") if best["formula"].endswith("_over_input_rms") else best["formula"],
            "threshold": float(best["threshold"]),
            "max_age": 6,
            "eps": 1e-6,
        }

    if output is None:
        output = trace_dir / "offline_block_score_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    gate_out = output.with_name("offline_block_score_recommended_gate.json")
    with open(gate_out, "w", encoding="utf-8") as f:
        json.dump(report["recommended_gate"], f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "report": str(output),
        "recommended_gate": str(gate_out),
        "block_types": list(report["results_by_block_type"].keys()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
