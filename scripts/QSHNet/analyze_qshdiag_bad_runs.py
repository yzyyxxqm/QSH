#!/usr/bin/env python3
"""Compare QSHDiag trajectories between good and bad iterations."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def load_diag_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_metric_by_itr(results_root: Path) -> dict[int, float]:
    metric_by_itr = {}
    for metric_path in sorted(results_root.glob("iter*/eval_*/metric.json")):
        itr_name = metric_path.parents[1].name
        if not itr_name.startswith("iter"):
            continue
        try:
            itr = int(itr_name.removeprefix("iter"))
        except ValueError:
            continue
        with metric_path.open("r", encoding="utf-8") as file:
            metric = json.load(file)
        if "MSE" in metric:
            metric_by_itr[itr] = float(metric["MSE"])
    return metric_by_itr


def select_bad_itrs(metric_by_itr: dict[int, float], threshold: float | None, top_k: int) -> set[int]:
    if threshold is not None:
        return {itr for itr, mse in metric_by_itr.items() if mse >= threshold}
    sorted_itrs = sorted(metric_by_itr, key=metric_by_itr.get, reverse=True)
    return set(sorted_itrs[:top_k])


def mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def summarize(rows: list[dict[str, str]], metric_by_itr: dict[int, float], bad_itrs: set[int], max_epoch: int):
    excluded = {"source", "itr", "stage", "epoch"}
    numeric_fields = sorted(
        field
        for field in {key for row in rows for key in row}
        if field not in excluded
    )
    by_epoch_group = defaultdict(lambda: {"good": defaultdict(list), "bad": defaultdict(list)})
    for row in rows:
        itr = int(row["itr"])
        epoch = int(row["epoch"])
        if epoch > max_epoch or itr not in metric_by_itr:
            continue
        group = "bad" if itr in bad_itrs else "good"
        for field in numeric_fields:
            by_epoch_group[epoch][group][field].append(parse_float(row.get(field)))

    summaries = []
    for epoch in sorted(by_epoch_group):
        groups = by_epoch_group[epoch]
        for field in numeric_fields:
            good_mean = mean(groups["good"][field])
            bad_mean = mean(groups["bad"][field])
            if math.isnan(good_mean) or math.isnan(bad_mean):
                continue
            summaries.append({
                "epoch": epoch,
                "field": field,
                "good_mean": good_mean,
                "bad_mean": bad_mean,
                "delta_bad_minus_good": bad_mean - good_mean,
            })
    return summaries


def print_metric_table(metric_by_itr: dict[int, float], bad_itrs: set[int]) -> None:
    print("Final MSE by itr:")
    for itr, mse in sorted(metric_by_itr.items()):
        label = "bad" if itr in bad_itrs else "good"
        print(f"  iter{itr}: {mse:.6f} ({label})")


def print_top_differences(summaries: list[dict], top_n: int) -> None:
    print("\nLargest early-epoch QSHDiag differences:")
    ranked = sorted(summaries, key=lambda item: abs(item["delta_bad_minus_good"]), reverse=True)
    for item in ranked[:top_n]:
        print(
            "  epoch={epoch:<3} {field}: good={good_mean:.6g} bad={bad_mean:.6g} delta={delta_bad_minus_good:+.6g}".format(
                **item
            )
        )


def write_summary_csv(summaries: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "field", "good_mean", "bad_mean", "delta_bad_minus_good"]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare QSHDiag rows between good and bad runs.")
    parser.add_argument("--diag_csv", type=Path, required=True, help="CSV produced by extract_qshdiag.py.")
    parser.add_argument("--results_root", type=Path, required=True, help="Run folder containing iter*/eval_*/metric.json.")
    parser.add_argument("--bad_mse_threshold", type=float, default=None, help="Classify itr with MSE >= threshold as bad.")
    parser.add_argument("--bad_top_k", type=int, default=3, help="If no threshold is given, classify top-k MSE itrs as bad.")
    parser.add_argument("--max_epoch", type=int, default=30, help="Only compare early epochs up to this value.")
    parser.add_argument("--top_n", type=int, default=30, help="Number of largest differences to print.")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional detailed comparison CSV.")
    args = parser.parse_args()

    rows = load_diag_rows(args.diag_csv)
    metric_by_itr = load_metric_by_itr(args.results_root)
    bad_itrs = select_bad_itrs(metric_by_itr, args.bad_mse_threshold, args.bad_top_k)
    summaries = summarize(rows, metric_by_itr, bad_itrs, args.max_epoch)

    print_metric_table(metric_by_itr, bad_itrs)
    print_top_differences(summaries, args.top_n)
    if args.output_csv is not None:
        write_summary_csv(summaries, args.output_csv)
        print(f"\nWrote comparison CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
