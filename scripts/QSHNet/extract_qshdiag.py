#!/usr/bin/env python3
"""Extract QSHDiag lines from training logs into a CSV table."""

import argparse
import csv
import re
import sys
from pathlib import Path


HEADER_RE = re.compile(r"\[QSHDiag\]\[itr=(?P<itr>\d+)\]\[stage=(?P<stage>\d+)\]\[epoch=(?P<epoch>\d+)\]")
KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(-?nan|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)", re.IGNORECASE)
LAYER_RE = re.compile(r"\bL(?P<layer>\d+)\s+")


def parse_qshdiag_line(line: str) -> dict[str, str] | None:
    if "[QSHDiag]" not in line:
        return None

    header = HEADER_RE.search(line)
    if header is None:
        return None

    row = header.groupdict()
    active_layer = ""
    for chunk in line.split("|"):
        layer_match = LAYER_RE.search(chunk)
        if layer_match is not None:
            active_layer = f"L{layer_match.group('layer')}_"

        for match in KV_RE.finditer(chunk):
            key = match.group("key")
            value = match.group(0).split("=", 1)[1]
            if key in {"itr", "stage", "epoch"}:
                continue
            if active_layer and not key.startswith("L"):
                key = active_layer + key
            row[key] = value
    return row


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                row = parse_qshdiag_line(line)
                if row is not None:
                    row["source"] = str(path)
                    rows.append(row)
    return rows


def numeric_sort_key(name: str) -> tuple[int, str]:
    priority = {
        "source": 0,
        "itr": 1,
        "stage": 2,
        "epoch": 3,
        "train_loss": 4,
        "vali_loss": 5,
        "lr": 6,
    }
    return priority.get(name, 100), name


def write_csv(rows: list[dict[str, str]], output: Path | None) -> None:
    if not rows:
        print("No QSHDiag rows found.", file=sys.stderr)
        return

    fieldnames = sorted({key for row in rows for key in row}, key=numeric_sort_key)
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract QSHDiag lines from one or more log files.")
    parser.add_argument("logs", type=Path, nargs="+", help="Log files produced by running main.py with tee.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    rows = load_rows(args.logs)
    write_csv(rows, args.output)
    if args.output is not None:
        print(f"Wrote {len(rows)} QSHDiag rows to {args.output}")


if __name__ == "__main__":
    main()
