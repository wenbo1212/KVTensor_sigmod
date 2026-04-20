#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CopySpec:
    source: Path
    target: Path
    required: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def evaluation_root() -> Path:
    return repo_root() / "evaluation"


def copy_if_present(spec: CopySpec) -> bool:
    if not spec.source.is_file():
        if spec.required:
            raise FileNotFoundError(f"required source table not found: {spec.source}")
        return False
    spec.target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.source, spec.target)
    return True


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_merged63(section63_path: Path, preload_path: Path, output_path: Path) -> None:
    section_rows = load_csv(section63_path)
    preload_rows = load_csv(preload_path)
    preload_index: dict[tuple[str, str, str], dict[str, str]] = {}

    for row in preload_rows:
        key = (row["platform_id"], row["chunk_size"], row["static_ratio"])
        preload_index[key] = row

    merged_rows: list[dict[str, object]] = []
    for row in section_rows:
        key = (row["platform_id"], row["chunk_size"], row["static_ratio"])
        preload_row = preload_index.get(key)
        preload_s = float(preload_row["median_preload_s"]) if preload_row else 0.0
        prefill_s = float(row["median_prefill_s"])
        merged = dict(row)
        merged["median_preload_s"] = preload_s
        merged["real_prefill_time"] = max(prefill_s - preload_s, 0.0)
        merged_rows.append(merged)

    fieldnames = list(section_rows[0].keys()) + ["median_preload_s", "real_prefill_time"]
    write_csv(output_path, fieldnames, merged_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy experiment summary tables into evaluation/ and rebuild merged63.csv.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if an expected source table is missing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    eval_root = evaluation_root()

    specs = [
        CopySpec(
            source=root / "experiment" / "results" / "derived" / "section63_summary.csv",
            target=eval_root / "section63.csv",
            required=args.strict,
        ),
        CopySpec(
            source=root / "experiment" / "results" / "preload_cost" / "derived" / "preload_cost_summary.csv",
            target=eval_root / "preload.csv",
            required=args.strict,
        ),
        CopySpec(
            source=root / "experiment" / "results" / "stable_diffusion" / "stable_diffusion_summary.csv",
            target=eval_root / "diffusion.csv",
            required=args.strict,
        ),
    ]

    copied = 0
    for spec in specs:
        if copy_if_present(spec):
            copied += 1
            print(f"copied {spec.source} -> {spec.target}")
        else:
            print(f"skipped missing source: {spec.source}")

    section63_path = eval_root / "section63.csv"
    preload_path = eval_root / "preload.csv"
    if section63_path.is_file() and preload_path.is_file():
        build_merged63(section63_path, preload_path, eval_root / "merged63.csv")
        print(f"rebuilt {eval_root / 'merged63.csv'}")
    elif args.strict:
        raise FileNotFoundError("cannot rebuild merged63.csv without both section63.csv and preload.csv")
    else:
        print("skipped merged63 rebuild because section63.csv or preload.csv is missing")

    print(f"completed sync; copied {copied} table(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
