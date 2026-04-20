#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DTYPE_SIZES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int8": 1,
}

PRIORITY_PREFIXES = [
    "ffn_gate_up_proj",
    "ffn_down_proj",
    "attn_qkv_proj",
    "attn_o_proj",
]


@dataclass(frozen=True)
class CandidateMatrix:
    matrix_id: str
    priority: int
    bytes: int


@dataclass(frozen=True)
class PreloadPlan:
    target_static_bytes: int
    selected_static_bytes: int
    matrix_ids: list[str]
    total_candidate_bytes: int


def _dtype_size(dtype: str) -> int:
    try:
        return DTYPE_SIZES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype in metadata: {dtype}") from exc


def _candidate_priority(matrix_id: str) -> int | None:
    if matrix_id == "output.output_proj":
        return len(PRIORITY_PREFIXES)

    for index, suffix in enumerate(PRIORITY_PREFIXES):
        if matrix_id.startswith("transformer.") and matrix_id.endswith(f".{suffix}"):
            return index
    return None


def load_candidates(metadata_path: Path) -> list[CandidateMatrix]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    candidates: list[CandidateMatrix] = []
    seen_ids: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {metadata_path}") from exc

            matrix_id = item["matrix_id"]
            priority = _candidate_priority(matrix_id)
            if priority is None:
                continue
            if matrix_id in seen_ids:
                raise ValueError(f"duplicate matrix_id in metadata: {matrix_id}")

            rows, cols = item["shape"]
            dtype = item["dtype"]
            size_bytes = int(rows) * int(cols) * _dtype_size(dtype)
            candidates.append(CandidateMatrix(matrix_id=matrix_id, priority=priority, bytes=size_bytes))
            seen_ids.add(matrix_id)

    candidates.sort(key=lambda item: (item.priority, item.matrix_id))
    return candidates


def build_preload_plan(candidates: Iterable[CandidateMatrix], target_static_bytes: int) -> PreloadPlan:
    if target_static_bytes < 0:
        raise ValueError("target_static_bytes must be non-negative")

    selected_ids: list[str] = []
    selected_bytes = 0
    candidate_list = list(candidates)

    for candidate in candidate_list:
        if selected_bytes + candidate.bytes > target_static_bytes:
            break
        selected_ids.append(candidate.matrix_id)
        selected_bytes += candidate.bytes

    total_candidate_bytes = sum(candidate.bytes for candidate in candidate_list)
    return PreloadPlan(
        target_static_bytes=target_static_bytes,
        selected_static_bytes=selected_bytes,
        matrix_ids=selected_ids,
        total_candidate_bytes=total_candidate_bytes,
    )


def write_preload_file(output_path: Path, matrix_ids: Iterable[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(matrix_ids)
    if text:
        text += "\n"
    output_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic Llama preload file from metadata.jsonl.",
    )
    parser.add_argument("--db-path", required=True, help="Path to the generated SimpleDB directory")
    parser.add_argument(
        "--target-static-bytes",
        required=True,
        type=int,
        help="Requested static-resident byte budget",
    )
    parser.add_argument("--out", required=True, help="Output preload text file path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.db_path) / "metadata.jsonl"
    candidates = load_candidates(metadata_path)
    plan = build_preload_plan(candidates, args.target_static_bytes)
    write_preload_file(Path(args.out), plan.matrix_ids)

    payload = {
        "db_path": str(Path(args.db_path).resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "target_static_bytes": plan.target_static_bytes,
        "selected_static_bytes": plan.selected_static_bytes,
        "total_candidate_bytes": plan.total_candidate_bytes,
        "matrix_count": len(plan.matrix_ids),
        "matrix_ids": plan.matrix_ids,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
