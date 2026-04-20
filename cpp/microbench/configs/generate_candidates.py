#!/usr/bin/env python3
import argparse
import itertools
import json
import re
from typing import List


def _parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _platform_threads(platform_id: str) -> int:
    match = re.match(r"(\d+)c", platform_id)
    if not match:
        raise ValueError(f"Could not infer thread count from platform_id '{platform_id}'")
    return int(match.group(1))


def _platform_machine_gib(platform_id: str) -> int:
    match = re.search(r"(\d+)g$", platform_id)
    if not match:
        raise ValueError(f"Could not infer machine memory from platform_id '{platform_id}'")
    return int(match.group(1))


def _legacy_id(
    chunk_size: int,
    prefetch_window: int,
    machine_gib: int,
    static_ratio: float,
    thread_count: int,
) -> str:
    static_gib = int(round(machine_gib * static_ratio))
    buffer_gib = machine_gib - static_gib
    return (
        f"cfg_c{chunk_size}_w{prefetch_window}_"
        f"ms{static_gib}g_mb{buffer_gib}g_t{thread_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate config grid for the ranking model.")
    parser.add_argument("--platform-id", default="", help="Optional platform id like 4c8g or 8c16g")
    parser.add_argument("--total-memory-bytes", type=int, default=0, help="Total usable memory bytes")
    parser.add_argument("--memory-fraction", type=float, default=0.7, help="Usable memory fraction when platform-id is used")
    parser.add_argument("--threads", type=int, default=0, help="Thread count override")
    parser.add_argument("--chunk-sizes", default="64,128,256,512", help="Comma-separated chunk sizes")
    parser.add_argument("--prefetch-windows", default="1", help="Comma-separated prefetch windows")
    parser.add_argument("--static-ratios", default="0,0.25,0.5,0.75", help="Comma-separated static ratios")
    parser.add_argument("--dtype", default="bfloat16", help="Candidate dtype")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    thread_count = args.threads
    total_memory_bytes = args.total_memory_bytes
    machine_gib = 0

    if args.platform_id:
        machine_gib = _platform_machine_gib(args.platform_id)
        if thread_count <= 0:
            thread_count = _platform_threads(args.platform_id)
        if total_memory_bytes <= 0:
            total_memory_bytes = int(args.memory_fraction * machine_gib * 1024**3)

    if thread_count <= 0:
        raise ValueError("Missing thread count. Provide --threads or use --platform-id.")
    if total_memory_bytes <= 0:
        raise ValueError("Missing total memory. Provide --total-memory-bytes or use --platform-id.")
    if machine_gib <= 0:
        machine_gib = int(round(total_memory_bytes / float(1024**3)))

    chunk_sizes = _parse_int_list(args.chunk_sizes)
    prefetch_windows = _parse_int_list(args.prefetch_windows)
    static_ratios = _parse_float_list(args.static_ratios)

    candidates = []
    for chunk_size, prefetch_window, static_ratio in itertools.product(
        chunk_sizes,
        prefetch_windows,
        static_ratios,
    ):
        requested_static_bytes = int(total_memory_bytes * static_ratio)
        bufferpool_bytes = total_memory_bytes - requested_static_bytes
        candidates.append(
            {
                "id": _legacy_id(chunk_size, prefetch_window, machine_gib, static_ratio, thread_count),
                "platform_id": args.platform_id,
                "chunk_size": chunk_size,
                "prefetch_window": prefetch_window,
                "thread_count": thread_count,
                "dtype": args.dtype,
                "static_ratio": static_ratio,
                "requested_static_bytes": requested_static_bytes,
                "selected_static_bytes": requested_static_bytes,
                "static_memory_bytes": requested_static_bytes,
                "bufferpool_bytes": bufferpool_bytes,
            }
        )

    payload = {
        "system": {
            "platform_id": args.platform_id,
            "thread_count": thread_count,
            "total_memory_bytes": total_memory_bytes,
        },
        "candidates": sorted(
            candidates,
            key=lambda c: (c["chunk_size"], c["prefetch_window"], c["static_ratio"]),
        ),
    }

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {len(candidates)} candidates to {args.out}")


if __name__ == "__main__":
    main()
