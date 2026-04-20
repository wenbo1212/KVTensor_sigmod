#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import statistics
import subprocess
from pathlib import Path

from generate_preload_file import build_preload_plan, load_candidates, write_preload_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiment" / "results" / "preload_cost"
PLATFORM_MEMORY_FRACTION = 0.7

PLATFORM_BYTES = {
    "4c8g": int(PLATFORM_MEMORY_FRACTION * 8 * 1024**3),
    "8c16g": int(PLATFORM_MEMORY_FRACTION * 16 * 1024**3),
}

RAW_FIELDNAMES = [
    "run_id",
    "platform_id",
    "threads",
    "chunk_size",
    "static_ratio",
    "requested_static_bytes",
    "selected_static_bytes",
    "matrix_count",
    "repeat_idx",
    "preload_s",
    "throughput_mb_s",
    "db_path",
    "preload_file",
    "report_json",
]

SUMMARY_FIELDNAMES = [
    "platform_id",
    "threads",
    "chunk_size",
    "static_ratio",
    "requested_static_bytes",
    "selected_static_bytes",
    "matrix_count",
    "median_preload_s",
    "median_throughput_mb_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure real preload time by invoking llama_inference_example in preload-only mode.",
    )
    parser.add_argument("--build-dir", default=str(REPO_ROOT / "cpp" / "build"))
    parser.add_argument("--db-root", default=str(REPO_ROOT / "experiment" / "results" / "raw" / "dbs"))
    parser.add_argument("--chunk-sizes", default="64,128,256,512")
    parser.add_argument("--platforms", default="4c8g")
    parser.add_argument("--static-ratios", default="0,0.25,0.5,0.75")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=20)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16", "int8"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-regenerate-dbs", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_csv_ratios(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def ensure_binary(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required binary not found: {path} ({name})")
    return path


def format_ratio_id(ratio: float) -> str:
    return f"{int(round(ratio * 100.0)):02d}"


def run_command(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(shlex.quote(token) for token in command) + "\n\n")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {log_path}")


def load_report(report_path: Path) -> dict[str, object]:
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    build_dir = Path(args.build_dir).resolve()
    db_root = Path(args.db_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    preload_dir = results_dir / "raw" / "preload"
    runs_dir = results_dir / "raw" / "runs"
    logs_dir = results_dir / "raw" / "logs"

    generator_bin = ensure_binary(build_dir / "simpledb_generate_weights", "simpledb_generate_weights")
    inference_bin = ensure_binary(build_dir / "llama_inference_example", "llama_inference_example")

    chunk_sizes = parse_csv_ints(args.chunk_sizes)
    static_ratios = parse_csv_ratios(args.static_ratios)
    selected_platforms = [item.strip() for item in args.platforms.split(",") if item.strip()]

    print(
        f"[preload_cost] Using memory budget fraction={PLATFORM_MEMORY_FRACTION:.2f} "
        "for all platform presets"
    )
    for platform_id in selected_platforms:
        print(f"[preload_cost] Platform {platform_id}: total_memory_bytes={PLATFORM_BYTES[platform_id]}")

    raw_rows: list[dict[str, object]] = []
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}

    for chunk_size in chunk_sizes:
        db_path = db_root / f"llama_chunk{chunk_size}"
        metadata_path = db_path / "metadata.jsonl"
        should_generate = args.force_regenerate_dbs or not metadata_path.is_file()
        if should_generate:
            print(f"[preload_cost] Generating DB for chunk_size={chunk_size}: {db_path}")
            if db_path.exists() and args.force_regenerate_dbs:
                shutil.rmtree(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            generate_command = [
                str(generator_bin),
                "--db-path",
                str(db_path),
                "--hidden-dim",
                str(args.hidden_dim),
                "--num-layers",
                str(args.num_layers),
                "--num-heads",
                str(args.num_heads),
                "--num-kv-heads",
                str(args.num_kv_heads),
                "--chunk-size",
                str(chunk_size),
                "--vocab-size",
                str(args.vocab_size),
                "--seed",
                str(args.seed),
                "--precision",
                args.precision,
            ]
            run_command(generate_command, os.environ.copy(), logs_dir / f"generate_chunk{chunk_size}.log")
        else:
            print(f"[preload_cost] Reusing DB for chunk_size={chunk_size}: {db_path}")

        candidates = load_candidates(metadata_path)
        for platform_id in selected_platforms:
            total_memory_bytes = PLATFORM_BYTES[platform_id]
            threads = 4 if platform_id == "4c8g" else 8
            print(
                f"[preload_cost] Measuring preload cost: platform={platform_id} chunk_size={chunk_size}"
            )
            for static_ratio in static_ratios:
                if static_ratio < 0.0 or static_ratio > 1.0:
                    raise ValueError(f"static ratio must be in [0, 1]: {static_ratio}")

                requested_static_bytes = int(total_memory_bytes * static_ratio)
                plan = build_preload_plan(candidates, requested_static_bytes)
                preload_path = preload_dir / (
                    f"{platform_id}_chunk{chunk_size}_static{format_ratio_id(static_ratio)}.txt"
                )
                write_preload_file(preload_path, plan.matrix_ids)

                print(
                    "[preload_cost] Prepared plan:",
                    f"platform={platform_id}",
                    f"chunk_size={chunk_size}",
                    f"static_ratio={int(round(static_ratio * 100.0))}%",
                    f"requested_static_bytes={requested_static_bytes}",
                    f"selected_static_bytes={plan.selected_static_bytes}",
                    f"matrices={len(plan.matrix_ids)}",
                )

                for repeat_idx in range(args.repeats):
                    run_id = (
                        f"{platform_id}_chunk{chunk_size}"
                        f"_s{format_ratio_id(static_ratio)}_r{repeat_idx}"
                    )
                    report_path = runs_dir / f"{run_id}.json"
                    log_path = logs_dir / f"{run_id}.log"
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = str(threads)
                    command = [
                        str(inference_bin),
                        "--db-path",
                        str(db_path),
                        "--hidden-dim",
                        str(args.hidden_dim),
                        "--num-layers",
                        str(args.num_layers),
                        "--num-heads",
                        str(args.num_heads),
                        "--num-kv-heads",
                        str(args.num_kv_heads),
                        "--chunk-size",
                        str(chunk_size),
                        "--vocab-size",
                        str(args.vocab_size),
                        "--seed",
                        str(args.seed),
                        "--report-json",
                        str(report_path),
                        "--preload-only",
                    ]
                    if plan.matrix_ids:
                        command.extend(["--preload-file", str(preload_path)])

                    if args.force_rerun or not report_path.is_file():
                        print(
                            f"[preload_cost] Running platform={platform_id} "
                            f"chunk_size={chunk_size} "
                            f"static_ratio={int(round(static_ratio * 100.0))}% repeat={repeat_idx}"
                        )
                        run_command(command, env, log_path)
                    else:
                        print(
                            f"[preload_cost] Reusing existing report for platform={platform_id} "
                            f"chunk_size={chunk_size} "
                            f"static_ratio={int(round(static_ratio * 100.0))}% repeat={repeat_idx}"
                        )

                    report = load_report(report_path)
                    preload = report["preload"]
                    preload_s = float(preload["elapsed_s"])
                    throughput_mb_s = (
                        (plan.selected_static_bytes / (1024.0 * 1024.0)) / preload_s
                        if preload_s > 0.0
                        else 0.0
                    )
                    row = {
                        "run_id": run_id,
                        "platform_id": platform_id,
                        "threads": threads,
                        "chunk_size": chunk_size,
                        "static_ratio": static_ratio,
                        "requested_static_bytes": requested_static_bytes,
                        "selected_static_bytes": plan.selected_static_bytes,
                        "matrix_count": len(plan.matrix_ids),
                        "repeat_idx": repeat_idx,
                        "preload_s": preload_s,
                        "throughput_mb_s": throughput_mb_s,
                        "db_path": str(db_path),
                        "preload_file": str(preload_path),
                        "report_json": str(report_path),
                    }
                    raw_rows.append(row)
                    key = (
                        platform_id,
                        threads,
                        chunk_size,
                        static_ratio,
                        requested_static_bytes,
                        plan.selected_static_bytes,
                        len(plan.matrix_ids),
                    )
                    grouped.setdefault(key, []).append(row)

        if db_path.exists():
            print(f"[preload_cost] Deleting DB for chunk_size={chunk_size}: {db_path}")
            shutil.rmtree(db_path)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        summary_rows.append(
            {
                "platform_id": key[0],
                "threads": key[1],
                "chunk_size": key[2],
                "static_ratio": key[3],
                "requested_static_bytes": key[4],
                "selected_static_bytes": key[5],
                "matrix_count": key[6],
                "median_preload_s": statistics.median(float(row["preload_s"]) for row in rows),
                "median_throughput_mb_s": statistics.median(
                    float(row["throughput_mb_s"]) for row in rows
                ),
            }
        )

    write_csv(results_dir / "derived" / "preload_cost_results.csv", RAW_FIELDNAMES, raw_rows)
    write_csv(results_dir / "derived" / "preload_cost_summary.csv", SUMMARY_FIELDNAMES, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
