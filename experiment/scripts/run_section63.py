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
from dataclasses import dataclass
from pathlib import Path

from generate_preload_file import build_preload_plan, load_candidates, write_preload_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiment" / "results"
PLATFORM_MEMORY_FRACTION = 0.7


@dataclass(frozen=True)
class PlatformConfig:
    platform_id: str
    threads: int
    total_memory_bytes: int


PLATFORMS = {
    "4c8g": PlatformConfig(
        platform_id="4c8g",
        threads=4,
        total_memory_bytes=int(PLATFORM_MEMORY_FRACTION * 8 * 1024**3),
    ),
    "8c16g": PlatformConfig(
        platform_id="8c16g",
        threads=8,
        total_memory_bytes=int(PLATFORM_MEMORY_FRACTION * 16 * 1024**3),
    ),
}


RAW_FIELDNAMES = [
    "run_id",
    "platform_id",
    "threads",
    "chunk_size",
    "prefetch_window",
    "static_ratio",
    "static_budget_bytes",
    "static_selected_bytes",
    "bufferpool_bytes",
    "db_path",
    "preload_file",
    "repeat_idx",
    "prefill_s",
    "decode_avg_s",
    "decode_ms_per_token",
    "decode_tokens_per_s",
]

SUMMARY_FIELDNAMES = [
    "platform_id",
    "chunk_size",
    "prefetch_window",
    "static_ratio",
    "static_budget_bytes",
    "static_selected_bytes",
    "bufferpool_bytes",
    "median_prefill_s",
    "median_decode_avg_s",
    "median_decode_ms_per_token",
    "median_decode_tokens_per_s",
]


def parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_csv_ratios(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --env entry (expected KEY=VALUE): {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --env entry (empty key): {item}")
        env[key] = value
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 6.3 Llama experiment grid.")
    parser.add_argument("--build-dir", default=str(REPO_ROOT / "cpp" / "build"))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--command-prefix", default="", help="Optional launcher prefix, split with shlex")
    parser.add_argument("--env", action="append", default=[], help="Extra environment override as KEY=VALUE")
    parser.add_argument("--platforms", default="4c8g")
    parser.add_argument("--chunk-sizes", default="64,128,256,512")
    parser.add_argument("--prefetch-windows", default="1")
    parser.add_argument("--static-ratios", default="0,0.25,0.5,0.75")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=20)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16", "int8"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-regenerate-dbs", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def ensure_binary(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required binary not found: {path} ({name})")
    return path


def format_ratio_id(ratio: float) -> str:
    return f"{int(round(ratio * 100.0)):02d}"


def describe_run(
    platform_id: str,
    chunk_size: int,
    prefetch_window: int,
    static_ratio: float,
    repeat_idx: int,
) -> str:
    static_percent = int(round(static_ratio * 100.0))
    return (
        f"platform={platform_id} chunk_size={chunk_size} "
        f"prefetch_window={prefetch_window} static_ratio={static_percent}% "
        f"repeat={repeat_idx}"
    )


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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_report(report_path: Path) -> dict[str, object]:
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_rows(
    reports: list[dict[str, object]],
    run_specs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_rows: list[dict[str, object]] = []
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}

    for report, spec in zip(reports, run_specs):
        prefill = report["prefill"]
        decode = report["decode"]
        row = {
            "run_id": spec["run_id"],
            "platform_id": spec["platform_id"],
            "threads": spec["threads"],
            "chunk_size": spec["chunk_size"],
            "prefetch_window": spec["prefetch_window"],
            "static_ratio": spec["static_ratio"],
            "static_budget_bytes": spec["static_budget_bytes"],
            "static_selected_bytes": spec["static_selected_bytes"],
            "bufferpool_bytes": spec["bufferpool_bytes"],
            "db_path": spec["db_path"],
            "preload_file": spec["preload_file"],
            "repeat_idx": spec["repeat_idx"],
            "prefill_s": prefill["elapsed_s"],
            "decode_avg_s": decode["avg_time_s"],
            "decode_ms_per_token": decode["ms_per_token"],
            "decode_tokens_per_s": decode["tokens_per_s"],
        }
        raw_rows.append(row)

        key = (
            spec["platform_id"],
            spec["chunk_size"],
            spec["prefetch_window"],
            spec["static_ratio"],
            spec["static_budget_bytes"],
            spec["static_selected_bytes"],
            spec["bufferpool_bytes"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        summary_rows.append(
            {
                "platform_id": key[0],
                "chunk_size": key[1],
                "prefetch_window": key[2],
                "static_ratio": key[3],
                "static_budget_bytes": key[4],
                "static_selected_bytes": key[5],
                "bufferpool_bytes": key[6],
                "median_prefill_s": statistics.median(float(row["prefill_s"]) for row in rows),
                "median_decode_avg_s": statistics.median(float(row["decode_avg_s"]) for row in rows),
                "median_decode_ms_per_token": statistics.median(float(row["decode_ms_per_token"]) for row in rows),
                "median_decode_tokens_per_s": statistics.median(float(row["decode_tokens_per_s"]) for row in rows),
            }
        )

    raw_rows.sort(key=lambda row: str(row["run_id"]))
    return raw_rows, summary_rows


def main() -> int:
    args = parse_args()

    build_dir = Path(args.build_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    raw_dir = results_dir / "raw"
    derived_dir = results_dir / "derived"
    db_dir = raw_dir / "dbs"
    preload_dir = raw_dir / "preload"
    runs_dir = raw_dir / "runs"
    logs_dir = raw_dir / "logs"

    generator_bin = ensure_binary(build_dir / "simpledb_generate_weights", "simpledb_generate_weights")
    inference_bin = ensure_binary(build_dir / "llama_inference_example", "llama_inference_example")

    selected_platforms = [PLATFORMS[item.strip()] for item in args.platforms.split(",") if item.strip()]
    chunk_sizes = parse_csv_ints(args.chunk_sizes)
    prefetch_windows = parse_csv_ints(args.prefetch_windows)
    static_ratios = parse_csv_ratios(args.static_ratios)
    env_overrides = parse_env_overrides(args.env)
    command_prefix = shlex.split(args.command_prefix)
    reports: list[dict[str, object]] = []
    run_specs: list[dict[str, object]] = []

    print(
        f"[section63] Using memory budget fraction={PLATFORM_MEMORY_FRACTION:.2f} "
        "for all platform presets"
    )
    for platform in selected_platforms:
        print(
            f"[section63] Platform {platform.platform_id}: "
            f"threads={platform.threads} total_memory_bytes={platform.total_memory_bytes}"
        )

    for ratio in static_ratios:
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"static ratio must be in [0, 1]: {ratio}")

    for chunk_size in chunk_sizes:
        db_path = db_dir / f"llama_chunk{chunk_size}"
        metadata_path = db_path / "metadata.jsonl"
        try:
            should_generate = args.force_regenerate_dbs or not metadata_path.is_file()
            if should_generate:
                print(f"[section63] Generating DB for chunk_size={chunk_size}: {db_path}")
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
                print(f"[section63] Reusing DB for chunk_size={chunk_size}: {db_path}")

            candidates = load_candidates(metadata_path)
            for platform in selected_platforms:
                for static_ratio in static_ratios:
                    requested_static_bytes = int(platform.total_memory_bytes * static_ratio)
                    plan = build_preload_plan(candidates, requested_static_bytes)
                    preload_path = preload_dir / (
                        f"{platform.platform_id}_chunk{chunk_size}_static{format_ratio_id(static_ratio)}.txt"
                    )
                    write_preload_file(preload_path, plan.matrix_ids)
                    print(
                        "[section63] Prepared preload plan:",
                        f"platform={platform.platform_id}",
                        f"chunk_size={chunk_size}",
                        f"static_ratio={int(round(static_ratio * 100.0))}%",
                        f"requested_static_bytes={requested_static_bytes}",
                        f"selected_static_bytes={plan.selected_static_bytes}",
                        f"matrices={len(plan.matrix_ids)}",
                    )

                    bufferpool_bytes = platform.total_memory_bytes - requested_static_bytes
                    arena_size_mb = bufferpool_bytes // (1024**2)

                    for prefetch_window in prefetch_windows:
                        for repeat_idx in range(args.repeats):
                            run_id = (
                                f"{platform.platform_id}_c{chunk_size}_w{prefetch_window}"
                                f"_s{format_ratio_id(static_ratio)}_r{repeat_idx}"
                            )
                            report_path = runs_dir / f"{run_id}.json"
                            log_path = logs_dir / f"{run_id}.log"
                            run_env = os.environ.copy()
                            run_env.update(env_overrides)
                            run_env["OMP_NUM_THREADS"] = str(platform.threads)

                            command = command_prefix + [
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
                                "--prefetch-window",
                                str(prefetch_window),
                                "--arena-size-mb",
                                str(arena_size_mb),
                                "--seq-len",
                                str(args.seq_len),
                                "--decode-steps",
                                str(args.decode_steps),
                                "--report-json",
                                str(report_path),
                            ]
                            if plan.matrix_ids:
                                command.extend(["--preload-file", str(preload_path)])

                            description = describe_run(
                                platform.platform_id,
                                chunk_size,
                                prefetch_window,
                                static_ratio,
                                repeat_idx,
                            )
                            if args.force_rerun or not report_path.is_file():
                                print(f"[section63] Running {description}")
                                run_command(command, run_env, log_path)
                            else:
                                print(f"[section63] Reusing existing report for {description}")

                            report = load_report(report_path)
                            run_spec = {
                                "run_id": run_id,
                                "platform_id": platform.platform_id,
                                "threads": platform.threads,
                                "chunk_size": chunk_size,
                                "prefetch_window": prefetch_window,
                                "static_ratio": static_ratio,
                                "static_budget_bytes": requested_static_bytes,
                                "static_selected_bytes": plan.selected_static_bytes,
                                "bufferpool_bytes": bufferpool_bytes,
                                "db_path": str(db_path),
                                "preload_file": str(preload_path) if plan.matrix_ids else "",
                                "repeat_idx": repeat_idx,
                            }
                            reports.append(report)
                            run_specs.append(run_spec)
        finally:
            if db_path.exists():
                print(f"[section63] Deleting DB for chunk_size={chunk_size}: {db_path}")
                shutil.rmtree(db_path)

    if not reports:
        raise RuntimeError("no runs were executed or loaded")

    raw_rows, summary_rows = resolve_rows(reports, run_specs)
    write_csv(derived_dir / "section63_results.csv", RAW_FIELDNAMES, raw_rows)
    write_csv(derived_dir / "section63_summary.csv", SUMMARY_FIELDNAMES, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
