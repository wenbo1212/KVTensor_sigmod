#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiment" / "results" / "stable_diffusion"
ARENA_MEMORY_FRACTION = 0.5


@dataclass(frozen=True)
class PlatformConfig:
    platform_id: str
    threads: int
    platform_memory_bytes: int

    @property
    def arena_bytes(self) -> int:
        return int(self.platform_memory_bytes * ARENA_MEMORY_FRACTION)

    @property
    def arena_size_mb(self) -> int:
        return max(1, self.arena_bytes // (1024**2))


PLATFORMS = {
    "4c8g": PlatformConfig("4c8g", threads=4, platform_memory_bytes=8 * 1024**3),
    "8c16g": PlatformConfig("8c16g", threads=8, platform_memory_bytes=16 * 1024**3),
}


RUN_FIELDNAMES = [
    "run_id",
    "platform_id",
    "threads",
    "platform_memory_bytes",
    "arena_size_mb",
    "chunk_size",
    "prefetch_window",
    "repeat_idx",
    "db_path",
    "graph_path",
    "report_json",
    "log_path",
    "total_wall_ms",
    "preload_ms",
    "prefetch_warmup_ms",
    "execution_ms_excluding_preload",
    "pipeline_ms_excluding_preload_and_prefetch_warmup",
    "output_rows",
    "output_cols",
    "used_bufferpool",
    "bufferpool_wait_ms",
    "bufferpool_prefetch_get_time_ms",
    "hidden_io_ms_est",
    "hidden_io_ratio_est",
    "step_count",
    "profiled_total_ms",
    "profiled_compute_ms",
    "profiled_other_compute_ms",
    "profiled_decompress_ms",
    "profiled_io_ms",
    "profiled_system_overhead_ms",
    "profiled_total_compute_like_ms",
    "profiled_bytes_read_mb",
    "profiled_gemm_flops",
    "profiled_pipeline_share",
    "unprofiled_pipeline_ms",
    "profiled_compute_like_share",
    "profiled_io_share",
    "profiled_system_overhead_share",
]

STEP_FIELDNAMES = [
    "run_id",
    "platform_id",
    "threads",
    "arena_size_mb",
    "chunk_size",
    "prefetch_window",
    "repeat_idx",
    "step",
    "timestep",
    "total_ms",
    "compute_ms",
    "other_compute_ms",
    "decompress_ms",
    "kv_read_ms",
    "overhead_ms",
    "bytes_read_mb",
    "io_throughput_mb_s",
    "gemm_flops",
    "gemm_throughput_gflops",
]

SUMMARY_FIELDNAMES = [
    "platform_id",
    "threads",
    "platform_memory_bytes",
    "arena_size_mb",
    "chunk_size",
    "prefetch_window",
    "median_execution_ms_excluding_preload",
    "median_pipeline_ms_excluding_preload_and_prefetch_warmup",
    "median_prefetch_warmup_ms",
    "median_hidden_io_ratio_est",
    "median_hidden_io_ms_est",
    "median_profiled_total_ms",
    "median_profiled_compute_ms",
    "median_profiled_other_compute_ms",
    "median_profiled_decompress_ms",
    "median_profiled_io_ms",
    "median_profiled_system_overhead_ms",
    "median_profiled_pipeline_share",
    "median_unprofiled_pipeline_ms",
    "median_profiled_compute_like_share",
    "median_profiled_io_share",
    "median_profiled_system_overhead_share",
]


STEP_LINE_RE = re.compile(
    r"^\[Profile:SDUNet\] step=(?P<step>\d+) timestep=(?P<timestep>\d+) "
    r"total_ms=(?P<total_ms>[-+0-9.eE]+) "
    r"compute_ms=(?P<compute_ms>[-+0-9.eE]+) "
    r"other_compute_ms=(?P<other_compute_ms>[-+0-9.eE]+) "
    r"kv_read_ms=(?P<kv_read_ms>[-+0-9.eE]+) "
    r"decompress_ms=(?P<decompress_ms>[-+0-9.eE]+) "
    r"overhead_ms=(?P<overhead_ms>[-+0-9.eE]+)$"
)

STEP_IO_RE = re.compile(
    r"^\[Profile:SDUNet\] bytes_read_mb=(?P<bytes_read_mb>[-+0-9.eE]+) "
    r"io_throughput_mb_s=(?P<io_throughput_mb_s>[-+0-9.eE]+) "
    r"io_lower_bound_ms=(?P<io_lower_bound_ms>[-+0-9.eE]+)$"
)

STEP_GEMM_RE = re.compile(
    r"^\[Profile:SDUNet\] gemm_flops=(?P<gemm_flops>\d+) "
    r"gemm_throughput_gflops=(?P<gemm_throughput_gflops>[-+0-9.eE]+) "
    r"compute_lower_bound_ms=(?P<compute_lower_bound_ms>[-+0-9.eE]+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stable Diffusion chunk-size experiment and collect preload-excluded timing, "
            "UNet profile breakdown, and an estimated hidden-I/O ratio from buffer-pool overlap."
        ),
    )
    parser.add_argument("--build-dir", default=str(REPO_ROOT / "cpp" / "build"))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--command-prefix", default="", help="Optional launcher prefix, split with shlex")
    parser.add_argument("--env", action="append", default=[], help="Extra environment override as KEY=VALUE")
    parser.add_argument("--platforms", default="4c8g,8c16g")
    parser.add_argument("--chunk-sizes", default="64,128,256,512")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefetch-window", type=int, default=1)
    parser.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16", "int8"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="a photo of a cat")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--text-hidden", type=int, default=768)
    parser.add_argument("--text-layers", type=int, default=12)
    parser.add_argument("--text-vocab-size", type=int, default=49408)
    parser.add_argument("--text-max-length", type=int, default=77)
    parser.add_argument("--unet-hidden", type=int, default=320)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--sample-height", type=int, default=64)
    parser.add_argument("--sample-width", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--vae-conv-layers", type=int, default=4)
    parser.add_argument("--force-regenerate-dbs", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


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


def ensure_binary(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required binary not found: {path} ({name})")
    return path


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


def parse_step_profiles(log_path: Path) -> list[dict[str, float | int]]:
    steps: list[dict[str, float | int]] = []
    last_step: dict[str, float | int] | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = STEP_LINE_RE.match(line)
            if match:
                last_step = {
                    "step": int(match.group("step")),
                    "timestep": int(match.group("timestep")),
                    "total_ms": float(match.group("total_ms")),
                    "compute_ms": float(match.group("compute_ms")),
                    "other_compute_ms": float(match.group("other_compute_ms")),
                    "kv_read_ms": float(match.group("kv_read_ms")),
                    "decompress_ms": float(match.group("decompress_ms")),
                    "overhead_ms": float(match.group("overhead_ms")),
                    "bytes_read_mb": 0.0,
                    "io_throughput_mb_s": 0.0,
                    "gemm_flops": 0,
                    "gemm_throughput_gflops": 0.0,
                }
                steps.append(last_step)
                continue

            if last_step is None:
                continue

            match = STEP_IO_RE.match(line)
            if match:
                last_step["bytes_read_mb"] = float(match.group("bytes_read_mb"))
                last_step["io_throughput_mb_s"] = float(match.group("io_throughput_mb_s"))
                continue

            match = STEP_GEMM_RE.match(line)
            if match:
                last_step["gemm_flops"] = int(match.group("gemm_flops"))
                last_step["gemm_throughput_gflops"] = float(match.group("gemm_throughput_gflops"))

    return steps


def aggregate_steps(steps: list[dict[str, float | int]], pipeline_ms: float) -> dict[str, float]:
    profiled_total_ms = sum(float(step["total_ms"]) for step in steps)
    profiled_compute_ms = sum(float(step["compute_ms"]) for step in steps)
    profiled_other_compute_ms = sum(float(step["other_compute_ms"]) for step in steps)
    profiled_decompress_ms = sum(float(step["decompress_ms"]) for step in steps)
    profiled_io_ms = sum(float(step["kv_read_ms"]) for step in steps)
    profiled_system_overhead_ms = sum(float(step["overhead_ms"]) for step in steps)
    profiled_total_compute_like_ms = (
        profiled_compute_ms + profiled_other_compute_ms + profiled_decompress_ms
    )
    profiled_bytes_read_mb = sum(float(step["bytes_read_mb"]) for step in steps)
    profiled_gemm_flops = float(sum(int(step["gemm_flops"]) for step in steps))
    profiled_pipeline_share = (profiled_total_ms / pipeline_ms) if pipeline_ms > 0.0 else 0.0
    unprofiled_pipeline_ms = max(0.0, pipeline_ms - profiled_total_ms)
    profiled_compute_like_share = (
        profiled_total_compute_like_ms / profiled_total_ms if profiled_total_ms > 0.0 else 0.0
    )
    profiled_io_share = profiled_io_ms / profiled_total_ms if profiled_total_ms > 0.0 else 0.0
    profiled_system_overhead_share = (
        profiled_system_overhead_ms / profiled_total_ms if profiled_total_ms > 0.0 else 0.0
    )
    return {
        "profiled_total_ms": profiled_total_ms,
        "profiled_compute_ms": profiled_compute_ms,
        "profiled_other_compute_ms": profiled_other_compute_ms,
        "profiled_decompress_ms": profiled_decompress_ms,
        "profiled_io_ms": profiled_io_ms,
        "profiled_system_overhead_ms": profiled_system_overhead_ms,
        "profiled_total_compute_like_ms": profiled_total_compute_like_ms,
        "profiled_bytes_read_mb": profiled_bytes_read_mb,
        "profiled_gemm_flops": profiled_gemm_flops,
        "profiled_pipeline_share": profiled_pipeline_share,
        "unprofiled_pipeline_ms": unprofiled_pipeline_ms,
        "profiled_compute_like_share": profiled_compute_like_share,
        "profiled_io_share": profiled_io_share,
        "profiled_system_overhead_share": profiled_system_overhead_share,
    }


def summarize_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["platform_id"],
            row["threads"],
            row["platform_memory_bytes"],
            row["arena_size_mb"],
            row["chunk_size"],
            row["prefetch_window"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        summary_rows.append(
            {
                "platform_id": key[0],
                "threads": key[1],
                "platform_memory_bytes": key[2],
                "arena_size_mb": key[3],
                "chunk_size": key[4],
                "prefetch_window": key[5],
                "median_execution_ms_excluding_preload": statistics.median(
                    float(row["execution_ms_excluding_preload"]) for row in group
                ),
                "median_pipeline_ms_excluding_preload_and_prefetch_warmup": statistics.median(
                    float(row["pipeline_ms_excluding_preload_and_prefetch_warmup"]) for row in group
                ),
                "median_prefetch_warmup_ms": statistics.median(
                    float(row["prefetch_warmup_ms"]) for row in group
                ),
                "median_hidden_io_ratio_est": statistics.median(
                    float(row["hidden_io_ratio_est"]) for row in group
                ),
                "median_hidden_io_ms_est": statistics.median(
                    float(row["hidden_io_ms_est"]) for row in group
                ),
                "median_profiled_total_ms": statistics.median(
                    float(row["profiled_total_ms"]) for row in group
                ),
                "median_profiled_compute_ms": statistics.median(
                    float(row["profiled_compute_ms"]) for row in group
                ),
                "median_profiled_other_compute_ms": statistics.median(
                    float(row["profiled_other_compute_ms"]) for row in group
                ),
                "median_profiled_decompress_ms": statistics.median(
                    float(row["profiled_decompress_ms"]) for row in group
                ),
                "median_profiled_io_ms": statistics.median(
                    float(row["profiled_io_ms"]) for row in group
                ),
                "median_profiled_system_overhead_ms": statistics.median(
                    float(row["profiled_system_overhead_ms"]) for row in group
                ),
                "median_profiled_pipeline_share": statistics.median(
                    float(row["profiled_pipeline_share"]) for row in group
                ),
                "median_unprofiled_pipeline_ms": statistics.median(
                    float(row["unprofiled_pipeline_ms"]) for row in group
                ),
                "median_profiled_compute_like_share": statistics.median(
                    float(row["profiled_compute_like_share"]) for row in group
                ),
                "median_profiled_io_share": statistics.median(
                    float(row["profiled_io_share"]) for row in group
                ),
                "median_profiled_system_overhead_share": statistics.median(
                    float(row["profiled_system_overhead_share"]) for row in group
                ),
            }
        )
    return summary_rows


def main() -> int:
    args = parse_args()

    build_dir = Path(args.build_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    raw_dir = results_dir / "raw"
    db_dir = raw_dir / "dbs"
    run_dir = raw_dir / "runs"
    log_dir = raw_dir / "logs"
    report_dir = raw_dir / "reports"

    generator_bin = ensure_binary(build_dir / "simpledb_generate_sd_weights", "simpledb_generate_sd_weights")
    inference_bin = ensure_binary(
        build_dir / "stable_diffusion_inference_example",
        "stable_diffusion_inference_example",
    )

    chunk_sizes = parse_csv_ints(args.chunk_sizes)
    selected_platforms = [PLATFORMS[item.strip()] for item in args.platforms.split(",") if item.strip()]
    command_prefix = shlex.split(args.command_prefix)
    env_overrides = parse_env_overrides(args.env)

    print(
        f"[sd_experiment] Using arena fraction={ARENA_MEMORY_FRACTION:.2f} of platform memory "
        "for all platform presets"
    )
    for platform in selected_platforms:
        print(
            f"[sd_experiment] Platform {platform.platform_id}: "
            f"threads={platform.threads} "
            f"platform_memory_bytes={platform.platform_memory_bytes} "
            f"arena_size_mb={platform.arena_size_mb}"
        )

    run_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []

    for chunk_size in chunk_sizes:
        db_path = db_dir / f"sd_chunk{chunk_size}"
        metadata_path = db_path / "metadata.jsonl"
        graph_path = db_path / "prefetch_graph.txt"
        should_generate = (
            args.force_regenerate_dbs
            or not metadata_path.is_file()
            or not graph_path.is_file()
        )
        if should_generate:
            print(f"[sd_experiment] Generating DB for chunk_size={chunk_size}: {db_path}")
            if db_path.exists() and args.force_regenerate_dbs:
                shutil.rmtree(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            generate_command = [
                str(generator_bin),
                "--db-path",
                str(db_path),
                "--graph-path",
                str(graph_path),
                "--chunk-size",
                str(chunk_size),
                "--precision",
                args.precision,
                "--seed",
                str(args.seed),
                "--text-hidden",
                str(args.text_hidden),
                "--text-layers",
                str(args.text_layers),
                "--text-vocab-size",
                str(args.text_vocab_size),
                "--unet-hidden",
                str(args.unet_hidden),
                "--latent-channels",
                str(args.latent_channels),
                "--image-channels",
                str(args.image_channels),
                "--vae-conv-layers",
                str(args.vae_conv_layers),
                "--steps",
                str(args.steps),
            ]
            run_command(generate_command, os.environ.copy(), log_dir / f"generate_chunk{chunk_size}.log")
        else:
            print(f"[sd_experiment] Reusing DB for chunk_size={chunk_size}: {db_path}")

        for platform in selected_platforms:
            for repeat_idx in range(args.repeats):
                run_id = f"{platform.platform_id}_chunk{chunk_size}_r{repeat_idx}"
                report_path = report_dir / f"{run_id}.json"
                log_path = log_dir / f"{run_id}.log"
                env = os.environ.copy()
                env.update(env_overrides)
                env.setdefault("OMP_NUM_THREADS", str(platform.threads))
                env.setdefault("MKL_NUM_THREADS", str(platform.threads))
                command = command_prefix + [
                    str(inference_bin),
                    "--db-path",
                    str(db_path),
                    "--prompt",
                    args.prompt,
                    "--steps",
                    str(args.steps),
                    "--guidance-scale",
                    str(args.guidance_scale),
                    "--chunk-size",
                    str(chunk_size),
                    "--arena-size-mb",
                    str(platform.arena_size_mb),
                    "--prefetch-window",
                    str(args.prefetch_window),
                    "--prefetch-graph",
                    str(graph_path),
                    "--seed",
                    str(args.seed),
                    "--text-hidden",
                    str(args.text_hidden),
                    "--text-layers",
                    str(args.text_layers),
                    "--text-vocab-size",
                    str(args.text_vocab_size),
                    "--text-max-length",
                    str(args.text_max_length),
                    "--unet-hidden",
                    str(args.unet_hidden),
                    "--latent-channels",
                    str(args.latent_channels),
                    "--sample-height",
                    str(args.sample_height),
                    "--sample-width",
                    str(args.sample_width),
                    "--image-size",
                    str(args.image_size),
                    "--image-channels",
                    str(args.image_channels),
                    "--vae-conv-layers",
                    str(args.vae_conv_layers),
                    "--profile",
                    "--report-json",
                    str(report_path),
                ]

                if args.force_rerun or not report_path.is_file():
                    print(
                        f"[sd_experiment] Running platform={platform.platform_id} "
                        f"chunk_size={chunk_size} repeat={repeat_idx}"
                    )
                    run_command(command, env, log_path)
                else:
                    print(
                        f"[sd_experiment] Reusing existing report for platform={platform.platform_id} "
                        f"chunk_size={chunk_size} repeat={repeat_idx}"
                    )

                report = load_report(report_path)
                steps = parse_step_profiles(log_path)
                if not steps:
                    raise RuntimeError(f"no SDUNet profile lines found in {log_path}")

                timing = report["timing"]
                bufferpool = report["bufferpool"]
                overlap = report["io_overlap_estimate"]
                output = report["output"]
                pipeline_ms = float(timing["pipeline_ms_excluding_preload_and_prefetch_warmup"])
                aggregates = aggregate_steps(steps, pipeline_ms)

                run_row = {
                    "run_id": run_id,
                    "platform_id": platform.platform_id,
                    "threads": platform.threads,
                    "platform_memory_bytes": platform.platform_memory_bytes,
                    "arena_size_mb": platform.arena_size_mb,
                    "chunk_size": chunk_size,
                    "prefetch_window": args.prefetch_window,
                    "repeat_idx": repeat_idx,
                    "db_path": str(db_path),
                    "graph_path": str(graph_path),
                    "report_json": str(report_path),
                    "log_path": str(log_path),
                    "total_wall_ms": float(timing["total_wall_ms"]),
                    "preload_ms": float(timing["preload_ms"]),
                    "prefetch_warmup_ms": float(timing["prefetch_warmup_ms"]),
                    "execution_ms_excluding_preload": float(timing["execution_ms_excluding_preload"]),
                    "pipeline_ms_excluding_preload_and_prefetch_warmup": pipeline_ms,
                    "output_rows": int(output["rows"]),
                    "output_cols": int(output["cols"]),
                    "used_bufferpool": bool(bufferpool["used_bufferpool"]),
                    "bufferpool_wait_ms": float(bufferpool["wait_ms"]),
                    "bufferpool_prefetch_get_time_ms": float(overlap["prefetch_get_time_ms"]),
                    "hidden_io_ms_est": float(overlap["hidden_io_ms"]),
                    "hidden_io_ratio_est": float(overlap["hidden_io_ratio"]),
                    "step_count": len(steps),
                }
                run_row.update(aggregates)
                run_rows.append(run_row)

                for step in steps:
                    step_rows.append(
                        {
                            "run_id": run_id,
                            "platform_id": platform.platform_id,
                            "threads": platform.threads,
                            "arena_size_mb": platform.arena_size_mb,
                            "chunk_size": chunk_size,
                            "prefetch_window": args.prefetch_window,
                            "repeat_idx": repeat_idx,
                            "step": int(step["step"]),
                            "timestep": int(step["timestep"]),
                            "total_ms": float(step["total_ms"]),
                            "compute_ms": float(step["compute_ms"]),
                            "other_compute_ms": float(step["other_compute_ms"]),
                            "decompress_ms": float(step["decompress_ms"]),
                            "kv_read_ms": float(step["kv_read_ms"]),
                            "overhead_ms": float(step["overhead_ms"]),
                            "bytes_read_mb": float(step["bytes_read_mb"]),
                            "io_throughput_mb_s": float(step["io_throughput_mb_s"]),
                            "gemm_flops": int(step["gemm_flops"]),
                            "gemm_throughput_gflops": float(step["gemm_throughput_gflops"]),
                        }
                    )

    run_rows.sort(key=lambda row: str(row["run_id"]))
    step_rows.sort(key=lambda row: (str(row["run_id"]), int(row["step"])))
    summary_rows = summarize_runs(run_rows)

    write_csv(results_dir / "stable_diffusion_runs.csv", RUN_FIELDNAMES, run_rows)
    write_csv(results_dir / "stable_diffusion_steps.csv", STEP_FIELDNAMES, step_rows)
    write_csv(results_dir / "stable_diffusion_summary.csv", SUMMARY_FIELDNAMES, summary_rows)

    print(f"[sd_experiment] Wrote run table: {results_dir / 'stable_diffusion_runs.csv'}")
    print(f"[sd_experiment] Wrote step table: {results_dir / 'stable_diffusion_steps.csv'}")
    print(f"[sd_experiment] Wrote summary table: {results_dir / 'stable_diffusion_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
