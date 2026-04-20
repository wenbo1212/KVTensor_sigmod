#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from generate_preload_file import write_preload_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiment" / "results" / "llama_prefill_logits"
PLATFORM_MEMORY_FRACTION = 0.7
EPSILON = 1e-12
T = TypeVar("T")


@dataclass(frozen=True)
class PlatformConfig:
    platform_id: str
    threads: int
    total_memory_bytes: int

    @property
    def arena_size_mb(self) -> int:
        return self.total_memory_bytes // (1024**2)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int = 32000


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

MODELS = {
    "3b": ModelConfig(
        model_id="3b",
        hidden_dim=3072,
        num_layers=20,
        num_heads=24,
        num_kv_heads=8,
    ),
    "8b": ModelConfig(
        model_id="8b",
        hidden_dim=4096,
        num_layers=20,
        num_heads=32,
        num_kv_heads=8,
    ),
}

PREFILL_RAW_FIELDNAMES = [
    "run_id",
    "experiment",
    "platform_id",
    "threads",
    "model_id",
    "chunk_size",
    "prefetch_window",
    "arena_size_mb",
    "seq_len",
    "repeat_idx",
    "db_path",
    "report_json",
    "log_path",
    "preload_s",
    "prefill_s",
]

PREFILL_SUMMARY_FIELDNAMES = [
    "platform_id",
    "threads",
    "model_id",
    "chunk_size",
    "prefetch_window",
    "arena_size_mb",
    "seq_len",
    "median_prefill_s",
    "median_preload_s",
]

LOGITS_RAW_FIELDNAMES = [
    "run_id",
    "experiment",
    "platform_id",
    "threads",
    "model_id",
    "chunk_size",
    "prefetch_window",
    "arena_size_mb",
    "seq_len",
    "repeat_idx",
    "db_path",
    "full_preload_file",
    "bufferpool_report_json",
    "preload_report_json",
    "bufferpool_log_path",
    "preload_log_path",
    "bufferpool_prefill_s",
    "bufferpool_decode_avg_s",
    "preload_materialization_s",
    "preload_prefill_s",
    "preload_decode_avg_s",
    "logit_count",
    "bufferpool_top1_id",
    "preload_top1_id",
    "top1_match",
    "max_abs_error",
    "mean_abs_error",
    "rmse",
    "relative_l2_error",
    "cosine_similarity",
]

LOGITS_SUMMARY_FIELDNAMES = [
    "platform_id",
    "threads",
    "model_id",
    "chunk_size",
    "prefetch_window",
    "arena_size_mb",
    "seq_len",
    "repeat_count",
    "top1_match_rate",
    "median_max_abs_error",
    "median_mean_abs_error",
    "median_rmse",
    "median_relative_l2_error",
    "median_cosine_similarity",
]


def parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_csv_names(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


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


def select_configs(mapping: dict[str, T], names: list[str], label: str) -> list[T]:
    selected: list[T] = []
    missing: list[str] = []
    for name in names:
        value = mapping.get(name)
        if value is None:
            missing.append(name)
        else:
            selected.append(value)
    if missing:
        raise ValueError(
            f"unknown {label}: {missing}; available {label}: {sorted(mapping)}"
        )
    return selected


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


def load_all_matrix_ids(metadata_path: Path) -> list[str]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")
    matrix_ids: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {metadata_path}") from exc
            matrix_id = payload.get("matrix_id")
            if not matrix_id:
                raise ValueError(f"missing matrix_id on line {line_number} of {metadata_path}")
            matrix_ids.add(str(matrix_id))
    return sorted(matrix_ids)


def build_base_env(platform: PlatformConfig, env_overrides: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(env_overrides)
    env.setdefault("OMP_NUM_THREADS", str(platform.threads))
    env.setdefault("MKL_NUM_THREADS", str(platform.threads))
    return env


def build_inference_command(
    inference_bin: Path,
    model: ModelConfig,
    platform: PlatformConfig,
    chunk_size: int,
    prefetch_window: int,
    seq_len: int,
    seed: int,
    db_path: Path,
    report_path: Path,
    dump_first_token_logits: bool,
    decode_steps: int,
    preload_file: Path | None = None,
    disable_bufferpool: bool = False,
) -> list[str]:
    command = [
        str(inference_bin),
        "--db-path",
        str(db_path),
        "--hidden-dim",
        str(model.hidden_dim),
        "--num-layers",
        str(model.num_layers),
        "--num-heads",
        str(model.num_heads),
        "--num-kv-heads",
        str(model.num_kv_heads),
        "--chunk-size",
        str(chunk_size),
        "--vocab-size",
        str(model.vocab_size),
        "--seed",
        str(seed),
        "--prefetch-window",
        str(prefetch_window),
        "--arena-size-mb",
        str(platform.arena_size_mb),
        "--seq-len",
        str(seq_len),
        "--decode-steps",
        str(decode_steps),
        "--report-json",
        str(report_path),
    ]
    if dump_first_token_logits:
        command.append("--dump-first-token-logits")
    if preload_file is not None:
        command.extend(["--preload-file", str(preload_file)])
    if disable_bufferpool:
        command.append("--disable-bufferpool")
    return command


def summarize_prefill(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["platform_id"],
            row["threads"],
            row["model_id"],
            row["chunk_size"],
            row["prefetch_window"],
            row["arena_size_mb"],
            row["seq_len"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        summary_rows.append(
            {
                "platform_id": key[0],
                "threads": key[1],
                "model_id": key[2],
                "chunk_size": key[3],
                "prefetch_window": key[4],
                "arena_size_mb": key[5],
                "seq_len": key[6],
                "median_prefill_s": statistics.median(float(row["prefill_s"]) for row in group),
                "median_preload_s": statistics.median(float(row["preload_s"]) for row in group),
            }
        )
    return summary_rows


def summarize_logits(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["platform_id"],
            row["threads"],
            row["model_id"],
            row["chunk_size"],
            row["prefetch_window"],
            row["arena_size_mb"],
            row["seq_len"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        summary_rows.append(
            {
                "platform_id": key[0],
                "threads": key[1],
                "model_id": key[2],
                "chunk_size": key[3],
                "prefetch_window": key[4],
                "arena_size_mb": key[5],
                "seq_len": key[6],
                "repeat_count": len(group),
                "top1_match_rate": sum(int(row["top1_match"]) for row in group) / len(group),
                "median_max_abs_error": statistics.median(float(row["max_abs_error"]) for row in group),
                "median_mean_abs_error": statistics.median(float(row["mean_abs_error"]) for row in group),
                "median_rmse": statistics.median(float(row["rmse"]) for row in group),
                "median_relative_l2_error": statistics.median(float(row["relative_l2_error"]) for row in group),
                "median_cosine_similarity": statistics.median(float(row["cosine_similarity"]) for row in group),
            }
        )
    return summary_rows


def extract_first_token_logits(report: dict[str, object]) -> list[float]:
    decode = report["decode"]
    first_token = decode["first_generated_token"]
    if not first_token.get("logits_dumped", False):
        raise ValueError("report does not contain dumped first-token logits")
    logits = first_token.get("logits")
    if not isinstance(logits, list):
        raise ValueError("first-token logits are missing or malformed")
    return [float(value) for value in logits]


def argmax(values: list[float]) -> int:
    if not values:
        raise ValueError("cannot compute argmax of an empty list")
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def compute_error_metrics(bufferpool_logits: list[float], preload_logits: list[float]) -> dict[str, float | int]:
    if len(bufferpool_logits) != len(preload_logits):
        raise ValueError(
            "logit length mismatch: "
            f"bufferpool={len(bufferpool_logits)} preload={len(preload_logits)}"
        )
    if not bufferpool_logits:
        raise ValueError("logit arrays are empty")

    sum_abs = 0.0
    sum_sq = 0.0
    dot = 0.0
    bufferpool_sq = 0.0
    preload_sq = 0.0
    max_abs = 0.0

    for bufferpool_value, preload_value in zip(bufferpool_logits, preload_logits):
        diff = bufferpool_value - preload_value
        abs_diff = abs(diff)
        if abs_diff > max_abs:
            max_abs = abs_diff
        sum_abs += abs_diff
        sum_sq += diff * diff
        dot += bufferpool_value * preload_value
        bufferpool_sq += bufferpool_value * bufferpool_value
        preload_sq += preload_value * preload_value

    length = len(bufferpool_logits)
    rmse = math.sqrt(sum_sq / length)
    preload_norm = math.sqrt(preload_sq)
    cosine_denominator = math.sqrt(bufferpool_sq) * preload_norm
    bufferpool_top1 = argmax(bufferpool_logits)
    preload_top1 = argmax(preload_logits)
    return {
        "logit_count": length,
        "bufferpool_top1_id": bufferpool_top1,
        "preload_top1_id": preload_top1,
        "top1_match": int(bufferpool_top1 == preload_top1),
        "max_abs_error": max_abs,
        "mean_abs_error": sum_abs / length,
        "rmse": rmse,
        "relative_l2_error": math.sqrt(sum_sq) / max(preload_norm, EPSILON),
        "cosine_similarity": dot / max(cosine_denominator, EPSILON),
    }


def generate_model_db(
    generator_bin: Path,
    model: ModelConfig,
    chunk_size: int,
    precision: str,
    seed: int,
    db_path: Path,
    log_dir: Path,
    force_regenerate_dbs: bool,
) -> Path:
    metadata_path = db_path / "metadata.jsonl"
    should_generate = force_regenerate_dbs or not metadata_path.is_file()
    if should_generate:
        print(
            f"[llama_experiment] Generating DB for model={model.model_id} "
            f"chunk_size={chunk_size}: {db_path}"
        )
        if db_path.exists() and force_regenerate_dbs:
            shutil.rmtree(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        generate_command = [
            str(generator_bin),
            "--db-path",
            str(db_path),
            "--hidden-dim",
            str(model.hidden_dim),
            "--num-layers",
            str(model.num_layers),
            "--num-heads",
            str(model.num_heads),
            "--num-kv-heads",
            str(model.num_kv_heads),
            "--chunk-size",
            str(chunk_size),
            "--vocab-size",
            str(model.vocab_size),
            "--seed",
            str(seed),
            "--precision",
            precision,
        ]
        run_command(
            generate_command,
            os.environ.copy(),
            log_dir / f"generate_{model.model_id}_chunk{chunk_size}.log",
        )
    else:
        print(
            f"[llama_experiment] Reusing DB for model={model.model_id} "
            f"chunk_size={chunk_size}: {db_path}"
        )
    return metadata_path


def prepare_full_preload_file(
    model: ModelConfig,
    metadata_path: Path,
    preload_dir: Path,
    chunk_size: int,
) -> Path:
    preload_path = preload_dir / f"{model.model_id}_chunk{chunk_size}_all_matrices.txt"
    matrix_ids = load_all_matrix_ids(metadata_path)
    write_preload_file(preload_path, matrix_ids)
    return preload_path


def run_prefill_experiment(
    inference_bin: Path,
    selected_models: list[ModelConfig],
    prefill_platforms: list[PlatformConfig],
    prefill_seq_lens: list[int],
    chunk_size: int,
    prefetch_window: int,
    seed: int,
    repeats: int,
    db_paths: dict[str, Path],
    env_overrides: dict[str, str],
    command_prefix: list[str],
    report_dir: Path,
    log_dir: Path,
    force_rerun: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    print("[llama_experiment] Starting experiment 1: prefill latency without preload")
    for model in selected_models:
        db_path = db_paths[model.model_id]
        for platform in prefill_platforms:
            env = build_base_env(platform, env_overrides)
            for seq_len in prefill_seq_lens:
                for repeat_idx in range(repeats):
                    run_id = (
                        f"{platform.platform_id}_{model.model_id}"
                        f"_seq{seq_len}_prefill_r{repeat_idx}"
                    )
                    report_path = report_dir / "prefill" / f"{run_id}.json"
                    current_log_path = log_dir / "prefill" / f"{run_id}.log"
                    command = command_prefix + build_inference_command(
                        inference_bin=inference_bin,
                        model=model,
                        platform=platform,
                        chunk_size=chunk_size,
                        prefetch_window=prefetch_window,
                        seq_len=seq_len,
                        seed=seed,
                        db_path=db_path,
                        report_path=report_path,
                        dump_first_token_logits=False,
                        decode_steps=1,
                    )

                    if force_rerun or not report_path.is_file():
                        print(
                            f"[llama_experiment] Running prefill "
                            f"platform={platform.platform_id} model={model.model_id} "
                            f"seq_len={seq_len} repeat={repeat_idx}"
                        )
                        run_command(command, env, current_log_path)
                    else:
                        print(
                            f"[llama_experiment] Reusing prefill report "
                            f"platform={platform.platform_id} model={model.model_id} "
                            f"seq_len={seq_len} repeat={repeat_idx}"
                        )

                    report = load_report(report_path)
                    rows.append(
                        {
                            "run_id": run_id,
                            "experiment": "prefill_no_static",
                            "platform_id": platform.platform_id,
                            "threads": platform.threads,
                            "model_id": model.model_id,
                            "chunk_size": chunk_size,
                            "prefetch_window": prefetch_window,
                            "arena_size_mb": platform.arena_size_mb,
                            "seq_len": seq_len,
                            "repeat_idx": repeat_idx,
                            "db_path": str(db_path),
                            "report_json": str(report_path),
                            "log_path": str(current_log_path),
                            "preload_s": float(report["preload"]["elapsed_s"]),
                            "prefill_s": float(report["prefill"]["elapsed_s"]),
                        }
                    )
    return rows


def run_logits_experiment(
    inference_bin: Path,
    selected_models: list[ModelConfig],
    logits_platform: PlatformConfig,
    logits_seq_lens: list[int],
    chunk_size: int,
    prefetch_window: int,
    seed: int,
    repeats: int,
    db_paths: dict[str, Path],
    full_preload_files: dict[str, Path],
    env_overrides: dict[str, str],
    command_prefix: list[str],
    report_dir: Path,
    log_dir: Path,
    force_rerun: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    env = build_base_env(logits_platform, env_overrides)

    print("[llama_experiment] Starting experiment 2: first-token logits error on 8c16g")
    for model in selected_models:
        db_path = db_paths[model.model_id]
        preload_file = full_preload_files[model.model_id]
        for seq_len in logits_seq_lens:
            for repeat_idx in range(repeats):
                run_id = (
                    f"{logits_platform.platform_id}_{model.model_id}"
                    f"_seq{seq_len}_logits_r{repeat_idx}"
                )
                bufferpool_report = report_dir / "logits" / f"{run_id}_bufferpool.json"
                preload_report = report_dir / "logits" / f"{run_id}_preload.json"
                bufferpool_log = log_dir / "logits" / f"{run_id}_bufferpool.log"
                preload_log = log_dir / "logits" / f"{run_id}_preload.log"

                bufferpool_command = command_prefix + build_inference_command(
                    inference_bin=inference_bin,
                    model=model,
                    platform=logits_platform,
                    chunk_size=chunk_size,
                    prefetch_window=prefetch_window,
                    seq_len=seq_len,
                    seed=seed,
                    db_path=db_path,
                    report_path=bufferpool_report,
                    dump_first_token_logits=True,
                    decode_steps=1,
                )
                preload_command = command_prefix + build_inference_command(
                    inference_bin=inference_bin,
                    model=model,
                    platform=logits_platform,
                    chunk_size=chunk_size,
                    prefetch_window=prefetch_window,
                    seq_len=seq_len,
                    seed=seed,
                    db_path=db_path,
                    report_path=preload_report,
                    dump_first_token_logits=True,
                    decode_steps=1,
                    preload_file=preload_file,
                    disable_bufferpool=True,
                )

                if force_rerun or not bufferpool_report.is_file():
                    print(
                        f"[llama_experiment] Running bufferpool logits "
                        f"platform={logits_platform.platform_id} model={model.model_id} "
                        f"seq_len={seq_len} repeat={repeat_idx}"
                    )
                    run_command(bufferpool_command, env, bufferpool_log)
                else:
                    print(
                        f"[llama_experiment] Reusing bufferpool logits report "
                        f"platform={logits_platform.platform_id} model={model.model_id} "
                        f"seq_len={seq_len} repeat={repeat_idx}"
                    )

                if force_rerun or not preload_report.is_file():
                    print(
                        f"[llama_experiment] Running full-preload logits "
                        f"platform={logits_platform.platform_id} model={model.model_id} "
                        f"seq_len={seq_len} repeat={repeat_idx}"
                    )
                    run_command(preload_command, env, preload_log)
                else:
                    print(
                        f"[llama_experiment] Reusing full-preload logits report "
                        f"platform={logits_platform.platform_id} model={model.model_id} "
                        f"seq_len={seq_len} repeat={repeat_idx}"
                    )

                bufferpool_payload = load_report(bufferpool_report)
                preload_payload = load_report(preload_report)
                metrics = compute_error_metrics(
                    extract_first_token_logits(bufferpool_payload),
                    extract_first_token_logits(preload_payload),
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "experiment": "first_token_logits_error",
                        "platform_id": logits_platform.platform_id,
                        "threads": logits_platform.threads,
                        "model_id": model.model_id,
                        "chunk_size": chunk_size,
                        "prefetch_window": prefetch_window,
                        "arena_size_mb": logits_platform.arena_size_mb,
                        "seq_len": seq_len,
                        "repeat_idx": repeat_idx,
                        "db_path": str(db_path),
                        "full_preload_file": str(preload_file),
                        "bufferpool_report_json": str(bufferpool_report),
                        "preload_report_json": str(preload_report),
                        "bufferpool_log_path": str(bufferpool_log),
                        "preload_log_path": str(preload_log),
                        "bufferpool_prefill_s": float(bufferpool_payload["prefill"]["elapsed_s"]),
                        "bufferpool_decode_avg_s": float(bufferpool_payload["decode"]["avg_time_s"]),
                        "preload_materialization_s": float(preload_payload["preload"]["elapsed_s"]),
                        "preload_prefill_s": float(preload_payload["prefill"]["elapsed_s"]),
                        "preload_decode_avg_s": float(preload_payload["decode"]["avg_time_s"]),
                        **metrics,
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two Llama experiments in one script: "
            "(1) prefill latency without static residency, then "
            "(2) first-token logits error comparing bufferpool-only against full preload."
        )
    )
    parser.add_argument("--build-dir", default=str(REPO_ROOT / "cpp" / "build"))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--db-root", default="")
    parser.add_argument("--command-prefix", default="", help="Optional launcher prefix, split with shlex")
    parser.add_argument("--env", action="append", default=[], help="Extra environment override as KEY=VALUE")
    parser.add_argument("--experiments", default="prefill,logits")
    parser.add_argument("--models", default="3b,8b")
    parser.add_argument("--prefill-platforms", default="4c8g,8c16g")
    parser.add_argument("--logits-platform", default="8c16g")
    parser.add_argument("--prefill-seq-lens", default="256,1024")
    parser.add_argument("--logits-seq-lens", default="64,256,1024")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--prefetch-window", type=int, default=1)
    parser.add_argument("--prefill-repeats", type=int, default=3)
    parser.add_argument("--logits-repeats", type=int, default=1)
    parser.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16", "int8"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-regenerate-dbs", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--cleanup-dbs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    build_dir = Path(args.build_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    raw_dir = results_dir / "raw"
    derived_dir = results_dir / "derived"
    db_root = Path(args.db_root).resolve() if args.db_root else raw_dir / "dbs"
    report_dir = raw_dir / "reports"
    log_dir = raw_dir / "logs"
    preload_dir = raw_dir / "preload"

    generator_bin = ensure_binary(build_dir / "simpledb_generate_weights", "simpledb_generate_weights")
    inference_bin = ensure_binary(build_dir / "llama_inference_example", "llama_inference_example")

    selected_experiments = set(parse_csv_names(args.experiments))
    invalid_experiments = selected_experiments.difference({"prefill", "logits"})
    if invalid_experiments:
        raise ValueError(f"unsupported experiments: {sorted(invalid_experiments)}")

    selected_models = select_configs(MODELS, parse_csv_names(args.models), "models")
    prefill_platforms = select_configs(
        PLATFORMS, parse_csv_names(args.prefill_platforms), "prefill platforms"
    )
    logits_platforms = select_configs(
        PLATFORMS, parse_csv_names(args.logits_platform), "logits platform"
    )
    if len(logits_platforms) != 1:
        raise ValueError("exactly one logits platform must be selected")
    logits_platform = logits_platforms[0]

    prefill_seq_lens = parse_csv_ints(args.prefill_seq_lens)
    logits_seq_lens = parse_csv_ints(args.logits_seq_lens)
    env_overrides = parse_env_overrides(args.env)
    command_prefix = shlex.split(args.command_prefix)

    print(
        f"[llama_experiment] Using memory budget fraction={PLATFORM_MEMORY_FRACTION:.2f} "
        "for platform presets"
    )
    for platform in sorted({*prefill_platforms, logits_platform}, key=lambda item: item.platform_id):
        print(
            f"[llama_experiment] Platform {platform.platform_id}: "
            f"threads={platform.threads} total_memory_bytes={platform.total_memory_bytes} "
            f"arena_size_mb={platform.arena_size_mb}"
        )

    db_paths: dict[str, Path] = {}
    full_preload_files: dict[str, Path] = {}
    created_db_paths: list[Path] = []

    print("[llama_experiment] Preparing model DBs and full-preload files")
    for model in selected_models:
        db_path = db_root / f"llama_{model.model_id}_chunk{args.chunk_size}"
        metadata_path = generate_model_db(
            generator_bin=generator_bin,
            model=model,
            chunk_size=args.chunk_size,
            precision=args.precision,
            seed=args.seed,
            db_path=db_path,
            log_dir=log_dir / "db_generation",
            force_regenerate_dbs=args.force_regenerate_dbs,
        )
        db_paths[model.model_id] = db_path
        full_preload_files[model.model_id] = prepare_full_preload_file(
            model=model,
            metadata_path=metadata_path,
            preload_dir=preload_dir,
            chunk_size=args.chunk_size,
        )
        created_db_paths.append(db_path)

    prefill_rows: list[dict[str, object]] = []
    logits_rows: list[dict[str, object]] = []

    if "prefill" in selected_experiments:
        prefill_rows = run_prefill_experiment(
            inference_bin=inference_bin,
            selected_models=selected_models,
            prefill_platforms=prefill_platforms,
            prefill_seq_lens=prefill_seq_lens,
            chunk_size=args.chunk_size,
            prefetch_window=args.prefetch_window,
            seed=args.seed,
            repeats=args.prefill_repeats,
            db_paths=db_paths,
            env_overrides=env_overrides,
            command_prefix=command_prefix,
            report_dir=report_dir,
            log_dir=log_dir,
            force_rerun=args.force_rerun,
        )

    if "logits" in selected_experiments:
        logits_rows = run_logits_experiment(
            inference_bin=inference_bin,
            selected_models=selected_models,
            logits_platform=logits_platform,
            logits_seq_lens=logits_seq_lens,
            chunk_size=args.chunk_size,
            prefetch_window=args.prefetch_window,
            seed=args.seed,
            repeats=args.logits_repeats,
            db_paths=db_paths,
            full_preload_files=full_preload_files,
            env_overrides=env_overrides,
            command_prefix=command_prefix,
            report_dir=report_dir,
            log_dir=log_dir,
            force_rerun=args.force_rerun,
        )

    if prefill_rows:
        prefill_rows.sort(key=lambda row: str(row["run_id"]))
        write_csv(derived_dir / "prefill_results.csv", PREFILL_RAW_FIELDNAMES, prefill_rows)
        write_csv(
            derived_dir / "prefill_summary.csv",
            PREFILL_SUMMARY_FIELDNAMES,
            summarize_prefill(prefill_rows),
        )

    if logits_rows:
        logits_rows.sort(key=lambda row: str(row["run_id"]))
        write_csv(derived_dir / "logits_error_results.csv", LOGITS_RAW_FIELDNAMES, logits_rows)
        write_csv(
            derived_dir / "logits_error_summary.csv",
            LOGITS_SUMMARY_FIELDNAMES,
            summarize_logits(logits_rows),
        )

    manifest = {
        "results_dir": str(results_dir),
        "db_root": str(db_root),
        "models": [model.model_id for model in selected_models],
        "prefill_platforms": [platform.platform_id for platform in prefill_platforms],
        "logits_platform": logits_platform.platform_id,
        "prefill_seq_lens": prefill_seq_lens,
        "logits_seq_lens": logits_seq_lens,
        "chunk_size": args.chunk_size,
        "prefetch_window": args.prefetch_window,
        "precision": args.precision,
        "seed": args.seed,
        "experiments": sorted(selected_experiments),
        "prefill_repeats": args.prefill_repeats,
        "logits_repeats": args.logits_repeats,
        "logits_comparison": {
            "baseline": "no_preload_bufferpool",
            "reference": "full_preload_disable_bufferpool",
        },
    }
    manifest_path = derived_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.cleanup_dbs:
        for db_path in created_db_paths:
            if db_path.exists():
                print(f"[llama_experiment] Deleting DB: {db_path}")
                shutil.rmtree(db_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
