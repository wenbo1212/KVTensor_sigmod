#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_binary(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required binary not found: {path} ({name})")
    return path


def run_command(command: list[str], env: dict[str, str], log_path: Optional[Path]) -> None:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(shlex.quote(token) for token in command) + "\n\n")
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"command failed with exit code {completed.returncode}: {log_path}")
        return

    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def db_matches_chunk_size(db_path: Path, chunk_size: int) -> bool:
    metadata_path = db_path / "metadata.jsonl"
    if not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                return int(payload.get("chunk_size", -1)) == chunk_size
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return False


def ensure_db(args: argparse.Namespace, env: dict[str, str], log_path: Optional[Path]) -> Path:
    build_dir = Path(args.build_dir).resolve()
    db_path = Path(args.db_path).resolve()
    generator_bin = ensure_binary(build_dir / "simpledb_generate_weights", "simpledb_generate_weights")

    if db_matches_chunk_size(db_path, args.chunk_size):
        return db_path

    if db_path.exists() and args.force_regenerate_db:
        shutil.rmtree(db_path)

    db_path.mkdir(parents=True, exist_ok=True)
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
        str(args.chunk_size),
        "--vocab-size",
        str(args.vocab_size),
        "--seed",
        str(args.seed),
        "--precision",
        args.precision,
    ]

    generator_log_path = None
    if log_path is not None:
        generator_log_path = log_path.with_name(log_path.stem + "_generate.log")
    run_command(generate_command, env, generator_log_path)
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run llama_inference_example once and emit a trace JSON report.")
    parser.add_argument("--build-dir", default=str(REPO_ROOT / "cpp" / "build"))
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--log", default="", help="Optional log path")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--arena-size-mb", type=int, required=True)
    parser.add_argument("--prefetch-window", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16", "int8"])
    parser.add_argument("--preload-file", default="")
    parser.add_argument("--preload-only", action="store_true")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--disable-bufferpool", action="store_true")
    parser.add_argument("--force-regenerate-db", action="store_true")
    parser.add_argument("--command-prefix", default="", help="Optional launcher prefix, split with shlex")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    binary = ensure_binary(build_dir / "llama_inference_example", "llama_inference_example")
    log_path = Path(args.log) if args.log else None

    env = os.environ.copy()
    if args.threads > 0:
        env["OMP_NUM_THREADS"] = str(args.threads)

    db_path = ensure_db(args, env, log_path)

    command = shlex.split(args.command_prefix) + [
        str(binary),
        "--db-path", str(db_path),
        "--hidden-dim", str(args.hidden_dim),
        "--num-layers", str(args.num_layers),
        "--num-heads", str(args.num_heads),
        "--num-kv-heads", str(args.num_kv_heads),
        "--seq-len", str(args.seq_len),
        "--decode-steps", str(args.decode_steps),
        "--chunk-size", str(args.chunk_size),
        "--arena-size-mb", str(args.arena_size_mb),
        "--prefetch-window", str(args.prefetch_window),
        "--vocab-size", str(args.vocab_size),
        "--seed", str(args.seed),
        "--profile",
        "--report-json", args.report_json,
    ]
    if args.preload_file:
        command.extend(["--preload-file", args.preload_file])
    if args.preload_only:
        command.append("--preload-only")
    if args.in_memory:
        command.append("--in-memory")
    if args.disable_bufferpool:
        command.append("--disable-bufferpool")
    run_command(command, env, log_path)

    print(f"Wrote trace report: {args.report_json}")


if __name__ == "__main__":
    main()
