#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _read_size_bytes(row: Dict[str, str]) -> int:
    direct = _to_int(row.get("read_size_bytes"), 0)
    if direct > 0:
        return direct
    direct = _to_int(row.get("window_bytes"), 0)
    if direct > 0:
        return direct
    chunk_bytes = _to_int(row.get("chunk_size_bytes"), 0)
    window_chunks = max(1, _to_int(row.get("window_chunks"), 1))
    if chunk_bytes > 0:
        return chunk_bytes * window_chunks
    bytes_transferred = _to_float(row.get("bytes_transferred"), 0.0)
    request_count = max(1, _to_int(row.get("request_count"), 1))
    if bytes_transferred > 0.0:
        return int(bytes_transferred / float(request_count))
    return 0


def fit_hardware_kv(kv_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, int], Dict[str, List[float]]] = defaultdict(
        lambda: {"latency_ms": [], "throughput_mb_s": []}
    )
    for row in kv_rows:
        read_size = _read_size_bytes(row)
        if read_size <= 0:
            continue
        thread_count = max(1, _to_int(row.get("thread_count"), 1))
        avg_ms = _to_float(row.get("avg_ms"), 0.0)
        throughput = _to_float(row.get("throughput_mb_s"), 0.0)
        if avg_ms <= 0.0 or throughput <= 0.0:
            continue
        key = (read_size, thread_count)
        groups[key]["latency_ms"].append(avg_ms)
        groups[key]["throughput_mb_s"].append(throughput)

    out: List[Dict[str, Any]] = []
    for (read_size, threads), vals in groups.items():
        out.append(
            {
                "read_size_bytes": read_size,
                "thread_count": threads,
                "lambda_ms": _mean(vals["latency_ms"]),
                "beta_mb_s": _mean(vals["throughput_mb_s"]),
                "samples": len(vals["latency_ms"]),
            }
        )
    out.sort(key=lambda r: (r["read_size_bytes"], r["thread_count"]))
    return out


def _map_operator_class(row: Dict[str, str]) -> str:
    op_class = str(row.get("operator_class", "")).strip().lower()
    if op_class:
        return op_class
    workload = str(row.get("workload", "")).strip().lower()
    if workload == "gemv":
        return "decode_gemv"
    if workload == "gemm":
        return "prefill_gemm"
    if workload in ("conv", "diffusion_gemm"):
        return "diffusion_gemm"
    return "prefill_gemm"


def fit_hardware_compute(compute_rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
    shape_groups: Dict[Tuple[str, str, int, int, int, int], List[float]] = defaultdict(list)
    class_groups: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for row in compute_rows:
        gflops = _to_float(row.get("throughput_gflops"), 0.0)
        if gflops <= 0.0:
            continue
        op_class = _map_operator_class(row)
        dtype = str(row.get("dtype", "float32")).strip().lower()
        threads = max(1, _to_int(row.get("thread_count"), 1))
        m = _to_int(row.get("m"), 0)
        k = _to_int(row.get("k"), 0)
        n = _to_int(row.get("n"), 0)
        shape_groups[(op_class, dtype, threads, m, k, n)].append(gflops)
        class_groups[(op_class, dtype, threads)].append(gflops)

    by_shape: List[Dict[str, Any]] = []
    for (op_class, dtype, threads, m, k, n), values in shape_groups.items():
        by_shape.append(
            {
                "operator_class": op_class,
                "dtype": dtype,
                "thread_count": threads,
                "m": m,
                "k": k,
                "n": n,
                "throughput_gflops": _mean(values),
                "samples": len(values),
            }
        )
    by_shape.sort(key=lambda r: (r["operator_class"], r["dtype"], r["thread_count"], r["m"], r["k"], r["n"]))

    by_class: List[Dict[str, Any]] = []
    for (op_class, dtype, threads), values in class_groups.items():
        by_class.append(
            {
                "operator_class": op_class,
                "dtype": dtype,
                "thread_count": threads,
                "throughput_gflops": _mean(values),
                "samples": len(values),
            }
        )
    by_class.sort(key=lambda r: (r["operator_class"], r["dtype"], r["thread_count"]))
    return {"by_shape": by_shape, "by_class": by_class}


def fit_beta_init(
    runtime_rows: List[Dict[str, str]],
    kv_profiles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    startup_samples: List[float] = []
    for row in runtime_rows:
        workload = str(row.get("workload", "")).strip().lower()
        if workload not in ("static_load", "preload", "startup"):
            continue
        t = _to_float(row.get("throughput_mb_s"), 0.0)
        if t > 0.0:
            startup_samples.append(t)

    if not startup_samples:
        for row in kv_profiles:
            t = _to_float(row.get("beta_mb_s"), 0.0)
            if t > 0.0:
                startup_samples.append(t)

    beta_init = _mean(startup_samples) if startup_samples else 1.0
    return {
        "beta_init_mb_s": beta_init,
        "samples": len(startup_samples),
    }


def fit_runtime_profile(runtime_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, int, str, int], Dict[str, List[float]]] = defaultdict(
        lambda: {
            "request_count": [],
            "bytes_transferred": [],
            "io_total_ms": [],
            "io_exposed_ms": [],
            "compute_ms": [],
            "merge_ms": [],
            "runtime_overhead_ms": [],
            "throughput_mb_s": [],
        }
    )

    for row in runtime_rows:
        chunk_size = _to_int(row.get("chunk_size_param"), 0)
        prefetch_window = _to_int(row.get("prefetch_depth"), 0)
        dtype = str(row.get("dtype", "")).strip().lower()
        threads = _to_int(row.get("thread_count"), 0)
        if chunk_size <= 0 or prefetch_window <= 0 or threads <= 0:
            continue
        key = (chunk_size, prefetch_window, dtype, threads)
        groups[key]["request_count"].append(_to_float(row.get("request_count"), 0.0))
        groups[key]["bytes_transferred"].append(_to_float(row.get("bytes_transferred"), 0.0))
        groups[key]["io_total_ms"].append(_to_float(row.get("io_total_ms"), 0.0))
        groups[key]["io_exposed_ms"].append(_to_float(row.get("io_exposed_ms"), 0.0))
        groups[key]["compute_ms"].append(_to_float(row.get("compute_ms"), 0.0))
        groups[key]["merge_ms"].append(_to_float(row.get("merge_ms"), 0.0))
        groups[key]["runtime_overhead_ms"].append(_to_float(row.get("runtime_overhead_ms"), 0.0))
        groups[key]["throughput_mb_s"].append(_to_float(row.get("throughput_mb_s"), 0.0))

    out: List[Dict[str, Any]] = []
    for (chunk_size, prefetch_window, dtype, threads), values in groups.items():
        io_total_ms = _mean(values["io_total_ms"])
        io_exposed_ms = _mean(values["io_exposed_ms"])
        out.append(
            {
                "chunk_size": chunk_size,
                "prefetch_window": prefetch_window,
                "dtype": dtype,
                "thread_count": threads,
                "request_count": _mean(values["request_count"]),
                "bytes_transferred": _mean(values["bytes_transferred"]),
                "io_total_ms": io_total_ms,
                "io_exposed_ms": io_exposed_ms,
                "io_exposed_ratio": (io_exposed_ms / io_total_ms) if io_total_ms > 0.0 else 1.0,
                "compute_ms": _mean(values["compute_ms"]),
                "merge_ms": _mean(values["merge_ms"]),
                "runtime_overhead_ms": _mean(values["runtime_overhead_ms"]),
                "merge_ms_per_request": _mean(values["merge_ms"]) / max(1e-9, _mean(values["request_count"])),
                "runtime_overhead_ms_per_request": _mean(values["runtime_overhead_ms"]) / max(1e-9, _mean(values["request_count"])),
                "throughput_mb_s": _mean(values["throughput_mb_s"]),
                "samples": len(values["io_total_ms"]),
            }
        )
    out.sort(key=lambda r: (r["chunk_size"], r["prefetch_window"], r["dtype"], r["thread_count"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hardware calibration profile for ranking model."
    )
    parser.add_argument("--kv-csv", required=True, help="Path to kv benchmark CSV")
    parser.add_argument("--compute-csv", required=True, help="Path to compute benchmark CSV")
    parser.add_argument("--runtime-csv", default="", help="Optional runtime benchmark CSV")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--out-json", default="hardware_profile.json", help="Output hardware profile JSON")
    parser.add_argument(
        "--stall-alpha",
        type=float,
        default=0.5,
        help="Alpha for phi(R)=alpha/R when R<R0",
    )
    parser.add_argument(
        "--stall-r0",
        type=float,
        default=2.0,
        help="R0 threshold for ring adequacy penalty",
    )
    args = parser.parse_args()

    kv_rows = _read_csv(args.kv_csv)
    compute_rows = _read_csv(args.compute_csv)
    runtime_rows = _read_csv(args.runtime_csv)

    kv_profiles = fit_hardware_kv(kv_rows)
    compute_profiles = fit_hardware_compute(compute_rows)
    startup = fit_beta_init(runtime_rows, kv_profiles)
    runtime_profiles = fit_runtime_profile(runtime_rows)

    hardware_profile = {
        "generated_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "inputs": {
            "kv_csv": args.kv_csv,
            "compute_csv": args.compute_csv,
            "runtime_csv": args.runtime_csv,
        },
        "sequential_io_profile": kv_profiles,
        "compute_profile": compute_profiles,
        "startup_profile": startup,
        "runtime_profile": {
            "rows": runtime_profiles,
        },
        "stall_model": {
            "alpha": args.stall_alpha,
            "r0": args.stall_r0,
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hardware_profile, f, indent=2)
    print(f"Wrote hardware profile: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
