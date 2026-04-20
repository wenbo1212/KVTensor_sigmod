#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any, Dict, List, Set, Tuple


WEIGHT_OPERATOR_CLASSES = {
    "attn_qkv_proj",
    "attn_o_proj",
    "ffn_down_proj",
    "ffn_gate_up_proj",
    "output_proj",
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_supported_dtype(dtype: str) -> str:
    value = str(dtype).lower()
    if value in ("float32", "fp32"):
        return "float32"
    if value in ("bfloat16", "bf16", "bfloat"):
        return "bfloat16"
    if value in ("int8", "i8"):
        return "int8"
    raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes: float32, bfloat16, int8.")


def _dtype_size_bytes(dtype: str) -> int:
    value = _normalize_supported_dtype(dtype)
    if value == "float32":
        return 4
    if value == "bfloat16":
        return 2
    return 1


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _filter_by_threads(rows: List[Dict[str, Any]], thread_count: int) -> List[Dict[str, Any]]:
    exact = [row for row in rows if _to_int(row.get("thread_count"), -1) == thread_count]
    return exact if exact else rows


def _chunk_size_bytes_from_workload(workload: Dict[str, Any], chunk_size: int, dtype: str) -> int:
    axis = str(workload.get("chunk_axis", "column")).strip().lower()
    dtype_bytes = _dtype_size_bytes(dtype)
    if axis == "column":
        chunk_rows = _to_int(workload.get("chunk_rows"), 0)
        if chunk_rows <= 0:
            raise ValueError("workload_profile must define chunk_rows for column chunking")
        return chunk_size * chunk_rows * dtype_bytes
    if axis == "row":
        chunk_cols = _to_int(workload.get("chunk_cols"), 0)
        if chunk_cols <= 0:
            raise ValueError("workload_profile must define chunk_cols for row chunking")
        return chunk_size * chunk_cols * dtype_bytes
    raise ValueError(f"Unsupported chunk_axis '{axis}'")


def _lookup_io_profile(hardware: Dict[str, Any], read_size_bytes: int, thread_count: int) -> Tuple[float, float]:
    rows = _filter_by_threads(hardware.get("sequential_io_profile", []), thread_count)
    if not rows:
        return 1.0, 1.0
    hit = min(rows, key=lambda row: abs(_to_int(row.get("read_size_bytes"), 0) - read_size_bytes))
    return (
        max(1e-9, _to_float(hit.get("lambda_ms"), 1.0)),
        max(1e-9, _to_float(hit.get("beta_mb_s"), 1.0)),
    )


def _lookup_compute_gflops(
    hardware: Dict[str, Any],
    operator_class: str,
    dtype: str,
    thread_count: int,
    m: int,
    k: int,
    n: int,
) -> float:
    compute_profile = hardware.get("compute_profile", {})
    rows = compute_profile.get("by_shape", []) or compute_profile.get("operator_classes", [])
    rows = [row for row in rows if str(row.get("operator_class", "")) == operator_class] or rows
    rows = [row for row in rows if str(row.get("dtype", "")) == dtype] or rows
    rows = _filter_by_threads(rows, thread_count)
    shape_rows = [row for row in rows if all(key in row for key in ("m", "k", "n"))]
    if shape_rows:
        hit = min(
            shape_rows,
            key=lambda row: (
                abs(_to_int(row.get("m"), m) - m),
                abs(_to_int(row.get("k"), k) - k),
                abs(_to_int(row.get("n"), n) - n),
            ),
        )
        gflops = max(1e-9, _to_float(hit.get("throughput_gflops"), 1.0))
        hit_m = max(1, _to_int(hit.get("m"), m))
        hit_n = max(1, _to_int(hit.get("n"), n))
        # Extrapolate conservatively when we are forced below the smallest measured shape.
        if m > 0 and m < hit_m:
            gflops *= max(0.2, math.sqrt(float(m) / float(hit_m)))
        if n > 0 and n < hit_n:
            gflops *= max(0.15, math.sqrt(float(n) / float(hit_n)))
        return max(1e-9, gflops)

    class_rows = compute_profile.get("by_class", []) or rows
    class_rows = [row for row in class_rows if str(row.get("operator_class", "")) == operator_class] or class_rows
    class_rows = [row for row in class_rows if str(row.get("dtype", "")) == dtype] or class_rows
    class_rows = _filter_by_threads(class_rows, thread_count)
    if not class_rows:
        return 1.0
    return max(1e-9, _to_float(class_rows[0].get("throughput_gflops"), 1.0))


def _lookup_runtime_row(
    hardware: Dict[str, Any],
    chunk_size: int,
    prefetch_window: int,
    dtype: str,
    thread_count: int,
) -> Dict[str, Any]:
    rows = hardware.get("runtime_profile", {}).get("rows", [])
    rows = [row for row in rows if str(row.get("dtype", "")) == dtype] or rows
    rows = _filter_by_threads(rows, thread_count)
    if not rows:
        return {}
    return min(
        rows,
        key=lambda row: (
            abs(_to_int(row.get("chunk_size"), 0) - chunk_size),
            abs(_to_int(row.get("prefetch_window"), 0) - prefetch_window),
        ),
    )


def _lookup_runtime_rows_for_chunk(
    hardware: Dict[str, Any],
    chunk_size: int,
    dtype: str,
    thread_count: int,
) -> List[Dict[str, Any]]:
    rows = hardware.get("runtime_profile", {}).get("rows", [])
    rows = [row for row in rows if str(row.get("dtype", "")) == dtype] or rows
    rows = _filter_by_threads(rows, thread_count)
    return [row for row in rows if _to_int(row.get("chunk_size"), 0) == chunk_size]


def _reference_structured_metrics(
    hardware: Dict[str, Any],
    workload: Dict[str, Any],
    dtype: str,
    thread_count: int,
) -> Dict[str, float]:
    reference = workload.get("reference", {})
    ref_chunk_size = max(1, _to_int(reference.get("chunk_size"), workload.get("system", {}).get("chunk_size", 0)))
    ref_prefetch_window = max(1, _to_int(reference.get("prefetch_window"), workload.get("system", {}).get("prefetch_window", 1)))
    chunk_size_bytes = _chunk_size_bytes_from_workload(workload, ref_chunk_size, dtype)

    io_total_ms = 0.0
    streamed_chunks = 0.0
    streamed_windows = 0.0
    for entry in workload.get("matrix_accesses", []):
        matrix_bytes = max(0.0, _to_float(entry.get("matrix_bytes"), 0.0))
        logical_passes = max(0.0, _to_float(entry.get("logical_passes"), 0.0))
        if matrix_bytes <= 0.0 or logical_passes <= 0.0:
            continue
        chunk_count = _matrix_chunk_count(entry, ref_chunk_size)
        if chunk_count <= 0:
            continue
        matrix_streamed_bytes = matrix_bytes * logical_passes
        matrix_streamed_chunks = chunk_count * logical_passes
        matrix_streamed_windows = math.ceil(chunk_count / float(ref_prefetch_window)) * logical_passes
        read_size_bytes = max(1, min(ref_prefetch_window, chunk_count) * chunk_size_bytes)
        lambda_ms, beta_mb_s = _lookup_io_profile(hardware, read_size_bytes, thread_count)
        beta_bytes_per_ms = beta_mb_s * 1024.0 * 1024.0 / 1000.0
        io_total_ms += matrix_streamed_windows * lambda_ms + (matrix_streamed_bytes / max(1e-9, beta_bytes_per_ms))
        streamed_chunks += matrix_streamed_chunks
        streamed_windows += matrix_streamed_windows

    runtime_row = _lookup_runtime_row(hardware, ref_chunk_size, ref_prefetch_window, dtype, thread_count)
    runtime_request_count = max(1.0, _to_float(runtime_row.get("request_count"), 1.0))
    raw_runtime_overhead_ms = max(0.0, _to_float(runtime_row.get("runtime_overhead_ms"), 0.0)) * (
        streamed_chunks / runtime_request_count if streamed_chunks > 0.0 else 0.0
    )
    raw_merge_ms = max(0.0, _to_float(runtime_row.get("merge_ms"), 0.0)) * (
        streamed_chunks / runtime_request_count if streamed_chunks > 0.0 else 0.0
    )
    raw_io_exposed_ms = io_total_ms * min(1.0, max(0.0, _to_float(runtime_row.get("io_exposed_ratio"), 1.0)))
    return {
        "chunk_size": float(ref_chunk_size),
        "prefetch_window": float(ref_prefetch_window),
        "chunk_size_bytes": float(chunk_size_bytes),
        "io_total_ms": io_total_ms,
        "streamed_chunks": streamed_chunks,
        "streamed_windows": streamed_windows,
        "raw_runtime_overhead_ms": raw_runtime_overhead_ms,
        "raw_merge_ms": raw_merge_ms,
        "raw_io_exposed_ms": raw_io_exposed_ms,
    }


def _beta_init_mb_s(hardware: Dict[str, Any], workload: Dict[str, Any]) -> float:
    startup = workload.get("startup", {})
    trace_beta = _to_float(startup.get("throughput_mb_s"), 0.0)
    if trace_beta > 0.0:
        return trace_beta
    return max(1e-9, _to_float(hardware.get("startup_profile", {}).get("beta_init_mb_s"), 1.0))


def _include_startup(mode: str, objective: str) -> bool:
    if objective == "cold":
        return True
    if objective == "warm":
        return False
    return mode == "prefill"


def _preload_candidates(workload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = workload.get("matrix_catalog", [])
    rows = [row for row in rows if _to_int(row.get("priority_rank"), -1) >= 0]
    return sorted(rows, key=lambda row: (_to_int(row.get("priority_rank"), -1), str(row.get("matrix_id", ""))))


def _select_static_matrices(workload: Dict[str, Any], target_static_bytes: int) -> Tuple[Set[str], int]:
    selected: Set[str] = set()
    selected_bytes = 0
    for row in _preload_candidates(workload):
        size_bytes = max(0, _to_int(row.get("size_bytes"), 0))
        matrix_id = str(row.get("matrix_id", ""))
        if not matrix_id or size_bytes <= 0:
            continue
        if selected_bytes + size_bytes > target_static_bytes:
            break
        selected.add(matrix_id)
        selected_bytes += size_bytes
    return selected, selected_bytes


def _matrix_chunk_count(entry: Dict[str, Any], candidate_chunk_size: int) -> int:
    rows = max(0, _to_int(entry.get("rows"), 0))
    cols = max(0, _to_int(entry.get("cols"), 0))
    split_mode = str(entry.get("split_mode", "column")).lower()
    if candidate_chunk_size <= 0:
        return 0
    if split_mode == "row":
        return (rows + candidate_chunk_size - 1) // candidate_chunk_size
    return (cols + candidate_chunk_size - 1) // candidate_chunk_size


def _matrix_full_output_width(entry: Dict[str, Any]) -> int:
    rows = max(0, _to_int(entry.get("rows"), 0))
    cols = max(0, _to_int(entry.get("cols"), 0))
    split_mode = str(entry.get("split_mode", "column")).lower()
    return rows if split_mode == "row" else cols


def _matrix_execution_mode(entry: Dict[str, Any], selected: bool) -> str:
    operator_class = str(entry.get("operator_class", entry.get("matrix_group", "")))
    if selected:
        mode = str(entry.get("resident_execution", _default_resident_execution_mode(operator_class))).strip().lower()
    else:
        mode = str(
            entry.get(
                "streamed_execution",
                _default_streamed_execution_mode(_to_int(entry.get("priority_rank"), -1)),
            )
        ).strip().lower()
    return mode or ("dense" if selected else "chunked")


def _default_resident_execution_mode(operator_class: str) -> str:
    return "chunked" if operator_class == "ffn_gate_up_proj" else "dense"


def _default_streamed_execution_mode(priority_rank: int) -> str:
    return "chunked" if priority_rank >= 0 else "none"


def _derive_matrix_compute_templates(workload: Dict[str, Any]) -> List[Dict[str, Any]]:
    templates = workload.get("matrix_compute_templates", [])
    if templates:
        return templates
    shape_by_class: Dict[str, Dict[str, int]] = {}
    for item in workload.get("compute_buckets", []):
        operator_class = str(item.get("operator_class", ""))
        if operator_class not in WEIGHT_OPERATOR_CLASSES:
            continue
        calls = _to_float(item.get("calls"), 0.0)
        current = shape_by_class.get(operator_class)
        if current is None or calls > current["calls"]:
            shape_by_class[operator_class] = {
                "m": _to_int(item.get("m"), 0),
                "k": _to_int(item.get("k"), 0),
                "n": _to_int(item.get("n"), 0),
                "calls": calls,
            }

    derived: List[Dict[str, Any]] = []
    for item in workload.get("matrix_accesses", []):
        operator_class = str(item.get("matrix_group", ""))
        if operator_class not in WEIGHT_OPERATOR_CLASSES:
            continue
        split_mode = str(item.get("split_mode", "column")).lower()
        rows = _to_int(item.get("rows"), 0)
        cols = _to_int(item.get("cols"), 0)
        shape_hint = shape_by_class.get(operator_class, {})
        derived.append(
            {
                "matrix_id": str(item.get("matrix_id", "")),
                "matrix_group": operator_class,
                "operator_class": operator_class,
                "resident_execution": str(item.get("resident_execution", _default_resident_execution_mode(operator_class))),
                "streamed_execution": str(item.get("streamed_execution", _default_streamed_execution_mode(_to_int(item.get("priority_rank"), -1)))),
                "m": _to_int(shape_hint.get("m"), 0),
                "k": rows if split_mode != "row" else cols,
                "full_n": cols if split_mode != "row" else rows,
                "reference_chunk_size": _to_int(item.get("reference_chunk_size"), 0),
                "logical_passes": _to_float(item.get("logical_passes"), 0.0),
            }
        )
    return derived


def _matmul_cost_ms(
    hardware: Dict[str, Any],
    operator_class: str,
    dtype: str,
    thread_count: int,
    m: int,
    k: int,
    n: int,
    calls: float,
) -> float:
    if m <= 0 or k <= 0 or n <= 0 or calls <= 0.0:
        return 0.0
    gflops = _lookup_compute_gflops(hardware, operator_class, dtype, thread_count, m, k, n)
    flops = 2.0 * float(m) * float(k) * float(n) * float(calls)
    return (flops / (gflops * 1e9)) * 1000.0


def _candidate_matrix_compute_ms(
    hardware: Dict[str, Any],
    workload: Dict[str, Any],
    selected_matrix_ids: Set[str],
    dtype: str,
    threads: int,
    chunk_size: int,
) -> Tuple[float, float, float]:
    matrix_templates = _derive_matrix_compute_templates(workload)
    if not matrix_templates:
        return 0.0, 0.0, 0.0

    compute_ms = 0.0
    chunked_execution_calls = 0.0
    resident_dense_calls = 0.0
    for entry in matrix_templates:
        matrix_id = str(entry.get("matrix_id", ""))
        if not matrix_id:
            continue
        operator_class = str(entry.get("operator_class", entry.get("matrix_group", "")))
        m = _to_int(entry.get("m"), 0)
        k = _to_int(entry.get("k"), 0)
        full_n = _to_int(entry.get("full_n"), 0)
        logical_passes = max(0.0, _to_float(entry.get("logical_passes"), 0.0))
        if operator_class not in WEIGHT_OPERATOR_CLASSES or m <= 0 or k <= 0 or full_n <= 0 or logical_passes <= 0.0:
            continue
        selected = matrix_id in selected_matrix_ids
        execution_mode = _matrix_execution_mode(entry, selected)
        if execution_mode == "dense":
            resident_dense_calls += logical_passes
            compute_ms += _matmul_cost_ms(hardware, operator_class, dtype, threads, m, k, full_n, logical_passes)
            continue

        candidate_chunk_n = max(1, min(chunk_size, full_n))
        full_chunks = full_n // candidate_chunk_n
        tail_n = full_n % candidate_chunk_n
        chunked_execution_calls += (full_chunks + (1 if tail_n > 0 else 0)) * logical_passes
        if full_chunks > 0:
            compute_ms += _matmul_cost_ms(
                hardware,
                operator_class,
                dtype,
                threads,
                m,
                k,
                candidate_chunk_n,
                full_chunks * logical_passes,
            )
        if tail_n > 0:
            compute_ms += _matmul_cost_ms(
                hardware,
                operator_class,
                dtype,
                threads,
                m,
                k,
                tail_n,
                logical_passes,
            )
    return compute_ms, chunked_execution_calls, resident_dense_calls


def _score_candidate_structured(
    hardware: Dict[str, Any],
    workload: Dict[str, Any],
    candidate: Dict[str, Any],
    system_total_memory_bytes: int,
    objective: str,
) -> Dict[str, Any]:
    candidate_id = str(candidate.get("id", "candidate"))
    dtype = _normalize_supported_dtype(str(candidate.get("dtype", workload.get("dtype", "bfloat16"))))
    threads = max(1, _to_int(candidate.get("thread_count", workload.get("system", {}).get("thread_count", 1)), 1))
    chunk_size = max(1, _to_int(candidate.get("chunk_size"), 0))
    prefetch_window = max(1, _to_int(candidate.get("prefetch_window"), 1))
    target_static_bytes = max(
        0,
        _to_int(
            candidate.get(
                "requested_static_bytes",
                candidate.get("static_memory_bytes", candidate.get("selected_static_bytes", 0)),
            ),
            0,
        ),
    )
    bufferpool_bytes = max(0, _to_int(candidate.get("bufferpool_bytes"), 0))

    feasible = True
    infeasible_reasons: List[str] = []
    if target_static_bytes + bufferpool_bytes > system_total_memory_bytes:
        feasible = False
        infeasible_reasons.append("requested_static_bytes + bufferpool_bytes exceeds total memory")

    chunk_size_bytes = _chunk_size_bytes_from_workload(workload, chunk_size, dtype)
    ring_chunks = bufferpool_bytes // max(1, chunk_size_bytes)
    if ring_chunks < 1:
        feasible = False
        infeasible_reasons.append("bufferpool capacity B=floor(M_b/c_bytes) is < 1")

    selected_matrix_ids, actual_selected_static_bytes = _select_static_matrices(workload, target_static_bytes)

    matrix_accesses = workload.get("matrix_accesses", [])
    io_total_ms = 0.0
    streamed_bytes = 0.0
    streamed_chunks = 0.0
    streamed_windows = 0.0
    resident_chunked_calls = 0.0
    for entry in matrix_accesses:
        matrix_id = str(entry.get("matrix_id", ""))
        if not matrix_id or matrix_id in selected_matrix_ids:
            if matrix_id and matrix_id in selected_matrix_ids and _matrix_execution_mode(entry, True) == "chunked":
                logical_passes = max(0.0, _to_float(entry.get("logical_passes"), 0.0))
                resident_chunked_calls += _matrix_chunk_count(entry, chunk_size) * logical_passes
            continue
        matrix_bytes = max(0.0, _to_float(entry.get("matrix_bytes"), 0.0))
        logical_passes = max(0.0, _to_float(entry.get("logical_passes"), 0.0))
        if matrix_bytes <= 0.0 or logical_passes <= 0.0:
            continue
        chunk_count = _matrix_chunk_count(entry, chunk_size)
        if chunk_count <= 0:
            continue
        matrix_streamed_bytes = matrix_bytes * logical_passes
        matrix_streamed_chunks = chunk_count * logical_passes
        matrix_windows_per_pass = math.ceil(chunk_count / float(prefetch_window))
        matrix_streamed_windows = matrix_windows_per_pass * logical_passes
        read_size_bytes = max(1, min(prefetch_window, chunk_count) * chunk_size_bytes)
        lambda_ms, beta_mb_s = _lookup_io_profile(hardware, read_size_bytes, threads)
        beta_bytes_per_ms = beta_mb_s * 1024.0 * 1024.0 / 1000.0
        io_total_ms += matrix_streamed_windows * lambda_ms + (matrix_streamed_bytes / max(1e-9, beta_bytes_per_ms))
        streamed_bytes += matrix_streamed_bytes
        streamed_chunks += matrix_streamed_chunks
        streamed_windows += matrix_streamed_windows

    compute_ms = 0.0
    chunked_execution_calls = streamed_chunks + resident_chunked_calls
    matrix_compute_ms, matrix_chunked_calls, resident_dense_calls = _candidate_matrix_compute_ms(
        hardware,
        workload,
        selected_matrix_ids,
        dtype,
        threads,
        chunk_size,
    )
    if matrix_compute_ms > 0.0:
        compute_ms += matrix_compute_ms
        chunked_execution_calls = max(chunked_execution_calls, matrix_chunked_calls)

    extra_compute_buckets = workload.get("extra_compute_buckets")
    if extra_compute_buckets is not None:
        bucket_source = extra_compute_buckets
    else:
        bucket_source = [
            bucket
            for bucket in workload.get("compute_buckets", [])
            if str(bucket.get("operator_class", "")) not in WEIGHT_OPERATOR_CLASSES
        ]
    for bucket in bucket_source:
        operator_class = str(bucket.get("operator_class", workload.get("operator_class", "prefill_gemm")))
        m = _to_int(bucket.get("m"), 0)
        k = _to_int(bucket.get("k"), 0)
        n = _to_int(bucket.get("n"), 0)
        flops = max(0.0, _to_float(bucket.get("flops"), 0.0))
        if flops <= 0.0:
            continue
        gflops = _lookup_compute_gflops(hardware, operator_class, dtype, threads, m, k, n)
        compute_ms += (flops / (gflops * 1e9)) * 1000.0
    if compute_ms <= 0.0:
        reference = workload.get("reference", {})
        gflops = _lookup_compute_gflops(
            hardware,
            str(workload.get("operator_class", "prefill_gemm")),
            dtype,
            threads,
            0,
            0,
            0,
        )
        gemm_flops = max(0.0, _to_float(reference.get("gemm_flops"), 0.0))
        compute_ms = (gemm_flops / (gflops * 1e9)) * 1000.0 if gemm_flops > 0.0 else _to_float(reference.get("compute_ms"), 0.0)

    reference = workload.get("reference", {})
    reference_metrics = _reference_structured_metrics(hardware, workload, dtype, threads)
    other_compute_ms = max(0.0, _to_float(reference.get("other_compute_ms"), 0.0))
    decompress_ms = max(0.0, _to_float(reference.get("decompress_ms"), 0.0))
    reference_overhead_ms = max(0.0, _to_float(reference.get("overhead_ms"), 0.0))
    reference_kv_read_ms = max(0.0, _to_float(reference.get("kv_read_ms"), 0.0))

    runtime_row = _lookup_runtime_row(hardware, chunk_size, prefetch_window, dtype, threads)
    runtime_request_count = max(1.0, _to_float(runtime_row.get("request_count"), 1.0))
    merge_scale = chunked_execution_calls / runtime_request_count if chunked_execution_calls > 0.0 else 0.0
    runtime_units = streamed_chunks + (resident_chunked_calls * 0.25)
    runtime_scale = runtime_units / runtime_request_count if runtime_units > 0.0 else 0.0
    raw_merge_ms = max(0.0, _to_float(runtime_row.get("merge_ms"), 0.0)) * merge_scale
    raw_runtime_overhead_ms = max(0.0, _to_float(runtime_row.get("runtime_overhead_ms"), 0.0)) * runtime_scale
    reference_runtime_estimate_ms = max(0.0, reference_metrics.get("raw_runtime_overhead_ms", 0.0))
    runtime_overhead_calibration = (
        reference_overhead_ms / reference_runtime_estimate_ms
        if reference_overhead_ms > 0.0 and reference_runtime_estimate_ms > 0.0
        else 0.0
    )
    merge_ms = raw_merge_ms
    runtime_overhead_ms = raw_runtime_overhead_ms * runtime_overhead_calibration

    ref_bp = reference.get("bufferpool", {})
    reference_wait_ms = max(0.0, _to_float(ref_bp.get("wait_ms"), 0.0))
    runtime_chunk_rows = _lookup_runtime_rows_for_chunk(hardware, chunk_size, dtype, threads)
    base_io_exposed_ratio = _to_float(runtime_row.get("io_exposed_ratio"), 1.0)
    if (
        str(workload.get("mode", "")).strip().lower() == "decode"
        and reference_wait_ms > 0.0
        and runtime_chunk_rows
    ):
        # The decode microbench tends to overstate the prefetch-window effect relative
        # to end-to-end inference. Use the best observed overlap for this chunk size as
        # a lower envelope instead of charging the exact synthetic ratio.
        base_io_exposed_ratio = min(
            base_io_exposed_ratio,
            min(_to_float(row.get("io_exposed_ratio"), base_io_exposed_ratio) for row in runtime_chunk_rows),
        )
    overlap_penalty = min(1.0, max(0.0, base_io_exposed_ratio))
    io_exposed_ms = io_total_ms * overlap_penalty
    reference_raw_io_exposed_ms = max(0.0, reference_metrics.get("raw_io_exposed_ms", 0.0))
    if reference_raw_io_exposed_ms > 0.0 and reference_kv_read_ms > 0.0:
        mode = str(workload.get("mode", "")).strip().lower()
        if mode == "prefill" or (mode == "decode" and reference_wait_ms <= 0.0):
            io_exposed_ms *= reference_kv_read_ms / reference_raw_io_exposed_ms

    system_streamable_bytes = max(
        0.0,
        _to_float(workload.get("system", {}).get("streamable_weight_bytes"), 0.0),
    )
    mode = str(workload.get("mode", "")).strip().lower()
    streamed_fraction = 1.0
    if system_streamable_bytes > 0.0:
        streamed_fraction = min(1.0, streamed_bytes / system_streamable_bytes)
    bufferpool_fraction = min(1.0, bufferpool_bytes / float(max(1, system_total_memory_bytes)))
    dynamic_pressure_ms = reference_wait_ms * max(0.0, 1.0 - bufferpool_fraction) * streamed_fraction
    structural_pressure_ms = 0.0
    chunk_pressure_ms = 0.0
    aggressive_prefetch_ms = 0.0
    resident_pressure_ms = 0.0
    if mode == "decode":
        ref_bufferpool_bytes = max(0, _to_int(ref_bp.get("memory_total_bytes"), 0))
        if ref_bufferpool_bytes > 0 and bufferpool_bytes < ref_bufferpool_bytes:
            shortfall = 1.0 - (bufferpool_bytes / float(ref_bufferpool_bytes))
            structural_baseline_ms = max(
                other_compute_ms + decompress_ms + runtime_overhead_ms,
                reference_overhead_ms,
            )
            structural_pressure_ms = structural_baseline_ms * shortfall

        ref_chunk_size = max(0, _to_int(workload.get("system", {}).get("chunk_size"), 0))
        if reference_wait_ms > 0.0 and ref_chunk_size > 0 and chunk_size > ref_chunk_size:
            chunk_pressure_ms = reference_wait_ms * 0.15 * math.log2(chunk_size / float(ref_chunk_size))
            if prefetch_window > 1:
                aggressive_prefetch_ms = (
                    reference_wait_ms
                    * math.log2(chunk_size / float(ref_chunk_size))
                    * math.log2(float(prefetch_window))
                )
        selected_fraction = min(1.0, actual_selected_static_bytes / float(max(1, system_total_memory_bytes)))
        resident_pressure_ms = reference_overhead_ms * 0.25 * selected_fraction * selected_fraction

    stall_ms = (
        dynamic_pressure_ms
        + structural_pressure_ms
        + chunk_pressure_ms
        + aggressive_prefetch_ms
        + resident_pressure_ms
    )

    execution_ms = compute_ms + other_compute_ms + decompress_ms + merge_ms + runtime_overhead_ms + io_exposed_ms + stall_ms

    beta_init_mb_s = _beta_init_mb_s(hardware, workload)
    beta_init_bytes_per_ms = beta_init_mb_s * 1024.0 * 1024.0 / 1000.0
    static_load_ms = actual_selected_static_bytes / max(1e-9, beta_init_bytes_per_ms)
    score_ms = execution_ms + (static_load_ms if _include_startup(str(workload.get("mode", "prefill")), objective) else 0.0)

    return {
        "id": candidate_id,
        "feasible": feasible,
        "infeasible_reasons": infeasible_reasons,
        "dtype": dtype,
        "thread_count": threads,
        "chunk_size": chunk_size,
        "chunk_size_bytes": chunk_size_bytes,
        "prefetch_window": prefetch_window,
        "target_static_bytes": target_static_bytes,
        "selected_static_bytes": actual_selected_static_bytes,
        "selected_matrix_count": len(selected_matrix_ids),
        "bufferpool_bytes": bufferpool_bytes,
        "breakdown_ms": {
            "startup": static_load_ms,
            "io_total": io_total_ms,
            "io_exposed": io_exposed_ms,
            "compute": compute_ms,
            "other_compute": other_compute_ms,
            "decompress": decompress_ms,
            "merge": merge_ms,
            "runtime_overhead": runtime_overhead_ms,
            "stall": stall_ms,
            "execution": execution_ms,
        },
        "derived": {
            "streamed_bytes": streamed_bytes,
            "streamed_chunks": streamed_chunks,
            "streamed_windows": streamed_windows,
            "chunked_execution_calls": chunked_execution_calls,
            "resident_chunked_calls": resident_chunked_calls,
            "resident_dense_calls": resident_dense_calls,
            "ring_chunks": ring_chunks,
            "selected_matrix_ids": sorted(selected_matrix_ids),
            "io_exposed_ratio": overlap_penalty,
            "beta_init_mb_s": beta_init_mb_s,
        },
        "score_ms": score_ms,
        "score_source": "trace+microbench+matrix_model",
    }


def _score_candidate_legacy(
    hardware: Dict[str, Any],
    workload: Dict[str, Any],
    candidate: Dict[str, Any],
    system_total_memory_bytes: int,
    objective: str,
) -> Dict[str, Any]:
    candidate_id = str(candidate.get("id", "candidate"))
    dtype = _normalize_supported_dtype(str(candidate.get("dtype", workload.get("dtype", "bfloat16"))))
    threads = max(1, _to_int(candidate.get("thread_count", workload.get("system", {}).get("thread_count", 1)), 1))
    chunk_size = max(1, _to_int(candidate.get("chunk_size"), 0))
    prefetch_window = max(1, _to_int(candidate.get("prefetch_window"), 1))
    selected_static_bytes = max(
        0,
        _to_int(
            candidate.get(
                "selected_static_bytes",
                candidate.get("static_memory_bytes", candidate.get("requested_static_bytes", 0)),
            ),
            0,
        ),
    )
    bufferpool_bytes = max(0, _to_int(candidate.get("bufferpool_bytes"), 0))

    feasible = True
    infeasible_reasons: List[str] = []
    if selected_static_bytes + bufferpool_bytes > system_total_memory_bytes:
        feasible = False
        infeasible_reasons.append("selected_static_bytes + bufferpool_bytes exceeds total memory")

    chunk_size_bytes = _chunk_size_bytes_from_workload(workload, chunk_size, dtype)
    ring_chunks = bufferpool_bytes // max(1, chunk_size_bytes)
    if ring_chunks < 1:
        feasible = False
        infeasible_reasons.append("bufferpool capacity B=floor(M_b/c_bytes) is < 1")

    reference = workload.get("reference", {})
    operator_class = str(workload.get("operator_class", "prefill_gemm"))
    streamable_weight_bytes = max(0.0, _to_float(workload.get("system", {}).get("streamable_weight_bytes"), 0.0))
    ref_selected_static = max(0.0, _to_float(reference.get("selected_static_bytes"), 0.0))
    ref_bytes_read = max(0.0, _to_float(reference.get("bytes_read"), 0.0))
    ref_chunk_size = max(1, _to_int(reference.get("chunk_size"), chunk_size))
    ref_chunk_size_bytes = _chunk_size_bytes_from_workload(workload, ref_chunk_size, dtype)

    ref_static_frac = min(0.95, ref_selected_static / streamable_weight_bytes) if streamable_weight_bytes > 0.0 else 0.0
    logical_stream_bytes = ref_bytes_read / max(1e-6, 1.0 - ref_static_frac) if ref_bytes_read > 0.0 else streamable_weight_bytes
    candidate_static_frac = min(0.95, selected_static_bytes / streamable_weight_bytes) if streamable_weight_bytes > 0.0 else 0.0
    streamed_bytes = logical_stream_bytes * (1.0 - candidate_static_frac)
    streamed_chunks = streamed_bytes / float(max(1, chunk_size_bytes))
    streamed_windows = math.ceil(streamed_chunks / float(prefetch_window)) if streamed_chunks > 0.0 else 0

    read_size_bytes = prefetch_window * chunk_size_bytes
    lambda_ms, beta_mb_s = _lookup_io_profile(hardware, read_size_bytes, threads)
    beta_bytes_per_ms = beta_mb_s * 1024.0 * 1024.0 / 1000.0
    io_total_ms = streamed_windows * lambda_ms + (streamed_bytes / max(1e-9, beta_bytes_per_ms))

    gflops = _lookup_compute_gflops(hardware, operator_class, dtype, threads, 0, 0, 0)
    gemm_flops = max(0.0, _to_float(reference.get("gemm_flops"), 0.0))
    compute_ms = (gemm_flops / (gflops * 1e9)) * 1000.0 if gemm_flops > 0.0 else _to_float(reference.get("compute_ms"), 0.0)

    other_compute_ms = max(0.0, _to_float(reference.get("other_compute_ms"), 0.0))
    decompress_ms = max(0.0, _to_float(reference.get("decompress_ms"), 0.0))

    runtime_row = _lookup_runtime_row(hardware, chunk_size, prefetch_window, dtype, threads)
    runtime_bytes = max(1.0, _to_float(runtime_row.get("bytes_transferred"), max(ref_bytes_read, 1.0)))
    runtime_scale = streamed_bytes / runtime_bytes if runtime_bytes > 0.0 else 1.0
    io_exposed_ratio = _to_float(runtime_row.get("io_exposed_ratio"), 1.0)
    merge_ms = max(0.0, _to_float(runtime_row.get("merge_ms"), 0.0)) * runtime_scale
    runtime_overhead_ms = max(0.0, _to_float(runtime_row.get("runtime_overhead_ms"), 0.0)) * runtime_scale

    ref_bp = reference.get("bufferpool", {})
    ref_wait_ms = max(0.0, _to_float(ref_bp.get("wait_ms"), 0.0))
    ref_get_calls = max(0, _to_int(ref_bp.get("get_chunk_calls"), 0))
    ref_miss_ratio = _to_float(ref_bp.get("cache_misses"), 0.0) / float(ref_get_calls) if ref_get_calls > 0 else 0.0
    ring_penalty = max(1.0, float(prefetch_window) / float(max(1, ring_chunks)))
    stall_ms = ref_wait_ms * max(0.25, ring_penalty) * max(0.25, 0.5 + ref_miss_ratio)

    exposed_io_ms = io_total_ms * io_exposed_ratio
    execution_ms = compute_ms + other_compute_ms + decompress_ms + merge_ms + runtime_overhead_ms + exposed_io_ms + stall_ms

    beta_init_mb_s = _beta_init_mb_s(hardware, workload)
    beta_init_bytes_per_ms = beta_init_mb_s * 1024.0 * 1024.0 / 1000.0
    static_load_ms = selected_static_bytes / max(1e-9, beta_init_bytes_per_ms)
    score_ms = execution_ms + (static_load_ms if _include_startup(str(workload.get("mode", "prefill")), objective) else 0.0)

    return {
        "id": candidate_id,
        "feasible": feasible,
        "infeasible_reasons": infeasible_reasons,
        "dtype": dtype,
        "thread_count": threads,
        "chunk_size": chunk_size,
        "chunk_size_bytes": chunk_size_bytes,
        "prefetch_window": prefetch_window,
        "selected_static_bytes": selected_static_bytes,
        "bufferpool_bytes": bufferpool_bytes,
        "breakdown_ms": {
            "startup": static_load_ms,
            "io_total": io_total_ms,
            "io_exposed": exposed_io_ms,
            "compute": compute_ms,
            "other_compute": other_compute_ms,
            "decompress": decompress_ms,
            "merge": merge_ms,
            "runtime_overhead": runtime_overhead_ms,
            "stall": stall_ms,
            "execution": execution_ms,
        },
        "derived": {
            "streamed_bytes": streamed_bytes,
            "streamed_chunks": streamed_chunks,
            "streamed_windows": streamed_windows,
            "ring_chunks": ring_chunks,
            "io_exposed_ratio": io_exposed_ratio,
            "beta_init_mb_s": beta_init_mb_s,
        },
        "score_ms": score_ms,
        "score_source": "trace+microbench+legacy_model",
    }


def score_candidate(
    hardware: Dict[str, Any],
    workload: Dict[str, Any],
    candidate: Dict[str, Any],
    system_total_memory_bytes: int,
    objective: str,
) -> Dict[str, Any]:
    if workload.get("matrix_accesses"):
        return _score_candidate_structured(hardware, workload, candidate, system_total_memory_bytes, objective)
    return _score_candidate_legacy(hardware, workload, candidate, system_total_memory_bytes, objective)


def _write_report(path: str, workload: Dict[str, Any], objective: str, ranking: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append(f"# Ranking Report: {workload.get('name', 'workload')}")
    lines.append("")
    lines.append(f"Objective: `{objective}`")
    lines.append("")
    lines.append("| rank | id | feasible | score_ms | startup | execution | io_total | io_exposed | compute | merge | runtime_overhead | stall |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, row in enumerate(ranking, start=1):
        b = row["breakdown_ms"]
        lines.append(
            f"| {index} | {row['id']} | {str(row['feasible']).lower()} | {row['score_ms']:.4f} | "
            f"{b['startup']:.4f} | {b['execution']:.4f} | {b['io_total']:.4f} | {b['io_exposed']:.4f} | "
            f"{b['compute']:.4f} | {b['merge']:.4f} | {b['runtime_overhead']:.4f} | {b['stall']:.4f} |"
        )
    lines.append("")
    if ranking:
        top = ranking[0]
        lines.append("## Recommended")
        lines.append("")
        lines.append(
            f"Use `{top['id']}` with `c={top['chunk_size']}`, `w={top['prefetch_window']}`, "
            f"`selected_static_bytes={top['selected_static_bytes']}`, `bufferpool_bytes={top['bufferpool_bytes']}`, "
            f"`threads={top['thread_count']}`, `dtype={top['dtype']}`."
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate configs from microbench hardware profile and one workload trace.")
    parser.add_argument("--hardware-profile", required=True, help="Path to hardware_profile.json")
    parser.add_argument("--workload-profile", required=True, help="Path to trace-derived workload profile JSON")
    parser.add_argument("--candidates", required=True, help="Path to candidate JSON")
    parser.add_argument("--objective", default="auto", choices=["auto", "cold", "warm"], help="Whether to include startup cost")
    parser.add_argument("--out", default="recommended_config.json", help="Output JSON")
    parser.add_argument("--report", default="ranking_report.md", help="Output markdown report")
    args = parser.parse_args()

    hardware = _load_json(args.hardware_profile)
    workload = _load_json(args.workload_profile)
    candidates_obj = _load_json(args.candidates)
    if isinstance(candidates_obj, list):
        candidates = candidates_obj
        system_cfg = {}
    else:
        candidates = candidates_obj.get("candidates", [])
        system_cfg = candidates_obj.get("system", {})
    if not candidates:
        raise ValueError("No candidates found")

    system_total_memory_bytes = _to_int(system_cfg.get("total_memory_bytes"), 0)
    if system_total_memory_bytes <= 0:
        raise ValueError("Missing system.total_memory_bytes in candidates JSON")

    objective = args.objective
    if objective == "auto":
        objective = "cold" if str(workload.get("mode", "prefill")) == "prefill" else "warm"

    scored = [
        score_candidate(hardware, workload, candidate, system_total_memory_bytes, objective)
        for candidate in candidates
    ]
    ranking = sorted(scored, key=lambda row: (0 if row["feasible"] else 1, _to_float(row["score_ms"], float("inf"))))
    feasible = [row for row in ranking if row["feasible"]]
    if not feasible:
        raise ValueError("All candidates are infeasible")

    payload = {
        "workload_name": workload.get("name", "workload"),
        "mode": workload.get("mode", "prefill"),
        "objective": objective,
        "recommended": feasible[0],
        "ranking": ranking,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    _write_report(args.report, workload, objective, ranking)
    print(f"Wrote recommendation: {os.path.abspath(args.out)}")
    print(f"Wrote ranking report: {os.path.abspath(args.report)}")


if __name__ == "__main__":
    main()
