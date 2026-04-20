#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


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


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / float(len(vals))


def _default_operator_class(mode: str) -> str:
    return "decode_gemv" if mode == "decode" else "prefill_gemm"


WEIGHT_OPERATOR_CLASSES = {
    "attn_qkv_proj",
    "attn_o_proj",
    "ffn_down_proj",
    "ffn_gate_up_proj",
    "output_proj",
}


def _resident_execution_mode(matrix_group: str) -> str:
    if matrix_group == "ffn_gate_up_proj":
        return "chunked"
    return "dense"


def _streamed_execution_mode(priority_rank: int) -> str:
    return "chunked" if priority_rank >= 0 else "none"


def _chunk_count(rows: int, cols: int, split_mode: str, chunk_size: int) -> int:
    if chunk_size <= 0:
        return 0
    if str(split_mode).lower() == "row":
        return max(0, (rows + chunk_size - 1) // chunk_size)
    return max(0, (cols + chunk_size - 1) // chunk_size)


def _matrix_catalog(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    items = report.get("system", {}).get("matrix_catalog", [])
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        matrix_id = str(item.get("matrix_id", ""))
        if not matrix_id:
            continue
        out[matrix_id] = {
            "matrix_id": matrix_id,
            "matrix_group": str(item.get("matrix_group", "other")),
            "preload_group": str(item.get("preload_group", "non_preload")),
            "resident_execution": str(
                item.get("resident_execution", _resident_execution_mode(str(item.get("matrix_group", "other"))))
            ),
            "streamed_execution": str(
                item.get("streamed_execution", _streamed_execution_mode(_to_int(item.get("priority_rank"), -1)))
            ),
            "priority_rank": _to_int(item.get("priority_rank"), -1),
            "rows": _to_int(item.get("rows"), 0),
            "cols": _to_int(item.get("cols"), 0),
            "chunk_size": _to_int(item.get("chunk_size"), 0),
            "dtype": str(item.get("dtype", report.get("system", {}).get("dtype", "bfloat16"))),
            "split_mode": str(item.get("split_mode", "column")),
            "size_bytes": _to_int(item.get("size_bytes"), 0),
        }
    return out


def _aggregate_named_rows(
    rows_list: List[List[Dict[str, Any]]],
    key_fields: Tuple[str, ...],
    value_fields: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    counts: Dict[Tuple[Any, ...], int] = defaultdict(int)

    for rows in rows_list:
        for row in rows:
            key = tuple(row.get(field) for field in key_fields)
            entry = groups.setdefault(
                key,
                {field: row.get(field) for field in key_fields} | {field: 0.0 for field in value_fields},
            )
            counts[key] += 1
            for field in value_fields:
                entry[field] += _to_float(row.get(field), 0.0)

    out: List[Dict[str, Any]] = []
    for key, entry in groups.items():
        denom = max(1, counts[key])
        row = {field: entry[field] for field in key_fields}
        for field in value_fields:
            value = entry[field] / float(denom)
            if field.endswith("_reads") or field in ("calls", "flops", "bytes_read"):
                row[field] = value
            else:
                row[field] = value
        out.append(row)
    out.sort(key=lambda item: tuple(item.get(field) for field in key_fields))
    return out


def _phase_from_decode(report: Dict[str, Any]) -> Dict[str, Any]:
    traces = report.get("decode", {}).get("token_traces", [])
    if not traces:
        return {}

    def collect(name: str) -> float:
        values = [_to_float(item.get("trace", {}).get(name), 0.0) for item in traces]
        return _mean(values)

    def collect_int(name: str) -> int:
        values = [_to_int(item.get("trace", {}).get(name), 0) for item in traces]
        return int(round(_mean(float(v) for v in values)))

    def collect_bp(name: str) -> float:
        values = [
            _to_float(item.get("trace", {}).get("bufferpool", {}).get(name), 0.0)
            for item in traces
        ]
        return _mean(values)

    def collect_bp_int(name: str) -> int:
        values = [
            _to_int(item.get("trace", {}).get("bufferpool", {}).get(name), 0)
            for item in traces
        ]
        return int(round(_mean(float(v) for v in values)))

    matrix_accesses = _aggregate_named_rows(
        [item.get("trace", {}).get("matrix_accesses", []) for item in traces],
        (
            "matrix_id",
            "matrix_group",
            "preload_group",
            "priority_rank",
            "rows",
            "cols",
            "chunk_size",
            "dtype",
            "split_mode",
        ),
        ("bytes_read", "chunk_reads"),
    )
    gemm_buckets = _aggregate_named_rows(
        [item.get("trace", {}).get("gemm_buckets", []) for item in traces],
        ("operator_class", "m", "k", "n"),
        ("calls", "flops"),
    )

    return {
        "elapsed_ms": _to_float(report.get("decode", {}).get("ms_per_token"), 0.0),
        "prefetch_warmup_ms": collect("prefetch_warmup_ms"),
        "compute_ms": collect("compute_ms"),
        "other_compute_ms": collect("other_compute_ms"),
        "kv_read_ms": collect("kv_read_ms"),
        "decompress_ms": collect("decompress_ms"),
        "overhead_ms": collect("overhead_ms"),
        "bytes_read": collect_int("bytes_read"),
        "gemm_flops": collect_int("gemm_flops"),
        "matrix_accesses": matrix_accesses,
        "gemm_buckets": gemm_buckets,
        "bufferpool": {
            "get_chunk_calls": collect_bp_int("get_chunk_calls"),
            "cache_misses": collect_bp_int("cache_misses"),
            "wait_ms": collect_bp("wait_ms"),
            "memory_total_bytes": collect_bp_int("memory_total_bytes"),
        },
    }


def _build_matrix_access_profile(
    phase: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    system_chunk_size: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in phase.get("matrix_accesses", []):
        matrix_id = str(item.get("matrix_id", ""))
        if not matrix_id:
            continue
        meta = catalog.get(matrix_id, {})
        rows = _to_int(item.get("rows"), _to_int(meta.get("rows"), 0))
        cols = _to_int(item.get("cols"), _to_int(meta.get("cols"), 0))
        split_mode = str(item.get("split_mode", meta.get("split_mode", "column"))).lower()
        chunk_size = _to_int(item.get("chunk_size"), _to_int(meta.get("chunk_size"), system_chunk_size))
        size_bytes = _to_int(meta.get("size_bytes"), 0)
        if size_bytes <= 0 and rows > 0 and cols > 0:
            dtype = str(item.get("dtype", meta.get("dtype", "bfloat16"))).lower()
            dtype_bytes = 4 if dtype == "float32" else 1 if dtype == "int8" else 2
            size_bytes = rows * cols * dtype_bytes
        reference_chunk_count = _chunk_count(rows, cols, split_mode, chunk_size)
        chunk_reads = _to_float(item.get("chunk_reads"), 0.0)
        bytes_read = _to_float(item.get("bytes_read"), 0.0)
        logical_passes = 0.0
        if reference_chunk_count > 0:
            logical_passes = chunk_reads / float(reference_chunk_count)
        elif size_bytes > 0:
            logical_passes = bytes_read / float(size_bytes)
        out.append(
            {
                "matrix_id": matrix_id,
                "matrix_group": str(item.get("matrix_group", meta.get("matrix_group", "other"))),
                "preload_group": str(item.get("preload_group", meta.get("preload_group", "non_preload"))),
                "resident_execution": str(
                    item.get("resident_execution", meta.get("resident_execution", _resident_execution_mode(str(item.get("matrix_group", meta.get("matrix_group", "other"))))))
                ),
                "streamed_execution": str(
                    item.get("streamed_execution", meta.get("streamed_execution", _streamed_execution_mode(_to_int(item.get("priority_rank"), _to_int(meta.get("priority_rank"), -1)))))
                ),
                "priority_rank": _to_int(item.get("priority_rank"), _to_int(meta.get("priority_rank"), -1)),
                "rows": rows,
                "cols": cols,
                "dtype": str(item.get("dtype", meta.get("dtype", "bfloat16"))),
                "split_mode": split_mode,
                "matrix_bytes": size_bytes,
                "reference_chunk_size": chunk_size,
                "reference_chunk_count": reference_chunk_count,
                "reference_bytes_read": bytes_read,
                "reference_chunk_reads": chunk_reads,
                "logical_passes": logical_passes,
            }
        )
    out.sort(key=lambda row: row["matrix_id"])
    return out


def _matrix_compute_templates(
    matrix_accesses: List[Dict[str, Any]],
    compute_buckets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    shape_by_class: Dict[str, Dict[str, int]] = {}
    for item in compute_buckets:
        operator_class = str(item.get("operator_class", ""))
        if operator_class not in WEIGHT_OPERATOR_CLASSES:
            continue
        current = shape_by_class.get(operator_class)
        candidate = {
            "m": _to_int(item.get("m"), 0),
            "k": _to_int(item.get("k"), 0),
            "n": _to_int(item.get("n"), 0),
            "calls": _to_float(item.get("calls"), 0.0),
        }
        if current is None or candidate["calls"] > current["calls"]:
            shape_by_class[operator_class] = candidate

    out: List[Dict[str, Any]] = []
    for item in matrix_accesses:
        operator_class = str(item.get("matrix_group", ""))
        if operator_class not in WEIGHT_OPERATOR_CLASSES:
            continue
        rows = _to_int(item.get("rows"), 0)
        cols = _to_int(item.get("cols"), 0)
        split_mode = str(item.get("split_mode", "column")).lower()
        if split_mode == "row":
            k_dim = cols
            full_n = rows
        else:
            k_dim = rows
            full_n = cols
        shape_hint = shape_by_class.get(operator_class, {})
        out.append(
            {
                "matrix_id": str(item.get("matrix_id", "")),
                "matrix_group": operator_class,
                "operator_class": operator_class,
                "resident_execution": str(item.get("resident_execution", _resident_execution_mode(operator_class))),
                "streamed_execution": str(item.get("streamed_execution", _streamed_execution_mode(_to_int(item.get("priority_rank"), -1)))),
                "m": _to_int(shape_hint.get("m"), 0),
                "k": k_dim if k_dim > 0 else _to_int(shape_hint.get("k"), 0),
                "full_n": full_n if full_n > 0 else _to_int(shape_hint.get("n"), 0),
                "reference_chunk_size": _to_int(item.get("reference_chunk_size"), 0),
                "logical_passes": _to_float(item.get("logical_passes"), 0.0),
            }
        )
    out.sort(key=lambda row: row["matrix_id"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a workload profile JSON from one traced llama_inference report.")
    parser.add_argument("--trace-json", required=True, help="Path to llama_inference report JSON")
    parser.add_argument("--mode", required=True, choices=["prefill", "decode"], help="Workload mode to extract")
    parser.add_argument("--name", default="", help="Optional workload profile name")
    parser.add_argument("--chunk-axis", default="column", choices=["column", "row"], help="Chunking axis")
    parser.add_argument("--chunk-rows", type=int, default=0, help="Chunk rows for column chunking")
    parser.add_argument("--chunk-cols", type=int, default=0, help="Chunk cols for row chunking")
    parser.add_argument("--operator-class", default="", help="Override operator class")
    parser.add_argument("--out", required=True, help="Output workload profile JSON")
    args = parser.parse_args()

    report = json.load(open(args.trace_json, "r", encoding="utf-8"))
    system = report.get("system", {})
    catalog = _matrix_catalog(report)

    chunk_rows = args.chunk_rows if args.chunk_rows > 0 else _to_int(system.get("hidden_dim"), 0)
    chunk_cols = args.chunk_cols
    if args.chunk_axis == "column" and chunk_rows <= 0:
        raise ValueError("chunk_rows must be provided or inferrable from trace for column chunking")
    if args.chunk_axis == "row" and chunk_cols <= 0:
        raise ValueError("chunk_cols must be provided for row chunking")

    if args.mode == "prefill":
        phase = report.get("prefill", {}).get("trace", {})
        elapsed_ms = _to_float(report.get("prefill", {}).get("elapsed_s"), 0.0) * 1000.0
    else:
        phase = _phase_from_decode(report)
        elapsed_ms = _to_float(report.get("decode", {}).get("ms_per_token"), 0.0)

    operator_class = args.operator_class or _default_operator_class(args.mode)
    preload = report.get("preload", {})
    preload_selected_bytes = _to_int(preload.get("selected_bytes"), 0)
    preload_elapsed_s = _to_float(preload.get("elapsed_s"), 0.0)
    preload_throughput = _to_float(preload.get("throughput_mb_s"), 0.0)
    system_chunk_size = _to_int(system.get("chunk_size"), 0)

    matrix_accesses = _build_matrix_access_profile(phase, catalog, system_chunk_size)
    compute_buckets = [
        {
            "operator_class": str(item.get("operator_class", operator_class)),
            "m": _to_int(item.get("m"), 0),
            "k": _to_int(item.get("k"), 0),
            "n": _to_int(item.get("n"), 0),
            "calls": _to_float(item.get("calls"), 0.0),
            "flops": _to_float(item.get("flops"), 0.0),
        }
        for item in phase.get("gemm_buckets", [])
    ]
    matrix_compute_templates = _matrix_compute_templates(matrix_accesses, compute_buckets)
    extra_compute_buckets = [
        item for item in compute_buckets if str(item.get("operator_class", "")) not in WEIGHT_OPERATOR_CLASSES
    ]

    payload = {
        "name": args.name or f"trace_{args.mode}",
        "mode": args.mode,
        "dtype": str(system.get("dtype", "bfloat16")),
        "chunk_axis": args.chunk_axis,
        "chunk_rows": chunk_rows,
        "chunk_cols": chunk_cols,
        "operator_class": operator_class,
        "system": {
            "hidden_dim": _to_int(system.get("hidden_dim"), 0),
            "num_layers": _to_int(system.get("num_layers"), 0),
            "num_heads": _to_int(system.get("num_heads"), 0),
            "num_kv_heads": _to_int(system.get("num_kv_heads"), 0),
            "seq_len": _to_int(system.get("seq_len"), 0),
            "decode_steps": _to_int(system.get("decode_steps"), 0),
            "chunk_size": system_chunk_size,
            "prefetch_window": _to_int(system.get("prefetch_window"), 0),
            "thread_count": _to_int(system.get("thread_count"), 0),
            "streamable_weight_bytes": _to_int(system.get("streamable_weight_bytes"), 0),
            "streamable_matrix_count": _to_int(system.get("streamable_matrix_count"), 0),
        },
        "matrix_catalog": list(catalog.values()),
        "matrix_accesses": matrix_accesses,
        "matrix_compute_templates": matrix_compute_templates,
        "compute_buckets": compute_buckets,
        "extra_compute_buckets": extra_compute_buckets,
        "reference": {
            "elapsed_ms": elapsed_ms,
            "chunk_size": system_chunk_size,
            "prefetch_window": _to_int(system.get("prefetch_window"), 0),
            "thread_count": _to_int(system.get("thread_count"), 0),
            "bufferpool_bytes": _to_int(system.get("arena_size_mb"), 0) * 1024 * 1024,
            "selected_static_bytes": preload_selected_bytes,
            "selected_matrix_ids": preload.get("selected_matrix_ids", []),
            "compute_ms": _to_float(phase.get("compute_ms"), 0.0),
            "other_compute_ms": _to_float(phase.get("other_compute_ms"), 0.0),
            "kv_read_ms": _to_float(phase.get("kv_read_ms"), 0.0),
            "decompress_ms": _to_float(phase.get("decompress_ms"), 0.0),
            "overhead_ms": _to_float(phase.get("overhead_ms"), 0.0),
            "bytes_read": _to_int(phase.get("bytes_read"), 0),
            "gemm_flops": _to_int(phase.get("gemm_flops"), 0),
            "prefetch_warmup_ms": _to_float(phase.get("prefetch_warmup_ms"), 0.0),
            "bufferpool": phase.get("bufferpool", {}),
        },
        "startup": {
            "selected_static_bytes": preload_selected_bytes,
            "selected_matrix_ids": preload.get("selected_matrix_ids", []),
            "elapsed_ms": preload_elapsed_s * 1000.0,
            "throughput_mb_s": preload_throughput,
        },
    }

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote workload profile: {args.out}")


if __name__ == "__main__":
    main()
