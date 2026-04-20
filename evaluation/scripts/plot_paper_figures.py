#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to render evaluation figures. "
        "Install it in the active Python environment and rerun this script."
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURE_EXTENSIONS = ("pdf", "png")
SYSTEM_COLORS = {
    "KVTensor": "#1f3c88",
    "FlexGen": "#d97706",
    "Llama.cpp": "#6b7280",
    "HFA": "#b91c1c",
}
PLATFORM_COLORS = {
    "4c8g": "#0f766e",
    "8c16g": "#1d4ed8",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def evaluation_root() -> Path:
    return repo_root() / "evaluation"


def figures_root() -> Path:
    return evaluation_root() / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in FIGURE_EXTENSIONS:
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)


def plot_kv_calibration(eval_root: Path, output_dir: Path) -> None:
    rows = read_csv(eval_root / "calibration_fixed" / "kv_calibration_621.csv")
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["platform"]].append((int(row["chunk_size"]), float(row["throughput_mb_s"])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for platform, points in sorted(grouped.items()):
        points.sort()
        xs = [chunk for chunk, _ in points]
        ys = [throughput for _, throughput in points]
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=platform, color=PLATFORM_COLORS.get(platform))

    ax.set_title("KV Read Throughput Calibration")
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("Throughput (MB/s)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "kv_calibration_621", output_dir)


def plot_runtime_calibration(eval_root: Path, output_dir: Path) -> None:
    rows = read_csv(eval_root / "calibration_fixed" / "runtime_calibration_623.csv")
    platforms = sorted({row["platform"] for row in rows})
    fig, axes = plt.subplots(1, len(platforms), figsize=(12, 4.5), sharey=True)
    if len(platforms) == 1:
        axes = [axes]

    for ax, platform in zip(axes, platforms):
        platform_rows = [row for row in rows if row["platform"] == platform]
        platform_rows.sort(key=lambda row: int(row["chunk_size"]))
        xs = range(len(platform_rows))
        chunks = [row["chunk_size"] for row in platform_rows]
        io_vals = [float(row["io_exposed_ms"]) for row in platform_rows]
        compute_vals = [float(row["compute_ms"]) for row in platform_rows]
        overhead_vals = [float(row["runtime_overhead_ms"]) for row in platform_rows]

        ax.bar(xs, compute_vals, label="Compute", color="#1d4ed8")
        ax.bar(xs, io_vals, bottom=compute_vals, label="Exposed I/O", color="#0f766e")
        stacked = [compute_vals[i] + io_vals[i] for i in range(len(io_vals))]
        ax.bar(xs, overhead_vals, bottom=stacked, label="Runtime overhead", color="#d97706")
        ax.set_xticks(list(xs), chunks)
        ax.set_title(platform)
        ax.set_xlabel("Chunk size")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Time (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Runtime Calibration Breakdown", y=1.02)
    save_figure(fig, "runtime_decomposition_623", output_dir)


def plot_validation_summary(eval_root: Path, output_dir: Path) -> None:
    rows = read_csv(eval_root / "calibration_fixed" / "validation_625_summary.csv")
    labels = [f"{row['model_size']} {row['platform']} {row['mode']}" for row in rows]
    spearman = [float(row["spearman"]) for row in rows]
    slowdown = [float(row["top1_slowdown_pct"]) for row in rows]
    xs = range(len(rows))

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].bar(xs, spearman, color="#1d4ed8")
    axes[0].set_ylabel("Spearman")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_title("Validation Summary")

    axes[1].bar(xs, slowdown, color="#d97706")
    axes[1].set_ylabel("Top-1 slowdown (%)")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_xticks(list(xs), labels, rotation=25, ha="right")
    save_figure(fig, "validation_625", output_dir)


def plot_baselines(eval_root: Path, output_dir: Path) -> None:
    rows = read_csv(eval_root / "section63_figure1.csv")
    groups = sorted({(row["model"], row["platform"]) for row in rows})
    systems = sorted({row["system"] for row in rows}, key=lambda name: (name != "KVTensor", name))
    x_positions = range(len(groups))
    width = 0.18

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    metrics = [("prefill_s", "Prefill (s)"), ("decode_s", "Decode (s/token)")]
    for ax, (metric, label) in zip(axes, metrics):
        for system_index, system in enumerate(systems):
            values = []
            for group in groups:
                value = next(
                    float(row[metric]) for row in rows if (row["model"], row["platform"], row["system"]) == (*group, system)
                )
                values.append(value)
            offsets = [x + (system_index - (len(systems) - 1) / 2.0) * width for x in x_positions]
            ax.bar(offsets, values, width=width, label=system, color=SYSTEM_COLORS.get(system))
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_title("System Comparison")
    axes[1].set_xticks(list(x_positions), [f"{model}-{platform}" for model, platform in groups])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(systems), frameon=False)
    save_figure(fig, "section63_figure1", output_dir)


def plot_prefill_decode_error(eval_root: Path, output_dir: Path) -> None:
    rows = read_csv(eval_root / "prefill_decode_error.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["model"])].append(row)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    metrics = [("prefill_s", "Prefill (s)"), ("decode_s", "Decode (s/token)")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for (system, model), group_rows in sorted(grouped.items()):
            group_rows.sort(key=lambda row: int(row["token_len"]))
            xs = [int(row["token_len"]) for row in group_rows]
            ys = [float(row[metric]) for row in group_rows]
            ax.plot(xs, ys, marker="o", linewidth=2, label=f"{system} {model}")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].set_title("Prompt-Length Sensitivity")
    axes[1].set_xlabel("Prompt length")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    save_figure(fig, "section63_prefill_decode_error", output_dir)


def plot_memory_budget(eval_root: Path, output_dir: Path) -> None:
    rows = read_csv(eval_root / "section63_figure2.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["platform"]].append(row)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    metrics = [("prefill_s", "Prefill (s)"), ("decode_s", "Decode (s/token)")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for platform, platform_rows in sorted(grouped.items()):
            platform_rows.sort(key=lambda row: float(row["memory_budget_gb"]))
            xs = [float(row["memory_budget_gb"]) for row in platform_rows]
            ys = [float(row[metric]) for row in platform_rows]
            ax.plot(xs, ys, marker="o", linewidth=2.2, label=platform, color=PLATFORM_COLORS.get(platform))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].set_title("Memory Budget Sensitivity")
    axes[1].set_xlabel("Memory budget (GB)")
    axes[0].legend(frameon=False)
    save_figure(fig, "section63_figure2", output_dir)


def plot_chunk_size(eval_root: Path, output_dir: Path) -> None:
    rows = read_csv(eval_root / "section63_figure3.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["platform"])].append(row)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    metrics = [("prefill_s", "Prefill (s)"), ("decode_s", "Decode (s/token)")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for (model, platform), group_rows in sorted(grouped.items()):
            group_rows.sort(key=lambda row: int(row["chunk_size"]))
            xs = [int(row["chunk_size"]) for row in group_rows]
            ys = [float(row[metric]) for row in group_rows]
            ax.plot(xs, ys, marker="o", linewidth=2, label=f"{model}-{platform}")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].set_title("Chunk Size Sensitivity")
    axes[1].set_xlabel("Chunk size")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    save_figure(fig, "section63_figure3", output_dir)


def plot_cache_policy(eval_root: Path, output_dir: Path) -> None:
    rows = read_csv(eval_root / "section63_figure4.csv")
    policies = [row["cache_policy"] for row in rows]
    prefill = [float(row["prefill_s"]) for row in rows]
    decode = [float(row["decode_s"]) for row in rows]
    xs = range(len(rows))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].bar(xs, prefill, color="#1d4ed8")
    axes[0].set_ylabel("Prefill (s)")
    axes[0].set_title("Cache Policy Comparison")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(xs, decode, color="#0f766e")
    axes[1].set_ylabel("Decode (s/token)")
    axes[1].set_xticks(list(xs), policies, rotation=15, ha="right")
    axes[1].grid(axis="y", alpha=0.2)
    save_figure(fig, "section63_figure4", output_dir)


def plot_diffusion_breakdown(eval_root: Path, output_dir: Path) -> None:
    rows = read_csv(eval_root / "diffusion.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["platform_id"], row["img_size"])].append(row)

    ordered_groups = sorted(grouped)
    fig, axes = plt.subplots(len(ordered_groups), 1, figsize=(10, 3.6 * len(ordered_groups)), sharex=False)
    if len(ordered_groups) == 1:
        axes = [axes]

    for ax, key in zip(axes, ordered_groups):
        platform_id, img_size = key
        group_rows = grouped[key]
        group_rows.sort(key=lambda row: int(row["chunk_size"]))
        labels = [row["chunk_size"] for row in group_rows]
        compute_like = [
            (
                float(row["median_profiled_compute_ms"])
                + float(row["median_profiled_other_compute_ms"])
                + float(row["median_profiled_decompress_ms"])
            ) / 1000.0
            for row in group_rows
        ]
        exposed_io = [
            (float(row["median_profiled_io_ms"]) + float(row["median_unprofiled_pipeline_ms"])) / 1000.0
            for row in group_rows
        ]
        system_overhead = [
            (float(row["median_profiled_system_overhead_ms"]) + float(row["median_prefetch_warmup_ms"])) / 1000.0
            for row in group_rows
        ]
        xs = range(len(group_rows))
        ax.bar(xs, compute_like, color="#1d4ed8", label="Compute-like")
        ax.bar(xs, exposed_io, bottom=compute_like, color="#0f766e", label="Exposed pipeline")
        bottoms = [compute_like[i] + exposed_io[i] for i in range(len(group_rows))]
        ax.bar(xs, system_overhead, bottom=bottoms, color="#d97706", label="System overhead")
        ax.set_xticks(list(xs), labels)
        ax.set_ylabel("Seconds")
        ax.set_title(f"{platform_id}, {img_size}x{img_size}")
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Stable Diffusion Runtime Breakdown", y=1.01)
    save_figure(fig, "diffusion_runtime_breakdown", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render evaluation figures from curated CSV inputs.")
    parser.add_argument("--output-dir", default=str(figures_root()))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_root = evaluation_root()
    output_dir = Path(args.output_dir).resolve()

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_kv_calibration(eval_root, output_dir)
    plot_runtime_calibration(eval_root, output_dir)
    plot_validation_summary(eval_root, output_dir)
    plot_baselines(eval_root, output_dir)
    plot_prefill_decode_error(eval_root, output_dir)
    plot_memory_budget(eval_root, output_dir)
    plot_chunk_size(eval_root, output_dir)
    plot_cache_policy(eval_root, output_dir)
    plot_diffusion_breakdown(eval_root, output_dir)
    print(f"wrote figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
