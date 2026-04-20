#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _legacy_id(platform_id: str, chunk_size: int, prefetch_window: int, static_ratio: float) -> str:
    machine_gib = 8 if platform_id == "4c8g" else 16
    threads = 4 if platform_id == "4c8g" else 8
    static_gib = int(round(machine_gib * static_ratio))
    buffer_gib = machine_gib - static_gib
    return f"cfg_c{chunk_size}_w{prefetch_window}_ms{static_gib}g_mb{buffer_gib}g_t{threads}"


def _load_ranking(path: str) -> List[Dict[str, float]]:
    payload = json.load(open(path, "r", encoding="utf-8"))
    rows = payload.get("ranking", payload if isinstance(payload, list) else [])
    return [
        {
            "id": str(row["id"]),
            "score_ms": float(row["score_ms"]),
        }
        for row in rows
    ]


def _rank(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _spearman(xs: List[float], ys: List[float]) -> float:
    rx = _rank(xs)
    ry = _rank(ys)
    n = len(xs)
    if n == 0:
        return float("nan")
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ranking output against merged63.csv without using it for fitting.")
    parser.add_argument("--ranking-json", required=True)
    parser.add_argument("--merged-csv", required=True)
    parser.add_argument("--platform-id", required=True, choices=["4c8g", "8c16g"])
    parser.add_argument("--mode", required=True, choices=["prefill", "decode"])
    args = parser.parse_args()

    ranking = _load_ranking(args.ranking_json)
    actual_rows = list(csv.DictReader(open(args.merged_csv, "r", encoding="utf-8")))
    value_col = "median_prefill_s" if args.mode == "prefill" else "median_decode_ms_per_token"
    actual: List[Tuple[str, float]] = []
    for row in actual_rows:
        if row.get("platform_id") != args.platform_id:
            continue
        actual.append(
            (
                _legacy_id(
                    args.platform_id,
                    int(float(row["chunk_size"])),
                    int(float(row["prefetch_window"])),
                    float(row["static_ratio"]),
                ),
                float(row[value_col]) * (1000.0 if args.mode == "prefill" else 1.0),
            )
        )

    actual_by_id = dict(actual)
    aligned = [row for row in ranking if row["id"] in actual_by_id]
    predicted_scores = [row["score_ms"] for row in aligned]
    actual_scores = [actual_by_id[row["id"]] for row in aligned]
    aligned_actual = [(row["id"], actual_by_id[row["id"]]) for row in aligned]
    actual_rank = {cid: idx + 1 for idx, (cid, _) in enumerate(sorted(aligned_actual, key=lambda item: item[1]))}
    top_actual = sorted(aligned_actual, key=lambda item: item[1])
    top5_actual = {cid for cid, _ in top_actual[:5]}
    top5_pred = {row["id"] for row in aligned[:5]}
    worst = sorted(
        (
            (
                abs((idx + 1) - actual_rank[row["id"]]),
                row["id"],
                idx + 1,
                actual_rank[row["id"]],
                actual_by_id[row["id"]],
            )
            for idx, row in enumerate(aligned)
        ),
        reverse=True,
    )[:5]

    payload = {
        "platform_id": args.platform_id,
        "mode": args.mode,
        "spearman": _spearman(predicted_scores, actual_scores),
        "top1_predicted": aligned[0]["id"] if aligned else "",
        "top1_actual": top_actual[0][0] if top_actual else "",
        "top5_hit_count": len(top5_pred & top5_actual),
        "worst_rank_inversions": [
            {
                "abs_rank_gap": gap,
                "id": cid,
                "predicted_rank": pred_rank,
                "actual_rank": act_rank,
                "actual_value_ms": actual_value,
            }
            for gap, cid, pred_rank, act_rank, actual_value in worst
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
