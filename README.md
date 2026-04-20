# KVTensor

This repository is organized around a C++ runtime plus Python experiment runners.

## Project Structure

- `cpp/`
  C++ source code, headers, CMake build, and example binaries.
- `experiment/scripts/`
  Python scripts that generate experiment databases, run workloads, and write result tables under `experiment/results/`.
- `evaluation/`
  Curated CSV inputs, plotting scripts, and generated figures.
- `paper/`
  Paper source files.

## Compile

Requirements:

- C++17 compiler
- `cmake`
- `leveldb`
- `dnnl`

Build:

```bash
cd cpp
cmake -S . -B build
cmake --build build -j
```

Main binaries:

- `cpp/build/llama_inference_example`
- `cpp/build/stable_diffusion_inference_example`
- `cpp/build/simpledb_generate_weights`
- `cpp/build/simpledb_generate_sd_weights`

## Experiment Scripts

All experiment scripts live in `experiment/scripts/`.

### 1. Section 6.3 Llama Grid

Script:

```bash
python3 experiment/scripts/run_section63.py
```

Important options:

- `--build-dir`
  Path to compiled binaries. Default: `cpp/build`
- `--results-dir`
  Output root. Default: `experiment/results`
- `--platforms`
  Comma-separated platform presets, usually `4c8g` and/or `8c16g`
- `--chunk-sizes`
  Comma-separated chunk sizes
- `--prefetch-windows`
  Comma-separated prefetch window sizes
- `--static-ratios`
  Comma-separated static residency ratios in `[0,1]`
- `--repeats`
  Number of repeated runs per config
- `--seq-len`
  Prompt length
- `--decode-steps`
  Number of generated decode steps
- `--hidden-dim`, `--num-layers`, `--num-heads`, `--num-kv-heads`, `--vocab-size`, `--precision`
  Synthetic model configuration used for DB generation and inference
- `--command-prefix`
  Optional launcher prefix
- `--env KEY=VALUE`
  Extra environment overrides
- `--force-regenerate-dbs`
  Rebuild weight DBs
- `--force-rerun`
  Re-run existing experiment points

Example:

```bash
python3 experiment/scripts/run_section63.py \
  --platforms 4c8g,8c16g \
  --chunk-sizes 64,128,256,512 \
  --prefetch-windows 1,2,4 \
  --static-ratios 0,0.25,0.5,0.75 \
  --repeats 3 \
  --seq-len 512 \
  --decode-steps 128
```

Outputs:

- raw logs, DBs, preload files, and per-run JSONs under `experiment/results/raw/`
- summary CSVs under `experiment/results/derived/`

### 2. Preload Cost Measurement

Script:

```bash
python3 experiment/scripts/measure_preload_cost.py
```

Important options:

- `--build-dir`
- `--db-root`
- `--results-dir`
- `--platforms`
- `--chunk-sizes`
- `--static-ratios`
- `--repeats`
- `--hidden-dim`, `--num-layers`, `--num-heads`, `--num-kv-heads`, `--vocab-size`, `--precision`
- `--force-regenerate-dbs`
- `--force-rerun`

Example:

```bash
python3 experiment/scripts/measure_preload_cost.py \
  --platforms 4c8g,8c16g \
  --chunk-sizes 64,128,256,512 \
  --static-ratios 0,0.25,0.5,0.75 \
  --repeats 3
```

### 3. Stable Diffusion Experiment

Script:

```bash
python3 experiment/scripts/run_stable_diffusion_experiment.py
```

Important options:

- `--build-dir`
- `--results-dir`
- `--platforms`
- `--chunk-sizes`
- `--prefetch-window`
- `--repeats`
- `--prompt`, `--steps`, `--guidance-scale`
- `--text-hidden`, `--text-layers`, `--text-vocab-size`, `--text-max-length`
- `--unet-hidden`, `--latent-channels`, `--sample-height`, `--sample-width`
- `--image-size`, `--image-channels`, `--vae-conv-layers`
- `--precision`
- `--command-prefix`
- `--env KEY=VALUE`
- `--force-regenerate-dbs`
- `--force-rerun`

Example:

```bash
python3 experiment/scripts/run_stable_diffusion_experiment.py \
  --platforms 4c8g,8c16g \
  --chunk-sizes 64,128,256,512 \
  --prefetch-window 1 \
  --repeats 3 \
  --image-size 512 \
  --steps 20
```

### 4. 3B Prefill / Logit Comparison

Script:

```bash
python3 experiment/scripts/run_llama_prefill_logits_experiment.py
```

Use this when you want the 3B validation/error artifacts used by the evaluation pipeline.

## Evaluation Helpers

Sync experiment summaries into `evaluation/`:

```bash
python3 evaluation/scripts/sync_experiment_data.py
```

Render figures from curated CSVs:

```bash
python3 evaluation/scripts/plot_paper_figures.py
```
