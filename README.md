# LMDataPrepare

Standalone data preparation project for language model pretraining corpora.

This repository focuses only on corpus ingestion, cleaning, sharding, and optional
Hugging Face Hub publishing. Training code lives in a separate repository.

## Scope

- Official Wikipedia dump preprocessing (`pages-articles-multistream.xml.bz2`)
- Official MS MARCO docs preprocessing (`msmarco-docs.tsv.gz`)
- Clean title/body extraction
- Parquet sharding with manifest and run summary
- Resume-safe processing
- Deterministic filtering (PII redaction, length/title checks, boilerplate cleanup)
- Exact dedup (LMDB) and near dedup (SimHash + LSH over LMDB)
- Optional model-based quality filtering (fastText)
- Optional push to Hugging Face Hub dataset repos

## Project structure

- `config/preprocess/`: Hydra configs for preprocess jobs
- `script/preprocess/`: corpus preprocess entrypoints
- `src/data/preprocess/`: download, clean, shard, hub, state helpers
- `src/runtime/`: lightweight script setup and logging helpers
- `test/`: preprocess-focused unit tests

## Prerequisites

- Python 3.10+ (Python 3.13 recommended for Hydra-based preprocess entrypoints)
- Network access for dataset downloads
- Enough disk space for:
  - raw downloads in `data/raw/*`
  - processed parquet shards in `data/processed/*`
  - LMDB indexes when dedup is enabled

## Install

Recommended setup uses a virtual environment:

```bash
python -m venv .venv --without-pip
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt
export PYTHONPATH=.
```

If your Python installation is externally managed (PEP 668), install dependencies only
inside `.venv` and run commands with `.venv/bin/python` as shown below.

You can still use plain `python` after activating the environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=.
```

## Configs

- `config/preprocess/_base.yaml`: shared defaults
- `config/preprocess/wikipedia_dump.yaml`: Wikimedia dump workflow
- `config/preprocess/msmarco_docs.yaml`: MS MARCO docs workflow

Key config groups:

- `filtering.*`: deterministic post-clean filtering and canonical text settings
- `dedup.exact.*`: exact SHA-256 dedup index (LMDB path, map size, batch size)
- `dedup.near.*`: near dedup controls (SimHash bits, LSH banding, threshold)
- `quality.*`: fastText quality scorer controls
- `performance.*`: chunksize tuning and resume-state save throttling
- `output.use_fixed_schema`: build Arrow tables with fixed schema for faster writes

Additional high-impact options:

- `filtering.enabled`: enable/disable deterministic filtering stage (title/body length, PII, boilerplate)
- `filtering.canonical_fields`: fields used to compute canonical text/hash for dedup and quality cache keys
- `filtering.boilerplate_patterns`: regex patterns stripped from body text before filtering/dedup
- `dedup.exact.map_size_bytes` / `dedup.near.map_size_bytes`: LMDB map size, tune based on corpus size
- `dedup.near.hamming_threshold` / `dedup.near.band_bits`: near-duplicate strictness and LSH candidate selection
- `quality.positive_label`: fastText label treated as keep/high-quality class
- `quality.cache_by_hash`: cache quality scores by canonical hash to reduce repeated scoring cost

## Quickstart (first run)

Run a small smoke preprocess first to verify your environment before full datasets.

Wikipedia smoke run:

```bash
.venv/bin/python script/preprocess/preprocess_wikipedia_dump.py \
  run.max_rows=2000 \
  run.resume=false \
  processing.num_proc=4 \
  output.output_dir=data/processed/wikipedia_smoke
```

MS MARCO smoke run:

```bash
.venv/bin/python script/preprocess/preprocess_msmarco_docs.py \
  run.max_rows=5000 \
  run.resume=false \
  processing.num_proc=4 \
  output.output_dir=data/processed/msmarco_smoke
```

Check artifacts:

```bash
ls data/processed/wikipedia_smoke
ls data/processed/msmarco_smoke
```

You should see parquet shards plus:

- `manifest.json`
- `run_summary.json`
- `resume_state.json`
- generated dataset card `README.md`

## Wikipedia dump to Parquet

```bash
.venv/bin/python script/preprocess/preprocess_wikipedia_dump.py \
  processing.num_proc=32 \
  output.output_dir=data/processed/wikipedia_dump \
  run.resume=true
```

`wikipedia_dump.yaml` already enables PII redaction and exact/near dedup by default.

Output schema:

- `id`
- `title`
- `body`
- `url`
- `source`
- `dump_date`

## MS MARCO docs to Parquet

```bash
.venv/bin/python script/preprocess/preprocess_msmarco_docs.py \
  processing.num_proc=32 \
  output.output_dir=data/processed/msmarco_docs \
  run.resume=true
```

`msmarco_docs.yaml` already enables PII redaction and exact/near dedup by default.

Output schema:

- `doc_id`
- `url`
- `title`
- `body`
- `source`
- `version`

MS MARCO malformed row behavior is controlled by `processing.malformed_row_policy`:

- `skip` (default): count malformed rows and continue
- `raise`: fail fast on the first malformed row

## Resume semantics

Each run writes `resume_state.json` under `output.output_dir`.

- `run.resume=true`: resume from previously saved counters/shard index.
- `run.resume=false`: start a fresh run (ignore prior state).

When dedup is enabled:

- `run.resume=true` reuses existing LMDB dedup indexes.
- `run.resume=false` resets LMDB indexes for a fresh pass.
- Resume now validates dedup compatibility:
  - saved dedup paths must match current config
  - required LMDB files must exist
  - otherwise the run fails fast with an actionable error

Artifacts written per run:

- `*.parquet` shards
- `manifest.json`
- `run_summary.json`
- generated dataset card `README.md`

`run_summary.json` includes throughput and filtering signals:

- `rows_per_second`
- `drop_reasons`
- `dedup` stats
- `quality` score stats and quantiles
- `stage_seconds`

## Push to Hugging Face Hub

Set token and enable `hub.push`:

```bash
export HF_TOKEN=<your_hf_token>
.venv/bin/python script/preprocess/preprocess_wikipedia_dump.py \
  hub.push=true \
  hub.repo_id=<org_or_user>/<dataset_name> \
  hub.path_in_repo=data
```

Retry knobs:

- `hub.retries`
- `hub.retry_backoff_seconds`

## Docker

Build image from repository root:

```bash
docker build -f docker/Dockerfile -t lm_data_prepare:local .
```

Open an interactive container with the project mounted:

```bash
docker compose -f docker/docker-compose.yml run --rm lm_data_prepare
```

Inside the container you can run the same preprocess scripts:

```bash
python script/preprocess/preprocess_msmarco_docs.py run.max_rows=1000 run.resume=false
```

## Optional quality model training

Train a fastText supervised classifier from labeled lines in the format:

```text
__label__high_quality document text here
__label__low_quality document text here
```

Example:

```bash
.venv/bin/python script/preprocess/train_quality_classifier.py \
  --train-file data/quality/train.txt \
  --model-out data/models/quality.bin \
  --validation-file data/quality/val.txt \
  --calibration-out data/models/quality_calibration.json \
  --positive-label __label__high_quality
```

Calibration output includes `calibration.threshold`; use that value for `quality.min_score`.

Then enable scoring at preprocess time:

```bash
.venv/bin/python script/preprocess/preprocess_wikipedia_dump.py \
  quality.enabled=true \
  quality.model_path=data/models/quality.bin \
  quality.min_score=0.55 \
  quality.positive_label=__label__high_quality
```

## Throughput presets

Throughput-focused:

```bash
.venv/bin/python script/preprocess/preprocess_msmarco_docs.py \
  performance.processpool_chunksize_divisor=2 \
  performance.processpool_min_chunksize=128 \
  performance.state_save_interval_rows=200000 \
  dedup.near.max_candidates_per_doc=64
```

Balanced quality/throughput:

```bash
.venv/bin/python script/preprocess/preprocess_msmarco_docs.py \
  performance.processpool_chunksize_divisor=4 \
  performance.processpool_min_chunksize=64 \
  dedup.near.hamming_threshold=3 \
  quality.enabled=true
```

Strict filtering:

```bash
.venv/bin/python script/preprocess/preprocess_msmarco_docs.py \
  filtering.pii_block_on_match=true \
  quality.enabled=true \
  quality.min_score=0.7 \
  dedup.near.hamming_threshold=2
```

## Quick validation

```bash
.venv/bin/python -m compileall src script test
.venv/bin/python -m pytest test/test_preprocess_text_clean.py \
  test/test_preprocess_wikipedia_dump.py \
  test/test_preprocess_msmarco_docs.py \
  test/test_preprocess_parquet_writer.py \
  test/test_preprocess_pipeline.py \
  test/test_preprocess_metrics.py \
  test/test_preprocess_runtime.py \
  test/test_preprocess_download.py \
  test/test_preprocess_hub.py \
  test/test_train_quality_classifier.py
```

Additional regression + benchmark commands:

```bash
.venv/bin/python -m pytest test/test_preprocess_filtering.py \
  test/test_preprocess_dedup_exact.py \
  test/test_preprocess_dedup_near.py \
  test/test_preprocess_quality.py \
  test/test_preprocess_resume.py

.venv/bin/python -m pytest -m benchmark test/bench_preprocess_stages.py --benchmark-only
```

Benchmarks are marked with `benchmark` and excluded from default `pytest` runs.

## Troubleshooting

- `externally-managed-environment` during install:
  - use the `.venv` setup shown in `Install`
- `No module named pytest`:
  - run tests with `.venv/bin/python -m pytest ...`
- `ValueError: badly formed help string` when running preprocess scripts on Python 3.14:
  - use Python 3.13 for script execution (`hydra-core==1.3.2` has argparse incompatibility on 3.14)
- Resume or dedup index confusion:
  - set `run.resume=false` for a clean run
  - this resets dedup LMDB indexes
  - if `run.resume=true` fails due dedup path/index validation, keep output path stable
    and restore LMDB files, or start a clean run
