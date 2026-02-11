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
- Optional push to Hugging Face Hub dataset repos

## Project structure

- `config/preprocess/`: Hydra configs for preprocess jobs
- `script/preprocess/`: corpus preprocess entrypoints
- `src/data/preprocess/`: download, clean, shard, hub, state helpers
- `src/runtime/`: lightweight script setup and logging helpers
- `test/`: preprocess-focused unit tests

## Install

```bash
python -m pip install -r requirements.txt
```

## Configs

- `config/preprocess/_base.yaml`: shared defaults
- `config/preprocess/wikipedia_dump.yaml`: Wikimedia dump workflow
- `config/preprocess/msmarco_docs.yaml`: MS MARCO docs workflow

## Wikipedia dump to Parquet

```bash
python script/preprocess/preprocess_wikipedia_dump.py \
  processing.num_proc=32 \
  output.output_dir=data/processed/wikipedia_dump \
  run.resume=true
```

Output schema:

- `id`
- `title`
- `body`
- `url`
- `source`
- `dump_date`

## MS MARCO docs to Parquet

```bash
python script/preprocess/preprocess_msmarco_docs.py \
  processing.num_proc=32 \
  output.output_dir=data/processed/msmarco_docs \
  run.resume=true
```

Output schema:

- `doc_id`
- `url`
- `title`
- `body`
- `source`
- `version`

## Resume semantics

Each run writes `resume_state.json` under `output.output_dir`.

- `run.resume=true`: resume from previously saved counters/shard index.
- `run.resume=false`: start a fresh run (ignore prior state).

Artifacts written per run:

- `*.parquet` shards
- `manifest.json`
- `run_summary.json`
- generated dataset card `README.md`

## Push to Hugging Face Hub

Set token and enable `hub.push`:

```bash
export HF_TOKEN=<your_hf_token>
python script/preprocess/preprocess_wikipedia_dump.py \
  hub.push=true \
  hub.repo_id=<org_or_user>/<dataset_name> \
  hub.path_in_repo=data
```

Retry knobs:

- `hub.retries`
- `hub.retry_backoff_seconds`

## Quick validation

```bash
python -m compileall src script test
pytest test/test_preprocess_text_clean.py \
  test/test_preprocess_wikipedia_dump.py \
  test/test_preprocess_msmarco_docs.py \
  test/test_preprocess_parquet_writer.py
```
