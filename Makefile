PY=.venv/bin/python
.PHONY: bootstrap test lint synth smoke download run compare

bootstrap:            ## create venv, install, fetch data if keys are present, run tests (idempotent)
	uv venv -q .venv || true
	uv pip install -q --python $(PY) -e ".[dev]"
	$(PY) -c "import prc, lightgbm, polars; print('env ok')"
	@if [ -n "$$OSN_ACCESS_KEY" ]; then \
	  $(PY) -c "from prc.io_minio import download_all; print(len(download_all()), 'files')" || echo "download failed"; \
	else echo "OSN_ACCESS_KEY not set: skipping download"; fi
	$(PY) -m pytest -q

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

download:             ## fetch the competition parquet files
	$(PY) -c "from prc.io_minio import download_all; print(len(download_all()), 'files')"

synth:                ## small synthetic dataset, used by the tests
	$(PY) -c "from pathlib import Path; from prc.synth import write_synthetic_dataset as w; print(w(Path('data/synth')))"

smoke:                ## end-to-end run on synthetic data (logs to experiments/synthetic_smoke.jsonl)
	$(PY) -m prc.pipeline --config configs/v8_seed_avg.yaml --synthetic --raw-dir data/synth --out-dir outputs/synth

run:                  ## CV + ranking prediction + gate, no upload: make run CONFIG=configs/v8_seed_avg.yaml
	$(PY) -m prc.pipeline --config $(CONFIG)

compare:              ## adopt/reject two experiments: make compare A=<exp_id> B=<exp_id>
	$(PY) -m prc.compare $(A) $(B)
