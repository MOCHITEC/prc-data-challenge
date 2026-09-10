PY=.venv/bin/python
.PHONY: bootstrap test lint synth download features cv predict

bootstrap:            ## create venv, install, verify env, fetch data if keys are present, run tests (idempotent)
	uv venv -q .venv || true
	uv pip install -q --python $(PY) -e ".[dev]"
	$(PY) -c "import prc, lightgbm, polars; print('env ok')"
	@if [ -n "$$OSN_ACCESS_KEY" ]; then $(PY) -c "from prc.io_minio import download_all; print(len(download_all()), 'files')"; else echo "OSN_ACCESS_KEY not set: skipping download"; fi
	$(PY) -m pytest -q

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

synth:                ## small synthetic dataset for pipeline development
	$(PY) -c "from pathlib import Path; from prc.synth import write_synthetic_dataset as w; print(w(Path('data/synth')))"

download:
	$(PY) -c "from prc.io_minio import download_all; print(len(download_all()), 'files')"
