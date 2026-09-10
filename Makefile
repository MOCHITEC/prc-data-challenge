PY=.venv/bin/python
.PHONY: bootstrap test lint synth download board profile anatomy ab doctor v1 v2

bootstrap:            ## create venv, install, verify env, fetch data if keys are present, run tests (idempotent)
	uv venv -q .venv || true
	uv pip install -q --python $(PY) -e ".[dev]"
	$(PY) -c "import prc, lightgbm, polars; print('env ok')"
	@if [ -n "$$OSN_ACCESS_KEY" ] || [ -n "$$OSN_ID" ]; then $(PY) -c "from prc.io_minio import download_all; print(len(download_all()), 'files')"; else echo "OSN_ACCESS_KEY not set: skipping download"; fi
	$(PY) -m pytest -q

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

synth:                ## small synthetic dataset for pipeline development
	$(PY) -c "from pathlib import Path; from prc.synth import write_synthetic_dataset as w; print(w(Path('data/synth')))"

download:
	$(PY) -c "from prc.io_minio import download_all; print(len(download_all()), 'files')"

board:                ## snapshot the official leaderboard and show movement vs the previous snapshot
	$(PY) -m prc.leaderboard

profile:              ## answer the docs/01 open questions from the real files
	$(PY) -m prc.profile

anatomy:              ## where the squared error sits, per clock source (docs/06)
	$(PY) -m prc.anatomy

ab:                   ## A/B two OOF parquets by airport and clock source: make ab BASE=... NEW=...
	$(PY) -m prc.anatomy --compare $(BASE) $(NEW)

v1:                   ## baseline submission (gate only, no upload)
	$(PY) -m prc.pipeline --config configs/v1_baseline.yaml

v2:                   ## two-clock model submission (gate only, no upload)
	$(PY) -m prc.pipeline --config configs/v2_two_clock.yaml

doctor:               ## verify the OpenSky credentials and discover the buckets (read-only)
	$(PY) -m prc.doctor
