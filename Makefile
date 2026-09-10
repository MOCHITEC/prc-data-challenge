PY=.venv/bin/python
.PHONY: bootstrap test lint synth smoke download board profile doctor v1 v2

bootstrap:            ## create venv, install, verify env, fetch data if keys are present, run tests (idempotent)
	uv venv -q .venv || true
	uv pip install -q --python $(PY) -e ".[dev]"
	$(PY) -c "import prc, lightgbm, polars; print('env ok')"
	@if [ -n "$$OSN_ACCESS_KEY" ] || [ -n "$$OSN_ID" ]; then \
	  $(PY) -c "from prc.io_minio import download_all; print(len(download_all()), 'files')" \
	  || echo "download failed: run 'make doctor' for the diagnosis (bootstrap continues)"; \
	else echo "OSN_ACCESS_KEY not set: skipping download"; fi
	$(PY) -m pytest -q

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

synth:                ## small synthetic dataset for pipeline development
	$(PY) -c "from pathlib import Path; from prc.synth import write_synthetic_dataset as w; print(w(Path('data/synth')))"

smoke:                ## end-to-end run on the synthetic data (logs to experiments/synthetic_smoke.jsonl)
	$(PY) -m prc.pipeline --config configs/v1_baseline.yaml --synthetic --raw-dir data/synth --out-dir outputs/synth

download:
	$(PY) -c "from prc.io_minio import download_all; print(len(download_all()), 'files')"

board:                ## snapshot the official leaderboard and show movement vs the previous snapshot
	$(PY) -m prc.leaderboard

profile:              ## answer the docs/01 open questions from the real files
	$(PY) -m prc.profile

v1:                   ## baseline submission (gate only, no upload)
	$(PY) -m prc.pipeline --config configs/v1_baseline.yaml

v2:                   ## two-clock model submission (gate only, no upload)
	$(PY) -m prc.pipeline --config configs/v2_two_clock.yaml

doctor:               ## verify the OpenSky credentials and discover the buckets (read-only)
	$(PY) -m prc.doctor
