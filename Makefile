PY ?= python

.PHONY: install pipeline data models reports test clean

install:
	$(PY) -m pip install -r requirements.txt

pipeline:
	$(PY) run_pipeline.py

data:
	$(PY) run_pipeline.py --steps load sql

models:
	$(PY) run_pipeline.py --steps train

reports:
	$(PY) run_pipeline.py --steps eda segment excel powerbi

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/warehouse/*.db data/processed/* models/*.joblib models/*.pt models/*.json
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
