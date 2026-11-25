.PHONY: help install lint test check serve load-test ranker monitor mlflow-ui docker-build docker-run

-include .env
export

PORT ?= 8000
WORKERS ?= 1
RUN = OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1 | sort

install:
	pip install -r requirements.txt && pip install -e . --no-deps

lint:
	ruff check .

test:
	$(RUN) pytest -q

check: lint test

serve:
	$(RUN) uvicorn recsys.api:app --port $(PORT) --workers $(WORKERS)

load-test:
	python scripts/load_test.py --seconds 10 --levels 1 8 32

ranker:
	$(RUN) python scripts/run_ranker.py

monitor:
	$(RUN) python scripts/run_monitor.py

mlflow-ui:
	mlflow ui --backend-store-uri $${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}

docker-build:
	docker build -t recsys-api .

docker-run:
	docker run --rm -p $(PORT):8000 \
	  -v "$(CURDIR)/artifacts:/app/artifacts:ro" \
	  -v "$(CURDIR)/data/processed/items.parquet:/app/data/items.parquet:ro" \
	  recsys-api