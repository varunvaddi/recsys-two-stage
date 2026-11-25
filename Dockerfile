FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# serving needs far fewer packages than training (no torch, implicit, mlflow, faiss)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# model artifacts are big and not in git: mount them at run time (see Makefile: docker-run)
ENV ARTIFACT_DIR=/app/artifacts \
    ITEMS_PATH=/app/data/items.parquet \
    OMP_NUM_THREADS=1 \
    PORT=8000 \
    WORKERS=1

RUN useradd --create-home app
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os,urllib.request as u; u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"

CMD ["sh", "-c", "uvicorn recsys.api:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]