import os
import threading
import time
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel

from recsys import serve


class Item(BaseModel):
    item_id: int
    title: str | None = None
    score: float | None = None
    sources: list[str] | None = None


class RecommendResponse(BaseModel):
    user_id: int
    cold_start: bool
    items: list[Item]


class RelatedResponse(BaseModel):
    item_id: int
    fallback: bool
    items: list[Item]


class UpNextResponse(BaseModel):
    user_id: int
    anchor: Item | None = None
    items: list[Item]


class TrendingResponse(BaseModel):
    items: list[Item]


class LatencyStats:
    """Rolling per-endpoint latency, kept in memory and exposed at /metrics."""

    def __init__(self, keep=10_000):
        self._lat = defaultdict(lambda: deque(maxlen=keep))
        self._count, self._errors = Counter(), Counter()
        self.cold_start = 0
        self._lock = threading.Lock()

    def record(self, name, ms, status):
        with self._lock:
            self._lat[name].append(ms)
            self._count[name] += 1
            self._errors[name] += status >= 500

    def note_cold_start(self):
        with self._lock:
            self.cold_start += 1

    def snapshot(self):
        with self._lock:
            out = {}
            for name, d in self._lat.items():
                p50, p95, p99 = np.percentile(np.array(d), [50, 95, 99])
                out[name] = {"count": self._count[name], "errors": self._errors[name],
                             "p50_ms": round(float(p50), 3), "p95_ms": round(float(p95), 3),
                             "p99_ms": round(float(p99), 3)}
            return {"endpoints": out, "cold_start_responses": self.cold_start}


def create_app(recommender=None):
    """Pass a ready Recommender (tests), or let it load from ARTIFACT_DIR / ITEMS_PATH."""

    @asynccontextmanager
    async def lifespan(app):
        app.state.rec = recommender or serve.Recommender.load(
            os.environ.get("ARTIFACT_DIR", "artifacts"),
            os.environ.get("ITEMS_PATH", "data/processed/items.parquet"))
        app.state.stats = LatencyStats()
        yield

    app = FastAPI(title="Two-stage recommender", lifespan=lifespan)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        route = request.scope.get("route")
        app.state.stats.record(route.path if route else request.url.path, ms, response.status_code)
        response.headers["X-Process-Time-Ms"] = f"{ms:.3f}"
        return response

    @app.get("/health")
    def health(request: Request):
        rec = request.app.state.rec
        return {"status": "ok", "n_users": int(rec.a.X.shape[0]), "n_items": len(rec.a.pop),
                "sources": rec.sources}

    @app.get("/metrics")
    def metrics(request: Request):
        return request.app.state.stats.snapshot()

    @app.get("/recommend/{user_id}", response_model=RecommendResponse)
    def recommend(request: Request, user_id: int, k: int = Query(10, ge=1, le=50),
                  lam: float = Query(0.0, ge=0.0, le=5.0), n_new: int = Query(0, ge=0, le=10)):
        out = request.app.state.rec.recommend(user_id, k, lam=lam, n_new=min(n_new, k))
        if out["cold_start"]:
            request.app.state.stats.note_cold_start()
        return out

    @app.get("/related/{item_id}", response_model=RelatedResponse)
    def related(request: Request, item_id: int, k: int = Query(10, ge=1, le=50)):
        return request.app.state.rec.related(item_id, k)

    @app.get("/up-next/{user_id}", response_model=UpNextResponse)
    def up_next(request: Request, user_id: int, k: int = Query(10, ge=1, le=50)):
        return request.app.state.rec.up_next(user_id, k)

    @app.get("/trending", response_model=TrendingResponse)
    def trending(request: Request, k: int = Query(10, ge=1, le=50)):
        return request.app.state.rec.trending(k)

    return app


app = create_app()