import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_serve import NAMES, _world

from recsys import api, ranker, serve


@pytest.fixture(scope="module")
def client():
    _, X, _, _, art = _world()
    rng = np.random.default_rng(1)
    f = pd.DataFrame(rng.random((600, len(NAMES))), columns=NAMES)
    f["user_id"], f["item_id"] = np.repeat(np.arange(30), 20), np.tile(np.arange(20), 30)
    f["label"] = (f["content_sim"] > 0.8).astype(int)
    booster = ranker.fit_ranker(f, NAMES, n_estimators=5, min_child_samples=1).booster_
    rec = serve.Recommender(art, booster=booster)
    with TestClient(api.create_app(rec)) as c:
        c.seen = X
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["n_items"] == 40


def test_recommend_returns_k_unseen_items_with_timing_header(client):
    r = client.get("/recommend/3?k=5")
    assert r.status_code == 200
    body = r.json()
    ids = [i["item_id"] for i in body["items"]]
    assert len(ids) == 5 and len(set(ids)) == 5 and body["cold_start"] is False
    assert not set(ids) & set(client.seen[3].indices.tolist())
    assert float(r.headers["X-Process-Time-Ms"]) > 0


def test_recommend_options_are_validated(client):
    assert client.get("/recommend/3?k=0").status_code == 422
    assert client.get("/recommend/3?k=100").status_code == 422
    assert client.get("/recommend/3?lam=-1").status_code == 422
    assert client.get("/recommend/3?k=5&lam=0.5&n_new=2").status_code == 200


def test_unknown_user_gets_trending_not_an_error(client):
    r = client.get("/recommend/999999?k=4")
    assert r.status_code == 200
    assert r.json()["cold_start"] is True and len(r.json()["items"]) == 4


def test_other_surfaces(client):
    assert len(client.get("/related/2?k=3").json()["items"]) == 3
    assert client.get("/related/99999").json()["fallback"] is True
    assert len(client.get("/up-next/3?k=4").json()["items"]) == 4
    assert len(client.get("/trending?k=5").json()["items"]) == 5


def test_metrics_report_per_endpoint_latency(client):
    for _ in range(3):
        client.get("/recommend/3?k=5")
    m = client.get("/metrics").json()
    rec = m["endpoints"]["/recommend/{user_id}"]
    assert rec["count"] >= 3 and rec["p50_ms"] > 0 and rec["errors"] == 0
    assert m["cold_start_responses"] >= 1