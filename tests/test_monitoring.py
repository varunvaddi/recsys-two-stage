import numpy as np
import pandas as pd

from recsys import monitoring as mon


def test_psi_is_near_zero_for_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    assert mon.psi(ref, rng.normal(0, 1, 20_000)) < 0.02
    assert mon.psi(ref, rng.normal(1.0, 1, 20_000)) > 0.25
    assert mon.psi(ref, rng.normal(0, 3, 20_000)) > mon.WARN


def test_psi_ignores_nans_and_handles_constant_features():
    assert mon.psi([1.0, np.nan, 2.0, 3.0, 4.0] * 50, [1.0, 2.0, 3.0, 4.0] * 50) < 0.05
    assert mon.psi(np.ones(100), np.ones(50)) == 0.0
    assert mon.psi(np.ones(100), np.full(50, 2.0)) == float("inf")


def test_status_thresholds():
    assert [mon.status(v) for v in (0.0, 0.1, 0.3)] == ["ok", "warn", "alert"]


def test_drift_report_flags_only_the_shifted_feature():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(2, 1, 5000)})
    rep = mon.drift_report(ref, cur, ["a", "b"])
    assert rep.iloc[0]["feature"] == "b" and rep.iloc[0]["status"] == "alert"
    assert rep.set_index("feature").loc["a", "status"] == "ok"


def test_snapshot_uses_only_history_before_the_cutoff():
    day = 86400
    inter = pd.DataFrame({"user_id": [0, 0, 1, 1], "item_id": [0, 1, 0, 1],
                          "ts": [1 * day, 5 * day, 2 * day, 20 * day]})
    users, items = mon.snapshot(inter, t_end=10 * day)
    assert items.loc[0, "item_pop"] == 2 and items.loc[1, "item_pop"] == 1  # the day-20 event is excluded
    assert users.loc[1, "user_n_hist"] == 1 and users.loc[0, "user_last_days"] == 5
    assert items.loc[0, "item_age_days"] == 9


def test_quality_alert():
    assert mon.quality_alert(0.025, 0.034) is True
    assert mon.quality_alert(0.032, 0.034) is False