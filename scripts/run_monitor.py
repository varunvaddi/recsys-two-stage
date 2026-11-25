import argparse

import mlflow
import pandas as pd

from recsys import data, monitoring, tracking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--out", default="reports/drift.csv")
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    train, val, _ = data.time_split(df)
    t_ref, t_cur = train["ts"].max(), val["ts"].max()  # the ranker's cutoff vs one period later
    gap = (t_cur - t_ref) / 86400
    u_ref, i_ref = monitoring.snapshot(df, t_ref)
    u_cur, i_cur = monitoring.snapshot(df, t_cur)

    parts = [monitoring.drift_report(u_ref, u_cur, list(u_ref.columns)),
             monitoring.drift_report(i_ref, i_cur, list(i_ref.columns))]
    report = pd.concat(parts, ignore_index=True).sort_values("psi", ascending=False)

    print(f"reference: data up to the ranker cutoff; current: {gap:.0f} days later")
    print(f"users {len(u_ref):,} -> {len(u_cur):,}, items {len(i_ref):,} -> {len(i_cur):,}\n")
    print(report.round(4).to_string(index=False))
    report.to_csv(a.out, index=False)

    with tracking.run("monitoring", "feature-drift", {"gap_days": round(gap), "n_users": len(u_cur),
                                                        "n_items": len(i_cur)}):
        for _, r in report.iterrows():
            mlflow.log_metric(tracking.clean(f"psi/{r['feature']}"), r["psi"])
        mlflow.log_metric("psi_max", report["psi"].max())
        mlflow.log_artifact(a.out)


if __name__ == "__main__":
    main()