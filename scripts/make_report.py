"""Rebuild the result tables in README.md and docs/offline_eval_report.md from reports/*.csv."""
import argparse
from pathlib import Path

import pandas as pd

from recsys import report


def read(reports, name, **kw):
    p = Path(reports) / name
    return pd.read_csv(p, **kw) if p.exists() else None


def build_sections(r):
    s = {}
    for key, name in [("test", "final_test.csv"), ("dev", "ranker.csv"), ("ablations", "ablations.csv"),
                      ("bias", "popularity_bias.csv"), ("together", "together.csv"), ("seq", "seq.csv"),
                      ("faiss", "faiss.csv"), ("faiss_scale", "faiss_scale.csv"), ("serving", "serving.csv")]:
        df = read(r, name, index_col=0)
        if df is not None:
            s[key] = report.md_table(df)
    drift = read(r, "drift.csv")
    if drift is not None:
        s["drift"] = report.md_table(drift, index=False)
    shap = read(r, "shap_summary.csv", index_col=0)
    if shap is not None:
        s["shap"] = report.md_table(shap.sort_values(shap.columns[0], ascending=False).head(10), index_name="feature")
    rr = read(r, "rerank.csv")
    if rr is not None:
        rr = rr[rr["new_slots"].isin([0, 2]) & rr["lambda"].isin([0.0, 0.5, 1.0])]
        s["rerank"] = report.md_table(rr, index=False)
    loads = [x for x in (read(r, "load_test.csv"), read(r, "load_test_w4.csv")) if x is not None]
    if loads:
        lt = pd.concat(loads)
        s["load"] = report.md_table(lt[lt["endpoint"] == "ALL"].drop(columns=["endpoint"]), index=False)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--files", nargs="+", default=["README.md", "docs/offline_eval_report.md"])
    a = ap.parse_args()
    sections = build_sections(a.reports)
    for path in a.files:
        p = Path(path)
        if not p.exists():
            print(f"skip {path} (not found)")
            continue
        text, done = p.read_text(), []
        for name, body in sections.items():
            try:
                text = report.inject(text, name, body)
                done.append(name)
            except KeyError:
                pass
        p.write_text(text)
        print(f"{path}: updated {', '.join(done) if done else 'nothing'}")


if __name__ == "__main__":
    main()