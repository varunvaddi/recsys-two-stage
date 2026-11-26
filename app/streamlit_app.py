import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from recsys import demo, serve

ART = os.environ.get("ARTIFACT_DIR", "artifacts")
ITEMS = os.environ.get("ITEMS_PATH", "data/processed/items.parquet")
REPORTS = Path(os.environ.get("REPORTS_DIR", "reports"))

st.set_page_config(page_title="Two-stage recommender", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load():
    return serve.Recommender.load(ART, ITEMS)


def show(items, pop, key):
    df = demo.items_frame(items, pop)
    st.dataframe(df, hide_index=True, width="stretch", key=key,
                 column_config={"new": st.column_config.CheckboxColumn("new item"),
                                "plays": st.column_config.NumberColumn("plays (train)")})
    return df


def read_report(name):
    p = REPORTS / name
    return pd.read_csv(p, index_col=0) if p.exists() else None


rec = load()
pop, X = rec.a.pop, rec.a.X
users = demo.example_users(X, n=30)

st.title("Two-stage video-game recommender")
st.caption(f"{X.shape[0]:,} users, {X.shape[1]:,} games (Amazon Reviews 2023, Video Games). "
           "Candidates from 5 sources, ranked by LightGBM, then re-ranked for popularity bias.")

tab_rec, tab_bt, tab_next, tab_trend, tab_res, tab_ops = st.tabs(
    ["Recommended for you", "Frequently bought together", "Up next", "Trending", "Results", "Operations"])

with tab_rec:
    c1, c2, c3 = st.columns([2, 2, 2])
    uid = c1.selectbox("Example user", users, format_func=lambda u: f"user {u} ({X[u].nnz} games)")
    lam = c2.slider("Popularity penalty (lambda)", 0.0, 2.0, 0.0, 0.25,
                    help="Higher = push down very popular games")
    n_new = c3.slider("Reserved new-game slots", 0, 5, 2,
                      help="Slots kept for games that nobody bought in the training period")
    owned = [rec.titles.get(int(i), f"item {i}") if rec.titles is not None else f"item {i}"
             for i in X[uid].indices[:8]]
    st.write("**Already owns (sample):** " + "; ".join(owned))
    left, right = st.columns(2)
    t0 = time.perf_counter()
    plain = rec.recommend(int(uid), 10)
    ms = (time.perf_counter() - t0) * 1000
    tuned = rec.recommend(int(uid), 10, lam=lam, n_new=n_new)
    with left:
        st.subheader("Ranker only")
        st.caption(f"{ms:.1f} ms")
        a = show(plain["items"], pop, "plain")
    with right:
        st.subheader("With your settings")
        b = show(tuned["items"], pop, "tuned")
    st.write(f"Median plays of the list: **{a['plays'].median():,.0f}** vs **{b['plays'].median():,.0f}**; "
             f"new games: **{int(a['new'].sum())}** vs **{int(b['new'].sum())}**.")

with tab_bt:
    q = st.text_input("Search for a game", "")
    found = demo.search_titles(rec.titles, q) if rec.titles is not None else pd.Series(dtype=str)
    if len(found):
        item = st.selectbox("Pick one", list(found.index), format_func=lambda i: found[i])
        out = rec.related(int(item), 10)
        if out["fallback"]:
            st.info("Not enough co-purchase data for this game, showing trending games instead.")
        show(out["items"], pop, "bt")
    else:
        st.write("Type part of a title, e.g. *zelda*.")

with tab_next:
    u2 = st.selectbox("User", users, key="u2", format_func=lambda u: f"user {u}")
    out = rec.up_next(int(u2), 10)
    if out["anchor"]:
        st.write(f"After their last game: **{out['anchor']['title'] or out['anchor']['item_id']}**")
    show(out["items"], pop, "next")

with tab_trend:
    st.caption("Time-decayed popularity, half-life 30 days.")
    show(rec.trending(15)["items"], pop, "trend")

with tab_res:
    final, ranker_dev = read_report("final_test.csv"), read_report("ranker.csv")
    if final is not None:
        st.subheader("Final test period (never used for tuning)")
        st.dataframe(final.round(4), width="stretch")
        st.bar_chart(final["recall@10"])
        st.caption("Recall@10 with 95% bootstrap intervals in the table. The ranker was trained on an "
                   "earlier period and frozen; sources and features were refit on train+validation.")
    if ranker_dev is not None:
        st.subheader("Development (validation period)")
        st.dataframe(ranker_dev.round(4), width="stretch")
    if final is None and ranker_dev is None:
        st.info("No report files found. Run the scripts in scripts/ first.")

with tab_ops:
    drift, load_test = pd.read_csv(REPORTS / "drift.csv") if (REPORTS / "drift.csv").exists() else None, None
    lt = REPORTS / "load_test_w4.csv"
    if lt.exists():
        load_test = pd.read_csv(lt)
    if drift is not None:
        st.subheader("Feature drift (PSI)")
        st.dataframe(drift.round(4), hide_index=True, width="stretch")
    if load_test is not None:
        st.subheader("Load test")
        st.dataframe(load_test[load_test["endpoint"] == "ALL"], hide_index=True, width="stretch")
    st.caption(f"Model sources: {', '.join(rec.sources)}")