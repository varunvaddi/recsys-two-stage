import numpy as np
import pandas as pd
from scipy import sparse

from recsys import demo


def test_search_titles_is_case_insensitive_and_literal():
    titles = pd.Series(["Super Mario Bros", "mario kart", "Zelda (2017)", None], index=[3, 4, 5, 6])
    assert demo.search_titles(titles, "MARIO").index.tolist() == [3, 4]
    assert demo.search_titles(titles, "(2017)").index.tolist() == [5]  # not treated as a regex
    assert len(demo.search_titles(titles, "  ")) == 0


def test_example_users_have_enough_history_and_are_reproducible():
    X = sparse.csr_matrix(np.array([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]))
    a = demo.example_users(X, n=5, min_hist=3, seed=1)
    assert a.tolist() == [0, 2]
    assert demo.example_users(X, n=1, min_hist=3, seed=1).tolist() == demo.example_users(X, n=1, min_hist=3, seed=1).tolist()


def test_items_frame_flags_new_items_and_handles_missing_titles():
    pop = np.array([10, 0, 5])
    df = demo.items_frame([{"item_id": 1, "title": None, "sources": ["pop"]}, {"item_id": 0, "title": "A"}], pop)
    assert df["new"].tolist() == [True, False] and df["plays"].tolist() == [0, 10]
    assert df.loc[0, "title"] == "item 1" and df.loc[1, "sources"] == ""