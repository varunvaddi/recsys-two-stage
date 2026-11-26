import numpy as np
import pandas as pd
import pytest

from recsys import report


def test_fmt_picks_sensible_precision_and_handles_nan():
    assert report.fmt(0.034009946) == "0.0340"
    assert report.fmt(12.3456) == "12.35"
    assert report.fmt(1234.7) == "1,235"
    assert report.fmt(np.nan) == "–"
    assert report.fmt("a|b") == "a\\|b"


def test_md_table_has_header_separator_and_rows():
    df = pd.DataFrame({"recall": [0.0243, 0.0198]}, index=["ranker", "pop"])
    lines = report.md_table(df, index_name="model").splitlines()
    assert lines[0] == "| model | recall |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| ranker | 0.0243 |" and len(lines) == 4


def test_inject_replaces_only_the_marked_block_and_is_repeatable():
    text = "a\n<!-- t:START -->\nold\n<!-- t:END -->\nb"
    once = report.inject(text, "t", "NEW")
    assert once == "a\n<!-- t:START -->\nNEW\n<!-- t:END -->\nb"
    assert report.inject(once, "t", "NEW") == once
    with pytest.raises(KeyError):
        report.inject(text, "missing", "x")