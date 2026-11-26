"""Turn the CSV files in reports/ into Markdown tables, so numbers in the README are never typed by hand."""
import re

import numpy as np


def fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        a = abs(v)
        return f"{v:,.0f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.4f}"
    return str(v).replace("|", "\\|")


def md_table(df, index=True, index_name=""):
    d = df.reset_index() if index else df.copy()
    cols = [index_name if (index and i == 0) else str(c) for i, c in enumerate(d.columns)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def inject(text, name, body):
    """Replace what sits between <!-- name:START --> and <!-- name:END --> with `body`."""
    pat = re.compile(rf"(<!-- {re.escape(name)}:START -->)(.*?)(<!-- {re.escape(name)}:END -->)", re.DOTALL)
    if not pat.search(text):
        raise KeyError(name)
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text, count=1)