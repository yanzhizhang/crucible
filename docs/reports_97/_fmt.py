"""Rebuild polars-printed tables (UTF8_FULL_CONDENSED, left aligned, dtypes and shape hidden) from transcribed rows."""

def cell(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)

def tbl(cols, rows):
    s = [[cell(v) for v in r] for r in rows]
    w = [max([len(c)] + [len(r[i]) for r in s]) for i, c in enumerate(cols)]
    line = lambda l, m, r, f: l + m.join(f * (x + 2) for x in w) + r
    row = lambda vals: "│ " + " ┆ ".join(v.ljust(w[i]) for i, v in enumerate(vals)) + " │"
    out = [line("┌", "┬", "┐", "─"), row(cols), line("╞", "╪", "╡", "═")]
    out += [row(r) for r in s]
    out.append(line("└", "┴", "┘", "─"))
    return "\n".join(out)

def parse(block):
    """'a b 0.1 2' lines -> rows; numbers with '.' -> float, digits -> int."""
    rows = []
    for ln in block.strip().splitlines():
        r = []
        for t in ln.split("\t") if "\t" in ln else ln.split():
            if t.lstrip("-").replace(".", "", 1).isdigit() and "." in t: r.append(float(t))
            elif t.lstrip("-").isdigit(): r.append(int(t))
            else: r.append(t)
        rows.append(r)
    return rows
