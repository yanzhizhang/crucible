"""HTML tearsheets.

Self-contained single-file output: charts are embedded as base64 PNGs, so a
report can be emailed or archived without dragging an asset directory along and
still renders identically in five years.

Every chart carries its **sample period and universe definition** in the
caption. That is not decoration. A tearsheet without them is unfalsifiable --
an IC of 0.06 means one thing on CSI 300 over ten years and something entirely
different on all A-shares over eight months, and by the time the chart reaches
a second reader the context is gone.

``matplotlib`` is imported lazily so the rest of crucible stays importable
without it.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from assay.results import DecayResult, ICResult, QuantileResult, TurnoverResult
from crucible.frames import TS, Frame, to_polars

__all__ = ["Caption", "factor_tearsheet", "strategy_tearsheet"]

_CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; max-width: 1080px; padding: 32px 20px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 8px; padding-bottom: 6px;
     border-bottom: 1px solid rgba(128,128,128,.3); }
.sub { opacity: .7; font-size: 13px; margin: 0 0 24px; }
figure { margin: 0 0 28px; }
figure img { width: 100%; height: auto; display: block; }
figcaption { font-size: 12px; opacity: .72; margin-top: 6px; font-style: italic; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px;
         border-bottom: 1px solid rgba(128,128,128,.22); }
th:first-child, td:first-child { text-align: left; }
thead th { font-weight: 600; opacity: .8; }
.warn { background: rgba(220,120,0,.14); border-left: 3px solid #d87800;
        padding: 10px 14px; margin: 16px 0; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px;
        margin: 16px 0 24px; }
.stat { border: 1px solid rgba(128,128,128,.25); border-radius: 8px; padding: 10px 12px; }
.stat .k { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; opacity: .65; }
.stat .v { font-size: 19px; font-variant-numeric: tabular-nums; margin-top: 2px; }
"""


@dataclass(frozen=True)
class Caption:
    """Provenance attached to every chart in a report.

    Parameters
    ----------
    universe:
        How the universe was defined, in words -- ``"CSI 500, ex-ST,
        ex-suspended"``. A rule, not a count: "1843 stocks" does not let a
        reader reproduce anything.
    """

    universe: str
    sample_start: dt.date | dt.datetime | None = None
    sample_end: dt.date | dt.datetime | None = None
    n_names: int | None = None
    n_periods: int | None = None
    frequency: str = "daily"
    note: str = ""

    def text(self) -> str:
        """One-line provenance string."""
        bits = [f"Universe: {self.universe}"]
        if self.sample_start and self.sample_end:
            bits.append(f"Sample: {_d(self.sample_start)} to {_d(self.sample_end)}")
        if self.n_periods:
            bits.append(f"{self.n_periods} {self.frequency} periods")
        if self.n_names:
            bits.append(f"{self.n_names} names")
        if self.note:
            bits.append(self.note)
        return " | ".join(bits)

    @classmethod
    def from_frame(cls, df: Frame, universe: str, **kw: Any) -> Caption:
        """Derive the sample span from a frame's own timestamps."""
        lf = to_polars(df)
        if TS not in lf.columns or lf.height == 0:
            return cls(universe=universe, **kw)
        return cls(
            universe=universe,
            sample_start=lf[TS].min(),  # type: ignore[arg-type]
            sample_end=lf[TS].max(),  # type: ignore[arg-type]
            n_periods=int(lf[TS].n_unique()),
            **kw,
        )


def _d(x: object) -> str:
    return x.strftime("%Y-%m-%d") if isinstance(x, (dt.date, dt.datetime)) else str(x)


def _fig_uri(fig: Any) -> str:
    """Render a matplotlib figure to an embeddable data URI."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _figure(fig: Any, caption: str) -> str:
    return (
        f'<figure><img alt="{html.escape(caption)}" src="{_fig_uri(fig)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _stats(pairs: Sequence[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="stat"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div></div>'
        for k, v in pairs
    )
    return f'<div class="grid">{cells}</div>'


def _table(df: pl.DataFrame, *, float_fmt: str = "{:.4f}") -> str:
    if df.height == 0:
        return "<p><em>no rows</em></p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
    body = []
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"<td>{float_fmt.format(v)}</td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _page(title: str, caption: Caption, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='sub'>{html.escape(caption.text())}</p>"
        f"{body}</body></html>"
    )


def _write(html_text: str, out_path: str | Path | None) -> str:
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_text, encoding="utf-8")
    return html_text


def factor_tearsheet(
    name: str,
    caption: Caption,
    *,
    ic: ICResult | None = None,
    quantiles: QuantileResult | None = None,
    decay: DecayResult | None = None,
    turnover: TurnoverResult | None = None,
    coverage: Frame | None = None,
    out_path: str | Path | None = None,
) -> str:
    """Render a single-factor report.

    Includes the IC series and its cumulative sum, quantile curves, the decay
    curve, turnover, and coverage over time. Sections for arguments left as
    ``None`` are simply omitted.

    Returns
    -------
    The HTML as a string, also written to ``out_path`` when given.

    Notes
    -----
    Thin samples are called out in the body rather than left for the reader to
    infer from an axis label. If :attr:`ICResult.is_thin` is set, the report
    says so at the top.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts: list[str] = []

    if ic is not None:
        parts.append(
            _stats(
                [
                    ("Mean IC", f"{ic.mean:+.4f}"),
                    ("ICIR", f"{ic.icir:+.3f}"),
                    ("t-stat", f"{ic.t_stat:+.2f}"),
                    ("Positive rate", f"{ic.positive_rate:.1%}"),
                    ("Periods", f"{ic.n_periods}"),
                    ("Mean breadth", f"{ic.mean_breadth:.0f}"),
                ]
            )
        )
        if ic.is_thin:
            parts.append(
                '<div class="warn"><strong>Thin sample.</strong> '
                f"{ic.n_periods} cross-sections is below the 30 needed for the IC "
                "moments to mean much. Treat every statistic here as indicative.</div>"
            )
        if ic.newey_west_lags == 0:
            parts.append(
                '<div class="warn">t-statistic is <strong>not</strong> corrected for '
                "overlapping label windows. If the label horizon exceeds one period, "
                "recompute with <code>newey_west_lags = horizon - 1</code>; the naive "
                "value overstates significance.</div>"
            )

        series = ic.series.filter(pl.col("ic").is_not_null())
        if series.height:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            x = series[TS].to_list()
            y = series["ic"].to_numpy()
            ax1.bar(x, y, width=0.9, color=["#2a7" if v > 0 else "#c44" for v in y])
            ax1.axhline(0, lw=0.8, color="black")
            ax1.set_ylabel("IC")
            ax2.plot(x, y.cumsum(), color="#357", lw=1.4)
            ax2.set_ylabel("cumulative IC")
            ax2.axhline(0, lw=0.8, color="black")
            fig.autofmt_xdate()
            parts.append(
                "<h2>Information coefficient</h2>"
                + _figure(
                    fig,
                    f"{ic.method.title()} rank IC per cross-section and its cumulative "
                    f"sum. {caption.text()}",
                )
            )

    if quantiles is not None and quantiles.curves.height:
        fig, ax = plt.subplots(figsize=(10, 4))
        for bucket, block in sorted(
            quantiles.curves.partition_by("bucket", as_dict=True).items()
        ):
            b = bucket[0] if isinstance(bucket, tuple) else bucket
            ax.plot(block[TS].to_list(), block["cum_ret"].to_numpy(), lw=1.2, label=f"Q{b}")
        ax.axhline(0, lw=0.8, color="black")
        ax.set_ylabel("cumulative return")
        ax.legend(ncol=5, fontsize=8, frameon=False)
        fig.autofmt_xdate()
        parts.append(
            "<h2>Quantile returns</h2>"
            + _stats(
                [
                    ("Long-short spread", f"{quantiles.spread_mean:+.4%}"),
                    ("Spread Sharpe", f"{quantiles.spread_sharpe:+.2f}"),
                    ("Monotonicity", f"{quantiles.monotonicity:+.2f}"),
                    ("Buckets", f"{quantiles.n_buckets}"),
                ]
            )
            + _figure(
                fig,
                f"Cumulative return by factor quantile, ranked within each period. "
                f"{caption.text()}",
            )
            + _table(quantiles.by_bucket)
        )
        if quantiles.monotonicity < 0.5:
            parts.append(
                '<div class="warn">Monotonicity below 0.5: the extreme buckets may be '
                "driven by outliers rather than a monotone exposure. Check the middle "
                "buckets before trading the spread.</div>"
            )

    if decay is not None and decay.curve.height:
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.plot(decay.curve["horizon"].to_list(), decay.curve["ic"].to_numpy(), "o-", color="#357")
        ax.axhline(0, lw=0.8, color="black")
        ax.set_xlabel("horizon (periods)")
        ax.set_ylabel("IC")
        hl = "not reached" if decay.half_life == float("inf") else f"{decay.half_life:.1f} periods"
        parts.append(
            "<h2>Signal decay</h2>"
            + _figure(
                fig,
                f"IC by holding horizon; peak {decay.peak_ic:+.4f} at horizon "
                f"{decay.peak_horizon}, half-life {hl}. t-stats are Newey-West "
                f"corrected per horizon. {caption.text()}",
            )
        )

    if turnover is not None and turnover.series.height:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(
            turnover.series[TS].to_list(), turnover.series["turnover"].to_numpy(), lw=1.1, color="#853"
        )
        ax.set_ylabel("rank turnover")
        ax.set_ylim(0, 1)
        fig.autofmt_xdate()
        parts.append(
            "<h2>Turnover</h2>"
            + _stats(
                [
                    ("Mean turnover", f"{turnover.mean:.3f}"),
                    ("Implied holding", f"{turnover.implied_holding_periods:.1f} periods"),
                ]
            )
            + _figure(
                fig,
                "1 - rank correlation with the previous cross-section, matched by "
                f"symbol. Compare implied holding against the decay half-life above -- "
                f"turning over faster than the alpha decays pays costs for nothing. "
                f"{caption.text()}",
            )
        )

    if coverage is not None:
        cov = to_polars(coverage)
        if cov.height and TS in cov.columns:
            fig, ax = plt.subplots(figsize=(10, 2.8))
            col = "n" if "n" in cov.columns else cov.columns[-1]
            ax.plot(cov[TS].to_list(), cov[col].to_numpy(), lw=1.1, color="#487")
            ax.set_ylabel("names")
            ax.set_ylim(bottom=0)
            fig.autofmt_xdate()
            parts.append(
                "<h2>Coverage</h2>"
                + _figure(
                    fig,
                    "Names with a valid factor value per period. A sharp drop is a data "
                    f"problem, not a regime change. {caption.text()}",
                )
            )

    return _write(_page(f"Factor tearsheet — {name}", caption, "".join(parts)), out_path)


def strategy_tearsheet(
    name: str,
    pnl: Frame,
    caption: Caption,
    *,
    positions: Frame | None = None,
    costs: Frame | None = None,
    return_col: str = "net_return",
    periods_per_year: float = 244.0,
    out_path: str | Path | None = None,
) -> str:
    """Render a strategy report: equity, drawdown, rolling Sharpe, cost attribution.

    Parameters
    ----------
    pnl:
        Per-period frame with ``ts`` and ``return_col``, e.g.
        :attr:`ledger.BacktestResult.pnl`.
    costs:
        Costed trades from :meth:`toll.CostModel.apply`. When supplied, PnL loss
        is attributed to commission, stamp duty, transfer fee and impact -- the
        breakdown that tells you whether the strategy is fighting tax or
        capacity, which imply completely different fixes.

    Returns
    -------
    The HTML as a string, also written to ``out_path`` when given.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from assay.time_series import performance

    lf = to_polars(pnl)
    perf = performance(lf, column=return_col, periods_per_year=periods_per_year)
    parts: list[str] = [
        _stats(
            [
                ("Sharpe", f"{perf.sharpe:+.2f}"),
                ("Ann. return", f"{perf.ann_return:+.2%}"),
                ("Ann. vol", f"{perf.ann_vol:.2%}"),
                ("Max drawdown", f"{perf.max_drawdown:.2%}"),
                ("DD duration", f"{perf.drawdown_duration} periods"),
                ("Calmar", f"{perf.calmar:+.2f}"),
                ("Monthly win", f"{perf.monthly_winrate:.1%}"),
                ("Periods", f"{perf.n}"),
            ]
        )
    ]
    if perf.is_thin:
        parts.append(
            '<div class="warn"><strong>Under one year of data.</strong> '
            "Annualised figures extrapolate a short sample and drawdown is almost "
            "certainly understated -- the worst period has not happened yet.</div>"
        )

    r = lf[return_col].fill_null(0.0).to_numpy()
    if r.size:
        eq = (1.0 + r).cumprod()
        peak = eq.copy()
        for i in range(1, len(peak)):
            peak[i] = max(peak[i - 1], eq[i])
        x = lf[TS].to_list() if TS in lf.columns else list(range(len(r)))

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        ax1.plot(x, eq, lw=1.4, color="#357")
        ax1.set_ylabel("equity (x)")
        ax2.fill_between(x, eq / peak - 1.0, 0, color="#c44", alpha=0.45)
        ax2.set_ylabel("drawdown")
        fig.autofmt_xdate()
        parts.append(
            "<h2>Equity and drawdown</h2>"
            + _figure(fig, f"Compounded net-of-cost equity curve. {caption.text()}")
        )

        window = max(20, int(periods_per_year / 4))
        if r.size > window:
            roll = pl.Series("r", r)
            rs = (
                roll.rolling_mean(window) / roll.rolling_std(window) * (periods_per_year**0.5)
            ).to_numpy()
            fig, ax = plt.subplots(figsize=(10, 2.8))
            ax.plot(x, rs, lw=1.1, color="#585")
            ax.axhline(0, lw=0.8, color="black")
            ax.set_ylabel(f"{window}p rolling Sharpe")
            fig.autofmt_xdate()
            parts.append(
                "<h2>Rolling Sharpe</h2>"
                + _figure(
                    fig,
                    f"Trailing {window}-period Sharpe, annualised at {periods_per_year:g} "
                    f"periods/year. {caption.text()}",
                )
            )

    if costs is not None:
        c = to_polars(costs)
        components = [
            x
            for x in ("commission", "stamp_duty", "transfer_fee", "impact_temporary", "impact_permanent")
            if x in c.columns
        ]
        if components:
            totals = {k: float(c[k].sum()) for k in components}
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.barh(list(totals), list(totals.values()), color="#853")
            ax.set_xlabel("total cost (CNY)")
            parts.append(
                "<h2>Cost attribution</h2>"
                + _figure(
                    fig,
                    "PnL loss by cost component. Tax and commission scale linearly with "
                    "turnover; impact scales with the square root of participation, so a "
                    f"large impact share means a capacity limit, not a fee problem. "
                    f"{caption.text()}",
                )
                + _table(
                    pl.DataFrame(
                        {"component": list(totals), "total_cny": list(totals.values())}
                    ),
                    float_fmt="{:,.0f}",
                )
            )

    if positions is not None:
        p = to_polars(positions)
        if "industry" in p.columns and "weight" in p.columns:
            expo = (
                p.group_by("industry")
                .agg(pl.col("weight").mean().alias("mean_weight"))
                .sort("mean_weight", descending=True)
            )
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.barh(expo["industry"].to_list(), expo["mean_weight"].to_numpy(), color="#487")
            ax.axvline(0, lw=0.8, color="black")
            ax.set_xlabel("mean net weight")
            parts.append(
                "<h2>Industry exposure</h2>"
                + _figure(
                    fig,
                    f"Average net weight by industry over the sample. {caption.text()}",
                )
            )

    return _write(_page(f"Strategy tearsheet — {name}", caption, "".join(parts)), out_path)
