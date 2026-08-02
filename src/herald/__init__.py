"""herald -- reporting.

Self-contained HTML tearsheets with charts embedded as data URIs. Every chart
carries its sample period and universe definition, because a factor statistic
without that context cannot be checked by a second reader.
"""

from __future__ import annotations

from herald.report import Caption, factor_tearsheet, strategy_tearsheet

__all__ = ["Caption", "factor_tearsheet", "strategy_tearsheet"]
