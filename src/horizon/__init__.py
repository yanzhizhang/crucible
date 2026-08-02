"""horizon -- labels.

Forward-looking targets built by explicit slot-index joins rather than
``shift``, so a gap in the panel produces null instead of a plausibly wrong
pairing. VWAP-to-VWAP is the default basis and entry is lagged one slot behind
the decision, because the price stamped at ``t`` is already complete and cannot
be traded.
"""

from __future__ import annotations

from horizon.labels import (
    assert_no_overlap,
    forward_return,
    slot_index,
    ternary_label,
    vol_normalized_return,
)

__all__ = [
    "assert_no_overlap",
    "forward_return",
    "slot_index",
    "ternary_label",
    "vol_normalized_return",
]
