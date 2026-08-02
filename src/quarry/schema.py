"""Schema fingerprinting for prism factor dumps.

crucible does not compute factors; it reads what prism produced. That makes the
producer/consumer contract the weakest link in the whole stack, and it fails
*silently*: prism reorders two columns, or changes a lookback from 20 to 22,
and Python happily loads the file. Every downstream IC, every admission
decision, every model is now attributed to the wrong factor, and nothing
errors.

The fingerprint closes that hole. It covers factor **names, parameters and
order**, plus the producer build id. A mismatch raises
:class:`~crucible.errors.SchemaMismatch` and refuses the file.

The fingerprint deliberately does *not* cover row counts, date ranges or file
sizes. Those legitimately change between dumps; the schema must not.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from crucible.determinism import stable_hash
from crucible.errors import SchemaMismatch

__all__ = ["FactorSpec", "SchemaFingerprint", "validate_fingerprint"]


def _freeze(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Sort params into a canonical, hashable form.

    Sorting matters: ``{"win": 20, "lag": 1}`` and ``{"lag": 1, "win": 20}``
    describe the same factor and must fingerprint identically, or every dict
    literal rewrite would look like a producer change.
    """
    return tuple(sorted((str(k), v) for k, v in params.items()))


@dataclass(frozen=True)
class FactorSpec:
    """One factor column as prism declares it.

    Parameters
    ----------
    name:
        Column name in the dump.
    params:
        The factor's parameters. Two factors sharing a name but differing in
        lookback are different factors, and this is what distinguishes them.
    dtype:
        Optional expected arrow/polars dtype string. Included in the digest
        when present, so a silent float64 -> float32 downcast is caught.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    dtype: str | None = None

    def key(self) -> tuple[str, tuple[tuple[str, Any], ...], str | None]:
        """Canonical hashable identity."""
        return (self.name, _freeze(self.params), self.dtype)

    def describe(self) -> str:
        """Human-readable form used in mismatch messages."""
        if not self.params:
            return self.name if self.dtype is None else f"{self.name}:{self.dtype}"
        p = ",".join(f"{k}={v!r}" for k, v in _freeze(self.params))
        base = f"{self.name}({p})"
        return base if self.dtype is None else f"{base}:{self.dtype}"


@dataclass(frozen=True)
class SchemaFingerprint:
    """The full identity of a prism dump's factor schema.

    Order is significant. Two dumps with the same factors in a different order
    are different schemas, because positional access anywhere downstream would
    silently transpose them.
    """

    factors: tuple[FactorSpec, ...]
    producer: str = "unknown"

    @classmethod
    def of(
        cls, factors: Iterable[FactorSpec | str], producer: str = "unknown"
    ) -> SchemaFingerprint:
        """Build from specs or bare names.

        Bare strings are accepted for the common case of a dump whose factors
        carry no parameters, but a parameterised factor passed as a string
        fingerprints as parameterless and will not catch a lookback change.
        """
        specs = tuple(FactorSpec(f) if isinstance(f, str) else f for f in factors)
        return cls(specs, producer)

    @property
    def names(self) -> tuple[str, ...]:
        """Factor column names in declaration order."""
        return tuple(f.name for f in self.factors)

    def digest(self) -> str:
        """Process-stable hex digest over names, params, order and producer.

        Uses :func:`crucible.determinism.stable_hash` rather than ``hash()``,
        which is salted per process and would produce a different fingerprint
        on every run.
        """
        return stable_hash("schema-v1", self.producer, tuple(f.key() for f in self.factors))

    def validate(self, other: SchemaFingerprint, *, source: str = "<dump>") -> None:
        """Raise :class:`SchemaMismatch` unless ``other`` matches exactly.

        The message names the specific difference -- missing, unexpected,
        reordered, or changed parameters -- because "schema mismatch" alone
        sends you diffing Parquet metadata by hand.
        """
        if self.digest() == other.digest():
            return

        mine, theirs = list(self.factors), list(other.factors)
        my_names, their_names = [f.name for f in mine], [f.name for f in theirs]

        missing = [n for n in my_names if n not in their_names]
        extra = [n for n in their_names if n not in my_names]
        problems: list[str] = []
        if missing:
            problems.append(f"missing from dump: {missing}")
        if extra:
            problems.append(f"unexpected in dump: {extra}")
        if not missing and not extra and my_names != their_names:
            problems.append(f"column order differs: expected {my_names}, dump has {their_names}")

        by_name = {f.name: f for f in theirs}
        for f in mine:
            g = by_name.get(f.name)
            if g is not None and f.key() != g.key():
                problems.append(f"{f.name}: expected {f.describe()}, dump has {g.describe()}")

        if self.producer != other.producer:
            problems.append(f"producer: expected {self.producer!r}, dump has {other.producer!r}")

        detail = "; ".join(problems) or "digests differ but no field-level difference was found"
        raise SchemaMismatch(
            f"refusing {source}: prism factor schema does not match the expected fingerprint. "
            f"{detail}. Re-dump from the matching prism build rather than relaxing this check -- "
            f"a mismatched producer silently reattributes every downstream result."
        )

    def to_json(self) -> str:
        """Serialise for storage alongside a dump or in an admission log."""
        return json.dumps(
            {
                "producer": self.producer,
                "factors": [
                    {"name": f.name, "params": dict(f.params), "dtype": f.dtype}
                    for f in self.factors
                ],
                "digest": self.digest(),
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> SchemaFingerprint:
        """Inverse of :meth:`to_json`, verifying the embedded digest."""
        obj = json.loads(payload)
        fp = cls(
            tuple(
                FactorSpec(f["name"], f.get("params") or {}, f.get("dtype")) for f in obj["factors"]
            ),
            obj.get("producer", "unknown"),
        )
        recorded = obj.get("digest")
        if recorded is not None and recorded != fp.digest():
            raise SchemaMismatch(
                f"fingerprint payload is self-inconsistent: recorded digest {recorded} "
                f"but recomputes to {fp.digest()}. The file has been edited by hand."
            )
        return fp

    def __len__(self) -> int:
        return len(self.factors)


def validate_fingerprint(
    expected: SchemaFingerprint | None,
    actual: SchemaFingerprint,
    *,
    source: str = "<dump>",
) -> SchemaFingerprint:
    """Check ``actual`` against ``expected``, returning ``actual``.

    ``expected=None`` skips validation and is the escape hatch for genuinely
    exploratory reads of an unknown dump. It is *not* the default anywhere in
    :mod:`quarry`: opting out of the producer contract has to be a visible
    choice at the call site, not an omission.
    """
    if expected is not None:
        expected.validate(actual, source=source)
    return actual
