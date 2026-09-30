"""Public A-share data sources, loaded from the ``a-stock-data`` skill's own embedded code.

The skill (``.claude/skills/a-stock-data/SKILL.md``) is the maintained source of truth for
dozens of free endpoints and their hard-won edge cases. Copying functions out of it by hand
would freeze a fork that silently drifts from the skill; instead this module extracts the
fenced Python blocks that define the functions we use and executes them into one namespace,
so updating the skill updates the fetchers.

Safety: blocks mix definitions with usage examples (network calls, prints, deliberate
``ValueError`` demos). Each block is parsed with :mod:`ast` and only imports, function/class
definitions, ``try``-imports and constant assignments are kept; any top-level statement that
calls a skill-defined function, and every bare expression / loop, is dropped.

Extranet only (these hosts are public internet). Results are pandas frames exactly as the
skill returns them; :mod:`astock.fetch` converts and caches them as Parquet.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import ModuleType
from typing import Any

SKILL = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "a-stock-data" / "SKILL.md"

#: Functions we need; the blocks defining them (and their shared helpers) are loaded in order.
WANTED = (
    "def get_prefix(",
    "def norm_ticker(",
    "def em_get(",  # Eastmoney rate-limited GET (anti-ban), used by st_stock_list
    "def bs_session(",  # baostock login context, the st_stock_list fallback when Eastmoney is down
    "def _v39_http(",  # v3.9 shared helpers: http, contract, frame, date/number parsing
    "def tencent_kline(",
    "def tdx_daily_package(",
    "def tencent_ticks(",
    "def sina_adjust_factor(",
    "def sw_industry_history(",
    "def st_stock_list(",
    "def index_constituents(",
)

_KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Try)


def _python_blocks(text: str) -> list[tuple[int, str]]:
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "```python":
            j = i + 1
            while j < len(lines) and lines[j].strip() != "```":
                j += 1
            out.append((i + 2, "\n".join(lines[i + 1 : j])))
            i = j
        i += 1
    return out


def _defined_names(tree: ast.Module) -> set[str]:
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)}


def _calls(node: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
        for n in ast.walk(node)
    )


def _definitions_only(src: str, skill_funcs: set[str]) -> str:
    tree = ast.parse(src)
    kept = []
    for node in tree.body:
        if isinstance(node, _KEEP) or (
            isinstance(node, ast.Assign | ast.AnnAssign) and not _calls(node, skill_funcs)
        ):
            kept.append(node)
    return ast.unparse(ast.Module(body=kept, type_ignores=[]))


def _use_os_trust_store() -> None:
    """Verify TLS with the OS certificate store (still verifying, never ``verify=False``).

    certifi's bundle lacks some Chinese sites' intermediates (swsresearch.com fails with
    "unable to get local issuer certificate"); the Windows store fetches missing
    intermediates and also holds any corporate root CA.
    """
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def load(skill: Path = SKILL) -> ModuleType:
    """Build a module from the skill's code blocks that define :data:`WANTED`."""
    _use_os_trust_store()
    text = skill.read_text(encoding="utf-8")
    version = re.search(r"^version:\s*(\S+)", text, re.M)
    blocks = _python_blocks(text)
    chosen: list[tuple[int, str]] = []
    for want in WANTED:
        hits = [(ln, b) for ln, b in blocks if want in b]
        if not hits:
            raise RuntimeError(f"skill no longer defines {want!r} -- update WANTED")
        for h in hits:
            if h not in chosen:
                chosen.append(h)
    chosen.sort()
    funcs: set[str] = set()
    for _, b in chosen:
        funcs |= _defined_names(ast.parse(b))
    mod = ModuleType("astock_skill")
    mod.__dict__["__skill_version__"] = version.group(1) if version else "?"
    ns: dict[str, Any] = mod.__dict__
    for line, b in chosen:
        code = compile(_definitions_only(b, funcs), f"{skill.name}:{line}", "exec")
        exec(code, ns)
    return mod


skill = load()
__all__ = ["SKILL", "WANTED", "load", "skill"]
