"""Index registry (verbatim from shtcommon ``configs.IndexSpec``): name, JY InnerCodes, Wind codes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexSpec:
    name: str
    jy_codes: tuple[str, ...]  # JY InnerCode, first is primary
    wind_codes: tuple[str, ...]  # Wind code, SH/SZ twins allowed, first is primary
    cc_slug: str | None = None

    @property
    def jy_code(self) -> str:
        return self.jy_codes[0]

    @property
    def wind_code(self) -> str:
        return self.wind_codes[0]


INDICES: list[IndexSpec] = [
    IndexSpec("SSE 50 Index", ("46",), ("000016.SH",), "cw_sh50"),
    IndexSpec("CSI 300 Index", ("3145", "3146"), ("000300.SH", "399300.SZ"), "cw_hs300"),
    IndexSpec("CSI A500 Index", ("636661",), ("000510.SH",), None),
    IndexSpec("CSI Smallcap 500 Index", ("4978",), ("000905.SH", "399905.SZ"), "cw_zz500"),
    IndexSpec("CSI 1000 Index", ("39144",), ("000852.SH", "399852.SZ"), "cw_zz1000"),
    IndexSpec("CSI 2000 Index", ("561230",), ("932000.CSI",), None),
    IndexSpec("CNI 300 Index", ("3470",), ("399312.SZ",), None),
    IndexSpec("CNI 1000 Index", ("3469",), ("399311.SZ",), None),
    IndexSpec("CNI 2000 Index", ("33792",), ("399303.SZ",), None),
]
REGISTRY: dict[str, IndexSpec] = {s.name: s for s in INDICES}
NAMES: list[str] = [s.name for s in INDICES]
WIND_CODE_MAP: dict[str, str] = {s.name: s.wind_code for s in INDICES}
WIND_CODE_INDEX_MAP: dict[str, str] = {wc: s.name for s in INDICES for wc in s.wind_codes}
JY_CODE_MAP: dict[str, str] = {s.name: s.jy_code for s in INDICES}
JY_CODE_INDEX_MAP: dict[str, str] = {jc: s.name for s in INDICES for jc in s.jy_codes}
WIND_WEIGHT_TABLE = "AIndexHS300FreeWeight"
