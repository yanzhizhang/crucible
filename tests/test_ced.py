"""research/ced (CED port): rules on synthetic data, and full SQL -> Parquet paths with a fake database."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from ced import calendar as cal_mod  # noqa: E402
from ced import check, concept, daily, db, index_universe, store, st, wind_div, zy_div  # noqa: E402
from ced.calendar import SSE, SZSE, calendar  # noqa: E402

DAYS = ["20260427", "20260428", "20260429", "20260430", "20260506", "20260507"]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE", tmp_path / "ced")
    monkeypatch.setattr(cal_mod, "CACHE", tmp_path / "ced" / "calendar")
    for ex in (SSE, SZSE):
        calendar.set_days(DAYS, ex)
    yield
    calendar.invalidate()


def fake_db(monkeypatch, tables: dict[str, pd.DataFrame]) -> list[str]:
    """Route read_sql to frames keyed by a table name found in the SQL; returns the SQL log."""
    seen: list[str] = []

    def read_sql(sql: str, dbname: str = db.WIND, *, what: str | None = None) -> pd.DataFrame:
        seen.append(sql)
        for key, df in tables.items():
            if key in sql:
                return df.copy()
        return pd.DataFrame()

    monkeypatch.setattr(db, "read_sql", read_sql)
    return seen


# ----------------------------------------------------------------------------- calendar / store
def test_calendar_prev_next_semantics():
    assert calendar.prev("20260430") == "20260429"
    assert calendar.prev("20260501") == "20260430"  # not a trading day: strictly before
    assert calendar.next("20260430") == "20260506"
    assert calendar.trade_days("20260428", "20260506") == ["20260428", "20260429", "20260430", "20260506"]
    with pytest.raises(ValueError):
        calendar.offset("20260501", 1)


def test_calendar_offline_uses_cache(monkeypatch, tmp_path):
    calendar.invalidate()
    monkeypatch.delenv("CRUCIBLE_WIND_URL", raising=False)
    monkeypatch.delenv("CRUCIBLE_SHTCOMMON", raising=False)
    monkeypatch.delenv("CRUCIBLE_DB_USER", raising=False)
    monkeypatch.setattr(db, "DB_USER", "")  # a locally filled-in account must not leak into the test
    monkeypatch.setattr(db, "DB_PASSWORD", "")
    cal_mod.CACHE.mkdir(parents=True)
    pd.DataFrame({"trade_day": DAYS}).to_parquet(cal_mod.CACHE / "exchange=SSE.parquet", index=False)
    assert calendar.trade_days("20260429", "20260506") == ["20260429", "20260430", "20260506"]


def test_store_roundtrip_and_skip():
    df = pd.DataFrame({"symbol": ["600000.SH"], "x": [1.0]})
    store.write(df, "demo", "20260429")
    store.write(df.assign(x=2.0), "demo", "20260429", overwrite=False)
    assert store.read_day("demo", "20260429")["x"].item() == 1.0
    store.write(df.assign(x=3.0), "demo", "20260430")
    got = store.read("demo", "20260429", "20260430").collect().sort("date")
    assert got["date"].to_list() == ["20260429", "20260430"] and got["x"].to_list() == [1.0, 3.0]


# ----------------------------------------------------------------------------- daily rules
def test_round_half_up_is_not_bankers():
    assert daily._round_half_up(np.array([3.75 * 1.1]))[0] == pytest.approx(4.13)
    assert daily._round_half_up(np.array([10.7 * 0.9]))[0] == pytest.approx(9.63)


def _noex() -> pd.DataFrame:
    cols = ["wind_code", "ex_date", "cash", "bonus", "conversed", "rights_ratio", "rights_price"]
    return daily._exdiv_on(pd.DataFrame(columns=cols), "none")


def _base(**over) -> pd.DataFrame:
    row = {"freeshare": 100.0, "circshare": 200.0, "totshare": 300.0, "prev_close": 10.0,
           "prev_preclose": 9.5, "prev_limit": 10.45, "prev_stopping": 8.55, "list_board": "主板",
           "list_date": "20100101", "delist_date": ""}
    row.update(over)
    return pd.DataFrame([row], index=pd.Index(["600000.SH"], name="wind_code"))


def test_live_sod_limits_ret_adj_and_shares():
    ex = pd.DataFrame({"wind_code": ["600000.SH"], "ex_date": ["20260429"], "cash": [0.5], "bonus": [0.0],
                       "conversed": [0.0], "rights_ratio": [0.0], "rights_price": [0.0]}).set_index("wind_code")
    out = daily._build_sod_one("20260429", _base(), ex, set(), None).iloc[0]
    assert out["ret_adj"] == pytest.approx(9.5 / 10.0)  # reference price 9.50
    assert out["maxp_allowed"] == pytest.approx(10.45) and out["minp_allowed"] == pytest.approx(8.55)
    assert out["freeshare"] == 1_000_000 and out["is_traded"] == 1


def test_live_sod_bse_rounding_suspension_and_no_band():
    base = _base(prev_close=50.02, prev_limit=np.nan, prev_preclose=np.nan, list_board="北证")
    base.index = pd.Index(["830000.BJ"], name="wind_code")
    out = daily._build_sod_one("20260429", base, _noex(), {"830000.BJ"}, None).iloc[0]
    assert out["maxp_allowed"] == pytest.approx(65.02) and out["minp_allowed"] == pytest.approx(35.02)
    assert out["is_traded"] == 0
    new = _base(list_date="20260428")  # 2nd trading day of a new listing: no band
    out = daily._build_sod_one("20260429", new, _noex(), set(), None).iloc[0]
    assert out["maxp_allowed"] == 0.0 and out["minp_allowed"] == 0.0


def test_shares_switch_only_when_effective():
    src = pd.DataFrame({"wind_code": ["600000.SH", "600000.SH"], "change_dt": ["20260101", "20260429"],
                        "ann_dt": ["20260101", "20260429"], "eff_dt": ["20260101", "20260429"],
                        "freeshare": [100.0, 150.0]})
    out = daily._with_shares_asof(_base(), [(src, ("freeshare",), "AShareFreeFloat")], "20260428", "20260429")
    assert out.loc["600000.SH", "freeshare"] == 150.0
    out = daily._with_shares_asof(_base(), [(src.iloc[:1], ("freeshare",), "AShareFreeFloat")], "20260428", "20260429")
    assert out.loc["600000.SH", "freeshare"] == 100.0


def test_eod_and_sod_hist_end_to_end(monkeypatch):
    eod = pd.DataFrame({"trade_dt": ["20260429"], "wind_code": ["600000.SH"], "open": [10.0], "high": [11.0],
                        "low": [9.8], "close": [10.5], "preclose": [9.9], "adj_factor": [2.0],
                        "maxp_allowed": [10.89], "minp_allowed": [8.91], "volume_lot": [12.0],
                        "amount_kyuan": [12.6], "vwap": [10.5], "list_date": ["20100101"]})
    hist = pd.DataFrame({"trade_dt": ["20260428", "20260429"], "wind_code": ["600000.SH"] * 2,
                         "prev_close": [9.9, 10.0], "maxp_allowed": [10.89, 11.0], "minp_allowed": [8.91, 9.0],
                         "adj_factor": [2.0, 2.1], "prev_adj_factor": [2.0, 2.0], "trade_status_code": [-1, 0],
                         "freeshare": [1.0, 1.0], "circshare": [2.0, 2.0], "totshare": [3.0, 3.0],
                         "list_board": ["主板"] * 2, "list_date": ["20100101"] * 2, "delist_date": [""] * 2})
    fake_db(monkeypatch, {"LAG(e.S_DQ_ADJFACTOR)": hist, "AShareEODPrices e": eod})
    daily.convert_range_eod("20260429", "20260429")
    e = store.read_day("daily_eod", "20260429").iloc[0]
    assert e["vol"] == 1200 and e["tot"] == pytest.approx(12600.0) and e["ret_o_pc"] == pytest.approx(10 / 9.9)
    daily.convert_range_sod_hist("20260429", "20260429")
    s = store.read_day("daily_sod_hist", "20260429").iloc[0]
    assert s["is_traded"] == 0 and s["ret_adj"] == pytest.approx(2.0 / 2.1) and s["totshare"] == 30_000


def test_sod_hist_listing_from_eod_rows_only():
    base = pd.DataFrame({"prev_close": [10.0, 5.0], "maxp_allowed": [11.0, 0.0], "minp_allowed": [9.0, 0.0],
                         "adj_factor": [1.0, 1.0], "prev_adj_factor": [1.0, np.nan], "trade_status_code": [-1, 1],
                         "freeshare": [1.0, 1.0], "circshare": [1.0, 1.0], "totshare": [1.0, 1.0],
                         "list_board": ["", ""], "list_date": ["", ""], "delist_date": ["20200101", ""]},
                        index=pd.Index(["600000.SH", "600001.SH"], name="wind_code"))
    out = daily._build_sod_hist_one("20260429", base, ["600000.SH", "600001.SH", "600002.SH"]).set_index("symbol")
    # Description says delisted / no list date, but EOD has the rows: EOD decides
    assert out.loc["600000.SH", "is_traded"] == 1 and out.loc["600001.SH", "is_traded"] == 1
    assert out.loc["600001.SH", "ret_adj"] == 1.0  # first day: no previous factor
    assert out.loc["600002.SH", "is_traded"] == 0  # no EOD row


# ----------------------------------------------------------------------------- dividends / ST
def test_dividend_aggregation_rules():
    rec = pd.DataFrame({"wind_code": ["A", "A", "B", "B"], "ex_dt": ["d"] * 4,
                        "cash_dvd_per_sh_pre_tax": [0.4921, 0.3906, 1.25, 0.31],
                        "cash_dvd_per_sh_after_tax": [0.4, 0.3, 1.0, 0.3],
                        "div_object": ["普通股股东", "大股东", "", ""], "is_public": [True, False, True, True]})
    out = wind_div.aggregate_per_share(rec, "t").set_index("symbol")
    assert out.loc["A", "cash_dvd_per_sh_pre_tax"] == pytest.approx(0.4921)  # differentiated row dropped
    assert out.loc["B", "cash_dvd_per_sh_pre_tax"] == pytest.approx(1.56)  # two plans added


def test_zy_div_rules(monkeypatch):
    raw = pd.DataFrame({
        "stock_code": ["600989", "600989", "002975", "000001"], "ex_dt": ["20260429"] * 4,
        "pre_tax_per_10": [4.921, 3.0, None, 5.0], "pre_tax_min_per_10": [None] * 4,
        "after_tax_per_10": [4.0, 2.5, None, 4.5], "distri_type": ["1002", "1002", "1004", "1002"],
        "scheme_type": ["1001", "1001", "1001", "1002"], "object_type": ["1002", "1001", "1001", "1001"],
        "object_desc": ["", "", "", ""], "is_newest": [1, 1, 1, 1], "is_valid": [1, 1, 1, 1]})
    fake_db(monkeypatch, {"bas_stk_hisdistribution": raw})
    zy_div.convert_range("20260429", "20260429")
    out = store.read_day("zy_div", "20260429").set_index("symbol")
    assert out.loc["600989.SH", "cash_dvd_per_sh_pre_tax"] == pytest.approx(0.4921)  # circulating row only
    assert out.loc["002975.SZ", "cash_dvd_per_sh_pre_tax"] == 0.0  # bonus only: 0 row kept
    assert "000001.SZ" not in out.index  # draft plan dropped


def test_st_active_rule(monkeypatch):
    ev = pd.DataFrame({"wind_code": ["A", "B"], "st_type": ["S", "Y"], "entry_dt": ["20260428", "20260101"],
                       "remove_dt": ["", "20260429"]})
    fake_db(monkeypatch, {"AShareST": ev})
    st.convert_range("20260429", "20260429")
    out = store.read_day("st", "20260429")
    assert out["symbol"].tolist() == ["A"] and out.loc[0, "st_S"] == 1 and out.loc[0, "st_Y"] == 0


# ----------------------------------------------------------------------------- concept
def test_concept_listing_day_not_counted_single_or_range():
    rng = pd.DataFrame({"instrument": ["X.SH"] * 3 + ["Y.SH"] * 2, "concept_name": ["c"] * 5,
                        "concept_code": ["K1", "K2", "K3", "K1", "K2"], "index_code": pd.array([647090000] * 5, "Int32"),
                        "entry_dt": ["20260101"] * 5, "remove_dt": [concept._FAR_FUTURE] * 5,
                        "list_dt": ["20260101", "20260429", "20260101", "20260101", "20260101"],
                        "expire_dt": [concept._FAR_FUTURE] * 5})
    out = concept.apply_filters(concept.slice_asof(rng, "20260429"), "20260429")
    assert "K2" not in out.loc[out["instrument"] == "X.SH", "concept_code"].tolist()
    out = concept.apply_filters(concept.slice_asof(rng, "20260430"), "20260430")
    assert "K2" in out.loc[out["instrument"] == "X.SH", "concept_code"].tolist()


# ----------------------------------------------------------------------------- index universe
def test_index_drift_model():
    anchors = pd.DataFrame({("CSI 300 Index", "A"): [0.5], ("CSI 300 Index", "B"): [0.5]}, index=["20260428"])
    anchors.columns = pd.MultiIndex.from_tuples(anchors.columns, names=["index_name", "wind_code"])
    ret = pd.DataFrame({"A": [1.1], "B": [1.0]}, index=["20260429"])
    oadj = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=["20260430"])
    wide = index_universe._carry_forward_morning(anchors, ret, oadj, ["20260429", "20260430"], "20260428")
    d29 = index_universe.day_frame(wide, "20260429", ["CSI 300 Index"]).set_index("symbol")["CSI 300 Index"]
    assert d29["A"] == pytest.approx(0.5)  # pre-open 29 = anchor of 28
    d30 = index_universe.day_frame(wide, "20260430", ["CSI 300 Index"]).set_index("symbol")["CSI 300 Index"]
    wa, wb = 0.55 / 1.05, 0.5 / 1.05 * 2.0  # drift by 29's return, then overnight factor
    assert d30["A"] == pytest.approx(wa / (wa + wb)) and d30["B"] == pytest.approx(wb / (wa + wb))


# ----------------------------------------------------------------------------- checks
def test_compare_frames_codes_and_mismatch_table():
    left = pd.DataFrame({"maxp_allowed": [10.0, 11.0], "freeshare": [1.0, 2.0]}, index=["600000.SH", "000001.SZ"])
    right = left.copy()
    right.loc["000001.SZ", "freeshare"] = 3.0
    r = check.compare_frames(left, right, names=("live", "hist"), rtol=1e-6, atol=1e-9, warn_frac=0.01,
                             col_rules=check.HIST_COL_RULES["daily_sod"])
    assert r.rc == check.RC_MINOR  # share columns never alarm
    right.loc["600000.SH", "maxp_allowed"] = 10.01
    r = check.compare_frames(left, right, names=("live", "hist"), rtol=1e-6, atol=1e-9, warn_frac=0.01,
                             col_rules=check.HIST_COL_RULES["daily_sod"])
    assert r.rc == check.RC_WARN and len(r.mismatches()) == 2
