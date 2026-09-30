"""CED (common equity data) for crucible: a port of shtcommon ``ced`` writing Parquet, not ``.xr``.

Same SQL, same rules, same datasets as the PM's CED (Wind / JYDB / 朝阳永续 databases and the
Barra delivery files), so the research panels match what the PM used; the storage is one
hive-partitioned Parquet table per dataset (:mod:`ced.store`) instead of one netCDF file per
day, and every range is one query sliced per day (identical to running each day).

Datasets (``store/ced/<name>/date=YYYYMMDD/part.parquet``, one row per symbol unless noted):

=======================  =================================================  ==============
dataset                  content                                            CED file
=======================  =================================================  ==============
daily_sod                live SOD: is_traded, limits, shares, ret_adj       daily/*_SOD.xr
daily_sod_hist           hist SOD (AShareEODPrices row T, authoritative)     daily/*_SOD.xr
daily_eod                OHLC, preclose, adj_factor, limits, vol, tot, ...  daily/*_EOD.xr
index_eod                index bars (fixed code list, null rows kept)       daily/*_idx_EOD.xr
index_universe           pre-open index weights (live drift model)          index/*.xr
index_universe_hist      same, hist overnight factors                       index/*.xr
st                       st_<type> 0/1 per symbol with an active ST event    st/*.xr
wind_div / zy_div        cash dividend per share taking effect that day     wind_div, zy_div
sw_industry              level1..3 codes (JYDB 申万, standard 38)            sw_industry/*_ind{n}.xr
wind_industry            level1..4 codes (Wind '62' tree)                   wind_industry/*_ind{n}.xr
concept                  long: (symbol, concept_code)                       concept/*.xr
barra_{exp,stats,rate,fret,cov}  Barra CNE6                                 barra/*_{kind}.xr
=======================  =================================================  ==============

Catalogs (code -> Chinese name): ``store/ced/<dataset>/_static/catalog.parquet``.
Trading calendar: :mod:`ced.calendar` (Wind ASHARECALENDAR, cached to Parquet).

Which day to read (CED's rule, unchanged): from 08:45 on day T read T's files, except
daily_eod / index_eod / barra and the industry / concept tables, which are read at T-1.

Run: ``python research/run_ced.py <job> [--date D | --start S --end E] [--hist] [--overwrite]``;
``all-hist`` rebuilds every dataset for a range on the authoritative (after-close) basis.
Database URLs: see :mod:`ced.db` (environment only).
"""
