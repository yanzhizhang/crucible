# Performance log

Performance is a first-class requirement of the PM-reproduction track. Every heavy step runs
under `bench.resmon.ResourceMonitor`, which appends one row per run to
`/work/crucible_data/bench/history.parquet` (wall time, per-core CPU, peak RSS of the process
tree, read throughput, GPU when present). This file records the decisions those numbers drove.

## Machines

| machine | CPU | RAM | GPU | notes |
|---|---|---|---|---|
| dev box (Windows + WSL2 Ubuntu 26.04) | i5-10500, 6C/12T | 15.7 GB host, **11 GB + 8 GB swap for WSL** (was 7 GB until 2026-09-29) | Intel UHD 630 only, no CUDA | raw dumps live on `C:` and are read through the 9p `/mnt/c` mount |
| 97 (intranet) | tbd (`lscpu`) | tbd | tbd (`nvidia-smi`) | |

GPU paths (LightGBM CUDA, cudf-polars, CuPy) cannot be validated on the dev box.

## W1 -- decode one day of feitu v3 dumps

Input: ~29 GB of zstd capnp per day (order 12 GB, quotation 11 GB, transaction 6.9 GB,
index 0.1 GB), one file per minute per stream.

### Python (pycapnp) vs C++ (kernels/, nanobind)

One busy minute (10:30, 20260615), single thread, decode + read every field:

| stream | records | pycapnp | C++ kernel | speed-up |
|---|---:|---:|---:|---:|
| order | 1.61 M | 0.15 M rec/s | 1.40 M rec/s | ~9x |
| transaction | 0.98 M | 0.16 M rec/s | 1.60 M rec/s | ~10x |
| quotation (10-level book) | 0.18 M | -- | 0.35 M rec/s | |

The C++ numbers include zstd decompression; pycapnp's exclude it. Field-by-field parity of the
two decoders was checked on 200k order records and 1k quotation books (identical).

Decision: pycapnp would need several hours per day single-threaded (~6.9e8 records), so the
C++ kernel is the production decoder. It streams zstd (the decompressed minute never exists in
memory whole) and releases the GIL, so a Python thread pool scales it across cores.

### Full day, C++ kernel + pyarrow Parquet writer

| day | threads | stream | records | wall | CPU avg | RSS peak | read |
|---|---:|---|---:|---:|---:|---:|---:|
| 20260615 | 12 | order | 378.2 M | 145 s | 66% | 4.2 GB | 79 MB/s |
| 20260615 | 12 | transaction | 248.4 M | 98 s | 62% | 4.3 GB | 72 MB/s |
| 20260615 | 12 | quotation | 57.1 M | 133 s | 61% | 2.7 GB | 79 MB/s |
| 20260615 | 12 | index | 3.5 M | 3 s | 38% | 0.6 GB | 41 MB/s |
| 20260805 | 8 | order | 382.5 M | 204 s | 54% | 3.0 GB | 54 MB/s |
| 20260805 | 8 | transaction | 246.7 M | 105 s | 39% | 2.9 GB | 67 MB/s |
| 20260805 | 8 | quotation | 56.0 M | 139 s | 37% | 1.8 GB | 70 MB/s |

A whole day decodes in about 6.5 minutes (target was 30).

**Bottleneck: reading from the Windows drive, not CPU.** CPU never saturates while read
throughput sits at 55-80 MB/s, the practical ceiling of WSL's 9p mount under parallel reads.
Next steps, in order of payoff:
1. Keep raw dumps on the WSL ext4 disk (`/work`), or decode on the machine that holds them
   (97 / the colo box) -- expected to lift the ceiling to NVMe speed and make it CPU-bound.
2. Then profile the kernel with `perf` (packed-capnp unpacking is the likely hotspot).

### Memory

WSL saw 7 GB at the time. The open-of-day minute files are up to 224 MB compressed (~20 M records), so
decoding with 12 threads unbounded ran the VM out of memory. Fixes now in place:

* streaming zstd input in the kernel (no whole-file decompressed buffer; reads chunked, because
  9p fails very large single reads with ENOMEM);
* a weighted memory budget in `research/decode_feitu_day.py` (5x compressed size per task,
  default 60% of available memory, largest files scheduled first);
* quality-check latency quantiles from a log-binned histogram instead of an exact quantile
  over a 3e8-row column (which was OOM-killed).

## W0 -- quality gate on one day

Peak RSS per check, 20260615 (6.9e8 records), measured one check per process:

| check | before | after | fix |
|---|---:|---:|---|
| book sanity (crossed top of book) | 6.1 GB | 0.4 GB | per-file evaluation; polars does not stream `arr.get` on fixed-size lists, so a whole-day scan materialised both 10-level books |
| sequence contiguity | 2.1 GB | 2.1 GB | rewritten as two streaming passes with a `uint8` presence counter per channel (no sort); peak now bounded by the id range, not the file layout |
| latency (order stream) | -- | 2.2 GB | log-binned histogram instead of exact quantiles (the exact version was OOM-killed) |
| trade prices vs limits | -- | 3.3 GB | largest remaining; streaming join of 2.5e8 trades |

Whole run: 5.5 GB -> 3.3 GB peak, ~250 s, CPU ~80-95%. The book check is now single-threaded
(24 s); parallelising it over files is the next easy win.

## W2 -- 1-minute bars

Per-minute-file partial aggregation (first/last carried with their `(exchange_ts, seq)` keys),
merged at the end: 80-96 s per day, 2.4-3.3 GB peak, 1.23 M bar rows (5,189 stocks x 238 slots).
Validated: end-of-day volume identity exact for every stock, amount relative error 2e-15, bar
close equals the last snapshot price.

Two WSL service crashes (`Wsl/Service/E_UNEXPECTED`) happened while heavy jobs from two
workstreams shared the 7 GB VM. Raising the WSL memory limit in `.wslconfig` (host has
15.7 GB) was done on 2026-09-29: `memory=11GB`, `swap=8GB`, `autoMemoryReclaim=gradual`.

## Classify after aggregating (2026-09-29)

The stock filter (`is_a_share`, a string test on the 6-digit code) applied per row before a
`group_by` over 2.5e8 trades cost **8.5 GB / 80 s** for the TDX cross-check; moving it after
the aggregation (5e3 symbols) made it **2.1 GB / 19 s** with identical results. Same change in
the three quote-side checks. Whole quality run on 20260615: **7.6 GB -> 2.4 GB** peak.
Rule: never evaluate a derived per-row predicate that only depends on the group key before a
large `group_by` -- aggregate on the key, then filter the groups.
