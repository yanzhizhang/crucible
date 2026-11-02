// The event file the C++ engine reads: what research/engine/export.py writes from the catalog.
#pragma once

#include <cstdint>

namespace eng {

// 16-byte header, then n_events records sorted by (sid, local receive stamp, kind).
struct file_header {
    char magic[8];  // "CRUEV001"
    std::uint64_t n_events;
};

constexpr char magic[8] = {'C', 'R', 'U', 'E', 'V', '0', '0', '1'};

// One snapshot or trade; every number a double so the engine sees exactly what Python saw.
struct event {
    std::int32_t sid;
    std::int16_t mkt;
    std::int16_t kind;  // 0 snapshot, 1 trade, 2 order (stamps only)
    double t;           // trading clock of local arrival (09:30 = 0, lunch removed)
    double s;           // exchange second of day
    double lsod;        // local second of day
    double price;
    double f[7];        // snapshot: b1 a1 hi lo op pc amount; trade: vol side 0 0 0 0 0
};

static_assert(sizeof(file_header) == 16, "header layout");
static_assert(sizeof(event) == 96, "event layout mirrors export.py's numpy dtype");

}  // namespace eng
