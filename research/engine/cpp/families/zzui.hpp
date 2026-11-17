// zzui: a 10-slot 10 s snapshot-price grid ending at 14:57:00 local (guess; the lead is a ~45 s price average).
#pragma once

#include "zzxo.hpp"

namespace eng {

struct zzui : price_grid<10> {
    static constexpr const char* stem = "zzui";
    static constexpr status st = status::guessed;
    void dump(std::string& o, const tape&) { dump_grid(o, "zzui.r."); }
};

}  // namespace eng
