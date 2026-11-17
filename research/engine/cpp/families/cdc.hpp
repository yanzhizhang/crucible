// cdc: PM's common block; cdc.lp = the last trade price.
#pragma once

#include "base.hpp"

namespace eng {

struct cdc : family {
    static constexpr const char* stem = "cdc";
    static constexpr status st = status::pinned;
    void dump(std::string& o, const tape& t) { put(o, "cdc.lp", t.last); }
};

}  // namespace eng
