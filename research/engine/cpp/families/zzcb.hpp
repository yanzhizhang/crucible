// zzcb: zzcb.r fits no clock x half-life x input yet; no code.
#pragma once

#include "base.hpp"

namespace eng {

struct zzcb : family {
    static constexpr const char* stem = "zzcb";
    static constexpr status st = status::open;
};

}  // namespace eng
