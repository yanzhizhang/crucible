// stock = the shared tape + one object per family, in the registry order of research/engine/families/__init__.py.
#pragma once

#include <tuple>

#include "zzpk.hpp"
#include "zzab.hpp"
#include "zzup.hpp"
#include "zzxu.hpp"
#include "zzvx.hpp"
#include "zzbd.hpp"
#include "zzeb.hpp"
#include "zzba.hpp"
#include "zzqj.hpp"
#include "zzeg.hpp"
#include "zzmh.hpp"
#include "zzvm.hpp"
#include "zzxo.hpp"
#include "zzui.hpp"
#include "zzhj.hpp"
#include "zzfb.hpp"
#include "zzjb.hpp"
#include "zzgh.hpp"
#include "cdc.hpp"
#include "zzcb.hpp"
#include "zztx.hpp"

namespace eng {

using family_set = std::tuple<zzpk, zzab, zzup, zzxu, zzvx, zzbd, zzeb, zzba, zzqj, zzeg, zzmh, zzvm, zzxo, zzui, zzhj, zzfb, zzjb, zzgh, cdc, zzcb, zztx>;

// The built-in rule table: every family's averages, registry order (must equal Python's families.EMAS; selftest checks).
inline std::vector<rule> builtin_rules() {
    std::vector<rule> o;
    family_set fs;
    std::apply([&](auto&... f) { (f.rules(o), ...); }, fs);
    return o;
}

struct stock : tape {
    family_set fams;
    explicit stock(const std::vector<rule>& rules) : tape(rules) {}
    void fams_snap(double t, double s, double lsod, double amount) {
        std::apply([&](auto&... f) { (f.on_snap(*this, t, s, lsod, amount), ...); }, fams);
    }
    void fams_trade(double t, double s, double amt, double side) {
        std::apply([&](auto&... f) { (f.on_trade(*this, t, s, amt, side), ...); }, fams);
    }
    void fams_dump(std::string& o) {
        std::apply([&](auto&... f) { (f.dump(o, *this), ...); }, fams);
    }
};

}  // namespace eng
