// One struct per PM factor family (mirror of research/engine/families/*.py): its status, its averages (the built-in
// rule table), its own state beyond the shared tape, its hooks and its keys under PM's names. guessed = subject to change.
#pragma once

#include <string>
#include <vector>

#include "../state.hpp"

namespace eng {

enum class status { pinned, guessed, open };
inline const char* status_name(status s) { return s == status::pinned ? "pinned" : s == status::guessed ? "guessed" : "open"; }

// Base: no averages, no extra state, no keys. Hooks run after the tape took the event.
struct family {
    static void rules(std::vector<rule>&) {}
    void on_snap(const tape&, double /*t*/, double /*s*/, double /*lsod*/, double /*amount*/) {}
    void on_trade(const tape&, double /*t*/, double /*s*/, double /*amt*/, double /*side*/) {}
    void dump(std::string&, const tape&) {}
};

}  // namespace eng
