#pragma once

namespace Loader {
struct Program;
// Demon's Souls 01.005.000: a job worker's loop over a list of up to 1024 ids (a GPU result read back) skips an id
// whose table entry, index array, object array or object is null or outside the guest's range instead of crashing on
// it, and a second loop over records naming such ids passes over a record with a bad index (demonsSoulsIdListGuard.cpp).
namespace DemonsSoulsIdListGuard {
void Install(Program* program);
} // namespace DemonsSoulsIdListGuard
} // namespace Loader
