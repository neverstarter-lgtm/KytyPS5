#pragma once

#include <cstdint>

namespace Loader {
struct Program;
// Demon's Souls 01.005.000: a job worker's loop over a list of up to 1024 ids (a GPU result read back) skips an id
// whose table entry, index array, object array or object is null or outside the guest's range instead of crashing on
// it, and a second loop over records naming such ids passes over a record with a bad index (demonsSoulsIdListGuard.cpp).
namespace DemonsSoulsIdListGuard {
void Install(Program* program);
// A fault at `pc` (reading `address`) inside one of the guards' checked loads: the address of that guard's skip path to
// resume at (the item is skipped as an invalid one), otherwise 0.
[[nodiscard]] uint64_t RecoveryTarget(uint64_t pc, uint64_t address);
} // namespace DemonsSoulsIdListGuard
} // namespace Loader
