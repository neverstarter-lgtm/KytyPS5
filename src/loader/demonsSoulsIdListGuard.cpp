// Demon's Souls 01.005.000 eboot+0xa10050: a "BPE JobWorkerThread" job walks a list of ids (count at list + 0x8000, at
// most 0x400; the ids from + 0x8004). The list is a GPU result the emulator writes back each frame (a "write-back
// readback" of 0x9084 bytes at 0x26a529000). Each id names a table (bits 16-23) and an element (bits 0-15): the table's
// entry (object + 0x50, eight bytes an entry), the entry's object array (+ 0xb0) and its u32 index array (+ 0xe0), then
// an object from the array (+ 0x20 of it read next). Three crashes read null there, all with the count at its 0x400
// cap and ids that name no live element (0x3e744c33, 0x1, 0x2):
//   eboot+0xa10170 (mov rcx, [rax + 0xb0] with rax 0; 11-20 07:48, a GTX 1080, after ~8 minutes of play)
//   eboot+0xa1017e (mov r13d, [rax + rsi * 4] with rax 0; 11-20 08:28, the same, after ~20 minutes; and 11-18 21:29)
// Here each of those pointers is checked: an id with a null one is skipped (the loop goes on with the next id), the
// first ones and then every power of two printed. The checks cover what crashed; an id that reaches a live but wrong
// object is not caught.
#include "loader/demonsSoulsIdListGuard.h"

#include "common/abi.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif
#include <xbyak/xbyak.h>

namespace Loader::DemonsSoulsIdListGuard {
namespace {
constexpr uint64_t PageSize = 0x4000, CaveOffset = 0x8000000 + 12 * PageSize;
// The guest's user range (kernel/memoryAddressSpace.inc: HOST_USER_MAX 0xfbffffffff); nothing is mapped below 64 KiB.
constexpr uint32_t MinPointer = 0x10000;
constexpr uint64_t MaxPointer = 0xfc00000000ull;

// 01.005.000 eboot+0xa10126, the loop's end and its body up to the object's load:
//   mov rax, [rbp - 0x78]; mov [r12 - 0x14], ecx; mov [r12 - 0x10], edx;
//   inc rax; cmp r14, rax; je <end>;                                   (+0x0e: the next id)
//   mov ecx, [rbx + rax * 4 + 0x8004]; mov [rbp - 0x78], rax; mov eax, 0x810; mov r15, [rdi + 0x30];
//   bextr edx, ecx, eax; mov rax, [rdi + 0x50]; movzx esi, cx; mov [rbp - 0x70], rcx; mov [rbp - 0x68], rdx;
//   mov [rbp - 0x60], rsi;
//   mov rax, [rax + rdx * 8]; mov rcx, [rax + 0xb0]; mov rax, [rax + 0xe0]; mov r13d, [rax + rsi * 4];  (+0x46: the site)
//   mov [rbp - 0x58], rax; mov eax, r13d; and eax, 0xffffff; mov rax, [rcx + rax * 8]; mov rcx, [rax + 0x20];
//   cmp r15, [rdi + 0x38]                                              (+0x70: back)
constexpr const char* Loop =
    "48 8b 45 88 41 89 4c 24 ec 41 89 54 24 f0 48 ff c0 49 39 c6 0f 84 ?? ?? ?? ?? 8b 8c 83 04 80 00 00 48 89 45 88 b8 10 08 "
    "00 00 4c 8b 7f 30 c4 e2 78 f7 d1 48 8b 47 50 0f b7 f1 48 89 4d 90 48 89 55 98 48 89 75 a0 48 8b 04 d0 48 8b 88 b0 00 "
    "00 00 48 8b 80 e0 00 00 00 44 8b 2c b0 48 89 45 a8 44 89 e8 25 ff ff ff 00 48 8b 04 c1 48 8b 48 20 4c 3b 7f 38";
constexpr uint64_t NextOffset = 0x0e, SiteOffset = 0x46, BackOffset = 0x70;
constexpr uint32_t SiteSize = BackOffset - SiteOffset; // 42 bytes, nothing jumps into them

std::atomic<uint32_t> g_skips {0};

void KYTY_SYSV_ABI Skipped(uint64_t offset, uint64_t id, uint64_t index, uint64_t count, uint64_t list) {
	const auto skips = g_skips.fetch_add(1, std::memory_order_relaxed) + 1;
	if (skips > 16 && (skips & (skips - 1)) != 0) return;
	std::printf("Demon's Souls id list guard: id 0x%08llx (%llu of %llu in the list at 0x%llx) names a null or invalid "
	            "entry, array or object; skipped at eboot+0x%llx (%u so far)\n",
	            static_cast<unsigned long long>(id & 0xffffffffu), static_cast<unsigned long long>(index),
	            static_cast<unsigned long long>(count), static_cast<unsigned long long>(list),
	            static_cast<unsigned long long>(offset), skips);
	std::fflush(stdout);
}

// 01.005.000 eboot+0xa11480, a second job worker loop over the same kind of ids (11-25: runs 101 and 110, both after
// ~5 and ~26 minutes, the second right after "bound as null"): records of 0xec bytes (count in rdi) are searched for the
// nearest one (+8 a distance, +0x48 a flag); a record's first u32 indexes an array of 24-byte entries (rdx) whose id
// (+4) names a table (rsi) entry, its object array (+0xb0) and an object. A record whose index was garbage
// (0x46xxxxxx..) read past the array: "mov r8d, [rdx + r8 * 8 + 4]" at +0x24 faulted (0x8d6312174, 0xfbe893c84).
//   add rcx, 0xec; dec rdi; je <end>; vmovss xmm1, [rcx + 8]; vucomiss xmm0, xmm1; jbe <top>; cmp dword [rcx + 0x48], 0;
//   je <top>;                                                                       (+0x1d: the site)
//   mov r8d, [rcx]; lea r8, [r8 + r8 * 2]; mov r8d, [rdx + r8 * 8 + 4]; bextr r9d, r8d, r11d; movzx r8d, r8w;
//   mov r9, [rsi + r9 * 8]; mov r9, [r9 + 0xb0]; cmp qword [r9 + r8 * 8], 0; je <top>;
//   vmovaps xmm0, xmm1; mov r13, rcx; jmp <top>                                      (+0x44: back)
// Here a record whose index is 0x100000 or more (24 MiB of entries; both crashes had ~1e9), or whose table entry or
// object array lies outside the guest's user range, is passed over like one with no object (the loop's own je <top>).
constexpr const char* RecordLoop =
    "48 81 c1 ec 00 00 00 48 ff cf 74 ?? c5 fa 10 49 08 c5 f8 2e c1 76 e9 83 79 48 00 74 e3 44 8b 01 4f 8d 04 40 46 8b 44 "
    "c2 04 c4 42 20 f7 c8 45 0f b7 c0 4e 8b 0c ce 4d 8b 89 b0 00 00 00 4b 83 3c c1 00 74 bc c5 f8 28 c1 49 89 cd eb b3";
constexpr uint64_t RecordSiteOffset = 0x1d, RecordBackOffset = 0x44, RecordCaveOffset = 0x8000000 + 14 * PageSize;
constexpr uint32_t RecordSiteSize = RecordBackOffset - RecordSiteOffset; // 39 bytes, nothing jumps into them
constexpr uint32_t RecordIndexLimit = 0x100000;

std::atomic<uint32_t> g_record_skips {0};

void KYTY_SYSV_ABI RecordSkipped(uint64_t offset, uint64_t index, uint64_t record) {
	const auto skips = g_record_skips.fetch_add(1, std::memory_order_relaxed) + 1;
	if (skips > 16 && (skips & (skips - 1)) != 0) return;
	std::printf("Demon's Souls id list guard: record 0x%llx has index 0x%08llx or an invalid table entry; passed over at "
	            "eboot+0x%llx (%u so far)\n",
	            static_cast<unsigned long long>(record), static_cast<unsigned long long>(index & 0xffffffffu),
	            static_cast<unsigned long long>(offset), skips);
	std::fflush(stdout);
}

void InstallRecordGuard(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program->mapped_size > RecordCaveOffset || program->base_vaddr > UINT64_MAX - RecordCaveOffset - PageSize) return;
	const auto loop = GuestCode::FindUnique(*program, GuestCode::Pattern(RecordLoop));
	if (!loop) return;
	const uint64_t top = *loop, site = *loop + RecordSiteOffset, back = *loop + RecordBackOffset;
	const auto     requested = program->base_vaddr + RecordCaveOffset;
	const auto     allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
        requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_record_guard", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(PageSize, reinterpret_cast<void*>(allocated));
	constexpr uint8_t save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	Xbyak::Label      bad, limit;
	auto              check = [&](const Xbyak::Reg64& reg) {
        c.cmp(reg, MinPointer);
        c.jb(bad, Xbyak::CodeGenerator::T_NEAR);
        c.cmp(reg, ptr[rip + limit]);
        c.jae(bad, Xbyak::CodeGenerator::T_NEAR);
	};
	// The moved instructions, the index and both pointers checked (flags are dead at <top> and after the je).
	c.mov(r8d, dword[rcx]);
	c.cmp(r8d, RecordIndexLimit);
	c.jae(bad, Xbyak::CodeGenerator::T_NEAR);
	c.lea(r8, ptr[r8 + r8 * 2]);
	c.mov(r8d, dword[rdx + r8 * 8 + 4]);
	c.bextr(r9d, r8d, r11d);
	c.movzx(r8d, r8w);
	c.mov(r9, ptr[rsi + r9 * 8]);
	check(r9);
	c.mov(r9, ptr[r9 + 0xb0]);
	check(r9);
	c.cmp(qword[r9 + r8 * 8], 0);
	c.je(reinterpret_cast<const void*>(top));
	c.jmp(reinterpret_cast<const void*>(back), Xbyak::CodeGenerator::T_NEAR);
	// Passed over: printed (registers and FP state kept; below the red zone, on a realigned stack), then <top>.
	c.L(bad);
	c.lea(rsp, ptr[rsp - 128]);
	for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx})
		c.push(reg);
	c.mov(rdx, rcx);
	c.mov(esi, dword[rcx]);
	c.mov(rdi, site - program->base_vaddr);
	c.mov(rbx, rsp);
	c.and_(rsp, -16);
	c.sub(rsp, 512);
	c.db(save_fp, sizeof(save_fp));
	c.mov(rax, reinterpret_cast<uint64_t>(&RecordSkipped));
	c.call(rax);
	c.fxrstor64(ptr[rsp]);
	c.mov(rsp, rbx);
	for (const auto& reg: {rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
		c.pop(reg);
	c.lea(rsp, ptr[rsp + 128]);
	c.jmp(reinterpret_cast<const void*>(top), Xbyak::CodeGenerator::T_NEAR);
	c.align(8);
	c.L(limit);
	c.dq(MaxPointer);
	c.ready();
	if (Xbyak::GetError() || !Common::VirtualMemory::FlushInstructionCache(allocated, c.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	uint8_t code[RecordSiteSize];
	std::memset(code, 0xcc, sizeof(code));
	code[0]                 = 0xe9;
	const auto displacement = static_cast<int32_t>(allocated - (site + 5));
	std::memcpy(code + 1, &displacement, sizeof(displacement));
	std::memcpy(reinterpret_cast<void*>(site), code, sizeof(code));
	Common::VirtualMemory::FlushInstructionCache(site, sizeof(code));
	std::printf("Demon's Souls id list guard: installed (the record loop at eboot+0x%llx passes over a record with a bad "
	            "index or table entry)\n",
	            static_cast<unsigned long long>(site - program->base_vaddr));
#endif
}
} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin" ||
	    program->mapped_size > CaveOffset || program->base_vaddr > UINT64_MAX - CaveOffset - PageSize)
		return;
	const auto loop = GuestCode::FindUnique(*program, GuestCode::Pattern(Loop));
	if (!loop) return; // (not 01.005.000)
	const uint64_t site = *loop + SiteOffset, back = *loop + BackOffset, next = *loop + NextOffset;
	const auto     requested = program->base_vaddr + CaveOffset;
	const auto     allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
        requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_id_list_guard", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(PageSize, reinterpret_cast<void*>(allocated));
	constexpr uint8_t save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	Xbyak::Label      bad, limit;
	// A pointer is used only if it lies in the guest's user range (KYTY_ID_LIST_GUARD_RANGE=0: null checks only, as before).
	// The list is a GPU result written back each frame; on Pascal (11-25, runs 102 and 109) a list id named a live table
	// entry whose pointers were garbage, not null (rax 0xbc802180c0b95555, rcx 0x0004a3ac00000441): the null checks let it
	// through and the moved loads at cave +0x0d / +0x3d faulted on a non-canonical address.
	const bool range = []() {
		const char* v = std::getenv("KYTY_ID_LIST_GUARD_RANGE");
		return v == nullptr || v[0] != '0';
	}();
	auto check = [&](const Xbyak::Reg64& reg) {
		if (range) {
			c.cmp(reg, MinPointer); // (null included)
			c.jb(bad, Xbyak::CodeGenerator::T_NEAR);
			c.cmp(reg, ptr[rip + limit]);
			c.jae(bad, Xbyak::CodeGenerator::T_NEAR);
		} else {
			c.test(reg, reg);
			c.jz(bad, Xbyak::CodeGenerator::T_NEAR);
		}
	};
	// The moved instructions, each pointer checked before its use.
	c.mov(rax, ptr[rax + rdx * 8]); // the table's entry
	check(rax);
	c.mov(rcx, ptr[rax + 0xb0]); // its object array
	c.mov(rax, ptr[rax + 0xe0]); // its index array
	check(rcx);
	check(rax);
	c.mov(r13d, dword[rax + rsi * 4]);
	c.mov(ptr[rbp - 0x58], rax);
	c.mov(eax, r13d);
	c.and_(eax, 0xffffff);
	c.mov(rax, ptr[rcx + rax * 8]); // the object
	check(rax);
	c.mov(rcx, ptr[rax + 0x20]);
	check(rcx);
	c.jmp(reinterpret_cast<const void*>(back));
	// Skipped: printed (registers and FP state kept; below the red zone, on a realigned stack), then the next id, as the
	// loop's own increment has it (rax: this id's index, kept at rbp - 0x78; rbx, rdi, r14 unchanged since the loop's top).
	c.L(bad);
	c.lea(rsp, ptr[rsp - 128]);
	for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx})
		c.push(reg);
	c.mov(rdi, site - program->base_vaddr);
	c.mov(rsi, ptr[rbp - 0x70]);
	c.mov(rdx, ptr[rbp - 0x78]);
	c.mov(rcx, r14);
	c.mov(r8, rbx);
	c.mov(rbx, rsp);
	c.and_(rsp, -16);
	c.sub(rsp, 512);
	c.db(save_fp, sizeof(save_fp));
	c.mov(rax, reinterpret_cast<uint64_t>(&Skipped));
	c.call(rax);
	c.fxrstor64(ptr[rsp]);
	c.mov(rsp, rbx);
	for (const auto& reg: {rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
		c.pop(reg);
	c.lea(rsp, ptr[rsp + 128]);
	c.mov(rax, ptr[rbp - 0x78]);
	c.jmp(reinterpret_cast<const void*>(next));
	c.align(8);
	c.L(limit);
	c.dq(MaxPointer);
	c.ready();
	if (Xbyak::GetError() || !Common::VirtualMemory::FlushInstructionCache(allocated, c.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	// Installation occurs before module initializers or guest threads run; program memory is writable to the host.
	// (The bytes after the jump are never reached: nothing jumps into the moved instructions.)
	uint8_t code[SiteSize];
	std::memset(code, 0xcc, sizeof(code));
	code[0]                 = 0xe9;
	const auto displacement = static_cast<int32_t>(allocated - (site + 5));
	std::memcpy(code + 1, &displacement, sizeof(displacement));
	std::memcpy(reinterpret_cast<void*>(site), code, sizeof(code));
	Common::VirtualMemory::FlushInstructionCache(site, sizeof(code));
	std::printf("Demon's Souls id list guard: installed (the id list loop at eboot+0x%llx skips an id with a null or "
	            "invalid entry, array or object)\n",
	            static_cast<unsigned long long>(site - program->base_vaddr));
	if (range) InstallRecordGuard(program);
#endif
}

} // namespace Loader::DemonsSoulsIdListGuard
