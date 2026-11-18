#ifndef EMULATOR_SRC_LOCAL_BARYCENTRIC_CENSUS_H_
#define EMULATOR_SRC_LOCAL_BARYCENTRIC_CENSUS_H_

// How much of the rendering goes through the barycentric emulation's geometry shader (GeForce GTX 10 series and older,
// BuildEmulatedBarycentricShader): the graphics pipelines made with and without it, and the draws of each kind, every 30 s
// in the log ("Barycentric emulation (30 s): ..."). Only counts: the pipelines are made as before (the stores stay valid).
// KYTY_BARYCENTRIC_CENSUS=0 turns the report off.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace BarycentricCensus {

// PipelineCache::Pipeline::barycentric_gs: what a graphics pipeline was made with.
enum Kind : uint8_t {
	NotCounted = 0, // a compute pipeline, or a GPU with its own barycentrics
	Plain      = 1, // a graphics pipeline without the geometry shader
	Emulated   = 2, // a graphics pipeline with it
};

struct Counts {
	std::atomic<uint64_t> pipelines[3] {};
	std::atomic<uint64_t> draws[3] {};     // draw calls (an indirect multi-draw counts once)
	std::atomic<uint64_t> triangles[3] {}; // of the draws whose counts the CPU knows: index (or vertex) count x instances / 3
	std::atomic<uint64_t> indirect[3] {};  // draws whose counts the GPU reads (no triangles counted)
	std::atomic<int64_t>  next_report_ms {0};
	std::atomic<uint32_t> calls {0};
};

inline Counts& Get() {
	static Counts counts;
	return counts;
}

inline bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_BARYCENTRIC_CENSUS");
		return value == nullptr || std::string_view(value) != "0";
	}();
	return on;
}

inline int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

inline double Percent(uint64_t part, uint64_t whole) {
	return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

inline void Report() {
	auto&          c        = Get();
	const uint64_t pe       = c.pipelines[Emulated].load(std::memory_order_relaxed);
	const uint64_t pp       = c.pipelines[Plain].load(std::memory_order_relaxed);
	const uint64_t de       = c.draws[Emulated].exchange(0, std::memory_order_relaxed);
	const uint64_t dp       = c.draws[Plain].exchange(0, std::memory_order_relaxed);
	const uint64_t te       = c.triangles[Emulated].exchange(0, std::memory_order_relaxed);
	const uint64_t tp       = c.triangles[Plain].exchange(0, std::memory_order_relaxed);
	const uint64_t ie       = c.indirect[Emulated].exchange(0, std::memory_order_relaxed);
	const uint64_t ip       = c.indirect[Plain].exchange(0, std::memory_order_relaxed);
	std::printf("Barycentric emulation (30 s): draws %llu of %llu through the geometry shader (%.1f%%), triangles %.1fM of "
	            "%.1fM (%.1f%%; indirect draws, not counted: %llu with, %llu without); pipelines made so far: %llu with, "
	            "%llu without (%.1f%%)\n",
	            static_cast<unsigned long long>(de), static_cast<unsigned long long>(de + dp), Percent(de, de + dp),
	            static_cast<double>(te) / 1e6, static_cast<double>(te + tp) / 1e6, Percent(te, te + tp),
	            static_cast<unsigned long long>(ie), static_cast<unsigned long long>(ip),
	            static_cast<unsigned long long>(pe), static_cast<unsigned long long>(pp), Percent(pe, pe + pp));
	std::fflush(stdout);
}

inline void NotePipeline(Kind kind) {
	if (kind == NotCounted) return;
	Get().pipelines[kind].fetch_add(1, std::memory_order_relaxed);
}

// A draw with a pipeline of this kind: `vertices` its index (or vertex) count per instance, 0 when the GPU reads it.
inline void NoteDraw(uint8_t kind, uint64_t vertices, uint64_t instances) {
	if (kind == NotCounted || kind > Emulated || !On()) return;
	auto& c = Get();
	c.draws[kind].fetch_add(1, std::memory_order_relaxed);
	if (vertices == 0) {
		c.indirect[kind].fetch_add(1, std::memory_order_relaxed);
	} else {
		c.triangles[kind].fetch_add(vertices * (instances == 0 ? 1 : instances) / 3, std::memory_order_relaxed);
	}
	// The clock only every 1024 draws.
	if ((c.calls.fetch_add(1, std::memory_order_relaxed) & 1023u) != 0) return;
	const int64_t now  = NowMs();
	int64_t       next = c.next_report_ms.load(std::memory_order_relaxed);
	if (next == 0) {
		c.next_report_ms.compare_exchange_strong(next, now + 30000, std::memory_order_relaxed);
		return;
	}
	if (now < next || !c.next_report_ms.compare_exchange_strong(next, now + 30000, std::memory_order_relaxed)) return;
	Report();
}

} // namespace BarycentricCensus

#endif // EMULATOR_SRC_LOCAL_BARYCENTRIC_CENSUS_H_
