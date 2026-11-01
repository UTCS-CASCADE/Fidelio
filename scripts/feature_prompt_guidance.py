from __future__ import annotations


def feature_prompt_guidance(feature_name: str, direction: str) -> list[str]:
    if feature_name == "perf_stat.IPC":
        if direction == "decrease":
            return [
                "PROVEN RECIPE: IPC is primarily controlled by the balance between useful ALU work and memory stall cycles. The 3-switch + pointer-chase pattern naturally produces IPC in the 1.5-2.0 range depending on iteration count and chain configuration. More iterations of the switch loop = lower IPC (more time in cache-missing pointer chases relative to fast ALU work).",
                "INLINE ASM FOR PRECISE IPC CONTROL: Use inline asm pointer-chase loads (`asm volatile(\"movq (%0), %0\" : \"+r\"(pos))`) in a loop body. Each serialized load takes ~5.3 cycles on L1d hit, giving IPC ≈ 1/(5.3 × insn_per_load). By controlling how many non-load instructions surround each chase load, you can dial IPC precisely. A loop of N chase loads with M ALU insns between them gives IPC ≈ (N+M)/(N×5.3+M).",
                "ITERATION COUNT AS PRIMARY IPC KNOB: When the benchmark framework controls total runtime (e.g. Folly fibers, fbthrift), the inner loop iteration count determines what fraction of total instructions come from your tuned code vs framework overhead. Higher iteration count = tuned code dominates more = IPC moves toward your code's IPC. Empirical: 50M iters → IPC 1.96, 100M → 1.70, 130M → 1.62, 160M → 1.63.",
                "Do not assume a backend stall is always the right first move: if frontend and branch counters are still too low, prefer one compact frontend or control-flow perturbation first.",
                "Only after frontend and branch counters are already near target should you lower IPC with one backend-oriented lever such as a dependent load-use chain, pointer-chase, or other memory-latency dependency in the steady-state loop.",
                "Keep the benchmark character close to the source benchmark. A controlled IPC drop is more valuable than a dramatic rewrite that perturbs many counters at once.",
            ]
        return [
            "To increase IPC, preserve the current benchmark's throughput-carrying structure and remove only the smallest avoidable stall source or dependency-depth issue.",
            "Do not change loop trip counts, benchmark scale, or the main steady-state work volume unless required.",
            "Do not add new critical-path work that can materially reduce IPC somewhere else in the timed loop.",
            "If a change risks lowering IPC, reduce its frequency or move it off the hottest path rather than rewriting the benchmark mechanism.",
        ]

    if feature_name == "perf_stat.L1-dcache-load-misses_MPKI":
        if direction == "increase":
            return [
                "PROVEN RECIPE: Use a linked-list pointer-chase through a 1 MB hot chain (16384 nodes × 64 bytes each). Each node contains a `next` pointer plus ~48 bytes of padding to fill a cache line. Initialize with Fisher-Yates shuffle to create a single Hamiltonian cycle. This exceeds the 48 KB L1d cache but fits in L2, generating reliable L1d misses without LLC misses.",
                "CRITICAL: Do NOT use hash-based or LCG-based random indexing into arrays — the hardware L2 stream prefetcher on modern Intel cores will learn the access pattern and prefetch ahead, resulting in near-zero L1-d misses even for large arrays. Only pointer-chasing (where the address of each load depends on the VALUE of the previous load) defeats the prefetcher.",
                "WORKING SET vs CACHE SIZE MODEL: L1d miss rate is controlled by the ratio of working set to L1d capacity. A chain that fits in L1d (≤48KB) → 0 MPKI. A chain at 1MB (21× L1d) → most accesses miss L1d but hit L2. A 'medium' chain at ~128KB (2.7× L1d, fits L2) gives moderate L1d misses. Use multiple independent medium-chain walkers to multiply the miss rate additively without growing working set beyond L2.",
                "INDEPENDENT HOT-CHAIN WALKERS: To boost L1d MPKI beyond what a single serialized chain provides, add 2-3 independent pointer-chase walkers that each maintain their own position in the same (or separate) hot chain. Each walker advances one step per loop iteration independently. This multiplies L1d miss rate without serializing the main loop. CAUTION: walkers add compact loop instructions that dilute L1i MPKI and branch-miss MPKI — use only when L1d is the priority and L1i/branch accuracy is acceptable.",
                "HOT CHAIN SIZING: The chain must exceed L1d (typically 48 KB) but should fit in L2 (typically 2 MB) to avoid LLC misses. 1 MB (16384 × 64B nodes) is the proven sweet spot. Use `struct ChainNode { ChainNode* next; char pad[48]; };` with 8-byte alignment on `next`.",
                "COLD-ACCESS SCHEDULING: To simultaneously control LLC misses, pair the hot chain with a separate cold chain (128 MB, exceeding LLC). Access the cold chain every N loop iterations using branchless arithmetic: `coldMask = -uint64_t(j % N == 0); coldPos = *(ChainNode*)((uintptr_t)coldPos->next & coldMask | (uintptr_t)coldPos & ~coldMask);`. This avoids injecting branches that pollute the TAGE predictor.",
                "MPKI DILUTION WARNING: MPKI = misses / (total_instructions / 1000). If the benchmark framework generates many instructions per invocation (e.g. Folly fibers at -O0), your inner loop must execute enough iterations (100K+) to dominate total instruction count. Aim for the tuned code to represent >70% of all instructions.",
            ]
        return [
            "To reduce L1-dcache miss rate, make the hot data working set smaller and more reusable so frequently accessed data stays resident in L1D.",
            "Prefer tighter locality, more repeated access to a compact hot set, and fewer cache-line touches per unit of work.",
        ]

    if feature_name == "perf_stat.L1-dcache-loads_LPKI":
        if direction == "increase":
            return [
                "PROVEN RECIPE — INLINE ASM LOADS: Use `asm volatile(\"movq (%0), %0\" : \"+r\"(pos))` in a loop. Each inline asm load is exactly 1 instruction producing exactly 1 L1-dcache-load. This gives a precise 1:1 ratio of loads per instruction, maximizing LPKI. N chase loads in a loop body = N loads per (N + overhead) instructions.",
                "LOOPED (NOT UNROLLED) ASM: Place N inline asm chase loads inside a for-loop rather than unrolling them sequentially. A 170-load loop body compiles to ~1.5KB of hot code; the same 170 loads unrolled produce ~12.5KB, which destroys icache accuracy. The compiler's loop structure reuses the same instruction bytes each iteration.",
                "INSTRUCTION DILUTION TRAP: LPKI = loads / (total_insn / 1000). Adding ANY non-load instructions (ALU, stores, branches) dilutes LPKI equally. If LPKI must increase, add more loads per iteration rather than reducing other instructions — removing non-loads risks destabilizing other counters.",
                "Avoid changes that mostly add control-flow complexity without increasing steady-state read traffic.",
            ]
        return [
            "To reduce L1-dcache-loads_LPKI, reuse values in registers or a compact cache-resident hot set instead of reloading them repeatedly.",
            "Prefer hoisting repeated loads, reusing previously fetched data across iterations, and simplifying the hot path so it needs fewer memory reads per unit of work.",
            "A common optimization pattern is to reduce loads by reusing data in registers or cache while also improving locality enough to lower miss MPKI.",
        ]

    if feature_name == "perf_stat.L1-dcache-stores_SPKI":
        if direction == "increase":
            return [
                "PROVEN RECIPE — INLINE ASM STORES: Use `asm volatile(\"movq %0, (%1)\" : : \"r\"(val), \"r\"(addr) : \"memory\")` in a loop. Each inline asm store is exactly 1 instruction producing exactly 1 L1-dcache-store. This gives precise control over SPKI. Pair with a small write buffer (e.g. 64-byte aligned scratch area) to keep stores in L1d.",
                "INSTRUCTION DILUTION TRAP: SPKI = stores / (total_insn / 1000). Adding non-store instructions dilutes SPKI. If SPKI must increase, add more store instructions per iteration rather than removing other work. Be aware that adding stores also adds instructions, so net SPKI gain is (new_stores - store_fraction × new_insns) / total_insns × 1000.",
                "Avoid one-time setup writes; the steady-state loop must repeatedly generate the additional store traffic.",
            ]
        return [
            "INTRACTABILITY WARNING: Stores SPKI is often dominated by framework overhead (e.g. Folly fibers promise/queue management, fbthrift protocol serialization) that generates ~3x more stores than user code can offset. If the target Stores SPKI is much lower than baseline and the benchmark uses a heavy framework, accept that Stores SPKI may be intractable and focus on other features.",
            "To reduce L1-dcache-stores_SPKI, batch or accumulate work in registers before storing results back to memory.",
            "Prefer delaying writes, combining multiple updates before a store, and keeping temporaries out of memory when possible.",
        ]

    if feature_name in {
        "perf_stat.L1-icache-load-misses_MPKI",
        "perf_stat.L2-icache-load-misses_MPKI",
    }:
        if direction == "increase":
            return [
                "PROVEN RECIPE: Create 3 __attribute__((noinline)) functions, each containing a 256-case switch statement with N ALU operations per case (add/xor/mul on a local accumulator). Rotate among them with `j % 3` inside the hot loop. The large switch bodies create a massive instruction footprint that cannot fit in L1I, and cycling among 3 functions forces continuous icache thrashing.",
                "The switch index should be `payload & 0xFF` where payload comes from a pointer-chase load, making the indirect branch target unpredictable. This simultaneously generates L1i misses (code footprint) and BadSpec/branch misses (indirect misprediction).",
                "EMPIRICAL SCALING: L1i MPKI scales approximately linearly with ALU ops per switch case on Intel Xeon: 8 ops/case → ~11 MPKI, 10 → ~10 MPKI, 12 → ~12 MPKI, 14 → ~15 MPKI. Use this to dial L1i MPKI to the target value.",
                "CRITICAL — LOOP, DO NOT UNROLL: When using inline asm loads/stores in the hot loop, keep them inside a for-loop rather than unrolling N copies sequentially. A 170-load loop body compiles to ~1.5KB; unrolled it becomes ~12.5KB. The extra code footprint thrashes L1i and can raise icache MPKI by 3-5× beyond target. The compiler reuses the same instruction bytes each loop iteration, preserving icache residency.",
                "Each switch case must contain genuinely different instruction sequences (different constants, different operation mixes) so the compiler cannot merge or deduplicate them. Use unique literal constants per case.",
                "Avoid adding mostly cold setup logic, rare slow paths, or one-time initialization. Those increase binary size without materially increasing instruction-cache misses.",
                "WARNING — INSTRUCTION DILUTION: Adding hot-chain walkers or extra inline asm loads/stores to boost L1d/LPKI/SPKI dilutes L1i MPKI because the compact loop instructions increase total instruction count without proportionally increasing icache misses. If L1i accuracy is important, prefer increasing ops-per-case over adding walkers or extra asm instructions.",
            ]
        return [
            "To reduce instruction-cache misses, make the hot path smaller and more repetitive rather than only removing cold code.",
            "Favor fewer hot helper bodies, more linear control flow, and repeated execution of the same compact code footprint.",
        ]

    if feature_name == "perf_stat.iTLB-load-misses_MPKI":
        if direction == "increase":
            return [
                "PROVEN RECIPE: The 3-noinline-function × 256-case-switch pattern naturally creates iTLB pressure because each switch body spans many 4 KB instruction pages and cycling among 3 such functions forces the iTLB to map a large code region. More ops-per-case = more code pages per switch = more iTLB misses.",
                "STLB CAPACITY MODEL: iTLB misses are governed by the L2 STLB (2048 entries on Intel Xeon). If the hot code + data together touch fewer than 2048 pages, most iTLB misses are absorbed by the STLB and iTLB-MPKI stays low. The formula: iTLB_misses_per_iter ≈ 1.73 × N_funcs × (total_pages - 2048) / total_pages. To increase iTLB misses, you need total touched pages > 2048.",
                "PAD-BASED CODE PAGE CONTROL: Each noinline function's size (and thus the number of 4KB pages it spans) is controlled by padding bytes between functions. With N functions of PAD_SIZE bytes each: total_code_pages ≈ N × (code_bytes + PAD_SIZE) / 4096. On Intel Xeon Gold 5512U, PAD_SIZE=380 with 8192 functions produces ~8640 code pages, exceeding the 2048 STLB capacity and generating reliable iTLB misses.",
                "1GB HUGEPAGES FOR STLB ISOLATION: Data allocations (cold chains, hot chains) on regular 4KB pages consume STLB entries and compete with instruction pages. Allocate large data buffers with 1GB hugepages (`mmap` with `MAP_HUGETLB | MAP_HUGE_1GB`) so each buffer uses only 1 STLB entry regardless of size. This reserves almost all 2048 STLB entries for instruction pages, maximizing iTLB pressure from code.",
                "Prefer repeatedly cycling through several noinline hot helpers placed in different parts of the binary so the front-end touches more instruction pages during steady-state execution.",
                "Avoid one-time setup code or cold branches; they rarely move iTLB misses in a stable way.",
            ]
        return [
            "To reduce iTLB misses, concentrate the hot path into fewer instruction pages and reuse a smaller set of hot helper bodies.",
        ]

    if feature_name == "perf_stat.LLC-load-misses_MPKI":
        if direction == "increase":
            return [
                "PROVEN RECIPE: Use a pointer-chase cold chain of 128 MB (2 million nodes × 64 bytes each) initialized with Fisher-Yates shuffle. This exceeds the LLC (typically 30-52 MB shared) so every access goes to DRAM. Access the cold chain every N loop iterations — N controls LLC MPKI rate. Smaller N = more LLC misses.",
                "CRITICAL: Use pointer-chasing through a linked-list chain larger than the LLC. Hash-based or LCG-based random indexing is defeated by hardware prefetchers and will NOT generate LLC misses. The chain must be a single Hamiltonian cycle (Fisher-Yates shuffle) so the access pattern is truly unpredictable to hardware.",
                "PARALLEL COLD LOADS > SERIAL: When issuing multiple cold loads per iteration, use INDEPENDENT parallel loads (each from a separate chain position) rather than a serial chain of dependent loads. Serial cold loads stall for N×200 cycles (each waiting for DRAM), during which the L1i prefetcher runs freely and icache accuracy tanks. Parallel independent loads exploit memory-level parallelism (MLP): the CPU issues all loads simultaneously, stalling for only ~200 cycles total regardless of count. This minimizes stall-window damage to other counters.",
                "1GB HUGEPAGES FOR COLD ALLOCATION: Allocate the cold chain buffer using `mmap` with `MAP_HUGETLB | MAP_HUGE_1GB` (flags = `MAP_HUGETLB | (30 << MAP_HUGE_SHIFT)`). A 128MB cold chain on 4KB pages consumes 32768 STLB entries, evicting instruction page mappings and destroying iTLB accuracy. With 1GB hugepages, the same 128MB uses only 1 STLB entry, leaving all 2048 STLB entries free for instruction pages.",
                "FEATURE COUPLING — TWO-TIER CHAIN STRATEGY: LLC misses and L1-d misses have conflicting data-size requirements. Use two separate chains: (1) a 1 MB hot chain (16384 × 64B nodes, fits L2 but not L1d) chased every iteration for L1d misses, and (2) a 128 MB cold chain (2M × 64B nodes, exceeds LLC) chased every N iterations for LLC misses. The cold-chain access frequency N is the primary knob for LLC MPKI.",
                "BRANCHLESS COLD-ACCESS SCHEDULING: Do NOT use `if (j % N == 0)` to schedule cold-chain accesses — this injects a branch into the TAGE global history register and unpredictably changes main-branch misprediction rates. Instead use branchless arithmetic: `coldMask = -uint64_t(j % N == 0); coldPos = *(ChainNode*)((uintptr_t)coldPos->next & coldMask | (uintptr_t)coldPos & ~coldMask);`",
                "COLD CHAIN SIZING: The chain must clearly exceed the total LLC size. On Intel Xeon with 52.5 MB shared L3, use 128 MB (2M nodes × 64B). On smaller L3 (30 MB), 64 MB may suffice. Check `/sys/devices/system/cpu/cpu0/cache/index3/size` for the actual L3 size.",
                "MPKI DILUTION WARNING: MPKI = misses / (total_instructions / 1000). If the benchmark framework generates many instructions per invocation (e.g. Folly fibers at -O0), your inner loop must execute enough iterations (100K+) to dominate total instruction count. Aim for the tuned code to represent >70% of all instructions.",
            ]
        return [
            "To reduce LLC-load misses, repeatedly use a smaller hot set of data so working data fits better inside the cache hierarchy.",
            "Prefer stronger temporal locality, more reuse of recently loaded data, and less scattered access patterns in the steady-state hot path.",
        ]

    if feature_name == "perf_stat.branch-misses_MPKI":
        if direction == "increase":
            return [
                "PROVEN RECIPE: Use 3 noinline functions each containing a 256-case switch. The switch index is `payload & 0xFF` where payload comes from a pointer-chase load (high-latency, unpredictable value). Rotate among the 3 functions with `j % 3`. The indirect branch for each switch is unpredictable to TAGE because the target depends on a cache-missing load value. This simultaneously generates branch misses AND L1i pressure.",
                "COMPACT ALTERNATIVE — COLLATZ BRANCHES: For branch misses without large code footprint, use Collatz iteration: `if (x & 1) { x = 3*x + 1; } else { x >>= 1; }`. This generates ~35% misprediction rate in a tiny code footprint (2 instructions per branch). Insert N_COLLATZ iterations per loop iteration to control branch-miss MPKI. A Collatz sequence seeded from a pointer-chase load value is unpredictable to TAGE. Proven: 75 Collatz iterations per loop iteration produces reliable branch misses with minimal icache impact.",
                "CRITICAL: Make branch conditions depend on the VALUE of a cache-missing load (e.g. a pointer-chasing load), not on register-computed state. The CPU resolves register-based conditions speculatively before they can cause mispredictions, but a condition gated on a high-latency load forces the predictor to guess and often miss.",
                "Keep the change structurally close to the source benchmark: prefer one cheap unpredictable conditional in the steady-state hot path rather than a large rewrite of both sides of the benchmark.",
                "Avoid only bloating code size; branch-misses primarily move when the predictor sees less predictable decisions, not merely more instructions.",
                "TAGE PREDICTOR WARNING: The TAGE branch predictor has ~32K-64K history table entries and can learn deterministic branch cycles shorter than ~32K steps. A pointer-chasing cycle of N nodes repeats every N branches — if N < 32K, TAGE will learn the pattern. Use a pre-filled random decision array with 1M+ entries indexed by a monotonic counter (not the cycle position). The array values must come from a high-quality PRNG (xorshift128+ or better), not LCG.",
                "COLD-ACCESS BRANCH COUPLING WARNING: Control-flow branches for cold-access scheduling (e.g. `if (j % N == 0) coldAccess()`) inject into the TAGE global history register and can unpredictably affect main-branch misprediction rates. Use branchless arithmetic selection (e.g. conditional move or `mask = -(j % N == 0); coldPos = cold[coldPos & mask]`) to decouple cold-access scheduling from branch prediction state.",
                "MPKI DILUTION WARNING: MPKI = misses / (total_instructions / 1000). If the benchmark framework generates many instructions per invocation (e.g. Folly fibers at -O0), your inner loop must execute enough iterations (100K+) to dominate total instruction count. Aim for the tuned code to represent >70% of all instructions.",
            ]
        return [
            "To reduce branch misses, make the common path obvious, repetitive, and easy for the branch predictor to learn.",
            "Prefer stable hot control flow with predictable taken/not-taken behavior and fewer data-dependent surprises on the steady-state path.",
        ]

    if feature_name == "perf_stat.branch-load-misses_MPKI":
        if direction == "increase":
            return [
                "To increase branch-load misses, stress the branch target structures and front-end by using a larger hot set of branch sites and jump targets.",
                "Prefer hot execution that bounces among many call targets, dispatch targets, or branch sites rather than repeatedly taking the same small set of branches.",
                "Increase the hot branch footprint moderately and locally; preserve the original benchmark shape and avoid turning the hot path into a large dispatcher tree unless that extra footprint is clearly needed.",
                "Prefer one or two additional hot branch sites or targets that are exercised repeatedly, ideally driven by lightweight memory-backed state, instead of many helper variants or deeply branchy control flow.",
                "Avoid aggressive helper fan-out that simultaneously explodes branch misses, icache misses, and IPC error; the goal is a controlled increase, not the largest possible branch footprint.",
                "Inlining or expanding code can help only if it materially increases the hot branch footprint; a huge cold body or one-time setup path is usually not enough.",
                "Interpreter-like dispatch, many distinct indirect or conditional branch targets, and movement across separated hot regions are more effective than one compact loop.",
            ]
        return [
            "To reduce branch-load misses, keep the hot branch footprint compact and repeatedly use a smaller set of branch sites and jump targets.",
            "Prefer simpler control flow, fewer hot dispatch targets, and less bouncing across distant hot regions of code.",
        ]

    return []
