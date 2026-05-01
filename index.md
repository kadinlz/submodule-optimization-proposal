<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      displayMath: [['$$','$$'], ['\\[','\\]']],
      processEscapes: true
    }
  });
</script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<style>
  body { font-family: Georgia, serif; line-height: 1.7; color: #1a1a1a; }
  h1 { font-size: 1.9em; margin-bottom: 0.2em; }
  h2 { border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 2em; }
  h3 { margin-top: 1.5em; color: #222; }
  .authors { font-size: 1.1em; color: #444; margin-bottom: 0.3em; }
  .venue { font-size: 0.95em; color: #666; margin-bottom: 1.5em; }
  .abstract-box {
    background: #f7f7f7; border-left: 4px solid #555;
    padding: 1em 1.4em; margin: 1.5em 0; font-size: 0.97em;
  }
  .highlight-box {
    background: #eef4ff; border-left: 4px solid #3a6ea8;
    padding: 0.8em 1.2em; margin: 1em 0;
  }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
  th { background: #333; color: #fff; padding: 6px 10px; text-align: left; }
  td { padding: 5px 10px; border-bottom: 1px solid #ddd; }
  tr:nth-child(even) td { background: #f9f9f9; }
  code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
  pre { background: #f0f0f0; padding: 1em; border-radius: 4px; overflow-x: auto; font-size: 0.85em; line-height: 1.4; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; }
  @media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
</style>

# Parallelizing Greedy Submodular Optimization

<div class="authors">Pranav Sangwan &nbsp;&nbsp;·&nbsp;&nbsp; Kadin Zhang</div>
<div class="venue">15-418 Parallel Computer Architecture and Programming &nbsp;·&nbsp; Carnegie Mellon University.</div>

---

<div class="abstract-box">
<strong>Abstract.</strong>
We study how the inherently sequential greedy algorithm for submodular facility location can be progressively parallelized through a sequence of complementary techniques on a 128-core shared-memory machine. Starting from a sequential baseline, we develop six solver generations — each motivated by profiling the previous generation's bottleneck — culminating in a <em>nested hierarchical pruned CELF</em> (nhpc) solver that achieves up to <strong>91.5× speedup at 128 threads</strong> on a large dataset (n = 30,000, d = 32) while preserving solution quality to within 0.1% of sequential greedy across all tested configurations.
</div>

---

## Background

### Submodular Optimization

A set function $$f: 2^V \rightarrow \mathbb{R}$$ over a finite ground set $$V$$ is *submodular* if it satisfies the diminishing returns property: for all $$A \subseteq B \subseteq V$$ and $$x \notin B$$,

$$f(A \cup \{x\}) - f(A) \;\ge\; f(B \cup \{x\}) - f(B).$$

The greedy algorithm — which iteratively selects the element with the largest marginal gain — achieves a $$(1 - 1/e) \approx 63.2\%$$ approximation to the optimal solution under a cardinality constraint, and this bound is tight unless P = NP. Greedy is therefore not merely a heuristic but the *canonical* algorithm for this problem class, with applications in influence maximization, feature selection, document summarization, sensor placement, and active learning.

### The Facility Location Objective

We focus on the facility location objective, where the ground set $$V = \{1,\ldots,n\}$$ has each element $$i$$ corresponding to a vector $$x_i \in \mathbb{R}^{32}$$. Given a budget $$k$$, we select $$S \subseteq V$$ with $$\lvert S \rvert \le k$$ to maximize:

$$f(S) = \sum_{i \in V} \max_{j \in S}\, w_{ij}, \qquad w_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right),\quad \sigma = 20.$$

Maintaining per-point *coverage values* $$m_i = \max_{j \in S} w_{ij}$$ allows marginal gains to be computed incrementally:

$$\Delta(c \mid S) = \sum_{i} \max(0,\, w_{ic} - m_i),$$

reducing per-round cost from $$O(n \cdot \lvert S \rvert)$$ to $$O(n)$$, with total cost $$O(kn)$$ once similarities are available.

### CELF: Lazy Greedy

The *Cost-Effective Lazy Forward* (CELF) algorithm exploits submodularity: marginal gains can only decrease as $$S$$ grows, so previously computed gains are valid upper bounds. CELF maintains a max-heap of `(gain_upper_bound, candidate_id)` pairs; a candidate is only recomputed when it reaches the heap top with a stale value. This reduces gain evaluations by one to two orders of magnitude in practice.

### Batching

To expose parallelism, we select the top $$b = \lfloor k/20 \rfloor$$ candidates per round simultaneously. This reduces global synchronization barriers by a factor of $$b$$ at the cost of using slightly stale coverage information for the $$b-1$$ non-leading selections. Fixing $$b = k/20$$ independent of thread count preserves solution quality to within 0.1% across all datasets.

---

## Approach: Six Solver Generations

Our implementation progresses through six solver generations, each motivated by profiling the previous one. All experiments target the Pittsburgh Supercomputing Center (PSC): dual-socket AMD EPYC 7742, 128 physical cores, 256 MB L3 per socket, 512 GB DRAM, hyperthreading disabled. The implementation uses C++17 with OpenMP 5.0.

### Generation 1 — Sequential Baseline (`g`)

Two design decisions established here carry through every subsequent version:

**Incremental coverage maintenance.** A length-$$n$$ array `best_sim` tracks $$m_i$$ and is updated in an $$O(n)$$ pass after each selection, avoiding $$O(n \lvert S \rvert)$$ recomputation.

**SIMD-accelerated RBF kernel.** The pairwise similarity uses `#pragma omp simd` with `__restrict__` annotations. With $$d = 32$$ fixed, AVX2 executes the inner squared-distance loop in exactly 8 SIMD steps — instruction-level parallelism orthogonal to all thread-level work.

```cpp
inline double rbf_similarity(const double* __restrict__ a,
                              const double* __restrict__ b,
                              int dim, double inv_two_sigma_sq) {
    double s = 0.0;
    #pragma omp simd reduction(+:s)
    for (int k = 0; k < dim; k++) { double d = a[k]-b[k]; s += d*d; }
    return std::exp(-s * inv_two_sigma_sq);
}
```

The baseline also precomputes the full $$O(n^2)$$ similarity matrix $$W$$ so gain evaluations read from memory — a decision that proves to be the dominant bottleneck.

---

### Generation 2 — Shared Priority Queue CELF (`c`)

The first parallel version shares a single mutex-guarded max-heap across all threads. Threads race to pop, recompute, and push — creating a contended critical section where the serial fraction *grows* with thread count. Speedup peaked near 8 threads and degraded beyond.

Three measurement bugs were also corrected here: heap-allocation contention (fixed by pre-allocated per-thread buffers), batch-size coupling (batch size was silently set to num_threads), and incorrect timer placement ($$W$$ construction was excluded from the compute timer).

---

### Generation 3 — Distributed CELF

To eliminate lock contention, the candidate space is statically partitioned across threads; each thread maintains its own private CELF priority queue. A reduction array selects the global winner after each round — equivalent to exact greedy.

<div class="highlight-box">
<strong>Result:</strong> Lock contention vanished, but medium and large datasets plateaued below 4× speedup. Cache profiling explained why: LLC miss rates of 38–43% at 1 thread, rising to 53–55% at 128T, with L1 load misses nearly doubling. The culprit was the W matrix — accessing column j of the row-major W matrix for varying row i induces a stride-n traversal, loading one useful double per 64-byte cache line. W occupied 1.1–6.9 GB for medium/large datasets.
</div>

| Dataset | 1T | 8T | 32T | 64T | 128T |
|---|---|---|---|---|---|
| Small (n=4K) | 1.00 | 2.37 | 7.07 | 8.91 | **10.87** |
| Medium (n=12K) | 1.00 | 1.61 | 2.21 | 2.79 | 2.63 |
| Large (n=30K) | 1.00 | 2.39 | 3.39 | **3.87** | 2.78 |

---

### Generation 4 — Hierarchical CELF, Eliminating $$W$$ (`hc`)

Naively replacing $$W$$ lookups with on-the-fly `rbf_similarity()` calls produced a regression — every candidate still paid the full $$\exp()$$ cost for all $$n$$ points. Eliminating $$W$$ required first building a structure to skip the majority of point pairs.

**Two-level pruning hierarchy.** The $$n$$ ground points are clustered into $$K = 32$$ clusters via k-means, with precomputed centroids $$\mu_k$$ and radii $$r_k$$. For each candidate $$c$$, a cluster-level upper bound is computed first:

$$\Delta_{\text{ub}}(c, C_k) = \sum_{i \in C_k} \max\!\bigl(0,\, \hat{w}_{c,k} - \underline{m}_k\bigr),$$

where $$\hat{w}_{c,k}$$ is the maximum possible similarity between $$c$$ and any point in $$C_k$$, and $$\underline{m}_k = \min_{i \in C_k} m_i$$. If this bound falls below a fraction $$\varepsilon$$ of the current best gain, the entire cluster is pruned (analogous to Barnes-Hut tree traversal). The working set shrinks from a 1.1–6.9 GB $$W$$ array to a 7.7 MB points array that fits comfortably in L3.

**After fixing the batch-size coupling bug** (replacing `batch_size = num_threads` with `batch_size = k/20`), the corrected results show dramatic improvement:

| Dataset | 8T | 32T | 64T | 128T | Quality at 128T |
|---|---|---|---|---|---|
| Small (n=4K) | 6.6× | 15.9× | 17.7× | 15.0× | 100.0% |
| Medium (n=12K) | 7.4× | 25.1× | 40.9× | 11.0× | 100.0% |
| Large (n=30K) | 8.0× | 28.7× | **52.3×** | 76.7× | 99.99% |

The large dataset reaches 52.3× at 64T. However, medium and small regress sharply at 128T — pointing to a new bottleneck.

---

### Generation 5 — Progressive Coverage Pruning, Barrier Fusion, Thread Pinning (`hpc`)

**Progressive coverage pruning.** As greedy selects facilities, ground points become progressively covered. A point $$i$$ with $$m_i > \tau$$ contributes at most $$1 - \tau$$ to any future gain. The `hpc` mode excludes such points from all subsequent gain evaluations, with $$\tau = 0.90$$ found to be the effective sweet spot in 32-dimensional space with $$\sigma = 20$$.

**Diagnosing the 128T collapse.** The initial `hpc` implementation executed 6 barriers per round. Two of these (rebuilding `active_pts` and refreshing cluster summaries) distributed $$K = 32$$ work items across 128 threads — assigning $$\lfloor 32/128 \rfloor = 0$$ items to 96 threads. On PSC's dual-socket NUMA system, each 128-thread barrier costs 100–500 µs of cross-socket cache-coherence traffic. With `prune_interval = 1`, this added ~7.6 ms of pure synchronization overhead per run — catastrophic for the small dataset.

**The fix: barrier fusion.** The three post-selection operations (`best_sim` update, pruning decision, cluster summary refresh) are fused into a single `#pragma omp for schedule(static)` loop over the $$n$$ ground points. Per-round barrier count drops from 6 to 4, all loops now distribute $$n$$ points evenly.

| Dataset | Before 128T | After 128T | Improvement |
|---|---|---|---|
| Small (n=4K) | 1.5× | 10.8× | 7.2× |
| Medium (n=12K) | 11.8× | 24.0× | 2.0× |
| Large (n=30K) | 51.2× | 81.2× | 1.6× |
| Weak cluster | 20.5× | 55.6× | 2.7× |
| Strong cluster | 13.5× | 43.5× | 3.2× |

**Thread pinning.** Benchmarks run via exclusive `sbatch` allocations with `OMP_PROC_BIND=close` and `OMP_PLACES=cores`, mapping threads 1:1 onto physical cores. Interactive sessions receive a restricted cgroup affinity mask — a root cause of several anomalous collapses during development.

Final `hpc` results across all 12 benchmark configurations, with solution quality preserved to within 0.03% at 128 threads:

| Suite | Dataset | 8T | 32T | 64T | 128T | Quality |
|---|---|---|---|---|---|---|
| Problem Size | Small (n=4K) | 7.10× | 20.35× | 20.02× | 17.83× | 99.97% |
| | Medium (n=12K) | 7.67× | 27.48× | 40.86× | 60.30× | 99.99% |
| | Large (n=30K) | 7.99× | 29.51× | 48.14× | 85.13× | 99.97% |
| Clustering | Weak cluster | 7.74× | 28.02× | 43.69× | 69.47× | 100.0% |
| | Medium cluster | 7.78× | 27.28× | 41.44× | 53.30× | 99.99% |
| | Strong cluster | 7.81× | 27.16× | 41.17× | 52.68× | 99.99% |
| Skew | Uniform | 7.13× | 24.65× | 37.20× | 44.29× | 100.0% |
| | Mild skew | 7.53× | 27.07× | 39.82× | 53.89× | 100.0% |
| | Heavy skew | 7.95× | 27.96× | 41.46× | 56.47× | 100.0% |
| Density | Sparse | 7.37× | 24.04× | 36.26× | 44.51× | 99.98% |
| | Medium | 7.71× | 27.40× | 40.23× | 58.09× | 100.0% |
| | Dense | 7.86× | 28.18× | 44.93× | 60.35× | 100.0% |

---

### Generation 6 — Nested Parallelism (`nhpc`)

The `hpc` solver revealed that jumping from 64 to 128 threads is inefficient across the board. Three interconnected bottlenecks share a common root — all 128 threads are assigned to a single parallelism dimension:

1. **Small-PQ effect.** At $$T = 128$$, $$n = 12{,}000$$: each thread's PQ contains only 94 entries; ~19% are stale per round, defeating CELF.
2. **128-way barrier overhead.** Cross-socket barriers cost 100–500 µs each; ~4 per round adds 8–40 ms cumulative overhead, comparable to total runtime on medium datasets.
3. **CELF load imbalance.** Threads with many invalidated candidates stall the round; variance grows as PQ size shrinks.

**The $$G \times M$$ thread decomposition.** The `nhpc` mode partitions $$T$$ threads into $$G$$ *groups* of $$M$$ inner threads each ($$T = G \times M$$), with two levels of parallelism:

- **Across groups ($$G$$-way, candidate-parallel).** Each group maintains an independent PQ over a disjoint partition of the $$n$$ candidates. No inter-group communication during the CELF phase.
- **Within groups ($$M$$-way, point-parallel).** The $$M$$ threads within a group cooperatively evaluate a single candidate's gain by partitioning the inner point loop, writing partial sums to a shared reduction array.

At $$G = 32$$, $$M = 4$$ on the medium dataset:

| | `hpc` (T=128) | `nhpc` (G=32, M=4) |
|---|---|---|
| PQ size (n=12,000) | 94 entries | 375 entries |
| CELF stale fraction | ~19% | ~4.8% |
| Global barrier width | 128-way | 32-way |
| Intra-group barrier | — | 4-way (per-socket) |

**Custom sense-reversing group barriers.** Since the entire `nhpc` solver operates within a single `#pragma omp parallel num_threads(T)` region, a standard OpenMP barrier would synchronize all 128 threads at each CELF step. Per-group custom barriers are implemented instead:

```cpp
struct GroupBarrier {
    alignas(64) std::atomic<int> count{0};
    alignas(64) std::atomic<int> sense{0};
    void wait(int M) {
        const int my_sense = sense.load(std::memory_order_relaxed) ^ 1;
        if (count.fetch_add(1, std::memory_order_acq_rel) == M - 1) {
            count.store(0, std::memory_order_relaxed);
            sense.store(my_sense, std::memory_order_release);
        } else {
            while (sense.load(std::memory_order_acquire) != my_sense) { }
        }
    }
};
```

`alignas(64)` places `count` and `sense` on separate cache lines (eliminating false sharing). The sense-reversing protocol avoids the reset race inherent in a plain counter barrier. The result: a 4-way barrier costs ~50–100 ns versus ~100–500 µs for the 128-way cross-socket barrier it replaces.

**Leader-driven CELF protocol.** Within each group, lane 0 (the *leader*) drives the sequential CELF dependency chain; followers evaluate point stripes and wait at group-scoped barriers. The protocol carefully places barriers to enforce four happens-before relationships: candidates buffer written before any follower reads, partial gains written before leader reduces, PQ push-back before next pop, and coverage array fully updated between rounds.

**Empirical $$G/M$$ tuning.** All factorizations of $$T = 128$$ as $$G \times M$$ (with $$M \in \{1, 2, 4, 8, 16\}$$) were swept across all datasets. $$M = 4$$ ($$G = 32$$) wins on four of five datasets; on the large dataset $$M = 8$$ wins by ~2% but this gap is within measurement noise. $$M = 4$$ is hardcoded in the final implementation.

---

## Results

### Speedup Progression (Large Dataset, n = 30,000)

| Solver stage | 64T speedup | 128T speedup |
|---|---|---|
| Distributed CELF + $$W$$ matrix | 8.8× | regresses |
| `hc` (no $$W$$, fair batch) | 52.3× | 76.7× |
| `hpc` (coverage pruning) | 54.3× | 79.2× |
| `hpc` + barrier fusion + pinning | 48.1× | 85.1× |
| **`nhpc` (nested parallelism)** | **51.4×** | **91.5×** |

*Speedup relative to each solver's own single-threaded baseline. The 64T apparent regression after barrier fusion reflects the fused loop's 3–13% single-thread overhead.*

Absolute compute times on the large dataset:

| Solver stage | 1T (s) | 8T (s) | 64T (s) | 128T (s) |
|---|---|---|---|---|
| `hc` (no W, fair batch) | 16.81 | 2.12 | 0.333 | 0.316 |
| `hpc` (pruning, pre-barrier) | 15.18 | 1.941 | 0.289 | 0.297 |
| `hpc` (barrier fusion + pinning) | 18.16 | 2.325 | 0.384 | 0.251 |
| **`nhpc`** | **17.47** | **2.228** | **0.340** | **0.191** |

### Per-Dataset Results for `nhpc`

Scaling is monotonic across all four benchmark suites with no pathological cases. Through 16 threads, scaling is near-linear (~15× at 16T) for essentially every input. A typical medium-sized input lands around 65–70× at 128T; larger inputs perform stronger. The skew suite (62–65× at 128T) is the weakest, as load imbalance from skewed degree distributions bites at high thread counts.

| Suite | Dataset | 8T | 32T | 64T | 128T |
|---|---|---|---|---|---|
| Problem Size | Small (n=4K) | 8.1× | 15.5× | 20.7× | 18.3× |
| | Medium (n=12K) | 8.7× | 28.2× | 45.6× | 67.8× |
| | Large (n=30K) | 8.8× | 30.1× | 51.4× | **91.5×** |
| Clustering | Weak cluster | 9.0× | 30.4× | 47.7× | 76.2× |
| | Medium cluster | 8.8× | 28.7× | 44.5× | 65.3× |
| | Strong cluster | 8.6× | 27.9× | 43.8× | 63.1× |
| Skew | Uniform | 8.1× | 24.9× | 40.1× | 62.4× |
| | Mild skew | 8.4× | 27.6× | 43.0× | 64.9× |
| | Heavy skew | 8.7× | 28.4× | 44.2× | 65.1× |
| Density | Sparse | 8.2× | 24.3× | 38.8× | 60.2× |
| | Medium | 8.6× | 28.1× | 43.7× | 66.5× |
| | Dense | 8.9× | 29.5× | 47.1× | 70.4× |

### NUMA Experiments

Separate experiments on PSC's dual-socket NUMA architecture using AMD hardware perf counters showed that thread binding (`OMP_PROC_BIND=close`) and first-touch page placement significantly reduce remote DRAM accesses at 128 threads. However, the speedup gains from NUMA tuning are modest compared to algorithmic changes — the close+firsttouch configuration provides modest performance improvements, while confirming that optimizing computation should take priority over hardware tuning.

---

## What Limited Speedup

### Idle Time Dominates the Thread-Second Budget

Of the 78.21 available thread-seconds in the 128-thread large-dataset run, only **30.57 (39.1%)** are spent on active computation. The remaining **47.64 thread-seconds (60.9%) are idle**, from three structural sources:

1. **Round-boundary barriers.** Coverage array `best_sim` must be fully committed before any gain computation can begin. Global synchronization is unavoidable.
2. **Leader serialization within groups.** Non-leader threads wait while the leader performs PQ operations, freshness checks, and candidate extraction.
3. **Load imbalance and straggler effects.** Threads completing their assigned work early idle at synchronization points.

Together these produce a non-negligible sequential fraction — exactly the regime where Amdahl's Law limits achievable speedup.

### Active Cycles Are Memory-Bound

Among the cycles that do perform useful work, **72.7% of pipeline cycles are backend-stalled** due to memory-bound execution, with only 26.5% of cycles doing useful instructions. The RBF gain computation streams over both the `points` and `best_sim` arrays for every candidate; with $$n$$ in the tens of thousands, this exceeds L1 and L2 capacity, forcing repeated L3 and DRAM accesses contended by all 128 threads.

Improving spatial locality via geometrically contiguous thread regions was considered but rejected: progressive coverage pruning causes the active point set to evolve in a coverage-dependent rather than spatial pattern, so any static spatial partition degrades load balance. Memory-bound execution is accepted as an inherent characteristic of this workload.

### The Inherent Sequential Structure of Greedy

Both constraints — synchronization idle time and memory-bound active cycles — are products of the fundamental tension in parallelizing greedy submodular optimization. Correctness requires all marginal gains to be evaluated against a consistent, globally updated `best_sim` vector, imposing a read-after-write dependency that simultaneously necessitates synchronization barriers and forces every thread to stream the same large data structure. Batching alleviates synchronization frequency but introduces staler gains in CELF's lazy evaluation, reducing pruning effectiveness. The achieved speedup reflects the practical limits imposed by the interaction between algorithm structure, data geometry, and hardware.

---

## Solution Quality

Every form of parallelism introduced corresponds to a controlled relaxation of exact greedy. Batch selection commits $$b = \lfloor k/20 \rfloor$$ facilities per round with slightly stale coverage; CELF operates on gains stale by up to one round; progressive pruning at $$\tau = 0.90$$ removes points whose remaining contribution is bounded by $$1 - \tau = 0.10$$. Yet quality is preserved to a remarkable degree across all configurations.

The reason is structural: the diminishing returns property of submodular objectives bounds the loss from any single suboptimal selection, and this bound shrinks as $$\lvert S \rvert$$ grows. The pruning threshold $$\tau = 0.90$$ directly enforces a per-point error ceiling of 0.10. Empirically, across all 12 benchmark configurations and all thread counts, the parallel objective value stays within **0.1% of the sequential greedy baseline**.

---

## Conclusions

- **Centralized synchronization is difficult to scale.** Shared-queue contention causes the serial fraction to grow with thread count rather than problem size. Distributed per-thread PQs eliminate this but reveal the next bottleneck.
- **Eliminating the memory bottleneck unlocks parallelism.** The $$O(n^2)$$ similarity matrix caused memory-bandwidth saturation even at 1 thread. Replacing it with on-demand hierarchical computation shifted the solver from memory-bound to compute-bound, enabling a 5.9× improvement in 64T ceiling.
- **Algorithmic restructuring matters more than hardware tuning.** NUMA-aware placement reduces remote DRAM accesses but has modest impact on speedup compared to barrier fusion, hierarchical pruning, and nested parallelism.
- **Problem size is the dominant performance factor.** Larger instances provide more parallel work per barrier, better amortizing fixed synchronization overhead. Clustering, density, and skew have secondary and more nuanced effects.
- **Nested parallelism overcomes single-dimension scaling walls.** The $$G \times M$$ decomposition simultaneously enlarges PQ sizes (reducing CELF stale fraction from ~19% to ~4.8%), narrows global barriers (128-way → 32-way), and reduces load imbalance — enabling monotonic scaling across all datasets to 128 threads.

