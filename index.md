# Parallelizing Submodular Optimization

**Authors:** Kadin Zhang, Pranav Sangwan

---

## URL
**Project Webpage:** [https://kadinlz.github.io/submodule-optimization-proposal/](https://kadinlz.github.io/submodule-optimization-proposal/)

---

## Summary
We are going to study parallelization strategies for greedy submodular optimization by exploring how relaxing its inherent sequential structure affects performance and solution quality. We will implement both shared-address-space and message-passing versions of the algorithm, and analyze tradeoffs introduced by batching, stale updates, and the implementation limitations of the 2 frameworks. Finally, we will attempt to use GPU-based acceleration of the marginal gain evaluation kernel as a potential optimization for the most compute-intensive portion of the workload.

---

## Background
Submodular optimization is a fundamental problem in combinatorial optimization which has many applications in machine learning, data summarization, and network analysis. The problem definition is as follows. A set function $f: 2^V \rightarrow \mathbb{R}$ is submodular if it satisfies the diminishing returns property: for all $A \subseteq B \subseteq V$ and $x \notin B$,

$$f(A \cup \{x\}) - f(A) \ge f(B \cup \{x\}) - f(B)$$

This condition essentially just formalizes the intuition that adding an element provides less benefit when the current set is larger by requiring diminishing returns. The goal of the problem is to maximize a submodular function $f$ value over all choices of input $S$ such that $|S| \leq k$. More formally:

$$\max_{|S| \leq k} f(S)$$

The standard way to do this is to use a greedy algorithm, which iteratively selects the next element to add to $S$ to be the one providing the largest marginal gain. This algorithm achieves a $(1 - 1/e)$ approximation to the optimal solution where $e$ is Euler’s constant. However, this algorithm is inherently sequential as each iteration depends on the current selected set $S$, and the marginal gains must be recomputed after every selection.

Despite this sequential structure, the algorithm contains components that suggest opportunities for parallel execution. In particular, each iteration requires evaluating marginal gains for a large number of candidate elements, which can be done in parallel. More interestingly, parallelism can be further increased by relaxing the strict sequential dependence of the greedy algorithm. For example, instead of selecting a single element at each iteration, multiple elements can be selected in a batch using marginal gains computed from a shared (possibly stale) state. Similarly, relaxations such as using stale marginal gains for selections can be explores which reduces synchronization.

Additional such relaxations that can be explored include evaluating only a subset of candidates (sampling) or partitioning data across processes in a distributed setting. All of these relaxations introduce a tradeoff where increasing parallelism improves performance but degrades solution quality due to outdated or incomplete information. In this project, we explore this design space of relaxations and evaluate how different choices impact performance, work efficiency, and solution optimality.

To justify why we are exploring this problem we note that submodular optimization is significant due to its wide range of real-world applications. For instance, in machine learning, it arises naturally in feature selection, where the goal is to choose a subset of informative features that maximizes predictive performance while avoiding redundancy. Another example is in data summarization tasks, where the goal is to select a representative subset of documents or images. These applications often involve large datasets and high computational cost, making efficient parallel implementations essential and the understanding how to effectively parallelize this workload valuable.

---

## The Challenge
Parallelizing greedy submodular optimization presents several challenges including:

* **Sequential Dependencies:** Each iteration updates the selected set $S$, and all future marginal gain computations depend on this updated state. This creates a strong dependency chain between iterations, making it difficult to parallelize across iterations without affecting correctness.
* **Synchronization Overhead:** In a parallel setting, threads must coordinate updates to the shared set $S$, which can limit scalability. Reducing synchronization may lead to wasted and incorrect optimization directions.
* **Memory Behavior:** The algorithm repeatedly evaluates marginal gains over a large candidate set which can involve irregular memory access patterns, leading to poor cache locality and increased memory bandwidth usage.
* **Parallelism and Solution Quality Tradeoffs:** Increasing parallelism requires relaxing strict sequential behavior. While this improves performance, it may lead to suboptimal choices.
* **GPU Acceleration Challenges:** Mapping evaluations to a GPU requires careful consideration of data layout and memory transfers. Irregular access patterns may limit speedups.
* **Load Imbalance and Scheduling:** The cost of evaluating different elements varies significantly, leading to load imbalance across threads.

---

## Resources
We will utilize the GHC cluster machines (equipped with 8-core Intel i7-9700 CPUs and NVIDIA RTX 2080 GPUs) for our baseline shared-memory and GPU acceleration experiments. For large-scale, NUMA-aware, and distributed testing, we will use the PSC Bridges-2 supercomputer, specifically the nodes featuring dual 64-core AMD EPYC processors.

The core algorithms will be written from scratch in C++20. We will rely on the standard thread and atomic libraries for lock-free concurrency, NUMA libraries for explicit thread pinning and memory binding, OpenMPI for the message-passing implementation, and NVIDIA CUDA for the GPU-accelerated marginal gain evaluator.

---

## Goals and Deliverables

### Plan to Achieve
* **Shared-Memory Solver:** A lock-free relaxed greedy algorithm in C++ using `std::atomic` and relaxed memory orderings to evaluate the limits of unsynchronized parallel execution.
* **Distributed Solver:** An asynchronous message-passing implementation using MPI to test network staleness bounds across multiple nodes.
* **GPU Acceleration:** A CUDA-based marginal gain evaluator utilizing shared memory tiling to parallelize the compute-intensive scanning of candidate elements.
* **Tradeoff Analysis:** Quantify the exact tradeoffs between increased parallelism (stale updates, batching), hardware utilization (cache misses), and the resulting degradation in solution quality.

### Hope to Achieve
* **NUMA Optimization:** Develop a NUMA-aware memory allocation strategy on the PSC Bridges-2 machines using `libnuma` to mitigate interconnect contention.
* **Unified Memory Profiling:** Compare our explicit GPU memory management strategy against CUDA Unified Memory to empirically measure the overhead of shared-address programming on the GPU.
* **Real-World Scale:** Apply our solvers to a massive real-world dataset (e.g., social network influence maximization) that stress tests our scaling.

---

## Platform Choice
For shared address space studies we will use C++ (without higher level parallel programming abstractions) using the PSC Bridges-2 and GHC machines. C++ provides the atomic library, allowing us to interface directly with hardware-level memory orderings. By opting for `std::memory_order_relaxed`, we can prevent unnecessary memory fences, maximizing memory bandwidth.

The GHC machines represent a standard UMA model, while the PSC Bridges-2 machines use a complex NUMA model. This allows us to test how well our lock-free approach scales when moving to chiplet architecture. The 128-core PSC nodes allow us to stress test cache coherence and interconnect contention.

For the GPU portion, we will utilize the NVIDIA GeForce RTX 2080 GPUs on the GHC cluster. The RTX 2080’s thousands of CUDA cores allow us to map each candidate element evaluation to a discrete thread, bypassing the sequential bottleneck of a CPU-bound scan.

---

## Schedule
* **Week 1:** Implement the baseline sequential greedy algorithm and the initial lock-free shared-memory version. Establish single-node benchmarking on GHC machines.
* **Week 2:** Port the shared-memory solver to PSC Bridges-2. Implement NUMA-aware memory binding. Begin development of the GPU CUDA kernel.
* **Week 3:** Finalize the GPU implementation with shared memory tiling. Begin the MPI-based distributed implementation for multi-node execution.
* **Week 4:** Complete the MPI implementation. Conduct comprehensive benchmarking across all platforms, gathering hardware performance counters.
* **Week 5:** Analyze collected data to quantify tradeoffs between parallelism and solution quality. Prepare the final presentation and report.
