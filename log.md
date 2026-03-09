# Dev Log — Sparse Flash Attention in CUDA

---

## Entry 1 — Naive attention kernel
Started with the simplest possible attention: one thread per output element `O[i,j]`. Each thread computes the full row of scores `S[i,:]`, softmaxes it, then dots with `V[:,j]`. Got the math right pretty quick — `Q @ K^T / sqrt(d)` then softmax then `@ V`. The tricky part was translating Python slices into flat C array indexing. `Q[i,j]` becomes `Q[i*head_dim + j]`. Once that clicked everything else followed.

**Bugs caught:** off-by-one in loop bounds (`t <= seq_len` instead of `t < seq_len`). Classic. Also initialized `max_val = 0.0f` for softmax which breaks for all-negative scores — changed to `-1e9f`.

---

## Entry 2 — Tiling + shared memory
The naive kernel reads from global memory (HBM) constantly. Every thread fetches its own copy of Q, K, V rows independently. Wasteful. The fix is shared memory — a fast, on-chip scratchpad (~48KB per block) where the whole thread block cooperates to load a tile once, then everyone reads from there.

Declared `__shared__ float Qs[Br][D]`, `Ks[Bc][D]`, `Vs[Bc][D]`. Tile sizes `Br=16`, `Bc=16`. Each thread loads one element into the tile cooperatively. Added `__syncthreads()` after loading — without it, fast threads start reading tiles that slow threads haven't written yet. Race condition. Added a second `__syncthreads()` at the end of each tile iteration to prevent overwriting the tile before everyone's done reading it.

**Bugs caught:** declared shared memory as `vs` (lowercase) instead of `Vs`. CUDA is case-sensitive. Took a compiler error to catch it.

---

## Entry 3 — Sparsity mask
Added a CPU-side function `build_sparse_mask()` that precomputes which tokens attend to which. Pattern: local window of size `w` (nearby tokens) + global stride every `s` tokens. Token `i` attends to `j` if `abs(i-j) <= w` or `j % s == 0`. Stored as flat int array `mask[N*N]`.

Inside the kernel, before loading a K/V tile, check if any token in the query tile attends to any token in the key tile. If not, `continue` — skip the tile entirely. This is the core of sparse attention: O(n·w) work instead of O(n²).

**Bugs caught:** used `N` (undefined inside kernel) instead of `seq_len` in several places. Also had a semicolon inside the function signature (`int* mask;`) which took embarrassingly long to find.

---

## Entry 4 — PyTorch extension wrapper
Wanted to call the kernel from Python for benchmarking. Built a PyTorch C++ extension using `torch/extension.h` and pybind11. The launcher function takes `torch::Tensor` objects, extracts raw GPU pointers via `data_ptr<float>()`, and launches the kernel.

`setup.py` compiles everything via `CUDAExtension`. First ran `python setup.py install` by mistake — this installed the module as an egg in the system Python path. Later recompiles didn't update it. Spent a while confused why new code wasn't running. Fixed by deleting the egg from `/usr/local/lib/python3.12/dist-packages/` and switching to `build_ext --inplace`.

**Bugs caught:** typo `torch.utlis` instead of `torch.utils` in setup.py. Launcher returned `0` instead of `O` (the tensor). `mask` type mismatch — kernel expected `int*` but tensor wasn't explicitly created as `torch.int32`.

---

## Entry 5 — First benchmark (v1 vs PyTorch dense)
Ran first three-way benchmark: sparse v1 vs PyTorch dense attention. Results were humbling — sparse kernel was ~16x slower at N=64, and got worse as N grew. Linear blowup vs PyTorch staying flat.

Root cause: `float S[1024]` on the stack. Every thread allocates N floats in local memory which spills to global memory. At N=512, each thread is doing 512 read/writes to the slowest memory on the GPU just for the score array. The sparsity optimization was being completely buried by this.

---

## Entry 6 — Online softmax (v2)
Replaced `S[1024]` with online softmax — the key idea from the original Flash Attention paper. Instead of storing all scores then softmaxing at the end, maintain running accumulators:
- `m`: running max (for numerical stability)
- `l`: running sum of exp (denominator)  
- `acc`: running output (single float per thread, not array)

As each tile comes in, update `m` and rescale the previous accumulator before adding the new tile. At the end divide `acc` by `l`. Memory per thread drops from O(N) to O(1) — four scalars regardless of sequence length.

**Bugs caught:** declared `float acc = 0.0f` but still used `acc[k]` as array in several places — left over from the old version. Also shadowed the outer variable `j = threadIdx.y` by redeclaring `int j` inside the sparsity loop. Renamed inner loop variable to `jj`. Hit `cudaErrorLaunchOutOfResources` before fixing this — 1024 threads × 64 floats = 65536 registers, exactly the T4's per-SM limit.

---

## Entry 7 — Second benchmark (v1 vs v2 vs PyTorch)
v2 is consistently ~35% faster than v1 across all sequence lengths. The O(1) register usage fixed the memory bottleneck. Both kernels are still slower than PyTorch dense — PyTorch uses cuBLAS which is optimized at the assembly level. Both custom kernels still grow linearly with N.

Linear growth means the sparsity check itself is the bottleneck now. The `any_active` check runs inside the kernel and reads global memory for every tile — doing extra work to avoid work. Next step would be precomputing the active tile list on CPU and passing it directly to the kernel, so the GPU never checks, it just skips.

---

## What's next
- Precompute active tile list on CPU, pass as a sorted array of `(row_tile, col_tile)` pairs
- Test at N=2048, 4096 where sparse attention's O(n·w) advantage should finally show
- Wrap everything in a proper Python class that matches the `nn.MultiheadAttention` interface