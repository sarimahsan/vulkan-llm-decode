# 🚀 vulkan-llm-decode: Project Status & Comprehensive Overview

> **Bandwidth-Roofline LLM Decode Engine on Vulkan**  
> Fused INT4/INT8 dequantization GEMV kernels on bandwidth-constrained Intel Xe-LP (UHD Graphics) unified memory architecture.

---

## 📍 1. Where We Are Right Now (Current State)

We have successfully engineered, validated, and benchmarked a complete **memory-roofline GEMV inference engine** on Vulkan 1.4:

1. **Hardware Memory Roofline Characterized:**  
   The physical DRAM ceiling on this unified memory architecture was measured at **~16.28 GB/s** using dedicated hardware timestamping (`vkCmdWriteTimestamp`, 52.0833 ns resolution).
2. **Five Working Subgroup-Accelerated Compute Kernels:**  
   - **FP16 Baseline GEMV** (`shaders/gemv_fp16.comp`)
   - **INT8 Fused Dequant-GEMV** (`shaders/gemv_int8.comp`)
   - **INT4 Per-Row Fused Dequant-GEMV** (`shaders/gemv_int4.comp`)
   - **Group-32 Linear INT4 Dequant-GEMV (AWQ/GPTQ style)** (`shaders/gemv_int4_group32.comp`)
   - **Native GGUF Q4_0 Interleaved Dequant-GEMV** (`shaders/gemv_q4_0.comp`)
3. **Multi-Precision Verification & Benchmark Runners:**  
   - [`src/gemv_benchmark.c`](src/gemv_benchmark.c): FP16 vs INT8 vs INT4 roofline baseline.
   - [`src/gemv_q4_benchmark.c`](src/gemv_q4_benchmark.c): Group-32 Linear vs Native GGUF `block_q4_0` interleaved format.
   - Bit-exact mathematical validation against CPU floating-point reference: **0.00000 discrepancy across all kernels**.
4. **Major Empirical Findings:**  
   - **Per-Row INT4 GEMV achieves a 3.97× speedup over FP16**, sustaining **15.46 GB/s (95.0% of physical DRAM)**.
   - **Group-32 Linear INT4 (AWQ) achieves 15.54 GB/s (95.5% of DRAM roofline)** with **2.431 ms latency** on $8K \times 8K$.
   - **Native GGUF Q4_0 Interleaved achieves 15.21 GB/s (93.4% of DRAM roofline)** with **2.484 ms latency** on $8K \times 8K$—proving **zero-copy reading of native GGUF files directly from disk without tensor conversion** sustains peak DRAM bandwidth!

---

## 📊 2. Key Experimental Results

Benchmarked on **Intel(R) UHD Graphics (Xe-LP / SIMD32 EUs)** against the **16.28 GB/s** physical memory roofline:

### Full DRAM Bus Saturation Benchmark ($8192 \times 8192$, 0.134 GFLOPs)

| Quantization Format | Storage Layout | Matrix Footprint | Bits/Param | Latency | Achieved Bandwidth | % DRAM Roofline | Throughput | Verification |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FP16 Baseline** | Dense FP16 | 128.00 MB | 16.0 | **8.627 ms** | **15.56 GB/s** | 95.6% | 115.9 inf/s | **PASSED (0.0000)** |
| **INT8 Fused Dequant** | Dense INT8 + row scale | 64.00 MB | 8.0 | **4.522 ms** (1.91×) | **14.85 GB/s** | 91.2% | 221.1 inf/s | **PASSED (0.0000)** |
| **INT4 Per-Row Dequant** | Packed INT4 + row scale | 32.00 MB | 4.0 | **2.173 ms** (3.97×) | **15.46 GB/s** | 95.0% | 460.3 inf/s | **PASSED (0.0000)** |
| **Group-32 Linear (AWQ)** | Split `uvec4` weights + FP16 scales | 36.00 MB | **4.5** | **2.431 ms** (**3.55×**) | **15.54 GB/s** | **95.5%** | **411.4 inf/s** | **PASSED (0.0000)** |
| **Native GGUF Q4_0** | Interleaved 18-byte `block_q4_0` | 36.00 MB | **4.5** | **2.484 ms** (**3.47×**) | **15.21 GB/s** | **93.4%** | **402.5 inf/s** | **PASSED (0.0000)** |

### LLaMA-3 / Qwen-2.5 Hidden Dimension ($4096 \times 4096$, 0.034 GFLOPs)

| Quantization Format | Matrix Size | Kernel Latency | Effective Bandwidth | Layer Inferences/sec | Verification |
|:---|:---:|:---:|:---:|:---:|:---:|
| **FP16 Baseline** | 32.00 MB | 2.745 ms | 12.23 GB/s | 364.3 inf/s | **PASSED (0.0000)** |
| **INT8 Fused Dequant** | 16.00 MB | 1.296 ms (2.12×) | 12.96 GB/s | 771.5 inf/s | **PASSED (0.0000)** |
| **Group-32 Linear (AWQ)** | 9.00 MB | 2.104 ms | 4.49 GB/s | 475.3 inf/s | **PASSED (0.0000)** |
| **Native GGUF Q4_0** | 9.00 MB | 2.043 ms | 4.63 GB/s | 489.4 inf/s | **PASSED (0.0000)** |

---

## 🛠️ 3. Core Architectural Accomplishments

### A. Minimalist Zero-Dependency Toolchain
- Compiles via **Clang** (`-nostdlib`) linked directly with minimal CRT import stubs ([`libs/kernel32.lib`](libs/kernel32.lib), [`libs/msvcrt.lib`](libs/msvcrt.lib)) and Vulkan 1.4 (`vulkan-1.lib`).
- No heavy Visual Studio SDK or runtime installation required.
- Executables are placed cleanly in [`bin/`](bin/) and kept out of Git tracking.

### B. Subgroup Warp-Level SIMD32 Reduction
- Every EU thread subgroup (32 threads) processes one row of output $y[i]$.
- Uses `GL_KHR_shader_subgroup_arithmetic` (`subgroupAdd(acc)`), completely eliminating shared memory allocation, barriers, and bank conflict overhead.

### C. In-Register INT4 Unpacking
- Loads 128 bits per memory transaction (`uvec4`), fetching **32 INT4 weights simultaneously**.
- Unpacks 4-bit nibbles directly inside the SIMD32 EU registers using bitwise shifts and masking (`& 0x0Fu - 8`).
- Multiplies against `f16vec4` activation vectors and scales per-row on lane 0.

---

## 📁 4. Project File Map

```
vulkan-llm-decode/
├── .vscode/               # VS Code tasks & GLSL syntax extensions
│   ├── extensions.json
│   ├── settings.json
│   └── tasks.json
├── bin/                   # Compiled benchmark binaries (git-ignored)
│   ├── bandwidth_benchmark.exe
│   └── gemv_benchmark.exe
├── libs/                  # Windows CRT import stubs
│   ├── kernel32.def / .lib
│   └── msvcrt.def / .lib
├── scripts/               # PowerShell build & execution scripts
│   ├── compile_shaders.ps1
│   ├── run_bandwidth_test.ps1
│   └── run_gemv_benchmark.ps1
├── shaders/               # GLSL compute kernels
│   ├── copy_bandwidth.comp
│   ├── vector_add_bandwidth.comp
│   ├── gemv_fp16.comp
│   ├── gemv_int8.comp
│   └── gemv_int4.comp
├── spirv/                 # Compiled SPIR-V bytecode & disassembly
├── src/                   # C benchmark runtimes
│   ├── bandwidth_benchmark.c
│   └── gemv_benchmark.c
├── .gitignore
├── PROJECT_STATUS.md      # Detailed project report (this file)
└── README.md              # Public repository documentation
```

---

## 🎯 5. The Agenda (Next Milestones When Resuming)

When we resume, here is the recommended roadmap to advance toward a complete end-to-end decode engine:

```
[Phase 1] ✅ Memory Roofline Baseline & Infrastructure (COMPLETE)
[Phase 2] ✅ FP16, INT8, and INT4 Fused GEMV Kernels (COMPLETE)
[Phase 3] ✅ Grouped Quantization: AWQ & Native GGUF Q4_0 (COMPLETE)
[Phase 4] ✅ Fused Transformer Activation Kernels (COMPLETE)
    │     ├── RMSNorm (40.05 us @ M=1, K=4096): PASSED (0.00000 diff)
    │     └── SwiGLU  (20.78 us @ M=1, N=11008): PASSED (0.00012 diff)
    ▼
[Phase 5] ✅ Paged KV-Cache & Single-Token Attention (COMPLETE)
    │     ├── RoPE (Q=16, KV=4, dim=128): 17.40 us (PASSED, 0.00045 diff)
    │     └── FlashDecode GQA Attention (seq_len=2048): 884.06 us (36.4x speedup, 0.00000 diff)
    ▼
[Phase 6] ✅ GGUF / SafeTensors Model Loader (COMPLETE)
    │     ├── Zero-dependency GGUF v3 parser (src/gguf.h) + 15-op single-CB Transformer layer (src/gguf_layer_runner.c)
    │     ├── End-to-End 15-Step Verification (Qwen-2.5-0.5B Layer 0): PASSED (0.00000 max diff over 896 dims)
    │     └── Single-Layer Latency: 1.897 ms best / 2.126 ms avg (~22.0 tokens/sec extrapolated 24-layer decode)
    ▼
[Phase 7] ✅ Full Multi-Layer Qwen2.5-0.5B-Instruct (Q4_0) Inference Engine (COMPLETE)
          ├── Added coalesced Group-32 INT4 GEMV + Q8->Q4 LM Head re-quantization, fused Gate+Up+SwiGLU (shaders/gemv_q4_swiglu_fused.comp), dynamic-pos static pre-recorded CBs, GPU argmax & BPE vocab decoder
          ├── Running real 408.87 MB Qwen2.5-0.5B-Instruct-Q4_0 GGUF (24 layers, 151,936 vocab) on Intel(R) UHD Graphics
          └── Observed Speed (30 tokens): 41.14 tokens/sec wall-clock (24.31 ms/tok; 22.83 ms GPU: 15.25 ms 24 layers + 7.32 ms Q4 LM Head)
```

