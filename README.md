# vulkan-llm-decode

> **Bandwidth-Roofline LLM Decode Engine on Vulkan**  
> Fused INT4/INT8 dequant-GEMV kernels for LLM decoding on bandwidth-constrained Intel Xe-LP (UHD Graphics) architectures.

---

## 🎯 Research Motivation & Goals
LLM autoregressive generation (token decoding at batch size $M=1$) is almost exclusively memory-bandwidth bound. Every token generation requires streaming all model parameters through the compute units to perform matrix-vector multiplications (GEMV):

$$\text{Tokens/sec} \approx \frac{\text{Memory Bandwidth (GB/s)}}{\text{Model Size (GB)}}$$

On integrated GPUs (such as Intel UHD Graphics / Xe-LP), the GPU shares system DRAM with the CPU. With physical DRAM bandwidth measured at **~16.28 GB/s**, a 0.5B model in INT4 (~295 MB) has a strict physical roofline ceiling:

$$\text{Roofline Ceiling} \approx \frac{16.28\text{ GB/s}}{0.295\text{ GB}} \approx \mathbf{55.2\text{ tokens/sec}}$$

### Core Research Questions
1. **Roofline Efficiency:** How close can a fused dequantize-and-multiply GEMV get to the physical DRAM memory roofline on Xe-LP?
2. **Fusion Impact:** How much throughput is lost when performing unfused dequantization (intermediate DRAM roundtrips) vs. in-register fused dequantization?
3. **Instruction Extensions:** What are the performance gains of DP4A (`VK_KHR_shader_integer_dot_product` / `shaderIntegerDotProduct`) and subgroup-level reductions (`subgroupAdd`)?
4. **Architecture Comparison:** CPU (i5 AVX2/VNNI) vs. iGPU (Vulkan EU SIMD32) on the identical unified physical memory bus.

---

## 📊 Hardware Baseline (Intel UHD Graphics)

* **Vulkan Driver:** 1.4.323
* **Hardware Timestamp Resolution:** 52.0833 ns (`vkCmdWriteTimestamp`)
* **Real Peak Copy Bandwidth:** **14.64 GB/s** (128 MB working set)
* **Real Peak Vector-Add Bandwidth:** **16.28 GB/s** (192 MB working set)

---

## 📁 Repository Structure
```
vulkan-llm-decode/
├── .vscode/        # VS Code tasks and shader language configurations
├── bin/            # Compiled benchmark executables (git-ignored)
├── libs/           # Windows CRT import libraries (kernel32.lib, msvcrt.lib)
├── scripts/        # PowerShell automation & benchmark build scripts
├── shaders/        # GLSL compute kernels (FP16, INT8 DP4A, INT4 fused GEMV)
├── spirv/          # Compiled SPIR-V bytecode & assembly (.spvasm)
├── src/            # Host benchmark & inference runtime (C / Vulkan)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Compile Shaders
```powershell
.\scripts\compile_shaders.ps1
```

### 2. Run Memory Bandwidth Benchmark
```powershell
.\scripts\run_bandwidth_test.ps1
```

