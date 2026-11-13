# Build & Run GGUF v3 Zero-Copy Full Transformer Decoder Layer Demo
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot

Write-Host "================ Compiling All Shaders ====================" -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File "$RootDir\scripts\compile_shaders.ps1"

$binDir = "$RootDir\bin"
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
}

# Generate a valid GGUF v3 file (Qwen-2.5-0.5B Layer 0 architecture) if not present
$ggufPath = "$binDir\test_layer.gguf"
if (-not (Test-Path $ggufPath)) {
    Write-Host "`n================ Generating Test GGUF v3 Layer ============" -ForegroundColor Cyan
    $H = 896
    $I = 4864
    $QHeads = 14
    $KVHeads = 2
    $KVDim = 128

    $fs = [System.IO.File]::Create($ggufPath)
    $bw = New-Object System.IO.BinaryWriter($fs)

    function Write-GgufStr([string]$s) {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($s)
        $bw.Write([uint64]$bytes.Length)
        $bw.Write($bytes)
    }

    # Header: Magic "GGUF" (0x46554747), Version 3, 9 tensors, 6 metadata KVs
    $bw.Write([uint32]0x46554747)
    $bw.Write([uint32]3)
    $bw.Write([uint64]9)
    $bw.Write([uint64]6)

    # Metadata KVs
    Write-GgufStr "qwen2.embedding_length";                  $bw.Write([uint32]4); $bw.Write([uint32]$H)
    Write-GgufStr "qwen2.feed_forward_length";               $bw.Write([uint32]4); $bw.Write([uint32]$I)
    Write-GgufStr "qwen2.attention.head_count";              $bw.Write([uint32]4); $bw.Write([uint32]$QHeads)
    Write-GgufStr "qwen2.attention.head_count_kv";           $bw.Write([uint32]4); $bw.Write([uint32]$KVHeads)
    Write-GgufStr "qwen2.attention.layer_norm_rms_epsilon";  $bw.Write([uint32]6); $bw.Write([float]1e-6)
    Write-GgufStr "qwen2.rope.freq_base";                    $bw.Write([uint32]6); $bw.Write([float]1000000.0)

    # Tensor definitions: [name, dims, type, byteSize]
    $tensors = @(
        @{ Name="blk.0.attn_norm.weight";   Dims=@([uint64]$H);              Type=1; Bytes=($H * 2) },
        @{ Name="blk.0.attn_q.weight";      Dims=@([uint64]$H, [uint64]$H);  Type=2; Bytes=(($H * $H / 32) * 18) },
        @{ Name="blk.0.attn_k.weight";      Dims=@([uint64]$H, [uint64]$KVDim); Type=2; Bytes=(($KVDim * $H / 32) * 18) },
        @{ Name="blk.0.attn_v.weight";      Dims=@([uint64]$H, [uint64]$KVDim); Type=2; Bytes=(($KVDim * $H / 32) * 18) },
        @{ Name="blk.0.attn_output.weight"; Dims=@([uint64]$H, [uint64]$H);  Type=2; Bytes=(($H * $H / 32) * 18) },
        @{ Name="blk.0.ffn_norm.weight";    Dims=@([uint64]$H);              Type=1; Bytes=($H * 2) },
        @{ Name="blk.0.ffn_gate.weight";    Dims=@([uint64]$H, [uint64]$I);  Type=2; Bytes=(($I * $H / 32) * 18) },
        @{ Name="blk.0.ffn_up.weight";      Dims=@([uint64]$H, [uint64]$I);  Type=2; Bytes=(($I * $H / 32) * 18) },
        @{ Name="blk.0.ffn_down.weight";    Dims=@([uint64]$I, [uint64]$H);  Type=2; Bytes=(($H * $I / 32) * 18) }
    )

    [uint64]$curOff = 0
    foreach ($t in $tensors) {
        Write-GgufStr $t.Name
        $bw.Write([uint32]$t.Dims.Length)
        foreach ($d in $t.Dims) { $bw.Write($d) }
        $bw.Write([uint32]$t.Type)
        $bw.Write($curOff)
        $aligned = [uint64](([math]::Ceiling($t.Bytes / 32.0)) * 32)
        $curOff += $aligned
    }

    # Align header to 32 bytes
    $pos = $fs.Position
    $pad = (32 - ($pos % 32)) % 32
    for ($p = 0; $p -lt $pad; $p++) { $bw.Write([byte]0) }

    # Populate Tensor Data
    # FP16 1.0 = 0x3C00; FP16 0.015625 (1/64) = 0x2C00
    foreach ($t in $tensors) {
        $buf = New-Object byte[] $t.Bytes
        if ($t.Type -eq 1) {
            # F16 weights = 1.0 (0x3C00)
            for ($k = 0; $k -lt $t.Bytes; $k += 2) {
                $buf[$k] = 0x00; $buf[$k + 1] = 0x3C
            }
        } else {
            # Q4_0 blocks: d = 0.015625 (0x2C00), deterministic nibbles around 8
            for ($b = 0; $b -lt $t.Bytes; $b += 18) {
                $buf[$b] = 0x00; $buf[$b + 1] = 0x2C
                for ($q = 0; $q -lt 16; $q++) {
                    $n0 = (($b + $q) % 5) + 6     # -2..+2
                    $n1 = (($b + $q + 2) % 5) + 6 # -2..+2
                    $buf[$b + 2 + $q] = [byte](($n1 -shl 4) -bor $n0)
                }
            }
        }
        $bw.Write($buf)
        $rem = (32 - ($t.Bytes % 32)) % 32
        for ($p = 0; $p -lt $rem; $p++) { $bw.Write([byte]0) }
    }

    $bw.Close()
    $fs.Close()
    Write-Host "[OK] Generated $ggufPath ($([math]::Round((Get-Item $ggufPath).Length / 1MB, 2)) MB)" -ForegroundColor Green
}

Write-Host "`n================ Compiling GGUF Layer Runner ==============" -ForegroundColor Cyan
$src = "$RootDir\src\benchmarks\gguf_layer_runner.c"
$outExe = "$binDir\gguf_layer_runner.exe"
$vulkanInc = "$env:VULKAN_SDK\Include"
$vulkanLib = "$env:VULKAN_SDK\Lib\vulkan-1.lib"
$k32Lib = "$RootDir\libs\kernel32.lib"
$crtLib = "$RootDir\libs\msvcrt.lib"

& H:\LLVM\bin\clang.exe -O3 -nostdlib -fno-builtin -Wno-builtin-requires-header -I "$vulkanInc" -I "$RootDir\src\engine" `
    -Xlinker /subsystem:console `
    -Xlinker /entry:mainCRTStartup `
    "$src" "$k32Lib" "$crtLib" "$vulkanLib" `
    -o "$outExe"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to compile GGUF layer runner!"
    exit 1
}

Write-Host "[OK] Built $outExe successfully!`n" -ForegroundColor Green

Write-Host "================ Running GGUF Transformer Layer ===========" -ForegroundColor Cyan
Set-Location $RootDir
& "$outExe"

