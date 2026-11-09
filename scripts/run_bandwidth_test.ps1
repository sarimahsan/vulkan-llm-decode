# Build & Run GPU Bandwidth Benchmark
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot

Write-Host "================ Compiling Bandwidth Shaders ================" -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File "$RootDir\scripts\compile_shaders.ps1"

Write-Host "`n================ Compiling Host Benchmark Runner =============" -ForegroundColor Cyan
$binDir = "$RootDir\bin"
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
}

$src = "$RootDir\src\bandwidth_benchmark.c"
$outExe = "$binDir\bandwidth_benchmark.exe"
$vulkanInc = "$env:VULKAN_SDK\Include"
$vulkanLib = "$env:VULKAN_SDK\Lib\vulkan-1.lib"
$k32Lib = "$RootDir\libs\kernel32.lib"
$crtLib = "$RootDir\libs\msvcrt.lib"

& H:\LLVM\bin\clang.exe -O3 -nostdlib -I "$vulkanInc" `
    -Xlinker /subsystem:console `
    -Xlinker /entry:mainCRTStartup `
    "$src" "$k32Lib" "$crtLib" "$vulkanLib" `
    -o "$outExe"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to compile bandwidth benchmark executable!"
    exit 1
}

Write-Host "[OK] Built $outExe successfully!`n" -ForegroundColor Green

Write-Host "================ Running GPU Bandwidth Test ================" -ForegroundColor Cyan
Set-Location $RootDir
& "$outExe"

