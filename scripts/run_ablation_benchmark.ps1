# Build & Run Research Question Ablation Benchmarks
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
& powershell -NoProfile -ExecutionPolicy Bypass -File "$RootDir\scripts\compile_shaders.ps1"

$binDir = "$RootDir\bin"
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
}

$src       = "$RootDir\src\benchmarks\ablation_benchmark.c"
$outExe    = "$binDir\ablation_benchmark.exe"
$vulkanInc = "$env:VULKAN_SDK\Include"
$vulkanLib = "$env:VULKAN_SDK\Lib\vulkan-1.lib"
$k32Lib    = "$RootDir\libs\kernel32.lib"
$crtLib    = "$RootDir\libs\msvcrt.lib"

& H:\LLVM\bin\clang.exe -O3 -nostdlib -I "$vulkanInc" -I "$RootDir\src\engine" `
    -Xlinker /subsystem:console `
    -Xlinker /entry:mainCRTStartup `
    "$src" "$k32Lib" "$crtLib" "$vulkanLib" `
    -o "$outExe"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to compile ablation_benchmark.exe!"
    exit 1
}

Write-Host "[OK] Built $outExe" -ForegroundColor Green
Set-Location $RootDir
& "$outExe"

