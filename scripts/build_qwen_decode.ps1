# Build & Run Qwen2.5-0.5B-Instruct Vulkan Decode Engine
param(
    [switch]$Run = $false
)
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
& powershell -NoProfile -ExecutionPolicy Bypass -File "$RootDir\scripts\compile_shaders.ps1"

$binDir = "$RootDir\bin"
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
}

$src       = "$RootDir\src\engine\qwen_decode.c"
$outExe    = "$binDir\qwen_decode.exe"
$vulkanInc = "$env:VULKAN_SDK\Include"
$vulkanLib = "$env:VULKAN_SDK\Lib\vulkan-1.lib"
$k32Lib    = "$RootDir\libs\kernel32.lib"
$crtLib    = "$RootDir\libs\msvcrt.lib"

Write-Host "`n================ Building qwen_decode.exe =================" -ForegroundColor Cyan
& H:\LLVM\bin\clang.exe -O3 -mno-stack-arg-probe -nostdlib -fno-builtin -Wno-builtin-requires-header `
    -I "$vulkanInc" -I "$RootDir\src\engine" `
    -Xlinker /subsystem:console `
    -Xlinker /entry:mainCRTStartup `
    "$src" "$k32Lib" "$crtLib" "$vulkanLib" `
    -o "$outExe"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to compile qwen_decode.exe!"
    exit 1
}

Write-Host "[OK] Built $outExe" -ForegroundColor Green

if ($Run) {
    Set-Location $RootDir
    & "$outExe"
}

