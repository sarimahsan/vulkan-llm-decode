# Shader compiler for vulkan-llm-decode
param (
    [string]$ShaderDir = "$PSScriptRoot\shaders",
    [string]$OutputDir = "$PSScriptRoot\spirv",
    [switch]$Disassemble = $true
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$glslc = Get-Command "glslc" -ErrorAction SilentlyContinue
if (-not $glslc) {
    Write-Error "glslc not found on PATH! Vulkan SDK is required."
    exit 1
}

$spirvDis = Get-Command "spirv-dis" -ErrorAction SilentlyContinue

Write-Host "================ Compiling LLM Decode Shaders ================" -ForegroundColor Cyan

$shaders = Get-ChildItem -Path $ShaderDir -File -Include *.comp -Recurse
foreach ($file in $shaders) {
    $outFile = Join-Path $OutputDir "$($file.BaseName).spv"
    $asmFile = Join-Path $OutputDir "$($file.BaseName).spvasm"
    
    Write-Host "Compiling: $($file.Name) -> $($outFile)" -ForegroundColor Yellow
    & glslc.exe --target-env=vulkan1.3 -O "$($file.FullName)" -o "$outFile"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $($file.Name)" -ForegroundColor Green
        if ($Disassemble -and $spirvDis) {
            & spirv-dis.exe "$outFile" -o "$asmFile"
            Write-Host "  [OK] Disassembled to $($file.BaseName).spvasm" -ForegroundColor DarkGray
        }
    } else {
        Write-Host "  [FAILED] $($file.Name)" -ForegroundColor Red
    }
}

Write-Host "================ Compilation Complete ========================" -ForegroundColor Cyan

