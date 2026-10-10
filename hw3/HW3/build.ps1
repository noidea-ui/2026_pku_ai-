$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$nvccCommand = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvccCommand) {
    Write-Error "nvcc was not found. Check CUDA 13.1 PATH and run from a VS 2022 x64 Native Tools terminal."
    exit 1
}

# RTX 5060 uses compute capability 12.0. Override for another GPU if needed:
#   $env:HW3_CUDA_ARCH = "sm_XX"
$CudaArch = $env:HW3_CUDA_ARCH
if ([string]::IsNullOrWhiteSpace($CudaArch)) {
    $CudaArch = "sm_120"
}

$Sources = @(
    "src\fc.cu",
    "src\conv2d.cu",
    "src\max_pool.cu",
    "src\softmax.cu",
    "src\cross_entropy.cu",
    "src\main.cu"
)
foreach ($Source in $Sources) {
    if (-not (Test-Path (Join-Path $ProjectRoot $Source))) {
        Write-Error "Missing source file: $Source"
        exit 1
    }
}

$BuildDir = Join-Path $ProjectRoot "build"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$OutputExe = Join-Path $BuildDir "hw3.exe"

$NvccArgs = @(
    "-std=c++17",
    "-O2",
    "-lineinfo",
    "-arch=$CudaArch",
    "-Iinclude",
    "-Itests"
) + $Sources + @(
    "-o", $OutputExe,
    "-lcublas",
    "-lcurand"
)

Write-Host "Project : $ProjectRoot"
Write-Host "Compiler: $($nvccCommand.Source)"
Write-Host "Arch    : $CudaArch"
Write-Host "Output  : $OutputExe"
Write-Host "Building HW3..."
& $nvccCommand.Source @NvccArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "nvcc compilation failed with exit code: $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host 'Build succeeded. Run with: .\build\hw3.exe' -ForegroundColor Green
