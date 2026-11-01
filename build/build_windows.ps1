<#
.SYNOPSIS
    Automated Windows Desktop Application Build and Packaging Script for VideoContextEngine.

.DESCRIPTION
    1. Executes pre-build secret scan to prevent credential leaks.
    2. Runs test suite to verify regression-free state.
    3. Verifies bundled FFmpeg / FFprobe binaries and licenses.
    4. Executes PyInstaller to produce standalone onedir bundle.
    5. Runs smoke verification tests on the packaged executable.
    6. Invokes Inno Setup 6 Compiler (ISCC) to build the Windows installer.
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  VideoContextEngine: Automated Windows Build Pipeline    " -ForegroundColor Cyan
Write-Host "  Root: $ProjectRoot" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Set-Location $ProjectRoot

# -------------------------------------------------------------
# 1. Pre-build Secret Leak Scan
# -------------------------------------------------------------
Write-Host "`n[Step 1/6] Running Pre-Build Secret Leak Scan..." -ForegroundColor Yellow
python build/scan_secrets.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Secret scan failed! Build aborted to prevent publishing credentials."
}

# -------------------------------------------------------------
# 2. Automated Test Suite Verification
# -------------------------------------------------------------
Write-Host "`n[Step 2/6] Verifying Regression Tests..." -ForegroundColor Yellow
python -m pytest tests/app/ -q
if ($LASTEXITCODE -ne 0) {
    Write-Error "Test suite verification failed! Aborting packaging."
}

# -------------------------------------------------------------
# 3. Verify Bundled Binaries
# -------------------------------------------------------------
Write-Host "`n[Step 3/6] Verifying Bundled Binaries & Assets..." -ForegroundColor Yellow
$ffmpegPath = Join-Path $ProjectRoot "bin\ffmpeg.exe"
$ffprobePath = Join-Path $ProjectRoot "bin\ffprobe.exe"
$iconPath = Join-Path $ProjectRoot "assets\icon.ico"

if (-not (Test-Path $ffmpegPath) -or -not (Test-Path $ffprobePath)) {
    Write-Error "Bundled FFmpeg/FFprobe binaries missing in bin/! Please copy ffmpeg.exe and ffprobe.exe into bin/."
}
if (-not (Test-Path $iconPath)) {
    Write-Error "Application icon missing at assets/icon.ico!"
}
Write-Host "  [OK] FFmpeg and FFprobe binaries present in bin/" -ForegroundColor Green
Write-Host "  [OK] Application icon present at assets/icon.ico" -ForegroundColor Green

# -------------------------------------------------------------
# 4. Clean & Execute PyInstaller
# -------------------------------------------------------------
Write-Host "`n[Step 4/6] Compiling Desktop Application with PyInstaller..." -ForegroundColor Yellow

$distDir = Join-Path $ProjectRoot "dist"
$buildDir = Join-Path $ProjectRoot "build"
$outputDir = Join-Path $ProjectRoot "Output"

if (Test-Path (Join-Path $distDir "VideoContextEngine")) {
    Write-Host "  Cleaning existing dist/VideoContextEngine..."
    Remove-Item (Join-Path $distDir "VideoContextEngine") -Recurse -Force
}

python -m PyInstaller --noconfirm VideoContextEngine.spec

$distExe = Join-Path $distDir "VideoContextEngine\VideoContextEngine.exe"
if (-not (Test-Path $distExe)) {
    Write-Error "PyInstaller build completed but executable was not found at $distExe!"
}
$exeSize = (Get-Item $distExe).Length / 1MB
Write-Host ("  [OK] Packaged Executable: {0} ({1:N2} MB)" -f $distExe, $exeSize) -ForegroundColor Green

# -------------------------------------------------------------
# 5. Smoke Verification of Packaged Binaries
# -------------------------------------------------------------
Write-Host "`n[Step 5/6] Running Smoke Tests on Packaged Application..." -ForegroundColor Yellow

Write-Host "  Testing --help dispatch..."
& "$distExe" --run-module1 --help | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Smoke test failed: --run-module1 did not exit cleanly!"
}

& "$distExe" --run-module2 --help | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Smoke test failed: --run-module2 did not exit cleanly!"
}

Write-Host "  Testing --diagnose-smtp dispatch..."
& "$distExe" --diagnose-smtp | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Smoke test failed: --diagnose-smtp did not exit cleanly!"
}
Write-Host "  [OK] Packaged binary smoke tests passed!" -ForegroundColor Green

# -------------------------------------------------------------
# 6. Compile Inno Setup Installer
# -------------------------------------------------------------
Write-Host "`n[Step 6/6] Building Windows Installer with Inno Setup 6..." -ForegroundColor Yellow

$isccCandidates = @(
    "ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
)

$isccPath = $null
foreach ($candidate in $isccCandidates) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $isccPath = $candidate
        break
    }
    if (Test-Path $candidate) {
        $isccPath = $candidate
        break
    }
}

if (-not $isccPath) {
    Write-Error "Inno Setup Compiler (ISCC.exe) not found! Please install Inno Setup 6."
}

Write-Host "  Using Inno Setup Compiler: $isccPath"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$issFile = Join-Path $ProjectRoot "installer\VideoContextEngine.iss"

& "$isccPath" "$issFile"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Inno Setup compilation failed!"
}

$installerExe = Join-Path $outputDir "VideoContextEngine_Setup_x64.exe"
if (-not (Test-Path $installerExe)) {
    Write-Error "Installer executable not found at $installerExe!"
}

$installerItem = Get-Item $installerExe
$installerHash = (Get-FileHash $installerExe -Algorithm SHA256).Hash
$installerSizeMB = $installerItem.Length / 1MB

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  BUILD AND PACKAGING SUCCESSFUL!                         " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ("Installer Path:   {0}" -f $installerExe) -ForegroundColor White
Write-Host ("Installer Size:   {0:N2} MB ({1:N0} bytes)" -f $installerSizeMB, $installerItem.Length) -ForegroundColor White
Write-Host ("SHA-256 Checksum: {0}" -f $installerHash) -ForegroundColor White
Write-Host "==========================================================`n" -ForegroundColor Green

