[CmdletBinding()]
param(
    [switch]$WithPdf,
    [switch]$WithPreprocessing,
    [switch]$Dev,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    @"
Usage: powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1 [options]

Options:
  -WithPdf             Install PDF rendering support.
  -WithPreprocessing   Install OpenCV deskew support.
  -Dev                 Install formatter, lint, type-check, and test tools.
  -Help                Show this help.
"@
    exit 0
}

$extras = @()
if ($WithPdf) { $extras += "pdf" }
if ($WithPreprocessing) { $extras += "preprocessing" }
if ($Dev) { $extras += "dev" }
$specifier = "."
if ($extras.Count -gt 0) {
    $specifier = ".[$($extras -join ',')]"
}

python --version
python -m pip install --upgrade pip
python -m pip install -e $specifier
Write-Host "Installed invoice-ocr-layout into the current Python environment."
Write-Host "Install PaddlePaddle/PyTorch separately for the CPU or CUDA version on this machine."

