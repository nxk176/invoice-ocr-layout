[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ServerHost,

    [Parameter(Mandatory = $true)]
    [string]$ServerUser,

    [Parameter(Mandatory = $true)]
    [string]$RemoteRunDirectory,

    [Parameter(Mandatory = $true)]
    [string]$LocalOutputDirectory,

    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    @"
Usage:
  .\scripts\download_results.ps1 `
    -ServerHost server.example.org `
    -ServerUser username `
    -RemoteRunDirectory /remote/repo/outputs/run_001 `
    -LocalOutputDirectory .\outputs\run_001

The script uses rsync when available and falls back to scp. Authentication is delegated
to SSH keys/agent configuration; no password or hostname is stored in this repository.
"@
    exit 0
}

$destination = [System.IO.Path]::GetFullPath($LocalOutputDirectory)
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$remote = "${ServerUser}@${ServerHost}:${RemoteRunDirectory.TrimEnd('/')}/"

if (Get-Command rsync -ErrorAction SilentlyContinue) {
    & rsync -av --partial $remote "$destination/"
    if ($LASTEXITCODE -ne 0) { throw "rsync failed with exit code $LASTEXITCODE" }
}
elseif (Get-Command scp -ErrorAction SilentlyContinue) {
    $parent = Split-Path -Parent $destination
    & scp -r "${ServerUser}@${ServerHost}:$RemoteRunDirectory" $parent
    if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }
}
else {
    throw "Neither rsync nor scp is available on PATH."
}

Write-Host "Results downloaded to $destination"

