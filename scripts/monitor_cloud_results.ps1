# Poll remote VMs on a timer; append lines to monitor_log.txt under this repo.
# Configure: GCP project, zone, VM names, and gcloud path before use.
# Stop: Ctrl+C.

$ErrorActionPreference = "Continue"
$CodeRoot = Split-Path $PSScriptRoot -Parent
$LogFile = Join-Path $CodeRoot "monitor_log.txt"

# Prefer gcloud on PATH; override with env GCLOUD e.g. "C:\...\google-cloud-sdk\bin\gcloud.ps1"
$Gcloud = $env:GCLOUD
if (-not $Gcloud) {
    $Gcloud = "gcloud"
}

# Set env GCP_PROJECT (and optionally GCP_ZONE) for your account
$Project = $env:GCP_PROJECT
if (-not $Project) {
    $Project = "your-gcp-project-id"
}
$Zone = $env:GCP_ZONE
if (-not $Zone) {
    $Zone = "us-central1-a"
}

# Space-separated VM names, e.g. $env:GRAPHCAL_MONITOR_VMS = "vm-a vm-b"
$VmList = @()
if ($env:GRAPHCAL_MONITOR_VMS) {
    $VmList = $env:GRAPHCAL_MONITOR_VMS -split "\s+"
} else {
    $VmList = @("your-vm-name")
}

# Remote directory listing on the VM (no spaces recommended)
$RemoteCodeDir = $env:GRAPHCAL_REMOTE_CODE_DIR
if (-not $RemoteCodeDir) {
    $RemoteCodeDir = "/home/ubuntu/2_Code"
}

$IntervalSec = 600
if ($env:GRAPHCAL_MONITOR_INTERVAL_SEC) {
    [int]$IntervalSec = $env:GRAPHCAL_MONITOR_INTERVAL_SEC
}

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss K"), $Message
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
    Write-Host $line
}

function Invoke-RemoteCheck {
    param([string]$VmName)
    # Single-quoted --command so $(hostname) expands on the Linux VM, not in PowerShell.
    $bash = 'echo ===$(hostname)===; echo ---dir---; ls -la ' + $RemoteCodeDir + ' 2>/dev/null | tail -n 20'
    & $Gcloud "compute" "ssh" $VmName "--zone=$Zone" "--project=$Project" "--quiet" "--command" $bash 2>&1
}

Write-Log "Monitor started (interval ${IntervalSec}s). Log: $LogFile"

while ($true) {
    Write-Log "----- poll cycle -----"
    foreach ($vm in $VmList) {
        Write-Log ">>> $vm"
        try {
            $out = Invoke-RemoteCheck -VmName $vm
            if ($null -eq $out) { Write-Log "  (no output)" }
            else { $out | ForEach-Object { Write-Log "  $_" } }
        }
        catch {
            Write-Log "  ERROR: $($_.Exception.Message)"
        }
    }
    Start-Sleep -Seconds $IntervalSec
}
