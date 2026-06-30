# daily_update.ps1 - Automated Daily Update System for TJK Racing Prediction
$ErrorActionPreference = "Continue"

# Configure working directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ScriptDir

# Get today's date for logging
$LogDate = Get-Date -Format "yyyy_MM_dd"
$LogFile = "logs\update_$LogDate.log"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$Script:PipelineFailures = @()

Function Write-InfoLog($Message) {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $FullMsg = "[$Timestamp] [INFO] $Message"
    Write-Output $FullMsg
    Add-Content -Path $LogFile -Value $FullMsg
}

Function Write-WarningLog($Message) {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $FullMsg = "[$Timestamp] [WARNING] $Message"
    Write-Warning $FullMsg
    Add-Content -Path $LogFile -Value $FullMsg
}

Function Write-ErrorLog($Message) {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $FullMsg = "[$Timestamp] [ERROR] $Message"
    Write-Error $FullMsg
    Add-Content -Path $LogFile -Value $FullMsg
}

Function Invoke-Script($ScriptName, $Arguments = "") {
    Write-InfoLog "Running: python $ScriptName $Arguments"
    
    # Run python script and capture exit code
    if ($Arguments -eq "") {
        python $ScriptName
    } else {
        python $ScriptName $Arguments
    }
    
    $ExitCode = $LASTEXITCODE
    Write-InfoLog "Exit code for $ScriptName`: $ExitCode"
    if ($ExitCode -eq 0) {
        Write-InfoLog "Successfully completed: $ScriptName"
        return $true
    } else {
        Write-WarningLog "Script $ScriptName failed with exit code $ExitCode."
        $Script:PipelineFailures += [PSCustomObject]@{
            Script = $ScriptName
            ExitCode = $ExitCode
        }
        # Append to failed updates CSV
        $DateStr = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $FailRow = """$DateStr"",""daily_update.ps1"",""$ScriptName"",""ExitCodeError"",""Script returned exit code $ExitCode"""
        if (-not (Test-Path "failed_updates.csv")) {
            Add-Content -Path "failed_updates.csv" -Value "date,script,entity,error_type,error_message"
        }
        Add-Content -Path "failed_updates.csv" -Value $FailRow
        return $false
    }
}

# --- PIPELINE START ---
Write-InfoLog "=== STARTING AUTOMATED DAILY UPDATE PIPELINE ==="

# 1. Yarış Programı
Invoke-Script "update_race_programs.py"

# 2. At Profilleri
Invoke-Script "update_profiles.py"

# 3. Yarış Geçmişi
Invoke-Script "update_races.py"

# 4. İstatistikler
Invoke-Script "update_statistics.py"

# 5. AGF
Invoke-Script "download_agfv2.py" "--today"

# 6. Komiser Raporları
Invoke-Script "komiser.py" "--today"

# 7. Komiser Parse
Invoke-Script "process_komiser.py" "--today"

# 8. Pist Bilgileri
Invoke-Script "update_track_conditions.py"

# 9. İdman Bilgileri
Invoke-Script "update_workouts.py"

# 10. Yarış Sonuçları (Gün sonu çalışmasında yararlı)
Invoke-Script "update_results.py"

# 11. Certified snapshot-only feature build.
# The builder runs leakage CI and provenance validation fail-closed.
$AsOfBuildOk = Invoke-Script "build_asof_features.py"

# 12. Predictions are forbidden when the leakage gate fails.
if ($AsOfBuildOk) {
    $ShadowOk = Invoke-Script "shadow_mode.py"
    Invoke-Script "predict_today.py"
    if ($ShadowOk) {
        Invoke-Script "shadow_monitor.py"
    } else {
        Write-WarningLog "Skipping shadow monitoring because prediction archival failed."
    }
} else {
    Write-WarningLog "Skipping predictions because the as-of leakage gate failed."
}

Write-InfoLog "=== COMPLETED AUTOMATED DAILY UPDATE PIPELINE ==="

if ($Script:PipelineFailures.Count -gt 0) {
    $FailureSummary = ($Script:PipelineFailures | ForEach-Object { "$($_.Script):$($_.ExitCode)" }) -join ", "
    Write-ErrorLog "=== PIPELINE FAILED: $($Script:PipelineFailures.Count) step(s) failed: $FailureSummary ==="
    exit 1
}

Write-InfoLog "=== PIPELINE SUCCESS: all steps completed with exit code 0 ==="
exit 0
