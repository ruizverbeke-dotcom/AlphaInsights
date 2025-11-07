# AlphaInsights — Daily Intelligence Memo Generator
# Usage: .\tools\new_memo.ps1 YYYY-MM-DD "Short Memo Title"

param(
    [string]$Date = (Get-Date -Format "yyyy-MM-dd"),
    [string]$Title = "Untitled Memo"
)

$templatePath = "docs\_templates\intelligence_memo_template.md"
$outputDir = "docs\intelligence_logs"
$outputFile = "$outputDir\$Date" + "_grand_technical_memo.md"

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

Copy-Item $templatePath $outputFile -Force
(Get-Content $outputFile) -replace "YYYY-MM-DD", $Date | Set-Content $outputFile
Write-Host "Created new memo:" $outputFile
Start-Process notepad.exe $outputFile
