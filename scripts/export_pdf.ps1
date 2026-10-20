# Export docs/presentation.pptx to docs/presentation.pdf with PowerPoint (Windows).
# Optional: -PreviewDir <folder> also writes one PNG per slide (for a visual check).
param([string]$PreviewDir = "")

$root = Split-Path -Parent $PSScriptRoot
$pptx = Join-Path $root "docs\presentation.pptx"
$pdf = Join-Path $root "docs\presentation.pdf"

$app = New-Object -ComObject PowerPoint.Application
try {
    $deck = $app.Presentations.Open($pptx, $true, $false, $false)
    $deck.SaveAs($pdf, 32)  # 32 = ppSaveAsPDF
    if ($PreviewDir) {
        New-Item -ItemType Directory -Force $PreviewDir | Out-Null
        $deck.Export($PreviewDir, "PNG", 1280, 720)
    }
    $deck.Close()
    Write-Output "wrote $pdf"
} finally {
    $app.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($app) | Out-Null
}
