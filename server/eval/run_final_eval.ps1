# LumenX final evaluation - run on the GPU laptop from D:\LumenX in PowerShell:
#   powershell -ExecutionPolicy Bypass -File server\eval\run_final_eval.ps1
# Fill in the dataset paths below. Leave a path as "" to skip that dataset.
# Resumable: if it stops, run it again and finished images are skipped.

$Data = "C:\Users\Manoj D\Desktop\CV_hackathon"

# Paired datasets: low-light folder + matching reference (ground-truth) folder
$LOLv1_Low  = "$Data\LOLdataset\eval15\low"
$LOLv1_High = "$Data\LOLdataset\eval15\high"
$LOLv2_Low  = ""   # e.g. "$Data\LOL-v2\Real_captured\Test\Low"   (path is in core\HANDOFF.md)
$LOLv2_High = ""   # e.g. "$Data\LOL-v2\Real_captured\Test\Normal"
$College_Low  = ""   # college TEST split, low-light images
$College_High = ""   # its references, or "" if the college set has no ground truth

# Unpaired datasets (no ground truth), optional
$LIME = ""
$DICM = ""

# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$env:TMP = "D:\piptemp"; $env:TEMP = "D:\piptemp"; $env:PIP_CACHE_DIR = "D:\pipcache"
$Py   = ".\core\.venv\Scripts\python.exe"
$Eval = "server\eval\evaluate.py"
$Out  = "server\results"
$Pre  = "server\eval\pretrained_zerodce.yaml"
$Modes = "classical,dl,zerodce,hybrid"
$Abl = "no_tuner,no_fusion,no_classical,no_dl,no_gain_denoise,no_recoverability,no_guardrails,no_color_correction"

function Run([string[]]$EvalArgs) {
    Write-Host "`n>>> $Py $Eval $($EvalArgs -join ' ')" -ForegroundColor Cyan
    & $Py $Eval @EvalArgs --output $Out --save-images
    if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed (exit $LASTEXITCODE)" }
}

function Paired([string]$Name, [string]$Low, [string]$High, [switch]$Ablate) {
    if (-not $Low) { Write-Host "skip $Name (no path)"; return }
    $root = Split-Path $Low -Parent
    if ($High) {
        Run @("--dataset", $Name, "--root", $root, "--input-dir", $Low, "--reference-dir", $High, "--paired", "--modes", $Modes)
        # Pretrained Zero-DCE++ for the fine-tuning comparison
        Run @("--dataset", "$Name-zerodce-pretrained", "--root", $root, "--input-dir", $Low, "--reference-dir", $High, "--paired", "--modes", "zerodce", "--config", $Pre)
        if ($Ablate) { Run @("--dataset", "$Name-ablation", "--root", $root, "--input-dir", $Low, "--reference-dir", $High, "--paired", "--modes", "hybrid", "--ablations", $Abl) }
    } else {
        Unpaired $Name $Low
        if ($Ablate) { Run @("--dataset", "$Name-ablation", "--root", $root, "--input-dir", $Low, "--modes", "hybrid", "--ablations", $Abl) }
    }
}

function Unpaired([string]$Name, [string]$Dir) {
    if (-not $Dir) { Write-Host "skip $Name (no path)"; return }
    Run @("--dataset", $Name, "--root", $Dir, "--input-dir", $Dir, "--modes", $Modes)
}

$sw = [Diagnostics.Stopwatch]::StartNew()
Paired "College" $College_Low $College_High -Ablate
Paired "LOL-v1-eval15" $LOLv1_Low $LOLv1_High -Ablate
Paired "LOL-v2-Real-test" $LOLv2_Low $LOLv2_High
Unpaired "LIME" $LIME
Unpaired "DICM" $DICM

Write-Host "`n>>> aggregate + failure mining" -ForegroundColor Cyan
& $Py server\eval\aggregate.py --results $Out
& $Py server\eval\mine_failures.py --results $Out
Write-Host "`nDone in $([math]::Round($sw.Elapsed.TotalMinutes,1)) min. Results: $Out\benchmark_summary.csv" -ForegroundColor Green
