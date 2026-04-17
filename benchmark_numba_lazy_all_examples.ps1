# ── benchmark_numba.ps1 ──

$examples = @(
    "examples/biaxial_column.yaml",
    "examples/example_tee.yaml",
    "examples/example_biaxial_column.yaml",
    "examples/example_annulus.yaml",
    "examples/example_input.yaml",
    "examples/example_custom.yaml",
    "examples/vcaslu_1.yaml",
    "examples/example_v2.1.yaml"
)

function Run-Suite {
    param([string]$Label)
    
    Write-Host "`n$('=' * 60)" -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Cyan
    Write-Host "$('=' * 60)" -ForegroundColor Cyan
    
    # Check numba availability
    uv run python -c "
try:
    import numba; print(f'  Numba: {numba.__version__}')
except ImportError:
    print('  Numba: NOT INSTALLED')
"
    
    $totalMs = 0
    $results = @()
    
    foreach ($ex in $examples) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($ex)
        $outdir = "results/bench_$Label/$name"
        
        $elapsed = (Measure-Command {
            uv run gensec run $ex --output-dir $outdir 2>&1 | Out-Null
        }).TotalSeconds
        
        $totalMs += $elapsed
        $results += [PSCustomObject]@{
            Example = $name
            Seconds = [math]::Round($elapsed, 1)
        }
        Write-Host "  $($name.PadRight(35)) $([math]::Round($elapsed, 1))s"
    }
    
    Write-Host "`n  TOTAL: $([math]::Round($totalMs, 1))s" -ForegroundColor Yellow
    return $totalMs
}

# ── Run WITHOUT numba ──
Write-Host "`nInstalling WITHOUT numba..."
uv sync --all-groups
$t_no = Run-Suite "WITHOUT NUMBA"

# ── Run WITH numba ──
Write-Host "`nInstalling WITH numba..."
uv sync --all-groups --all-extras
# Warm up JIT once (first import compiles cached kernels)
uv run python -c "from gensec.materials import Concrete; Concrete().stress_array(__import__('numpy').zeros(10))"
$t_yes = Run-Suite "WITH NUMBA"

# ── Summary ──
Write-Host "`n$('=' * 60)" -ForegroundColor Green
Write-Host "  COMPARISON" -ForegroundColor Green
Write-Host "$('=' * 60)" -ForegroundColor Green
Write-Host "  Without numba:  $([math]::Round($t_no, 1))s"
Write-Host "  With numba:     $([math]::Round($t_yes, 1))s"
$speedup = if ($t_yes -gt 0) { [math]::Round($t_no / $t_yes, 2) } else { "N/A" }
Write-Host "  Speedup:        ${speedup}x" -ForegroundColor Yellow