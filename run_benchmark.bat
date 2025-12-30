@echo off
chcp 65001 >nul
REM ============================================================================
REM Run Parallel NN Benchmark
REM ============================================================================

echo.
echo ========================================
echo  Running Parallel NN Benchmark
echo ========================================
echo.

if not exist parallel_nn.exe (
    echo ERROR: parallel_nn.exe not found!
    echo Please run build_windows.bat first.
    pause
    exit /b 1
)

REM Run benchmark with verification
echo Running with 5000x5000 matrix (verification enabled)...
echo.
parallel_nn.exe -s 5000 -b 10 -v

echo.
echo ========================================
echo  Running larger matrix test...
echo ========================================
echo.

parallel_nn.exe -s 10000 -b 15

echo.
echo ========================================
echo  Scalability Test
echo ========================================
echo.

for %%s in (1000 2000 5000 10000 20000) do (
    echo.
    echo --- Matrix Size: %%s x %%s ---
    parallel_nn.exe -s %%s -b 10 -r 5
)

echo.
echo Done!
pause
