@echo off
chcp 65001 >nul
REM ============================================================================
REM Run Full Benchmark: NN vs DCT Comparison
REM ============================================================================

echo.
echo ================================================================
echo  PARALLEL SPARSE MATRIX SCALING - FULL BENCHMARK
echo  NN vs DCT Comparison on RTX 4060
echo ================================================================
echo.

if not exist parallel_nn.exe (
    echo ERROR: parallel_nn.exe not found!
    echo Run build_all.bat first.
    pause
    exit /b 1
)

if not exist parallel_dct.exe (
    echo ERROR: parallel_dct.exe not found!
    echo Run build_all.bat first.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo  TEST 1: Nearest Neighbor (NN) Method
echo ================================================================
echo.
parallel_nn.exe -s 5000 -b 10 -v

echo.
echo ================================================================
echo  TEST 2: DCT Method (cuFFT)
echo ================================================================
echo.
parallel_dct.exe -s 1000 -b 10

echo.
echo ================================================================
echo  SCALABILITY TEST - NN Method
echo ================================================================
echo.

for %%s in (1000 2000 5000 10000) do (
    echo.
    echo --- Matrix Size: %%s x %%s ---
    parallel_nn.exe -s %%s -b 10 -r 5
)

echo.
echo ================================================================
echo  SCALABILITY TEST - DCT Method
echo ================================================================
echo.

for %%s in (500 1000 2000) do (
    echo.
    echo --- Matrix Size: %%s x %%s ---
    parallel_dct.exe -s %%s -b 10 -r 5
)

echo.
echo ================================================================
echo  BENCHMARK COMPLETE!
echo ================================================================
echo.

pause
