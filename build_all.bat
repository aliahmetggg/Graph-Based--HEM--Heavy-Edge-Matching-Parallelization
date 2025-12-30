@echo off
chcp 65001 >nul
REM ============================================================================
REM Build Both NN and DCT CUDA Programs for RTX 4060
REM ============================================================================

echo.
echo ========================================
echo  Building Parallel Sparse Matrix Scaling
echo  NN + DCT Methods (RTX 4060)
echo ========================================
echo.

where nvcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: nvcc not found! Install CUDA Toolkit.
    pause
    exit /b 1
)

echo CUDA found:
nvcc --version
echo.

REM Build NN
echo [1/2] Building parallel_nn.exe (Nearest Neighbor)...
nvcc -O3 -arch=sm_89 -o parallel_nn.exe parallel_nn_v2.cu
if %ERRORLEVEL% neq 0 (
    echo FAILED to build parallel_nn.exe
) else (
    echo SUCCESS: parallel_nn.exe built
)

echo.

REM Build DCT (requires cuFFT)
echo [2/2] Building parallel_dct.exe (DCT + cuFFT)...
nvcc -O3 -arch=sm_89 -o parallel_dct.exe parallel_dct.cu -lcufft
if %ERRORLEVEL% neq 0 (
    echo FAILED to build parallel_dct.exe
) else (
    echo SUCCESS: parallel_dct.exe built
)

echo.
echo ========================================
echo  Build Complete!
echo ========================================
echo.
echo Run tests:
echo   parallel_nn.exe -s 5000 -v
echo   parallel_dct.exe -s 1000
echo.

pause
