@echo off
chcp 65001 >nul
REM ============================================================================
REM Parallel NN Sparse Matrix Scaling - Windows Build Script
REM RTX 4060 (sm_89 - Ada Lovelace)
REM ============================================================================
REM Authors: Ali Ahmet Taskesen, Omer Yildirim
REM ============================================================================

echo.
echo ========================================
echo  Parallel NN Sparse Matrix Scaling
echo  Building for RTX 4060 (sm_89)
echo ========================================
echo.

REM Check if nvcc exists
where nvcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: nvcc not found!
    echo Please install CUDA Toolkit from:
    echo https://developer.nvidia.com/cuda-downloads
    echo.
    echo After installation, make sure CUDA bin folder is in PATH:
    echo Usually: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
    pause
    exit /b 1
)

echo CUDA Compiler found:
nvcc --version
echo.

REM Compile
echo Compiling parallel_nn.cu...
nvcc -O3 -arch=sm_89 -o parallel_nn.exe parallel_nn_v2.cu

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================
    echo  BUILD SUCCESSFUL!
    echo ========================================
    echo.
    echo Run with: parallel_nn.exe
    echo Or:       parallel_nn.exe -s 10000 -v
    echo.
) else (
    echo.
    echo BUILD FAILED!
    echo.
)

pause
