\xEF\xBB\xBF@echo off
chcp 65001 >nul
echo.
echo ==================================================================
echo     NN vs DCT Karsilastirma Benchmark
echo     Ali Ahmet Taskesen, Omer Yildirim
echo ==================================================================
echo.

echo [1/4] NN Metodu - 1000x1000 matris
echo -----------------------------------
parallel_nn.exe -s 1000 -v
echo.

echo [2/4] DCT Metodu - 1000x1000 matris  
echo -----------------------------------
parallel_dct.exe -s 1000 -v
echo.

echo [3/4] NN Metodu - 5000x5000 matris
echo -----------------------------------
parallel_nn.exe -s 5000 -v
echo.

echo [4/4] DCT Metodu - 500x500 matris (bellek siniri)
echo -----------------------------------
parallel_dct.exe -s 500 -v
echo.

echo ==================================================================
echo     Benchmark Tamamlandi!
echo ==================================================================
echo.
pause
