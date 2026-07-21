@echo off
setlocal

set "DATA_PATH=%~dp0"
if "%DATA_PATH:~-1%"=="\" set "DATA_PATH=%DATA_PATH:~0,-1%"

set "DEFAULT_IMAGE=w_cideconvolve_benchmark"
if "%IMAGE%"=="" set "IMAGE=%DEFAULT_IMAGE%"

docker run --rm --gpus all ^
	-v "%DATA_PATH%\infolder:/data/in" ^
	-v "%DATA_PATH%\outfolder:/data/out" ^
	%IMAGE% ^
	--infolder /data/in ^
	--outfolder /data/out ^
	--local ^
	%*

endlocal
