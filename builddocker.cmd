@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "DOCKER_BUILDKIT=1"

REM Local image identity is declared in the Bilayers config.
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$c = Get-Content '%SCRIPT_DIR%\config.yaml' -Raw; if ($c -match '(?m)^\s*name:\s*(w_cideconvolve_benchmark)\s*$') { $Matches[1] }"`) do set "IMAGE_NAME=%%I"

REM Read version from version.txt
set /p VERSION=<"%SCRIPT_DIR%\version.txt"
if not defined IMAGE_NAME (
    echo Failed to read image name from config.yaml
    exit /b 1
)
if not defined VERSION (
    echo Failed to read version from version.txt
    exit /b 1
)

pushd "%SCRIPT_DIR%" >nul
if errorlevel 1 (
    echo Failed to change directory to %SCRIPT_DIR%
    exit /b 1
)

docker build --progress=plain -t "%IMAGE_NAME%:%VERSION%" -t "%IMAGE_NAME%:latest" %* .
set "EXITCODE=%ERRORLEVEL%"

popd >nul
endlocal & exit /b %EXITCODE%
