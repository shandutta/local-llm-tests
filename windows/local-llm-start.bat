@echo off
setlocal

REM Launches the Local LLM stack inside WSL (model container + API + frontend).
REM Optional: setx LOCAL_LLM_WSL_DISTRO "YourDistroName" and/or setx LOCAL_LLM_WSL_HOME "/home/you".

pushd "%~dp0" >nul

set "WSL_WORKDIR=/home/shan/local-llm-tests"
set "DISTRO_LABEL=%LOCAL_LLM_WSL_DISTRO%"
set "WSL_HOME=%LOCAL_LLM_WSL_HOME%"
if "%WSL_HOME%"=="" set "WSL_HOME=/home/shan"

set "LAUNCH_CMD=cd %WSL_WORKDIR% && ./bin/local-llm-launcher.sh start"
set "LAUNCH_CMD=%LAUNCH_CMD:"=\"%"

set "WSL_BASE=wsl.exe"
set "DISPLAY_NAME=default"
if not "%DISTRO_LABEL%"=="" goto useCustomDistro
goto buildCommand

:useCustomDistro
set "DISPLAY_NAME=%DISTRO_LABEL%"
set "WSL_BASE=wsl.exe --distribution \"%DISTRO_LABEL%\""

:buildCommand

echo Starting Local LLM stack in WSL distro "%DISPLAY_NAME%"...
%WSL_BASE% --cd "%WSL_HOME%" -- bash -lc "%LAUNCH_CMD%"
set "WSL_EXIT=%ERRORLEVEL%"
echo.
echo WSL command finished with exit code %WSL_EXIT%

if "%WSL_EXIT%"=="0" goto success_start
goto error_start

:success_start
echo.
echo Local LLM stack requested to start. Check the terminal/logs for progress.
goto end_start

:error_start
echo.
echo [ERROR] Failed to start stack (see output above).

:end_start

echo.
pause
popd >nul
