@echo off
setlocal

REM Stops the Local LLM stack inside WSL (frontend + API + Docker container).

pushd "%~dp0" >nul

set "WSL_WORKDIR=/home/shan/local-llm-tests"
set "DISTRO_LABEL=%LOCAL_LLM_WSL_DISTRO%"
set "WSL_HOME=%LOCAL_LLM_WSL_HOME%"
if "%WSL_HOME%"=="" set "WSL_HOME=/home/shan"

set "LAUNCH_CMD=cd %WSL_WORKDIR% && ./bin/local-llm-launcher.sh stop"
set "LAUNCH_CMD=%LAUNCH_CMD:"=\"%"

set "WSL_BASE=wsl.exe"
set "DISPLAY_NAME=default"
if not "%DISTRO_LABEL%"=="" goto useCustomDistro
goto buildCommand

:useCustomDistro
set "DISPLAY_NAME=%DISTRO_LABEL%"
set "WSL_BASE=wsl.exe --distribution \"%DISTRO_LABEL%\""

:buildCommand

echo Stopping Local LLM stack in WSL distro "%DISPLAY_NAME%"...
%WSL_BASE% --cd "%WSL_HOME%" -- bash -lc "%LAUNCH_CMD%"
set "WSL_EXIT=%ERRORLEVEL%"
echo.
echo WSL command finished with exit code %WSL_EXIT%

if "%WSL_EXIT%"=="0" goto success_stop
goto error_stop

:success_stop
echo.
echo Local LLM stack requested to stop.
goto end_stop

:error_stop
echo.
echo [ERROR] Failed to stop stack (see output above).

:end_stop

echo.
pause
popd >nul
