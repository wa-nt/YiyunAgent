@echo off
rem Build the desktop app: PyInstaller one-folder, windowed (no console).
rem Output: dist\SecondBrainAgent\  -> double-click SecondBrainAgent.exe
rem NOTE: keep this file pure ASCII - cmd.exe parses .bat in the OEM codepage (GBK
rem on zh-CN systems) and UTF-8 Chinese comments break line-continuation parsing.
setlocal
cd /d %~dp0
set PY=.venv\Scripts\python.exe
set OUT=dist\SecondBrainAgent
set STASH=%TEMP%\sba-runtime-stash

rem ---- Refuse to build while the app is running ----
rem PyInstaller --noconfirm deletes OUT wholesale, but a running instance holds
rem _internal\*.dll (clr_loader, vec0, pythonnet). The delete then fails halfway
rem and leaves OUT mangled - no schema.sql, no data - which looks like a broken build.
tasklist /FI "IMAGENAME eq SecondBrainAgent.exe" 2>nul | find /I "SecondBrainAgent.exe" >nul
if not errorlevel 1 (
  echo ERROR: SecondBrainAgent.exe is still running.
  echo Quit it first from the tray icon menu, then re-run this script.
  exit /b 1
)

rem ---- Preserve runtime state across the rebuild ----
rem The rebuild would also take the .env edited from the settings panel and the
rem data\ knowledge base. Stash them first, restore after. If OUT has nothing yet,
rem seed from the project root instead, so the very first build is usable as-is.
if exist "%STASH%" rmdir /S /Q "%STASH%"
mkdir "%STASH%" 2>nul
if exist "%OUT%\.env" copy /Y "%OUT%\.env" "%STASH%\.env" >nul
if not exist "%STASH%\.env" if exist ".env" copy /Y ".env" "%STASH%\.env" >nul
if exist "%OUT%\data" xcopy /E /I /Y /Q "%OUT%\data" "%STASH%\data" >nul
if not exist "%STASH%\data" if exist "data" xcopy /E /I /Y /Q "data" "%STASH%\data" >nul

rem Generate the exe icon (idempotent)
%PY% -m app.desktop --make-icon
if errorlevel 1 goto fail

rem --paths .  : entry is inside the app package, so let analysis resolve `app.*`
rem --add-data  : app/schema.sql is read via Path(__file__) at runtime, not importable
rem --collect-all: pywebview/pythonnet runtimes and sqlite-vec native dll
%PY% -m PyInstaller --noconfirm --clean --windowed ^
  --name SecondBrainAgent ^
  --icon web\app.ico ^
  --paths . ^
  --add-data "web;web" ^
  --add-data "app\schema.sql;app" ^
  --collect-all webview ^
  --collect-all pythonnet ^
  --collect-all sqlite_vec ^
  app\desktop.py
if errorlevel 1 goto fail

rem skills are loaded relative to CWD (see config.py); ship them next to the exe
xcopy /E /I /Y /Q skills "%OUT%\skills" >nul

call :restore
rmdir /S /Q "%STASH%"
echo.
echo Done: %OUT%\SecondBrainAgent.exe
echo .env and data\ were carried over from the previous build (or the project root).
exit /b 0

:fail
rem Put the runtime state back even on failure: PyInstaller may have already wiped
rem OUT, and losing the settings-panel .env / knowledge base would be the real damage.
echo.
echo Build FAILED - restoring .env and data\ so the output dir stays usable.
call :restore
exit /b 1

:restore
if exist "%STASH%\.env" copy /Y "%STASH%\.env" "%OUT%\.env" >nul
rem Clear OUT\data first: a failed build can leave it half-deleted, and copying on
rem top of a partial tree is how you end up with a nested data\data. The stash is
rem the authoritative copy, so dropping the broken one loses nothing.
if exist "%STASH%\data" rmdir /S /Q "%OUT%\data" 2>nul
if exist "%STASH%\data" xcopy /E /I /Y /Q "%STASH%\data" "%OUT%\data" >nul
goto :eof
