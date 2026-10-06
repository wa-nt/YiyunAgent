@echo off
rem Build the Windows installer (Inno Setup) from the PyInstaller output.
rem NOTE: keep this file pure ASCII - cmd.exe parses .bat in the OEM codepage.
rem
rem Prerequisite: dist\YiyunAgent\ must exist. Run build_desktop.bat first.
rem Output: dist\installer\YiyunAgent-0.1.0-win64-setup.exe
setlocal
cd /d %~dp0

set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist %ISCC% (
  echo ERROR: Inno Setup 6 not found ^(ISCC.exe^).
  echo Install it from https://jrsoftware.org/isdl.php then re-run.
  exit /b 1
)

if not exist "dist\YiyunAgent\YiyunAgent.exe" (
  echo ERROR: dist\YiyunAgent\YiyunAgent.exe not found.
  echo Run build_desktop.bat first.
  exit /b 1
)

%ISCC% installer\YiyunAgent.iss
if errorlevel 1 (
  echo.
  echo Installer build FAILED.
  exit /b 1
)

echo.
echo Done: dist\installer\YiyunAgent-0.1.0-win64-setup.exe
exit /b 0
