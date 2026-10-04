@echo off
rem LOCITIZE - the single double-click launcher (Milestone 13 consolidation).
rem Opens the one desktop window, which reaches every capability: Models,
rem Fine-tune, Talk, Voice Setup, Vision, Memory, Chat, Settings.
rem For the old text menu instead, run:  locitize.bat --terminal
rem Resolves everything relative to this file, so it works from any clone
rem location, and prefers the project virtual environment when present.
setlocal
cd /d "%~dp0"
if exist "..\.venv\Scripts\python.exe" (
    set "PY=..\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
rem M15 first-run gate. A fresh clone has no venv and no PySide6, so launching the
rem Qt desktop straight away would fail with a traceback in a console nobody asked
rem for. needs_setup() is one file check plus one import probe on stdlib only, so a
rem normal start pays almost nothing; "locitize.bat --setup" forces the wizard so a
rem user can come back and add features later.
rem The gate FAILS OPEN by design: no wizard file means no gate, so a checkout
rem without it (or a trimmed deployment) still launches normally instead of being
rem diverted into a setup that cannot run.
if /i "%~1"=="--uninstall" goto :uninstall
if /i "%~1"=="--setup" goto :setup
if not exist "setup_wizard.py" goto :launch
"%PY%" -c "import setup_wizard, sys; sys.exit(1 if setup_wizard.needs_setup() else 0)"
if errorlevel 1 goto :setup

:launch
"%PY%" launcher.py --desktop %*
goto :eof

:uninstall
rem Safe uninstall (M18.16): stdlib-only, so it runs on bare python even while
rem the venvs it removes are the thing being removed.
python uninstall.py
goto :eof

:setup
rem Deliberately bare "python": the venv this is about to build may not exist yet,
rem and the wizard is stdlib-only precisely so it runs on whatever is on PATH.
rem M15.8: on stock Windows "python" is a Microsoft Store alias that opens a
rem store page instead of running anything - the one link the bootstrap chain
rem could not carry itself. Probe for a REAL interpreter first (the alias fails
rem the --version probe), and if none exists say exactly what to install and
rem where, instead of flashing a store window at a confused new user.
python --version >nul 2>&1
if errorlevel 1 (
    echo locitize needs Python 3.11 or newer, which this machine does not have yet.
    echo.
    echo   1. Install it from  https://www.python.org/downloads/
    echo      ^(tick "Add python.exe to PATH" in the installer^)
    echo   2. Double-click locitize.vbs again.
    echo.
    start https://www.python.org/downloads/
    pause
    goto :eof
)
python setup_wizard.py
goto :eof
