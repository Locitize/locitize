@echo off
rem claude-local - run Claude Code against the model LOCITIZE is serving.
rem Thin wrapper; the logic lives in claude-local.ps1 beside this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0claude-local.ps1" %*
