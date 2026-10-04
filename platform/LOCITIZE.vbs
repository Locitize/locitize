' LOCITIZE.vbs - the official no-flash double-click launcher (M17.12).
'
' Double-clicking a .bat forces Windows to open a black cmd console window for
' the whole life of the launcher - the flash an owner reported on startup. A
' .vbs opens no console at all. This runs LOCITIZE.bat hidden, so a normal launch
' shows only the desktop window, with zero console flash. All launch logic still
' lives in LOCITIZE.bat; this only changes WHETHER a console is shown.
'
' It stays visible on purpose in the two cases where console output matters:
'   - first run (no virtual environment yet): the setup wizard and any
'     "install Python first" message must be seen, not swallowed.
'   - an explicit argument (--setup / --terminal): the terminal menu and the
'     wizard need a real console to read from and write to.
' Only a bare double-click on an already-set-up install launches hidden.
'
' Any arguments are forwarded to LOCITIZE.bat, so "LOCITIZE.vbs --terminal" and
' "LOCITIZE.vbs --setup" behave like the batch equivalents (and stay visible).
Option Explicit

Dim shell, fso, here, bat, i, cmdLine, style, venvUp, venvHere

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Resolve everything relative to this file so it works from any clone location.
here = fso.GetParentFolderName(WScript.ScriptFullName)
bat = here & "\LOCITIZE.bat"
shell.CurrentDirectory = here

' Build: cmd /c ""<bat>" <args...>"  - the outer quotes let cmd /c keep the
' quoted path intact, and each argument is quoted in case it contains spaces.
cmdLine = "cmd /c """"" & bat & """"
For i = 0 To WScript.Arguments.Count - 1
    cmdLine = cmdLine & " """ & WScript.Arguments(i) & """"
Next
cmdLine = cmdLine & """"

' Hide the console only for a plain, already-set-up launch. The venv lives at
' ..\.venv (normal, alongside platform\) or .venv (flat checkout).
venvUp = fso.FileExists(fso.GetParentFolderName(here) & "\.venv\Scripts\python.exe")
venvHere = fso.FileExists(here & "\.venv\Scripts\python.exe")
If WScript.Arguments.Count = 0 And (venvUp Or venvHere) Then
    style = 0   ' hidden window - no console flash
Else
    style = 1   ' visible - first-run setup, or an explicit --setup / --terminal
End If

' bWaitOnReturn False = launch and return at once so the launcher keeps running
' after this script exits.
shell.Run cmdLine, style, False
