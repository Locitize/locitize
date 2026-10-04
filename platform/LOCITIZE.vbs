' LOCITIZE.vbs - the official no-flash double-click launcher (M17.12).
'
' Double-clicking a .bat forces Windows to open a black cmd console window for
' the whole life of the launcher - the flash an owner reported on startup. A
' .vbs opens no console at all. This runs LOCITIZE.bat hidden, so a normal launch
' shows only the desktop window, with zero console flash. All launch logic still
' lives in LOCITIZE.bat; this only changes WHETHER a console is shown.
'
' The setup wizard is a window of its own, so first run and --setup launch
' hidden too - a new user sees only the installer. The one message the console
' used to carry ("install Python first") is shown here as a dialog instead,
' before anything is launched.
'
' The console stays visible only where a person types into it:
'   --terminal (the text menu) and --uninstall (asks for confirmation).
'
' Any arguments are forwarded to LOCITIZE.bat, so "LOCITIZE.vbs --terminal" and
' "LOCITIZE.vbs --setup" behave like the batch equivalents.
Option Explicit

Dim shell, fso, here, bat, i, cmdLine, style, venvUp, venvHere, firstArg

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

' The venv lives at ..\.venv (normal, alongside platform\) or .venv (flat
' checkout). Without one, LOCITIZE.bat runs the wizard on the system Python.
venvUp = fso.FileExists(fso.GetParentFolderName(here) & "\.venv\Scripts\python.exe")
venvHere = fso.FileExists(here & "\.venv\Scripts\python.exe")
firstArg = ""
If WScript.Arguments.Count > 0 Then firstArg = LCase(WScript.Arguments(0))

' Setup runs on bare "python". On stock Windows that may be missing or only the
' Microsoft Store alias (which fails --version), so check it here, hidden, and
' explain in a dialog - the hidden console could not show the message.
If (Not (venvUp Or venvHere)) Or firstArg = "--setup" Then
    If shell.Run("cmd /c python --version", 0, True) <> 0 Then
        MsgBox "LOCITIZE needs Python 3.11 or newer, which this machine does not have yet." & vbCrLf & vbCrLf & _
               "1. Install it from https://www.python.org/downloads/" & vbCrLf & _
               "   (tick ""Add python.exe to PATH"" in the installer)" & vbCrLf & _
               "2. Double-click LOCITIZE.vbs again.", vbInformation, "LoCiTiZe setup"
        shell.Run "https://www.python.org/downloads/", 1, False
        WScript.Quit 1
    End If
End If

If firstArg = "--terminal" Or firstArg = "--uninstall" Then
    style = 1   ' visible - these read from the keyboard
Else
    style = 0   ' hidden - the desktop and the setup wizard are windows of their own
End If

' bWaitOnReturn False = launch and return at once so the launcher keeps running
' after this script exits.
shell.Run cmdLine, style, False
