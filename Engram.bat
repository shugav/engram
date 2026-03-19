@echo off
:: Double-click to open Engram on your server via Cursor Remote SSH.
:: Replace YOUR_HOST and /path/to/engram with your actual values.

where cursor >nul 2>nul
if %errorlevel%==0 (
    start "" cursor --remote ssh-remote+YOUR_HOST /path/to/engram
) else (
    start "" "%LOCALAPPDATA%\Programs\cursor\Cursor.exe" --remote ssh-remote+YOUR_HOST /path/to/engram
)
