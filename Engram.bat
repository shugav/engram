@echo off
:: Double-click to open Engram on a remote host via Cursor Remote SSH
:: Replace "your-server" with your SSH hostname (e.g. a Tailscale hostname)

where cursor >nul 2>nul
if %errorlevel%==0 (
    start "" cursor --remote ssh-remote+your-server /path/to/engram
) else (
    start "" "%LOCALAPPDATA%\Programs\cursor\Cursor.exe" --remote ssh-remote+your-server /path/to/engram
)
