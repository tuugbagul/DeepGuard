@echo off
setlocal

python demo.py %*
if errorlevel 1 (
    echo.
    echo DeepGuard demo baslatilamadi.
    pause
)
