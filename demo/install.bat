@echo off
setlocal

echo Installing dependencies...
pip install --quiet -e "%~dp0..\python" || goto :error
pip install --quiet -e "%~dp0" || goto :error
exit /b 0

:error
echo Install failed.
exit /b 1
