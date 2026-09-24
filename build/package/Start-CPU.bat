@echo off
rem Same as Start.bat but never uses the GPU (for troubleshooting).
call "%~dp0Start.bat" --cpu
