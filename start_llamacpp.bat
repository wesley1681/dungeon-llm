@echo off
set LLAMACPP=C:\Users\Wesley\Desktop\Python\llama.cpp-tq3\build\bin\Release
set MODEL=C:\Users\Wesley\Desktop\Python\models\Qwen3.6-27B-TQ3_4S\Qwen3.6-27B-TQ3_4S.gguf
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%LLAMACPP%;%CUDA_PATH%\bin;%PATH%

echo Starting llama-server on port 11435...
echo Model: %MODEL%
echo.

"%LLAMACPP%\llama-server.exe" -m "%MODEL%" -ngl 99 -c 32768 -ctk q4_0 -ctv tq3_0 -fa on --jinja --port 11435 --host 127.0.0.1

pause
