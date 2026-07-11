# Single control point for which model trpg uses this run.
#
# Run this in its own terminal BEFORE starting the game (start.bat / python -m trpg).
# It always writes _server_config.json; trpg/cli.py has no backend/model constants
# of its own and just reads whatever this script decided.
#
# For a local GGUF model this also launches llama-server and blocks the terminal
# (leave it running, start the game in another terminal). For "ollama:" / "deepseek:"
# it just writes the config file and exits immediately -- Ollama itself keeps
# running as its own background service, nothing to launch here.

# --- Change model here (the ONLY place to pick a model) ---
#   local GGUF: qwen3.6-mtp-q4 / qwen3-30b / gemma4-google-q4 / gemma4-huihui-q4 / gemma4-hauhau-q4
#   Ollama:     ollama:<model name>       (e.g. ollama:qwen3.6-prism)
#   DeepSeek:   deepseek:<model name>     (e.g. deepseek:deepseek-v4-flash; api_key still lives in _deepseek_config.json)
$selectedModel = "gemma4-hauhau-q4"

$exe = "C:\Users\Wesley\Desktop\Python\llama.cpp-tq3\build\bin\Release\llama-server.exe"

$modelPaths = @{
    "qwen3.6-mtp-q4"   = "C:\Users\Wesley\Desktop\Python\llama.cpp-tq3\models\qwen3.6-35b-mtp\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    "qwen3-30b"        = "C:\Users\Wesley\Desktop\Python\llama.cpp-tq3\models\qwen3-30b-a3b\Qwen3-30B-A3B-Q4_K_M.gguf"
    "gemma4-google-q4" = "C:\Users\Wesley\Desktop\Python\llama.cpp-tq3\models\gemma4-26b\gemma-4-26B-A4B-it-ollama-text.gguf"
    "gemma4-huihui-q4" = "C:\Users\Wesley\Desktop\Python\llama.cpp-tq3\models\gemma4-26b\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    "gemma4-hauhau-q4" = "C:\Users\Wesley\Desktop\Python\llama.cpp-tq3\models\gemma4-26b\Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
}

# Per-model ncmoe (MoE expert layers on CPU). Tune per benchmark.
$ncmoeValues = @{
    "qwen3.6-mtp-q4"   = 16
    "qwen3-30b"        = 32
    "gemma4-google-q4" = 8
    "gemma4-huihui-q4" = 8
    "gemma4-hauhau-q4" = 8
}

$serverConfigPath = Join-Path $PSScriptRoot "_server_config.json"

if ($selectedModel -like "ollama:*") {
    $model = $selectedModel.Substring(7)
    $config = @{ backend = "ollama"; model = $model; base_url = "http://localhost:11434" } | ConvertTo-Json
    $config | Set-Content $serverConfigPath
    Write-Host "trpg backend: Ollama  model=$model"
    exit 0
}

if ($selectedModel -like "deepseek:*") {
    $model = $selectedModel.Substring(9)
    $config = @{ backend = "deepseek"; model = $model; base_url = "https://api.deepseek.com" } | ConvertTo-Json
    $config | Set-Content $serverConfigPath
    Write-Host "trpg backend: DeepSeek  model=$model (api_key comes from _deepseek_config.json)"
    exit 0
}

# --- Local GGUF: launch llama-server ---
$model = $modelPaths[$selectedModel]
if (-not $model) { $model = $selectedModel }
$ncmoe = $ncmoeValues[$selectedModel]
if (-not $ncmoe) { $ncmoe = 32 }

# base_url is bare host:port -- trpg's backend.py appends /v1/chat/completions itself
$config = @{ backend = "llamacpp"; model = $selectedModel; base_url = "http://127.0.0.1:8090" } | ConvertTo-Json
$config | Set-Content $serverConfigPath
Write-Host "trpg backend: llama.cpp  model=$selectedModel  ncmoe=$ncmoe"

& $exe -m $model `
    -ngl 99 -fa on `
    -ctk q8_0 -ctv q8_0 `
    -ncmoe $ncmoe `
    -c 32768 -b 512 -ub 512 `
    --parallel 1 `
    --cache-ram 0 `
    --jinja --reasoning auto `
    --host 127.0.0.1 --port 8090
