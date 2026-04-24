import requests
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4-abliterix:latest"
SYSTEM_PROMPT = "你是一個有用的助手，請使用繁體中文回應。"

history = []


def chat(user_message, think=False):
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": MODEL, "messages": messages, "stream": True, "think": think},
        stream=True,
    )
    resp.raise_for_status()

    full_response = ""
    in_thinking = False

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        msg = data.get("message", {})

        if msg.get("thinking"):
            if not in_thinking:
                in_thinking = True
                print("\n[思考中]\n", flush=True)
            print(msg["thinking"], end="", flush=True)
        elif msg.get("content"):
            if in_thinking:
                in_thinking = False
                print("\n\n[回答]\n", flush=True)
            content = msg["content"]
            full_response += content
            print(content, end="", flush=True)

        if data.get("done"):
            break

    print()
    history.append({"role": "assistant", "content": full_response})
    return full_response


def chat_with_image(image_path, question, think=False):
    import base64

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    messages = [
        {
            "role": "user",
            "content": question,
            "images": [image_b64],
        }
    ]

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": MODEL, "messages": messages, "stream": True, "think": think},
        stream=True,
    )
    resp.raise_for_status()

    full_response = ""
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        content = data.get("message", {}).get("content", "")
        full_response += content
        print(content, end="", flush=True)
        if data.get("done"):
            break

    print()
    return full_response


if __name__ == "__main__":
    # 支援命令列指定模型：python chat.py qwen3:14b
    if len(sys.argv) > 1:
        MODEL = sys.argv[1]

    # 確認 Ollama 服務可用
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        installed = [m["name"] for m in r.json().get("models", [])]
        if MODEL not in installed:
            print(f"錯誤: 找不到模型 '{MODEL}'")
            print(f"已安裝的模型: {installed}")
            print(f"請先執行: ollama pull {MODEL}")
            sys.exit(1)
    except requests.ConnectionError:
        print("錯誤: 無法連線 Ollama 服務。請確認 Ollama 正在運行。")
        sys.exit(1)

    print(f"=== 模型: {MODEL} ===")
    print("輸入文字開始對話，輸入 'quit' 退出")
    print("輸入 'model <名稱>' 切換模型，例如: model qwen3:14b")
    print("輸入 'clear' 清除對話記憶，重新開始")
    print("輸入 'think <問題>' 啟用思考模式，例如: think 解釋量子力學")
    print("輸入 'image <路徑> <問題>' 分析圖片，例如: image C:\\photo.jpg 這是什麼?")
    print()

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再見！")
            break

        # 切換模型
        if user_input.lower().startswith("model "):
            MODEL = user_input[6:].strip()
            history.clear()
            print(f"已切換至模型: {MODEL}，對話記憶已清除\n")
            continue

        # 清除對話記憶
        if user_input.lower() == "clear":
            history.clear()
            print("對話記憶已清除\n")
            continue

        think = False
        if user_input.lower().startswith("think "):
            think = True
            user_input = user_input[6:]

        if user_input.lower().startswith("image "):
            parts = user_input[6:].split(" ", 1)
            if len(parts) < 2:
                print("格式: image <圖片路徑> <問題>\n")
                continue
            image_path, question = parts
            print(f"\n{MODEL}: ", end="")
            chat_with_image(image_path, question, think=think)
        else:
            print(f"\n{MODEL}: ", end="")
            chat(user_input, think=think)

        print()
